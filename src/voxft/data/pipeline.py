from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..paths import DATA_PROCESSED, DATA_RAW

TARGET_SR = 16000


@dataclass
class Options:
    min_dur: float = 3.0
    max_dur: float = 30.0
    ref_audio_ratio: float = 0.4  # 官方建议 30-50% 样本带 ref_audio
    val_ratio: float = 0.02
    utmos_min: float | None = None       # 如 3.5；None = 不做 UTMOS 过滤
    whisper_lang: str | None = None      # "th"/"tl"/"zh"；None = 不做转写校验
    whisper_min_sim: float = 0.55
    seed: int = 42


def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(path, dtype="float32", always_2d=True)
    return arr.mean(axis=1), sr


def trim_silence(wav: np.ndarray, sr: int, floor: float = 1e-3,
                 tail_keep: float = 0.3) -> np.ndarray:
    """裁掉首尾静音；尾部最多保留 tail_keep 秒（官方要求 <0.5s，防生成失控）。

    floor 相对峰值幅度（1e-3 ≈ -60dB）。
    """
    peak = float(np.abs(wav).max())
    if peak < 1e-8:
        return wav
    thr = peak * floor
    nz = np.nonzero(np.abs(wav) > thr)[0]
    if len(nz) == 0:
        return wav
    start = max(0, nz[0] - int(0.05 * sr))
    end = min(len(wav), nz[-1] + 1 + int(tail_keep * sr))
    return wav[start:end]


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(wav).max())
    return wav * (peak / m) if m > 1e-8 else wav


def _whisper_model(lang: str):
    from faster_whisper import WhisperModel  # 可选依赖：uv sync --group qc
    return WhisperModel("medium", device="auto", compute_type="auto")


def _whisper_similarity(model, wav: np.ndarray, sr: int, lang: str,
                        ref_text: str) -> float:
    segs, _ = model.transcribe(wav, language=lang, vad_filter=True)
    hyp = "".join(s.text for s in segs).lower().replace(" ", "")
    ref = ref_text.lower().replace(" ", "")
    return SequenceMatcher(None, hyp, ref).ratio()


def _read_manifest(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_dataset(source_id: str, out_name: str | None = None,
                    opts: Options | None = None, max_items: int | None = None,
                    progress=None) -> dict:
    """加工 data/raw/<source_id>/manifest.jsonl → data/processed/<out_name>/

    步骤：16k 重采样 → 裁尾静音 → 响度归一化 → 时长过滤 → 可选 Whisper 校验 →
    可选 UTMOS 过滤 → ref_audio 配对 → 说话人分层切分 → train/val JSONL。
    返回统计信息。
    """
    opts = opts or Options()
    rng = random.Random(opts.seed)
    src = DATA_RAW / source_id
    manifest = src / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"先运行下载：python -m voxft.data.download --source {source_id}")
    out = DATA_PROCESSED / (out_name or source_id)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest)
    if max_items:
        rows = rows[:max_items]

    whisper = None
    if opts.whisper_lang:
        if progress:
            progress(f"加载 Whisper 质检模型（{opts.whisper_lang}，首次需下载）...")
        whisper = _whisper_model(opts.whisper_lang)
        if progress:
            progress("Whisper 质检模型就绪")
    if opts.utmos_min is not None:
        from ..qc.utmos import get_scorer, last_error
        if get_scorer() is None:
            raise RuntimeError(f"UTMOS 评分器不可用，无法质检: {last_error()}")
        from ..qc.utmos import score_wav
        if progress:
            progress(f"UTMOS 质检已启用（阈值 {opts.utmos_min}）")
    else:
        score_wav = None

    kept, stats = [], {"total": len(rows), "drop_decode": 0, "drop_duration": 0,
                       "drop_whisper": 0, "drop_utmos": 0}
    for i, row in enumerate(rows):
        if progress and i % 50 == 0:
            progress(f"加工 {source_id}: {i}/{len(rows)}")
        try:
            wav, sr = load_wav_mono(row["audio"])
        except Exception:
            stats["drop_decode"] += 1
            continue
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        wav = trim_silence(wav, TARGET_SR)
        if not (opts.min_dur <= len(wav) / TARGET_SR <= opts.max_dur):
            stats["drop_duration"] += 1
            continue
        if whisper is not None:
            try:
                sim = _whisper_similarity(whisper, wav, TARGET_SR,
                                          opts.whisper_lang, row["text"])
            except Exception:
                sim = 0.0
            if sim < opts.whisper_min_sim:
                stats["drop_whisper"] += 1
                continue
        if score_wav is not None:
            mos = score_wav(wav, TARGET_SR)
            if mos is None or mos < opts.utmos_min:
                stats["drop_utmos"] += 1
                continue
        wav = peak_normalize(wav)
        dst = audio_dir / f"{i:07d}.wav"
        sf.write(dst, wav, TARGET_SR)
        kept.append({
            "audio": str(dst), "text": row["text"],
            "speaker": row.get("speaker", "default"),
            "duration": round(len(wav) / TARGET_SR, 2),
        })

    stats["kept"] = len(kept)
    if not kept:
        raise RuntimeError(f"{source_id}: 过滤后无剩余样本，请放宽质检参数")

    # ref_audio：同说话人随机配另一条（官方建议 30-50%）
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for rec in kept:
        by_spk[rec["speaker"]].append(rec)
    for rec in kept:
        pool = by_spk[rec["speaker"]]
        if len(pool) > 1 and rng.random() < opts.ref_audio_ratio:
            ref = rng.choice([r for r in pool if r is not rec])
            rec["ref_audio"] = ref["audio"]

    # 说话人分层切 val；单说话人时随机切
    spks = list(by_spk)
    val: list[dict] = []
    if len(spks) > 1:
        rng.shuffle(spks)
        target = max(1, int(len(kept) * opts.val_ratio))
        while spks and len(val) < target and len(spks) > 1:
            val.extend(by_spk[spks.pop()])
        train = [r for r in kept if r["speaker"] in set(spks)]
    else:
        rng.shuffle(kept)
        n_val = max(1, int(len(kept) * opts.val_ratio))
        val, train = kept[:n_val], kept[n_val:]
    rng.shuffle(train)

    _write_jsonl(train, out / "train.jsonl")
    _write_jsonl(val, out / "val.jsonl")
    stats.update({"train": len(train), "val": len(val),
                  "speakers": len(by_spk),
                  "with_ref_audio": sum(1 for r in train if "ref_audio" in r),
                  "output": str(out)})
    (out / "stats.json").write_text(
        json.dumps({"source_id": source_id, "options": asdict(opts), **stats},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def mix_manifests(parts: list[tuple[str, float]], out_name: str,
                  seed: int = 42) -> dict:
    """按权重混合多个已加工数据集（重复行拼接）。

    parts: [(processed_name, weight), ...]，如 [("fleurs_th", 0.85), ("fleurs_zh", 0.15)]
    输出 data/processed/<out_name>/{train,val}.jsonl。
    """
    if not parts:
        raise ValueError("parts 不能为空")
    rng = random.Random(seed)
    total_w = sum(w for _, w in parts)
    out = DATA_PROCESSED / out_name
    summary = {}
    for split in ("train", "val"):
        merged, counts = [], []
        for name, _w in parts:
            p = DATA_PROCESSED / name / f"{split}.jsonl"
            if not p.exists():
                raise FileNotFoundError(p)
            counts.append(_read_manifest(p))
        n_total = sum(len(c) for c in counts)
        mixed: list[dict] = []
        for (name, w), rows in zip(parts, counts):
            if not rows:
                continue
            target = max(1, round(w / total_w * n_total))
            tiled = (rows * ((target // len(rows)) + 1))[:target]
            summary[f"{name}/{split}"] = target
            mixed.extend(tiled)
        rng.shuffle(mixed)
        _write_jsonl(mixed, out / f"{split}.jsonl")
    (out / "mix.json").write_text(
        json.dumps({"parts": parts, "counts": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return {"output": str(out), **summary}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--utmos-min", type=float, default=None)
    ap.add_argument("--whisper-lang", default=None)
    args = ap.parse_args()
    o = Options(utmos_min=args.utmos_min, whisper_lang=args.whisper_lang)
    print(json.dumps(process_dataset(args.source, args.out, o, args.max_items),
                     ensure_ascii=False, indent=2))
