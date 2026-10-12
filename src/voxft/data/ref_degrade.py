"""确定性参考音降质器：模拟「差参考音」（多人声串入 / 频带受限）做鲁棒性实验。

背景：线上参考音不总是干净的（已观测多人声参考、SNR 4.27dB 参考）。训练侧把
「降质 ref + 干净 target」配对（loss 只算 target，见官方
packers.process_tts_data_with_ref），教模型「条件脏也要出干净音」；评测侧用同一套
profile 量化 base/r8 在差参考下的退化幅度。

确定性契约：同一 (源 wav 内容, profile, seed) 产出逐字节一致——全部随机性来自
np.random.RandomState(seed)；批量入口（CLI / degrade_train_refs）用 derive_seed
把全局 seed + 源文件 SHA + profile 派生成 per-(文件, profile) 的种子，跨进程确定。
只改参考音，不改 target；profile 矩阵冻结在 PROFILES，新增退化类型先改这里。
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, lfilter

from .pipeline import _frame_hop, _frame_rms, _read_manifest, _write_jsonl, load_wav_mono

# profile 矩阵冻结：两类用户圈定的退化（串音 / 低通）各两档 severity + 一个组合档。
# 增删退化类型必须连文档（docs/ref_robustness_plan_20261012.md）一起改，别散落字面量。
PROFILES: dict[str, dict] = {
    "babble_sir15": {"kind": "babble", "sir_db": 15.0, "severity": "moderate"},
    "babble_sir8": {"kind": "babble", "sir_db": 8.0, "severity": "severe"},
    "lowpass4k": {"kind": "lowpass", "cutoff_hz": 4000, "severity": "moderate"},
    "lowpass3k": {"kind": "lowpass", "cutoff_hz": 3000, "severity": "severe"},
    "combo_babble8_lp3k": {"kind": "combo", "sir_db": 8.0, "cutoff_hz": 3000,
                           "severity": "severe"},
}

_PEAK_LIMIT = 0.99


def _sha256(path: str | Path) -> str:
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def derive_seed(seed: int, src_sha: str, profile: str) -> int:
    """per-(文件, profile) 的种子：sha256(f"{seed}:{src_sha}:{profile}") 前 8 字节转 int。

    不依赖调用顺序与进程状态，同一输入文件在任何机器、任何批次里拿到同一个种子。
    """
    digest = hashlib.sha256(f"{seed}:{src_sha}:{profile}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _voiced_rms(wav: np.ndarray, sr: int) -> float:
    """ref 有声段 RMS：帧 RMS（pipeline._frame_rms 同口径）p40 以上帧覆盖的样本。

    SIR 必须按有声段定标而不是整段：参考首尾的静音会稀释整段 RMS，
    同一标称 SIR 下短参考的实际串音会更响。
    """
    rms = _frame_rms(wav, sr)
    if not rms.size:
        return 0.0
    thr = max(float(np.percentile(rms, 40)), 1e-5)
    hop = _frame_hop(sr)
    mask = np.zeros(len(wav), dtype=bool)
    for i, ok in enumerate(rms >= thr):
        if ok:
            mask[i * hop:(i + 1) * hop] = True
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav[mask].astype(np.float64)))))


def _entry_wav(entry: dict) -> tuple[np.ndarray, int]:
    """干扰池条目可直接给 wav/sr，或给 audio 路径按需加载（只加载被选中的条目）。"""
    if "wav" in entry:
        return np.asarray(entry["wav"], dtype=np.float32), int(entry["sr"])
    if entry.get("audio"):
        return load_wav_mono(entry["audio"])
    raise ValueError("干扰池条目必须提供 wav/sr 或 audio 路径")


def _mix_babble(wav: np.ndarray, sr: int, sir_db: float, rs: np.random.RandomState,
                interferers: list[dict] | None, ref_speaker: str | None,
                ref_origin: str | None) -> tuple[np.ndarray, list[str]]:
    """混入 1-2 条他人语音并定标到目标 SIR。返回 (干扰信号, 实际用到的 origin_audio 列表)。

    SIR 定义：20*log10(rms(ref_voiced) / rms(interferer_scaled))——ref 侧取有声段 RMS，
    干扰侧取拼到等长后的整段 RMS。同 speaker / 同 origin_audio 的条目必须拒绝：
    把 ref 本人的声音混回去既不是串音、又污染身份实验；过滤后为空就显式失败，
    不静默降级成无干扰。
    """
    pool = [e for e in (interferers or [])
            if e.get("speaker") != ref_speaker and e.get("origin_audio") != ref_origin]
    if not pool:
        raise ValueError("过滤同 speaker/同 origin_audio 后干扰池为空，无法合成 babble")
    ref_voiced = _voiced_rms(wav, sr)
    if ref_voiced < 1e-8:
        raise ValueError("参考音没有可定标的有声段（近静音），无法合成 babble")
    n = int(rs.randint(1, min(3, len(pool) + 1)))
    picks = sorted(int(i) for i in rs.choice(len(pool), size=n, replace=False))
    mixed = np.zeros(len(wav), dtype=np.float64)
    origins: list[str] = []
    for i in picks:
        entry = pool[i]
        iwav, isr = _entry_wav(entry)
        if isr != sr:
            iwav = librosa.resample(iwav, orig_sr=isr, target_sr=sr)
        if not iwav.size:
            continue
        # 随机起点循环拼接到等长：不同参考的串音不从干扰源同一帧开始
        tiled = np.tile(iwav.astype(np.float64), len(wav) // len(iwav) + 2)
        start = int(rs.randint(0, len(iwav)))
        mixed += tiled[start:start + len(wav)]
        origins.append(str(entry.get("origin_audio") or entry.get("audio")))
    if not origins:
        raise ValueError("干扰池条目均为空音频")
    cur = float(np.sqrt(np.mean(np.square(mixed))))
    if cur < 1e-8:
        raise ValueError("干扰源近静音，无法定标 SIR")
    mixed *= ref_voiced * 10.0 ** (-sir_db / 20.0) / cur
    return mixed, origins


def _lowpass(wav: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """4 阶 Butterworth + filtfilt 零相位（scipy 是 librosa 传递依赖，不新增依赖）。"""
    if not (0 < cutoff_hz < sr / 2):
        raise ValueError(f"截止频率 {cutoff_hz} 超出奈奎斯特范围（sr={sr}）")
    b, a = butter(4, cutoff_hz, fs=sr)
    # filtfilt 默认 padlen=3*max(len(a),len(b))；比这个还短的退化输入走 lfilter，
    # 边缘瞬变可接受（正常 ref 是秒级，不会走到这）。
    if len(wav) <= 3 * max(len(a), len(b)):
        return lfilter(b, a, wav)
    return filtfilt(b, a, wav)


def degrade_wav(wav: np.ndarray, sr: int, profile: str, *, seed: int,
                interferers: list[dict] | None = None, ref_speaker: str | None = None,
                ref_origin: str | None = None) -> tuple[np.ndarray, dict]:
    """对单条参考音施加指定 profile 的确定性降质；返回 (降质后 wav, 实际参数记录)。

    所有 profile 末尾把整段 RMS 对齐回原始 ref（生产侧本就有电平对齐，退化只改
    "质"不改电平），并防削波（峰值 >0.99 时整体回退到 0.99）。combo = 先 babble
    再 lowpass（真实差参考常两者叠加）。
    """
    spec = PROFILES.get(profile)
    if spec is None:
        raise ValueError(f"未知 profile: {profile}（可选 {sorted(PROFILES)}）")
    wav = np.asarray(wav, dtype=np.float64)
    if not wav.size:
        raise ValueError("空音频无法降质")
    rs = np.random.RandomState(int(seed) % (2 ** 32))  # RandomState 只接受 32 位种子
    kind = spec["kind"]
    record: dict = {"profile": profile, "severity": spec["severity"], "kind": kind}
    out = wav
    if kind in ("babble", "combo"):
        mixed, origins = _mix_babble(wav, sr, float(spec["sir_db"]), rs, interferers,
                                     ref_speaker, ref_origin)
        out = wav + mixed
        record.update(sir_db=float(spec["sir_db"]), interferer_origins=origins)
    if kind in ("lowpass", "combo"):
        out = _lowpass(out, sr, int(spec["cutoff_hz"]))
        record["cutoff_hz"] = int(spec["cutoff_hz"])
    ref_rms = float(np.sqrt(np.mean(np.square(wav))))
    out_rms = float(np.sqrt(np.mean(np.square(out))))
    level_gain = ref_rms / out_rms if ref_rms > 1e-8 and out_rms > 1e-8 else 1.0
    out = out * level_gain
    peak = float(np.abs(out).max())
    if peak > _PEAK_LIMIT:
        out = out * (_PEAK_LIMIT / peak)
    record["level_gain_db"] = round(20.0 * float(np.log10(max(level_gain, 1e-8))), 3)
    record["clipped"] = bool(peak > _PEAK_LIMIT)
    return out, record


def degrade_file(src: str | Path, dst: str | Path, profile: str, *, seed: int,
                 interferers: list[dict] | None = None, ref_speaker: str | None = None,
                 ref_origin: str | None = None) -> dict:
    """读 src、降质、写 dst（与 pipeline 同款的 float32 写法）；源文件不动。

    返回 sidecar 记录：source sha256 / profile / seed / params / dst 文件名。
    批量调用方把记录交给 append_sidecars 落 degradations.jsonl。
    """
    src, dst = Path(src), Path(dst)
    src_sha = _sha256(src)
    wav, sr = load_wav_mono(src)
    out, params = degrade_wav(wav, sr, profile, seed=seed, interferers=interferers,
                              ref_speaker=ref_speaker, ref_origin=ref_origin)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, out.astype(np.float32), sr)
    return {"source_sha256": src_sha, "profile": profile, "seed": int(seed),
            "params": params, "dst": dst.name}


def append_sidecars(records: list[dict], path: str | Path) -> None:
    """把降质记录追加写 degradations.jsonl：读旧 → 合并 → 原子写回（同 pipeline._write_jsonl）。"""
    path = Path(path)
    existing = _read_manifest(path) if path.exists() else []
    _write_jsonl(existing + records, path)


def _manifest_rows(manifest: Path) -> list[dict]:
    """--manifest 可以是 JSONL（每行至少 audio，可选 speaker/origin_audio）或目录（取 *.wav）。"""
    if manifest.is_dir():
        return [{"audio": str(p)} for p in sorted(manifest.glob("*.wav"))]
    rows = _read_manifest(manifest)
    if any(not row.get("audio") for row in rows):
        raise ValueError(f"manifest 每行必须含 audio: {manifest}")
    return rows


def main(argv: list[str] | None = None, progress=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="参考音 JSONL 或含 *.wav 的目录")
    ap.add_argument("--out", required=True, help="输出目录；产物为 <out>/<profile>/<原文件名>")
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--pool", default=None,
                    help="干扰池 JSONL（每行 audio/speaker/origin_audio）；含 babble 类 profile 时必填")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    need_pool = any(PROFILES[p]["kind"] in ("babble", "combo") for p in args.profiles)
    pool = _read_manifest(Path(args.pool)) if args.pool else None
    if need_pool:
        if not pool:
            ap.error("--pool 是 babble 类 profile 的必填项："
                     "干扰源必须带 speaker/origin_audio，否则无法拒绝同人/同源")
        if any(not r.get("audio") or not r.get("speaker") or not r.get("origin_audio")
               for r in pool):
            raise ValueError("干扰池每行必须提供 audio/speaker/origin_audio")
    rows = _manifest_rows(Path(args.manifest))
    out = Path(args.out)
    records = []
    for row in rows:
        src = Path(row["audio"])
        src_sha = _sha256(src)
        for profile in args.profiles:
            dst = out / profile / src.name
            rec = degrade_file(src, dst, profile,
                               seed=derive_seed(args.seed, src_sha, profile),
                               interferers=pool,
                               ref_speaker=row.get("speaker"),
                               ref_origin=row.get("origin_audio"))
            rec.update(source=str(src), dst=str(dst.relative_to(out)))
            records.append(rec)
            if progress:
                progress(f"{src.name} × {profile} -> {dst.relative_to(out)}")
    append_sidecars(records, out / "degradations.jsonl")
    if progress:
        progress(f"完成 {len(records)} 条降质（{len(rows)} 参考 × {len(args.profiles)} profile），"
                 f"sidecar: {out / 'degradations.jsonl'}")


if __name__ == "__main__":
    main(progress=print)
