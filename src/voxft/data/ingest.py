"""成片音轨 → 切分 → 转写 → 追加到自备语料（drama_tl / drama_th / drama_vi / drama_id / replay_en）。

Tagalog 没有可商用的开源真人表演语料：Common Voice tl 官方 0 小时、YODAS/YODAS2 的
224 个语种子集里没有 tl、OpenSLR 无菲律宾语资源、HF 上带 audio 的只有厂商 sample。
vi / id 处境相同：本轮未能核实到任何可商用的开源真人情感/表演语料（见
docs/vi_id_support.md 的待核实清单）。所以 drama_* 只能来自自有授权素材，本模块把
"手工切片 + 手写字段"换成"自动切分转写 + 人工试听标注"，并保证追加后重新加工不泄漏
旧验证集（holdout.json）。

只在远程 GPU 机跑；解码视频容器需要 PyAV（uv sync --group qc）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import pipeline
from .pipeline import TARGET_SR, _assert_control_lang
from .registry import SOURCES, get_source

VERDICTS = ("保留", "丢弃", "待定")
ALL = "（全部批次）"


def ingest_dir(source_id: str, video_id: str) -> Path:
    return pipeline.DATA_RAW / source_id / "ingest" / video_id


def _safe_id(name: str) -> str:
    """素材 ID 会变成 DATA_RAW 下的目录名，必须挡掉路径穿越。"""
    vid = re.sub(r"[^\w.\-]+", "_", str(name).strip())[:80]
    if not vid or vid in (".", ".."):
        raise ValueError(f"素材 ID 非法: {name!r}")
    return vid


def decode_to_wav(src: str | Path, dst: Path, progress=None) -> Path:
    """PyAV 解码视频/音频容器 → 16k 单声道 WAV。soundfile 读不了 mp4/mkv。"""
    try:
        import av  # 可选依赖：uv sync --group qc（faster-whisper 本来就带 PyAV）
    except ImportError as exc:
        raise RuntimeError(
            "解码视频容器需要 PyAV：uv sync --group qc\n"
            "项目不依赖系统 ffmpeg；PyAV 自带 ffmpeg 库，本地与远程都可用。") from exc
    dst.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    with av.open(str(src)) as container:
        if not container.streams.audio:
            raise RuntimeError(f"{src}: 没有音轨")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)
        for frame in [*container.decode(container.streams.audio[0]), None]:
            for out in resampler.resample(frame):   # 末尾 None 冲出重采样器缓冲
                arr = out.to_ndarray().reshape(-1)
                chunks.append(arr.astype(np.float32) / 32768.0)
    wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    if wav.size < TARGET_SR:
        raise RuntimeError(f"{src}: 只解出 {wav.size / TARGET_SR:.2f}s 音频，检查文件是否完整")
    sf.write(dst, wav, TARGET_SR, subtype="PCM_16")
    if progress:
        progress(f"解码 {Path(src).name}：{wav.size / TARGET_SR / 60:.1f} 分钟 → {dst}")
    return dst


def speech_regions(wav: np.ndarray, lang: str, progress=None) -> list[tuple[float, float]]:
    """整轨过一遍 Whisper 只取 VAD 边界（medium 够快）；文本留给 large-v3 逐条转写，
    因为语种检测是逐条的，整轨只能给一个语种。"""
    model = pipeline._whisper_model(lang, "medium", progress)
    segs, _info = model.transcribe(wav, vad_filter=True)
    return [(float(s.start), float(s.end)) for s in segs]


def _split_long(wav: np.ndarray, start: float, end: float,
                max_dur: float) -> list[tuple[float, float]]:
    """超长区间在最安静的一帧切开，不切在词中间。"""
    if end - start <= max_dur:
        return [(start, end)]
    lo, hi = start + max_dur * 0.4, start + max_dur * 0.9
    seg = wav[int(lo * TARGET_SR):int(hi * TARGET_SR)]
    if seg.size < 512:
        cut = (lo + hi) / 2
    else:
        rms = pipeline._frame_rms(seg, TARGET_SR)
        cut = lo + int(np.argmin(rms)) * pipeline._frame_hop(TARGET_SR) / TARGET_SR
    return _split_long(wav, start, cut, max_dur) + _split_long(wav, cut, end, max_dur)


def group_regions(regions, wav: np.ndarray, min_dur: float = 3.0, max_dur: float = 30.0,
                  max_gap: float = 0.7, pad: float = 0.15) -> tuple[list[tuple[float, float]], int]:
    """VAD 区间并成 min_dur-max_dur 的候选；返回 (区间, 因过短丢弃的条数)。

    短剧台词本来就碎，隔 max_gap 以内的相邻区间合并成一条完整语流；
    不拼接没有真实连续时间关系的句子（合并只发生在同一段连续语音内）。
    """
    total = len(wav) / TARGET_SR
    merged: list[list[float]] = []
    for start, end in sorted(regions):
        s, e = max(0.0, start - pad), min(total, end + pad)
        if merged and s - merged[-1][1] <= max_gap and e - merged[-1][0] <= max_dur:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = [c for s, e in merged for c in _split_long(wav, s, e, max_dur)]
    kept = [(s, e) for s, e in out if e - s >= min_dur]
    return kept, len(out) - len(kept)


def list_batches(source_id: str) -> list[str]:
    root = pipeline.DATA_RAW / source_id / "ingest"
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "candidates.jsonl").exists())


def load_candidates(source_id: str, batch: str = ALL) -> list[dict]:
    """读候选清单；batch=ALL 时按素材 ID 顺序合并所有批次。"""
    rows = []
    for vid in ([batch] if batch and batch != ALL else list_batches(source_id)):
        p = ingest_dir(source_id, vid) / "candidates.jsonl"
        if p.exists():
            rows.extend(pipeline._read_manifest(p))
    return rows


def save_candidates(source_id: str, rows: list[dict]) -> int:
    """按素材 ID 分组原子写回各自的 candidates.jsonl（坏例与"丢弃"也留着，便于复查）。"""
    by_vid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vid[r["ingest_video"]].append(r)
    for vid, rs in by_vid.items():
        pipeline._write_jsonl(rs, ingest_dir(source_id, vid) / "candidates.jsonl")
    return len(rows)


def clip_label(i: int, row: dict) -> str:
    dur = float(row.get("end", 0)) - float(row.get("start", 0))
    mark = f"听不清({row['transcribe_error'].split(':')[0]})" if row.get("transcribe_error") \
        else (row.get("verdict") or "待定")
    return f"{row.get('ingest_video', '?')}:{i:04d} | {dur:4.1f}s | {mark} | {row.get('speaker') or '未标说话人'}"


def ingest_file(path: str | Path, source_id: str = "drama_tl", video_id: str | None = None,
                min_dur: float = 3.0, max_dur: float = 30.0, max_items: int | None = None,
                progress=None) -> dict:
    """解码 → VAD 定边界 → 切 3-30s → large-v3 逐条转写 + 语种过滤 → candidates.jsonl。

    转写走 pipeline._transcribe_manifest，因此每 100 条落盘、重跑跳过已转写行、
    坏例只排除不删除——万级素材中途挂掉不用白跑。
    """
    src = get_source(source_id)
    if src.kind != "local":
        raise ValueError(f"{source_id} 不是自备语料源；有下载入口的源用 python -m voxft.data.download")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"素材不存在: {path}")
    vid = _safe_id(video_id or path.stem)
    out = ingest_dir(source_id, vid)
    wav_path = decode_to_wav(path, out / "source.wav", progress)
    wav, _sr = pipeline.load_wav_mono(wav_path)
    regions = speech_regions(wav, src.lang, progress)
    if progress:
        progress(f"{vid}: VAD 找到 {len(regions)} 段语音，正在合并成 {min_dur}-{max_dur}s 候选")
    clips, too_short = group_regions(regions, wav, min_dur, max_dur)
    if max_items:
        clips = clips[:max_items]
    if not clips:
        raise RuntimeError(f"{vid}: 没切出 {min_dur}-{max_dur}s 的候选"
                           f"（VAD 区间 {len(regions)} 段，过短丢弃 {too_short} 段）")
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (s, e) in enumerate(clips):
        dst = clips_dir / f"{i:04d}_s{s:08.2f}_e{e:08.2f}.wav"
        sf.write(dst, wav[int(s * TARGET_SR):int(e * TARGET_SR)], TARGET_SR, subtype="PCM_16")
        rows.append({"audio": str(dst.resolve()), "lang": src.lang, "session": vid,
                     "ingest_video": vid, "start": round(s, 2), "end": round(e, 2),
                     "verdict": "待定"})
    candidates = out / "candidates.jsonl"
    if candidates.exists():   # 断点续跑：同一素材重跑跳过已转写的候选
        done = {r["audio"]: r for r in pipeline._read_manifest(candidates)}
        rows = [done.get(r["audio"], r) for r in rows]
    opts = pipeline.options_for(source_id)
    kept, bad = pipeline._transcribe_manifest(
        rows, src.lang, src.languages(), progress,
        checkpoint=lambda rs: pipeline._write_jsonl(rs, candidates), checkpoint_every=100,
        min_logprob=opts.asr_min_logprob, max_no_speech=opts.asr_max_no_speech)
    if progress:
        progress(f"{vid}: 切出 {len(rows)} 条，转写通过 {len(kept)}，排除 {bad}，过短丢弃 {too_short}")
    return {"video_id": vid, "output": str(out), "candidates": str(candidates),
            "clips": len(rows), "transcribed": len(kept), "dropped": bad, "too_short": too_short}


def validate_annotation(text, speaker, speaker_ok, emotion, emotion_ok,
                        control_zh, control_en, control_ok, verdict) -> dict:
    """人工标注是训练数据的入口，规则必须与加工阶段一致，别等加工时才炸。"""
    text = str(text or "").strip()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict 必须是 {'/'.join(VERDICTS)} 之一")
    if verdict == "保留" and not text:
        raise ValueError("标为「保留」的样本必须有台词文本")
    if text.startswith(("(", "（")):
        raise ValueError("原始 text 请提供裸台词，控制指令放 control_zh/control_en 列")
    speaker = str(speaker or "").strip()
    if speaker_ok and speaker in ("", "default"):
        raise ValueError("speaker_verified 必须为布尔值；可信身份必须提供真实 speaker ID")
    row = {"text": text, "verdict": verdict, "speaker": speaker,
           # 只有人工核实过身份才写 true：drama_tl 是 local 源，加工阶段不会替你默认
           "speaker_verified": bool(speaker_ok and speaker),
           "emotion": str(emotion or "").strip(),
           "emotion_verified": bool(emotion_ok), "control_verified": bool(control_ok)}
    for name, value in (("control_zh", control_zh), ("control_en", control_en)):
        if str(value or "").strip():
            row[name] = _assert_control_lang(str(value).strip())
    return row


def pin_holdout(source_id: str, video_ids) -> tuple[str, ...]:
    """把素材 ID 钉进验证集：追加后重新加工，这些素材永远不会被重排进训练集。"""
    pin = pipeline.DATA_RAW / source_id / "holdout.json"
    current = pipeline._load_holdout(pin.parent / "manifest.jsonl")
    sessions = sorted(set(current) | {_safe_id(v) for v in video_ids})
    tmp = pin.with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, pin)   # 钉住文件写坏会静默泄漏验证集，必须原子替换
    return tuple(sessions)


def append_to_source(source_id: str, rows: list[dict], holdout=(), progress=None) -> dict:
    """把标注为「保留」且转写通过的候选追加进 data/raw/<source>/manifest.jsonl。

    同一素材重切会替换它上次追加的行（按 ingest_video），再按音频绝对路径去重。
    holdout 是要钉进验证集的素材 ID（通常只挑一集永久当评测集），空=不动钉住文件。
    """
    accepted = [r for r in rows if r.get("verdict") == "保留" and r.get("text")
                and not r.get("transcribe_error")]
    if not accepted:
        return {"appended": 0, "total": 0,
                "reason": "没有标注为「保留」且转写通过的候选；先试听标注"}
    manifest = pipeline.DATA_RAW / source_id / "manifest.jsonl"
    old = pipeline._read_manifest(manifest) if manifest.exists() else []
    vids = sorted({r["ingest_video"] for r in accepted})
    replacing = set(vids)
    seen, merged = set(), []
    for r in [r for r in old if r.get("ingest_video") not in replacing] + accepted:
        key = str(Path(r["audio"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        row = {k: v for k, v in r.items() if k != "verdict"}
        row["audio"] = key
        merged.append(row)
    pipeline._write_jsonl(merged, manifest)
    pinned = list(pin_holdout(source_id, holdout)) if holdout else []
    if progress:
        progress(f"追加 {len(accepted)} 条 → {manifest}（素材 {vids}，清单共 {len(merged)} 条）")
        if holdout:
            progress(f"验证集钉住: {pinned}")
    return {"appended": len(accepted), "total": len(merged), "videos": vids,
            "holdout": pinned}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True,
                    help="远程视频/音频文件（mp4/mkv/wav/flac），可一次给多个")
    ap.add_argument("--source", default="drama_tl",
                    choices=[s.id for s in SOURCES if s.kind == "local"])
    ap.add_argument("--video-id", default=None,
                    help="素材 ID（空=文件名）。同时作为 session 与 holdout 键")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--max-items", type=int, default=None, help="试跑：只切前 N 条，不追加")
    ap.add_argument("--append", action="store_true",
                    help="把已标注为「保留」的候选追加进原始清单")
    ap.add_argument("--holdout", nargs="+", default=(), metavar="VID",
                    help="把这些素材 ID 钉进验证集（holdout.json）；通常只挑一集永久当评测集")
    ap.add_argument("--process", action="store_true", help="追加后重新加工")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.video_id and len(args.input) > 1:
        ap.error("--video-id 只能在单个 --input 时指定")
    if args.append and args.max_items:
        ap.error("--max-items 是试跑，不能与 --append 同时用")
    if (args.process or args.holdout) and not args.append:
        ap.error("--process/--holdout 需要 --append")

    summaries = [ingest_file(p, args.source, args.video_id, args.min_dur, args.max_dur,
                             args.max_items, progress=print) for p in args.input]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if args.append:
        rows = [r for s in summaries for r in load_candidates(args.source, s["video_id"])]
        print(json.dumps(append_to_source(args.source, rows, args.holdout, progress=print),
                         ensure_ascii=False, indent=2))
        if args.process:
            print(json.dumps(pipeline.process_dataset(
                args.source, args.out,
                pipeline.options_for(args.source, min_dur=args.min_dur, max_dur=args.max_dur),
                progress=print), ensure_ascii=False, indent=2))
