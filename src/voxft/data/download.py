from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

from ..paths import DATA_RAW, env, load_dotenv
from .registry import SOURCES, Source, get_source

_CJK = re.compile(r"[\u4e00-\u9fff]")


def _detect_cols(row: dict) -> tuple[str | None, str | None, str | None]:
    """探测音频/文本/说话人列名，兼容不同数据集的命名习惯。"""
    audio = next(
        (k for k, v in row.items() if isinstance(v, dict) and "array" in v), None
    ) or ("audio" if "audio" in row else None)
    text = next((k for k in ("sentence", "text", "transcript") if k in row), None)
    speaker = next(
        (k for k in ("client_id", "speaker_id", "speaker", "speaker_name") if k in row),
        None,
    )
    return audio, text, speaker


def _download_hf(source: Source, dest: Path, max_samples: int | None,
                 progress=None) -> int:
    from datasets import load_dataset

    kwargs = {"split": source.split, "streaming": True}
    if source.config:
        kwargs["name"] = source.config
    token = env("HF_TOKEN")
    if token:
        kwargs["token"] = token
    try:
        ds = load_dataset(source.repo, **kwargs)
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "401" in msg or "403" in msg:
            raise RuntimeError(
                f"{source.repo} 为受限（gated）数据集：请先在 "
                f"https://huggingface.co/datasets/{source.repo} 页面同意条款，"
                f"并在 .env 配置有效的 HF_TOKEN"
            ) from exc
        raise

    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.jsonl"
    n = 0
    with manifest.open("w", encoding="utf-8") as f:
        for row in ds:
            if max_samples is not None and n >= max_samples:
                break
            a_col, t_col, s_col = _detect_cols(row)
            if a_col is None or t_col is None or not str(row[t_col]).strip():
                continue
            a = row[a_col]
            array, sr = a.get("array"), a.get("sampling_rate") or 16000
            path = audio_dir / f"{n:07d}.wav"
            sf.write(path, array, sr)
            rec = {"audio": str(path), "text": str(row[t_col]).strip()}
            if s_col:
                rec["speaker"] = str(row[s_col])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if progress and n % 100 == 0:
                progress(f"{source.id}: 已下载 {n} 条")
    return n


def _download_aishell3(source: Source, dest: Path, max_samples: int | None,
                       progress=None) -> int:
    tgz = dest / "data_aishell3.tgz"
    if not tgz.exists():
        tgz.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载 {source.repo} （约 20GB，请耐心等待）...")
        urllib.request.urlretrieve(source.repo, tgz)
    wav_root = dest / "data_aishell3"
    if not wav_root.exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(dest)
    content = wav_root / "train" / "content.txt"
    if not content.exists():
        candidates = list(dest.rglob("content.txt"))
        if not candidates:
            raise RuntimeError("AISHELL-3 解压后未找到 content.txt")
        content = candidates[0]
    manifest = dest / "manifest.jsonl"
    n = 0
    with content.open(encoding="utf-8") as fin, manifest.open("w", encoding="utf-8") as fout:
        for line in fin:
            if max_samples is not None and n >= max_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            name = parts[0]
            text = next((p for p in reversed(parts) if _CJK.search(p)), "")
            if not text:
                continue
            audio = wav_root / "train" / "wav" / name
            if not audio.exists():
                audio = next(iter(dest.rglob(name)), None)
                if audio is None:
                    continue
            fout.write(json.dumps(
                {"audio": str(audio), "text": text.replace(" ", ""),
                 "speaker": name[:7]},
                ensure_ascii=False) + "\n")
            n += 1
    return n


def download_source(source_id: str, max_samples: int | None = None,
                    progress=None) -> Path:
    """下载数据源到 data/raw/<id>/，返回目录。重复调用会覆盖 manifest。"""
    source = get_source(source_id)
    dest = DATA_RAW / source.id
    if source.kind == "openslr":
        n = _download_aishell3(source, dest, max_samples, progress)
    else:
        n = _download_hf(source, dest, max_samples, progress)
    if n == 0:
        raise RuntimeError(f"{source_id}: 未下载到任何样本，请检查数据源/权限")
    print(f"[{source_id}] 完成，共 {n} 条 → {dest}/manifest.jsonl")
    return dest


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="下载微调数据集")
    ap.add_argument("--source", required=True, choices=[s.id for s in SOURCES])
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    download_source(args.source, args.max_samples)


if __name__ == "__main__":
    main()
