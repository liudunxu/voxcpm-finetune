from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
import urllib.request
from pathlib import Path

import soundfile as sf

from ..log import _fmt_size
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


_GATED_HINT = ("为受限（gated）数据集：请先在 "
               "数据集页面同意条款，并在 .env 配置有效的 HF_TOKEN")


def _check_gated(exc: Exception, repo: str) -> None:
    msg = str(exc)
    if "gated" in msg.lower() or "401" in msg or "403" in msg:
        raise RuntimeError(f"{repo} {_GATED_HINT} "
                           f"(https://huggingface.co/datasets/{repo})") from exc


def _tree(endpoint: str, repo: str, path: str, token: str) -> list[dict]:
    import requests
    r = requests.get(
        f"{endpoint}/api/datasets/{repo}/tree/refs%2Fconvert%2Fparquet/{path}",
        headers={"Authorization": f"Bearer {token}"} if token else {},
        timeout=30)
    if r.status_code in (401, 403):
        _check_gated(RuntimeError(str(r.status_code)), repo)
    r.raise_for_status()
    return r.json()


def _parquet_files(repo: str, config: str, split: str,
                   token: str) -> list[tuple[str, str]]:
    """返回 (仓库内相对路径, revision) 列表。

    首选遍历 refs/convert/parquet 分支树（可精确过滤 config/split）；
    失败时回退到 /parquet 索引 API（无法过滤，返回全部条目）。
    """
    import os
    import requests

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    try:
        root = config if config else ""
        files: list[tuple[str, str]] = []
        subdirs = [root, f"{root}/{split}"] if root else []
        # 无 config 的数据集：先列出顶层目录作为候选
        if not root:
            top = _tree(endpoint, repo, "", token)
            dirs = [e["path"] for e in top if e["type"] == "directory"]
            subdirs = [f"{d}/{split}" for d in dirs] if split else dirs
        for sub in subdirs:
            try:
                entries = _tree(endpoint, repo, sub, token)
            except Exception:
                continue
            for e in entries:
                if not e["path"].endswith(".parquet"):
                    continue
                # tree API 返回的 path 相对所查目录，需补全目录前缀
                p = e["path"] if (not sub or e["path"].startswith(sub + "/")) \
                    else f"{sub}/{e['path']}"
                files.append((p, "refs/convert/parquet"))
        if files:
            return files
    except Exception as exc:
        _check_gated(exc, repo)
        print(f"[{repo}] parquet 分支遍历失败（{exc}），回退索引 API")

    params, headers = {}, {}
    if config:
        params["config"] = config
    if split:
        params["split"] = split
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{endpoint}/api/datasets/{repo}/parquet",
                     params=params, headers=headers, timeout=30)
    if r.status_code in (401, 403):
        _check_gated(RuntimeError(str(r.status_code)), repo)
    r.raise_for_status()
    out: list[tuple[str, str]] = []
    for cfg, splits in r.json().items():
        for sp, fl in splits.items():
            if config and cfg != config:
                continue
            if split and sp != split:
                continue
            out.extend((u, "") for u in fl)
    return out


def _write_record(f, audio_dir: Path, n: int, wav, sr: int, text: str,
                  speaker: str | None) -> None:
    path = audio_dir / f"{n:07d}.wav"
    sf.write(path, wav, sr)
    rec = {"audio": str(path), "text": text}
    if speaker:
        rec["speaker"] = speaker
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _resolve_parquet_ref(entry: str) -> tuple[str, str]:
    """把 parquet 索引条目解析为 (仓库内相对路径, git revision)。

    索引可能返回三种形式：
    - API URL: .../api/datasets/<repo>/parquet/<cfg>/<split>/0.parquet
      → parquet 导出实际位于分支 refs/convert/parquet
    - resolve URL: .../resolve/<rev>/<path>
    - 纯相对路径
    """
    if not entry.startswith("http"):
        return entry.lstrip("/"), "main"
    from urllib.parse import unquote, urlparse
    path = urlparse(entry).path
    if "/parquet/" in path:
        return path.split("/parquet/", 1)[1], "refs/convert/parquet"
    if "/resolve/" in path:
        rest = path.split("/resolve/", 1)[1]
        rev, _, fp = rest.partition("/")
        return unquote(fp), unquote(rev)
    return path.lstrip("/"), "main"


def _download_parquet(source: Source, files: list[str], dest: Path,
                      max_samples: int | None, token: str, progress=None) -> int:
    """逐分片下载（hf_hub_download 自带断点续传与缓存）并解析。"""
    import io
    from functools import partial

    import pandas as pd
    import torchaudio
    from huggingface_hub import hf_hub_download

    from ..log import LogBar

    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.jsonl"
    bar = partial(LogBar, log=progress) if progress else None
    n = 0
    with manifest.open("w", encoding="utf-8") as f:
        for fi, (repo_file, revision) in enumerate(files):
            if max_samples is not None and n >= max_samples:
                break
            if not revision:  # 索引 API 回退路径：条目是 URL
                repo_file, revision = _resolve_parquet_ref(repo_file)
            name = Path(repo_file).name
            if progress:
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 下载 {name}")
            local = hf_hub_download(repo_id=source.repo, filename=repo_file,
                                    revision=revision,
                                    repo_type="dataset", token=token or None,
                                    **({"tqdm_class": bar} if bar else {}))
            if progress:
                size_mb = Path(local).stat().st_size / 1024 / 1024
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 下载完成"
                         f"（{size_mb:.1f}MB），解析中...")
            df = pd.read_parquet(local)
            t_col = next((c for c in ("sentence", "text", "transcript",
                                      "transcription", "raw_transcription")
                          if c in df.columns), None)
            s_col = next((c for c in ("client_id", "speaker_id", "speaker", "speaker_name")
                          if c in df.columns), None)
            if t_col is None or "audio" not in df.columns:
                if progress:
                    progress(f"{source.id}: 分片 {name} 缺少 audio/text 列"
                             f"（实际列: {list(df.columns)}），跳过")
                continue
            if progress:
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 含 {len(df)} 条，写入音频...")
            before = n
            for _, row in df.iterrows():
                if max_samples is not None and n >= max_samples:
                    break
                text = str(row[t_col] or "").strip()
                if not text:
                    continue
                a = row["audio"]
                raw = a["bytes"] if isinstance(a, dict) else a
                if raw is None:
                    continue
                try:
                    wav, sr = torchaudio.load(io.BytesIO(raw))
                except Exception:
                    continue
                _write_record(f, audio_dir, n, wav.mean(0).numpy(), sr, text,
                              str(row[s_col]) if s_col else None)
                n += 1
                if progress and n % 200 == 0:
                    progress(f"{source.id}: 已写入 {n} 条")
            if progress:
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 完成"
                         f"（本分片写入 {n - before} 条，累计 {n} 条）")
    return n


def _download_stream(source: Source, dest: Path, max_samples: int | None,
                     progress=None) -> int:
    from datasets import load_dataset

    kwargs = {"split": source.split, "streaming": True}
    if source.config:
        kwargs["name"] = source.config
    token = env("HF_TOKEN")
    if token:
        kwargs["token"] = token
    if progress:
        progress(f"{source.id}: 建立流式连接...")
    try:
        ds = load_dataset(source.repo, **kwargs)
    except Exception as exc:
        _check_gated(exc, source.repo)
        raise
    if progress:
        progress(f"{source.id}: 流式下载中（逐条写入）")

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
            _write_record(f, audio_dir, n, array, sr, str(row[t_col]).strip(),
                          str(row[s_col]) if s_col else None)
            n += 1
            if progress and n % 100 == 0:
                progress(f"{source.id}: 已下载 {n} 条")
    return n


def _download_hf(source: Source, dest: Path, max_samples: int | None,
                 progress=None) -> int:
    token = env("HF_TOKEN")
    if progress:
        progress(f"{source.id}: 解析分片列表（repo={source.repo} "
                 f"config={source.config or '-'} split={source.split}）...")
    try:
        files = _parquet_files(source.repo, source.config, source.split, token)
    except Exception as exc:
        _check_gated(exc, source.repo)
        if progress:
            progress(f"{source.id}: parquet 索引不可用（{exc}），回退流式下载")
        files = []
    if files:
        if progress:
            progress(f"{source.id}: 共 {len(files)} 个分片待下载")
        return _download_parquet(source, files, dest, max_samples, token, progress)
    return _download_stream(source, dest, max_samples, progress)


def _download_aishell3(source: Source, dest: Path, max_samples: int | None,
                       progress=None) -> int:
    tgz = dest / "data_aishell3.tgz"
    if not tgz.exists():
        tgz.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(f"{source.id}: 开始下载 {source.repo}（约 20GB，请耐心等待）")
        with urllib.request.urlopen(source.repo) as resp, tgz.open("wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last = 0, 0.0
            while True:
                chunk = resp.read(1 << 20)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if progress and now - last >= 5:  # 每 5 秒报一次进度
                    last = now
                    if total:
                        progress(f"{source.id}: 已下载 {_fmt_size(done)}"
                                 f"/{_fmt_size(total)} ({100 * done / total:.1f}%)")
                    else:
                        progress(f"{source.id}: 已下载 {_fmt_size(done)}")
        if progress:
            progress(f"{source.id}: 下载完成（{_fmt_size(done)}）")
    wav_root = dest / "data_aishell3"
    if not wav_root.exists():
        if progress:
            progress(f"{source.id}: 解压 data_aishell3.tgz（约 20GB，耗时较长）...")
        with tarfile.open(tgz) as tf:
            tf.extractall(dest)
        if progress:
            progress(f"{source.id}: 解压完成")
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
    if progress:
        progress(f"{source_id}: 完成，共 {n} 条 → {dest}/manifest.jsonl")
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
