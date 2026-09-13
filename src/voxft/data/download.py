from __future__ import annotations

import argparse
import json
import re
import socket
import tarfile
import time
import urllib.request
from pathlib import Path

import soundfile as sf

from ..log import _fmt_size
from ..paths import DATA_RAW, env, load_dotenv
from .registry import SOURCES, Source, get_source, row_passes

_CJK = re.compile(r"[\u4e00-\u9fff]")

# 流式读经 fsspec/httpx，连接卡死时没有任何超时：实测 fleurs 流式取首行永久挂起，
# 进程零字节读入、无输出、无网络连接。setdefaulttimeout 是单次 recv 级别的，慢速但
# 仍在传输的下载不受影响，只有真卡死才抛。
_STREAM_SOCKET_TIMEOUT = 120

_TEXT_COLS = ("sentence", "text", "transcript", "transcription", "raw_transcription")
_SPK_COLS = ("client_id", "speaker_id", "speaker", "speaker_name")
_MISSING = {"", "none", "nan", "null"}


def _pick(candidates, columns) -> str | None:
    return next((c for c in candidates if c in columns), None)


def _detect_cols(row: dict, source: Source | None = None
                 ) -> tuple[str | None, str | None, str | None]:
    """探测音频/文本/说话人列名；数据源在 registry 里指定了列则优先用它的。"""
    cols = list(row)
    audio = (source.audio_column(cols) if source else None) or next(
        (k for k, v in row.items() if isinstance(v, dict) and "array" in v), None
    ) or ("audio" if "audio" in row else None)
    if source and source.audio_cols:
        audio = source.audio_column(cols)  # 显式麦克风白名单不能回退到 mic_zoom
    text = _pick((source.text_cols if source else ()) or _TEXT_COLS, cols)
    speaker = _pick((source.speaker_cols if source else ()) or _SPK_COLS, cols)
    return audio, text, speaker


def _clean_text(value) -> str:
    """把 'None'/'nan'/空 统一成空串（THAI-SER 的 impro 轮次 script_sent 就是字符串 'None'）。"""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING else text


def _aishell_text(value: str) -> str:
    """AISHELL-3 的正文是汉字与拼音交错排列，不把拼音当 TTS 正文。"""
    return "".join(_CJK.findall(value))


def _metadata(source: Source, get) -> dict:
    rec = {"source_id": source.id, "lang": source.lang,
           "speaker_verified": source.has_speaker,
           "emotion_verified": source.id == "thai_ser"}
    for col in ("audio_id", "utt_id", "uuid", "turn_type", "script_intensity",
                "situation_desc", "situation_turn", "actor_gender", "actor_age",
                "agreement", "grade_avg", "dnsmos_overall"):
        value = get(col)
        if _clean_text(value):
            rec[col] = value.item() if hasattr(value, "item") else value
    return rec


def _emotion(source: Source, value) -> str:
    """把情绪列的取值规范成小写名字；ClassLabel 整数按 label_names 解码。"""
    if value is None:
        return ""
    if source.label_names is not None and source.label_names:
        try:
            return source.label_names[int(value)]
        except (TypeError, ValueError, IndexError):
            pass
    return _clean_text(value).lower()


def _load_audio(value, source: Source, token: str = ""):
    """共享音频读取：内嵌 bytes、HF 地址/相对路径、流式已解码数组。"""
    import io
    from urllib.parse import urlparse

    import numpy as np
    from huggingface_hub import HfFileSystem, hf_hub_download

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError("音频列必须是单个音频，不能为空或包含多个音轨")
        value = value[0]
    if isinstance(value, dict) and value.get("array") is not None:
        wav = np.asarray(value["array"], dtype=np.float32)
        sr = value.get("sampling_rate")
        if wav.ndim != 1 or not wav.size or not sr:
            raise ValueError("已解码音频缺少单声道数组或采样率")
        return wav, int(sr)

    raw = value.get("bytes") if isinstance(value, dict) else value
    if isinstance(raw, (bytes, bytearray, memoryview)) and len(raw):
        location = io.BytesIO(bytes(raw))
    else:
        path = (value.get("path") or value.get("src")) if isinstance(value, dict) else value
        if not isinstance(path, str) or not path.strip():
            raise ValueError("音频列没有 bytes 或 path/src")
        revision = "main"
        if path.startswith("hf://"):
            resolved = HfFileSystem(token=token or None).resolve_path(path)
            if resolved.repo_type != "dataset" or resolved.repo_id != source.repo:
                raise ValueError("音频地址不属于当前数据集仓库")
            filename, revision = resolved.path_in_repo, resolved.revision
        elif urlparse(path).scheme:
            url = urlparse(path)
            hosts = {"huggingface.co", "hf-mirror.com", urlparse(env("HF_ENDPOINT")).netloc}
            if (url.scheme not in ("http", "https") or url.netloc not in hosts
                    or not url.path.startswith(f"/datasets/{source.repo}/resolve/")):
                raise ValueError("音频 URL 必须是当前 HF 数据集的 resolve 地址")
            filename, revision = _resolve_parquet_ref(path)
        else:
            filename = path
        if not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("音频相对路径无效")
        # 音频在原仓库的 main/固定 revision，不在 refs/convert/parquet；
        # 走官方下载器保留缓存、认证及 .env 的镜像/大盘配置。
        location = hf_hub_download(repo_id=source.repo, filename=filename,
                                   repo_type="dataset", revision=revision, token=token or None)
    wav, sr = sf.read(location, dtype="float32", always_2d=True)
    if not len(wav):
        raise ValueError("解码得到空音频")
    return wav.mean(axis=1), sr


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
                   token: str, progress=None) -> list[tuple[str, str]]:
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
        if progress:
            progress(f"[{repo}] parquet 分支遍历失败（{exc}），回退索引 API")

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
                  speaker: str | None, emotion: str = "",
                  session: str = "", metadata: dict | None = None) -> None:
    path = audio_dir / f"{n:07d}.wav"
    sf.write(path, wav, sr)
    rec = {**(metadata or {}), "audio": str(path), "text": text}
    speaker = _clean_text(speaker)
    if speaker:
        rec["speaker"] = speaker
    else:
        rec["speaker_verified"] = False
    if emotion:
        rec["emotion"] = emotion     # → 加工时转成 (情绪) 控制前缀
    if session:
        rec["session"] = session     # → 训练/验证集隔离
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


def _download_parquet(source: Source, files: list[tuple[str, str]], dest: Path,
                      max_samples: int | None, token: str, progress=None) -> int:
    """逐分片下载（hf_hub_download 自带断点续传与缓存）并解析。"""
    from functools import partial

    import pandas as pd
    from huggingface_hub import hf_hub_download

    from ..log import LogBar

    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.jsonl"
    bar = partial(LogBar, log=progress) if progress else None
    log = progress or print
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
            dl_kwargs = dict(repo_id=source.repo, filename=repo_file,
                             revision=revision,
                             repo_type="dataset", token=token or None)
            if bar:
                dl_kwargs["tqdm_class"] = bar
            for attempt in range(3):
                try:
                    local = hf_hub_download(**dl_kwargs)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"{source.id}: 分片 {name} 下载失败"
                                           f"（已重试 3 次）: {exc}") from exc
                    if progress:
                        progress(f"{source.id}: 分片 {name} 下载出错（{exc}），"
                                 f"5 秒后重试 {attempt + 2}/3")
                    time.sleep(5)
            if progress:
                size_mb = Path(local).stat().st_size / 1024 / 1024
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 下载完成"
                         f"（{size_mb:.1f}MB），解析中...")
            df = pd.read_parquet(local)
            cols = list(df.columns)
            a_col = source.audio_column(cols)
            t_col = _pick(source.text_cols or _TEXT_COLS, cols)
            s_col = _pick(source.speaker_cols or _SPK_COLS, cols)
            e_col = source.emotion_col if source.emotion_col in cols else None
            g_col = source.session_col if source.session_col in cols else None
            if a_col is None or (t_col is None and not source.needs_transcribe):
                if progress:
                    progress(f"{source.id}: 分片 {name} 缺少 audio/text 列"
                             f"（实际列: {cols}），跳过")
                continue
            if progress:
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 含 {len(df)} 条"
                         f"（音频列 {a_col}，文本列 {t_col or '无→待转写'}），写入音频...")
            before, dropped, missing_text, bad_audio = n, 0, 0, 0
            for _, row in df.iterrows():
                if max_samples is not None and n >= max_samples:
                    break
                if not row_passes(source, lambda c: row[c] if c in row else None):
                    dropped += 1
                    continue
                text = _clean_text(row[t_col]) if t_col else ""
                if t_col and not text and not source.needs_transcribe:
                    missing_text += 1
                    continue
                try:
                    wav, sr = _load_audio(row[a_col], source, token)
                except (sf.LibsndfileError, ValueError, TypeError) as exc:
                    bad_audio += 1
                    if bad_audio <= 3:
                        log(f"{source.id}: 分片 {name} 第 {row.name} 行音频读取失败：{type(exc).__name__}: {exc}")
                    continue
                _write_record(f, audio_dir, n, wav, sr, text,
                              str(row[s_col]) if s_col else None,
                              _emotion(source, row[e_col]) if e_col else "",
                              source.session_of(row[g_col]) if g_col else "",
                              _metadata(source, row.get))
                n += 1
                if progress and (n == 1 or n % 25 == 0):
                    progress(f"{source.id}: 已写入 {n} 条")
            log(f"{source.id}: 分片 {fi + 1}/{len(files)} 完成"
                f"（写入 {n - before} 条，行过滤 {dropped} 条，缺文本 {missing_text} 条，"
                f"音频读取失败 {bad_audio} 条，累计 {n} 条）")
            if bad_audio and n == before:
                raise RuntimeError(f"{source.id}: 分片 {name} 的候选音频全部读取失败；"
                                   "请看上方首批具体错误，不是行过滤或分片下载失败")
    return n


def _download_stream(source: Source, dest: Path, max_samples: int | None,
                     progress=None) -> int:
    from datasets import load_dataset

    kwargs = {"split": source.split, "streaming": True, "trust_remote_code": True}
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
    prev_timeout = socket.setdefaulttimeout(_STREAM_SOCKET_TIMEOUT)
    try:
        with manifest.open("w", encoding="utf-8") as f:
            for row in ds:
                if max_samples is not None and n >= max_samples:
                    break
                a_col, t_col, s_col = _detect_cols(row, source)
                if a_col is None or (t_col is None and not source.needs_transcribe):
                    continue
                if not row_passes(source, row.get):
                    continue
                text = _clean_text(row[t_col]) if t_col else ""
                if t_col and not text and not source.needs_transcribe:
                    continue
                array, sr = _load_audio(row[a_col], source, token)
                _write_record(f, audio_dir, n, array, sr, text,
                              str(row[s_col]) if s_col else None,
                              _emotion(source, row.get(source.emotion_col))
                              if source.emotion_col else "",
                              source.session_of(row.get(source.session_col))
                              if source.session_col else "", _metadata(source, row.get))
                n += 1
                if progress and n % 100 == 0:
                    progress(f"{source.id}: 已下载 {n} 条")
    except TimeoutError as exc:
        raise RuntimeError(
            f"{source.id}: 流式下载卡死（{_STREAM_SOCKET_TIMEOUT}s 未收到数据，已写入 {n} 条）。"
            "多为代理/镜像到该 repo 直链不通；parquet 路径可用时优先走它，"
            "或换 HF_ENDPOINT 后重试") from exc
    finally:
        socket.setdefaulttimeout(prev_timeout)
    return n


def _download_hf(source: Source, dest: Path, max_samples: int | None,
                 progress=None) -> int:
    token = env("HF_TOKEN")
    if progress:
        progress(f"{source.id}: 解析分片列表（repo={source.repo} "
                 f"config={source.config or '-'} split={source.split}）...")
    try:
        files = _parquet_files(source.repo, source.config, source.split, token, progress)
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
            text = _aishell_text(next((p for p in reversed(parts) if _CJK.search(p)), ""))
            if not text:
                continue
            audio = content.parent / "wav" / name[:7] / name
            if not audio.exists():
                audio = next(iter(dest.rglob(name)), None)
                if audio is None:
                    continue
            fout.write(json.dumps(
                {"audio": str(audio), "text": text,
                 "speaker": name[:7], "speaker_verified": True,
                 "source_id": source.id, "lang": "zh"},
                ensure_ascii=False) + "\n")
            n += 1
    return n


def download_source(source_id: str, max_samples: int | None = None,
                    progress=None) -> Path:
    """下载数据源到 data/raw/<id>/，返回目录。重复调用会覆盖 manifest。"""
    source = get_source(source_id)
    dest = DATA_RAW / source.id
    if source.kind == "local":
        raise ValueError(f"{source_id} 使用已审核的真人语料，请在加工页填写原始 JSONL 路径，"
                         f"或放到 {dest}/manifest.jsonl；此入口不下载或生成录音")
    if source.kind == "openslr":
        n = _download_aishell3(source, dest, max_samples, progress)
    else:
        n = _download_hf(source, dest, max_samples, progress)
    if n == 0:
        raise RuntimeError(f"{source_id}: 未下载到任何样本，请检查数据源/权限")
    if progress:
        progress(f"{source_id}: 完成，共 {n} 条 → {dest}/manifest.jsonl")
    return dest


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="下载微调数据集")
    ap.add_argument("--source", required=True, choices=[s.id for s in SOURCES])
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    download_source(args.source, args.max_samples, progress=print)


if __name__ == "__main__":
    main()
