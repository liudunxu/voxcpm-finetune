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

# raw_transcription 排在 transcription 之前：FLEURS 的 transcription 是全小写、去标点的
# 归一变体，而线上送进模型的是保留大小写与标点的原始台词。标点承载停顿与句末收束信号，
# 训成无标点文本会让"生成停不下来"少一半可用线索。
_TEXT_COLS = ("sentence", "text", "transcript", "raw_transcription", "transcription")
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


def _tree(endpoint: str, repo: str, path: str, token: str,
          revision: str = "refs%2Fconvert%2Fparquet") -> list[dict]:
    import requests
    r = requests.get(
        f"{endpoint}/api/datasets/{repo}/tree/{revision}/{path}",
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


def _is_404(exc: Exception) -> bool:
    """只认「资源不存在」；网络/权限等其它错误不许掉进直链回退。"""
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    if isinstance(exc, EntryNotFoundError):
        return True
    resp = getattr(exc, "response", None)
    return isinstance(exc, HfHubHTTPError) and resp is not None \
        and resp.status_code == 404


def _download_parquet_url(url: str, dest: Path, token: str, progress=None) -> str:
    """parquet 索引 API URL 的直链下载，落到 dest 下的临时文件并续传。

    实测（2026-09-15，sarulab-speech/yodas2_sidon ms000）：仓库没有 refs/convert/parquet
    分支时索引 API 照样返回 URL，解析成分支路径下载必 404，但 API URL 本身跟随跳转可
    直接 GET（Range 返回 206），镜像侧同样可用。主机改写到 HF_ENDPOINT；残留的
    .dl-* 临时文件下次按 Range 续传。
    """
    from urllib.parse import urlparse, urlunparse

    import requests

    parts = urlparse(url)
    endpoint = env("HF_ENDPOINT").rstrip("/")
    if endpoint:
        mirror = urlparse(endpoint)
        if mirror.netloc and mirror.netloc != parts.netloc:
            parts = parts._replace(scheme=mirror.scheme, netloc=mirror.netloc)
            url = urlunparse(parts)
    tmp = dest / f".dl-{Path(parts.path).name}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for attempt in range(3):
        pos = tmp.stat().st_size if tmp.exists() else 0
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:      # Range 越过末尾：已下完
                    return str(tmp)
                r.raise_for_status()
                # 服务端不理会 Range（200）就从头重写，只在真续传（206）时追加
                with tmp.open("ab" if pos and r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return str(tmp)
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"索引 API URL 直链下载失败（已重试 3 次）: {exc}") from exc
            if progress:
                progress(f"直链下载出错（{exc}），5 秒后重试 {attempt + 2}/3")
            time.sleep(5)
    raise RuntimeError("unreachable")


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
                api_url, repo_file, revision = repo_file, *_resolve_parquet_ref(repo_file)
            else:
                api_url = ""
            name = Path(repo_file).name
            if progress:
                progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 下载 {name}")
            dl_kwargs = dict(repo_id=source.repo, filename=repo_file,
                             revision=revision,
                             repo_type="dataset", token=token or None)
            if bar:
                dl_kwargs["tqdm_class"] = bar
            local = None
            for attempt in range(3):
                try:
                    local = hf_hub_download(**dl_kwargs)
                    break
                except Exception as exc:
                    # 仓库没有 refs/convert/parquet 分支时 404 重试无意义，
                    # 直接回退到索引 API 的原始 URL 直链下载（yodas2_sidon 实测）
                    if api_url and _is_404(exc):
                        break
                    if attempt == 2:
                        raise RuntimeError(f"{source.id}: 分片 {name} 下载失败"
                                           f"（已重试 3 次）: {exc}") from exc
                    if progress:
                        progress(f"{source.id}: 分片 {name} 下载出错（{exc}），"
                                 f"5 秒后重试 {attempt + 2}/3")
                    time.sleep(5)
            if local is None:
                if progress:
                    progress(f"{source.id}: parquet 分支 404，回退索引 API URL 直链下载 {name}")
                local = _download_parquet_url(api_url, dest, token, progress)
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
            # WebDataset 原样成员列（yodas2_sidon 实测）：flac 是 HF Audio 结构
            # {bytes=整段视频音频, path}，metadata.json 给逐句 start/end/text/utt_id
            # 与 video_id——一行一个视频，按 utterances 切成句级样本
            wds = a_col is None and "flac" in cols and "metadata.json" in cols
            if not wds and (a_col is None or (t_col is None and not source.needs_transcribe)):
                if progress:
                    progress(f"{source.id}: 分片 {name} 缺少 audio/text 列"
                             f"（实际列: {cols}），跳过")
                continue
            if wds:
                if progress:
                    progress(f"{source.id}: 分片 {fi + 1}/{len(files)} 含 {len(df)} 个视频"
                             "（WebDataset 成员列，按 utterances 切句），写入音频...")
                before, missing_text, bad_audio = n, 0, 0
                for _, row in df.iterrows():
                    if max_samples is not None and n >= max_samples:
                        break
                    md = row["metadata.json"] or {}
                    utts = md.get("utterances") or {}

                    def _lst(key):
                        v = utts.get(key)
                        return [] if v is None else list(v)

                    texts = _lst("text")
                    if not texts:        # 无句级标注的视频不解码，直接跳过
                        continue
                    starts, ends, utt_ids = _lst("start"), _lst("end"), _lst("utt_id")
                    try:
                        wav, sr = _load_audio(row["flac"], source, token)
                    except (sf.LibsndfileError, ValueError, TypeError) as exc:
                        bad_audio += 1
                        if bad_audio <= 3:
                            log(f"{source.id}: 分片 {name} 第 {row.name} 行音频读取失败：{type(exc).__name__}: {exc}")
                        continue
                    session = _clean_text(md.get("video_id"))
                    for ui, utext in enumerate(texts):
                        if max_samples is not None and n >= max_samples:
                            break
                        text = _clean_text(utext)
                        if not text:
                            missing_text += 1
                            continue
                        try:
                            s = max(0, int(round(float(starts[ui]) * sr)))
                            e = min(len(wav), int(round(float(ends[ui]) * sr)))
                        except (IndexError, TypeError, ValueError):
                            bad_audio += 1
                            continue
                        clip = wav[s:e]
                        if not clip.size:
                            bad_audio += 1
                            continue
                        # utt_id 形如 <video_id>-00000-00001694-00002270，
                        # rsplit("-", 3)[0] 即视频 ID（同 yodas_th 约定）；
                        # 优先用显式 video_id 字段。speaker 是视频级近似身份，不强凑
                        utt_id = str(utt_ids[ui]) if ui < len(utt_ids) else ""
                        _write_record(f, audio_dir, n, clip, sr, text, None, "",
                                      session=session or utt_id.rsplit("-", 3)[0],
                                      metadata=_metadata(source, {"utt_id": utt_id}.get))
                        n += 1
                        if progress and (n == 1 or n % 200 == 0):
                            progress(f"{source.id}: 已写入 {n} 条")
                log(f"{source.id}: 分片 {fi + 1}/{len(files)} 完成"
                    f"（写入 {n - before} 条，缺文本 {missing_text} 条，"
                    f"音频/切片失败 {bad_audio} 条，累计 {n} 条）")
                if bad_audio and n == before:
                    raise RuntimeError(f"{source.id}: 分片 {name} 的候选音频全部读取失败；"
                                       "请看上方首批具体错误，不是分片下载失败")
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


def _sentence_case(text: str) -> str:
    """全大写转写 → 句首大写。

    gigaspeech2 的 tsv 是全大写，而线上送进模型的是句首大写的原始台词。越南语/印尼语的
    大小写不参与语法，str.lower() 对变音符号是安全的；代价是句内英文专有名词会被小写，
    没有真实大小写恢复器之前先接受这个损失。
    """
    t = text.strip()
    return t[:1].upper() + t[1:].lower() if t else t


def _download_hf_tar(source: Source, dest: Path, max_samples: int | None,
                     token: str, progress=None) -> int:
    """WebDataset 布局：`data/<config>/<split>.tar.gz`（每条一个 wav）+ 同名 `.tsv`（id\\t文本）。

    gigaspeech2 就是这个形态。它的 `refs/convert/parquet` 分支根本不存在，但 parquet 索引
    API 照样返回 200 和一串 URL，`_resolve_parquet_ref` 解析后下载必 404——所以只能按仓库内
    真实路径取。dev/test 分片约 1GB/语种（8-9h），train 单片 3.4-6.6GB，优先用 dev。
    """
    import tarfile
    from functools import partial

    from huggingface_hub import hf_hub_download

    from ..log import LogBar

    log = progress or print
    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"data/{source.config}/{source.split}"
    kw = dict(repo_id=source.repo, repo_type="dataset", token=token or None)
    tsv = hf_hub_download(filename=f"{prefix}.tsv", **kw)
    texts: dict[str, str] = {}
    with open(tsv, encoding="utf-8") as fh:
        for line in fh:
            uid, _, text = line.rstrip("\n").partition("\t")
            if uid and text.strip():
                texts[uid] = text
    if not texts:
        raise RuntimeError(f"{source.id}: {prefix}.tsv 没解析出任何 id<TAB>文本")
    log(f"{source.id}: TSV 转写 {len(texts)} 条；开始取 {prefix}.tar.gz（整片下载，"
        "约 1GB，进度条不细分到条目）")
    bar = partial(LogBar, log=progress) if progress else None
    tar_path = hf_hub_download(filename=f"{prefix}.tar.gz",
                               **({"tqdm_class": bar} if bar else {}), **kw)

    n = no_text = bad = 0
    with (dest / "manifest.jsonl").open("w", encoding="utf-8") as out, \
            tarfile.open(tar_path, "r:*") as tf:
        for member in tf:
            if max_samples is not None and n >= max_samples:
                break
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            uid = Path(member.name).stem          # dev/22/22-52.wav → 22-52
            text = _clean_text(texts.get(uid))
            if not text:
                no_text += 1
                continue
            handle = tf.extractfile(member)
            if handle is None:
                bad += 1
                continue
            try:
                wav, sr = _load_audio(handle.read(), source, token)
            except (sf.LibsndfileError, ValueError, TypeError) as exc:
                bad += 1
                if bad <= 3:
                    log(f"{source.id}: {member.name} 音频读取失败：{type(exc).__name__}: {exc}")
                continue
            if source.sentence_case:
                text = _sentence_case(text)
            # tar 内路径的第二级就是 YouTube 视频 ID，用它做会话隔离，
            # 否则同一视频的切片会跨 train/val 泄漏
            _write_record(out, audio_dir, n, wav, sr, text, None, "",
                          session=uid.split("-")[0], metadata=_metadata(source, {}.get))
            n += 1
            if progress and (n == 1 or n % 200 == 0):
                progress(f"{source.id}: 已写入 {n} 条")
    log(f"{source.id}: tar 遍历结束（写入 {n} 条，无转写 {no_text} 条，音频读取失败 {bad} 条）")
    return n


# Common Voice 22 只取官方切分；other 是未进切分的 validated 余量、invalidated 是
# 被投票否决的，都不进训练。
_CV22_SPLITS = ("train", "dev", "test")


def _download_cv22(source: Source, dest: Path, max_samples: int | None,
                   token: str, progress=None) -> int:
    """Common Voice 22 社区镜像（fsicoli/common_voice_22_0）的自包含下载。

    仓库是脚本式数据集，加载脚本把数据文件 URL 硬编码到 huggingface.co，
    HF_ENDPOINT 管不到，国内直连必失败——绕开脚本按真实布局直拉（2026-09-15
    镜像 tree API 核实）：`transcript/<lang>/<split>.tsv`（表头
    client_id/path/sentence_id/sentence/...）+ `audio/<lang>/<split>/
    <lang>_<split>_<n>.tar`（成员 `<lang>_<split>_<n>/common_voice_<lang>_<id>.mp3`，
    48kHz，libsndfile 1.2 直接解码）。
    client_id 是账号级持久身份（非聚类猜测）：写进 speaker 与 session（供 train/val
    隔离），r9 起 registry 标 has_speaker=True，经 _metadata 自动标
    speaker_verified、参与 ref 配对；只做匿名分组，不识别真人（CV 条款红线）。
    """
    import csv
    from functools import partial

    from huggingface_hub import hf_hub_download

    from ..log import LogBar

    log = progress or print
    audio_dir = dest / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(repo_id=source.repo, repo_type="dataset", token=token or None)
    bar = partial(LogBar, log=progress) if progress else None
    n = 0
    with (dest / "manifest.jsonl").open("w", encoding="utf-8") as out:
        for split in _CV22_SPLITS:
            if max_samples is not None and n >= max_samples:
                break
            tsv = hf_hub_download(
                filename=f"transcript/{source.config}/{split}.tsv", **kw)
            clips: dict[str, tuple[str, str]] = {}
            with open(tsv, encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    text = _clean_text(row.get("sentence"))
                    if row.get("path") and text:
                        clips[row["path"]] = (text, _clean_text(row.get("client_id")))
            if not clips:
                raise RuntimeError(f"{source.id}: transcript/{source.config}/{split}.tsv "
                                   "没解析出任何条目")
            endpoint = env("HF_ENDPOINT").rstrip("/") or "https://huggingface.co"
            tars = sorted(e["path"] for e in
                          _tree(endpoint, source.repo,
                                f"audio/{source.config}/{split}", token, "main")
                          if e["type"] == "file" and e["path"].endswith(".tar"))
            if not tars:
                raise RuntimeError(f"{source.id}: audio/{source.config}/{split} 下没有 tar")
            log(f"{source.id}: split {split} 转写 {len(clips)} 条，{len(tars)} 个 tar")
            for ti, tar_name in enumerate(tars):
                if max_samples is not None and n >= max_samples:
                    break
                tar_path = hf_hub_download(
                    filename=tar_name,
                    **({"tqdm_class": bar} if bar else {}), **kw)
                no_text = bad = 0
                with tarfile.open(tar_path, "r:*") as tf:
                    for member in tf:
                        if max_samples is not None and n >= max_samples:
                            break
                        if not member.isfile() or not member.name.endswith(".mp3"):
                            continue
                        hit = clips.get(Path(member.name).name)
                        if not hit:
                            no_text += 1
                            continue
                        handle = tf.extractfile(member)
                        if handle is None:
                            bad += 1
                            continue
                        try:
                            wav, sr = _load_audio(handle.read(), source, token)
                        except (sf.LibsndfileError, ValueError, TypeError) as exc:
                            bad += 1
                            if bad <= 3:
                                log(f"{source.id}: {member.name} 音频读取失败："
                                    f"{type(exc).__name__}: {exc}")
                            continue
                        text, client_id = hit
                        _write_record(out, audio_dir, n, wav, sr, text,
                                      client_id or None, "", session=client_id,
                                      metadata=_metadata(source, {}.get))
                        n += 1
                        if progress and (n == 1 or n % 200 == 0):
                            progress(f"{source.id}: 已写入 {n} 条")
                log(f"{source.id}: {split} 分片 {ti + 1}/{len(tars)} 完成"
                    f"（累计 {n} 条，无转写 {no_text} 条，音频读取失败 {bad} 条）")
    return n


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
    elif source.kind == "hf_tar":
        n = _download_hf_tar(source, dest, max_samples, env("HF_TOKEN"), progress)
    elif source.kind == "cv22":
        n = _download_cv22(source, dest, max_samples, env("HF_TOKEN"), progress)
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
