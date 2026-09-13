"""parquet 索引条目解析自测（离线）。"""
import io
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from voxft.data.download import _resolve_parquet_ref


def test_resolve_parquet_ref():
    f, rev = _resolve_parquet_ref(
        "https://huggingface.co/api/datasets/google/fleurs/parquet/af_za/test/0.parquet")
    assert (f, rev) == ("af_za/test/0.parquet", "refs/convert/parquet")

    f, rev = _resolve_parquet_ref(
        "https://hf-mirror.com/api/datasets/x/y/parquet/tl/train/3.parquet")
    assert (f, rev) == ("tl/train/3.parquet", "refs/convert/parquet")

    f, rev = _resolve_parquet_ref(
        "https://huggingface.co/datasets/x/y/resolve/refs%2Fconvert%2Fparquet/a/0.parquet")
    assert rev == "refs/convert/parquet" and f == "a/0.parquet"

    f, rev = _resolve_parquet_ref("data/train-00000-of-00002.parquet")
    assert (f, rev) == ("data/train-00000-of-00002.parquet", "main")


def test_xet_disabled_only_on_mirror(monkeypatch):
    """镜像不代理 xet 的 CAS 服务器，不关会 401；直连官方时不该动它。"""
    import os

    from voxft import paths

    def _run(endpoint, preset=None):
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        if preset is not None:
            monkeypatch.setenv("HF_HUB_DISABLE_XET", preset)
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        paths._disable_xet_on_mirror()
        return os.environ.get("HF_HUB_DISABLE_XET")

    assert _run("https://hf-mirror.com") == "1"
    assert _run("https://huggingface.co") is None
    assert _run("https://hf-mirror.com", preset="0") == "0"   # 显式设置优先


def test_prefetch_builds_whisper_repo_and_uses_env(monkeypatch):
    """预取入口必须走 .env 的镜像设置，且能拼出 faster-whisper 仓库名。"""
    from voxft.data import prefetch as pf

    assert pf.WHISPER_REPO.format(size="large-v3") == "Systran/faster-whisper-large-v3"

    seen = {}
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kw: seen.update(kw) or "/tmp/snap")
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    assert pf.prefetch("Systran/faster-whisper-medium", progress=lambda m: None) == "/tmp/snap"
    assert seen["repo_id"] == "Systran/faster-whisper-medium"


def test_aishell_mixed_pinyin_and_source_metadata():
    from voxft.data.download import _aishell_text, _metadata, _detect_cols
    from voxft.data.registry import get_source, row_passes
    assert _aishell_text("广 guang3 州 zhou1，欢迎 huan1 ying2") == "广州欢迎"
    src = get_source("thai_ser")
    row = {"turn_type": "impro", "script_intensity": "high", "actor_gender": "female",
           "agreement": 0.9, "mic_zoom": {"array": [0]}}
    assert _metadata(src, row.get)["script_intensity"] == "high"
    assert _metadata(src, row.get)["emotion_verified"] is True
    assert _detect_cols(row, src)[0] is None
    assert row_passes(src, row.get)
    assert not row_passes(src, {**row, "agreement": float("nan")}.get)
    assert _metadata(get_source("yodas_th"), {}.get)["speaker_verified"] is False


def test_audio_bytes_paths_and_stream_share_reader(tmp_path, monkeypatch):
    """只用合成音频和模拟下载；不访问或下载真实语料。"""
    from types import SimpleNamespace
    from voxft.data.download import _load_audio
    from voxft.data.registry import get_source
    source = get_source("filswitch")
    wav = np.column_stack([np.full(160, 0.1), np.full(160, 0.3)])
    data = io.BytesIO()
    sf.write(data, wav, 16000, format="FLAC")
    local = tmp_path / "sample.flac"
    local.write_bytes(data.getvalue())
    downloads = []
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kw: downloads.append(kw) or str(local))
    monkeypatch.setattr("huggingface_hub.HfFileSystem.resolve_path", lambda self, path:
                        SimpleNamespace(repo_type="dataset", repo_id=source.repo,
                                        path_in_repo="train/sample.flac", revision="pinned"))
    forms = [
        {"bytes": data.getvalue(), "path": "unused.flac"},
        np.array([{"bytes": None, "path": f"hf://datasets/{source.repo}@pinned/train/sample.flac"}]),
        {"bytes": None, "path": f"https://huggingface.co/datasets/{source.repo}/resolve/pinned/train/sample.flac"},
        [{"src": f"https://hf-mirror.com/datasets/{source.repo}/resolve/pinned/train/sample.flac"}],
        "train/sample.flac",
        {"array": np.full(160, 0.2), "sampling_rate": 16000},
    ]
    for value in forms:
        mono, sr = _load_audio(value, source, "test-token")
        assert sr == 16000 and mono.shape == (160,)
        assert np.allclose(mono, 0.2, atol=1e-4)
    assert len(downloads) == 4
    assert [call["revision"] for call in downloads] == ["pinned", "pinned", "pinned", "main"]
    assert all(call["repo_id"] == source.repo and call["token"] == "test-token"
               and call["filename"] == "train/sample.flac" for call in downloads)
    for bad in ({"bytes": None, "path": None}, "/etc/passwd", "../outside.wav",
                "https://example.com/private.wav", ["a.wav", "b.wav"]):
        with pytest.raises(ValueError):
            _load_audio(bad, source)
    assert len(downloads) == 4  # 非当前 HF 仓库地址不带 token 发请求


def test_path_only_parquet_writes_audio_and_logs_bad_rows(tmp_path, monkeypatch):
    from voxft.data import download as dl
    from voxft.data.registry import get_source
    parquet = tmp_path / "metadata.parquet"
    frame = pd.DataFrame([
        {"audio": {"bytes": None, "path": "train/sample.flac"}, "text": "hello", "uuid": "ok"},
        {"audio": {"bytes": None, "path": None}, "text": "missing audio", "uuid": "bad"},
        {"audio": {"bytes": None, "path": None}, "text": "", "uuid": "empty text"},
        {"audio": {"bytes": b"broken", "path": None}, "text": "broken audio", "uuid": "broken"},
    ])
    frame.to_parquet(parquet)
    flac = tmp_path / "sample.flac"
    sf.write(flac, np.full(160, 0.2), 16000)
    calls = []
    def download(**kw):
        calls.append(kw)
        return str(parquet if kw["filename"].endswith(".parquet") else flac)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    log = []
    source = get_source("filswitch")
    count = dl._download_parquet(source, [("default/train/0000.parquet", "refs/convert/parquet")],
                                 tmp_path / "output", None, "", log.append)
    rows = [json.loads(line) for line in (tmp_path / "output/manifest.jsonl").read_text().splitlines()]
    assert count == len(rows) == 1 and rows[0]["uuid"] == "ok"
    assert sf.info(rows[0]["audio"]).frames == 160
    assert [call["revision"] for call in calls] == ["refs/convert/parquet", "main"]
    assert any("缺文本 1 条" in line and "音频读取失败 2 条" in line for line in log)
    assert any("音频列没有 bytes 或 path/src" in line for line in log)

    # 单个坏例可定位，全分片解码失败则中止，不再笼统提示检查权限。
    frame.iloc[[1, 3]].to_parquet(parquet)
    with pytest.raises(RuntimeError, match="候选音频全部读取失败"):
        dl._download_parquet(source, [("bad.parquet", "main")],
                             tmp_path / "failed", None, "", log.append)

    # HTTP/认证/下载异常不能被当成单条坏音频吞掉。
    frame.iloc[[0]].to_parquet(parquet)
    def network_error(**kw):
        if kw["filename"].endswith(".parquet"):
            return str(parquet)
        raise OSError("audio download unavailable")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", network_error)
    with pytest.raises(OSError, match="audio download unavailable"):
        dl._download_parquet(source, [("bad.parquet", "main")],
                             tmp_path / "network_failure", None, "", log.append)


def test_stream_stall_raises_instead_of_hanging(tmp_path, monkeypatch):
    """流式回退卡死要变成可诊断异常，且退出后恢复 socket 超时（UI 进程有长连接）。"""
    import socket

    from voxft.data import download as dl
    from voxft.data.registry import get_source

    class Stalled:
        def __iter__(self):
            # 迭代期间超时必须已生效，否则卡死仍然无声无息
            assert socket.getdefaulttimeout() == dl._STREAM_SOCKET_TIMEOUT
            raise TimeoutError("recv timed out")

    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: Stalled())
    before = socket.getdefaulttimeout()
    with pytest.raises(RuntimeError, match="流式下载卡死"):
        dl._download_stream(get_source("filswitch"), tmp_path / "out", 5)
    assert socket.getdefaulttimeout() == before
