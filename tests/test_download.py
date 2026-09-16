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


def test_fleurs_prefers_verbatim_over_normalized_text():
    """线上送进模型的是带标点与大小写的原始台词；FLEURS 的 transcription 是全小写去标点
    的归一变体，选错会让训练文本与推理文本形态不一致（标点还承载句末收束信号）。"""
    from voxft.data.download import _detect_cols
    from voxft.data.registry import get_source
    row = {"audio": {"array": [0.0]}, "transcription": "walang punctuation dito",
           "raw_transcription": "Walang punctuation dito."}
    assert _detect_cols(row, get_source("fleurs_tl"))[1] == "raw_transcription"


def test_sentence_case_keeps_vietnamese_diacritics():
    """gigaspeech2 的 tsv 整库全大写；.lower() 必须保住越南语变音符号，
    否则声调信息就没了（vi 是 6 声调语言）。"""
    from voxft.data.download import _sentence_case
    assert _sentence_case("TRONG MỘT THẾ GIỚI LUÔN THAY ĐỔI") == "Trong một thế giới luôn thay đổi"
    assert _sentence_case("  GIÁ VÉ LÀ 250.000 ĐỒNG ") == "Giá vé là 250.000 đồng"
    assert _sentence_case("") == ""


def test_hf_tar_reads_webdataset_and_isolates_sessions(tmp_path, monkeypatch):
    """gigaspeech2 布局：tar 内每条一个 wav + 同名 tsv 给 id\\t文本。
    会话必须取 YouTube 视频 ID，否则同一视频的切片会跨 train/val 泄漏。"""
    import io
    import tarfile

    import numpy as np
    import soundfile as sf
    from voxft.data import download as dl
    from voxft.data.registry import get_source

    src = get_source("gigaspeech2_vi")
    assert src.kind == "hf_tar" and src.sentence_case is True

    tsv = tmp_path / "dev.tsv"
    tsv.write_text("7-1\tXIN CHÀO ANH\n7-2\tGIÁ VÉ LÀ 250.000 ĐỒNG\n9-1\tTẠM BIỆT\n",
                   encoding="utf-8")
    tar = tmp_path / "dev.tar.gz"

    def _wav(seconds):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
        return buf.getvalue()

    with tarfile.open(tar, "w:gz") as tf:
        for name, payload in (("dev/7/7-1.wav", _wav(0.4)), ("dev/7/7-2.wav", _wav(0.5)),
                              ("dev/9/9-1.wav", _wav(0.3)), ("dev/7/7-9.wav", _wav(0.3)),
                              ("dev/", b"")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))

    paths = {"data/vi/dev.tsv": str(tsv), "data/vi/dev.tar.gz": str(tar)}
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda filename=None, **kw: paths[filename])
    n = dl._download_hf_tar(src, tmp_path / "out", None, "tok", progress=lambda m: None)
    assert n == 3                      # 7-9 没有转写，必须跳过而不是写成空文本
    rows = [__import__("json").loads(line)
            for line in (tmp_path / "out" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == ["Xin chào anh", "Giá vé là 250.000 đồng", "Tạm biệt"]
    assert [r["session"] for r in rows] == ["7", "7", "9"]
    assert all(r["lang"] == "vi" and r["speaker_verified"] is False for r in rows)

    # max_samples 要在写满即止，不能把整片 tar 遍历完
    n2 = dl._download_hf_tar(src, tmp_path / "out2", 2, "tok", progress=None)
    assert n2 == 2


def test_cv22_reads_tsv_and_tar_without_loader_script(tmp_path, monkeypatch):
    """cv22 绕开把 URL 硬编码到 huggingface.co 的加载脚本，按真实布局直拉：
    transcript/<lang>/<split>.tsv（client_id/path/sentence 表头）+
    audio/<lang>/<split>/<lang>_<split>_<n>.tar（成员名前缀一层同名目录）。
    client_id 是账号级持久身份：进 speaker/session 供 train/val 隔离，
    r9 起 has_speaker=True、标 speaker_verified 并参与 ref 配对。"""
    import io
    import tarfile

    import numpy as np
    import soundfile as sf
    from voxft.data import download as dl
    from voxft.data.registry import get_source

    src = get_source("cv22_th")
    assert src.kind == "cv22" and src.has_speaker

    def _mp3(seconds):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(48000 * seconds), dtype=np.float32), 48000,
                 format="MP3")
        return buf.getvalue()

    header = "client_id\tpath\tsentence_id\tsentence\tsentence_domain\n"
    tsvs = {
        "train": header + "client-a\tcommon_voice_th_1.mp3\ts1\tประโยคที่หนึ่ง\t\n"
                          "client-a\tcommon_voice_th_2.mp3\ts2\tประโยคที่สอง\t\n"
                          "client-b\tcommon_voice_th_3.mp3\ts3\t\t\n",  # 空文本跳过
        "dev": header + "client-c\tcommon_voice_th_4.mp3\ts4\tประโยคที่สี่\t\n",
        "test": header + "client-d\tcommon_voice_th_5.mp3\ts5\tประโยคที่ห้า\t\n",
    }
    files = {}
    for split, body in tsvs.items():
        p = tmp_path / f"{split}.tsv"
        p.write_text(body, encoding="utf-8")
        files[f"transcript/th/{split}.tsv"] = str(p)
    tar = tmp_path / "th_train_0.tar"
    with tarfile.open(tar, "w") as tf:
        for name, payload in (("th_train_0/common_voice_th_1.mp3", _mp3(0.2)),
                              ("th_train_0/common_voice_th_2.mp3", _mp3(0.2)),
                              ("th_train_0/common_voice_th_3.mp3", _mp3(0.2)),
                              ("th_train_0/common_voice_th_9.mp3", _mp3(0.2)),  # 无转写
                              ("th_train_0/", b"")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    files["audio/th/train/th_train_0.tar"] = str(tar)
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda filename=None, **kw: files[filename])
    monkeypatch.setattr(dl, "_tree", lambda *a, **k: [
        {"type": "file", "path": "audio/th/train/th_train_0.tar"}])

    log = []
    n = dl._download_cv22(src, tmp_path / "out", None, "tok", log.append)
    assert n == 2   # 空文本与无转写各丢一条
    rows = [json.loads(line)
            for line in (tmp_path / "out/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == ["ประโยคที่หนึ่ง", "ประโยคที่สอง"]
    assert [r["speaker"] for r in rows] == ["client-a", "client-a"]
    assert [r["session"] for r in rows] == ["client-a", "client-a"]
    assert all(r["speaker_verified"] is True and r["lang"] == "th" for r in rows)
    assert any("无转写 2 条" in line for line in log)  # 空文本行也不进 clips 映射

    # max_samples 在 tar 遍历中间就停，不扫后续 split
    n2 = dl._download_cv22(src, tmp_path / "out2", 1, "tok", None)
    assert n2 == 1


def test_parquet_404_falls_back_to_api_url(tmp_path, monkeypatch):
    """仓库没有 refs/convert/parquet 分支时（yodas2_sidon 实测），hf_hub_download 404
    要回退到索引 API 的原始 URL 直链下载；非 404 错误不许回退。"""
    import io

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from huggingface_hub.errors import EntryNotFoundError
    from voxft.data import download as dl
    from voxft.data.registry import get_source

    src = get_source("yodas2_ms")
    flac = io.BytesIO()
    sf.write(flac, np.full(160, 0.2, dtype=np.float32), 16000, format="FLAC")
    buf = io.BytesIO()
    pd.DataFrame([{"audio": {"bytes": flac.getvalue(), "path": "a.flac"},
                   "text": "selamat pagi"}]).to_parquet(buf)
    payload = buf.getvalue()

    class Resp:
        status_code = 200
        def __init__(self, data): self._data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_content(self, n): yield self._data

    urls = []
    monkeypatch.setattr("requests.get", lambda url, **kw: urls.append(url) or Resp(payload))
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kw: (_ for _ in ()).throw(EntryNotFoundError("no entry")))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    api_url = ("https://huggingface.co/api/datasets/sarulab-speech/yodas2_sidon"
               "/parquet/ms000/train/0.parquet")
    log = []
    n = dl._download_parquet(src, [(api_url, "")], tmp_path / "out", None, "tok", log.append)
    assert n == 1
    rows = [json.loads(line)
            for line in (tmp_path / "out/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "selamat pagi" and rows[0]["lang"] == "ms"
    assert urls == [api_url.replace("huggingface.co", "hf-mirror.com")]  # 主机改写到镜像
    assert any("回退索引 API URL 直链下载" in line for line in log)

    # 非 404 错误（网络/权限）照旧重试后报错，不许静默掉进直链回退
    urls.clear()
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **kw: (_ for _ in ()).throw(OSError("conn reset")))
    with pytest.raises(RuntimeError, match="下载失败"):
        dl._download_parquet(src, [(api_url, "")], tmp_path / "out3", None, "", None)
    assert urls == []


def test_wds_member_columns_sliced_by_utterances(tmp_path, monkeypatch):
    """yodas2_sidon 转换 parquet 是 WebDataset 原样成员列：flac={bytes=整段视频音频},
    metadata.json={video_id, utterances:{start,end,text,utt_id}}——按 utterances 切句，
    session 取 video_id（同 yodas_th 的 utt_id.rsplit("-",3)[0] 约定），speaker 不强凑。"""
    import io

    import numpy as np
    import pandas as pd
    import soundfile as sf
    from voxft.data import download as dl
    from voxft.data.registry import get_source

    src = get_source("yodas2_ms")

    def _flac(seconds, value):
        buf = io.BytesIO()
        sf.write(buf, np.full(int(8000 * seconds), value, dtype=np.float32), 8000,
                 format="FLAC")
        return buf.getvalue()

    frame = pd.DataFrame([
        {"flac": {"bytes": _flac(6.0, 0.3), "path": "v1.flac"},
         "metadata.json": {"duration": 6.0, "id": "0", "video_id": "abc123-x",
                           "utterances": {"start": [0.5, 2.0], "end": [1.5, 4.0],
                                          "text": ["selamat pagi", ""],
                                          "utt_id": ["abc123-x-00000-00000050-00000150",
                                                     "abc123-x-00001-00000200-00000400"]}},
         "__key__": "0", "__url__": "hf://datasets/x@rev/ms000/train-00000.tar.gz"},
        {"flac": {"bytes": _flac(2.0, 0.4), "path": "v2.flac"},
         "metadata.json": {"duration": 2.0, "id": "1", "video_id": "def456",
                           "utterances": {"start": [], "end": [], "text": [], "utt_id": []}},
         "__key__": "1", "__url__": "hf://datasets/x@rev/ms000/train-00000.tar.gz"},
        {"flac": {"bytes": b"broken", "path": "v3.flac"},
         "metadata.json": {"duration": 1.0, "id": "2", "video_id": "ghi789",
                           "utterances": {"start": [0.0], "end": [1.0],
                                          "text": ["tidak mengapa"], "utt_id": ["ghi789-00000-0-1"]}},
         "__key__": "2", "__url__": "hf://datasets/x@rev/ms000/train-00000.tar.gz"},
    ])
    parquet = tmp_path / "0.parquet"
    frame.to_parquet(parquet)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(parquet))

    log = []
    n = dl._download_parquet(src, [("ms000/train/0.parquet", "refs/convert/parquet")],
                             tmp_path / "out", None, "", log.append)
    assert n == 1   # 空文本句、无 utterances 的视频、坏音频各跳一条
    rows = [json.loads(line)
            for line in (tmp_path / "out/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rec = rows[0]
    assert rec["text"] == "selamat pagi" and rec["lang"] == "ms"
    assert rec["session"] == "abc123-x" and rec["utt_id"] == "abc123-x-00000-00000050-00000150"
    assert rec["speaker_verified"] is False and "speaker" not in rec
    assert sf.info(rec["audio"]).frames == 8000   # 0.5s-1.5s 切出 1.0s，不是整段 6s
    assert any("缺文本 1 条" in line and "音频/切片失败 1 条" in line for line in log)
