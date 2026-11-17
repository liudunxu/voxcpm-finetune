"""流程审计修复的回归测试：假停止/桥接断流/坏权重/半截下载/评测续跑等。"""
import io
import json
import os
import signal
import subprocess
import threading

import numpy as np
import pytest
import soundfile as sf


def test_stop_local_kills_whole_process_group():
    """shell=True 启动时 terminate 只杀 shell；必须按进程组带走训练子进程。"""
    from voxft.train import launcher
    proc = subprocess.Popen("sleep 30 & wait", shell=True, start_new_session=True)
    launcher._PROC = proc
    try:
        assert launcher.stop_local() is True
        assert proc.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)  # 组内子进程（sleep）也必须已退出
    finally:
        launcher._PROC = None
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    assert launcher.stop_local() is False


def test_bridge_serve_survives_transient_backend_failure(monkeypatch, capsys):
    from voxft.train import tb_wandb_bridge as bridge
    stop = threading.Event()
    calls = []

    def flaky(tb_dir, run_name, state):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("SwanLab 网络抖动")
        stop.set()
        return 0

    monkeypatch.setattr(bridge, "sync_once", flaky)
    bridge.serve("unused", "run", interval=0.01, stop_event=stop)
    assert len(calls) == 2  # 第一次失败后没有死掉，重试成功
    assert "桥接异常" in capsys.readouterr().err


def test_merge_refuses_partial_match(tmp_path):
    import torch
    from safetensors.torch import save_file
    from voxft.lora.merge import merge_lora
    base = tmp_path / "base"
    base.mkdir()
    save_file({"other.layer.weight": torch.randn(4, 4)}, str(base / "model.safetensors"))
    (base / "config.json").write_text("{}")
    lora = tmp_path / "lora"
    lora.mkdir()
    save_file({"lm.layer.lora_A": torch.randn(2, 4), "lm.layer.lora_B": torch.randn(4, 2)},
              str(lora / "lora_weights.safetensors"))
    (lora / "lora_config.json").write_text(json.dumps({"r": 2, "alpha": 4}))
    with pytest.raises(ValueError, match="合并不完整"):
        merge_lora(base, lora, tmp_path / "merged")


def test_aishell_incomplete_download_not_treated_as_complete(tmp_path, monkeypatch):
    from voxft.data import download as dl
    from voxft.data.registry import get_source

    class ShortResp:
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    resp = ShortResp()
    reads = []

    def read(n):
        reads.append(1)
        return b"x" * 50 if len(reads) == 1 else b""

    resp.read = read
    monkeypatch.setattr(dl.urllib.request, "urlopen", lambda url: resp)
    with pytest.raises(RuntimeError, match="下载不完整"):
        dl._download_aishell3(get_source("aishell3"), tmp_path, None)
    assert not (tmp_path / "data_aishell3.tgz").exists()
    assert (tmp_path / "data_aishell3.tgz.part").exists()


def test_stream_tolerates_bad_rows_like_parquet(tmp_path, monkeypatch):
    from voxft.data import download as dl
    from voxft.data.registry import get_source
    rows = [{"audio": {"array": np.full(160, 0.2), "sampling_rate": 16000}, "text": f"s{i}"}
            for i in range(3)]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: iter(rows))
    real_load = dl._load_audio
    calls = []

    def flaky(value, source, token=""):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("broken row")
        return real_load(value, source, token)

    monkeypatch.setattr(dl, "_load_audio", flaky)
    n = dl._download_stream(get_source("filswitch"), tmp_path / "out", None)
    assert n == 2  # 坏行被跳过，其余照常写入
    manifest = (tmp_path / "out/manifest.jsonl").read_text().splitlines()
    assert len(manifest) == 2

    # 全部坏行则中止，与 parquet 路径同口径
    monkeypatch.setattr(dl, "_load_audio",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("all bad")))
    with pytest.raises(RuntimeError, match="全部读取失败"):
        dl._download_stream(get_source("filswitch"), tmp_path / "bad", None)


def test_model_card_reflects_current_license_policy(tmp_path):
    from voxft.hub.sync import _model_card
    card = _model_card(tmp_path)
    assert "一律禁止对外分发" not in card  # 2026-10-20 已废止的 SA 禁令
    assert "2026-10-20" in card and "share-alike" in card
    assert "GigaSpeech" in card  # 2026-11-17 风险记录


def test_speaker_gain_write_is_atomic(tmp_path, monkeypatch):
    from voxft.data import pipeline
    wav_path = tmp_path / "a.wav"
    sf.write(wav_path, np.full(1600, 0.2), 16000)
    original = wav_path.read_bytes()
    rec = {"audio": str(wav_path), "speaker": "s", "speaker_verified": True,
           "_rms_dbfs": -20.0}

    def crash(path, data, sr):
        with open(path, "wb") as f:
            f.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(pipeline.sf, "write", crash)
    with pytest.raises(OSError, match="disk full"):
        pipeline.apply_speaker_gain([rec], -24.0)
    assert wav_path.read_bytes() == original  # 原音频不被半截写破坏
    assert not (tmp_path / "a.wav.gain.tmp.wav").exists()


def test_whisper_model_cached_within_process(monkeypatch):
    pytest.importorskip("faster_whisper")
    from voxft.data import pipeline
    builds = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None):
            builds.append(name)

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)
    monkeypatch.setattr(pipeline, "_WHISPER_CACHE", {})
    a = pipeline._whisper_model("th", "large-v3")
    b = pipeline._whisper_model("tl", "large-v3")
    assert a is b and len(builds) == 1
    pipeline._whisper_model("th", "medium")
    assert len(builds) == 2


def _stub_eval(monkeypatch, tmp_path, run):
    from voxft import eval as evaluation, infer
    from voxft.data import pipeline
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_whisper_model", lambda lang, size: object())
    monkeypatch.setattr(infer, "get_model", lambda *a: object())
    monkeypatch.setattr(infer, "_run", run)
    monkeypatch.setattr(evaluation, "_transcribe", lambda *a: "สวัสดีค่ะ")
    monkeypatch.setattr(evaluation, "_acoustics", lambda *a: {
        "f0_std_st": 1.0, "audio_sec": 1.0, "chars_per_sec": 5.0,
        "metallic": False, "low_snr": False, "speaker_sim": 0.9})


def test_evaluate_resumes_after_crash(tmp_path, monkeypatch):
    from voxft import eval as evaluation
    cases = [{"text": "สวัสดีค่ะ", "lang": "th"}, {"text": "ยินดีต้อนรับ", "lang": "th"}]
    calls = []

    def crash_second(model, kw, **extra):
        calls.append(kw["text"])
        if len(calls) == 2:
            raise RuntimeError("GPU 掉了")
        return ("fake.wav", 0.1)

    _stub_eval(monkeypatch, tmp_path, crash_second)
    with pytest.raises(RuntimeError, match="GPU 掉了"):
        evaluation.evaluate("base", "th", cases)
    partial = tmp_path / "base.partial.jsonl"
    assert partial.exists()
    assert len(partial.read_text().splitlines()) == 2  # 指纹 + 1 条完成

    calls.clear()
    report = evaluation.evaluate("base", "th", cases)
    assert calls == ["ยินดีต้อนรับ"]  # 只补跑缺失的那条
    assert len(report["items"]) == 2
    assert not partial.exists()  # 完成后清理

    # case 集变化 → 指纹不符，从头跑，不复用旧结果
    calls.clear()
    changed = [{"text": "ประโยคใหม่", "lang": "th"}]
    report = evaluation.evaluate("base", "th", changed)
    assert calls == ["ประโยคใหม่"] and len(report["items"]) == 1