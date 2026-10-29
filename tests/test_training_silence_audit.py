import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_training_silence")


def test_tail_is_not_internal_pause_and_original_is_unchanged(module, tmp_path):
    audio = np.r_[np.full(16000, .1), np.zeros(16000), np.full(16000, .1), np.zeros(16000)]
    path = tmp_path / "tail.wav"
    sf.write(path, audio, 16000)
    before = path.read_bytes()
    item = module.measure(path, 4., module.sha256(path), lambda a, sr: [
        {"start": 0, "end": 16000}, {"start": 32000, "end": 48000}])
    assert item["tail_silence_sec"] == pytest.approx(1., abs=.04)
    assert item["vad"]["tail_seconds"] == 1.
    assert item["vad"]["speech_seconds"] == 2.
    assert item["near_zero_tail_seconds"] == 1.
    assert set(item["review_reasons"]) == {"energy_tail_over_0_5s", "vad_tail_over_0_5s"}
    assert path.read_bytes() == before


def test_low_level_is_unknown_not_zero_and_gain_is_diagnostic_only(module, tmp_path):
    path = tmp_path / "quiet.wav"
    sf.write(path, np.full(48000, .0001), 16000, subtype="FLOAT")
    before = path.read_bytes()
    item = module.measure(path, 3., module.sha256(path), lambda a, sr:
                          [{"start": 0, "end": len(a)}] if np.max(a) > .1 else [])
    assert item["tail_silence_sec"] is None
    assert item["vad"]["tail_seconds"] is None
    assert item["gain_diagnostic"]["speech_seconds"] == 3.
    assert item["review_status"] == "pending_not_confirmed_bad_audio"
    assert module.summarize([item])["energy_tail"]["unknown"] == 1
    assert path.read_bytes() == before


def test_silence_and_invalid_vad_cannot_look_like_clean_audio(module, tmp_path):
    path = tmp_path / "zero.wav"
    sf.write(path, np.zeros(48000), 16000)
    digest = module.sha256(path)
    item = module.measure(path, 3., digest, lambda a, sr: [])
    assert "all_zero_waveform" in item["review_reasons"]
    assert item["gain_diagnostic"] is None
    assert item["nonzero_samples"] == 0
    with pytest.raises(ValueError, match="VAD"):
        module.measure(path, 3., digest, lambda a, sr: [{"start": 0, "end": 48001}])
    with pytest.raises(ValueError, match="changed"):
        module.measure(path, 3., "stale", lambda a, sr: [])
    with pytest.raises(ValueError, match="duration"):
        module.measure(path, 5., digest, lambda a, sr: [])


def test_existing_output_is_never_replaced(module, tmp_path):
    marker = tmp_path / "summary.json"
    marker.write_text("frozen", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path / "missing", tmp_path)
    assert marker.read_text(encoding="utf-8") == "frozen"


def test_complete_audit_covers_external_refs_and_keeps_all_inputs(module, monkeypatch, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    hashes = {}
    for name in ("one", "two", "ref"):
        path = run / f"{name}.wav"
        sf.write(path, np.full(48000, .1), 16000)
        hashes[str(path)] = module.sha256(path)
    rows = [{"audio": str(run / f"{name}.wav"), "duration": 3., "lang": "id",
             "source_id": "cv22_id", "ref_audio": str(run / "ref.wav"),
             "ref_duration": 3., "ref_lang": "id"} for name in ("one", "two")]
    manifest = run / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    hashes[str(manifest)] = module.sha256(manifest)
    (run / "plan.json").write_text(json.dumps({"inputs_sha256": hashes}), encoding="utf-8")
    originals = {path: path.read_bytes() for path in run.iterdir()}

    def timestamps(audio, options, sampling_rate):
        return [{"start": 0, "end": len(audio)}]

    monkeypatch.setitem(sys.modules, "faster_whisper.vad", SimpleNamespace(
        VadOptions=lambda **kwargs: SimpleNamespace(threshold=.5, **kwargs),
        get_speech_timestamps=timestamps,
        get_vad_model=lambda: SimpleNamespace(
            session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))))
    monkeypatch.setitem(sys.modules, "faster_whisper.utils",
                        SimpleNamespace(get_assets_path=lambda: str(tmp_path)))
    output = tmp_path / "audit"
    result = module.prepare(run, output)
    assert result["status"] == "complete" and result["unique_files"] == 3
    assert result["reference_usages"] == 2 and result["unique_references"] == 1
    assert result["flagged_files"] == 0
    assert len((output / "measurements.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert all(path.read_bytes() == content for path, content in originals.items())
    manifest.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        module.prepare(run, tmp_path / "stale")
