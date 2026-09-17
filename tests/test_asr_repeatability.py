import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_repeatability_distinguishes_score_stability_from_quality(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("asr_repeatability", scripts / "asr_repeatability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PROFILES == {"default": {}, "temperature_zero": {"temperature": 0.0}}
    samples = [{"model": model, "case_id": "th_case", "seed": 42, "original_cer": 0.1}
               for model in ("base", "r8")]
    rows = [{**sample, "profile": profile, "repeat": repeat,
             "hyp": f"word {repeat}" if profile == "default" else "wrong but stable",
             "cer": repeat / 10 if profile == "default" else 0.9,
             "segments": [{"temperature": 0.6 if profile == "default" else 0.0}]}
            for profile in module.PROFILES for sample in samples for repeat in range(3)]
    summary = module.summarize(rows, samples)
    assert summary == module.summarize(list(reversed(rows)), samples)
    assert all(row["unique_normalized_hypotheses"] == 3 and row["cer_range"] == 0.2
               for row in summary["default"])
    assert all(row["unique_normalized_hypotheses"] == 1 and row["cer_min"] == 0.9
               for row in summary["temperature_zero"])
    for broken in (rows[:-1], rows + [rows[0]], [dict(row, repeat=7) for row in rows]):
        with pytest.raises(ValueError, match="Incomplete or duplicate"):
            module.summarize(broken, samples)


def test_alignment_changes_only_word_timestamps_and_preserves_originals(monkeypatch, tmp_path):
    import soundfile

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("asr_alignment", scripts / "asr_repeatability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = tmp_path / "asr_repeatability"
    previous.mkdir()
    audio = tmp_path / "test.wav"
    audio.touch()
    rows = [{"model": model, "case_id": "fleurs_test_vi_1695", "seed": 42,
             "profile": "temperature_zero", "repeat": 0, "text": "test", "hyp": "test test",
             "wav": str(audio), "audio_sha256": module.sha256(audio), "cer": 1,
             "segments": [{"end": 30.0}]} for model in ("base", "r8")]
    source = previous / "results.json"
    source.write_text(json.dumps({"complete": True, "items": rows}))
    original = source.read_bytes()
    (previous / "plan.json").write_text(json.dumps({"asr_model_sha256": {}, "asr_model_path": "/model"}))
    monkeypatch.setattr(module, "runtime_environment", lambda work: None)
    monkeypatch.setattr(soundfile, "info", lambda path: SimpleNamespace(duration=12.0))

    @dataclass
    class Segment:
        text: str = "test"
        end: float = 11.9

    @dataclass
    class Options:
        word_timestamps: bool = True

    calls = []

    class Whisper:
        def transcribe(self, wav, **kwargs):
            calls.append(kwargs)
            return [Segment()], SimpleNamespace(transcription_options=Options())

    monkeypatch.setattr(module, "_whisper_model", lambda *args: Whisper())
    module.align_vi(tmp_path)
    result = json.loads((tmp_path / "asr_alignment/results.json").read_text())
    assert result["complete"] and len(result["items"]) == 2
    assert all(row["baseline_timestamp_overrun_sec"] == 18 and row["aligned_timestamp_overrun_sec"] == 0
               for row in result["items"])
    assert calls == [{"language": "vi", "vad_filter": True, "temperature": 0.0,
                      "word_timestamps": True}] * 2
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        module.align_vi(tmp_path)
