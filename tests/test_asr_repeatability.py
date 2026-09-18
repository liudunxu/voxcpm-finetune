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


def test_random_controls_keep_frozen_selection_and_reject_missing_strata(monkeypatch, tmp_path):
    import soundfile

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("asr_controls", scripts / "asr_repeatability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "done").touch()
    selection = tmp_path / "diagnostic_sample/plan.json"
    selection.parent.mkdir()
    audio = tmp_path / "test.wav"
    audio.touch()
    rows = [{"case_id": f"{lang}_{ref}", "lang": lang, "ref_lang": ref, "seed": 42,
             "text": "unchanged", "wav": str(audio), "cer": 0.9}
            for lang in ("th", "tl", "vi", "id", "ms") for ref in ("zh", "en", "tl")]
    controls = [{key: row[key] for key in ("case_id", "seed")} for row in reversed(rows)]
    hashes = {}
    for model in ("base", "r8"):
        source = tmp_path / f"{model}_report.json"
        source.write_text(model)
        hashes[model] = module.sha256(source)
    plan = {"samples": {"random_control": controls, "diagnostic": controls[:1]},
            "source_reports_sha256": hashes}
    selection.write_text(json.dumps(plan))
    originals = selection.read_bytes()
    monkeypatch.setattr(soundfile, "info", lambda path: SimpleNamespace(duration=12.0))
    index = {(row["case_id"], row["seed"]): row for row in rows}
    monkeypatch.setattr(module.evaluation, "_review_reports", lambda *args: (
        {who: {"asr_model": "large-v3"} for who in ("a", "b")},
        {who: index for who in ("a", "b")}))
    result = module.prepare(tmp_path, random_controls=True)
    assert result["sample_group"] == "random_control" and len(result["samples"]) == 30
    assert [row["case_id"] for row in result["samples"]] == [row["case_id"] for row in controls] * 2
    assert selection.read_bytes() == originals
    assert all(row["source_audio_sec"] == 12 for row in result["samples"])
    plan["samples"]["random_control"][-1] = controls[0]
    selection.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="15 target/ref-language"):
        module.prepare(tmp_path, random_controls=True)
