import importlib.util
import json
from pathlib import Path

import pytest


def test_diagnostic_export_preserves_sources_and_independent_controls(monkeypatch, tmp_path):
    from voxft import eval as evaluation

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("holdout_diagnostics", scripts / "holdout_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "reports")
    cases = [{"case_id": f"{lang}_{reference}", "text": "Test sentence.", "lang": lang,
              "ref_lang": reference, "ref_audio": f"/{reference}.wav", "numeric": False}
             for lang in ("th", "tl", "vi", "id", "ms") for reference in ("zh", "en", "tl")]
    seeds = [42, 43, 44, 45, 49]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    module.write_json(tmp_path / "plan.json", {
        "evaluation_seeds": seeds, "inputs_sha256": {str(cases_path): module.sha256(cases_path)},
    })
    originals = {}
    for model in ("base", "r8"):
        audio = tmp_path / f"{model}.wav"
        audio.touch()
        rows = [{**case, "seed": seed, "wav": str(audio), "cer": 0.1, "wer": 0.1, "hyp": "test",
                 "speaker_sim": 0.9, "audio_sec": 4.0, "f0_std_st": 1.0,
                 "human_review": {"acoustic_status": None}}
                for case in cases for seed in seeds]
        if model == "r8":
            for row in rows:
                if row["seed"] == 49:
                    row.update(cer=0.6, speaker_sim=0.5)
        source = tmp_path / f"{model}_report.json"
        module.write_json(source, {"items": rows, "cohorts": {"stale": 75}, "label": model})
        originals[source] = source.read_bytes()
    hashes = {model: module.sha256(tmp_path / f"{model}_report.json") for model in ("base", "r8")}
    paired = [{"case_id": case["case_id"], "cer": [0.1, 0.2], "speaker_sim": [0.9, 0.82]}
              for case in cases]
    module.write_json(tmp_path / "comparison.json", {
        "inputs_sha256": hashes, "r8_minus_base": {"by_case": paired},
    })
    _, indices = evaluation._review_reports(str(tmp_path / "base_report.json"),
                                            str(tmp_path / "r8_report.json"))
    selected = module.select_samples(cases, paired, indices)
    assert selected == module.select_samples(list(reversed(cases)), list(reversed(paired)), indices)
    altered = [{**row, "cer": [0.9, 0.1], "speaker_sim": [0.4, 0.9]} for row in paired]
    assert selected["random_control"] == module.select_samples(cases, altered, indices)["random_control"]
    assert len(selected["diagnostic"]) == 10
    assert {row["seed"] for row in selected["diagnostic"]} == {49}
    assert {row["seed"] for row in selected["random_control"]} == {42}
    missing = tmp_path / "r8.wav"
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        module.collect(tmp_path)
    assert not (tmp_path / "diagnostic_sample").exists()
    missing.touch()
    source = tmp_path / "base_report.json"
    source.write_text("changed")
    with pytest.raises(ValueError, match="report changed"):
        module.collect(tmp_path)
    source.write_bytes(originals[source])
    module.collect(tmp_path)
    saved = json.loads((tmp_path / "diagnostic_sample/reports.json").read_text())
    metadata = json.loads((tmp_path / "diagnostic_sample/plan.json").read_text())
    assert len(metadata["overlap_case_ids"]) == 10
    for group, paths in saved.items():
        session = evaluation.review_session(paths["base"], paths["r8"])
        assert len(session) == len(selected[group])
        for path in paths.values():
            report = json.loads(Path(path).read_text())
            assert "cohorts" not in report
            assert sum(row["cases"] for row in report["by_lang"].values()) == len(selected[group])
            assert all(row["human_review"]["acoustic_status"] is None for row in report["items"])
    assert all(path.read_bytes() == data for path, data in originals.items())
    with pytest.raises(FileExistsError):
        module.collect(tmp_path)
    from types import SimpleNamespace
    from voxft.data import pipeline

    calls = []

    class Whisper:
        def transcribe(self, wav, **kwargs):
            calls.append((wav, kwargs))
            return [SimpleNamespace(start=0, end=4, text="Test sentence.")], SimpleNamespace()

    monkeypatch.setattr(module, "runtime_environment", lambda work: None)
    monkeypatch.setattr(pipeline, "_whisper_model", lambda *args: Whisper())
    module.recheck_asr(tmp_path)
    result = json.loads((tmp_path / "asr_recheck/results.json").read_text())
    assert result["complete"] is True and len(result["items"]) == 12
    assert all(row["cer"] == 0 for row in result["items"])
    for original, variant in zip(calls[::2], calls[1::2]):
        assert original[0] == variant[0]
        assert original[1]["vad_filter"] is True and variant[1]["vad_filter"] is False
        assert {**original[1], "vad_filter": False} == variant[1]
    assert all(path.read_bytes() == data for path, data in originals.items())
    with pytest.raises(FileExistsError):
        module.recheck_asr(tmp_path)
