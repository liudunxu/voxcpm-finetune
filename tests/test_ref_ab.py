import importlib.util
import json
from pathlib import Path

import pytest


def test_ref_ab_pair_validation_and_only_reference_fields_change(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ref_ab_test", scripts / "ref_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = {"audio": "/target.wav", "origin_audio": "/original_target.wav",
              "duration": 4.0, "text": "target words", "lang": "th",
              "source_id": "cv22_th", "speaker": "verified",
              "speaker_verified": True, "ref_audio": "/reference.wav",
              "ref_origin_audio": "/original_reference.wav", "ref_duration": 4.0,
              "ref_speaker": "verified"}
    reference = {"audio": "/reference.wav", "origin_audio": "/original_reference.wav",
                 "duration": 4.0, "text": "different words", "lang": "th",
                 "source_id": "cv22_th", "speaker": "verified", "speaker_verified": True}
    assert module.checked_pairs([target, reference]) == [target]
    assert module.checked_pairs([target, reference], {"targetwords"}) == []
    assert module.checked_pairs([target, reference], {"differentwords"}) == []
    stripped = module.without_ref(target)
    assert stripped["text"] == target["text"] and stripped["audio"] == target["audio"]
    assert not any(key.startswith("ref_") for key in stripped)
    assert "ref_audio" in target
    for invalid in ({**target, "speaker_verified": False},
                    {**target, "ref_speaker": "different"},
                    {**target, "origin_audio": reference["origin_audio"]},
                    {**target, "ref_duration": float("nan")},
                    {**target, "ref_duration": 9}):
        with pytest.raises(ValueError, match="Invalid"):
            module.checked_pairs([invalid, reference])
    with pytest.raises(ValueError, match="Invalid"):
        module.checked_pairs([target])
    assert module.checked_pairs([target, {**reference, "text": target["text"]}]) == []
    assert module.memberships([target]) & module.memberships([reference])
    assert not module.memberships([target]) & module.memberships([
        {"audio": "/val.wav", "origin_audio": "/val_original.wav",
         "speaker_verified": True, "speaker": "held_out"}])
    cases = [{"case_id": "case", "text": "test", "lang": "th"}]
    report = {"cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
              "asr_model": "large-v3",
              "items": [{**cases[0], "seed": seed} for seed in module.SEEDS]}
    module.verify_comparison({"off": report, "on": report}, cases)
    with pytest.raises(ValueError, match="complete"):
        module.verify_comparison({"missing": {**report, "items": report["items"][:-1]}}, cases)
    with pytest.raises(ValueError, match="conditions"):
        module.verify_comparison({"cfg": {**report, "cfg_value": 2.0}}, cases)


def test_ref_ab_collection_preserves_historical_reports(monkeypatch, tmp_path):
    from copy import deepcopy
    from voxft import eval as evaluation
    from voxft.train import runlog

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ref_ab_collection", scripts / "ref_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = {"case_id": "case", "text": "test", "lang": "th", "cohort": "r8_unseen_text"}
    report = {"label": "historical", "cfg_value": 1.8, "inference_timesteps": 20,
              "retry_badcase": False, "asr_model": "large-v3", "note": "original",
              "items": [{**case, "seed": seed, "numeric": False, "wer": None,
                         "cer": 0.1, "audio_sec": 4.0, "f0_std_st": 1}
                        for seed in module.SEEDS]}
    source = tmp_path / "historical.json"
    module.write_json(source, report)
    original = source.read_bytes()
    (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n")
    module.write_json(tmp_path / "plan.json", {
        "historical_reports": {"base": {"path": str(source), "sha256": module.sha256(source)},
                               "r8": {"path": str(source), "sha256": module.sha256(source)}},
        "r8_seen_text_case_ids": [],
        "configs": {arm: {"summary": {"hours": 1}} for arm in module.ARMS},
    })
    monkeypatch.setattr(module, "runtime_environment", lambda work: None)
    monkeypatch.setattr(evaluation, "merge_reports", lambda paths:
                        {**deepcopy(report), "label": Path(paths[0]).stem})
    monkeypatch.setattr(runlog, "build_record", lambda *args, **kwargs: "# Test record")
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: None)
    module.collect(tmp_path)
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert summary["reports"]["off"]["overall"]["mean_audio_sec_non_numeric"] == 4
    assert summary["reports"]["on"]["cohorts"]["r8_unseen_text"]["mean_cer"] == 0.1
    assert source.read_bytes() == original
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stage"] == "complete" and status["production_changed"] is False
    assert status["native_review"] == "unverified"
