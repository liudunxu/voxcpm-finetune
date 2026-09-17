import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


def test_cpu_diagnostics_exposure_and_pairing_guards(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ref_diagnostics", scripts / "ref_ab_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = {"audio": "/ref.wav", "origin_audio": "/ref_original.wav", "text": "Reference!",
                 "lang": "th", "speaker": "known", "speaker_verified": True}
    target = {"audio": "/train.wav", "text": "target", "ref_audio": "/ref.wav",
              "ref_origin_audio": "/ref_original.wav", "ref_speaker": "known"}
    used = module.expand_references([target], {"/ref.wav": reference})
    with pytest.raises(ValueError, match="unavailable"):
        module.expand_references([target], {})
    candidates = [{"audio": "/new.wav", "text": "REFERENCE", "lang": "th"},
                  {"audio": "/other.wav", "text": "new", "speaker": "known", "speaker_verified": True},
                  {"audio": "/val.wav", "text": "validation"},
                  {"audio": "/unused.wav", "text": "untouched"}]
    rows = module.audit_candidates(candidates, {"r10/train": used, "r8/val": [candidates[2]]}, set())
    assert rows[0]["overlaps"] == {"r10/train": ["text"]}
    assert rows[1]["identity_status"] == "matched" and not rows[1]["training_clean"]
    assert rows[2]["training_clean"] and not rows[2]["training_validation_clean"]
    assert rows[3]["training_validation_clean"] and rows[3]["identity_status"] == "unknown"
    assert not module.audit_candidates([candidates[3]], {}, {"untouched"})[0]["training_validation_clean"]
    report = {"cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
              "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
              "items": [{"case_id": case_id, "seed": seed, "text": str(case_id), "lang": "th",
                         "numeric": False, "cer": 0.0, "audio_sec": 2.0}
                        for case_id in range(3) for seed in range(42, 47)]}
    changed = deepcopy(report)
    for row in changed["items"]:
        row["cer"] = row["case_id"] * 0.1
    result = module.paired_summary(report, changed)
    metric = result["by_cohort"]["all"]["th"]["non_numeric"]["cer"]
    assert metric["cases"] == 3 and metric["delta_b_minus_a"] == pytest.approx(0.1)
    assert metric["case_bootstrap_95"][0] <= 0.1 <= metric["case_bootstrap_95"][1]
    assert result == module.paired_summary(report, {**changed, "items": list(reversed(changed["items"]))})
    changed["items"][0]["cer"] = None
    changed["items"][5]["audio_sec"] = float("nan")
    metrics = module.paired_summary(report, changed)["by_cohort"]["all"]["th"]["all"]
    assert metrics["cer"]["cases"] == metrics["audio_sec"]["cases"] == 1
    assert metrics["cer"]["case_bootstrap_95"] is None
    assert metrics["speaker_sim"]["cases"] == 0 and metrics["speaker_sim"]["mean_b"] is None
    for invalid in ({**report, "items": report["items"][:-1]},
                    {**report, "items": report["items"] + report["items"][:1]},
                    {**report, "cfg_value": 2.0}):
        with pytest.raises(ValueError):
            module.paired_summary(report, invalid)
    for field, value in (("text", "different"), ("numeric", True), ("ref_audio", "/different.wav")):
        invalid = deepcopy(report)
        invalid["items"][0][field] = value
        with pytest.raises(ValueError, match="conditions"):
            module.paired_summary(report, invalid)


def test_training_pool_audio_and_holdout_guards(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("training_pool", scripts / "training_pool.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "test.wav"
    waveform = np.full(48000, 0.1, dtype=np.float32)
    row = {"audio": str(path), "duration": 3.0, "text": "Target!",
           "origin_audio": "/original.wav", "speaker": "known", "speaker_verified": True}
    sf.write(path, waveform, 16000, subtype="FLOAT")
    clean = module.inspect_audio(row)
    assert clean["max_full_scale_run_ms"] == 0 and clean["near_zero_tail_sec"] == 0
    with pytest.raises(ValueError, match="duration"):
        module.inspect_audio({**row, "duration": float("nan")})
    waveform[:32] = 1.0
    waveform[-16000:] = 0
    sf.write(path, waveform, 16000, subtype="FLOAT")
    flagged = module.inspect_audio(row)
    assert flagged["max_full_scale_run_ms"] == 2 and flagged["near_zero_tail_sec"] == 1
    assert flagged["pcm_sha256"] != clean["pcm_sha256"]
    for amplitude in (0, 1.1, float("nan")):
        sf.write(path, np.full(48000, amplitude, dtype=np.float32), 16000, subtype="FLOAT")
        with pytest.raises(ValueError):
            module.inspect_audio(row)
    assert module.metadata_rejection(row, set(), {"target"}) == "evaluation_or_validation_text"
    assert module.metadata_rejection(row, {("speaker", "known")}, set()) == "validation_audio_speaker_or_session"
    assert module.metadata_rejection(row, set(), set()) is None
    assert module.metadata_rejection({**row, "origin_audio": ""}, set(), set()) == "missing_origin"
