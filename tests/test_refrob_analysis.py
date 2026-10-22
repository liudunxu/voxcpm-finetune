from copy import deepcopy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("refrob_stage2_analysis")


def report():
    return {"cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
            "asr_model": "large-v3",
            "items": [{"case_id": "case", "seed": seed, "lang": "id", "text": "test",
                       "ref_audio": "ref.wav", "control": "", "cer": 0.1}
                      for seed in (42, 43, 44)]}


def test_direct_pairing_and_difference_in_differences(module):
    baseline = report()
    candidate = deepcopy(baseline)
    for row in candidate["items"]:
        row["cer"] = 0.04
    direct = module.paired_changes(candidate, baseline, "cer", "id")
    assert direct == pytest.approx({"case": -0.06})
    assert module.delta(direct, {"case": -0.02}) == pytest.approx({"case": -0.04})
    assert baseline == report()
    assert module.boot_ci(direct, n=20) == pytest.approx((-0.06, -0.06, -0.06))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "ref", "cfg", "metric"])
def test_pairing_fails_closed(module, fault):
    baseline = report()
    candidate = deepcopy(baseline)
    if fault == "missing":
        candidate["items"].pop()
    elif fault == "duplicate":
        candidate["items"].append(candidate["items"][0])
    elif fault == "ref":
        candidate["items"][0]["ref_audio"] = "different.wav"
    elif fault == "cfg":
        candidate["cfg_value"] = 2
    else:
        candidate["items"][0]["cer"] = None
    with pytest.raises(ValueError):
        module.paired_changes(candidate, baseline, "cer")


def test_no_silent_intersection_or_overwriting(module, tmp_path):
    with pytest.raises(ValueError, match="Incomplete"):
        module.delta({"a": 1}, {"b": 2})
    output = tmp_path / "old_report.json"
    output.write_bytes(b"frozen")
    with pytest.raises(FileExistsError):
        module.main(tmp_path, output)
    assert output.read_bytes() == b"frozen"
