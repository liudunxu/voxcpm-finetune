from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("control_ab_run")


def fixture_report(module):
    cases = [{"case_id": condition, "sentence_id": "one", "condition": condition,
              "lang": "id", "text": "hello", "ref_audio": "/ref.wav",
              "control": "" if condition == "plain" else "angry"}
             for condition in ("plain", "control")]
    report = {
        "target": "base", "base": "/base", "cfg_value": 1.8, "inference_timesteps": 20,
        "retry_badcase": False, "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
        "lora_strength": 0,
        "items": [{**case, "seed": seed, "cer": 0.2, "speaker_sim": 0.9}
                  for case in cases for seed in module.SEEDS],
    }
    return cases, report, {"base_path": "/base", "targets": {"base": "base"}}


@pytest.mark.parametrize("fault", ["none", "missing", "duplicate", "text", "seed", "backend"])
def test_reports_require_complete_frozen_conditions(module, fault):
    cases, report, plan = fixture_report(module)
    if fault == "none":
        module.verify_report(report, plan, cases, "base")
        return
    if fault == "missing":
        report["items"].pop()
    elif fault == "duplicate":
        report["items"].append(report["items"][0])
    elif fault == "backend":
        report["asr_model"] = "different"
    else:
        report["items"][0][fault] = "changed"
    with pytest.raises(ValueError):
        module.verify_report(report, plan, cases, "base")


def test_control_variants_and_seeds_are_one_text_cluster(module):
    cases, baseline, _ = fixture_report(module)
    candidate = deepcopy(baseline)
    for row in candidate["items"]:
        if row["condition"] == "control":
            row["cer"] = 0.1
            row["speaker_sim"] = 0.8
    changes = module.cluster_changes(candidate, baseline, cases, "cer", "id")
    assert changes == pytest.approx({"one": -0.05})
    assert module.control_effect(candidate, "speaker_sim", "id") == pytest.approx({"one": -0.1})
    for report in (candidate, baseline):
        report["items"].append({**report["items"][0], "case_id": "old",
                                "cohort": "historical_regression", "cer": 20})
    assert module.cluster_changes(candidate, baseline, cases, "cer", "id") == changes


def test_explicit_parent_comparison_uses_matching_frozen_reports(module, monkeypatch, tmp_path):
    cases, baseline, plan = fixture_report(module)
    plan.update(inputs_sha256={}, additional_comparisons=[["r29", "r28"]])
    plan["targets"].update(r28="/old/latest", r29="/new/latest")
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases))
    for name, cer in (("base", .2), ("r28", .1), ("r29", .05)):
        report = deepcopy(baseline)
        report.update(target=plan["targets"][name], lora_strength=0 if name == "base" else 1)
        for row in report["items"]:
            row.update(cer=cer, suspected_truncation=False, over_read=False)
        (tmp_path / f"{name}_report.json").write_text(json.dumps(report))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(module, "LANGS", ("id",))
    monkeypatch.setattr(module.evaluation, "_report_metrics", lambda items: {})
    result = module.collect(tmp_path, ("base", "r28", "r29"), tmp_path / "comparison.json")
    assert result["comparisons"]["r29_minus_r28"]["id"]["cer"]["mean"] == pytest.approx(-.05)
    assert result["comparisons"]["r29_minus_r28"]["id"]["cer"]["independent_texts"] == 1
    assert "r29_minus_base" in result["comparisons"]
    plan["additional_comparisons"] = [["r29", "missing"]]
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Invalid requested comparison"):
        module.collect(tmp_path, ("base", "r28", "r29"), tmp_path / "invalid.json")


def test_fresh_training_never_reuses_a_checkpoint(module, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    module.require_fresh_checkpoint({"save_path": str(checkpoint)})
    checkpoint.mkdir()
    with pytest.raises(FileExistsError, match="implicit resume"):
        module.require_fresh_checkpoint({"save_path": str(checkpoint)})


@pytest.mark.parametrize("unexpected_file", [False, True])
def test_preflight_created_empty_directories_are_not_resumed(
        module, tmp_path, monkeypatch, unexpected_file):
    configs = {}
    for arm in ("off", "on"):
        path = tmp_path / f"{arm}.yaml"
        checkpoint = tmp_path / f"checkpoint_{arm}"
        path.write_text(yaml.safe_dump({"save_path": str(checkpoint),
                                       "tensorboard": str(checkpoint / "logs")}))
        configs[arm] = str(path)
    (tmp_path / "baseline_comparison.json").write_text('{"inputs_sha256": {}}')
    monkeypatch.setattr(module, "check_inputs", lambda work: {"configs": configs, "num_iters": 1})

    def preflight(path, gpus):
        config = yaml.safe_load(Path(path).read_text())
        Path(config["save_path"]).mkdir()
        if unexpected_file:
            (Path(config["save_path"]) / "unexpected").write_text("preserve")
        return []

    class Bridge:
        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    trained = []

    def training(command, **kwargs):
        config = yaml.safe_load(Path(command[-1]).read_text())
        assert Path(config["tensorboard"]).is_dir()
        latest = Path(config["save_path"]) / "latest"
        latest.mkdir()
        (latest / "training_state.json").write_text(json.dumps({"step": 1}))
        trained.append(command[-1])

    monkeypatch.setattr(module, "preflight", preflight)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Bridge())
    monkeypatch.setattr(module.subprocess, "run", training)
    if unexpected_file:
        with pytest.raises(OSError):
            module.train(tmp_path)
        assert (tmp_path / "checkpoint_off/unexpected").read_text() == "preserve"
        assert trained == []
        return
    module.train(tmp_path)
    assert trained == list(configs.values())
    assert all((tmp_path / f"trained_{arm}.json").is_file() for arm in configs)
