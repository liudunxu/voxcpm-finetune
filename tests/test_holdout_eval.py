import importlib.util
import json
from pathlib import Path

import pytest


def test_holdout_collection_requires_frozen_complete_conditions(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("holdout_eval", scripts / "holdout_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = [{"case_id": "new_id", "lang": "id", "text": "Test sentence.",
              "ref_audio": "/ref.wav", "cohort": "new_text", "numeric": False}]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row) + "\n" for row in cases))
    seeds = [42, 43, 44, 45, 49]
    plan = {"base_path": "/base", "targets": {"base": "base", "r8": "/r8"},
            "evaluation_seeds": seeds, "inputs_sha256": {str(cases_path): module.sha256(cases_path)}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    for model in ("base", "r8"):
        report = {"target": plan["targets"][model], "base": "/base",
                  "lora_strength": 0 if model == "base" else 1, "label": model,
                  "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
                  "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
                  "items": [{**cases[0], "seed": seed, "cer": 0.1, "wer": 0.1, "audio_sec": 4,
                             "f0_std_st": 1.0} for seed in seeds], "by_lang": {}}
        (tmp_path / f"{model}_report.json").write_text(json.dumps(report))
    module.collect(tmp_path)
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert summary["r8_minus_base"]["seeds"] == seeds
    assert summary["production_changed"] is False and summary["native_review"] == "unverified"
    path = tmp_path / "r8_report.json"
    original = path.read_text()
    for change in ("missing", "duplicate", "seed", "text", "ref_audio", "cfg_value", "target"):
        report = json.loads(original)
        if change == "missing":
            report["items"].pop()
        elif change == "duplicate":
            report["items"].append(report["items"][0])
        elif change in ("seed", "text", "ref_audio"):
            report["items"][0][change] = "changed"
        else:
            report[change] = "changed"
        path.write_text(json.dumps(report))
        with pytest.raises(ValueError):
            module.collect(tmp_path)
    cases_path.write_text("changed")
    with pytest.raises(ValueError, match="Frozen input"):
        module.check_inputs(tmp_path)


def test_frozen_plan_uses_utf8_after_runtime_locale_changes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from holdout_eval import check_inputs, sha256

    source = tmp_path / "case.json"
    source.write_text('{"text": "ราคาตั๋ว"}', encoding="utf-8")
    plan = {"text": "ราคาตั๋ว", "inputs_sha256": {str(source): sha256(source)}}
    (tmp_path / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    read_text = Path.read_text

    def ascii_default(path, encoding=None, **kwargs):
        return read_text(path, encoding=encoding or "ascii", **kwargs)

    monkeypatch.setattr(Path, "read_text", ascii_default)
    assert check_inputs(tmp_path) == plan
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input"):
        check_inputs(tmp_path)
