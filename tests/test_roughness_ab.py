import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_pinned_model_ab_budget_inputs_and_no_overwrite(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("roughness_ab", scripts / "roughness_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle, base, lora = (tmp_path / name for name in ("bundle", "base", "lora"))
    for directory in (bundle, base, lora):
        directory.mkdir()
        (directory / "fixture").write_bytes(directory.name.encode())
    plan = {
        "scope": "model diagnostic only", "generation_count": 12, "retry_badcase": False,
        "base_sha256": {"fixture": module.sha256(base / "fixture")},
        "lora_sha256": {"fixture": module.sha256(lora / "fixture")},
        "cases": [{"case_id": name, "text": "test", "seed": 42, "seeds": [42, 43, 49],
                   "cfg_value": 1.8, "inference_timesteps": 28, "control": "outburst",
                   "reference": "fixture", "reference_sha256": module.sha256(bundle / "fixture")}
                  for name in ("first", "last")],
    }
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert module.check_inputs(bundle, base, lora) == plan
    (lora / "fixture").write_bytes(b"wrong adapter")
    with pytest.raises(ValueError, match="Frozen input changed"):
        module.check_inputs(bundle, base, lora)
    (lora / "fixture").write_bytes(b"lora")
    plan["cases"][1]["seeds"] = [42, 43, 44]
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="adjacent seeds"):
        module.check_inputs(bundle, base, lora)
    plan["cases"][1]["seeds"] = [42, 43, 49]
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(module.infer.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "runtime_environment", lambda _: None)
    calls = []

    def synthesize(text, base_path, lora_path, **kwargs):
        import soundfile as sf

        calls.append((text, base_path, lora_path, kwargs))
        directory = module.infer.CHECKPOINT_DIR / "auditions"
        directory.mkdir(exist_ok=True)
        takes = []
        for arm in ("base", "r8"):
            path = directory / f"fixture_{arm}.wav"
            sf.write(path, [0.0, 0.1, -0.1], 16000)
            takes.append((str(path), 0.0))
        return *takes, "mock, not GPU verification"

    monkeypatch.setattr(module.infer, "synthesize_ab", synthesize)
    original_output = module.infer.CHECKPOINT_DIR
    output = tmp_path / "output"
    module.run(bundle, base, lora, output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete" and len(report["items"]) == 12
    assert len(calls) == 6 and len(list((output / "auditions").glob("*.wav"))) == 12
    assert [call[3]["seed"] for call in calls] == [42, 43, 49] * 2
    assert all(call[3]["control"] == "outburst" for call in calls)
    assert module.infer.CHECKPOINT_DIR == original_output
    with pytest.raises(FileExistsError):
        module.run(bundle, base, lora, output)
    assert len(calls) == 6
