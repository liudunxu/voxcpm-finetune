import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fresh_baseline_eval")


def test_frozen_budget_id_first_without_mutating_cases(module):
    cases = module.frozen._read_manifest(module.CASES)
    original = json.dumps(cases, ensure_ascii=False)
    tasks = [(model, *task) for model in module.MODELS for task in module.tasks(cases, model)]
    assert len(tasks) == 1800
    assert len({identifier for _, identifier, _, _ in tasks}) == 1800
    assert {(model, case["case_id"], seed) for model, _, case, seed in tasks} == {
        (model, case["case_id"], seed)
        for model in module.MODELS for case in cases for seed in range(42, 47)}
    assert all(case["lang"] == "id" for _, _, case, _ in tasks[:300])
    assert all(case["lang"] != "id" for _, _, case, _ in tasks[300:900])
    assert json.dumps(cases, ensure_ascii=False) == original
    with pytest.raises(ValueError, match="Unapproved"):
        list(module.tasks(cases, "r27"))


def test_existing_work_and_reports_cannot_be_overwritten(module, tmp_path):
    with pytest.raises(FileExistsError, match="implicitly resume"):
        module.prepare(tmp_path)
    module.save_new(tmp_path / "started.json", {"existing": True})
    with pytest.raises(FileExistsError, match="implicitly resume"):
        module.run(tmp_path)
    with pytest.raises(FileExistsError):
        module.save_new(tmp_path / "started.json", {})
    assert json.loads((tmp_path / "started.json").read_text()) == {"existing": True}


@pytest.mark.parametrize("fault", ["none", "seed", "budget", "target", "training", "input", "asr"])
def test_plan_guards_authorized_scope_and_input_hashes(module, tmp_path, fault):
    path = tmp_path / "cases.jsonl"
    path.write_bytes(module.CASES.read_bytes())
    plan = {
        **module.SETTINGS, "base_path": str(module.BASE),
        "targets": {"base": "base", "r26": str(module.R26)},
        "asr_path": str(module.WHISPER), "speaker_model_path": str(module.SPEAKER),
        "cases": 180, "generation_budget": 1800, "max_workers": 2,
        "training_authorized": False, "inputs_sha256": {str(path): module.CASES_SHA},
    }
    if fault == "seed":
        plan["evaluation_seeds"] = [42]
    elif fault == "budget":
        plan["generation_budget"] = 1808
    elif fault == "target":
        plan["targets"] = {"base": "base", "r26": "/r27"}
    elif fault == "training":
        plan["training_authorized"] = True
    elif fault == "asr":
        plan["asr_path"] = "/another/model"
    elif fault == "input":
        path.write_text("{}\n")
    module.save_new(tmp_path / "plan.json", plan)
    if fault == "none":
        assert module.checked(tmp_path) == plan
    else:
        with pytest.raises(ValueError):
            module.checked(tmp_path)


@pytest.fixture
def experiment(module, tmp_path, monkeypatch):
    cases = [{"case_id": f"{lang}_{condition}", "sentence_id": lang, "condition": condition,
              "lang": lang, "text": f"hello {lang}", "ref_audio": "/ref",
              "control": "" if condition == "plain" else "angry"}
             for lang in ("th", "tl", "id") for condition in ("plain", "control")]
    plan = {
        **module.SETTINGS, "base_path": "/base", "targets": {"base": "base", "r26": "/r26"},
        "asr_path": "/whisper", "speaker_model_path": "/speaker", "generation_budget": 60,
    }
    monkeypatch.setattr(module, "checked", lambda work: plan)
    monkeypatch.setattr(module.frozen, "check_inputs", lambda work: plan)
    monkeypatch.setattr(module.frozen, "_read_manifest", lambda path: cases)
    monkeypatch.setattr(module.frozen, "runtime_environment", lambda path: None)
    monkeypatch.setattr(module, "cached_services", lambda cases: nullcontext())
    monkeypatch.setattr(module, "gpu_services", lambda *args: {"asr_device": "cuda"})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(module.frozen.evaluation, "EVAL_DIR", tmp_path / "raw")
    monkeypatch.setattr(module.frozen.evaluation.infer, "CHECKPOINT_DIR", tmp_path)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VOXFT_WHISPER_MODEL_LARGE",
                "VOXFT_SPK_EMB_MODEL", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(key, "test")
    generated = []

    def evaluate(target, lang, texts, *, base, seeds, cfg_value, inference_timesteps):
        assert cfg_value == 1.8 and inference_timesteps == 20 and len(texts) == len(seeds) == 1
        generated.append((target, texts[0]["case_id"], seeds[0]))
        path = tmp_path / f"audio{len(generated)}.wav"
        path.write_bytes(b"preserved")
        return {
            "target": target, "base": base, "label": "base" if target == "base" else "r26",
            "lora_strength": 0 if target == "base" else 1,
            "cfg_value": cfg_value, "inference_timesteps": inference_timesteps,
            "retry_badcase": False, "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
            "items": [{**texts[0], "seed": seeds[0], "wav": str(path), "cer": 0.1, "wer": 0.1,
                       "speaker_sim": 0.9, "numeric": False, "suspected_truncation": False,
                       "over_read": False, "audio_sec": 2.0, "f0_std_st": 1.0}],
        }

    monkeypatch.setattr(module.frozen.evaluation, "evaluate", evaluate)
    return cases, plan, generated


@pytest.mark.parametrize("cuda_available,asr_device", [(False, "cuda"), (True, "cpu")])
def test_gpu_failures_never_fall_back_to_cpu(module, monkeypatch, cuda_available, asr_device):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(module.pipeline, "_whisper_model",
                        lambda *args: SimpleNamespace(model=SimpleNamespace(device=asr_device)))
    with pytest.raises(RuntimeError, match="CUDA|CPU fallback"):
        module.gpu_services({}, [], "base")


def test_worker_preserves_completed_items_on_failure(module, tmp_path, monkeypatch, experiment):
    _, _, generated = experiment
    for name in ("claims", "items"):
        (tmp_path / name).mkdir()
    original = module.frozen.evaluation.evaluate

    def fail_second(*args, **kwargs):
        if generated:
            raise RuntimeError("deliberate")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.frozen.evaluation, "evaluate", fail_second)
    with pytest.raises(RuntimeError, match="deliberate"):
        module.worker(tmp_path, "r26")
    report = json.loads((tmp_path / "items/r26_0004_42.json").read_text())
    assert report["items"][0]["lang"] == "id" and len(generated) == 1
    assert report["wav_sha256"] == {
        report["items"][0]["wav"]: module.frozen.sha256(report["items"][0]["wav"])}
    assert (tmp_path / "claims/r26_0004_43.json").exists()
    with pytest.raises(FileExistsError):
        module.worker(tmp_path, "r26")
    assert len(generated) == 1


@pytest.mark.parametrize("fault", ["none", "wav", "seed", "metadata", "missing", "worker"])
def test_controller_requires_complete_unchanged_results(
        module, tmp_path, monkeypatch, experiment, fault):
    _, plan, generated = experiment
    processes = []

    class Process:
        pid = 100

        def __init__(self, command, **kwargs):
            assert command[-2] == "--model" and command[-1] in module.MODELS
            processes.append(self)
            module.worker(tmp_path, command[-1])
            if len(processes) == 2:
                path = tmp_path / "items/r26_0004_42.json"
                report = json.loads(path.read_text())
                if fault == "wav":
                    Path(report["items"][0]["wav"]).write_bytes(b"changed")
                elif fault == "seed":
                    report["items"][0]["seed"] = 99
                    module.frozen.write_json(path, report)
                elif fault == "metadata":
                    report["items"][0]["control"] = "changed"
                    module.frozen.write_json(path, report)
                elif fault == "missing":
                    path.unlink()

        def poll(self):
            return 1 if fault == "worker" else 0

    monkeypatch.setattr(module.subprocess, "check_output",
                        lambda command, **kwargs: "" if "--query-compute-apps=pid" in command else "40000")
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    if fault != "none":
        with pytest.raises((ValueError, RuntimeError)):
            module.run(tmp_path)
        assert json.loads((tmp_path / "status.json").read_text())["stage"] == "failed"
        assert not (tmp_path / "comparison.json").exists()
        return
    module.run(tmp_path)
    assert len(generated) == plan["generation_budget"]
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "complete"
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert set(comparison["reports"]) == set(module.MODELS)
    assert set(comparison["comparisons"]) == {"r26_minus_base"}
    for model in module.MODELS:
        report = json.loads((tmp_path / f"{model}_report.json").read_text())
        assert len(report["items"]) == len(report["wav_sha256"]) == 30
