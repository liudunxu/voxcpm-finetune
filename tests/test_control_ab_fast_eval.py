import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("control_ab_fast_eval")


def test_tasks_preserve_all_models_cases_and_seeds(module):
    cases = [{"case_id": "one"}, {"case_id": "two"}]
    tasks = list(module.jobs(cases))
    assert len(tasks) == 2 * 2 * 5
    assert len({identifier for identifier, *_ in tasks}) == len(tasks)
    assert {(model, case["case_id"], seed) for _, model, case, seed in tasks} == {
        (model, case["case_id"], seed)
        for model in ("off", "on") for case in cases for seed in module.frozen.SEEDS}
    reverse = list(module.jobs(cases, "on"))
    assert reverse[0][1] == "on"
    assert {row[0] for row in reverse} == {row[0] for row in tasks}


def test_claim_never_reuses_an_inflight_or_finished_task(module, tmp_path):
    assert module.claim(tmp_path, "off_0000_42") is True
    original = (tmp_path / "off_0000_42.json").read_bytes()
    assert module.claim(tmp_path, "off_0000_42") is False
    assert (tmp_path / "off_0000_42.json").read_bytes() == original


def test_reference_cache_keeps_values_and_restores_functions(module, monkeypatch):
    reference = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    calls, whisper_calls, caches = [], [], []

    def embedding(waveform, sample_rate):
        calls.append(module.pcm_key(waveform, sample_rate))
        return np.array([float(np.sum(waveform)), sample_rate])

    def whisper(*args, **kwargs):
        whisper_calls.append(args)
        return object()

    def acoustics(wav_path, text, ref_path, ref_cache):
        caches.append(ref_cache)
        return {"speaker_sim": float(np.sum(module.qc.speaker_embedding(target, 16000)))}

    monkeypatch.setattr(module.pipeline, "load_wav_mono", lambda path: (reference.copy(), 16000))
    monkeypatch.setattr(module.qc, "speaker_embedding", embedding)
    monkeypatch.setattr(module.pipeline, "_whisper_model", whisper)
    monkeypatch.setattr(module.frozen.evaluation, "_acoustics", acoustics)
    expected = embedding(reference, 16000)
    calls.clear()
    with pytest.raises(ValueError, match="deliberate"):
        with module.cached_services([{"ref_audio": "/ref"}]):
            for _ in range(3):
                assert np.array_equal(module.qc.speaker_embedding(reference.copy(), 16000), expected)
                module.qc.speaker_embedding(target.copy(), 16000)
            module.qc.speaker_embedding(reference.copy(), 8000)
            assert module.pipeline._whisper_model("th", "large-v3") is module.pipeline._whisper_model(
                "th", "large-v3")
            module.frozen.evaluation._acoustics("/wav", "text", "/ref", {})
            module.frozen.evaluation._acoustics("/wav", "text", "/ref", {})
            raise ValueError("deliberate")
    assert calls.count(module.pcm_key(reference, 16000)) == 1
    assert calls.count(module.pcm_key(target, 16000)) == 5
    assert calls.count(module.pcm_key(reference, 8000)) == 1
    assert len(whisper_calls) == 1 and caches[0] is caches[1]
    assert module.qc.speaker_embedding is embedding
    assert module.pipeline._whisper_model is whisper
    assert module.frozen.evaluation._acoustics is acoustics


def test_existing_run_is_not_overwritten(module, tmp_path):
    (tmp_path / "fast_eval").mkdir()
    with pytest.raises(FileExistsError, match="implicitly resume"):
        module.run(tmp_path)


@pytest.fixture
def experiment(module, tmp_path, monkeypatch):
    cases = [{"case_id": "one", "text": "hello", "lang": "id", "control": ""},
             {"case_id": "two", "text": "สวัสดี", "lang": "th", "control": "开心"}]
    plan = {"base_path": "/base", "targets": {"off": "/off", "on": "/on"},
            "cfg_value": 1.8, "inference_timesteps": 20, "num_iters": 393, "cases": len(cases)}
    module.frozen.write_json(tmp_path / "plan.json", plan)
    module.frozen.write_json(tmp_path / "fast_eval_profile.json", {
        "metrics_exactly_equal": True, "source_sha256": module.frozen.sha256(module.__file__)})
    for model in ("off", "on"):
        weight = tmp_path / f"{model}.weights"
        weight.write_bytes(b"frozen")
        module.frozen.write_json(tmp_path / f"trained_{model}.json", {
            "step": 393, "inputs_sha256": {str(weight): module.frozen.sha256(weight)}})
    monkeypatch.setattr(module.frozen, "check_inputs", lambda work: plan)
    monkeypatch.setattr(module.frozen, "_read_manifest", lambda path: cases)
    monkeypatch.setattr(module.frozen, "runtime_environment", lambda work: None)
    monkeypatch.setattr(module, "cached_services", lambda cases: nullcontext())
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=22 * 1024**3))
    monkeypatch.setattr(module.frozen.evaluation, "EVAL_DIR", tmp_path / "raw")
    monkeypatch.setattr(module.frozen.evaluation.infer, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(module.frozen.evaluation.infer, "get_model", lambda base, lora: lora)
    monkeypatch.setattr(module.pipeline, "_whisper_model", lambda *args: object())
    monkeypatch.setattr(module.frozen.evaluation, "_transcribe", lambda *args: "hello")
    monkeypatch.setattr(module.frozen.evaluation, "_acoustics",
                        lambda *args: {"audio_sec": 1.0, "speaker_sim": 0.9, "f0_std_st": 1.0})
    generated = []

    def generate(model, kwargs):
        assert kwargs["retry_badcase"] is False
        assert kwargs["cfg_value"] == 1.8 and kwargs["inference_timesteps"] == 20
        generated.append((model, kwargs["seed"], kwargs["text"]))
        path = tmp_path / f"audio{len(generated)}.wav"
        path.write_bytes(b"preserved")
        return str(path), 1.0

    monkeypatch.setattr(module.frozen.evaluation.infer, "_run", generate)
    return cases, plan, generated


def test_worker_persists_each_item_and_skips_claimed_tasks(module, tmp_path, monkeypatch, experiment):
    cases, plan, generated = experiment
    directory = tmp_path / "fast_eval"
    (directory / "claims").mkdir(parents=True)
    (directory / "items").mkdir()
    module.frozen.write_json(directory / "plan.json", {"inputs_sha256": {}})
    original = module.frozen.evaluation.evaluate

    def fail_second(*args, **kwargs):
        if generated:
            raise RuntimeError("deliberate")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.frozen.evaluation, "evaluate", fail_second)
    with pytest.raises(RuntimeError, match="deliberate"):
        module.worker(tmp_path, directory, 0)
    report_path = directory / "items/off_0000_42.json"
    preserved = report_path.read_bytes()
    report = json.loads(preserved)
    assert len(report["items"]) == 1 and len(generated) == 1
    assert report["wav_sha256"] == {
        report["items"][0]["wav"]: module.frozen.sha256(report["items"][0]["wav"])}
    assert (directory / "claims/off_0000_43.json").exists()
    monkeypatch.setattr(module.frozen.evaluation, "evaluate", original)
    module.worker(tmp_path, directory, 1)
    assert report_path.read_bytes() == preserved
    assert len(generated) == len(cases) * len(module.frozen.SEEDS) * 2 - 1
    assert not (directory / "items/off_0000_43.json").exists()


@pytest.mark.parametrize("failure", ["none", "worker", "wav", "seed", "checkpoint"])
def test_controller_merges_exact_tasks_or_stops_safely(
        module, tmp_path, monkeypatch, experiment, failure):
    cases, plan, generated = experiment
    collected, processes = [], []
    directory = tmp_path / "fast_eval"

    def collect(work, names, output):
        collected.append(names)
        module.frozen.write_json(output, {"complete": True})

    def query(command, **kwargs):
        if "--query-compute-apps=pid" in command:
            return ""
        return "40000" if not processes else "8000"

    class Process:
        pid = 100

        def __init__(self, command, **kwargs):
            assert command[-2:] == ["--index", "0"]
            processes.append(self)
            module.worker(tmp_path, directory, 0)
            if failure == "wav":
                (tmp_path / "audio1.wav").write_bytes(b"modified")
            if failure == "seed":
                path = directory / "items/off_0000_42.json"
                report = json.loads(path.read_text())
                report["items"][0]["seed"] = 99
                module.frozen.write_json(path, report)
            if failure == "checkpoint":
                (tmp_path / "off.weights").write_bytes(b"modified")

        def poll(self):
            return 1 if failure == "worker" else 0

    monkeypatch.setattr(module.frozen, "collect", collect)
    monkeypatch.setattr(module.subprocess, "check_output", query)
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    if failure != "none":
        with pytest.raises((ValueError, RuntimeError)):
            module.run(tmp_path)
        assert collected == [] and not (tmp_path / "on_report.json").exists()
        assert json.loads((tmp_path / "status.json").read_text())["stage"] == "evaluate_failed"
        assert len(list((directory / "items").glob("*.json"))) == 20
        return
    module.run(tmp_path)
    assert len(generated) == 20
    for model in ("off", "on"):
        report = json.loads((tmp_path / f"{model}_report.json").read_text())
        module.frozen.verify_report(report, plan, cases, model)
        assert len(report["items"]) == len(report["wav_sha256"]) == 10
    assert collected == [("base", "r23", "off", "on")]
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "evaluate_complete"
