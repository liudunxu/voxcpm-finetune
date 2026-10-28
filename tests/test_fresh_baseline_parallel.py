import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fresh_baseline_parallel")


@pytest.fixture
def experiment(module, tmp_path, monkeypatch):
    baseline = module.baseline
    cases = [{"case_id": f"{lang}_{condition}", "sentence_id": lang, "condition": condition,
              "lang": lang, "text": f"hello {lang}", "ref_audio": "/ref",
              "control": "" if condition == "plain" else "angry"}
             for lang in ("th", "tl", "id") for condition in ("plain", "control")]
    plan = {
        **baseline.SETTINGS, "base_path": "/base", "targets": {"base": "base", "r26": "/r26"},
        "asr_path": "/whisper", "speaker_model_path": "/speaker", "generation_budget": 60,
        "inputs_sha256": {},
    }
    monkeypatch.setattr(baseline, "checked", lambda work: plan)
    monkeypatch.setattr(module.frozen, "runtime_environment", lambda path: None)
    monkeypatch.setattr(baseline, "cached_services", lambda cases: nullcontext())
    monkeypatch.setattr(baseline, "gpu_services", lambda *args: {"asr_device": "cuda"})
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
        directory = module.frozen.evaluation.infer.CHECKPOINT_DIR / "auditions"
        directory.mkdir(exist_ok=True)
        path = directory / f"audio{len(generated)}.wav"
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
    for name in ("claims", "items", "parallel4"):
        (tmp_path / name).mkdir()
    (tmp_path / "cases.jsonl").write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
    baseline.save_new(tmp_path / "plan.json", plan)
    completed, inputs = set(), {}
    for model in baseline.MODELS:
        identifier, case, seed = next(baseline.tasks(cases, model))
        assert baseline.claim(tmp_path / "claims", identifier)
        report = evaluate(plan["targets"][model], "th", [case], base=plan["base_path"],
                          seeds=[seed], cfg_value=1.8, inference_timesteps=20)
        wav = Path(report["items"][0]["wav"])
        report["wav_sha256"] = {str(wav): module.frozen.sha256(wav)}
        path = tmp_path / "items" / f"{identifier}.json"
        baseline.save_new(path, report)
        completed.add(identifier)
        for artifact in (path, wav, tmp_path / "claims" / path.name):
            inputs[str(artifact)] = module.frozen.sha256(artifact)
    return SimpleNamespace(cases=cases, plan=plan, completed=completed, inputs=inputs,
                           generated=generated, evaluate=evaluate)


def prepare(module, work, experiment):
    module.baseline.save_new(work / "parallel4/preserved.json", {
        "completed": sorted(experiment.completed), "inputs_sha256": experiment.inputs,
    })
    module.prepare(work, work / "parallel4", experiment.completed)


def test_four_disjoint_lanes_cover_only_remaining_frozen_budget(module):
    cases = module.frozen._read_manifest(module.baseline.CASES)
    original = json.dumps(cases, ensure_ascii=False)
    tasks = [task for model in module.baseline.MODELS for task in module.baseline.tasks(cases, model)]
    completed = {task[0] for task in tasks[::7]}
    workers = module.assignments(cases, completed)
    assigned = [identifier for worker in workers for identifier in worker["tasks"]]
    assert [worker["model"] for worker in workers] == ["base", "base", "r26", "r26"]
    assert len(assigned) == len(set(assigned)) == 1800 - len(completed)
    assert set(assigned).isdisjoint(completed)
    assert set(assigned) | completed == {task[0] for task in tasks}
    assert json.dumps(cases, ensure_ascii=False) == original
    for worker in workers:
        ordered = [task for task in tasks if task[0] in worker["tasks"]]
        languages = [task[1]["lang"] for task in ordered]
        first_other = next(index for index, lang in enumerate(languages) if lang != "id")
        assert "id" not in languages[first_other:]


@pytest.mark.parametrize("fault", ["none", "wrong_parent", "reused_pid", "worker_error",
                                  "missing_report", "unmapped_wav"])
def test_drain_finishes_inflight_items_without_signalling_workers(
        module, tmp_path, monkeypatch, experiment, fault):
    supervisor, base_pid, r26_pid = 100, 101, 102
    states = {pid: {"state": "S", "start": 1000 + pid,
                    "parent": 1 if pid == supervisor else supervisor, "exit_code": 0}
              for pid in (supervisor, base_pid, r26_pid)}
    if fault == "wrong_parent":
        states[base_pid]["parent"] = 999
    module.baseline.save_new(tmp_path / "status.json", {
        "stage": "evaluating", "pids": [base_pid, r26_pid], "completed_items": 2})
    pending = {}
    for model in module.baseline.MODELS:
        pending[model] = list(module.baseline.tasks(experiment.cases, model))[1]
        assert module.baseline.claim(tmp_path / "claims", pending[model][0])
    original_read = Path.read_bytes

    def read_command(path):
        if path.parent.parent == Path("/proc"):
            pid = int(path.parent.name)
            command = ["python", str(Path(module.baseline.__file__).resolve()),
                       "run" if pid == supervisor else "worker", "--work", str(tmp_path)]
            if pid != supervisor:
                command.extend(["--model", "base" if pid == base_pid else "r26"])
            return b"\0".join(part.encode() for part in command)
        return original_read(path)

    signals = []

    def send_signal(pid, identity, action):
        assert pid == supervisor
        signals.append((pid, action))
        states[pid]["state"] = "T" if action == module.signal.SIGSTOP else "Z"
        if action == module.signal.SIGCONT:
            module.frozen.write_json(tmp_path / "status.json", {
                "stage": "failed", "error": "Worker failed; preserve outputs, no automatic retry"})

    def advance(delay):
        assert states[supervisor]["state"] == "T"
        for model, pid in (("base", base_pid), ("r26", r26_pid)):
            identifier, case, seed = pending[model]
            report = experiment.evaluate(
                experiment.plan["targets"][model], "th", [case], base="/base",
                seeds=[seed], cfg_value=1.8, inference_timesteps=20)
            report["wav_sha256"] = {
                report["items"][0]["wav"]: module.frozen.sha256(report["items"][0]["wav"])}
            if fault != "missing_report":
                module.baseline.save_new(tmp_path / "items" / f"{identifier}.json", report)
            barrier = module.read_json(tmp_path / f"parallel4/{model}_barrier.json")["identifier"]
            (tmp_path / f"{model}.log").write_text(
                f"RuntimeError: Task already claimed; no automatic retry: {barrier}\n")
            states[pid].update(state="Z", exit_code=256)
        if fault == "worker_error":
            (tmp_path / "base.log").write_text("RuntimeError: unrelated failure\n")
        elif fault == "reused_pid":
            states[base_pid]["start"] += 1
        elif fault == "unmapped_wav":
            (tmp_path / "auditions/orphan.wav").write_bytes(b"orphan")

    monkeypatch.setattr(Path, "read_bytes", read_command)
    monkeypatch.setattr(module, "process_state", lambda pid: dict(states[pid]))
    monkeypatch.setattr(module, "signal_same_process", send_signal)
    monkeypatch.setattr(module.time, "sleep", advance)
    if fault != "none":
        with pytest.raises((ValueError, RuntimeError)):
            module.drain(tmp_path, tmp_path / "parallel4", supervisor)
        assert not (tmp_path / "parallel4/preserved.json").exists()
        assert signals == ([] if fault == "wrong_parent" else
                           [(supervisor, module.signal.SIGSTOP), (supervisor, module.signal.SIGCONT)])
    else:
        completed = module.drain(tmp_path, tmp_path / "parallel4", supervisor)
        assert completed == experiment.completed | {task[0] for task in pending.values()}
        assert len(module.read_json(tmp_path / "parallel4/preserved.json")["completed"]) == 4
        assert len(signals) == 2
    assert all(module.frozen.sha256(path) == digest for path, digest in experiment.inputs.items())


@pytest.mark.parametrize("fault", ["scope", "assignment", "old_report", "old_plan", "child_cases"])
def test_schedule_rejects_changed_scope_or_preserved_evidence(
        module, tmp_path, experiment, fault):
    prepare(module, tmp_path, experiment)
    schedule = module.checked(tmp_path)
    if fault == "scope":
        schedule["max_workers"] = 5
    elif fault == "assignment":
        schedule["workers"][0]["tasks"].append(next(iter(experiment.completed)))
    elif fault == "old_report":
        next((tmp_path / "items").glob("*.json")).write_text("{}")
    elif fault == "old_plan":
        (tmp_path / "plan.json").write_text("{}")
    elif fault == "child_cases":
        (tmp_path / "parallel4/worker0/cases.jsonl").write_text("{}")
    if fault in ("scope", "assignment"):
        module.frozen.write_json(tmp_path / "parallel4/plan.json", schedule)
    with pytest.raises(ValueError):
        module.checked(tmp_path)


@pytest.mark.parametrize("fault", ["none", "worker", "low_memory", "wav", "seed", "settings", "missing"])
def test_four_workers_preserve_successes_and_require_exact_complete_merge(
        module, tmp_path, monkeypatch, experiment, fault):
    prepare(module, tmp_path, experiment)
    processes = []
    original_tasks = module.baseline.tasks

    class Process:
        def __init__(self, command, **kwargs):
            self.index = int(command[-1])
            self.pid = 1000 + self.index
            assert command[-2] == "--index" and command[5] == "worker"
            processes.append(self)
            module.worker(tmp_path, self.index)
            assert module.baseline.tasks is original_tasks
            if self.index == 3:
                path = next((tmp_path / "parallel4/worker3/items").glob("*.json"))
                report = module.read_json(path)
                if fault == "wav":
                    Path(report["items"][0]["wav"]).write_bytes(b"changed")
                elif fault == "seed":
                    report["items"][0]["seed"] = 99
                    module.frozen.write_json(path, report)
                elif fault == "settings":
                    report["cfg_value"] = 2.0
                    module.frozen.write_json(path, report)
                elif fault == "missing":
                    path.unlink()

        def poll(self):
            return 1 if fault == "worker" and self.index == 3 else 0

    def memory(command, **kwargs):
        if "--query-compute-apps=pid" in command:
            return ""
        return "10000" if fault == "low_memory" and len(processes) == 3 else "40000"

    monkeypatch.setattr(module.subprocess, "check_output", memory)
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    if fault != "none":
        with pytest.raises((ValueError, RuntimeError)):
            module.run(tmp_path)
        assert module.read_json(tmp_path / "status.json")["stage"] == "failed"
        assert not (tmp_path / "comparison.json").exists()
        assert {path.stem for path in (tmp_path / "items").glob("*.json")} == experiment.completed
    else:
        module.run(tmp_path)
        assert len(processes) == 4
        status = module.read_json(tmp_path / "status.json")
        assert status["stage"] == "complete" and status["max_workers"] == 4
        assert status["completed_items"] == experiment.plan["generation_budget"] == 60
        assert status["training_started"] is status["production_changed"] is False
        assert (tmp_path / "comparison.json").exists()
        for model in module.baseline.MODELS:
            report = module.read_json(tmp_path / f"{model}_report.json")
            assert len(report["items"]) == len(report["wav_sha256"]) == 30
    assert len(experiment.generated) == len(set(experiment.generated))
    assert all(module.frozen.sha256(path) == digest for path, digest in experiment.inputs.items())
    with pytest.raises(FileExistsError, match="implicitly resume"):
        module.run(tmp_path)
