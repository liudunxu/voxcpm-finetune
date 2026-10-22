import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("control_ab_overlap")


def test_overlap_refuses_existing_r23_artifacts(module, tmp_path):
    module.require_unstarted_r23(tmp_path)
    (tmp_path / "r23_shard0.log").write_text("preserve")
    with pytest.raises(FileExistsError):
        module.require_unstarted_r23(tmp_path)


def test_shard_requires_frozen_audio(module, tmp_path, monkeypatch):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"test")
    report = {"items": [{"wav": str(audio)}], "wav_sha256": {str(audio): module.frozen.sha256(audio)}}
    (tmp_path / "base_shard0.json").write_text(json.dumps(report))
    observed = []
    monkeypatch.setattr(module.frozen, "verify_report", lambda report, plan, cases, model:
                        observed.append((cases, model)))
    module.validate_shard(tmp_path, {}, [0, 1, 2, 3], "base_shard0")
    assert observed == [([0, 2], "base")]
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen WAV"):
        module.validate_shard(tmp_path, {}, [0, 1, 2, 3], "base_shard0")


@pytest.mark.parametrize("failure", ["none", "child", "low_memory", "wrong_parent"])
def test_handover_preserves_base_and_runs_each_r23_shard_once(
        module, tmp_path, monkeypatch, failure):
    supervisor, base_pids = 100, [101, 102]
    runner = Path(module.frozen.__file__).resolve()
    plan = {"inputs_sha256": {str(runner): module.frozen.sha256(runner)}}
    original = {"stage": "baseline", "time": 123}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "status.json").write_text(json.dumps(original))
    (tmp_path / "preserved.wav").write_bytes(b"original")
    states = {pid: {"state": "S", "parent": 1 if pid == supervisor else supervisor,
                    "start": pid + 1000, "exit_code": 0} for pid in [supervisor, *base_pids]}
    if failure == "wrong_parent":
        states[101]["parent"] = 99
    real_read = Path.read_bytes

    def read_command(path):
        if path.parent.parent == Path("/proc"):
            pid = int(path.parent.name)
            command = ["python", str(runner), "baseline" if pid == supervisor else "worker",
                       "--work", str(tmp_path)]
            if pid != supervisor:
                command += ["--model", "base", "--shard", str(base_pids.index(pid))]
            return b"\0".join(part.encode() for part in command)
        return real_read(path)

    signals = []

    def send_signal(pid, identity, action):
        assert states[pid]["start"] == identity["start"]
        signals.append((pid, action))
        states[pid]["state"] = ("T" if action == module.signal.SIGSTOP else
                                "S" if action == module.signal.SIGCONT else "Z")

    monkeypatch.setattr(Path, "read_bytes", read_command)
    monkeypatch.setattr(module, "process_state", lambda pid: dict(states[pid]) if pid in states else None)
    monkeypatch.setattr(module, "signal_same_process", send_signal)
    monkeypatch.setattr(module, "free_gpu_mib", lambda: (8 if failure == "low_memory" else 28) * 1024)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=22 * 1024**3))
    monkeypatch.setattr(module.frozen, "check_inputs", lambda path: plan)
    monkeypatch.setattr(module.frozen, "_read_manifest", lambda path: [])
    validated = []
    monkeypatch.setattr(module, "validate_shard", lambda work, plan, cases, name: validated.append(name))
    monkeypatch.setattr(module.frozen.evaluation, "merge_reports", lambda paths: {"paths": paths})
    monkeypatch.setattr(module.frozen, "verify_report", lambda *args: None)
    collected = []
    monkeypatch.setattr(module.frozen, "collect", lambda work, names, path: collected.append(names))
    spawned = []

    class Process:
        def __init__(self, command, **kwargs):
            assert command[command.index("--model") + 1] == "r23"
            self.shard = command[-1]
            self.pid = 200 + len(spawned)
            states[self.pid] = {"state": "S", "parent": 1, "start": self.pid + 1000, "exit_code": 0}
            spawned.append(self)
            assert sum(state["state"] == "S" for state in states.values()) <= 3

        def poll(self):
            return (-11 if failure == "child" else 0) if states[self.pid]["state"] == "Z" else None

        def wait(self):
            return self.poll()

    def tick(seconds):
        if failure != "child":
            for pid in base_pids:
                states[pid]["state"] = "Z"
        for process in spawned:
            states[process.pid]["state"] = "Z"

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.time, "sleep", tick)
    if failure in ("low_memory", "wrong_parent"):
        with pytest.raises((RuntimeError, ValueError)):
            module.run(tmp_path, supervisor, base_pids)
        assert not signals and not spawned
        assert json.loads((tmp_path / "status.json").read_text()) == original
    elif failure == "child":
        with pytest.raises(RuntimeError, match="r23 worker failed"):
            module.run(tmp_path, supervisor, base_pids)
        assert [process.shard for process in spawned] == ["0"]
        assert not collected
        assert json.loads((tmp_path / "status.json").read_text())["stage"] == "baseline_failed"
        assert all(state["state"] == "Z" for state in states.values())
    else:
        module.run(tmp_path, supervisor, base_pids)
        assert [process.shard for process in spawned] == ["0", "1"]
        assert set(validated) == {"base_shard0", "base_shard1", "r23_shard0", "r23_shard1"}
        assert collected == [("base", "r23")]
        assert json.loads((tmp_path / "status.json").read_text())["stage"] == "baseline_complete"
        assert signals == [(supervisor, module.signal.SIGSTOP), (supervisor, module.signal.SIGKILL)]
    assert (tmp_path / "preserved.wav").read_bytes() == b"original"
    assert (tmp_path / "plan.json").read_text() == json.dumps(plan)
