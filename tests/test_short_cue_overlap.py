import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("short_cue_overlap", scripts / "short_cue_overlap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linux_exit_status_and_recycled_pid_guard(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    directory = tmp_path / "17"
    directory.mkdir()
    fields = ["0"] * 50
    fields[0], fields[1], fields[19], fields[49] = "Z", "12", "456", str(7 << 8)
    (directory / "stat").write_text("17 (name with ) brackets) " + " ".join(fields))
    state = module.process_state(17, tmp_path)
    assert state == {"state": "Z", "parent": 12, "start": 456, "exit_code": 1792}
    assert module.process_state(18, tmp_path) is None
    signals = []
    monkeypatch.setattr(module.os, "kill", lambda *args: signals.append(args))
    monkeypatch.setattr(module, "process_state", lambda pid: {**state, "state": "S", "start": 999})
    module.signal_same_process(17, state, module.signal.SIGKILL)
    assert not signals


@pytest.mark.parametrize("fail", [False, True])
def test_overlap_keeps_live_workers_and_never_repeats_jobs(monkeypatch, tmp_path, fail):
    module = load_module(monkeypatch)
    supervisor = 100
    workers = {"control_short": 101, "short_short": 102}
    jobs = ("base_short", "r8_short", *workers, "control_control", "short_control")
    plan = {"jobs": dict.fromkeys(jobs, {}), "total_outputs": 990}
    original = {"stage": "evaluation", "jobs": list(workers), "pids": list(workers.values())}
    (tmp_path / "status.json").write_text(json.dumps(original))
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    for name in jobs[:2]:
        (tmp_path / f"{name}_report.json").write_text("{}")
    states = {pid: {"state": "S", "parent": 1 if pid == supervisor else supervisor,
                    "start": pid + 1000, "exit_code": 0} for pid in (supervisor, *workers.values())}
    real_read = Path.read_bytes

    def read_command(path):
        if path.parent.parent == Path("/proc"):
            pid = int(path.parent.name)
            command = ["python", "short_cue_eval.py", "run" if pid == supervisor else "worker",
                       "--output", str(tmp_path)]
            if pid != supervisor:
                command += ["--job", next(name for name, number in workers.items() if number == pid)]
            return b"\0".join(part.encode() for part in command)
        return real_read(path)

    def kill(pid, action):
        states[pid]["state"] = "T" if action == module.signal.SIGSTOP else "Z"

    monkeypatch.setattr(Path, "read_bytes", read_command)
    monkeypatch.setattr(module, "process_state", lambda pid: states.get(pid))
    monkeypatch.setattr(module.os, "kill", kill)
    monkeypatch.setattr(module, "check_inputs", lambda path: plan)
    monkeypatch.setattr(module, "runtime_environment", lambda path: None)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=4 * 1024**3))
    validated = []
    monkeypatch.setattr(module, "validate_report", lambda plan, job, report: validated.append(report))
    spawned = []

    class Process:
        def __init__(self, command, **kwargs):
            self.name = command[-1]
            self.pid = 200 + len(spawned)
            self.returncode = None
            states[self.pid] = {"state": "S", "parent": 1, "start": self.pid + 1000, "exit_code": 0}
            spawned.append(self)
            assert self.name in ("control_control", "short_control")
            assert sum(state["state"] == "S" for pid, state in states.items() if pid != supervisor) <= 3

        def poll(self):
            if states[self.pid]["state"] == "Z":
                self.returncode = -11 if fail else 0
            return self.returncode

        def wait(self):
            return self.poll()

    def tick(seconds):
        for name, pid in workers.items():
            if not fail:
                states[pid]["state"] = "Z"
                (tmp_path / f"{name}_report.json").write_text("{}")
        for process in spawned:
            states[process.pid]["state"] = "Z"
            if not fail:
                (tmp_path / f"{process.name}_report.json").write_text("{}")

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module.time, "sleep", tick)
    if fail:
        with pytest.raises(RuntimeError, match="Smoke worker failed"):
            module.run(tmp_path, supervisor)
        assert [process.name for process in spawned] == ["control_control"]
        assert not (tmp_path / "done").exists()
        assert json.loads((tmp_path / "status.json").read_text())["stage"] == "failed"
    else:
        module.run(tmp_path, supervisor)
        assert [process.name for process in spawned] == ["control_control", "short_control"]
        assert (tmp_path / "done").exists()
        assert json.loads((tmp_path / "status.json").read_text())["outputs"] == 990
        assert len(validated) == 12
    assert all(state["state"] == "Z" for state in states.values())
    assert (tmp_path / "plan.json").read_text() == json.dumps(plan)
