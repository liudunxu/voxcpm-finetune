import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["none", "wrong_parent", "baseline", "changed_source", "train"])
def test_handoff_never_signals_live_baseline_and_stops_on_failure(tmp_path, monkeypatch, failure):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("control_ab_fast_handoff")
    shell, supervisor = 100, 101
    runner = str(Path(module.frozen.__file__).resolve())
    coordinator = str(Path(module.overlap.__file__).resolve())
    states = {shell: {"state": "S", "start": 1000, "parent": 1, "exit_code": 0},
              supervisor: {"state": "S", "start": 1001, "parent": shell, "exit_code": 0}}
    if failure == "wrong_parent":
        states[supervisor]["parent"] = 999
    inputs = tmp_path / "source.py"
    inputs.write_text("original")
    for name, data in {
        "plan.json": {},
        "fast_eval_profile.json": {"metrics_exactly_equal": True,
                                  "source_sha256": module.frozen.sha256(module.fast.__file__)},
        "status.json": {"stage": "baseline_overlap", "supervisor_pid": supervisor},
        "overlap_schedule.json": {"source_sha256": {str(inputs): module.frozen.sha256(inputs)}},
        "base_report.json": {}, "r23_report.json": {},
        "baseline_comparison.json": {"inputs_sha256": {}},
    }.items():
        module.frozen.write_json(tmp_path / name, data)
    original_read = Path.read_bytes

    def read_command(path):
        if path.parent.parent == Path("/proc"):
            command = (["bash", "-c", f"{coordinator} {runner} {tmp_path}"] if
                       int(path.parent.name) == shell else
                       ["python", coordinator, "--work", str(tmp_path)])
            return b"\0".join(part.encode() for part in command)
        return original_read(path)

    signals, calls, validated = [], [], []

    def signal_process(pid, identity, action):
        assert pid == shell
        assert identity["start"] == states[pid]["start"]
        signals.append(action)
        states[pid]["state"] = "T" if action == module.signal.SIGSTOP else "Z"

    def advance(delay):
        assert states[shell]["state"] == "T"
        states[supervisor]["state"] = "Z"
        states[supervisor]["exit_code"] = 1 if failure == "baseline" else 0
        module.frozen.write_json(tmp_path / "status.json", {"stage": "baseline_complete"})
        if failure == "changed_source":
            inputs.write_text("changed")

    def run(command, **kwargs):
        assert states[shell]["state"] == "Z"
        calls.append(command)
        if failure == "train":
            raise RuntimeError("training failed")

    monkeypatch.setattr(Path, "read_bytes", read_command)
    monkeypatch.setattr(module.overlap, "process_state", lambda pid: dict(states[pid]))
    monkeypatch.setattr(module.overlap, "signal_same_process", signal_process)
    monkeypatch.setattr(module.overlap, "validate_shard",
                        lambda work, plan, cases, name: validated.append(name))
    monkeypatch.setattr(module.frozen, "check_inputs", lambda work: {})
    monkeypatch.setattr(module.frozen, "_read_manifest", lambda path: [])
    monkeypatch.setattr(module.frozen, "verify_report", lambda *args: None)
    monkeypatch.setattr(module.time, "sleep", advance)
    monkeypatch.setattr(module.subprocess, "run", run)
    if failure != "none":
        with pytest.raises((RuntimeError, ValueError)):
            module.handoff(tmp_path, shell, supervisor)
        assert len(calls) == (1 if failure == "train" else 0)
        if failure == "wrong_parent":
            assert signals == []
        else:
            assert json.loads((tmp_path / "fast_handoff_failed.json").read_text())[
                "baseline_not_interrupted"] is True
        return
    module.handoff(tmp_path, shell, supervisor)
    assert len(calls) == 2 and calls[0][5] == "train" and calls[1][5] == "run"
    assert validated == ["base_shard0", "base_shard1", "r23_shard0", "r23_shard1"]
    assert signals[0] == module.signal.SIGSTOP and signals[-1] == module.signal.SIGKILL
