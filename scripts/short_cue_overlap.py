"""One-off r11 schedule handover: retain live short jobs and overlap smoke tests, max three."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from short_cue_eval import ROOT, check_inputs, runtime_environment, sha256, validate_report, write_json


def process_state(pid, proc=Path("/proc")):
    try:
        fields = (proc / str(pid) / "stat").read_text().rpartition(")")[2].split()
    except FileNotFoundError:
        return None
    return {"state": fields[0], "parent": int(fields[1]),
            "start": int(fields[19]), "exit_code": int(fields[49])}


def signal_same_process(pid, identity, action):
    current = process_state(pid)
    if current and current["start"] == identity["start"] and current["state"] != "Z":
        os.kill(pid, action)


def run(output, supervisor):
    plan = check_inputs(output)
    original = json.loads((output / "status.json").read_text())
    if (original["stage"] != "evaluation"
            or original["jobs"] != ["control_short", "short_short"]
            or len(original["pids"]) != 2 or (output / "done").exists()):
        raise ValueError("Short A/B wave is not the active wave; refusing handover")
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise RuntimeError("Need at least 2GiB free for overlap")
    worker_script = ROOT / "scripts/short_cue_eval.py"
    identities = {pid: process_state(pid) for pid in [supervisor, *original["pids"]]}
    for pid, role in [(supervisor, "run"), *[(pid, "worker") for pid in original["pids"]]]:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if (identities[pid] is None or identities[pid]["state"] in ("Z", "T", "t")
                or role.encode() not in command or str(output).encode() not in command
                or not any(Path(os.fsdecode(part)).name == worker_script.name for part in command)):
            raise ValueError(f"Unexpected process identity: {pid}")
    for name, pid in zip(original["jobs"], original["pids"]):
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if identities[pid]["parent"] != supervisor or name.encode() not in command:
            raise ValueError(f"Worker is not owned by the old supervisor: {pid}")
    completed = {"base_short", "r8_short"}
    for name in completed:
        validate_report(plan, plan["jobs"][name], json.loads((output / f"{name}_report.json").read_text()))
    pending = ["control_control", "short_control"]
    for name in pending:
        if any((output / filename).exists() for filename in (name, f"{name}.log", f"{name}_report.json")):
            raise FileExistsError(f"Smoke job already started: {name}")
    record = output / "overlap_schedule.json"
    with record.open("x") as stream:
        json.dump({
            "time": time.time(), "previous_status": original, "supervisor": supervisor,
            "process_identities": identities, "max_workers": 3, "pending": pending,
            "plan_sha256": sha256(output / "plan.json"), "script_sha256": sha256(__file__),
            "note": "Scheduling only; old supervisor paused to prevent duplicate smoke jobs. "
                    "Live short workers and frozen evaluator/plan are unchanged. "
                    "Old supervisor retired after completion or fail-fast cleanup.",
        }, stream, indent=2)
    runtime_environment(output / "overlap_runtime")
    os.environ["PYTHONFAULTHANDLER"] = "1"
    adopted = dict(zip(original["jobs"], original["pids"]))
    children = {}
    taken_over = False
    try:
        signal_same_process(supervisor, identities[supervisor], signal.SIGSTOP)
        for attempt in range(20):
            if process_state(supervisor)["state"] == "T":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Old supervisor did not pause")
        if json.loads((output / "status.json").read_text()) != original:
            raise RuntimeError("Wave changed during handover")
        taken_over = True
        while adopted or children or pending:
            if process_state(supervisor)["state"] != "T":
                raise RuntimeError("Old supervisor resumed unexpectedly")
            for name, pid in list(adopted.items()):
                state = process_state(pid)
                if state is None or state["start"] != identities[pid]["start"]:
                    raise RuntimeError(f"Lost inherited worker identity: {name}")
                if state["state"] == "Z":
                    if state["exit_code"] != 0:
                        raise RuntimeError(f"Inherited worker failed: {name}, status={state['exit_code']}")
                    validate_report(plan, plan["jobs"][name],
                                    json.loads((output / f"{name}_report.json").read_text()))
                    completed.add(name)
                    del adopted[name]
            for name, process in list(children.items()):
                returncode = process.poll()
                if returncode is not None:
                    if returncode != 0:
                        raise RuntimeError(f"Smoke worker failed: {name}, code={returncode}")
                    validate_report(plan, plan["jobs"][name],
                                    json.loads((output / f"{name}_report.json").read_text()))
                    completed.add(name)
                    del children[name]
            if pending and len(adopted) + len(children) < 3:
                name = pending.pop(0)
                with (output / f"{name}.log").open("x") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(worker_script), "worker", "--output", str(output),
                         "--job", name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                children[name] = process
                identities[process.pid] = process_state(process.pid)
            active = {**adopted, **{name: process.pid for name, process in children.items()}}
            write_json(output / "status.json", {
                "stage": "evaluation", "jobs": list(active), "pids": list(active.values()),
                "completed_jobs": sorted(completed), "max_workers": 3, "time": time.time(),
                "supervisor_pid": os.getpid(), "schedule": str(record),
            })
            if active:
                time.sleep(5)
        check_inputs(output)
        for name, job in plan["jobs"].items():
            validate_report(plan, job, json.loads((output / f"{name}_report.json").read_text()))
        write_json(output / "status.json", {
            "stage": "pilot_audio_complete_not_accepted", "outputs": plan["total_outputs"],
            "time": time.time(), "production_changed": False, "native_review": "unverified",
            "schedule": str(record),
        })
        (output / "done").write_text("990 evaluations complete; analysis/acceptance pending.\n")
    except BaseException as error:
        if taken_over:
            write_json(output / "status.json", {
                "stage": "failed", "error": str(error), "time": time.time(), "schedule": str(record),
            })
        raise
    finally:
        if taken_over:
            remaining = [*adopted.values(), *[process.pid for process in children.values()]]
            for pid in remaining:
                signal_same_process(pid, identities[pid], signal.SIGTERM)
            deadline = time.monotonic() + 15
            while any((state := process_state(pid)) and state["state"] != "Z"
                      and state["start"] == identities[pid]["start"] for pid in remaining):
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            for pid in remaining:
                signal_same_process(pid, identities[pid], signal.SIGKILL)
            for process in children.values():
                process.wait()
            signal_same_process(supervisor, identities[supervisor], signal.SIGKILL)
        else:
            signal_same_process(supervisor, identities[supervisor], signal.SIGCONT)


def interrupted(signum, frame):
    raise InterruptedError(f"Supervisor received signal {signum}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervisor", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    run(args.output.resolve(), args.supervisor)
