"""Adopt live base shards and start r23 with at most three unchanged evaluation workers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import control_ab_run as frozen
from short_cue_overlap import interrupted, process_state, signal_same_process


def free_gpu_mib():
    return int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip())


def require_unstarted_r23(work):
    for name in ("r23_report.json", "baseline_comparison.json",
                 "r23_shard0.json", "r23_shard1.json", "r23_shard0.log", "r23_shard1.log"):
        if (work / name).exists():
            raise FileExistsError(work / name)


def validate_shard(work, plan, cases, name):
    model, shard = name.split("_shard")
    report = json.loads((work / f"{name}.json").read_text(encoding="utf-8"))
    frozen.verify_report(report, plan, cases[int(shard)::2], model)
    hashes = report["wav_sha256"]
    if (set(hashes) != {row["wav"] for row in report["items"]}
            or any(frozen.sha256(path) != digest for path, digest in hashes.items())):
        raise ValueError(f"Missing or changed frozen WAV: {name}")


def run(work, supervisor, base_pids):
    plan = frozen.check_inputs(work)
    runner = Path(frozen.__file__).resolve()
    if plan["inputs_sha256"].get(str(runner)) != frozen.sha256(runner):
        raise ValueError("Runner is not the frozen evaluator")
    original = json.loads((work / "status.json").read_text(encoding="utf-8"))
    if original["stage"] != "baseline" or (work / "base_report.json").exists():
        raise ValueError("Base wave is no longer active")
    require_unstarted_r23(work)
    if len(set([supervisor, *base_pids])) != 3:
        raise ValueError("Need one supervisor and two distinct base workers")
    identities = {pid: process_state(pid) for pid in [supervisor, *base_pids]}
    for pid, role in [(supervisor, "baseline"), *[(pid, "worker") for pid in base_pids]]:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        state = identities[pid]
        if (state is None or state["state"] in ("Z", "T", "t")
                or str(runner).encode() not in command or role.encode() not in command
                or str(work).encode() not in command):
            raise ValueError(f"Unexpected process identity: {pid}")
        if role == "worker":
            shard = base_pids.index(pid)
            if (state["parent"] != supervisor or b"--model" not in command
                    or command[command.index(b"--model") + 1] != b"base"
                    or b"--shard" not in command
                    or command[command.index(b"--shard") + 1] != str(shard).encode()):
                raise ValueError(f"Worker is not the expected live base shard: {pid}")
    if free_gpu_mib() < 16 * 1024 or shutil.disk_usage(work).free < 8 * 1024**3:
        raise RuntimeError("Need 16 GiB GPU headroom and 8 GiB disk before overlap")
    sources = {str(path): frozen.sha256(path) for path in Path(__file__).parent.glob("*.py")}
    record = work / "overlap_schedule.json"
    with record.open("x", encoding="utf-8") as stream:
        json.dump({
            "time": time.time(), "previous_status": original, "old_supervisor": supervisor,
            "process_identities": identities, "max_workers": 3,
            "plan_sha256": frozen.sha256(work / "plan.json"), "source_sha256": sources,
            "note": "User-authorized scheduling change only. Preserve live base shards; "
                    "start one r23 shard, queue the other until a worker slot is free. "
                    "Frozen cases, seeds, weights, evaluator and output budget unchanged. "
                    "Concurrent generation times are not isolated latency benchmarks.",
        }, stream, ensure_ascii=False, indent=2)
    cases = frozen._read_manifest(work / "cases.jsonl")
    adopted = dict(zip(("base_shard0", "base_shard1"), base_pids))
    pending, children, completed = ["r23_shard0", "r23_shard1"], {}, set()
    taken_over = False
    try:
        signal_same_process(supervisor, identities[supervisor], signal.SIGSTOP)
        for attempt in range(20):
            state = process_state(supervisor)
            if state and state["start"] == identities[supervisor]["start"] and state["state"] == "T":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Old supervisor did not pause")
        if json.loads((work / "status.json").read_text(encoding="utf-8")) != original:
            raise RuntimeError("Baseline changed during handover")
        require_unstarted_r23(work)
        taken_over = True
        while adopted or children or pending:
            state = process_state(supervisor)
            if not state or state["start"] != identities[supervisor]["start"] or state["state"] != "T":
                raise RuntimeError("Old supervisor no longer paused")
            for name, pid in list(adopted.items()):
                state = process_state(pid)
                if state is None or state["start"] != identities[pid]["start"]:
                    raise RuntimeError(f"Lost inherited worker identity: {name}")
                if state["state"] == "Z":
                    if state["exit_code"] != 0:
                        raise RuntimeError(f"Inherited worker failed: {name}")
                    validate_shard(work, plan, cases, name)
                    completed.add(name)
                    del adopted[name]
            for name, process in list(children.items()):
                code = process.poll()
                if code is not None:
                    if code != 0:
                        raise RuntimeError(f"r23 worker failed: {name}, code={code}")
                    validate_shard(work, plan, cases, name)
                    completed.add(name)
                    del children[name]
            if pending and len(adopted) + len(children) < 3 and free_gpu_mib() >= 16 * 1024:
                name = pending.pop(0)
                with (work / f"{name}.log").open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-X", "utf8", "-u", str(runner), "worker",
                         "--work", str(work), "--model", "r23", "--shard", name[-1]],
                        cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
                children[name] = process
                identities[process.pid] = process_state(process.pid)
                print(f"STARTED {name} pid={process.pid}", flush=True)
            active = {**adopted, **{name: process.pid for name, process in children.items()}}
            frozen.write_json(work / "status.json", {
                "stage": "baseline_overlap", "jobs": active, "pending": pending,
                "completed_jobs": sorted(completed), "max_workers": 3,
                "time": time.time(), "supervisor_pid": os.getpid(), "schedule": str(record),
            })
            if active or pending:
                time.sleep(5)
        frozen.check_inputs(work)
        if any(frozen.sha256(path) != digest for path, digest in sources.items()):
            raise ValueError("Handover source changed")
        frozen.evaluation.EVAL_DIR = work / "raw_reports"
        for name in ("base", "r23"):
            destination = work / f"{name}_report.json"
            if destination.exists():
                raise FileExistsError(destination)
            report = frozen.evaluation.merge_reports(
                [str(work / f"{name}_shard{shard}.json") for shard in range(2)])
            frozen.verify_report(report, plan, cases, name)
            frozen.write_json(destination, report)
        frozen.collect(work, ("base", "r23"), work / "baseline_comparison.json")
        frozen.write_json(work / "status.json", {
            "stage": "baseline_complete", "time": time.time(),
            "production_changed": False, "schedule": str(record),
        })
        print("BASELINE_COMPLETE; frozen training/evaluation pipeline may continue", flush=True)
    except BaseException as error:
        if taken_over:
            frozen.write_json(work / "status.json", {
                "stage": "baseline_failed", "error": str(error),
                "time": time.time(), "schedule": str(record),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--supervisor", type=int, required=True)
    parser.add_argument("--base-pids", type=int, nargs=2, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    run(args.work.resolve(), args.supervisor, args.base_pids)
