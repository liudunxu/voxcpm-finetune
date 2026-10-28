"""Drain the frozen two-worker baseline at item boundaries, then use four workers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import fresh_baseline_eval as baseline
from short_cue_overlap import interrupted, process_state, signal_same_process

frozen = baseline.frozen


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assignments(cases, completed):
    workers = []
    for model in baseline.MODELS:
        pending = [identifier for identifier, _, _ in baseline.tasks(cases, model)
                   if identifier not in completed]
        for shard in range(2):
            workers.append({"model": model, "tasks": pending[shard::2]})
    return workers


def validate_item(path, plan, model, case, seed):
    report = read_json(path)
    settings = {
        "target": plan["targets"][model], "base": plan["base_path"],
        "cfg_value": plan["cfg_value"], "inference_timesteps": plan["inference_timesteps"],
        "retry_badcase": False, "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
        "lora_strength": 0 if model == "base" else 1,
    }
    if len(report["items"]) != 1 or any(report.get(key) != value for key, value in settings.items()):
        raise ValueError(f"Item settings changed: {path}")
    row = report["items"][0]
    if row["seed"] != seed or any(row.get(key) != value for key, value in case.items()):
        raise ValueError(f"Item conditions changed: {path}")
    if report["wav_sha256"] != {row["wav"]: frozen.sha256(row["wav"])}:
        raise ValueError(f"Item WAV changed: {path}")
    return row


def live_processes(work, supervisor, status):
    if status["stage"] != "evaluating" or len(status["pids"]) != 2:
        raise ValueError("Expected the active two-worker baseline")
    identities, workers = {}, {}
    if len(set([supervisor, *status["pids"]])) != 3:
        raise ValueError("Expected three distinct process identities")
    for pid in [supervisor, *status["pids"]]:
        identity = process_state(pid)
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        role = b"run" if pid == supervisor else b"worker"
        cwd = (Path("/proc") / str(pid) / "cwd").resolve()
        scripts = [cwd / os.fsdecode(part) for part in command
                   if Path(os.fsdecode(part)).name == Path(baseline.__file__).name]
        if (identity is None or identity["state"] in ("Z", "T", "t")
                or role not in command or str(work).encode() not in command
                or not any(path.resolve() == Path(baseline.__file__).resolve() for path in scripts)):
            raise ValueError(f"Unexpected process identity: {pid}")
        if pid != supervisor:
            if identity["parent"] != supervisor or b"--model" not in command:
                raise ValueError(f"Worker parent/model mismatch: {pid}")
            model = os.fsdecode(command[command.index(b"--model") + 1])
            if model not in baseline.MODELS or model in workers:
                raise ValueError(f"Unexpected worker model: {pid}")
            workers[model] = pid
        identities[pid] = identity
    return identities, workers


def drain(work, directory, supervisor):
    plan = baseline.checked(work)
    status = read_json(work / "status.json")
    identities, workers = live_processes(work, supervisor, status)
    cases = frozen._read_manifest(work / "cases.jsonl")
    baseline.save_new(directory / "handoff.json", {
        "time": time.time(), "previous_status": status, "supervisor": supervisor,
        "identities": identities, "workers": workers,
        "note": "Only the controller is paused. Exclusive barrier claims stop old workers "
                "before another generation, after their current item is saved. "
                "Expected barrier exceptions are not failed or retried generation attempts.",
    })
    barriers = {}
    try:
        signal_same_process(supervisor, identities[supervisor], signal.SIGSTOP)
        for attempt in range(30):
            state = process_state(supervisor)
            if state and state["start"] == identities[supervisor]["start"] and state["state"] == "T":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Original controller did not pause")
        for model in baseline.MODELS:
            for identifier, _, _ in baseline.tasks(cases, model):
                path = work / "claims" / f"{identifier}.json"
                try:
                    baseline.save_new(path, {
                        "pid": os.getpid(), "time": time.time(), "scheduler_barrier": True,
                        "handoff": str(directory / "handoff.json"),
                    })
                except FileExistsError:
                    continue
                barriers[model] = identifier
                baseline.save_new(directory / f"{model}_barrier.json", {
                    "identifier": identifier, "claim_sha256": frozen.sha256(path)})
                break
            else:
                raise RuntimeError(f"No remaining task to drain: {model}")
        deadline = time.monotonic() + 600
        while True:
            state = process_state(supervisor)
            if not state or state["start"] != identities[supervisor]["start"] or state["state"] != "T":
                raise RuntimeError("Original controller lost its pinned paused identity")
            states = {model: process_state(pid) for model, pid in workers.items()}
            if any(state is None or state["start"] != identities[workers[model]]["start"]
                   for model, state in states.items()):
                raise RuntimeError("Lost an original worker identity")
            if all(state["state"] == "Z" for state in states.values()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Workers did not finish their current items within 600 seconds")
            time.sleep(1)
        expected = {}
        for model in baseline.MODELS:
            suffix = f"RuntimeError: Task already claimed; no automatic retry: {barriers[model]}"
            if (os.waitstatus_to_exitcode(states[model]["exit_code"]) != 1
                    or not (work / f"{model}.log").read_text(encoding="utf-8").rstrip().endswith(suffix)):
                raise RuntimeError(f"Worker did not exit at its scheduling barrier: {model}")
            for identifier, case, seed in baseline.tasks(cases, model):
                if identifier == barriers[model]:
                    break
                expected[identifier] = (model, case, seed)
        completed = {path.stem for path in (work / "items").glob("*.json")}
        claims = {path.stem for path in (work / "claims").glob("*.json")}
        if completed != expected.keys() or claims != completed | set(barriers.values()):
            raise ValueError("Unresolved, missing or unexpected original tasks; do not resume")
        inputs = {str(work / "claims" / f"{identifier}.json"):
                  frozen.sha256(work / "claims" / f"{identifier}.json")
                  for identifier in barriers.values()}
        wavs = set()
        for identifier, (model, case, seed) in expected.items():
            path = work / "items" / f"{identifier}.json"
            row = validate_item(path, plan, model, case, seed)
            wavs.add(row["wav"])
            for artifact in (path, Path(row["wav"]), work / "claims" / path.name):
                inputs[str(artifact)] = frozen.sha256(artifact)
        if wavs != {str(path) for path in (work / "auditions").glob("*.wav")}:
            raise ValueError("Unmapped original WAV; do not regenerate it")
        baseline.save_new(directory / "preserved.json", {
            "completed": sorted(completed), "inputs_sha256": inputs, "barriers": barriers,
        })
        print(f"DRAINED: preserved {len(completed)} outputs; no worker was signalled", flush=True)
    finally:
        signal_same_process(supervisor, identities[supervisor], signal.SIGCONT)
    deadline = time.monotonic() + 60
    while True:
        state = process_state(supervisor)
        if not state or state["start"] != identities[supervisor]["start"] or state["state"] == "Z":
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Original controller did not retire")
        time.sleep(1)
    baseline.save_new(directory / "legacy_status.json", read_json(work / "status.json"))
    return completed


def prepare(work, directory, completed):
    plan = baseline.checked(work)
    cases = frozen._read_manifest(work / "cases.jsonl")
    workers = assignments(cases, completed)
    if any(not worker["tasks"] for worker in workers):
        raise ValueError("Not enough remaining tasks for four workers")
    inputs = dict(read_json(directory / "preserved.json")["inputs_sha256"])
    paths = [Path(__file__).resolve(), Path(__file__).resolve().with_suffix(".sh"),
             work / "plan.json", *directory.glob("*.json")]
    for index in range(4):
        child = directory / f"worker{index}"
        child.mkdir()
        for name in ("items", "claims"):
            (child / name).mkdir()
        for name in ("plan.json", "cases.jsonl"):
            shutil.copyfile(work / name, child / name)
            paths.append(child / name)
    inputs.update({str(path): frozen.sha256(path) for path in paths})
    baseline.save_new(directory / "plan.json", {
        "inputs_sha256": inputs, "max_workers": 4, "generation_budget": plan["generation_budget"],
        "preserved": sorted(completed), "workers": workers,
        "note": "User-authorized scheduling-only amendment to the immutable two-worker plan. "
                "Two disjoint lanes per model; no repeated successful or in-flight samples. "
                "Original worker/evaluator, cases, references, seeds, parameters and scoring reused. "
                "Concurrent gen_sec is not isolated model latency. No training or promotion.",
    })
    checked(work)


def checked(work):
    plan = baseline.checked(work)
    directory = work / "parallel4"
    schedule = frozen.check_inputs(directory)
    cases = frozen._read_manifest(work / "cases.jsonl")
    expected = {identifier for model in baseline.MODELS
                for identifier, _, _ in baseline.tasks(cases, model)}
    completed = set(schedule["preserved"])
    if (schedule["max_workers"] != 4 or schedule["generation_budget"] != plan["generation_budget"]
            or not completed <= expected or len(completed) != len(schedule["preserved"])
            or schedule["preserved"] != read_json(directory / "preserved.json")["completed"]
            or schedule["workers"] != assignments(cases, completed)
            or schedule["inputs_sha256"].get(str(Path(__file__).resolve())) != frozen.sha256(__file__)):
        raise ValueError("Four-worker scheduling scope changed")
    return schedule


def worker(work, index):
    schedule = checked(work)
    assignment = schedule["workers"][index]
    selected = set(assignment["tasks"])
    original_tasks = baseline.tasks
    print(f"ASSIGNED worker{index} model={assignment['model']} items={len(selected)}", flush=True)

    def selected_tasks(cases, model):
        return (task for task in original_tasks(cases, model) if task[0] in selected)

    baseline.tasks = selected_tasks
    try:
        baseline.worker(work / "parallel4" / f"worker{index}", assignment["model"])
    finally:
        baseline.tasks = original_tasks
    checked(work)


def collect(work):
    schedule = checked(work)
    plan = baseline.checked(work)
    cases = frozen._read_manifest(work / "cases.jsonl")
    expected = {identifier: (model, case, seed) for model in baseline.MODELS
                for identifier, case, seed in baseline.tasks(cases, model)}
    if {path.stem for path in (work / "items").glob("*.json")} != set(schedule["preserved"]):
        raise ValueError("Original item directory changed before collection")
    paths = []
    for index, assignment in enumerate(schedule["workers"]):
        child = work / "parallel4" / f"worker{index}"
        actual = {path.stem for path in (child / "items").glob("*.json")}
        claims = {path.stem for path in (child / "claims").glob("*.json")}
        if actual != set(assignment["tasks"]) or claims != actual:
            raise ValueError(f"Incomplete or unexpected worker tasks: {index}")
        wavs = set()
        for identifier in assignment["tasks"]:
            path = child / "items" / f"{identifier}.json"
            row = validate_item(path, plan, *expected[identifier])
            wavs.add(row["wav"])
            paths.append(path)
        if wavs != {str(path) for path in (child / "auditions").glob("*.wav")}:
            raise ValueError(f"Unmapped worker WAV: {index}")
    for path in paths:
        os.link(path, work / "items" / path.name)
    baseline.collect(work)


def run(work):
    schedule = checked(work)
    directory = work / "parallel4"
    if (directory / "started.json").exists():
        raise FileExistsError("Refusing to implicitly resume four-worker evaluation")
    if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is still busy; do not overlap the old and new workers")
    if not baseline.claim(directory, "started"):
        raise FileExistsError("Four-worker evaluation already started")
    processes = []

    def status(stage, **extra):
        frozen.write_json(work / "status.json", {
            "stage": stage, "time": time.time(), "max_workers": 4,
            "pids": [process.pid for process in processes if process.poll() is None],
            "completed_items": len(schedule["preserved"]) + sum(
                len(list((directory / f"worker{index}/items").glob("*.json"))) for index in range(4)),
            "expected_items": schedule["generation_budget"], "schedule": str(directory / "plan.json"),
            "training_started": False, "production_changed": False, **extra,
        })

    try:
        for index in range(4):
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).strip())
            if free < 12 * 1024 or shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError("Insufficient memory/disk for four workers; conditions unchanged")
            with (directory / f"worker{index}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "worker",
                     "--work", str(work), "--index", str(index)],
                    cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            progress = directory / f"worker{index}" / f"{schedule['workers'][index]['model']}_progress.json"
            while not progress.exists() and process.poll() is None:
                if any(child.poll() not in (None, 0) for child in processes):
                    raise RuntimeError("Worker failed during startup; no automatic retry")
                status("starting_four_workers")
                time.sleep(2)
            if process.poll() not in (None, 0):
                raise RuntimeError(f"Worker {index} failed during startup")
            print(f"STARTED worker{index} pid={process.pid}", flush=True)
        while True:
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Worker failed; preserve outputs, no automatic retry")
            status("evaluating")
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(10)
        collect(work)
        status("complete")
    except BaseException as error:
        status("failed", error=str(error))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def handoff(work, supervisor):
    baseline.checked(work)
    directory = work / "parallel4"
    directory.mkdir()
    try:
        completed = drain(work, directory, supervisor)
        prepare(work, directory, completed)
    except BaseException as error:
        frozen.write_json(directory / "failed.json", {"time": time.time(), "error": str(error)})
        raise
    run(work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("handoff", "run", "worker"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--supervisor", type=int)
    parser.add_argument("--index", type=int, choices=range(4))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    if args.action == "worker":
        if args.index is None:
            parser.error("worker requires --index")
        worker(args.work.resolve(), args.index)
    elif args.action == "handoff":
        if args.supervisor is None:
            parser.error("handoff requires --supervisor")
        handoff(args.work.resolve(), args.supervisor)
    else:
        run(args.work.resolve())
