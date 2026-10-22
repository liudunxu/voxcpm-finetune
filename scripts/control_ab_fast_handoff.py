"""Replace only the future evaluation stage; leave the live baseline worker untouched."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import control_ab_fast_eval as fast
import control_ab_overlap as overlap

frozen = fast.frozen


def handoff(work, shell, supervisor):
    frozen.check_inputs(work)
    profile_path = work / "fast_eval_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profile["metrics_exactly_equal"] or profile["source_sha256"] != frozen.sha256(fast.__file__):
        raise ValueError("Need verified cache equivalence")
    status = json.loads((work / "status.json").read_text(encoding="utf-8"))
    if status["stage"] != "baseline_overlap" or status["supervisor_pid"] != supervisor:
        raise ValueError("Baseline overlap is no longer active")
    for name in ("train.started", "evaluate.started", "fast_eval", "comparison.json"):
        if (work / name).exists():
            raise FileExistsError(work / name)
    identities = {pid: overlap.process_state(pid) for pid in (shell, supervisor)}
    commands = {pid: (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                for pid in identities}
    runner = Path(frozen.__file__).resolve()
    coordinator = Path(overlap.__file__).resolve()
    if (any(state is None or state["state"] in ("T", "t", "Z") for state in identities.values())
            or identities[supervisor]["parent"] != shell
            or str(coordinator).encode() not in commands[supervisor]
            or str(work).encode() not in commands[supervisor]
            or Path(commands[shell][0].decode()).name != "bash"
            or commands[shell][1] != b"-c"
            or any(str(path).encode() not in commands[shell][2]
                   for path in (runner, coordinator, work))):
        raise ValueError("Pipeline process identity mismatch")
    schedule = json.loads((work / "overlap_schedule.json").read_text(encoding="utf-8"))
    inputs = {**schedule["source_sha256"], **{
        str(path): frozen.sha256(path) for path in
        (Path(__file__).resolve(), Path(fast.__file__).resolve(), work / "plan.json", profile_path)}}
    if any(frozen.sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Scheduling inputs changed")
    with (work / "fast_handoff.json").open("x", encoding="utf-8") as stream:
        json.dump({"time": time.time(), "shell": shell, "supervisor": supervisor,
                   "identities": identities, "inputs_sha256": inputs,
                   "note": "Pause only the outer shell; baseline controller/workers remain alive. "
                           "After successful baseline, retire shell, run frozen training then dynamic evaluation."},
                  stream, ensure_ascii=False, indent=2)
    adopted = False
    try:
        overlap.signal_same_process(shell, identities[shell], signal.SIGSTOP)
        for attempt in range(20):
            state = overlap.process_state(shell)
            if state and state["start"] == identities[shell]["start"] and state["state"] == "T":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Outer shell did not pause")
        if (work / "train.started").exists() or (work / "evaluate.started").exists():
            raise RuntimeError("Pipeline advanced during handoff")
        adopted = True
        print(f"WAIT_BASELINE shell={shell} paused; supervisor={supervisor} untouched", flush=True)
        while True:
            parent, child = overlap.process_state(shell), overlap.process_state(supervisor)
            if (not parent or parent["start"] != identities[shell]["start"] or parent["state"] != "T"
                    or not child or child["start"] != identities[supervisor]["start"]
                    or child["parent"] != shell):
                raise RuntimeError("Lost pinned pipeline identity")
            if child["state"] == "Z":
                if child["exit_code"] != 0:
                    raise RuntimeError("Baseline controller failed")
                break
            time.sleep(10)
        if json.loads((work / "status.json").read_text(encoding="utf-8"))["stage"] != "baseline_complete":
            raise ValueError("Baseline completion marker missing")
        if any(frozen.sha256(path) != digest for path, digest in inputs.items()):
            raise ValueError("Handoff inputs changed")
        plan = frozen.check_inputs(work)
        cases = frozen._read_manifest(work / "cases.jsonl")
        for model in ("base", "r23"):
            for shard in range(2):
                overlap.validate_shard(work, plan, cases, f"{model}_shard{shard}")
            frozen.verify_report(json.loads((work / f"{model}_report.json").read_text(
                encoding="utf-8")), plan, cases, model)
        comparison = json.loads((work / "baseline_comparison.json").read_text(encoding="utf-8"))
        if any(frozen.sha256(path) != digest for path, digest in comparison["inputs_sha256"].items()):
            raise ValueError("Baseline report changed")
        overlap.signal_same_process(shell, identities[shell], signal.SIGKILL)
        print("BASELINE_VERIFIED; frozen training then accelerated evaluation", flush=True)
        subprocess.run([sys.executable, "-X", "utf8", "-u", str(runner), "train",
                        "--work", str(work)], cwd=frozen.ROOT, check=True)
        subprocess.run([sys.executable, "-X", "utf8", "-u", str(Path(fast.__file__).resolve()),
                        "run", "--work", str(work)], cwd=frozen.ROOT, check=True)
    except BaseException as error:
        frozen.write_json(work / "fast_handoff_failed.json", {
            "error": str(error), "time": time.time(), "baseline_not_interrupted": True})
        raise
    finally:
        overlap.signal_same_process(shell, identities[shell],
                                    signal.SIGKILL if adopted else signal.SIGCONT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--shell", type=int, required=True)
    parser.add_argument("--supervisor", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, overlap.interrupted)
    handoff(args.work.resolve(), args.shell, args.supervisor)
