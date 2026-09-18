"""Eval-only r11 recovery; preserve training evidence and run at most two GPU workers."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.paths import CHECKPOINT_DIR, ROOT, VOXCPM_REPO

from holdout_eval import check_inputs
from quality_probe import write_json
from ref_ab import runtime_environment, sha256
from short_cue_ab import ARMS, SEEDS


def prepare(work, output, workers):
    if workers not in (1, 2):
        raise ValueError("Only one or two workers have been budgeted")
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((work / "plan.json").read_text())
    for group in ("inputs_sha256", "audio_sha256", "frozen_files", "source_sha256"):
        for path, digest in plan[group].items():
            if sha256(path) != digest:
                raise ValueError(f"Frozen training input changed: {path}")
    if plan["evaluation_seeds"] != SEEDS:
        raise ValueError("Frozen evaluation seeds changed")
    r8 = json.loads((work / "prelaunch_verification.json").read_text())["r8_files_sha256"]
    if any(sha256(path) != digest for path, digest in r8.items()):
        raise ValueError("r8 checkpoint changed")
    targets = {"base": "base", "r8": str(CHECKPOINT_DIR / "lora_omni5_r8/latest")}
    for label, name in ARMS.items():
        target = CHECKPOINT_DIR / name / "latest"
        state = json.loads((target / "training_state.json").read_text())
        if (state["step"] != plan["steps"]
                or state != json.loads((work / f"train_{label}.done.json").read_text())):
            raise ValueError(f"Training incomplete: {label}")
        targets[label] = str(target)
    jobs = {
        f"{label}_{cohort}": {"target": targets[label], "cases": str(work / f"{cohort}_cases.jsonl")}
        for cohort in ("short", "control")
        for label in ("base", "r8", *ARMS)
        if cohort == "short" or label in ARMS
    }
    total = sum(len(_read_manifest(Path(job["cases"]))) * len(SEEDS) for job in jobs.values())
    if total != 990:
        raise ValueError(f"Expected frozen 990 outputs, got {total}")
    if shutil.disk_usage(work).free < 2 * 1024**3:
        raise RuntimeError("Need at least 2GiB free before evaluation")
    inputs = {path: digest for group in ("inputs_sha256", "frozen_files", "source_sha256")
              for path, digest in plan[group].items()}
    paths = {work / "plan.json", work / "prelaunch_verification.json"}
    paths |= {path for target in targets.values() if target != "base"
              for path in Path(target).iterdir() if path.is_file()}
    paths |= set((ROOT / "src").rglob("*.py")) | set((VOXCPM_REPO / "src").rglob("*.py"))
    paths |= {Path(__file__).resolve(), ROOT / "scripts/holdout_eval.py",
              ROOT / "scripts/quality_probe.py", ROOT / "uv.lock"}
    inputs.update({str(path): sha256(path) for path in paths})
    output.mkdir(parents=True)
    for filename in ("status.json", "eval_base_short.log"):
        shutil.copyfile(work / filename, output / f"original_{filename}")
    recovery = {
        "training_work": str(work), "base_path": plan["base_path"], "jobs": jobs,
        "evaluation_seeds": SEEDS, "cfg_value": 1.8, "inference_timesteps": 20,
        "retry_badcase": False, "workers": workers, "total_outputs": total,
        "inputs_sha256": inputs,
        "training_audio_verified": len(plan["audio_sha256"]),
        "note": "Eval only; frozen training plan, checkpoints and failed attempt are unchanged. "
                "Fresh per-job runtime caches avoid the reproduced old Numba cache crash. "
                "No native-language acceptance or production changes; concurrent timings are not latency benchmarks.",
    }
    write_json(output / "plan.json", recovery)
    return recovery


def validate_report(plan, job, report):
    expected = {(row["case_id"], seed): row
                for row in _read_manifest(Path(job["cases"])) for seed in plan["evaluation_seeds"]}
    actual = {(row["case_id"], row["seed"]): row for row in report["items"]}
    settings = {
        "target": job["target"], "base": plan["base_path"],
        "lora_strength": 0 if job["target"] == "base" else 1,
        "cfg_value": plan["cfg_value"], "inference_timesteps": plan["inference_timesteps"],
        "retry_badcase": False, "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
    }
    if (actual.keys() != expected.keys() or len(report["items"]) != len(expected)
            or any(report.get(key) != value for key, value in settings.items())):
        raise ValueError("Incomplete or inconsistent evaluation report")
    for key, case in expected.items():
        if (any(actual[key].get(field) != value for field, value in case.items())
                or not Path(actual[key].get("wav", "")).is_file()):
            raise ValueError(f"Changed case metadata or missing audio: {key}")


def worker(output, name):
    runtime_environment(output / name)
    plan = check_inputs(output)
    job = plan["jobs"][name]
    destination = output / f"{name}_report.json"
    if destination.exists():
        raise FileExistsError(destination)

    def progress(message):
        print(message, flush=True)
        if shutil.disk_usage(output).free < 1024**3:
            raise RuntimeError("Less than 1GiB free; stopping to protect existing artifacts")
        match = re.match(r"\[(\d+)/(\d+)\]", message)
        if match:
            write_json(output / f"{name}_progress.json", {
                "completed": int(match[1]), "total": int(match[2]),
                "message": message, "time": time.time(),
            })

    progress(f"Starting {name}")
    report = evaluation.evaluate(
        job["target"], "tl", _read_manifest(Path(job["cases"])),
        base=plan["base_path"], seeds=plan["evaluation_seeds"],
        cfg_value=plan["cfg_value"], inference_timesteps=plan["inference_timesteps"],
        progress=progress)
    check_inputs(output)
    validate_report(plan, job, report)
    write_json(destination, report)
    print(f"Saved {destination}", flush=True)


def run(work, output, workers):
    plan = prepare(work, output, workers)
    runtime_environment(output)
    os.environ["PYTHONFAULTHANDLER"] = "1"
    processes = []
    try:
        names = list(plan["jobs"])
        for offset in range(0, len(names), workers):
            wave = []
            for name in names[offset:offset + workers]:
                with (output / f"{name}.log").open("x") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                         "--output", str(output), "--job", name], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                wave.append(process)
                write_json(output / "status.json", {
                    "stage": "evaluation", "jobs": names[offset:offset + len(wave)],
                    "pids": [process.pid for process in wave], "time": time.time(),
                })
                if len(processes) == 1:
                    while not (output / f"{name}_progress.json").exists() and process.poll() is None:
                        time.sleep(2)
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f"First worker failed: {name}")
            while any(process.poll() is None for process in wave):
                if any(process.poll() not in (None, 0) for process in wave):
                    raise RuntimeError("Evaluation worker failed; see job logs")
                time.sleep(5)
            if any(process.returncode != 0 for process in wave):
                raise RuntimeError("Evaluation worker failed; see job logs")
            for name in names[offset:offset + workers]:
                validate_report(plan, plan["jobs"][name],
                                json.loads((output / f"{name}_report.json").read_text()))
        check_inputs(output)
        write_json(output / "status.json", {
            "stage": "pilot_audio_complete_not_accepted", "outputs": plan["total_outputs"],
            "time": time.time(), "production_changed": False, "native_review": "unverified",
        })
        (output / "done").write_text("990 evaluations complete; analysis/acceptance pending.\n")
    except BaseException as error:
        write_json(output / "status.json", {"stage": "failed", "error": str(error), "time": time.time()})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "short_cue_ab_20260918")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.action == "worker":
        if not args.job:
            parser.error("worker requires --job")
        worker(args.output, args.job)
    else:
        run(args.work, args.output, args.workers)
