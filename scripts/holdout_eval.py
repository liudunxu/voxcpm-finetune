"""Frozen new-text base/r8 evaluation; two GPU workers, no training or promotion."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.data.registry import TARGET_LANGS
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, DATA_RAW, ROOT, VOXCPM_REPO

from quality_probe import write_json
from ref_ab import runtime_environment, sha256
from ref_ab_diagnostics import paired_summary


def check_inputs(work):
    plan = json.loads((work / "plan.json").read_text())
    for path, expected in plan["inputs_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return plan


def prepare(work, texts):
    source = json.loads((texts / "plan.json").read_text())
    cases = _read_manifest(texts / "cases.jsonl")
    if (sha256(texts / "cases.jsonl") != source["cases_sha256"]
            or Counter(row["lang"] for row in cases) != Counter({lang: 30 for lang in TARGET_LANGS})
            or len({row["case_id"] for row in cases}) != 150
            or len({evaluation._norm(row["text"]) for row in cases}) != 150
            or any(evaluation._is_numeric(row) for row in cases)
            or source["evaluation_seeds"] != [42, 43, 44, 45, 49]
            or source["cfg_value"] != 1.8 or source["inference_timesteps"] != 20
            or source["retry_badcase"] is not False):
        raise ValueError("Frozen holdout cases/settings differ")
    paths = set(DATA_PROCESSED.glob("*/train.jsonl")) | set(DATA_PROCESSED.glob("*/val.jsonl"))
    paths |= set(DATA_RAW.glob("*/manifest.jsonl"))
    paths |= set((CHECKPOINT_DIR.parent / "training_pool_20260917").glob("*/targets.jsonl"))
    paths |= {Path(row["ref_audio"]) for row in cases}
    for path in paths:
        if source["inputs_sha256"].get(str(path)) != sha256(path):
            raise ValueError(f"Holdout exclusion/reference audit needs refresh: {path}")
    old_plan = json.loads((CHECKPOINT_DIR.parent / "ref_ab_20260917/plan.json").read_text())
    base = Path(old_plan["base_path"])
    r8 = CHECKPOINT_DIR / "lora_omni5_r8/latest"
    for filename in ("lora_config.json", "lora_weights.safetensors"):
        if not (r8 / filename).is_file():
            raise FileNotFoundError(r8 / filename)
    if not base.is_dir() or not any(base.rglob("*.safetensors")):
        raise FileNotFoundError(base)
    paths |= {path for directory in (base, r8) for path in directory.rglob("*") if path.is_file()}
    paths |= set((ROOT / "src").rglob("*.py")) | set((ROOT / "scripts").glob("*.py"))
    paths |= set((VOXCPM_REPO / "src").rglob("*.py"))
    paths |= {ROOT / "uv.lock", ROOT / "pyproject.toml", texts / "cases.jsonl", texts / "plan.json"}
    if shutil.disk_usage(work.parent).free < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB free; no evaluation started")
    work.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(texts / "cases.jsonl", work / "cases.jsonl")
    paths.add(work / "cases.jsonl")
    write_json(work / "plan.json", {
        "base_path": str(base), "targets": {"base": "base", "r8": str(r8)},
        "evaluation_seeds": source["evaluation_seeds"], "cfg_value": 1.8,
        "inference_timesteps": 20, "retry_badcase": False,
        "cases": 150, "outputs_per_model": 750, "workers": 2,
        "inputs_sha256": {str(path): sha256(path) for path in sorted(paths)},
        "note": "New-text diagnostic only. No training, automatic retries or production changes. "
                "No claim of pretraining novelty, independent speakers or native-language quality. "
                "Two workers reuse the existing concurrency budget; not a throughput optimum.",
    })
    write_json(work / "status.json", {"stage": "prepared", "time": time.time()})
    print(f"Prepared 150 cases x 5 seeds x base/r8: {work}", flush=True)


def worker(work, model):
    runtime_environment(work)
    plan = check_inputs(work)
    output = work / f"{model}_report.json"
    if output.exists():
        raise FileExistsError(output)

    def progress(message):
        print(message, flush=True)
        if shutil.disk_usage(work).free < 2 * 1024**3:
            raise RuntimeError("Less than 2 GiB free; stopping to protect existing artifacts")

    progress(f"Starting {model}")
    report = evaluation.evaluate(
        plan["targets"][model], "th", _read_manifest(work / "cases.jsonl"),
        base=plan["base_path"], seeds=plan["evaluation_seeds"],
        cfg_value=plan["cfg_value"], inference_timesteps=plan["inference_timesteps"],
        progress=progress)
    check_inputs(work)
    write_json(output, report)
    print(f"Saved {output}", flush=True)


def collect(work):
    from voxft.train import runlog

    plan = check_inputs(work)
    reports = {model: json.loads((work / f"{model}_report.json").read_text()) for model in plan["targets"]}
    expected = {(row["case_id"], seed): row
                for row in _read_manifest(work / "cases.jsonl") for seed in plan["evaluation_seeds"]}
    for model, report in reports.items():
        actual = {(row["case_id"], row["seed"]): row for row in report["items"]}
        if (actual.keys() != expected.keys() or len(report["items"]) != len(expected)
                or report["target"] != plan["targets"][model] or report["base"] != plan["base_path"]
                or report["lora_strength"] != (0 if model == "base" else 1)
                or report["cfg_value"] != 1.8 or report["inference_timesteps"] != 20
                or report["retry_badcase"] is not False or report["asr_model"] != "large-v3"
                or report["asr_auto_detect_langs"] != ["tl"]):
            raise ValueError(f"Incomplete or inconsistent report: {model}")
        for key, case in expected.items():
            if any(actual[key].get(field) != value for field, value in case.items()):
                raise ValueError(f"Case metadata changed: {model} {key}")
    ordered = [reports["base"], reports["r8"]]
    write_json(work / "comparison.json", {
        "reports": {model: evaluation._report_metrics(report["items"]) for model, report in reports.items()},
        "cer": runlog._regressed(ordered), "duration": runlog._duration_gates(ordered),
        "r8_minus_base": paired_summary(*ordered),
        "inputs_sha256": {model: sha256(work / f"{model}_report.json") for model in reports},
        "production_changed": False, "native_review": "unverified",
    })


def run(work):
    runtime_environment(work)
    check_inputs(work)
    jobs = []
    try:
        for model in ("base", "r8"):
            with (work / f"{model}.log").open("x") as log:
                jobs.append(subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                     "--work", str(work), "--model", model], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT))
        write_json(work / "status.json", {"stage": "evaluation", "pids": [job.pid for job in jobs],
                                        "time": time.time()})
        while any(job.poll() is None for job in jobs):
            if any(job.poll() not in (None, 0) for job in jobs):
                raise RuntimeError("Evaluation worker failed; see model logs")
            time.sleep(10)
        if any(job.returncode != 0 for job in jobs):
            raise RuntimeError("Evaluation worker failed; see model logs")
        collect(work)
        write_json(work / "status.json", {"stage": "voice_stability", "time": time.time()})
        with (work / "voice_stability.log").open("x") as log:
            subprocess.run(
                [sys.executable, "-u", str(ROOT / "scripts/voice_stability.py"),
                 str(work / "base_report.json"), str(work / "r8_report.json")],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(work / "status.json", {"stage": "complete", "time": time.time(),
                                        "production_changed": False, "native_review": "unverified"})
        (work / "done").write_text(str(time.time()))
    except BaseException as error:
        write_json(work / "status.json", {"stage": "failed", "error": str(error), "time": time.time()})
        raise
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
                try:
                    job.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "collect"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "holdout_eval_20260917")
    parser.add_argument("--texts", type=Path, default=CHECKPOINT_DIR.parent / "holdout_texts_20260917")
    parser.add_argument("--model", choices=("base", "r8"), default="base")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.work, args.texts)
    elif args.action == "worker":
        worker(args.work, args.model)
    else:
        {"run": run, "collect": collect}[args.action](args.work)
