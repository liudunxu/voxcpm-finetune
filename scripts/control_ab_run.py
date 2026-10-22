"""Run frozen baseline, one-epoch control A/B, then matched evaluation. Never promote."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.paths import ROOT, VOXCPM_REPO
from voxft.train.launcher import preflight

from control_ab_gate import LANGS, SEEDS, WORK
from holdout_eval import check_inputs
from quality_probe import write_json
from ref_ab import runtime_environment, sha256
from refrob_stage2_analysis import boot_ci, paired_changes


def target_path(plan, name):
    if name in plan["targets"]:
        return plan["targets"][name]
    config = yaml.safe_load(Path(plan["configs"][name]).read_text(encoding="utf-8"))
    return str(Path(config["save_path"]) / "latest")


def verify_report(report, plan, cases, name):
    expected = {(row["case_id"], seed): row for row in cases for seed in SEEDS}
    actual = {(row["case_id"], row["seed"]): row for row in report["items"]}
    if len(actual) != len(report["items"]) or actual.keys() != expected.keys():
        raise ValueError("Incomplete or duplicated case/seed results")
    settings = {"target": target_path(plan, name), "base": plan["base_path"],
                "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
                "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
                "lora_strength": 0 if name == "base" else 1}
    if any(report.get(key) != value for key, value in settings.items()):
        raise ValueError("Evaluation/model settings differ")
    for key, case in expected.items():
        if any(actual[key].get(field) != value for field, value in case.items()):
            raise ValueError(f"Case changed: {key}")


def cluster_changes(candidate, baseline, cases, metric, lang):
    changes = paired_changes(independent_report(candidate), independent_report(baseline), metric, lang)
    sentences = {row["case_id"]: row["sentence_id"] for row in cases}
    groups = defaultdict(list)
    for case, difference in changes.items():
        groups[sentences[case]].append(difference)
    return {sentence: sum(values) / len(values) for sentence, values in groups.items()}


def control_effect(report, metric, lang):
    index = {(row["sentence_id"], row["condition"], row["seed"]): row
             for row in independent_report(report)["items"] if row["lang"] == lang}
    groups = defaultdict(list)
    for (sentence, condition, seed), row in index.items():
        if condition == "control":
            groups[sentence].append(float(row[metric]) - float(index[sentence, "plain", seed][metric]))
    return {sentence: sum(values) / len(values) for sentence, values in groups.items()}


def interval(values):
    estimate = boot_ci(values)
    return {"independent_texts": len(values), "mean": estimate[0],
            "ci95": list(estimate[1:])}


def independent_report(report):
    return {**report, "items": [row for row in report["items"]
                               if row.get("cohort", "independent_text") == "independent_text"]}


def collect(work, names, destination):
    plan = check_inputs(work)
    cases = _read_manifest(work / "cases.jsonl")
    reports = {}
    for name in names:
        report = json.loads((work / f"{name}_report.json").read_text(encoding="utf-8"))
        verify_report(report, plan, cases, name)
        reports[name] = report
    summary = {
        "reports": {name: evaluation._report_metrics(independent_report(report)["items"])
                    for name, report in reports.items()},
        "historical_regressions": {
            name: evaluation._report_metrics([row for row in report["items"]
                                              if row.get("cohort") == "historical_regression"])
            for name, report in reports.items()},
        "comparisons": {}, "control_minus_plain": {},
        "inputs_sha256": {str(work / f"{name}_report.json"): sha256(work / f"{name}_report.json")
                         for name in names},
        "note": "Bootstrap resamples independent texts, not control variants or seeds. "
                "Pointwise exploratory intervals, one training seed. "
                "No acoustic/native validation or automatic promotion.",
    }
    pairs = [(name, "base") for name in names if name != "base"]
    if "on" in names:
        pairs += [("on", "off")]
    for candidate, baseline in pairs:
        result = {}
        for lang in LANGS:
            result[lang] = {metric: interval(cluster_changes(
                reports[candidate], reports[baseline], cases, metric, lang))
                for metric in ("cer", "suspected_truncation", "over_read", "speaker_sim")}
        summary["comparisons"][f"{candidate}_minus_{baseline}"] = result
    for name, report in reports.items():
        summary["control_minus_plain"][name] = {
            lang: {metric: interval(control_effect(report, metric, lang))
                   for metric in ("cer", "speaker_sim")} for lang in LANGS}
    if "on" in names:
        summary["prefix_training_interaction"] = {
            lang: {metric: interval({
                sentence: value - control_effect(reports["off"], metric, lang)[sentence]
                for sentence, value in control_effect(reports["on"], metric, lang).items()})
                for metric in ("cer", "speaker_sim")} for lang in LANGS}
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, summary)
    return summary


def worker(work, name, shard):
    runtime_environment(work)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    plan = check_inputs(work)
    output = work / f"{name}_shard{shard}.json"
    if output.exists():
        raise FileExistsError(output)
    evaluation.EVAL_DIR = work / "raw_reports"
    evaluation.infer.CHECKPOINT_DIR = work

    def progress(message):
        print(message, flush=True)
        if shutil.disk_usage(work).free < 4 * 1024**3:
            raise RuntimeError("Less than 4 GiB free")

    report = evaluation.evaluate(
        target_path(plan, name), "th", _read_manifest(work / "cases.jsonl"),
        base=plan["base_path"], seeds=SEEDS, cfg_value=1.8,
        inference_timesteps=20, shard=(shard, 2), progress=progress)
    check_inputs(work)
    report["wav_sha256"] = {row["wav"]: sha256(row["wav"]) for row in report["items"]}
    write_json(output, report)


def evaluate_models(work, names):
    plan = check_inputs(work)
    for name in names:
        final = work / f"{name}_report.json"
        if final.exists():
            raise FileExistsError(final)
        jobs = []
        try:
            for shard in range(2):
                output = work / f"{name}_shard{shard}.json"
                if output.exists():
                    raise FileExistsError(output)
                with (work / f"{name}_shard{shard}.log").open("x", encoding="utf-8") as log:
                    jobs.append(subprocess.Popen(
                        [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                         "worker", "--work", str(work), "--model", name, "--shard", str(shard)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
            while any(job.poll() is None for job in jobs):
                if any(job.poll() not in (None, 0) for job in jobs):
                    raise RuntimeError(f"{name}: evaluation failed; inspect shard logs, no automatic retry")
                time.sleep(10)
            if any(job.returncode != 0 for job in jobs):
                raise RuntimeError(f"{name}: evaluation failed")
        finally:
            for job in jobs:
                if job.poll() is None:
                    job.terminate()
                    try:
                        job.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        job.kill()
                        job.wait()
        evaluation.EVAL_DIR = work / "raw_reports"
        report = evaluation.merge_reports([str(work / f"{name}_shard{shard}.json") for shard in range(2)])
        verify_report(report, plan, _read_manifest(work / "cases.jsonl"), name)
        write_json(final, report)
        print(f"EVAL_DONE {name}: {len(report['items'])}", flush=True)


def require_fresh_checkpoint(config):
    checkpoint = Path(config["save_path"])
    if checkpoint.exists():
        raise FileExistsError(f"Refusing implicit resume or overwrite: {checkpoint}")


def train(work):
    plan = check_inputs(work)
    comparison = json.loads((work / "baseline_comparison.json").read_text(encoding="utf-8"))
    if any(sha256(path) != digest for path, digest in comparison["inputs_sha256"].items()):
        raise ValueError("Baseline changed")
    for arm in ("off", "on"):
        config_path = Path(plan["configs"][arm])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require_fresh_checkpoint(config)
        issues = preflight(config_path, gpus=1)
        if any(not issue.startswith("警告") for issue in issues):
            raise ValueError(f"Preflight failed: {issues}")
        Path(config["save_path"]).rmdir()
    for arm in ("off", "on"):
        check_inputs(work)
        config_path = Path(plan["configs"][arm])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require_fresh_checkpoint(config)
        Path(config["tensorboard"]).mkdir(parents=True, exist_ok=False)
        with (work / f"train_{arm}.log").open("x", encoding="utf-8") as log, \
                (work / f"wandb_{arm}.log").open("x", encoding="utf-8") as bridge_log:
            bridge = subprocess.Popen(
                [sys.executable, "-X", "utf8", "-u", "-m", "voxft.train.tb_wandb_bridge",
                 config["tensorboard"], f"control_ab_20261022_{arm}_1ep"],
                cwd=ROOT, stdout=bridge_log, stderr=subprocess.STDOUT)
            try:
                print(f"TRAIN_STARTED {arm} steps={plan['num_iters']}", flush=True)
                subprocess.run(
                    [sys.executable, "-X", "utf8", "-u",
                     str(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py"),
                     "--config_path", str(config_path)], cwd=VOXCPM_REPO,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            finally:
                bridge.terminate()
                try:
                    bridge.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    bridge.kill()
                    bridge.wait()
        latest = Path(config["save_path"]) / "latest"
        state = json.loads((latest / "training_state.json").read_text(encoding="utf-8"))
        if state["step"] != plan["num_iters"]:
            raise ValueError(f"Training incomplete: {arm}")
        hashes = {str(path): sha256(path) for path in latest.iterdir() if path.is_file()}
        write_json(work / f"trained_{arm}.json", {"step": state["step"], "inputs_sha256": hashes})
        print(f"TRAIN_DONE {arm}", flush=True)
    check_inputs(work)


def run(work, action):
    runtime_environment(work)
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError("GPU has active compute jobs; no competing work started")
    with (work / f"{action}.started").open("x", encoding="utf-8") as marker:
        marker.write(str(time.time()))
    write_json(work / "status.json", {"stage": action, "time": time.time()})
    try:
        if action == "baseline":
            evaluate_models(work, ("base", "r23"))
            collect(work, ("base", "r23"), work / "baseline_comparison.json")
        elif action == "train":
            train(work)
        else:
            for arm in ("off", "on"):
                marker = json.loads((work / f"trained_{arm}.json").read_text(encoding="utf-8"))
                if any(sha256(path) != digest for path, digest in marker["inputs_sha256"].items()):
                    raise ValueError(f"Trained checkpoint changed: {arm}")
            evaluate_models(work, ("off", "on"))
            collect(work, ("base", "r23", "off", "on"), work / "comparison.json")
        write_json(work / "status.json", {"stage": f"{action}_complete", "time": time.time(),
                                        "production_changed": False})
    except BaseException as error:
        write_json(work / "status.json", {"stage": f"{action}_failed", "error": str(error),
                                        "time": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "train", "evaluate", "worker"))
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--model", choices=("base", "r23", "off", "on"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.work, args.model, args.shard)
    else:
        run(args.work, args.action)
