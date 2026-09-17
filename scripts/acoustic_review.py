"""Freeze a small acoustic-only review sample; reuse generated A/B audio after evaluation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.data.registry import TARGET_LANGS
from voxft.paths import CHECKPOINT_DIR

from quality_probe import write_json
from ref_ab import sha256


def select_cases(cases):
    rng = random.Random(20260917)
    selected = []
    for lang in TARGET_LANGS:
        for ref_lang in ("zh", "en", "tl"):
            pool = sorted((case for case in cases if case["lang"] == lang
                           and case.get("ref_lang") == ref_lang and case.get("ref_audio")
                           and not evaluation._is_numeric(case)),
                          key=lambda case: str(case["case_id"]))
            if not pool:
                raise ValueError(f"No nonnumeric acoustic sample: {lang}/{ref_lang}")
            selected.append(rng.choice(pool)["case_id"])
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate sampled case IDs")
    return selected


def prepare(work):
    output = work / "acoustic_sample"
    output.mkdir(exist_ok=False)
    cases = _read_manifest(work / "cases.jsonl")
    write_json(output / "plan.json", {
        "case_ids": select_cases(cases), "seeds": [42, 43], "selection_seed": 20260917,
        "source_cases_sha256": sha256(work / "cases.jsonl"),
        "note": "15 cases x 2 seeds = 30 pairs, stratified by target/ref language. "
                "Selection uses frozen case metadata only, not A/B outcome scores. "
                "Acoustic abnormalities only; native ratings may remain empty. "
                "Not an online badcase-rate estimate, not full stability/native validation.",
    })


def collect(work):
    directory = work / "acoustic_sample"
    if (directory / "reports.json").exists():
        raise FileExistsError(directory / "reports.json")
    plan = json.loads((directory / "plan.json").read_text())
    if sha256(work / "cases.jsonl") != plan["source_cases_sha256"]:
        raise ValueError("Frozen cases changed")
    sources, _ = evaluation._review_reports(str(work / "off_report.json"), str(work / "on_report.json"))
    expected = {(case_id, seed) for case_id in plan["case_ids"] for seed in plan["seeds"]}
    reports = {}
    for arm in ("off", "on"):
        source = work / f"{arm}_report.json"
        report = deepcopy(sources["a" if arm == "off" else "b"])
        items = [row for row in report["items"] if (row["case_id"], row["seed"]) in expected]
        if len(items) != len(expected) or {(row["case_id"], row["seed"]) for row in items} != expected:
            raise ValueError(f"Incomplete acoustic sample: {arm}")
        for row in items:
            if not Path(row["wav"]).is_file():
                raise FileNotFoundError(row["wav"])
        report.update(items=items, label=f"r10_acoustic_{arm}", source_report=str(source),
                      source_report_sha256=sha256(source), acoustic_sample=plan)
        report.pop("cohorts", None)
        report.update(evaluation._report_metrics(items))
        reports[arm] = report
    saved = {arm: evaluation._save_report(report)["report_path"] for arm, report in reports.items()}
    evaluation.review_session(saved["off"], saved["on"])
    write_json(directory / "reports.json", saved)
    print(f"Acoustic-only review reports ready: {saved}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "ref_ab_20260917")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.wait:
        if args.action != "collect":
            parser.error("--wait requires collect")
        deadline = time.monotonic() + 4 * 3600
        while not (args.work / "done").exists():
            status = json.loads((args.work / "status.json").read_text())
            if status["stage"] == "failed" or time.monotonic() > deadline:
                raise RuntimeError(f"Controller failed or wait expired: {status}")
            print(f"Waiting for controller: {status['stage']}", flush=True)
            time.sleep(60)
    if args.action == "prepare":
        prepare(args.work)
    else:
        collect(args.work)
