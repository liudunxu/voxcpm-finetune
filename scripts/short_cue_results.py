"""Summarize the frozen r11 pilot by source sentence; no new synthesis or scoring."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.paths import CHECKPOINT_DIR
from voxft.train.runlog import _duration_gates, _regressed

from quality_probe import write_json
from ref_ab import sha256
from ref_ab_diagnostics import METRICS, paired_summary
from short_cue_eval import validate_report


def sentence_summary(report_a, report_b):
    paired_cases = paired_summary(report_a, report_b)["by_case"]
    membership, texts = {}, {}
    for report in (report_a, report_b):
        for row in report["items"]:
            if not row.get("source_sentence_id") or evaluation._is_numeric(row):
                raise ValueError("Expected nonnumeric source sentence metadata")
            group = (row["lang"], str(row["source_sentence_id"]))
            if membership.setdefault(row["case_id"], group) != group:
                raise ValueError("Changed source sentence membership")
            if texts.setdefault(group, row["text"]) != row["text"]:
                raise ValueError("Changed text within source sentence")
    groups = defaultdict(list)
    for row in paired_cases:
        groups[membership[row["case_id"]]].append(row)
    sentences = []
    for (lang, sentence_id), rows in sorted(groups.items()):
        sentence = {"lang": lang, "source_sentence_id": sentence_id,
                    "case_ids": [row["case_id"] for row in rows]}
        for metric in METRICS:
            sentence[metric] = (
                np.mean([row[metric] for row in rows], axis=0).tolist()
                if all(row[metric] is not None for row in rows) else None)
        sentences.append(sentence)
    by_lang = {}
    for lang in sorted({row["lang"] for row in sentences}):
        selected = [row for row in sentences if row["lang"] == lang]
        by_lang[lang] = {}
        for metric in METRICS:
            values = np.array([row[metric] for row in selected if row[metric] is not None])
            delta = values[:, 1] - values[:, 0] if len(values) else np.array([])
            interval = None
            if len(delta) >= 2:
                rng = np.random.default_rng(20260917)
                samples = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
                interval = np.quantile(samples, [0.025, 0.975]).tolist()
            by_lang[lang][metric] = {
                "sentences": len(delta), "total_sentences": len(selected),
                "missing_sentence_ids": [row["source_sentence_id"] for row in selected
                                         if row[metric] is None],
                "mean_a": float(values[:, 0].mean()) if len(delta) else None,
                "mean_b": float(values[:, 1].mean()) if len(delta) else None,
                "delta_b_minus_a": float(delta.mean()) if len(delta) else None,
                "sentence_bootstrap_95": interval,
            }
    return {"by_lang": by_lang, "by_sentence": sentences}


def collect(work, holdout):
    destination = work / "paired_results.json"
    if destination.exists():
        raise FileExistsError(destination)
    if not (work / "done").is_file():
        raise ValueError("Evaluation not complete")
    plan = json.loads((work / "plan.json").read_text())
    expected_jobs = {f"{model}_short" for model in ("base", "r8", "control", "short")}
    expected_jobs |= {"control_control", "short_control"}
    if set(plan["jobs"]) != expected_jobs or plan["total_outputs"] != 990:
        raise ValueError("Unexpected pilot jobs")
    inputs = {str(path): sha256(path) for path in (work / "plan.json", work / "done")}
    for filename in ("short_cue_results.py", "ref_ab_diagnostics.py", "short_cue_eval.py"):
        source = Path(__file__).with_name(filename)
        inputs[str(source)] = sha256(source)
    reports = {"short": {}, "control": {}}
    for name, job in plan["jobs"].items():
        cases_path = Path(job["cases"])
        digest = sha256(cases_path)
        if plan["inputs_sha256"].get(str(cases_path)) != digest:
            raise ValueError(f"Frozen cases changed: {cases_path}")
        inputs[str(cases_path)] = digest
        path = work / f"{name}_report.json"
        report = json.loads(path.read_text())
        validate_report(plan, job, report)
        inputs[str(path)] = sha256(path)
        model, cohort = name.split("_")
        reports[cohort][model] = report
    control_path = plan["jobs"]["control_control"]["cases"]
    case_ids = {row["case_id"] for row in _read_manifest(Path(control_path))}
    for model in ("base", "r8"):
        path = holdout / f"{model}_report.json"
        report = json.loads(path.read_text())
        digest = sha256(path)
        if plan["inputs_sha256"].get(report["report_path"]) != digest:
            raise ValueError(f"Frozen baseline report changed: {path}")
        inputs[str(path)] = digest
        items = [row for row in report["items"] if row["case_id"] in case_ids]
        subset = {**report, "items": items, **evaluation._report_metrics(items)}
        job = {**plan["jobs"][f"{model}_short"], "cases": control_path}
        validate_report(plan, job, subset)
        reports["control"][model] = subset
    results = {}
    for cohort, models in reports.items():
        summaries = {
            model: {"outputs": len(report["items"]), **evaluation._report_metrics(report["items"])}
            for model, report in models.items()
        }
        comparisons = {}
        for baseline, candidate in (("control", "short"), ("r8", "control"), ("r8", "short"),
                                    ("base", "r8"), ("base", "control"), ("base", "short")):
            pair = [{**models[model], **summaries[model], "label": model}
                    for model in (baseline, candidate)]
            comparisons[f"{candidate}_minus_{baseline}"] = {
                **sentence_summary(*pair),
                "cer_gate": _regressed(pair)[candidate],
                "duration_gates": _duration_gates(pair)[candidate],
            }
        results[cohort] = {"models": summaries, "comparisons": comparisons}
    write_json(destination, {
        "inputs_sha256": inputs, "evaluation_seeds": plan["evaluation_seeds"], **results,
        "native_review": "unverified", "production_changed": False,
        "note": "All cases are nonnumeric. Paired seeds averaged per ref condition, then ref "
                "conditions averaged per (lang, source_sentence_id), then sentences weighted equally. "
                "5000 paired sentence bootstrap samples, RNG 20260917. Missing metrics exclude "
                "the whole sentence for that metric; CER/duration use joint completeness. "
                "Short TL: 14 independent texts, not 42 conditions or 210 independent trials. "
                "Control: three sentences per language, smoke only. Conditional on fixed refs, "
                "seeds and trained models; not training replication, population equivalence or "
                "native acceptance. No multiple-comparison correction. Gate thresholds unchanged. "
                "speaker_sim is output-vs-ref similarity, not pairwise cross-cue voice stability.",
    })
    print(f"Saved {destination}; original reports and production unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=(
        CHECKPOINT_DIR.parent / "short_cue_ab_20260918/eval_recovery_20260918"))
    parser.add_argument("--holdout", type=Path, default=CHECKPOINT_DIR.parent / "holdout_eval_20260917")
    args = parser.parse_args()
    collect(args.work, args.holdout)
