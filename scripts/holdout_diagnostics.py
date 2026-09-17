"""Export paired diagnostic and score-independent review samples without new inference."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.paths import CHECKPOINT_DIR

from acoustic_review import select_cases
from quality_probe import write_json
from ref_ab import runtime_environment, sha256


def select_samples(cases, paired_cases, indices):
    controls = [{"case_id": case_id, "seed": 42} for case_id in select_cases(cases)]
    selected = {}
    for metric, direction, limit in (("cer", 1, 10), ("speaker_sim", -1, 5)):
        ranked = sorted(
            (row for row in paired_cases if row[metric] is not None
             and direction * (row[metric][1] - row[metric][0]) > 0),
            key=lambda row: (-direction * (row[metric][1] - row[metric][0]), row["case_id"]),
        )
        for row in ranked[:limit]:
            case_id = row["case_id"]
            if case_id not in selected:
                seeds = sorted(seed for identifier, seed in indices["a"] if identifier == case_id)
                seed = max(seeds, key=lambda seed: direction * (
                    indices["b"][case_id, seed][metric] - indices["a"][case_id, seed][metric]))
                selected[case_id] = {"case_id": case_id, "seed": seed, "reasons": []}
            selected[case_id]["reasons"].append({
                "metric": metric, "mean_base": row[metric][0], "mean_r8": row[metric][1],
            })
    return {"diagnostic": list(selected.values()), "random_control": controls}


def collect(work):
    output = work / "diagnostic_sample"
    if output.exists():
        raise FileExistsError(output)
    comparison = json.loads((work / "comparison.json").read_text())
    plan = json.loads((work / "plan.json").read_text())
    cases_path = work / "cases.jsonl"
    if sha256(cases_path) != plan["inputs_sha256"][str(cases_path)]:
        raise ValueError("Frozen cases changed")
    sources = {model: work / f"{model}_report.json" for model in ("base", "r8")}
    for model, source in sources.items():
        if sha256(source) != comparison["inputs_sha256"][model]:
            raise ValueError(f"Collected report changed: {model}")
    reports, indices = evaluation._review_reports(str(sources["base"]), str(sources["r8"]))
    cases = _read_manifest(cases_path)
    expected = {(case["case_id"], seed) for case in cases for seed in plan["evaluation_seeds"]}
    if expected != indices["a"].keys():
        raise ValueError("Incomplete frozen case/seed coverage")
    samples = select_samples(cases, comparison["r8_minus_base"]["by_case"], indices)
    metadata = {
        "selection_seed": 20260917, "samples": samples,
        "source_cases_sha256": sha256(cases_path),
        "source_reports_sha256": comparison["inputs_sha256"],
        "comparison_sha256": sha256(work / "comparison.json"),
        "script_sha256": sha256(__file__),
        "overlap_case_ids": sorted({row["case_id"] for row in samples["diagnostic"]}
                                   & {row["case_id"] for row in samples["random_control"]}),
        "note": "Diagnostics: top 10 positive case-mean CER deltas plus top 5 ref-similarity "
                "drops; one worst paired seed for the first selecting metric. Not confirmed "
                "acoustic failures. Controls: one case per target/ref-language stratum, "
                "RNG 20260917 and seed 42; chosen after results exist but algorithm uses "
                "metadata only, never excludes diagnostic overlaps. Separate reports, no "
                "online badcase-rate estimate. Full five-seed evidence stays in originals. "
                "No native-quality validation. Samples used for tuning become development cases.",
    }
    exports = {}
    for group, selections in samples.items():
        if not selections:
            raise ValueError(f"Empty review sample: {group}")
        for who, model in (("a", "base"), ("b", "r8")):
            report = deepcopy(reports[who])
            items = [deepcopy(indices[who][row["case_id"], row["seed"]]) for row in selections]
            for item in items:
                if not Path(item["wav"]).is_file():
                    raise FileNotFoundError(item["wav"])
            report.update(items=items, label=f"holdout_{group}_{model}",
                          source_report=str(sources[model]),
                          source_report_sha256=comparison["inputs_sha256"][model],
                          diagnostic_sample=metadata)
            for field in ("cohorts", "human_review"):
                report.pop(field, None)
            report.update(evaluation._report_metrics(items))
            exports[group, model] = report
    output.mkdir()
    write_json(output / "plan.json", metadata)
    saved = {}
    for group in samples:
        saved[group] = {model: evaluation._save_report(exports[group, model])["report_path"]
                        for model in sources}
        evaluation.review_session(saved[group]["base"], saved[group]["r8"])
    write_json(output / "reports.json", saved)
    print(f"Saved review pairs: { {group: len(rows) for group, rows in samples.items()} }", flush=True)


def recheck_asr(work):
    from voxft.data.pipeline import _whisper_model

    output = work / "asr_recheck"
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((work / "diagnostic_sample/plan.json").read_text())
    reports = {}
    for model, expected in plan["source_reports_sha256"].items():
        source = work / f"{model}_report.json"
        if sha256(source) != expected:
            raise ValueError(f"Collected report changed: {model}")
        reports[model] = {(row["case_id"], row["seed"]): row
                          for row in json.loads(source.read_text())["items"]}
    cases = [row for row in plan["samples"]["diagnostic"]
             if any(reason["metric"] == "cer" for reason in row["reasons"])][:3]
    keys = sorted({(row["case_id"], seed) for row in cases for seed in (row["seed"], 49)})
    if not keys:
        raise ValueError("No CER diagnostic cases")
    audio_hashes = {row["wav"]: sha256(row["wav"])
                    for index in reports.values() for key in keys for row in [index[key]]}
    output.mkdir()
    result = {
        "source_reports_sha256": plan["source_reports_sha256"], "audio_sha256": audio_hashes,
        "script_sha256": sha256(__file__), "keys": keys, "asr_model": "large-v3",
        "note": "Same saved audio: repeat original VAD-on ASR, then change only vad_filter=False. "
                "Top 3 CER case regressions, selected worst seed and seed49 within-case control. "
                "Decoder sensitivity only; not native confirmation or new acceptance scores. "
                "Original reports and production remain unchanged.",
        "items": [], "complete": False,
    }
    write_json(output / "results.json", result)
    runtime_environment(work)
    whisper = _whisper_model("th", "large-v3")
    for model, index in reports.items():
        for key in keys:
            row = index[key]
            for vad_filter in (True, False):
                segments, info = whisper.transcribe(
                    row["wav"], language=None if row["lang"] in evaluation.AUTO_DETECT_LANGS
                    else row["lang"], vad_filter=vad_filter)
                segments = [{"start": segment.start, "end": segment.end, "text": segment.text}
                            for segment in segments]
                hyp = " ".join(segment["text"].strip() for segment in segments)
                result["items"].append({
                    "model": model, "case_id": key[0], "seed": key[1], "vad_filter": vad_filter,
                    "wav": row["wav"], "original_cer": row["cer"], "original_hyp": row["hyp"],
                    "hyp": hyp, "segments": segments,
                    "duration_after_vad": getattr(info, "duration_after_vad", None),
                    "cer": evaluation._error_rate(evaluation._norm(hyp), evaluation._norm(row["text"])),
                    "suspected_truncation": evaluation._is_truncated(hyp, row["text"]),
                    "over_read": evaluation._over_read(hyp, row["text"]),
                })
                write_json(output / "results.json", result)
                print(f"ASR recheck {len(result['items'])}/{len(keys) * len(reports) * 2}", flush=True)
    result["complete"] = True
    write_json(output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "holdout_eval_20260917")
    parser.add_argument("--recheck-asr", action="store_true")
    args = parser.parse_args()
    (recheck_asr if args.recheck_asr else collect)(args.work)
