"""r10 CPU sidecar: candidate exposure audit and case-paired diagnostics; no GPU work."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, ROOT

from ref_ab import memberships, sha256, verify_comparison, without_ref
from quality_probe import write_json

METRICS = ("cer", "wer", "suspected_truncation", "over_read", "audio_sec",
           "speaker_sim", "speech_ratio", "tail_silence_sec")


def expand_references(rows, index):
    expanded = [without_ref(row) for row in rows]
    for row in rows:
        if row.get("ref_audio"):
            key = str(Path(row["ref_audio"]).resolve())
            if key not in index:
                raise ValueError(f"Reference transcript/provenance unavailable: {key}")
            reference = index[key]
            if (row.get("ref_speaker") != reference.get("speaker")
                    or row.get("ref_origin_audio") != reference.get("origin_audio")):
                raise ValueError(f"Reference provenance differs: {key}")
            expanded.append(without_ref(reference))
    return expanded


def audit_candidates(candidates, exposures, evaluated_texts):
    indices = {name: (memberships(rows), {evaluation._norm(row["text"]) for row in rows})
               for name, rows in exposures.items()}
    results = []
    for row in candidates:
        target = without_ref(row)
        keys = memberships([target])
        text = evaluation._norm(row["text"])
        overlaps = {}
        for name, (used_keys, used_texts) in indices.items():
            reasons = {key[0] for key in keys & used_keys}
            if text in used_texts:
                reasons.add("text")
            if reasons:
                overlaps[name] = sorted(reasons)
        results.append({
            **target, "overlaps": overlaps, "evaluated_text": text in evaluated_texts,
            "training_clean": not any(name.endswith("/train") for name in overlaps),
            "training_validation_clean": not overlaps and text not in evaluated_texts,
            "identity_status": ("verified_id_not_matched" if row.get("speaker_verified") is True
                                and not any("speaker" in reasons for reasons in overlaps.values())
                                else "matched" if any("speaker" in reasons for reasons in overlaps.values())
                                else "unknown"),
        })
    return results


def audit(work):
    import soundfile as sf

    output = work / "cpu_audit"
    output.mkdir(exist_ok=False)
    recipe = json.loads((DATA_PROCESSED / "joint_omni8/mix.json").read_text())
    sources = sorted({name for name, _ in recipe["parts"]} | {"cv22_th", "cv22_id"})
    inputs, index, candidates = {}, {}, []
    for source in sources:
        for split in ("train", "val"):
            path = DATA_PROCESSED / source / f"{split}.jsonl"
            rows = _read_manifest(path)
            inputs[str(path)] = sha256(path)
            for row in rows:
                key = str(Path(row["audio"]).resolve())
                if key in index:
                    raise ValueError(f"Duplicate source audio path: {key}")
                index[key] = row
            if split == "val":
                candidates.extend(rows)
    exposures = {}
    for name in ("joint_omni8", "r10_ref_off", "r10_ref_on"):
        for split in ("train", "val"):
            path = DATA_PROCESSED / name / f"{split}.jsonl"
            inputs[str(path)] = sha256(path)
            exposures[f"{name}/{split}"] = expand_references(_read_manifest(path), index)
    case_paths = sorted((ROOT / "eval_cases").glob("*.jsonl")) + [work / "cases.jsonl"]
    evaluated_texts = set()
    for path in case_paths:
        inputs[str(path)] = sha256(path)
        evaluated_texts.update(evaluation._norm(row["text"]) for row in _read_manifest(path))
    rows = audit_candidates(candidates, exposures, evaluated_texts)
    for row in rows:
        try:
            info = sf.info(row["audio"])
            row["audio_header_valid"] = (
                info.samplerate == 16000 and info.channels == 1 and 3 <= info.duration <= 30
                and abs(info.duration - row["duration"]) <= 0.03)
        except (OSError, RuntimeError):
            row["audio_header_valid"] = False
    _write_jsonl(rows, output / "candidates.jsonl")
    summary = {}
    for lang in sorted({row["lang"] for row in rows}):
        group = [row for row in rows if row["lang"] == lang]
        clean = [row for row in group if row["training_clean"] and row["audio_header_valid"]]
        untouched = [row for row in clean if row["training_validation_clean"]]
        unique = {evaluation._norm(row["text"]) for row in untouched}
        summary[lang] = {
            "source_val_rows": len(group), "training_clean_rows": len(clean),
            "training_clean_unique_texts": len({evaluation._norm(row["text"]) for row in clean}),
            "training_validation_clean_rows": len(untouched),
            "training_validation_clean_unique_texts": len(unique),
            "verified_identity_clean_rows": sum(row["identity_status"] == "verified_id_not_matched"
                                                for row in untouched),
            "shortfall_to_30_unique_texts": max(0, 30 - len(unique)),
            "invalid_audio_headers": sum(not row["audio_header_valid"] for row in group),
            "overlap_rows": dict(Counter(name for row in group for name in row["overlaps"])),
            "overlap_reasons": dict(Counter(reason for row in group
                                             for reason in {reason for reasons in row["overlaps"].values()
                                                            for reason in reasons})),
        }
    write_json(output / "summary.json", {
        "by_lang": summary, "inputs_sha256": inputs,
        "note": "Scope: existing source validation rows against r8/r10 target AND reference inputs, "
                "including loss-validation exposure and existing local evaluation texts. "
                "Not a complete historical/pretraining audit. Unknown speakers remain unknown. "
                "Metadata paths/origins/IDs and audio headers only: no waveform duplicate detection "
                "or transcript/native-quality verification. Candidates are NOT new evaluation cases; "
                "current frozen benchmark unchanged. Counts across overlap reasons are not additive.",
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def paired_summary(report_a, report_b):
    for field in ("cfg_value", "inference_timesteps", "retry_badcase",
                  "asr_model", "asr_auto_detect_langs"):
        if field not in report_a or field not in report_b or report_a[field] != report_b[field]:
            raise ValueError(f"Missing or unequal report condition: {field}")
    indices = []
    for report in (report_a, report_b):
        index = {(row["case_id"], row["seed"]): row for row in report["items"]}
        if len(index) != len(report["items"]) or not index:
            raise ValueError("Empty report or duplicate case/seed")
        indices.append(index)
    if indices[0].keys() != indices[1].keys():
        raise ValueError("Incomplete paired case/seed sets")
    seeds = sorted({seed for _, seed in indices[0]})
    cases = defaultdict(list)
    fields = ("lang", "text", "ref_audio", "ref_lang", "control", "speaker", "cohort")
    for key in sorted(indices[0], key=lambda pair: (str(pair[0]), pair[1])):
        row_a, row_b = (index[key] for index in indices)
        if (any((row_a.get(field) or "") != (row_b.get(field) or "") for field in fields)
                or evaluation._is_numeric(row_a) != evaluation._is_numeric(row_b)):
            raise ValueError(f"Unequal case conditions: {key}")
        cases[key[0]].append((row_a, row_b))
    case_rows = []
    for case_id, pairs in cases.items():
        first = pairs[0][0]
        if len(pairs) != len(seeds) or any(
                any((row.get(field) or "") != (first.get(field) or "") for field in fields)
                or evaluation._is_numeric(row) != evaluation._is_numeric(first)
                for row, _ in pairs):
            raise ValueError(f"Incomplete seeds or changing case conditions: {case_id}")
        result = {key: first.get(key) for key in ("case_id", "lang", "cohort", "ref_audio")}
        result["numeric"] = evaluation._is_numeric(first)
        for metric in METRICS:
            required = ("cer", "audio_sec") if metric in ("cer", "audio_sec") else (metric,)
            valid = all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field])
                        for pair in pairs for row in pair for field in required)
            result[metric] = ([float(np.mean([pair[position][metric] for pair in pairs]))
                               for position in (0, 1)] if valid else None)
        case_rows.append(result)
    summaries = {}
    for cohort in ["all"] + sorted({row["cohort"] for row in case_rows if row["cohort"]}):
        summaries[cohort] = {}
        for lang in sorted({row["lang"] for row in case_rows}):
            group = [row for row in case_rows if row["lang"] == lang
                     and (cohort == "all" or row["cohort"] == cohort)]
            summaries[cohort][lang] = {}
            for subset in ("all", "non_numeric"):
                selected = [row for row in group if subset == "all" or not row["numeric"]]
                metrics = {}
                for metric in METRICS:
                    values = np.array([row[metric] for row in selected if row[metric] is not None])
                    delta = values[:, 1] - values[:, 0] if len(values) else np.array([])
                    interval = None
                    if len(delta) >= 2:
                        rng = np.random.default_rng(20260917)
                        resampled = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(axis=1)
                        interval = np.quantile(resampled, [0.025, 0.975]).tolist()
                    metrics[metric] = {
                        "cases": len(delta), "total_cases": len(selected),
                        "missing_case_ids": [row["case_id"] for row in selected if row[metric] is None],
                        "mean_a": float(values[:, 0].mean()) if len(delta) else None,
                        "mean_b": float(values[:, 1].mean()) if len(delta) else None,
                        "delta_b_minus_a": float(delta.mean()) if len(delta) else None,
                        "case_bootstrap_95": interval,
                        "positive_cases": int(np.sum(delta > 0)),
                        "negative_cases": int(np.sum(delta < 0)),
                        "tied_cases": int(np.sum(delta == 0)),
                    }
                summaries[cohort][lang][subset] = metrics
    return {
        "seeds": seeds, "by_cohort": summaries, "by_case": case_rows,
        "note": "B minus A. Average paired seeds within each case, then weight cases equally. "
                "5000 paired case bootstrap resamples, RNG 20260917, percentile 95% intervals. "
                "Missing/nonfinite measurements exclude the whole case for that metric; "
                "CER and duration share joint completeness. Fewer than two cases: interval null. "
                "Conditional diagnostics for these fixed refs/seeds/models, NOT training-seed "
                "replication, speaker-population uncertainty, equivalence or native quality. "
                "Shared refs/text dependencies and multiple comparisons are not corrected; "
                "zero-width intervals do not establish zero risk. No new promotion threshold.",
    }


def compare(work):
    cases = _read_manifest(work / "cases.jsonl")
    reports = {name: json.loads((work / f"{name}_report.json").read_text())
               for name in ("base", "r8", "off", "on")}
    verify_comparison(reports, cases)
    output = work / "paired_diagnostics.json"
    if output.exists():
        raise FileExistsError(output)
    results = {f"{candidate}_minus_{baseline}": paired_summary(reports[baseline], reports[candidate])
               for baseline, candidate in (("base", "r8"), ("off", "on"),
                                           ("r8", "off"), ("r8", "on"))}
    results["inputs_sha256"] = {name: sha256(work / f"{name}_report.json") for name in reports}
    write_json(output, results)
    print(f"Saved {output}; production unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "compare"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "ref_ab_20260917")
    parser.add_argument("--wait", action="store_true", help="Compare only: wait up to 4h for controller done")
    args = parser.parse_args()
    if args.wait:
        if args.action != "compare":
            parser.error("--wait requires compare")
        deadline = time.monotonic() + 4 * 3600
        while not (args.work / "done").exists():
            status = json.loads((args.work / "status.json").read_text())
            if status["stage"] == "failed":
                raise RuntimeError(f"Controller failed: {status}")
            if time.monotonic() > deadline:
                raise TimeoutError("Controller has not finished within 4h")
            print(f"Waiting for controller: {status['stage']}", flush=True)
            time.sleep(60)
    if args.action == "audit":
        audit(args.work)
    else:
        compare(args.work)
