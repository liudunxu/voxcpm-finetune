"""Audit existing training coverage and spot-check transcripts; never start training."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from statistics import median

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest, _whisper_model, _write_jsonl
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, DATA_RAW, ROOT

from quality_probe import write_json
from ref_ab import memberships, runtime_environment, sha256, without_ref
from training_pool import inspect_audio, metadata_rejection


def describe(rows):
    durations = [float(row["duration"]) for row in rows]
    if any(not math.isfinite(duration) or not 3 <= duration <= 30 for duration in durations):
        raise ValueError("Training duration outside 3-30 seconds")
    seconds = sum(durations)
    short_seconds = sum(duration for duration in durations if duration <= 8)
    return {
        "records": len(rows), "hours": seconds / 3600,
        "unique_origins": len({row["origin_audio"] for row in rows}),
        "unique_texts": len({evaluation._norm(row["text"]) for row in rows}),
        "max_origin_exposure": max(Counter(row["origin_audio"] for row in rows).values(), default=0),
        "duration_p50": median(durations) if durations else None,
        "normalized_chars_p50": median(len(evaluation._norm(row["text"])) for row in rows) if rows else None,
        "short_records": sum(duration <= 8 for duration in durations),
        "short_hours": short_seconds / 3600,
        "short_duration_share": short_seconds / seconds if seconds else None,
        "duration_bins": dict(Counter("3-5s" if duration <= 5 else "5-8s" if duration <= 8
                                      else "8-15s" if duration <= 15 else "15-30s"
                                      for duration in durations)),
    }


def provenance(rows, raw):
    counts, issues = Counter(), []
    for row in rows:
        original = raw.get((row["source_id"], row["origin_audio"]))
        reason = ("missing_raw_origin" if original is None else
                  "raw_language_mismatch" if original.get("lang") != row.get("lang") else
                  "exact_text_match" if original.get("text") == row["text"] else
                  "normalization_only_difference" if evaluation._norm(original["text"]) ==
                  evaluation._norm(row["text"]) else "raw_text_mismatch")
        counts[reason] += 1
        if reason != "exact_text_match":
            issues.append({"audio": row["audio"], "origin_audio": row["origin_audio"], "reason": reason})
    return {"counts": dict(counts), "issues": issues,
            "note": "Raw-manifest linkage only; not proof that speech matches the transcript."}


def review_sample(rows):
    rng = random.Random(20260918)
    selected = []
    for lang in ("tl", "ms"):
        for low, high in ((3, 5), (5, 8), (8, 30)):
            candidates = sorted((row for row in rows if row["lang"] == lang
                                 and (low <= row["duration"] if low == 3 else low < row["duration"])
                                 and row["duration"] <= high and not evaluation._is_numeric(row)),
                                key=lambda row: row["audio"])
            if len(candidates) < 5:
                raise ValueError(f"Insufficient non-digit review inventory: {lang}/{low}-{high}")
            selected.extend({**row, "review_stratum": f"{lang}/{low}-{high}s"}
                            for row in rng.sample(candidates, 5))
    return selected


def audit(work, pool, examples):
    work.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(path):
        inputs[str(path)] = sha256(path)
        return _read_manifest(path)

    previous = json.loads((pool / "summary.json").read_text())
    inputs[str(pool / "summary.json")] = sha256(pool / "summary.json")
    baseline = read(DATA_PROCESSED / "joint_omni8/train.jsonl")
    validations = read(DATA_PROCESSED / "joint_omni8/val.jsonl")
    for source in sorted(previous["sources"]):
        path = DATA_PROCESSED / source / "train.jsonl"
        if sha256(path) != previous["inputs_sha256"][str(path)]:
            raise ValueError(f"Candidate source changed since waveform audit: {source}")
        inputs[str(path)] = sha256(path)
        validations.extend(read(DATA_PROCESSED / source / "val.jsonl"))
    protected = memberships(validations)
    excluded = {evaluation._norm(row["text"]) for row in validations}
    for path in sorted((ROOT / "eval_cases").glob("*.jsonl")):
        excluded.update(evaluation._norm(row["text"]) for row in read(path))
    raw = {}
    for source in sorted(previous["sources"]):
        for row in read(DATA_RAW / source / "manifest.jsonl"):
            key = (source, row["audio"])
            if key in raw and raw[key] != row:
                raise ValueError(f"Ambiguous raw origin: {key}")
            raw[key] = row
    sources, eligible, rejected = {}, [], {}
    for source in sorted(previous["sources"]):
        rows, reasons = [], Counter()
        for original in read(pool / source / "targets.jsonl"):
            row = without_ref(original)
            reason = metadata_rejection(row, protected, excluded)
            if reason:
                reasons[reason] += 1
            else:
                rows.append(row)
        rejected[source] = dict(reasons)
        sources[source] = {"coverage": describe(rows), "provenance": provenance(rows, raw)}
        eligible.extend(rows)
    grouped = defaultdict(list)
    for row in baseline:
        grouped[row["lang"]].append(row)
    feasibility = {}
    for lang in ("tl", "ms"):
        coverage = sources[f"fleurs_{lang}"]["coverage"]
        feasibility[lang] = {
            "pilot_joint_hours": 12, "target_language_hours": 12 * 0.17,
            "candidate_short_duration_share": 0.25,
            "required_short_hours": 12 * 0.17 * 0.25,
            "available_short_hours_1x": coverage["short_hours"],
            "available_short_hours_3x": 3 * coverage["short_hours"],
            "feasible_without_repetition": coverage["short_hours"] >= 12 * 0.17 * 0.25,
            "max_short_share_at_r8_language_budget_3x":
                3 * coverage["short_hours"] / describe(grouped[lang])["hours"],
        }
    sample = review_sample([row for row in eligible if row["source_id"] in ("fleurs_tl", "fleurs_ms")])
    for row in sample:
        row["audio_sha256"] = sha256(row["audio"])
        row["waveform_check"] = inspect_audio(row)
    _write_jsonl(sample, work / "review_samples.jsonl")
    requests = read(examples)
    report = {
        "r8_by_lang": {lang: describe(rows) for lang, rows in sorted(grouped.items())},
        "r8_provenance": provenance(baseline, raw),
        "candidates_by_source": sources, "new_candidate_rejections": rejected,
        "feasibility": feasibility, "inputs_sha256": inputs,
        "production_examples": requests,
        "review_samples_sha256": sha256(work / "review_samples.jsonl"),
        "source_sha256": {str(Path(__file__)): sha256(__file__)},
        "note": "3-8s coverage uses audio-duration share, not record share. "
                "Candidate inventory is not added audio or an approved recipe. "
                "Two archived request examples do not estimate five-language production traffic. "
                "Text length is comparable only within a language; example text is pre-model input. "
                "25% is a feasibility probe, not an optimal ratio or demonstrated remedy. "
                "Review is fixed-random, five non-digit recordings per TL/MS duration stratum. "
                "Existing waveform audit is reused; only review WAVs are rechecked and hashed. "
                "No automatic transcript edits, waveform slicing, training or promotion.",
    }
    write_json(work / "coverage.json", report)
    (work / "audit_done").write_text("Coverage audit complete; training not started.\n")
    print(json.dumps({"r8_by_lang": report["r8_by_lang"], "feasibility": feasibility,
                      "r8_provenance": report["r8_provenance"]["counts"]}, ensure_ascii=False), flush=True)


def asr_check(work):
    from faster_whisper.utils import download_model

    output = work / "asr"
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((work / "coverage.json").read_text())
    sample_path = work / "review_samples.jsonl"
    if sha256(sample_path) != report["review_samples_sha256"]:
        raise ValueError("Frozen review samples changed")
    samples = _read_manifest(sample_path)
    model_path = Path(download_model("large-v3", local_files_only=True))
    output.mkdir()
    runtime_environment(output)
    plan = {
        "samples_sha256": sha256(sample_path),
        "model_path": str(model_path),
        "model_sha256": {path.name: sha256(path) for path in model_path.iterdir() if path.is_file()},
        "source_sha256": {str(Path(__file__)): sha256(__file__),
                          str(Path(evaluation.__file__)): sha256(evaluation.__file__)},
        "decode": {"temperature": 0.0, "vad_filter": False},
        "language": "TL auto-detect; MS forced ms, same as existing evaluation policy",
        "note": "ASR spot-check only, not verified alignment or language/acoustic acceptance. "
                "No automatic filtering or text replacement. Non-digit selection can still contain spoken numbers.",
    }
    write_json(output / "plan.json", plan)
    model = _whisper_model("ms", str(model_path))
    results = {"complete": False, "items": [], "plan_sha256": sha256(output / "plan.json")}
    write_json(output / "results.json", results)
    for position, row in enumerate(samples, 1):
        if sha256(row["audio"]) != row["audio_sha256"]:
            raise ValueError(f"Frozen audio changed: {row['audio']}")
        segments, info = model.transcribe(
            row["audio"], language=None if row["lang"] in evaluation.AUTO_DETECT_LANGS else row["lang"],
            **plan["decode"])
        segments = list(segments)
        hypothesis = " ".join(segment.text.strip() for segment in segments)
        results["items"].append({
            **{key: row[key] for key in ("audio", "audio_sha256", "text", "lang", "duration", "review_stratum")},
            "hyp": hypothesis, "detected_language": info.language,
            "cer": evaluation._error_rate(evaluation._norm(hypothesis), evaluation._norm(row["text"])),
            "segments": [asdict(segment) for segment in segments],
        })
        write_json(output / "results.json", results)
        print(f"{position}/{len(samples)} {row['review_stratum']} CER={results['items'][-1]['cer']:.4f}",
              flush=True)
    results["complete"] = True
    write_json(output / "results.json", results)
    (output / "done").write_text("ASR spot-check complete; no automatic training-data changes.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "asr"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "training_coverage_20260918")
    parser.add_argument("--pool", type=Path, default=CHECKPOINT_DIR.parent / "training_pool_20260917")
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        if args.examples is None:
            parser.error("audit requires sanitized --examples")
        audit(args.work, args.pool, args.examples)
    else:
        asr_check(args.work)
