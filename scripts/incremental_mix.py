"""Audit frozen mixes and guard incremental additions without changing old artifacts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from voxft.data.pipeline import _read_manifest, dataset_summary, text_duration_review
from voxft.eval import _norm
from voxft.paths import ROOT

from ref_ab import memberships, sha256, without_ref

REQUIRED_EVAL_FILES = (
    "omnivoice_prod_v2.jsonl", "fleurs_test_holdout_20260917.jsonl",
    "tl_short_holdout_20260918.jsonl", "id_prosody_probe_20261020.jsonl",
    "short_cue_probe_20261020.jsonl", "drama_chain_20261021.jsonl",
    "drama_chain_prod_20261021.jsonl",
)


def target_text(row):
    text = row.get("text") or ""
    if not isinstance(text, str):
        return ""
    prefix = f"({row.get('control')})"
    return text[len(prefix):] if row.get("control") and text.startswith(prefix) else text


def source_row(row):
    source = row.get("source_id") or ""
    for suffix in ("_r22boost", "_r23boost"):
        source = source.removesuffix(suffix)
    return {**row, "source_id": source}


def load_exclusions(directory=ROOT / "eval_cases"):
    for name in REQUIRED_EVAL_FILES:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    paths = sorted(directory.glob("*.jsonl"))
    hashes = {str(path): sha256(path) for path in paths}
    texts = {_norm(target_text(row)) for path in paths for row in _read_manifest(path)
             if row.get("text")}
    return texts, hashes


def row_issues(row, protected, excluded, max_dur: float = 20.0):
    issues = []
    text = target_text(row)
    if not isinstance(text, str) or not _norm(text):
        issues.append("empty_text")
    elif _norm(text) in excluded:
        issues.append("evaluation_or_validation_text")
    if not row.get("audio") or not row.get("origin_audio"):
        issues.append("missing_audio_or_origin")
    if row.get("lang") not in {"th", "tl", "id", "zh", "en"}:
        issues.append("unsupported_language")
    try:
        duration = float(row["duration"])
        valid_duration = math.isfinite(duration) and 3 <= duration <= max_dur
    except (KeyError, TypeError, ValueError):
        valid_duration = False
    if not valid_duration:
        issues.append(f"duration_outside_3_{max_dur:g}s")
    elif review := text_duration_review(row):
        issues.append(review["reason"])
    if row.get("speaker_verified") is True and not row.get("speaker"):
        issues.append("missing_verified_identity")
    elif memberships([source_row(row)]) & protected:
        issues.append("validation_audio_speaker_or_session")
    if row.get("ref_audio"):
        if (row.get("speaker_verified") is not True or not row.get("speaker")
                or row.get("speaker") != row.get("ref_speaker")
                or not row.get("ref_origin_audio")
                or not row.get("origin_audio")
                or Path(row["origin_audio"]).resolve() == Path(row["ref_origin_audio"]).resolve()):
            issues.append("unverified_or_same_origin_reference")
    return issues


def audit_mix(train, validation, excluded, max_dur: float = 20.0):
    if not math.isfinite(max_dur) or not 3 <= max_dur <= 30:
        raise ValueError("Audit duration limit must be within 3-30 seconds")
    protected = memberships([source_row(row) for row in validation])
    validation_texts = {_norm(target_text(row)) for row in validation if row.get("text")}
    evaluation_texts = excluded
    excluded = excluded | validation_texts
    violations = []
    for position, row in enumerate(train, 1):
        issues = row_issues(row, protected, excluded, max_dur)
        if issues:
            violations.append({"row": position, "audio": row.get("audio"),
                               "lang": row.get("lang"), "source_id": row.get("source_id"),
                               "issues": issues})
            if "text_duration_review_pending" in issues:
                violations[-1]["text_duration_diagnostic"] = text_duration_review(row)
    exposures = Counter(str(Path(row.get("origin_audio") or row["audio"]).resolve())
                        for row in train if row.get("audio"))
    return {
        "rows": len(train), "validation_rows": len(validation), "max_duration_sec": max_dur,
        "by_lang": {
            lang: {"rows": len(rows),
                   "with_ref": sum(bool(row.get("ref_audio")) for row in rows),
                   "with_control": sum(bool(row.get("control")) for row in rows),
                   "control_inoculation": sum(bool(row.get("control_inoculation")) for row in rows)}
            for lang in sorted({row.get("lang", "unknown") for row in train})
            for rows in [[row for row in train if row.get("lang", "unknown") == lang]]},
        "violation_counts": dict(Counter(issue for row in violations for issue in row["issues"])),
        "evaluation_text_rows": sum(_norm(target_text(row)) in evaluation_texts for row in train),
        "validation_text_rows": sum(_norm(target_text(row)) in validation_texts for row in train),
        "violations": violations,
        "origins_above_3x": {origin: count for origin, count in exposures.items() if count > 3},
        "note": "Metadata/path/text audit only; no PCM, audio quality, license or identity verification. "
                "Historical r22/r23 source suffixes are removed only for session comparison.",
    }


def validate_mix(train, validation, excluded):
    audit = audit_mix(train, validation, excluded)
    if not train or audit["violations"] or audit["origins_above_3x"]:
        raise ValueError(f"Unsafe training mix: {audit['violation_counts']}; "
                         f"origins_above_3x={len(audit['origins_above_3x'])}; rows={len(train)}")
    return audit


def eligible_additions(candidates, train, validation, excluded, max_dur: float = 20.0):
    protected = memberships([source_row(row) for row in validation])
    excluded = excluded | {_norm(target_text(row)) for row in validation if row.get("text")}
    used = {key for key in memberships([without_ref(row) for row in train]) if key[0] == "audio"}
    kept, rejected = [], Counter()
    for row in candidates:
        issues = row_issues(row, protected, excluded, max_dur)
        target_keys = {("audio", str(Path(row[field]).resolve()))
                       for field in ("audio", "origin_audio") if row.get(field)}
        if target_keys & used:
            issues.append("existing_or_duplicate_target")
        if issues:
            rejected.update(issues)
        else:
            kept.append(dict(row))
            used.update(target_keys)
    return kept, dict(rejected)


def describe_mix(rows):
    summary = dataset_summary(rows)
    seconds = Counter()
    for row in rows:
        seconds[row["lang"]] += row["duration"]
    return summary, {
        lang: {"requested": None, "actual": duration / summary["seconds"],
               "hours": duration / 3600, "rows": summary["languages"][lang]}
        for lang, duration in sorted(seconds.items())}


def align_addition(row, columns):
    extra = {key for key in row.keys() - columns if row[key] is not None}
    if extra:
        raise ValueError(f"Refusing to discard new metadata columns: {sorted(extra)}")
    return {key: row.get(key) for key in sorted(columns)}


def audit_directory(directory, eval_dir, max_dur: float = 20.0):
    excluded, hashes = load_exclusions(eval_dir)
    paths = [directory / f"{split}.jsonl" for split in ("train", "val")]
    hashes.update({str(path): sha256(path) for path in paths})
    result = audit_mix(*[_read_manifest(path) for path in paths], excluded, max_dur)
    training_texts = Counter(_norm(target_text(row)) for row in _read_manifest(paths[0]))
    result["evaluation_exposure"] = {}
    for path in sorted(eval_dir.glob("*.jsonl")):
        cases = _read_manifest(path)
        overlap = {_norm(target_text(row)) for row in cases} & training_texts.keys()
        if overlap:
            result["evaluation_exposure"][path.name] = {
                "training_rows": sum(training_texts[text] for text in overlap),
                "unique_texts": len(overlap),
                "case_ids": [row.get("case_id") for row in cases
                             if _norm(target_text(row)) in overlap],
            }
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("Input changed during audit")
    return {**result, "inputs_sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "eval_cases")
    parser.add_argument("--max-dur", type=float, default=20.0,
                        help="Audit the frozen recipe's duration cap; default 20, never above 30")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_directory(args.directory, args.eval_dir, args.max_dur)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"violations", "inputs_sha256", "origins_above_3x"}},
                     ensure_ascii=False, indent=2))
