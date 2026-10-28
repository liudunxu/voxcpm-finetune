"""Audit r26/r27 and reserve evaluation inputs using existing files only."""
import argparse
from collections import Counter
import json
from pathlib import Path

import soundfile as sf

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest, _write_jsonl, dataset_summary
from voxft.data.registry import get_source
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, DATA_RAW, ROOT, env

from control_ab_gate import build_cases, choose_references, pcm_conflicts, verify_provenance
from holdout_texts import REVISION, parse_tsv, select_texts
from incremental_mix import audit_directory, source_row, target_text
from ref_ab import memberships, sha256, without_ref
from ref_ab_diagnostics import expand_references
from training_pool import inspect_audio

RUNS = {"r26": "r26_r18_inoc10", "r27": "r27_scale25h"}


def coverage(rows):
    return {
        **dataset_summary(rows),
        "verified_speaker_rows": sum(row.get("speaker_verified") is True for row in rows),
        "verified_control_rows": sum(bool(row.get("control")) and (
            row.get("control_verified") is True or row.get("emotion_verified") is True)
                                    for row in rows),
        "control_inoculation_rows": sum(row.get("control_inoculation") is True for row in rows),
        "cross_language_ref_rows": sum(bool(row.get("ref_audio")) and bool(row.get("ref_lang"))
                                       and row["ref_lang"] != row["lang"] for row in rows),
    }


def check_audio(rows):
    files, failures, inputs = {}, [], {}
    for row in rows:
        for field, duration_field in (("audio", "duration"), ("ref_audio", "ref_duration")):
            path = row.get(field)
            if not path:
                continue
            if path in files:
                if abs(float(row[duration_field]) - files[path]["duration"]) > 0.03:
                    failures.append({"audio": path, "reason": "conflicting_manifest_duration"})
                continue
            try:
                digest = sha256(path)
                record = inspect_audio({"audio": path, "duration": row[duration_field]})
                if sha256(path) != digest:
                    raise ValueError("Audio changed during inspection")
                files[path] = {**record, "file_sha256": digest, "duration": float(row[duration_field])}
                inputs[path] = digest
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                failures.append({"audio": path, "reason": f"{type(error).__name__}: {error}"})
            if (len(files) + len(failures)) % 1000 == 0:
                print(f"PCM files={len(files)}, failures={len(failures)}", flush=True)
    return files, failures, inputs


def resolve_probes(rows, reference_root):
    ready, blocked = [], []
    for row in rows:
        path = (reference_root / row["ref_audio"]).resolve()
        status = row.get("reference_status") or ""
        reason = status if status.startswith("blocked") else (
            "missing_reference_file" if not path.is_file() else "")
        record = {**row, "ref_audio": str(path)}
        if reason:
            blocked.append({**record, "blocking_reason": reason})
        else:
            ready.append(record)
    return ready, blocked


def prepare(work, repo, eval_dir):
    work.mkdir(parents=True, exist_ok=False)
    inputs, manifests, excluded = {}, {}, set()

    def freeze(path):
        path = Path(path)
        digest = sha256(path)
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ValueError(f"Input changed: {path}")
        inputs[str(path)] = digest
        return path

    def write(name, value):
        path = work / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        return path

    paths = (set(DATA_PROCESSED.glob("*/train.jsonl")) | set(DATA_PROCESSED.glob("*/val.jsonl"))
             | set(DATA_RAW.glob("*/manifest.jsonl")) | set(eval_dir.glob("*.jsonl"))
             | set((repo / "eval_cases").glob("*.jsonl"))
             | set(DATA_PROCESSED.parent.parent.glob("*/cases.jsonl")))
    for path in sorted(paths):
        rows = _read_manifest(freeze(path))
        excluded.update(evaluation._norm(target_text(row)) for row in rows if row.get("text"))
        if path.parent.parent in {DATA_PROCESSED, DATA_RAW}:
            manifests[path] = rows
    for path in sorted((CHECKPOINT_DIR / "eval").glob("*.json")):
        report = json.loads(freeze(path).read_text(encoding="utf-8"))
        excluded.update(evaluation._norm(target_text(row)) for row in report.get("items", [])
                        if row.get("text"))
    for path in sorted(Path(__file__).parent.glob("*.py")):
        freeze(path)
    for path in sorted((ROOT / "src").rglob("*.py")):
        freeze(path)

    runs, summaries, reviews = {}, {}, []
    for name, directory in RUNS.items():
        source = DATA_PROCESSED / directory
        train, validation = (manifests[source / f"{split}.jsonl"] for split in ("train", "val"))
        runs[name] = (train, validation)
        audit = audit_directory(source, eval_dir, max_dur=30.0)
        write(f"{name}_metadata.json", audit)
        summaries[name] = {
            "metadata": {key: value for key, value in audit.items()
                         if key not in {"violations", "inputs_sha256"}},
            "by_lang": {lang: coverage([row for row in train if row["lang"] == lang])
                        for lang in sorted({row["lang"] for row in train})},
            "by_source": {source_id: coverage([row for row in train if row["source_id"] == source_id])
                          for source_id in sorted({row["source_id"] for row in train})},
        }
        for item in audit["violations"]:
            if "text_duration_review_pending" in item["issues"]:
                reviews.append({**train[item["row"] - 1], "run": name, "row": item["row"],
                                "text_duration_diagnostic": item["text_duration_diagnostic"],
                                "review_status": "unreviewed_not_confirmed_bad_audio"})
    _write_jsonl(reviews, work / "text_duration_review.jsonl")

    sources = {source_row(row)["source_id"] for train, validation in runs.values()
               for row in train + validation}
    index = {str(Path(row["audio"]).resolve()): row
             for path, rows in manifests.items()
             if path.parent.name in sources and path.name in {"train.jsonl", "val.jsonl"}
             for row in rows}
    raw = {str(Path(row["audio"]).resolve()): row for path, rows in manifests.items()
           if path.name == "manifest.jsonl" for row in rows}
    provenance = []
    for name, (train, validation) in runs.items():
        for row in train + validation:
            try:
                for original in expand_references([row], index):
                    verify_provenance(original, raw)
            except (KeyError, ValueError) as error:
                provenance.append({"run": name, "audio": row["audio"], "reason": str(error)})
    write("provenance_issues.json", provenance)

    all_rows = [row for train, validation in runs.values() for row in train + validation]
    reference_pool = [without_ref(row) for lang in ("zh", "en")
                      for row in manifests[DATA_PROCESSED / f"fleurs_{lang}/val.jsonl"]
                      if 3 <= row["duration"] <= 10]
    audio, failures, audio_inputs = check_audio(all_rows + reference_pool)
    inputs.update(audio_inputs)
    write("audio_audit.json", {"files": audio, "failures": failures})
    for name, (train, validation) in runs.items():
        complete = all(row[field] in audio for row in train + validation
                       for field in ("audio", "ref_audio") if row.get(field))
        summaries[name]["pcm"] = {
            "complete": complete,
            "conflicts": pcm_conflicts(train, validation, audio) if complete else None,
        }

    probes = _read_manifest(eval_dir / "language_reading_probes_20261026.jsonl")
    ready, blocked = resolve_probes(probes, repo / "eval_cases")
    probe_rows = [{"audio": path, "duration": sf.info(path).duration}
                  for path in sorted({row["ref_audio"] for row in ready})]
    probe_audio, probe_errors, probe_inputs = check_audio(probe_rows)
    inputs.update(probe_inputs)
    invalid_refs = {item["audio"] for item in probe_errors}
    blocked.extend({**row, "blocking_reason": "reference_audio_check_failed"}
                   for row in ready if row["ref_audio"] in invalid_refs)
    ready = [row for row in ready if row["ref_audio"] not in invalid_refs]
    _write_jsonl(ready, work / "reading_cross_reference_cases.jsonl")
    _write_jsonl(blocked, work / "reading_blocked_cases.jsonl")
    write("reading_readiness.json", {
        "ready_cases": len(ready), "blocked_cases": len(blocked), "references": probe_audio,
        "failures": probe_errors, "native_language_quality": "unverified",
        "scope": "Registered existing reference bytes, not independent speaker/provenance certification.",
    })

    fresh = {"prepared": False}
    if not failures and not provenance and all(
            entry["pcm"]["complete"] and not any(entry["pcm"]["conflicts"].values())
            for entry in summaries.values()):
        used = [row for train, _ in runs.values() for row in train]
        used_keys = memberships(used)
        used_pcm = {audio[row[field]]["pcm_sha256"] for row in used
                    for field in ("audio", "ref_audio") if row.get(field)}
        references, reference_records = {}, []
        for lang in ("zh", "en"):
            chosen = choose_references([row for row in reference_pool if row["lang"] == lang],
                                       used_keys, used_pcm, audio)
            for row in chosen:
                verify_provenance(row, raw)
            references[lang] = [row["audio"] for row in chosen]
            reference_records.extend(chosen)
        selected, counts = {}, {}
        snapshot = Path(env("HF_HOME")) / "hub/datasets--google--fleurs/snapshots" / REVISION
        for lang in ("th", "tl", "id"):
            path = freeze(snapshot / "data" / get_source(f"fleurs_{lang}").config / "test.tsv")
            selected[lang], counts[lang] = select_texts(
                parse_tsv(path.read_text(encoding="utf-8")), excluded, lang)
        fresh = {"prepared": all(not item["shortfall"] for item in counts.values()),
                 "by_lang": counts, "references": reference_records}
        if fresh["prepared"]:
            cases = build_cases(selected, references)
            _write_jsonl(cases, work / "independent_cases.jsonl")
            fresh.update(cases=len(cases), independent_texts=90,
                         cases_sha256=sha256(work / "independent_cases.jsonl"))
    else:
        fresh["blocked_by"] = "Incomplete audio/provenance checks or PCM isolation conflicts"
    write("fresh_evaluation.json", fresh)

    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Frozen input changed during CPU audit")
    summary = {
        "status": "cpu_preparation_only", "training_started": False, "inference_started": False,
        "runs": summaries, "audio_files": len(audio), "audio_failures": len(failures),
        "provenance_issues": len(provenance), "fresh_evaluation": fresh,
        "reading_ready": len(ready), "reading_blocked": len(blocked),
        "evaluation_seeds": [42, 43, 44, 45, 46], "cfg_value": 1.8,
        "inference_timesteps": 20, "retry_badcase": False, "asr_model": "large-v3",
        "source_mutations": False, "production_changed": False,
        "note": "Exact PCM/path/text/source-metadata checks only; unknown identities, near-duplicates, "
                "recording consent, acoustic and native-language quality remain unverified. "
                "Slow text/duration rows are pending review, not confirmed training defects. "
                "No source audio download, resampling, ASR, TTS, training or source-manifest edits. "
                "New cases require new matched baselines; historical scores are not reused.",
    }
    write("inputs_sha256.json", inputs)
    write("summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"runs", "fresh_evaluation"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "eval_cases")
    args = parser.parse_args()
    prepare(args.work.resolve(), args.repo.resolve(), args.eval_dir.resolve())
