"""Quarantine unresolved r26/r27 rows without changing frozen inputs or starting training."""
import argparse
from collections import Counter
import inspect
import json
from pathlib import Path

import soundfile as sf

from audit_finetune_readiness import RUNS, coverage
from control_ab_gate import pcm_conflicts
from incremental_mix import audit_mix, describe_mix, load_exclusions
from quality_probe import write_json
from ref_ab import sha256
from training_pool import inspect_audio
from voxft.data.pipeline import _read_manifest, text_duration_review
from voxft.paths import ROOT
from voxft.qc.audio import edge_silence, speech_ratio


def quarantine_rows(train, validation, excluded, audio):
    before = audit_mix(train, validation, excluded, max_dur=30.0)
    flagged = {item["row"]: item for item in before["violations"]}
    blocked_paths, blocked_pcm = set(), set()
    for position in flagged:
        row = train[position - 1]
        blocked_paths.update(str(Path(row[field]).resolve())
                             for field in ("audio", "origin_audio") if row.get(field))
        blocked_pcm.add(audio[row["audio"]]["pcm_sha256"])
    kept_positions, rejected = [], []
    for position, row in enumerate(train, 1):
        issues = list(flagged.get(position, {}).get("issues", []))
        for field, origin in (("audio", "origin_audio"), ("ref_audio", "ref_origin_audio")):
            if not row.get(field):
                continue
            if (any(str(Path(row[key]).resolve()) in blocked_paths
                    for key in (field, origin) if row.get(key))
                    or audio[row[field]]["pcm_sha256"] in blocked_pcm):
                if field == "ref_audio" or position not in flagged:
                    issues.append(f"quarantined_{field}")
        if issues:
            rejected.append({
                "row": position, "issues": issues, "sample": row,
                "text_duration_diagnostic": text_duration_review(row),
                "status": "excluded_from_candidate_not_confirmed_bad_audio",
            })
        else:
            kept_positions.append(position)
    kept = [train[position - 1] for position in kept_positions]
    after = audit_mix(kept, validation, excluded, max_dur=30.0)
    conflicts = pcm_conflicts(kept, validation, audio)
    if not kept or after["violations"] or after["origins_above_3x"] or any(conflicts.values()):
        raise ValueError("Candidate still has unresolved metadata or PCM conflicts")
    return kept_positions, rejected, {**after, "pcm": conflicts}


def duration_diagnostic(row, speech_timestamps):
    checks = inspect_audio(row)
    waveform, sample_rate = sf.read(row["audio"], dtype="float32")
    segments = speech_timestamps(waveform, sample_rate)
    previous_end = 0
    for segment in segments:
        if not previous_end <= segment["start"] < segment["end"] <= len(waveform):
            raise ValueError("Invalid VAD boundaries")
        previous_end = segment["end"]
    duration = len(waveform) / sample_rate
    speech_seconds = sum(segment["end"] - segment["start"] for segment in segments) / sample_rate
    gaps = [(right["start"] - left["end"]) / sample_rate
            for left, right in zip(segments, segments[1:])]
    review = text_duration_review(row)
    return {
        "sample": row, "text_duration_diagnostic": review, "audio_checks": checks,
        "actual_duration": duration, "raw_audio_available": Path(row["origin_audio"]).is_file(),
        "raw_processed_comparison": "not_performed",
        "energy_activity_ratio": speech_ratio(waveform, sample_rate),
        "energy_edge_silence": edge_silence(waveform, sample_rate),
        "vad_segments_seconds": [{"start": segment["start"] / sample_rate,
                                  "end": segment["end"] / sample_rate} for segment in segments],
        "vad_speech_seconds": speech_seconds,
        "vad_speech_ratio": speech_seconds / duration,
        "vad_head_seconds": segments[0]["start"] / sample_rate if segments else None,
        "vad_tail_seconds": (len(waveform) - segments[-1]["end"]) / sample_rate if segments else None,
        "vad_internal_gap_seconds": sum(gaps), "vad_max_internal_gap_seconds": max(gaps, default=0),
        "vad_words_per_second": review["word_count"] / speech_seconds if review and speech_seconds else None,
        "review_status": "unreviewed_not_confirmed_bad_audio",
        "transcript_consistency": "unverified", "native_language_quality": "unverified",
        "note": "VAD/energy measurements are diagnostic, never approval, transcription or trimming.",
    }


def prepare(audit_dir, eval_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def freeze(path, expected=None):
        path = Path(path)
        digest = sha256(path)
        if (expected is not None and digest != expected
                or str(path) in inputs and inputs[str(path)] != digest):
            raise ValueError(f"Frozen input changed: {path}")
        inputs[str(path)] = digest
        return path

    cache = json.loads(freeze(audit_dir / "audio_audit.json").read_text(encoding="utf-8"))
    if cache["failures"]:
        raise ValueError("Incomplete frozen audio audit")
    audio = cache["files"]
    excluded, evaluation_hashes = load_exclusions(eval_dir)
    for path, digest in evaluation_hashes.items():
        freeze(path, digest)
    for function in (prepare, coverage, audit_mix, pcm_conflicts, sha256, _read_manifest,
                     inspect_audio, speech_ratio):
        freeze(inspect.getfile(function))
    sources, audio_paths = {}, set()
    for run in RUNS:
        previous = json.loads(freeze(audit_dir / f"{run}_metadata.json").read_text(encoding="utf-8"))
        for path, digest in previous["inputs_sha256"].items():
            freeze(path, digest)
        directory = next(Path(path).parent for path in previous["inputs_sha256"]
                         if Path(path).name == "train.jsonl")
        lines = [line for line in (directory / "train.jsonl").read_bytes().splitlines(keepends=True)
                 if line.strip()]
        train = [json.loads(line) for line in lines]
        validation = _read_manifest(directory / "val.jsonl")
        sources[run] = (directory, lines, train, validation)
        audio_paths.update(row[field] for row in train + validation
                           for field in ("audio", "ref_audio") if row.get(field))
    for position, path in enumerate(sorted(audio_paths), 1):
        freeze(path, audio[path]["file_sha256"])
        if position % 1000 == 0:
            print(f"Verified frozen audio bytes: {position}/{len(audio_paths)}", flush=True)

    results, pending = {}, {}
    for run, (directory, lines, train, validation) in sources.items():
        positions, rejected, audit = quarantine_rows(train, validation, excluded, audio)
        kept = [train[position - 1] for position in positions]
        destination = output / run
        destination.mkdir()
        (destination / "train.jsonl").write_bytes(b"".join(lines[position - 1] for position in positions))
        (destination / "val.jsonl").write_bytes((directory / "val.jsonl").read_bytes())
        with (destination / "excluded.jsonl").open("x", encoding="utf-8") as stream:
            for record in rejected:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if record["text_duration_diagnostic"]:
                    pending[record["sample"]["audio"]] = record["sample"]
        train_summary, shares = describe_mix(kept)
        _, previous_shares = describe_mix(train)
        for lang, share in shares.items():
            share["parent_actual"] = previous_shares[lang]["actual"]
        results[run] = {
            "status": "candidate_only_not_approved_for_training",
            "parent_directory": str(directory), "before": coverage(train), "after": coverage(kept),
            "by_lang": {lang: {"before": coverage([row for row in train if row["lang"] == lang]),
                               "after": coverage([row for row in kept if row["lang"] == lang])}
                        for lang in sorted({row["lang"] for row in train})},
            "excluded_rows": len(rejected),
            "excluded_by_source": dict(Counter(record["sample"]["source_id"] for record in rejected)),
            "reasons": dict(Counter(reason for record in rejected for reason in record["issues"])),
            "train": train_summary, "language_shares": shares, "metadata_audit": audit,
            "training_approved": False, "retained_rows_byte_identical": True,
            "outputs_sha256": {name: sha256(destination / name)
                               for name in ("train.jsonl", "val.jsonl", "excluded.jsonl")},
        }
        write_json(destination / "mix.json", results[run])
        print(f"{run}: {len(train)} -> {len(kept)}, excluded={len(rejected)}", flush=True)

    from faster_whisper.utils import get_assets_path
    from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

    model = get_vad_model()
    providers = model.session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise ValueError(f"VAD must use CPU only: {providers}")
    freeze(inspect.getfile(get_speech_timestamps))
    for path in sorted(Path(get_assets_path()).glob("silero_vad*.onnx")):
        freeze(path)
    options = VadOptions(min_silence_duration_ms=500, speech_pad_ms=0)
    diagnostics = []
    for path, row in sorted(pending.items()):
        diagnostic = duration_diagnostic(row, lambda waveform, rate: get_speech_timestamps(
            waveform, options, sampling_rate=rate))
        if diagnostic["audio_checks"]["pcm_sha256"] != audio[path]["pcm_sha256"]:
            raise ValueError(f"PCM changed: {path}")
        diagnostics.append(diagnostic)
    write_json(output / "duration_diagnostics.json", {
        "scope": "pending_rows_only_not_full_corpus_language_validation",
        "vad": {"providers": providers, "threshold": options.threshold,
                "min_silence_duration_ms": 500, "speech_pad_ms": 0},
        "items": diagnostics,
    })
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during preparation: {path}")
    write_json(output / "inputs_sha256.json", inputs)
    summary = {
        "status": "cpu_quarantine_complete_not_training_approval", "runs": results,
        "verified_audio_files": len(audio_paths), "unique_duration_review_audio": len(pending),
        "duration_review_seconds": sum(row["duration"] for row in pending.values()),
        "raw_audio_missing": sum(not item["raw_audio_available"] for item in diagnostics),
        "training_started": False, "asr_started": False, "tts_started": False,
        "source_mutations": False, "threshold_changed": False, "production_changed": False,
        "note": "No new data, resampling, trimming, ref replacement or control relabeling. "
                "Excluded rows remain unapproved, not confirmed bad recordings. "
                "Unknown identity/consent and transcript/native-language quality remain unverified.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"},
                     ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "eval_cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.audit_dir.resolve(), args.eval_dir.resolve(), args.output.resolve())
