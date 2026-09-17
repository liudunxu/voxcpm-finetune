"""Prepare next-round candidate manifests from existing audio, without changing any experiment."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from voxft import eval as evaluation
from voxft.data.pipeline import Options, _read_manifest, _write_jsonl, dataset_summary, pair_references
from voxft.data.registry import get_source
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, ROOT

from quality_probe import write_json
from ref_ab import checked_pairs, memberships, sha256, without_ref


def inspect_audio(row):
    waveform, sample_rate = sf.read(row["audio"], dtype="float32", always_2d=True)
    if (sample_rate != 16000 or waveform.shape[1] != 1
            or not 3 <= len(waveform) / sample_rate <= 30
            or not math.isfinite(float(row["duration"]))
            or abs(len(waveform) / sample_rate - float(row["duration"])) > 0.03):
        raise ValueError("format_or_duration")
    if not np.isfinite(waveform).all():
        raise ValueError("nonfinite_waveform")
    amplitude = np.abs(waveform[:, 0])
    peak = float(amplitude.max())
    if not 0 < peak <= 1:
        raise ValueError("silent_or_out_of_range")
    full_scale = amplitude >= 1 - 1 / 32768
    boundaries = np.flatnonzero(np.diff(np.r_[False, full_scale, False]))
    longest = int(np.max(boundaries[1::2] - boundaries[::2])) if len(boundaries) else 0
    nonzero = np.flatnonzero(amplitude > 1 / 32768)
    return {
        "pcm_sha256": hashlib.sha256(waveform.astype("<f4", copy=False).tobytes()).hexdigest(),
        "peak": peak, "full_scale_ratio": float(full_scale.mean()),
        "max_full_scale_run_ms": longest / sample_rate * 1000,
        "near_zero_tail_sec": ((len(waveform) - 1 - int(nonzero[-1])) / sample_rate
                               if len(nonzero) else len(waveform) / sample_rate),
    }


def metadata_rejection(row, protected_keys, excluded_texts):
    if not isinstance(row.get("text"), str) or not evaluation._norm(row["text"]):
        return "empty_text"
    if evaluation._norm(row["text"]) in excluded_texts:
        return "evaluation_or_validation_text"
    if not row.get("origin_audio"):
        return "missing_origin"
    if row.get("speaker_verified") is True and not row.get("speaker"):
        return "missing_verified_identity"
    if memberships([without_ref(row)]) & protected_keys:
        return "validation_audio_speaker_or_session"
    return None


def prepare(output, experiment):
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads((DATA_PROCESSED / "joint_omni8/mix.json").read_text())
    sources = sorted({name for name, _ in recipe["parts"]} |
                     {"cv22_th", "cv22_id", "cv22_vi", "yodas2_ms"})
    manifests, inputs = {}, {}
    for source in sources:
        if get_source(source).license not in {"CC0", "CC-BY-3.0", "CC-BY-4.0", "Apache-2.0"}:
            raise ValueError(f"Source outside existing license allowlist: {source}")
        manifests[source] = {}
        for split in ("train", "val"):
            path = DATA_PROCESSED / source / f"{split}.jsonl"
            inputs[str(path)] = sha256(path)
            manifests[source][split] = _read_manifest(path)
    validation = [without_ref(row) for splits in manifests.values() for row in splits["val"]]
    protected_keys = memberships(validation)
    excluded_texts = {evaluation._norm(row["text"]) for row in validation}
    for path in sorted((ROOT / "eval_cases").glob("*.jsonl")) + [experiment / "cases.jsonl"]:
        inputs[str(path)] = sha256(path)
        excluded_texts.update(evaluation._norm(row["text"]) for row in _read_manifest(path))
    history = {}
    for name in ("joint_omni8", "r10_ref_on"):
        path = DATA_PROCESSED / name / "train.jsonl"
        inputs[str(path)] = sha256(path)
        history[name] = memberships(_read_manifest(path))
    protected_hashes = set()
    for row in validation:
        try:
            protected_hashes.add(inspect_audio(row)["pcm_sha256"])
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"Cannot fingerprint protected validation audio: {row['audio']}") from error
    fingerprints, all_summary, flags = set(), {}, []
    for source, splits in manifests.items():
        kept, rejected = [], Counter()
        for position, original in enumerate(splits["train"], 1):
            row = without_ref(original)
            reason = metadata_rejection(row, protected_keys, excluded_texts)
            if reason:
                rejected[reason] += 1
                continue
            if row.get("speaker_verified") is True and not get_source(source).has_speaker:
                raise ValueError(f"Unapproved verified speaker flag: {source}")
            try:
                audio = inspect_audio(row)
            except (OSError, RuntimeError, ValueError) as error:
                rejected[f"audio:{type(error).__name__}:{error}"] += 1
                flags.append({"source_id": source, "audio": row["audio"], "reason": str(error)})
                continue
            digest = audio["pcm_sha256"]
            if digest in protected_hashes or digest in fingerprints:
                rejected["validation_pcm" if digest in protected_hashes else "duplicate_pcm"] += 1
                continue
            if audio["max_full_scale_run_ms"] >= 1 or audio["near_zero_tail_sec"] > 0.5:
                rejected["acoustic_review_pending"] += 1
                flags.append({"source_id": source, "audio": row["audio"], **audio,
                              "reason": "suspected_clipping_or_long_zero_tail_not_confirmed_badcase"})
                continue
            fingerprints.add(digest)
            kept.append(row)
            if position % 1000 == 0:
                print(f"{source}: {position}/{len(splits['train'])}, kept {len(kept)}", flush=True)
        paired = pair_references([dict(row) for row in kept],
                                 Options(ref_audio_ratio=1.0), random.Random(42))
        eligible = checked_pairs(paired, excluded_texts)
        if memberships(kept) & protected_keys or memberships(eligible) & protected_keys:
            raise ValueError(f"Candidate split leakage: {source}")
        directory = output / source
        directory.mkdir()
        _write_jsonl(kept, directory / "targets.jsonl")
        _write_jsonl(eligible, directory / "ref_candidates.jsonl")
        summary = {
            "license": get_source(source).license,
            "input": dataset_summary(splits["train"]), "targets": dataset_summary(kept),
            "reference_candidates": dataset_summary(eligible),
            "rejected": dict(rejected),
            "duration_bins": dict(Counter("3-8s" if row["duration"] <= 8 else
                                         "8-15s" if row["duration"] <= 15 else "15-30s"
                                         for row in kept)),
            "unmatched_training_memberships": {
                name: sum(not memberships([row]) & used for row in kept)
                for name, used in history.items()},
            "status": ("separate_source_experiment_only" if source in ("cv22_vi", "yodas2_ms")
                       else "candidate_pool_not_a_training_recipe"),
        }
        all_summary[source] = summary
        write_json(directory / "summary.json", summary)
        print(f"{source}: kept={len(kept)}, ref_candidates={len(eligible)}, rejected={dict(rejected)}",
              flush=True)
    _write_jsonl(flags, output / "acoustic_review.jsonl")
    write_json(output / "summary.json", {
        "sources": all_summary, "inputs_sha256": inputs,
        "note": "Existing processed train splits only; all source val audio/IDs/sessions/texts and "
                "existing evaluation texts protected. No dataset downloads or audio modification. "
                "Exact decoded-PCM duplicates removed, not perceptual or near-duplicate detection. "
                "Full-scale plateaus >=1ms and near-zero tails >0.5s quarantined for acoustic review, "
                "not established perceptual failures or new production gates. "
                "Ref candidates use verified identities/different origins/different texts, same split. "
                "Ratio 1 enumerates a candidate inventory, NOT a proposed training ref ratio. "
                "No new identity/control labels, no synthetic audio, no model training/promotion. "
                "CV22 VI remains unverified identity; YODAS MS locale remains unverified. "
                "Membership counts include audio/identity/session, not proof of novel text/speaker. "
                "Input count != novel audio count; no new hours are claimed.",
    })
    (output / "done").write_text("Candidate inventory complete; no training started.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR.parent / "training_pool_20260917")
    parser.add_argument("--experiment", type=Path, default=CHECKPOINT_DIR.parent / "ref_ab_20260917")
    args = parser.parse_args()
    prepare(args.output, args.experiment)
