"""Read-only energy/VAD audit of a frozen run's actual training targets and refs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from control_ab_review import tail_summary
from ref_ab import sha256
from voxft.data.pipeline import _read_manifest
from voxft.qc.audio import edge_silence, speech_ratio


def vad_bounds(segments, samples, rate):
    previous = 0
    for segment in segments:
        if not previous <= segment["start"] < segment["end"] <= samples:
            raise ValueError("Invalid VAD boundaries")
        previous = segment["end"]
    return {
        "segments": [{"start": part["start"] / rate, "end": part["end"] / rate}
                     for part in segments],
        "speech_seconds": sum(part["end"] - part["start"] for part in segments) / rate,
        "head_seconds": segments[0]["start"] / rate if segments else None,
        "tail_seconds": (samples - segments[-1]["end"]) / rate if segments else None,
    }


def measure(path, duration, expected_sha, speech_timestamps):
    if not expected_sha or sha256(path) != expected_sha:
        raise ValueError(f"Frozen audio changed or missing SHA: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if (rate != 16000 or audio.shape[1] != 1 or not 3 <= len(audio) / rate <= 30
            or not math.isfinite(duration) or abs(len(audio) / rate - duration) > 0.03
            or not np.isfinite(audio).all()):
        raise ValueError(f"Invalid audio format/duration: {path}")
    audio = audio[:, 0]
    peak = float(np.abs(audio).max())
    if peak > 1:
        raise ValueError(f"Out-of-range audio: {path}")
    edges = {key: float(value) if value is not None else None
             for key, value in edge_silence(audio, rate).items()}
    vad = vad_bounds(speech_timestamps(audio, rate), len(audio), rate)
    above_lsb = np.flatnonzero(np.abs(audio) > 1 / 32768)
    reasons = []
    if peak == 0:
        reasons.append("all_zero_waveform")
    if edges["tail_silence_sec"] is None:
        reasons.append("no_energy_boundary")
    elif edges["tail_silence_sec"] > 0.5:
        reasons.append("energy_tail_over_0_5s")
    if not vad["segments"]:
        reasons.append("no_vad_speech")
    elif vad["tail_seconds"] > 0.5:
        reasons.append("vad_tail_over_0_5s")
    gain_check = None
    if 0 < peak < 0.1 and (not vad["segments"] or edges["tail_silence_sec"] is None):
        # Diagnose detector sensitivity only; never save or train on amplified audio.
        gain = 0.5 / peak
        gain_check = {"gain": gain, **vad_bounds(
            speech_timestamps(audio * gain, rate), len(audio), rate)}
    if sha256(path) != expected_sha:
        raise ValueError(f"Audio changed during audit: {path}")
    return {
        "audio": str(path), "file_sha256": expected_sha,
        "pcm_sha256": hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest(),
        "sample_rate": rate, "samples": len(audio), "duration": len(audio) / rate,
        "peak": peak, "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        "nonzero_samples": int(np.count_nonzero(audio)),
        "near_zero_tail_seconds": (len(audio) - 1 - int(above_lsb[-1])) / rate
        if above_lsb.size else len(audio) / rate,
        **edges, "energy_activity_ratio": speech_ratio(audio, rate), "vad": vad,
        "gain_diagnostic": gain_check, "review_reasons": reasons,
        "review_status": "pending_not_confirmed_bad_audio" if reasons else "no_silence_flag",
    }


def summarize(items):
    items = list(items)
    return {
        "energy_tail": tail_summary(items),
        "vad_tail": tail_summary([{"tail_silence_sec": item["vad"]["tail_seconds"]}
                                  for item in items]),
        "review_counts": dict(Counter(reason for item in items for reason in item["review_reasons"])),
    }


def prepare(run, output):
    if output.exists():
        raise FileExistsError(output)
    plan_path, manifest = run / "plan.json", run / "train.jsonl"
    plan_hash = sha256(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = plan["inputs_sha256"]
    if sha256(manifest) != expected[str(manifest)]:
        raise ValueError("Frozen training manifest changed")
    rows = _read_manifest(manifest)
    uses, durations = defaultdict(list), {}
    for position, row in enumerate(rows, 1):
        for role, prefix in (("target", ""), ("reference", "ref_")):
            path = row.get(prefix + "audio")
            if not path:
                continue
            duration = float(row[prefix + "duration"])
            if path in durations and abs(durations[path] - duration) > 0.03:
                raise ValueError(f"Conflicting manifest duration: {path}")
            durations[path] = duration
            uses[path].append({"row": position, "role": role,
                               "lang": row.get(prefix + "lang", "unknown"),
                               "target_source_id": row["source_id"],
                               "origin_audio": row.get(prefix + "origin_audio")})
    paths = sorted(uses, key=lambda path: (not any(use["lang"] == "id" for use in uses[path]), path))
    inputs = {str(plan_path): plan_hash, str(manifest): expected[str(manifest)]}
    for path in paths:
        inputs[path] = expected[path]

    from faster_whisper.utils import get_assets_path
    from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

    model = get_vad_model()
    providers = model.session.get_providers()
    if providers != ["CPUExecutionProvider"]:
        raise ValueError(f"VAD must use CPU only: {providers}")
    for function in (prepare, sha256, _read_manifest, edge_silence, speech_ratio,
                     tail_summary, get_speech_timestamps):
        path = inspect.getfile(function)
        inputs[path] = sha256(path)
    for path in sorted(Path(get_assets_path()).glob("silero_vad*.onnx")):
        inputs[str(path)] = sha256(path)
    options = VadOptions(min_silence_duration_ms=500, speech_pad_ms=0)
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)

    write("plan.json", {"run": str(run), "inputs_sha256": inputs, "workers": 4,
                        "unique_files": len(paths), "training_rows": len(rows),
                        "reference_usages": sum(bool(row.get("ref_audio")) for row in rows),
                        "vad": {"providers": providers, "threshold": options.threshold,
                                "min_silence_duration_ms": 500, "speech_pad_ms": 0},
                        "note": "Detection flags are review candidates, not deletion decisions. "
                                "Reference source groups identify the consuming training source."})

    def scan(path):
        return measure(path, durations[path], inputs[path], lambda audio, rate:
                       get_speech_timestamps(audio, options, sampling_rate=rate))

    metrics = {}
    with (output / "measurements.jsonl").open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for count, item in enumerate(executor.map(scan, paths), 1):
                item["uses"] = uses[item["audio"]]
                metrics[item["audio"]] = item
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                if count % 500 == 0 or count == len(paths):
                    stream.flush()
                    print(f"scanned {count}/{len(paths)}; flagged="
                          f"{sum(bool(row['review_reasons']) for row in metrics.values())}", flush=True)
    groups = defaultdict(list)
    for path, usages in uses.items():
        for use in usages:
            groups[use["role"], use["lang"], use["target_source_id"]].append(metrics[path])
    flagged = [item for item in metrics.values() if item["review_reasons"]]
    write("flagged.json", flagged)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during audit: {path}")
    summary = {
        "status": "complete", "training_rows": len(rows), "unique_files": len(paths),
        "unique_targets": len({row["audio"] for row in rows}),
        "reference_usages": sum(bool(row.get("ref_audio")) for row in rows),
        "unique_references": len({row["ref_audio"] for row in rows if row.get("ref_audio")}),
        "unique_file_metrics": summarize(metrics.values()), "flagged_files": len(flagged),
        "by_role_language_source": [
            {"role": key[0], "lang": key[1], "target_source_id": key[2], **summarize(items)}
            for key, items in sorted(groups.items())],
        "source_mutations": False, "training_started": False, "asr_started": False,
        "tts_started": False, "production_changed": False,
        "note": "No trimming, normalizing saved audio, exclusion, training or language acceptance. "
                "Unmeasurable energy/VAD boundaries remain unknown, never zero silence. "
                "VAD can miss quiet speech or include noise; review flags require corroboration.",
    }
    write("summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "by_role_language_source"}, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run.resolve(), args.output.resolve())
