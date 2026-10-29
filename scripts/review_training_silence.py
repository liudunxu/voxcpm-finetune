"""Unprompted ASR diagnostics for silence-audit flags; never relabel or delete audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import inspect
import json
from pathlib import Path

import soundfile as sf

from control_ab_gate import pcm_conflicts
from fresh_baseline_eval import save_new
from incremental_mix import audit_mix, describe_mix, load_exclusions
from ref_ab import sha256
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.eval import AUTO_DETECT_LANGS, _error_rate, _norm
from voxft.paths import ROOT


def transcribe(model, audio, lang, expected_texts):
    segments, info = model.transcribe(
        audio, language=None if lang in AUTO_DETECT_LANGS else lang,
        vad_filter=False, initial_prompt=None, beam_size=5, temperature=0,
        condition_on_previous_text=False)
    segments = list(segments)
    hyp = " ".join(segment.text.strip() for segment in segments)
    return {
        "hyp": hyp, "detected_language": info.language,
        "language_probability": info.language_probability,
        "segments": [{"start": part.start, "end": part.end, "text": part.text,
                      "avg_logprob": part.avg_logprob, "no_speech_prob": part.no_speech_prob}
                     for part in segments],
        "comparisons": [{"text": text, "cer": _error_rate(_norm(hyp), _norm(text))}
                        for text in expected_texts],
    }


def exclude_flagged(rows, flagged, audio):
    blocked_paths = {str(Path(path).resolve()) for item in flagged
                     for path in [item["audio"], *(use.get("origin_audio") for use in item["uses"])]
                     if path}
    blocked_pcm = {item["pcm_sha256"] for item in flagged}
    kept, rejected = [], []
    for position, row in enumerate(rows, 1):
        reasons = []
        for field, origin in (("audio", "origin_audio"), ("ref_audio", "ref_origin_audio")):
            if row.get(field) and (
                    any(str(Path(row[key]).resolve()) in blocked_paths
                        for key in (field, origin) if row.get(key))
                    or audio[row[field]]["pcm_sha256"] in blocked_pcm):
                reasons.append(f"suspect_{field}_path_origin_or_pcm")
        if reasons:
            rejected.append({"row": position, "sample": row, "reasons": reasons,
                             "status": "quarantined_suspect_not_confirmed_bad_audio"})
        else:
            kept.append(position)
    return kept, rejected


def quarantine(audit, output):
    if output.exists():
        raise FileExistsError(output)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    audit_plan = read(audit / "plan.json")
    if read(audit / "summary.json")["status"] != "complete":
        raise ValueError("Silence audit is incomplete")
    run = Path(audit_plan["run"])
    parent = read(run / "plan.json")
    inputs = dict(audit_plan["inputs_sha256"])
    for path in (audit / "plan.json", audit / "summary.json", audit / "flagged.json",
                 audit / "measurements.jsonl"):
        inputs[str(path)] = sha256(path)
    flagged = read(audit / "flagged.json")
    measured = {row["audio"]: row for row in _read_manifest(audit / "measurements.jsonl")}
    if (set(item["audio"] for item in flagged) !=
            {path for path, item in measured.items() if item["review_reasons"]}
            or any(item != measured[item["audio"]] for item in flagged)):
        raise ValueError("Flags differ from complete measurements")
    pcm_path, = [Path(path) for path in parent["inputs_sha256"]
                 if path.endswith("/cpu/audio_audit.json")]
    inputs[str(pcm_path)] = parent["inputs_sha256"][str(pcm_path)]
    audio = read(pcm_path)["files"]
    if any(item["pcm_sha256"] != audio[path]["pcm_sha256"]
           or item["file_sha256"] != inputs[path] for path, item in measured.items()):
        raise ValueError("Audited PCM differs from parent")
    validation_paths = [Path(path) for path in parent["inputs_sha256"]
                        if Path(path).name == "val.jsonl"]
    protected = {}
    for path in validation_paths:
        inputs[str(path)] = parent["inputs_sha256"][str(path)]
        for row in _read_manifest(path):
            protected[row["audio"], row.get("ref_audio"), row["text"]] = row
            for field in ("audio", "ref_audio"):
                if row.get(field):
                    inputs[row[field]] = parent["inputs_sha256"][row[field]]
    excluded, evaluation_hashes = load_exclusions(ROOT / "eval_cases")
    inputs.update(evaluation_hashes)
    for function in (quarantine, audit_mix, pcm_conflicts, sha256, _read_manifest):
        path = inspect.getfile(function)
        inputs[path] = sha256(path)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    train_path, val_path = run / "train.jsonl", run / "val.jsonl"
    rows = _read_manifest(train_path)
    positions, removed = exclude_flagged(rows, flagged, audio)
    retained = [rows[position - 1] for position in positions]
    checks = audit_mix(retained, list(protected.values()), excluded, max_dur=30)
    pcm = pcm_conflicts(retained, list(protected.values()), audio)
    if not retained or checks["violations"] or checks["origins_above_3x"] or any(pcm.values()):
        raise ValueError("Candidate still has metadata/PCM conflicts")
    if exclude_flagged(retained, flagged, audio)[1]:
        raise ValueError("Suspect audio leaked back into candidate")
    summary, shares = describe_mix(retained)
    for lang, values in shares.items():
        values["requested"] = parent["language_shares"][lang]["actual"]
    output.mkdir(parents=True, exist_ok=False)
    lines = [line for line in train_path.read_bytes().splitlines(keepends=True) if line.strip()]
    (output / "train.jsonl").write_bytes(b"".join(lines[position - 1] for position in positions))
    (output / "val.jsonl").write_bytes(val_path.read_bytes())
    _write_jsonl(removed, output / "excluded.jsonl")
    save_new(output / "plan.json", {
        "inputs_sha256": inputs, "parent_run": str(run), "selected_suspect_files": len(flagged),
        "authorization": "User: 原则就是把这些有问题的，以及疑似有问题的数据都删掉",
        "policy": "Exclude every audit flag plus target/ref path, origin and exact PCM reuse; "
                  "no ASR-based automatic pardon or claim of confirmed defects.",
        "source_mutations": False, "retained_rows_byte_identical": True,
    })
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during quarantine: {path}")
    result = {
        "status": "candidate_quarantine_complete", "before_rows": len(rows),
        "excluded_rows": len(removed), "train": summary, "language_shares": shares,
        "validation_rows": len(_read_manifest(val_path)), "protected_validation_rows": len(protected),
        "checks": checks, "pcm": pcm, "source_mutations": False,
        "training_approved": False, "training_started": False, "production_changed": False,
        "outputs_sha256": {name: sha256(output / name)
                           for name in ("train.jsonl", "val.jsonl", "excluded.jsonl", "plan.json")},
        "note": "All known/suspected flags excluded per user; originals and historical evidence kept. "
                "Candidate preparation does not assert native quality or prove a training benefit.",
    }
    save_new(output / "mix.json", result)
    print(json.dumps({"excluded_rows": len(removed), "train": summary}, ensure_ascii=False), flush=True)
    return result


def review(audit, output):
    if output.exists():
        raise FileExistsError(output)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    audit_plan = read(audit / "plan.json")
    if read(audit / "summary.json")["status"] != "complete":
        raise ValueError("Silence audit is incomplete")
    run = Path(audit_plan["run"])
    inputs = {str(path): sha256(path) for path in
              (audit / "plan.json", audit / "summary.json", audit / "flagged.json")}
    for path in (run / "plan.json", run / "train.jsonl"):
        inputs[str(path)] = audit_plan["inputs_sha256"][str(path)]
    parent = read(run / "plan.json")
    rows = _read_manifest(run / "train.jsonl")
    items = read(audit / "flagged.json")
    tasks = []
    for item in items:
        path = item["audio"]
        if item["file_sha256"] != audit_plan["inputs_sha256"][path]:
            raise ValueError("Flagged audio differs from frozen audit")
        inputs[path] = item["file_sha256"]
        langs = {use["lang"] for use in item["uses"]}
        if len(langs) != 1 or not langs <= {"id", "th", "tl", "zh", "en"}:
            raise ValueError(f"Ambiguous language: {path}")
        texts = sorted({rows[use["row"] - 1]["text"] for use in item["uses"]
                        if use["role"] == "target"})
        tasks.append((item, langs.pop(), texts))
    model_path = Path(parent["asr_path"])
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            inputs[str(path)] = parent["inputs_sha256"][str(path)]
    from faster_whisper import WhisperModel
    for function in (review, save_new, sha256, _norm, _read_manifest, WhisperModel):
        path = inspect.getfile(function)
        inputs[path] = sha256(path)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "plan.json", {
        "inputs_sha256": inputs, "asr_path": str(model_path), "workers": 4,
        "original_audio_budget": len(tasks),
        "gain_diagnostic_budget": sum(item["gain_diagnostic"] is not None for item in items),
        "vad_filter": False, "initial_prompt": None, "temperature": 0, "beam_size": 5,
        "condition_on_previous_text": False, "training_started": False,
        "note": "VAD bypass is local to this data diagnostic, never a change to frozen evaluation. "
                "ASR output/errors do not certify transcript correctness, speech boundaries or language quality.",
    })
    model = WhisperModel(str(model_path), device="cuda", compute_type="int8_float16",
                         num_workers=4, local_files_only=True)
    if model.model.device != "cuda":
        raise RuntimeError("GPU ASR required; no CPU fallback")
    save_new(output / "runtime.json", {
        "device": model.model.device, "compute_type": model.model.compute_type,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("faster-whisper", "ctranslate2", "soundfile")},
    })

    def scan(task):
        item, lang, texts = task
        result = {"audio": item["audio"], "file_sha256": item["file_sha256"],
                  "uses": item["uses"], "review_reasons": item["review_reasons"],
                  "original": transcribe(model, item["audio"], lang, texts)}
        if item["gain_diagnostic"]:
            audio, rate = sf.read(item["audio"], dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError("Invalid diagnostic audio")
            gain = item["gain_diagnostic"]["gain"]
            result["gain_diagnostic"] = {"gain": gain,
                **transcribe(model, audio * gain, lang, texts)}
        if sha256(item["audio"]) != item["file_sha256"]:
            raise ValueError("Audio changed during ASR")
        return result

    results = []
    with (output / "items.jsonl").open("x", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for count, result in enumerate(executor.map(scan, tasks), 1):
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"ASR diagnostic {count}/{len(tasks)}", flush=True)
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during ASR: {path}")
    save_new(output / "summary.json", {
        "status": "complete", "files": len(results),
        "gain_diagnostics": sum("gain_diagnostic" in row for row in results),
        "empty_original_transcripts": sum(not row["original"]["hyp"].strip() for row in results),
        "original_files_unchanged": True, "training_started": False, "tts_started": False,
        "source_manifests_changed": False, "production_changed": False,
        "native_quality": "unverified", "automatic_exclusions": 0,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", action="store_true",
                        help="Exclude all audit flags and their reuse; no ASR or training")
    args = parser.parse_args()
    (quarantine if args.quarantine else review)(args.audit.resolve(), args.output.resolve())
