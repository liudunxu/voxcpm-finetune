"""Archive reviewed DIS A/B clips as development-only candidates, never training rows."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shutil

import soundfile as sf


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wav_asset(content, folder):
    info = sf.info(io.BytesIO(content))
    if info.frames <= 0 or info.format != "WAV":
        raise ValueError("Expected a nonempty original WAV; do not silently convert conditioning")
    digest = hashlib.sha256(content).hexdigest()
    return {"status": "available", "clip": f"{folder}/{digest}.wav", "audio_sha256": digest,
            "duration_seconds": info.duration, "sample_rate": info.samplerate, "channels": info.channels}


def collect(pack, output):
    pack, output = Path(pack).resolve(), Path(output).resolve()
    feedback_path = pack / "feedback.json"
    feedback_bytes = feedback_path.read_bytes()
    feedback = json.loads(feedback_bytes)
    feedback_sha = hashlib.sha256(feedback_bytes).hexdigest()
    pack_id = feedback["pack_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pack_id):
        raise ValueError("Invalid pack ID")

    def frozen_path(name):
        path = (pack / name).resolve()
        expected = feedback["frozen_artifacts_sha256"].get(name)
        if not path.is_relative_to(pack) or not expected or sha256(path) != expected:
            raise ValueError(f"Frozen artifact missing or changed: {name}")
        return path

    rows, clips, conditioning = [], {}, []
    seen = set()
    for case in feedback["cases"]:
        for arm, take in case["takes"].items():
            if take.get("status") != "无明显异常":
                continue
            code = f"{case['code']}-{arm}"
            if not re.fullmatch(r"[A-Za-z0-9_-]+", code) or code in seen:
                raise ValueError("Invalid or duplicate audition code")
            seen.add(code)
            audio = frozen_path(f"audio/{code}.wav")
            audio_sha = sha256(audio)
            if audio_sha != take["audio_sha256"]:
                raise ValueError(f"Feedback/audio mismatch: {code}")
            request_path = frozen_path(f"evidence/{code}-request.json")
            request = json.loads(request_path.read_text(encoding="utf-8"))
            if not request.get("text") or not request.get("language"):
                raise ValueError(f"Missing target text/language: {code}")
            response_name = f"evidence/{code}-response.json"
            response = (json.loads(frozen_path(response_name).read_text(encoding="utf-8"))
                        if response_name in feedback["frozen_artifacts_sha256"] else {})
            inputs = {}
            for role in ("reference", "prompt"):
                encoded = request.get(f"{role}_audio_base64")
                if encoded:
                    content = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
                    inputs[role] = wav_asset(content, "conditioning")
                    clips[inputs[role]["clip"]] = content
                else:
                    inputs[role] = {"status": "voice_id_only_not_restored" if request.get("voice_id")
                                    else "not_supplied"}
            preprocessing = (response.get("adaptive_params") or {}).get("ref_preprocess") or {}
            conditioning.append({
                "pack_id": pack_id, "audition_code": code, "audio_sha256": audio_sha,
                "request_sha256": sha256(request_path), "feedback_sha256": feedback_sha,
                "response_sha256": feedback["frozen_artifacts_sha256"].get(response_name),
                "stage": "client_request", "training_eligible": False,
                "gpu_preprocessed_audio_persisted": False, "replay_complete": False,
                "reference_identity_verified": False, "reference_language": "unverified",
                **inputs,
                "requested_parameters": {key: request[key] for key in (
                    "text", "language", "seed", "cfg_value", "inference_timesteps", "model_id",
                    "control_instruction", "prompt_text", "reference_prompt_text", "voice_id",
                    "denoise", "normalize", "ref_level_match", "ref_active_target_db", "ref_quiet_boost",
                    "ref_dereverb", "reference_edge_trim", "reference_internal_cleanup",
                    "enable_multi_speaker_check", "noise_gate", "trim_leading_silence", "trim_silence_vad",
                    "voxcpm_cfg_auto", "voxcpm_steps_auto", "voxcpm_adaptive", "speed",
                    "retry_badcase", "max_generation_attempts", "best_of", "text_regen", "quality_retry"
                ) if key in request},
                "effective_control_instruction": response.get("control_instruction"),
                "prompt_demoted_to_ref": response.get("prompt_demoted_to_ref"),
                "reported_reference_preprocessing": {key: preprocessing[key] for key in (
                    "method", "dereverb", "dominant_speaker", "level_gain_db", "vocalization_removed"
                ) if key in preprocessing},
            })
            info = sf.info(audio)
            if info.frames <= 0:
                raise ValueError(f"Empty audio: {code}")
            clip = f"audio/{audio_sha}.wav"
            severe = take.get("automatic_severe_issues")
            text_qc = take.get("automatic_text_qc") or {}
            rows.append({
                "pack_id": pack_id, "audition_code": code, "clip": clip,
                "audio_sha256": audio_sha, "duration_seconds": info.duration,
                "sample_rate": info.samplerate, "channels": info.channels,
                "target_text": request["text"], "lang": request["language"],
                "source_kind": "synthetic_tts", "split": "development_only",
                "training_eligible": False, "language_quality": "unverified",
                "transcript_verified": False, "speaker_identity_verified": False,
                "training_authorization": "unverified",
                "human_acoustic_status": take["status"],
                "pair_preference": case.get("submitted_preference"),
                "pair_note": case.get("raw_note"),
                "automatic_severe_issues": severe,
                "automatic_text_qc_status": text_qc.get("status", "unknown"),
                "candidate_status": ("acoustics_only" if severe == [] and text_qc.get("status") == "pass"
                                     else "needs_qc_review"),
                "requested_model_id": request.get("model_id"),
                "loaded_weights_sha256": None,
                "effective_generation": {key: take.get("effective_generation", {}).get(key)
                                         for key in ("seed", "cfg_value", "inference_timesteps")},
                "source_pack": str(pack), "feedback_sha256": feedback_sha,
                "request_sha256": sha256(request_path),
            })
            clips[clip] = audio
    if not rows:
        return rows
    target = output / f"{pack_id}-{feedback_sha[:12]}" / "annotations.jsonl"
    manifests = {target: rows, target.with_name("conditioning.jsonl"): conditioning}
    contents = {path: "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
                for path, records in manifests.items()}
    for path, content in contents.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Candidate snapshot already exists with different annotations")
    for clip, source in clips.items():
        destination = output / clip
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if isinstance(source, bytes):
                destination.write_bytes(source)
            else:
                shutil.copyfile(source, destination)
        if sha256(destination) != destination.stem:
            raise ValueError(f"Archive hash mismatch: {destination.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for path, content in contents.items():
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "work/for_train/ab_candidates")
    args = parser.parse_args()
    for pack in args.packs:
        rows = collect(pack, args.output)
        print(f"{pack.name}: {len(rows)} acoustic-only candidates; 0 approved training rows")
