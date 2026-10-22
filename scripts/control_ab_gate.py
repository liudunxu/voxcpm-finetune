"""Freeze the October 22 control-prefix experiment; CPU checks only, no downloads."""
import argparse
from collections import Counter
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.data.registry import get_source
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, DATA_RAW, ROOT, VOXCPM_REPO, env
from voxft.train.yaml_builder import steps_for_epochs

from holdout_eval import check_inputs
from holdout_texts import REVISION, parse_tsv, select_texts
from incremental_mix import source_row, target_text, validate_mix
from quality_probe import write_json
from ref_ab import memberships, runtime_environment, sha256
from ref_ab_diagnostics import expand_references
from training_pool import inspect_audio

WORK = CHECKPOINT_DIR.parent / "control_ab_20261022"
BUNDLE = CHECKPOINT_DIR.parent / "review_20261022/control_ab"
LANGS = ("th", "tl", "id")
SEEDS = [42, 43, 44, 45, 46]
CONTROLS = ("angry and emotionally tense", "sad and emotionally weighted",
            "bright and excited")
ANCESTRY = ("r20_emofix", "r21_nothaiser", "r22_idboost", "r23_thtlboost")


def verify_arms(off, on):
    if not on or len(off) != len(on):
        raise ValueError("Incomplete paired training data")
    for plain, controlled in zip(off, on):
        expected = {**controlled, "text": target_text(controlled), "control": ""}
        if plain != expected:
            raise ValueError("A/B changes more than control prefixes")
        if controlled.get("control"):
            if (not (controlled.get("emotion_verified") is True
                     or controlled.get("control_verified") is True)
                    or not controlled["text"].startswith(f"({controlled['control']})")):
                raise ValueError("Unverified or inconsistent control prefix")


def pcm_conflicts(train, validation, audio):
    def digests(rows):
        return {audio[row[field]]["pcm_sha256"] for row in rows
                for field in ("audio", "ref_audio") if row.get(field)}

    shared = sorted(digests(train) & digests(validation))
    same_reference = [row["audio"] for row in train if row.get("ref_audio")
                      and audio[row["audio"]]["pcm_sha256"]
                      == audio[row["ref_audio"]]["pcm_sha256"]]
    exposures = Counter(audio[row["audio"]]["pcm_sha256"] for row in train)
    return {"train_validation_pcm": shared, "identical_target_reference": same_reference,
            "pcm_exposure_above_3": {key: count for key, count in exposures.items() if count > 3}}


def verify_provenance(row, raw):
    source = get_source(source_row(row)["source_id"])
    original = raw.get(str(Path(row["origin_audio"]).resolve()))
    if original is None:
        raise ValueError(f"Missing raw provenance: {row['origin_audio']}")
    if row.get("speaker_verified") is True:
        expected = f"{original.get('speaker_namespace') or source.id}:{original.get('speaker')}"
        if not source.has_speaker or expected != row["speaker"]:
            raise ValueError(f"Verified speaker differs from source metadata: {row['audio']}")


def trim_replay(rows):
    requested = {"zh": 0.04, "en": 0.01}
    target_seconds = sum(row["duration"] for row in rows if row["lang"] in LANGS)
    budget = target_seconds / (1 - sum(requested.values()))
    keep = {position for position, row in enumerate(rows) if row["lang"] not in requested}
    rng = random.Random(42)
    for lang, share in requested.items():
        positions = [position for position, row in enumerate(rows) if row["lang"] == lang]
        rng.shuffle(positions)
        remaining = budget * share
        for position in positions:
            if rows[position]["duration"] <= remaining:
                keep.add(position)
                remaining -= rows[position]["duration"]
        if not any(rows[position]["lang"] == lang for position in keep):
            raise ValueError(f"No replay examples fit the {lang} duration budget")
    selected = [row for position, row in enumerate(rows) if position in keep]
    seconds = Counter()
    for row in selected:
        seconds[row["lang"]] += row["duration"]
    total = sum(seconds.values())
    return selected, {
        "before_rows": len(rows), "after_rows": len(selected), "requested_replay_shares": requested,
        "actual_seconds": dict(seconds), "actual_shares": {lang: value / total for lang, value in seconds.items()},
        "note": "All ID/TH/TL targets and existing refs retained; only replay rows reduced identically in both arms.",
    }


def choose_references(rows, used_keys, used_pcm, audio):
    candidates = sorted(rows, key=lambda row: (abs(row["duration"] - 5), row["audio"]))
    selected, seen = [], set()
    for row in candidates:
        if not 3 <= row["duration"] <= 10 or memberships([row]) & used_keys:
            continue
        digest = audio[row["audio"]]["pcm_sha256"]
        if digest in used_pcm or digest in seen:
            continue
        selected.append(row)
        seen.add(digest)
        if len(selected) == 3:
            return selected
    raise ValueError("Need three traceable, training-isolated reference recordings per source language")


def build_cases(selected, references):
    cases = []
    for lang in LANGS:
        rows = selected[lang]
        if len(rows) != 30 or len({evaluation._norm(row["text"]) for row in rows}) != 30:
            raise ValueError(f"Need 30 independent texts: {lang}")
        counts = Counter()
        for position, row in enumerate(rows):
            ref_lang = "en" if position % 5 == 4 else "zh"
            pool = references[ref_lang]
            reference = pool[counts[ref_lang] % len(pool)]
            counts[ref_lang] += 1
            sentence = f"fresh_{lang}_{row['source_sentence_id']}"
            for condition, control in (("plain", ""), ("control", CONTROLS[(position // 5) % 3])):
                cases.append({
                    **row, "case_id": f"{sentence}_{condition}", "sentence_id": sentence,
                    "lang": lang, "ref_lang": ref_lang, "ref_audio": reference,
                    "control": control, "condition": condition, "cohort": "independent_text",
                    "source_dataset": "google/fleurs", "source_revision": REVISION,
                    "source_split": "test", "source_license": "CC-BY-4.0", "numeric": False,
                    "note": "Unseen in audited local text pools, not proof of pretraining novelty. "
                            "Read-speech text; requested emotion is not a verified performance label. "
                            "Reference identity novelty and native-language quality unverified.",
                })
    return cases


def regression_cases(manifests):
    directory = ROOT / "eval_cases"
    original = manifests[directory / "omnivoice_prod_v2.jsonl"]
    short = manifests[directory / "short_cue_probe_20261020.jsonl"]
    selected = []
    for lang in LANGS:
        rows = [row for row in original if row["lang"] == lang and row.get("ref_lang") in {"zh", "en"}]
        selected.append((next(row for row in rows if evaluation._is_numeric(row)), "numeric"))
        selected.append((next(row for row in rows if "code-switch" in row.get("note", "")), "mixed"))
        selected.append((min((row for row in short if row["lang"] == lang and row["ref_lang"] == "zh"),
                             key=lambda row: len(row["text"])), "short_probe"))
    probe = manifests[directory / "id_prosody_probe_20261020.jsonl"]
    selected.append((next(row for row in probe if row["case_id"] == "id_ms_colloquial_en"),
                     "historical_severe_continuation"))
    cases = []
    for row, kind in selected:
        sentence = f"regression_{row['case_id']}"
        for condition, control in (("plain", ""), ("control", CONTROLS[0])):
            cases.append({
                **{key: value for key, value in row.items() if key != "seed"},
                "case_id": f"{sentence}_{condition}", "sentence_id": sentence,
                "ref_audio": str((directory / row["ref_audio"]).resolve()),
                "control": control, "condition": condition, "cohort": "historical_regression",
                "regression_kind": kind, "historical_case_id": row["case_id"],
                "note": "Historical stress text; may be training-exposed. Not an independent quality gate. "
                        "Original seed replaced by the shared five-seed set, including severe seed 43.",
            })
    return cases


def verify_rng():
    sys.path.insert(0, str(VOXCPM_REPO / "src"))
    from voxcpm.training import Accelerator

    default = inspect.signature(Accelerator.__init__).parameters["seed"].default
    samples = []
    for _ in range(2):
        Accelerator(amp=False)
        samples.append((random.random(), float(np.random.random()), torch.rand(4).tolist()))
    if default != 42 or samples[0] != samples[1]:
        raise ValueError("Official training RNG initialization differs")
    return {"seed": default, "repeated_initialization_equal": True,
            "bitwise_training_determinism": "not_claimed"}


def prepare(work, bundle=BUNDLE):
    work.mkdir(parents=True, exist_ok=False)
    runtime_environment(work)
    original = check_inputs(bundle)
    inputs = dict(original["inputs_sha256"])

    def freeze(path):
        path = Path(path)
        if str(path) not in inputs:
            inputs[str(path)] = sha256(path)
        return path

    freeze(bundle / "plan.json")
    train = _read_manifest(freeze(bundle / "on.jsonl"))
    plain = _read_manifest(freeze(bundle / "off.jsonl"))
    validation = _read_manifest(freeze(bundle / "validation.jsonl"))
    verify_arms(plain, train)
    if sorted(evaluation.AUTO_DETECT_LANGS) != ["tl"]:
        raise ValueError("ASR language-hint policy changed")
    config = yaml.safe_load(freeze(bundle / "on.yaml").read_text(encoding="utf-8"))
    base = Path(config["pretrained_path"])
    candidate = CHECKPOINT_DIR / "lora_omni5_r23_thtlboost_cont/latest"
    state = json.loads(freeze(candidate / "training_state.json").read_text(encoding="utf-8"))
    if state["step"] != 2909:
        raise ValueError("Unexpected r23 checkpoint step")
    for directory in (base, candidate):
        for path in sorted(directory.glob("*")):
            if path.is_file():
                freeze(path)
    for name in ("model.safetensors", "audiovae.pth"):
        path = base / name
        if len(path.resolve().name) != 64 or inputs[str(path)] != path.resolve().name:
            raise ValueError(f"Base weight differs from cached content-addressed blob: {path}")
    for directory in (ROOT / "src", VOXCPM_REPO / "src"):
        for path in sorted(directory.rglob("*.py")):
            freeze(path)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        freeze(path)
    freeze(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py")
    for name in ("uv.lock", "pyproject.toml"):
        freeze(ROOT / name)

    excluded = set()
    manifests = {}
    paths = (set(DATA_PROCESSED.glob("*/train.jsonl"))
             | set(DATA_PROCESSED.glob("*/val.jsonl"))
             | set(DATA_RAW.glob("*/manifest.jsonl"))
             | set((ROOT / "eval_cases").glob("*.jsonl")))
    for path in sorted(paths):
        rows = _read_manifest(freeze(path))
        manifests[path] = rows
        excluded.update(evaluation._norm(target_text(row)) for row in rows if row.get("text"))
    for path in sorted(evaluation.EVAL_DIR.glob("*.json")):
        report = json.loads(freeze(path).read_text(encoding="utf-8"))
        excluded.update(evaluation._norm(target_text(row)) for row in report.get("items", [])
                        if row.get("text"))
    ancestry = [source_row(row) for name in ANCESTRY
                for row in manifests[DATA_PROCESSED / name / "train.jsonl"]]
    known_eval = {evaluation._norm(target_text(row))
                  for path, rows in manifests.items() if path.parent == ROOT / "eval_cases"
                  for row in rows if row.get("text")}
    validate_mix(train, validation, known_eval)
    train, replay = trim_replay(train)
    plain = [{**row, "text": target_text(row), "control": ""} for row in train]
    verify_arms(plain, train)
    for arm, rows in (("off", plain), ("on", train)):
        _write_jsonl(rows, work / f"{arm}.jsonl")
        freeze(work / f"{arm}.jsonl")
    for row in train:
        bound = len(row["text"].encode("utf-8")) + math.ceil(
            (row["duration"] + (row.get("ref_duration") or 0)) * 50) + 4
        if bound > config["max_batch_tokens"] // config["batch_size"]:
            raise ValueError("Control variant could trigger unequal training token filtering")
    raw = {str(Path(row["audio"]).resolve()): row
           for path, rows in manifests.items() if path.name == "manifest.jsonl"
           for row in rows}
    index = {str(Path(row["audio"]).resolve()): row
             for path, rows in manifests.items() if path.name in {"train.jsonl", "val.jsonl"}
             for row in rows}
    for row in expand_references(train, index):
        verify_provenance(row, raw)

    audio = {}
    for row in train + validation + ancestry:
        for field, duration_field in (("audio", "duration"), ("ref_audio", "ref_duration")):
            path = row.get(field)
            if not path or path in audio:
                continue
            freeze(path)
            audio[path] = inspect_audio({"audio": path, "duration": row[duration_field]})
            if len(audio) % 1000 == 0:
                print(f"PCM checked: {len(audio)}", flush=True)
    conflicts = pcm_conflicts(train, validation, audio)
    write_json(work / "audio_audit.json", {"files": audio, "conflicts": conflicts})
    freeze(work / "audio_audit.json")
    if any(conflicts.values()):
        raise ValueError(f"Audio isolation failed: {conflicts}")

    used_keys = memberships(train + ancestry)
    used_pcm = {audio[row[field]]["pcm_sha256"] for row in train + ancestry
                for field in ("audio", "ref_audio") if row.get(field)}
    old_references = {}
    for row in manifests[ROOT / "eval_cases/refrob_20261019_clean.jsonl"]:
        path = str((ROOT / "eval_cases" / row["ref_audio"]).resolve())
        if path in old_references:
            continue
        freeze(path)
        record = inspect_audio({"audio": path, "duration": sf.info(path).duration})
        old_references[path] = {
            **record, "matches_training_pcm": record["pcm_sha256"] in used_pcm,
            "matching_audited_paths": [name for name, entry in audio.items()
                                      if entry["pcm_sha256"] == record["pcm_sha256"]],
            "used_in_new_independent_set": False,
        }
    write_json(work / "old_reference_review.json", old_references)
    freeze(work / "old_reference_review.json")
    references = {}
    reference_audit = {}
    for lang in ("zh", "en"):
        rows = manifests[DATA_PROCESSED / f"fleurs_{lang}/val.jsonl"]
        for row in rows:
            if not 3 <= row["duration"] <= 10:
                continue
            freeze(row["audio"])
            if row["audio"] not in audio:
                audio[row["audio"]] = inspect_audio(row)
        chosen = choose_references(rows, used_keys, used_pcm, audio)
        references[lang] = [row["audio"] for row in chosen]
        for row in chosen:
            verify_provenance(row, raw)
            path = row["audio"]
            reference_audit[path] = {
                "file_sha256": inputs[path], **audio[path], "source_id": row["source_id"],
                "origin_audio": row["origin_audio"], "known_identity_overlap": False,
                "identity_novelty": "unverified",
            }
    selected, text_summary = {}, {}
    snapshot = Path(env("HF_HOME")) / "hub/datasets--google--fleurs/snapshots" / REVISION
    for lang in LANGS:
        source = get_source(f"fleurs_{lang}")
        path = freeze(snapshot / "data" / source.config / "test.tsv")
        selected[lang], text_summary[lang] = select_texts(
            parse_tsv(path.read_text(encoding="utf-8")), excluded, lang)
    cases = build_cases(selected, references)
    regressions = regression_cases(manifests)
    for case in regressions:
        freeze(case["ref_audio"])
    cases.extend(regressions)
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate frozen case IDs")
    _write_jsonl(cases, work / "cases.jsonl")
    freeze(work / "cases.jsonl")
    rng = verify_rng()
    configs = {}
    for arm in ("off", "on"):
        path = work / f"{arm}.jsonl"
        steps = steps_for_epochs(str(path), 1, config["batch_size"], config["grad_accum_steps"])
        checkpoint = work / f"checkpoint_{arm}"
        arm_config = {**config, "train_manifest": str(path), "num_iters": steps,
                      "max_steps": steps, "save_path": str(checkpoint),
                      "tensorboard": str(checkpoint / "logs")}
        config_path = work / f"{arm}_1ep.yaml"
        config_path.write_text(yaml.safe_dump(arm_config, allow_unicode=True), encoding="utf-8")
        freeze(config_path)
        configs[arm] = str(config_path)
    plan = {
        "status": "frozen_pending_baseline", "user_authorization_date": "2026-10-22",
        "base_path": str(base), "targets": {"base": "base", "r23": str(candidate)},
        "cases": len(cases), "independent_texts": 90, "regression_cases": len(regressions),
        "evaluation_seeds": SEEDS,
        "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
        "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
        "configs": configs, "epochs": 1, "num_iters": steps, "rng": rng,
        "text_selection": text_summary, "references": reference_audit,
        "replay": replay, "reference_language_shares": {"zh": 0.8, "en": 0.2},
        "training_rows": len(train), "training_controls": sum(bool(row.get("control")) for row in train),
        "source_licenses": {name: get_source(name).license for name in sorted({row["source_id"] for row in train})},
        "inputs_sha256": inputs, "audio_audit_sha256": sha256(work / "audio_audit.json"),
        "note": "User authorized baseline then gated paired training. First epoch only, no schedule extension. "
                "One shared base; fresh optimizer; no implicit resume or production promotion. "
                "Exact PCM and known identity/session isolation, not near-duplicate or unknown-speaker proof. "
                "Existing source metadata traced, not new certification of recording consent. "
                "No source audio download or native/acoustic quality certification.",
    }
    if shutil.disk_usage(work).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB free; no models started")
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Frozen input changed during preparation")
    write_json(work / "plan.json", plan)
    print(json.dumps({key: value for key, value in plan.items()
                      if key not in {"inputs_sha256", "references"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    prepare(args.work, args.bundle)
