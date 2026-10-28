"""Authorized cleaned-data, one-epoch candidate; preserve historical evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

import control_ab_run as frozen
import fresh_baseline_eval as baseline
from control_ab_fast_eval import cached_services, claim
from control_ab_gate import pcm_conflicts, verify_rng
from incremental_mix import audit_mix, load_exclusions, target_text
from voxft.data.pipeline import _read_manifest, _write_jsonl, dataset_summary
from voxft.data.registry import get_source
from voxft.train.launcher import preflight
from voxft.train.yaml_builder import steps_for_epochs

DATA = Path("/root/autodl-tmp")
QUARANTINE = DATA / "finetune_readiness_20261028/quarantine/r26"
OLD_RUN = DATA / "fresh_baseline_20261028/run"
RUN_NAME = "lora_joint3_r28_clean_1ep"
ENCODING = re.compile(r"(?:Ã[\x80-\xbf]|Â[\x80-\xbf]|â(?:€|[\x80-\xbf])|ðŸ|\ufffd)")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_rows(train, validation, audio):
    bad = [row for row in train + validation if ENCODING.search(target_text(row))]
    blocked_paths = {str(Path(row[key]).resolve()) for row in bad
                     for key in ("audio", "origin_audio") if row.get(key)}
    blocked_pcm = {audio[row["audio"]]["pcm_sha256"] for row in bad}
    cleaned, removed, prefixes = {}, [], []
    for split, rows in (("train", train), ("val", validation)):
        kept = []
        for position, row in enumerate(rows, 1):
            reasons = []
            if ENCODING.search(target_text(row)):
                reasons.append("encoding_artifact_in_text")
            for field, origin in (("audio", "origin_audio"), ("ref_audio", "ref_origin_audio")):
                if row.get(field) and (
                        any(str(Path(row[key]).resolve()) in blocked_paths
                            for key in (field, origin) if row.get(key))
                        or audio[row[field]]["pcm_sha256"] in blocked_pcm):
                    reasons.append(f"quarantined_{field}")
            if reasons:
                removed.append({"split": split, "row": position, "reasons": reasons, "sample": row})
                continue
            item = dict(row)
            if row.get("control"):
                if not row["text"].startswith(f"({row['control']})"):
                    raise ValueError("Control metadata/text mismatch")
                if not (row.get("emotion_verified") is True or row.get("control_verified") is True):
                    prefixes.append({"split": split, "row": position, "sample": row})
                    item["text"], item["control"] = target_text(row), ""
                    item.pop("control_inoculation", None)
            kept.append(item)
        if not kept:
            raise ValueError(f"Empty cleaned {split}")
        cleaned[split] = kept
    return cleaned, removed, prefixes


def candidate_config(template, work):
    config = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    expected = {"enable_lm": True, "enable_dit": True, "enable_proj": False,
                "r": 64, "alpha": 64, "dropout": 0.05}
    if config["lora"] != expected or config["learning_rate"] != 1e-4:
        raise ValueError("Historical template recipe changed")
    steps = steps_for_epochs(str(work / "train.jsonl"), 1, 2, 8, 1)
    return {**config, "train_manifest": str(work / "train.jsonl"),
            "val_manifest": str(work / "val.jsonl"), "batch_size": 2, "grad_accum_steps": 8,
            "num_iters": steps, "max_steps": steps, "warmup_steps": max(1, steps // 10),
            "save_interval": steps, "save_path": str(work / "checkpoint"),
            "tensorboard": str(work / "checkpoint/logs")}


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    baseline.checked(OLD_RUN)
    for path, digest in read_json(OLD_RUN / "comparison.json")["inputs_sha256"].items():
        if frozen.sha256(path) != digest:
            raise ValueError(f"Baseline report changed: {path}")
    quarantine = read_json(QUARANTINE / "mix.json")
    for name, digest in quarantine["outputs_sha256"].items():
        if frozen.sha256(QUARANTINE / name) != digest:
            raise ValueError(f"Quarantine input changed: {name}")
    audio_path = DATA / "finetune_readiness_20261028/cpu/audio_audit.json"
    original_hashes = read_json(QUARANTINE.parent / "inputs_sha256.json")
    if frozen.sha256(audio_path) != original_hashes[str(audio_path)]:
        raise ValueError("Audio audit changed")
    audio = read_json(audio_path)["files"]
    train, validation = (_read_manifest(QUARANTINE / f"{split}.jsonl") for split in ("train", "val"))
    cleaned, removed, prefixes = clean_rows(train, validation, audio)
    exclusions, inputs = load_exclusions(frozen.ROOT / "eval_cases")
    # Keep the entire old validation set protected, including newly excluded rows.
    audit = audit_mix(cleaned["train"], validation, exclusions, max_dur=30)
    pcm = pcm_conflicts(cleaned["train"], validation, audio)
    if audit["violations"] or audit["origins_above_3x"] or any(pcm.values()):
        raise ValueError(f"Unsafe cleaned candidate: {audit['violation_counts']}")
    template = frozen.ROOT / "configs/lora_omni5_r26_r18_inoc10.yaml"
    paths = [template, audio_path, QUARANTINE / "mix.json", QUARANTINE / "excluded.jsonl",
             QUARANTINE / "train.jsonl", QUARANTINE / "val.jsonl", OLD_RUN / "plan.json",
             OLD_RUN / "comparison.json", OLD_RUN / "cases.jsonl",
             OLD_RUN / "base_report.json", OLD_RUN / "r26_report.json"]
    for root in (frozen.ROOT / "src", frozen.ROOT / "scripts",
                 frozen.VOXCPM_REPO / "src", frozen.VOXCPM_REPO / "scripts"):
        paths.extend(root.rglob("*.py"))
        paths.extend(root.rglob("*.sh"))
    inputs.update({str(path): frozen.sha256(path) for path in paths})
    old_plan = read_json(OLD_RUN / "plan.json")
    for path, digest in old_plan["inputs_sha256"].items():
        if Path(path).is_relative_to(baseline.HUB):
            inputs[path] = digest
    for row in train + validation + _read_manifest(OLD_RUN / "cases.jsonl"):
        for field in ("audio", "ref_audio"):
            if row.get(field):
                inputs[row[field]] = audio[row[field]]["file_sha256"]
    for path, digest in inputs.items():
        if frozen.sha256(path) != digest:
            raise ValueError(f"Input changed: {path}")
    if shutil.disk_usage(work.parent).free < 6 * 1024**3:
        raise RuntimeError("Need at least 6 GiB for checkpoints and evaluation")
    work.mkdir(parents=True, exist_ok=False)
    for split, rows in cleaned.items():
        _write_jsonl(rows, work / f"{split}.jsonl")
    for name in ("cases.jsonl", "base_report.json", "r26_report.json"):
        shutil.copyfile(OLD_RUN / name, work / name)
    summary = dataset_summary(cleaned["train"])
    language_shares = {
        lang: {"requested": quarantine["language_shares"][lang]["actual"],
               "actual": sum(row["duration"] for row in cleaned["train"] if row["lang"] == lang)
                         / summary["seconds"]}
        for lang in sorted(summary["languages"])}
    baseline.save_new(work / "mix.json", {"train": summary, "language_shares": language_shares,
        "training_approved": True, "approval_scope": "User-authorized one-epoch candidate only",
        "source_manifests_unchanged": True, "verified_control_rows": 0,
        "acoustic_and_native_quality": "unverified"})
    config = candidate_config(template, work)
    config_path = work / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    frozen.require_fresh_checkpoint(config)
    issues = preflight(config_path, gpus=1)
    Path(config["save_path"]).rmdir()
    if any(not issue.startswith("警告") for issue in issues):
        raise ValueError(f"Training preflight failed: {issues}")
    for name in ("train.jsonl", "val.jsonl", "train.yaml", "mix.json", "cases.jsonl", "base_report.json", "r26_report.json"):
        inputs[str(work / name)] = frozen.sha256(work / name)
    sources = {}
    for source_id in sorted({row["source_id"] for row in cleaned["train"]}):
        source = get_source(source_id)
        if source.license not in {"CC0", "CC-BY-3.0", "CC-BY-4.0", "Apache-2.0"}:
            raise ValueError(f"Source outside existing allowlist: {source_id}")
        sources[source_id] = {"license": source.license, **dataset_summary(
            [row for row in cleaned["train"] if row["source_id"] == source_id])}
    plan = {
        **baseline.SETTINGS, "base_path": str(baseline.BASE),
        "targets": {"base": "base", "r26": str(baseline.R26), "r28": str(work / "checkpoint/latest")},
        "asr_path": str(baseline.WHISPER), "speaker_model_path": str(baseline.SPEAKER),
        "run_name": RUN_NAME, "config": str(config_path), "num_iters": config["num_iters"],
        "epochs": 1, "training_seed": 42, "rng_verification": verify_rng(), "training_authorized": True,
        "authorization": "User: 直接去掉有问题的训练集，然后按照你建议来训练吧; continued explicitly.",
        "max_workers": 4, "generation_budget": 900, "production_changed": False,
        "inputs_sha256": inputs, "train": summary, "language_shares": language_shares,
        "validation": dataset_summary(cleaned["val"]), "sources": sources,
        "audit": audit, "pcm": pcm, "preflight": issues,
        "inherited_excluded_rows": quarantine["excluded_rows"],
        "additional_excluded": removed, "removed_unverified_prefixes": prefixes,
        "note": "Cold-start one-epoch cleaned-data candidate, not a causal cleaning A/B against 3-epoch r26. "
                "Unverified neutral prefixes removed; zero verified control coverage is a limitation. "
                "Historical evaluation inputs/scores remain frozen, including known suspect text encoding. "
                "Fresh text set now serves development comparison; native quality unverified. "
                "No downloads, automatic retries, promotion, merge, or deletion of original data.",
    }
    baseline.save_new(work / "plan.json", plan)
    print(json.dumps({"train_rows": len(cleaned["train"]), "val_rows": len(cleaned["val"]),
                      "additional_excluded": len(removed), "prefixes_removed": len(prefixes),
                      "steps": config["num_iters"], "preflight": issues}, ensure_ascii=False), flush=True)


def tasks(cases):
    return [(f"r28_{position:04d}_{seed}", case, seed)
            for position, case in sorted(enumerate(cases), key=lambda item: item[1]["lang"] != "id")
            for seed in frozen.SEEDS]


def checked(work, trained=False):
    plan = frozen.check_inputs(work)
    if not plan["training_authorized"] or plan["epochs"] != 1 or plan["generation_budget"] != 900:
        raise ValueError("Candidate scope changed")
    if trained:
        state = read_json(work / "trained.json")
        if state["step"] != plan["num_iters"]:
            raise ValueError("Training incomplete")
        for path, digest in state["inputs_sha256"].items():
            if frozen.sha256(path) != digest:
                raise ValueError(f"Checkpoint changed: {path}")
    return plan


def worker(work, index):
    plan = checked(work, trained=True)
    directory = work / f"eval/worker{index}"
    directory.mkdir(parents=True, exist_ok=False)
    frozen.runtime_environment(directory / "runtime")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VOXFT_WHISPER_MODEL_LARGE=plan["asr_path"],
                      VOXFT_SPK_EMB_MODEL=plan["speaker_model_path"],
                      TORCHINDUCTOR_CACHE_DIR=str(directory / "runtime/inductor"))
    cases = _read_manifest(work / "cases.jsonl")
    frozen.evaluation.EVAL_DIR = directory / "raw_reports"
    frozen.evaluation.infer.CHECKPOINT_DIR = directory
    with cached_services(cases):
        baseline.save_new(directory / "runtime.json", baseline.gpu_services(plan, cases, "r28"))
        for count, (identifier, case, seed) in enumerate(tasks(cases)[index::4], 1):
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError("Less than 4 GiB free; preserve evidence")
            if not claim(work / "eval/claims", identifier):
                raise RuntimeError(f"Refusing retry: {identifier}")
            report = frozen.evaluation.evaluate(
                plan["targets"]["r28"], "th", [case], base=plan["base_path"], seeds=[seed],
                cfg_value=plan["cfg_value"], inference_timesteps=plan["inference_timesteps"])
            row, = report["items"]
            if row["seed"] != seed or any(row.get(k) != v for k, v in case.items()):
                raise ValueError(f"Unexpected item: {identifier}")
            report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"])}
            baseline.save_new(work / "eval/items" / f"{identifier}.json", report)
            frozen.write_json(directory / "progress.json", {"completed": count, "last_task": identifier,
                                                             "lang": case["lang"], "time": time.time()})
            print(f"DONE {identifier} completed={count}/225", flush=True)
    checked(work, trained=True)


def collect(work):
    plan = checked(work, trained=True)
    cases = _read_manifest(work / "cases.jsonl")
    expected = {identifier: (case, seed) for identifier, case, seed in tasks(cases)}
    files = {path.stem: path for path in (work / "eval/items").glob("*.json")}
    if files.keys() != expected.keys():
        raise ValueError("Incomplete or duplicate evaluation")
    for identifier, path in files.items():
        report = read_json(path)
        row, = report["items"]
        case, seed = expected[identifier]
        if row["seed"] != seed or any(row.get(k) != v for k, v in case.items()):
            raise ValueError(f"Case changed: {identifier}")
        if report["wav_sha256"] != {row["wav"]: frozen.sha256(row["wav"])}:
            raise ValueError(f"WAV changed: {identifier}")
    frozen.evaluation.EVAL_DIR = work / "eval/raw_reports"
    report = frozen.evaluation.merge_reports([str(files[key]) for key in sorted(files)])
    frozen.verify_report(report, plan, cases, "r28")
    report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"]) for row in report["items"]}
    baseline.save_new(work / "r28_report.json", report)
    frozen.collect(work, ("base", "r26", "r28"), work / "comparison.json")


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(work):
    plan = checked(work)
    config = yaml.safe_load(Path(plan["config"]).read_text(encoding="utf-8"))
    frozen.require_fresh_checkpoint(config)
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU busy; refusing overlapping training")
    if not claim(work, "started"):
        raise FileExistsError("Refusing implicit restart")
    frozen.runtime_environment(work / "runtime")
    processes, trainer, bridge = [], None, None

    def status(stage, **extra):
        frozen.write_json(work / "status.json", {"stage": stage, "time": time.time(),
            "expected_training_steps": plan["num_iters"], "expected_evaluation_items": 900,
            "completed_evaluation_items": len(list((work / "eval/items").glob("*.json"))),
            "training_pid": trainer.pid if trainer and trainer.poll() is None else None,
            "worker_pids": [p.pid for p in processes if p.poll() is None],
            "production_changed": False, **extra})

    try:
        if shutil.disk_usage(work).free < 6 * 1024**3:
            raise RuntimeError("Insufficient disk before training")
        for line in Path("/root/voxcpm-finetune/.env").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in {"WANDB_API_KEY", "WANDB_PROJECT"}:
                os.environ.setdefault(key.strip(), value.strip())
        with (work / "train.log").open("x", encoding="utf-8") as log, \
                (work / "wandb.log").open("x", encoding="utf-8") as bridge_log:
            trainer = subprocess.Popen([sys.executable, "-X", "utf8", "-u",
                str(frozen.VOXCPM_REPO / "scripts/train_voxcpm_finetune.py"),
                "--config_path", plan["config"]], cwd=frozen.VOXCPM_REPO, stdout=log, stderr=subprocess.STDOUT)
            if os.environ.get("WANDB_API_KEY"):
                bridge = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "-m",
                    "voxft.train.tb_wandb_bridge", config["tensorboard"], RUN_NAME],
                    cwd=frozen.ROOT, stdout=bridge_log, stderr=subprocess.STDOUT)
            while trainer.poll() is None:
                status("training")
                if shutil.disk_usage(work).free < 4 * 1024**3:
                    raise RuntimeError("Disk guard during training")
                time.sleep(10)
            if trainer.returncode:
                raise RuntimeError(f"Training exited {trainer.returncode}; see train.log")
        stop(bridge)
        latest = Path(config["save_path"]) / "latest"
        state = read_json(latest / "training_state.json")
        if state["step"] != plan["num_iters"]:
            raise ValueError("Training did not reach planned step")
        baseline.save_new(work / "trained.json", {"step": state["step"], "time": time.time(),
            "inputs_sha256": {str(p): frozen.sha256(p) for p in latest.iterdir() if p.is_file()}})
        checked(work, trained=True)
        for name in ("claims", "items"):
            (work / "eval" / name).mkdir(parents=True, exist_ok=False)
        for index in range(4):
            free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"], text=True).strip())
            if free < 12 * 1024:
                raise RuntimeError("Insufficient GPU memory for four workers")
            with (work / f"eval/worker{index}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                    "worker", "--work", str(work), "--index", str(index)],
                    cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            while not (work / f"eval/worker{index}/progress.json").exists() and process.poll() is None:
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError("Evaluation worker failed during startup")
                status("starting_evaluation")
                time.sleep(2)
            if process.poll() not in (None, 0):
                raise RuntimeError(f"Worker {index} failed")
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Evaluation failed; no automatic retry")
            status("evaluating")
            time.sleep(10)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Evaluation failed; preserve outputs")
        collect(work)
        status("complete", training_step=state["step"])
    except BaseException as error:
        status("failed", error=str(error))
        raise
    finally:
        stop(trainer)
        stop(bridge)
        for process in processes:
            stop(process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == "worker":
        if args.index is None:
            parser.error("worker requires --index")
        worker(args.work, args.index)
    else:
        {"prepare": prepare, "run": run}[args.action](args.work)
