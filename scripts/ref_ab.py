"""r10：固定预算的同人 ref 开/关对照；仅在远端运行，不升级生产。"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, ROOT, VOXCPM_REPO

from quality_probe import SEEDS, SOURCES, write_json

ARMS = {"off": "lora_omni5_r10_ref_off", "on": "lora_omni5_r10_ref_on"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def without_ref(row):
    return {key: value for key, value in row.items() if not key.startswith("ref_")}


def memberships(rows):
    result = set()
    for row in rows:
        for key in ("audio", "ref_audio", "origin_audio", "ref_origin_audio"):
            if row.get(key):
                result.add(("audio", str(Path(row[key]).resolve())))
        if row.get("speaker_verified") is True:
            result.add(("speaker", row["speaker"]))
        if row.get("session"):
            result.add(("session", row.get("source_id", ""), row["session"]))
    return result


def checked_pairs(rows, excluded_texts=frozenset()):
    from voxft.eval import _norm

    index = {str(Path(row["audio"]).resolve()): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("CV22 source contains duplicate target paths")
    pairs = []
    for row in rows:
        if not row.get("ref_audio"):
            continue
        reference = index.get(str(Path(row["ref_audio"]).resolve()))
        if (reference is None or row.get("speaker_verified") is not True
                or reference.get("speaker_verified") is not True
                or not row.get("speaker")
                or row["speaker"] != reference.get("speaker")
                or row["speaker"] != row.get("ref_speaker")
                or row.get("source_id") != reference.get("source_id")
                or row.get("ref_origin_audio") != reference.get("origin_audio")
                or not row.get("origin_audio")
                or Path(row["origin_audio"]).resolve() == Path(reference["origin_audio"]).resolve()
                or not math.isfinite(float(reference["duration"]))
                or not math.isfinite(float(row["ref_duration"]))
                or not 3 <= float(reference["duration"]) <= 10
                or abs(float(row["ref_duration"]) - float(reference["duration"])) > 0.01):
            raise ValueError(f"Invalid same-speaker, different-origin pair: {row['audio']}")
        if (_norm(row["text"]) != _norm(reference["text"])
                and _norm(row["text"]) not in excluded_texts
                and _norm(reference["text"]) not in excluded_texts):
            pairs.append(dict(row))
    return pairs


def runtime_environment(work):
    for key, directory in {
        "WANDB_DIR": "wandb", "WANDB_CACHE_DIR": "wandb_cache",
        "WANDB_CONFIG_DIR": "wandb_config", "TMPDIR": "tmp",
        "MPLCONFIGDIR": "matplotlib", "NUMBA_CACHE_DIR": "numba",
        "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda_cache",
    }.items():
        path = work / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.setdefault("XDG_CACHE_HOME", str(CHECKPOINT_DIR.parent / "cache"))
    os.environ.setdefault("TORCH_HOME", str(CHECKPOINT_DIR.parent / "torch_home"))
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONHASHSEED"] = "42"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def prepare(work):
    import numpy as np
    import soundfile as sf
    import yaml
    from voxft import eval as evaluation
    from voxft.data.pipeline import _read_manifest, _write_jsonl, dataset_summary, mix_manifests
    from voxft.data.registry import get_source
    from voxft.train.launcher import preflight
    from voxft.train.yaml_builder import build_yaml

    work.mkdir(parents=True, exist_ok=False)
    runtime_environment(work)
    recipe_path = DATA_PROCESSED / "joint_omni8/mix.json"
    recipe = json.loads(recipe_path.read_text())
    reports = {name: json.loads((evaluation.EVAL_DIR / SOURCES[name]).read_text())
               for name in ("base", "r8")}
    evaluation.review_session(reports["base"]["report_path"], reports["r8"]["report_path"])
    cases = {}
    keys = ("case_id", "text", "lang", "ref_audio", "ref_lang", "control", "numeric", "note")
    for item in reports["r8"]["items"]:
        cases[item["case_id"]] = {key: item[key] for key in keys if key in item}
    r8_texts = {evaluation._norm(row["text"]) for row in
                _read_manifest(DATA_PROCESSED / "joint_omni8/train.jsonl")}
    for case in cases.values():
        case["cohort"] = ("r8_seen_text" if evaluation._norm(case["text"]) in r8_texts
                          else "r8_unseen_text")
    _write_jsonl(list(cases.values()), work / "cases.jsonl")
    eval_texts = {evaluation._norm(row["text"]) for row in cases.values()}
    sources = {}
    aliases = {}
    for name in ("cv22_th", "cv22_id"):
        if not get_source(name).has_speaker:
            raise ValueError(f"Untrusted source identity: {name}")
        split_rows = {split: _read_manifest(DATA_PROCESSED / name / f"{split}.jsonl")
                      for split in ("train", "val")}
        if memberships(split_rows["train"]) & memberships(split_rows["val"]):
            raise ValueError(f"Source train/val leakage: {name}")
        alias = f"r10_{name}_pairs"
        if (DATA_PROCESSED / alias).exists():
            raise FileExistsError(DATA_PROCESSED / alias)
        aliases[alias] = name
        sources[name] = {"license": get_source(name).license}
        for split, rows in split_rows.items():
            pairs = checked_pairs(rows, eval_texts)
            if not pairs:
                raise ValueError(f"No eligible pairs: {name}/{split}")
            _write_jsonl(pairs, DATA_PROCESSED / alias / f"{split}.jsonl")
            sources[name][split] = {
                "input_sha256": sha256(DATA_PROCESSED / name / f"{split}.jsonl"),
                "input_rows": len(rows), "eligible_pairs": len(pairs),
            }
    parts = [(name, weight - 3 if name in ("fleurs_th", "fleurs_id") else weight)
             for name, weight in recipe["parts"]]
    parts.extend((alias, 3) for alias in aliases)
    for position, (name, weight) in enumerate(parts):
        if name in aliases:
            continue
        rows = _read_manifest(DATA_PROCESSED / name / "train.jsonl")
        clean = [row for row in rows if evaluation._norm(row["text"]) not in eval_texts]
        sources[name] = {
            "license": get_source(name).license,
            "input_sha256": sha256(DATA_PROCESSED / name / "train.jsonl"),
            "input_rows": len(rows), "excluded_eval_text_rows": len(rows) - len(clean),
        }
        if len(clean) != len(rows):
            alias = f"r10_{name}_eval_clean"
            if (DATA_PROCESSED / alias).exists():
                raise FileExistsError(DATA_PROCESSED / alias)
            _write_jsonl(clean, DATA_PROCESSED / alias / "train.jsonl")
            _write_jsonl(_read_manifest(DATA_PROCESSED / name / "val.jsonl"),
                         DATA_PROCESSED / alias / "val.jsonl")
            aliases[alias] = name
            parts[position] = (alias, weight)
    name_on = "r10_ref_on"
    name_off = "r10_ref_off"
    if any((DATA_PROCESSED / name).exists() for name in (name_on, name_off)):
        raise FileExistsError("r10 manifests already exist; refusing overwrite")
    mix_manifests(parts, name_on, seed=42, train_hours=12, progress=print)
    directory = DATA_PROCESSED / name_on
    train_rows = _read_manifest(directory / "train.jsonl")
    removed = len(train_rows) % 16
    if removed:
        train_rows = train_rows[:-removed]
    val_rows = [without_ref(row) for row in _read_manifest(directory / "val.jsonl")]
    if memberships(train_rows) & memberships(val_rows):
        raise ValueError("Mixed train/val leakage")
    if any(evaluation._norm(row["text"]) in eval_texts for row in train_rows):
        raise ValueError("Evaluation text appears in training targets")
    validated = set()
    audio_hashes = {}
    for row in train_rows + val_rows:
        for field, duration_field in (("audio", "duration"), ("ref_audio", "ref_duration")):
            if not row.get(field):
                continue
            audio_path = row[field]
            if audio_path in validated:
                continue
            info = sf.info(audio_path)
            if (info.samplerate != 16000 or info.channels != 1
                    or not 3 <= info.duration <= 30
                    or abs(info.duration - float(row[duration_field])) > 0.03):
                raise ValueError(f"Invalid processed audio metadata: {audio_path}")
            if row["source_id"] in ("cv22_th", "cv22_id"):
                waveform, _ = sf.read(audio_path, dtype="float32")
                if not np.isfinite(waveform).all() or not 0 < np.max(np.abs(waveform)) <= 1:
                    raise ValueError(f"Invalid CV22 waveform: {audio_path}")
                audio_hashes[audio_path] = sha256(audio_path)
            validated.add(audio_path)
        if row.get("ref_audio") and audio_hashes[row["audio"]] == audio_hashes[row["ref_audio"]]:
            raise ValueError(f"Identical target/ref audio content: {row['audio']}")
        max_tokens = (len(row["text"].encode("utf-8"))
                      + math.ceil((row["duration"] + row.get("ref_duration", 0)) * 50) + 4)
        if max_tokens > 4096:
            raise ValueError("Conservative token bound exceeds shared training limit")
    train_hashes = {audio_hashes[row[field]] for row in train_rows
                    for field in ("audio", "ref_audio") if row.get(field) in audio_hashes}
    val_hashes = {audio_hashes[row["audio"]] for row in val_rows if row["audio"] in audio_hashes}
    if train_hashes & val_hashes:
        raise ValueError("CV22 train/val share identical audio content")
    _write_jsonl(val_rows, work / "validation.jsonl")
    configs = {}
    summary_on = json.loads((directory / "mix.json").read_text())
    r8_config_path = ROOT / "configs/lora_omni5_r8.yaml"
    base_path = yaml.safe_load(r8_config_path.read_text())["pretrained_path"]
    if not Path(base_path).is_dir():
        raise FileNotFoundError(base_path)
    for arm, name in (("off", name_off), ("on", name_on)):
        rows = [without_ref(row) for row in train_rows] if arm == "off" else train_rows
        output = DATA_PROCESSED / name
        _write_jsonl(rows, output / "train.jsonl")
        _write_jsonl(val_rows, output / "val.jsonl")
        summary = {**deepcopy(summary_on), "train": dataset_summary(rows),
                   "val": dataset_summary(val_rows)}
        summary["language_shares"] = {
            lang: {**share, "hours": summary["train"]["language_hours"][lang],
                   "actual": round(summary["train"]["language_hours"][lang] /
                                   summary["train"]["hours"], 4)}
            for lang, share in summary_on["language_shares"].items()}
        for source_name, _ in parts:
            original = aliases.get(source_name, source_name)
            selected = [row for row in rows if row["source_id"] == original]
            summary["datasets"][f"{source_name}/train"] = {
                **dataset_summary(selected),
                "actual_duration_share": sum(row["duration"] for row in selected) /
                                         summary["train"]["seconds"]}
            summary["counts"][f"{source_name}/train"] = len(selected)
            selected_val = [row for row in val_rows if row["source_id"] == original]
            summary["datasets"][f"{source_name}/val"] = dataset_summary(selected_val)
            summary["counts"][f"{source_name}/val"] = len(selected_val)
        summary["ref_ab_arm"] = arm
        write_json(output / "mix.json", summary)
        if (CHECKPOINT_DIR / ARMS[arm]).exists() or (ROOT / "configs" / f"{ARMS[arm]}.yaml").exists():
            raise FileExistsError(ARMS[arm])
        config = build_yaml(ARMS[arm], base_path, str(output / "train.jsonl"),
                            str(work / "validation.jsonl"), epochs=1, gpus=1)
        issues = preflight(config, gpus=1)
        if any(not issue.startswith("警告") for issue in issues):
            raise ValueError("\n".join(issues))
        configs[arm] = {"path": str(config), "sha256": sha256(config), "warnings": issues,
                        "manifest_sha256": sha256(output / "train.jsonl"),
                        "summary": summary["train"]}
    off_rows = _read_manifest(DATA_PROCESSED / name_off / "train.jsonl")
    if off_rows != [without_ref(row) for row in train_rows]:
        raise ValueError("A/B targets or ordering differ")
    config_rows = [yaml.safe_load(Path(configs[arm]["path"]).read_text()) for arm in ARMS]
    allowed = {"train_manifest", "save_path", "tensorboard"}
    if ({key: value for key, value in config_rows[0].items() if key not in allowed}
            != {key: value for key, value in config_rows[1].items() if key not in allowed}):
        raise ValueError("A/B hyperparameters differ")
    plan = {
        "configs": configs, "sources": sources, "parts": parts, "requested_hours": 12,
        "dropped_to_full_batch": removed, "training_seed": 42, "epochs": 1,
        "num_iters": config_rows[0]["num_iters"], "initialization": "fresh base; not r8 continuation",
        "base_path": base_path, "recipe_sha256": sha256(recipe_path),
        "r8_config_sha256": sha256(r8_config_path),
        "official_train_sha256": sha256(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py"),
        "validation_sha256": sha256(work / "validation.jsonl"),
        "cases_sha256": sha256(work / "cases.jsonl"), "evaluation_seeds": SEEDS,
        "validated_audio_files": len(validated),
        "r8_seen_text_case_ids": sorted(case["case_id"] for case in cases.values()
                                        if case["cohort"] == "r8_seen_text"),
        "historical_reports": {name: {"path": str(evaluation.EVAL_DIR / SOURCES[name]),
                                     "sha256": sha256(evaluation.EVAL_DIR / SOURCES[name])}
                               for name in reports},
        "note": "Small-budget paired intervention, not a reproduction of r8. "
                "Only TH/ID have trusted ref training. Low coverage and one training seed "
                "limit negative conclusions. No production promotion or native-quality claim.",
    }
    write_json(work / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)


def execute(work):
    import yaml

    runtime_environment(work)
    plan = json.loads((work / "plan.json").read_text())
    if sha256(work / "cases.jsonl") != plan["cases_sha256"]:
        raise ValueError("Evaluation cases changed")
    for arm, config_info in plan["configs"].items():
        config_path = Path(config_info["path"])
        config = yaml.safe_load(config_path.read_text())
        if (sha256(config_path) != config_info["sha256"]
                or sha256(config["train_manifest"]) != config_info["manifest_sha256"]
                or sha256(config["val_manifest"]) != plan["validation_sha256"]
                or sha256(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py")
                != plan["official_train_sha256"]):
            raise ValueError("Frozen training inputs changed")
        marker = work / f"train_{arm}.done"
        if not marker.exists():
            if (Path(config["save_path"]) / "latest").exists():
                raise RuntimeError("Partial run detected; refusing implicit resume")
            write_json(work / "status.json", {"stage": f"train_{arm}", "started": time.time()})
            with (work / f"train_{arm}.log").open("x") as log, \
                    (work / f"wandb_{arm}.log").open("a") as bridge_log:
                bridge = subprocess.Popen(
                    [sys.executable, "-u", "-m", "voxft.train.tb_wandb_bridge",
                     config["tensorboard"], ARMS[arm]], cwd=ROOT,
                    stdout=bridge_log, stderr=subprocess.STDOUT)
                try:
                    subprocess.run(
                        [sys.executable, "-u", str(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py"),
                         "--config_path", str(config_path)], cwd=VOXCPM_REPO,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                finally:
                    bridge.terminate()
                    try:
                        bridge.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        bridge.kill()
                        bridge.wait()
            state = json.loads((Path(config["save_path"]) / "latest/training_state.json").read_text())
            if state["step"] != plan["num_iters"]:
                raise ValueError("Training stopped before planned step")
            marker.write_text(str(time.time()))
    jobs = []
    for arm in ARMS:
        write_json(work / "status.json", {"stage": f"eval_{arm}", "started": time.time()})
        for shard in range(2):
            marker = work / f"eval_{arm}_{shard}.json"
            if not marker.exists():
                with (work / f"eval_{arm}_{shard}.log").open("x") as log:
                    jobs.append(subprocess.Popen(
                        [sys.executable, "-u", str(Path(__file__).resolve()), "eval",
                         "--work", str(work), "--arm", arm, "--shard", str(shard)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        results = [job.wait() for job in jobs]
        if any(results):
            raise RuntimeError(f"Evaluation failed: {results}")
        jobs = []
    write_json(work / "status.json", {"stage": "evaluations_done", "finished": time.time()})
    collect(work)


def verify_comparison(reports, cases):
    fields = ("text", "lang", "ref_audio", "control")
    expected = {(case["case_id"], seed): tuple(case.get(key) or "" for key in fields)
                for case in cases for seed in SEEDS}
    for report in reports.values():
        actual = {(row["case_id"], row["seed"]): tuple(row.get(key) or "" for key in fields)
                  for row in report["items"]}
        if expected != actual or len(report["items"]) != len(expected):
            raise ValueError("Reports must cover identical complete case/seed/condition sets")
        if (report.get("cfg_value") != 1.8 or report.get("inference_timesteps") != 20
                or report.get("retry_badcase") is not False or report.get("asr_model") != "large-v3"):
            raise ValueError("Inference/ASR conditions differ from the frozen evaluation plan")


def collect(work):
    from voxft import eval as evaluation
    from voxft.train import runlog

    runtime_environment(work)
    write_json(work / "status.json", {"stage": "collect", "started": time.time()})
    plan = json.loads((work / "plan.json").read_text())
    reports = {arm: evaluation.merge_reports(
        [str(work / f"eval_{arm}_{shard}.json") for shard in range(2)]) for arm in ARMS}
    for name, source in plan["historical_reports"].items():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Frozen historical report changed")
        reports[name] = json.loads(Path(source["path"]).read_text())
    cases = [json.loads(line) for line in (work / "cases.jsonl").read_text().splitlines()]
    verify_comparison(reports, cases)
    cohorts = {case["case_id"]: case["cohort"] for case in cases}
    for name, report in reports.items():
        report["items"] = [{**row, "cohort": cohorts[row["case_id"]]} for row in report["items"]]
        report.update(evaluation._report_metrics(report["items"]))
        report["cohorts"] = {
            cohort: evaluation._report_metrics([row for row in report["items"] if row["cohort"] == cohort])
            for cohort in sorted(set(cohorts.values()))}
        write_json(work / f"{name}_report.json", report)
    summary = {
        "reports": {name: {"overall": evaluation._agg(report["items"]),
                           **{key: report[key] for key in ("by_lang", "seed_stability", "cohorts")}}
                    for name, report in reports.items()},
        "note": f"Historical r8 saw {len(plan['r8_seen_text_case_ids'])}/{len(cases)} evaluation texts. "
                "Seen/unseen cohorts are separate; "
                "the full v2 set is not an independent held-out test. "
                "A/B are fresh 12-hour experiments, not budget-matched r8 reproductions. "
                "No native-language validation or production promotion.",
    }
    for name in ("base", "r8", "off"):
        ordered = [reports[name]] + [reports[arm] for arm in ARMS if arm != name]
        summary[f"against_{name}"] = {
            "cer": runlog._regressed(ordered), "duration": runlog._duration_gates(ordered)}
    write_json(work / "comparison.json", summary)
    for arm in ARMS:
        record = runlog.build_record(
            ARMS[arm], [str(work / "base_report.json"), str(work / f"{arm}_report.json")],
            verdict="离线诊断完成，未转正；语言质量未验证",
            next_step="结合 ref 开/关与 r8 对照决定是否扩大预算，不自动上传",
            notes=f"同目标{plan['configs'][arm]['summary']['hours']}h/1epoch实验；"
                  "全量v2含r8已见文本，分组结果见 comparison.json。")
        (work / f"{ARMS[arm]}.md").write_text(record)
    write_json(work / "status.json", {"stage": "voice_stability", "started": time.time()})
    with (work / "voice_stability.log").open("a") as log:
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts/voice_stability.py"),
             str(work / "off_report.json"), str(work / "on_report.json")],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(work / "status.json", {"stage": "complete", "finished": time.time(),
                                    "production_changed": False, "native_review": "unverified"})
    (work / "done").write_text(str(time.time()))


def evaluate_arm(work, arm, shard):
    from voxft import eval as evaluation

    runtime_environment(work)
    plan = json.loads((work / "plan.json").read_text())
    if sha256(work / "cases.jsonl") != plan["cases_sha256"]:
        raise ValueError("Evaluation cases changed")
    destination = work / f"eval_{arm}_{shard}.json"
    if destination.exists():
        raise FileExistsError(destination)
    cases = [json.loads(line) for line in (work / "cases.jsonl").read_text().splitlines()]
    report = evaluation.evaluate(str(CHECKPOINT_DIR / ARMS[arm] / "latest"), "th", cases,
                                 base=plan["base_path"], seeds=SEEDS, cfg_value=1.8,
                                 inference_timesteps=20, shard=(shard, 2), progress=print)
    write_json(destination, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "eval", "collect"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "ref_ab_20260917")
    parser.add_argument("--arm", choices=tuple(ARMS), default="off")
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.work)
        elif args.action == "run":
            execute(args.work)
        elif args.action == "collect":
            collect(args.work)
        else:
            evaluate_arm(args.work, args.arm, args.shard)
    except Exception as error:
        if args.action in ("run", "collect") and args.work.is_dir():
            write_json(args.work / "status.json", {"stage": "failed", "error": str(error),
                                                  "failed_at": time.time()})
        raise
