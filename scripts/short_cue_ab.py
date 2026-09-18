"""r11 pilot: change only within-source TL duration sampling; no production promotion."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest, _write_jsonl, dataset_summary
from voxft.paths import CHECKPOINT_DIR, DATA_PROCESSED, DATA_RAW, ROOT, VOXCPM_REPO
from voxft.train.launcher import preflight
from voxft.train.yaml_builder import build_yaml

from holdout_texts import REVISION, parse_tsv
from quality_probe import write_json
from ref_ab import memberships, runtime_environment, sha256, without_ref
from training_coverage import describe

ARMS = {"control": "lora_omni5_r11_control", "short": "lora_omni5_r11_tl_short"}
SEEDS = [42, 43, 44, 45, 49]


def take_seconds(rows, seconds, seed):
    selected, consumed = [], 0.0
    for row in random.Random(seed).sample(rows, len(rows)):
        if consumed >= seconds:
            break
        selected.append(row)
        consumed += row["duration"]
    if consumed < seconds:
        raise ValueError("Insufficient unique audio; refusing repetition or budget redistribution")
    return selected


def make_arms(pools, parts):
    common = []
    for source, weight in parts:
        if source != "fleurs_tl":
            common.extend(take_seconds(pools[source], 12 * 3600 * weight / 100, f"r11:{source}"))
    target = 12 * 3600 * 0.17
    tl = pools["fleurs_tl"]
    baseline = take_seconds(tl, target, "r11:tl:control")
    short = take_seconds([row for row in tl if row["duration"] <= 8], target * 0.25, "r11:tl:short")
    short += take_seconds([row for row in tl if row["duration"] > 8], target * 0.75, "r11:tl:long")
    arms = {"control": common + baseline, "short": common + short}
    for rows in arms.values():
        if len({row["origin_audio"] for row in rows}) != len(rows):
            raise ValueError("Duplicate origin in pilot")
        random.Random(42).shuffle(rows)
    return arms


def prepare(work):
    import soundfile as sf
    import yaml

    if work.exists():
        raise FileExistsError(work)
    coverage_root = CHECKPOINT_DIR.parent / "training_coverage_20260918"
    if not (coverage_root / "asr/done").is_file():
        raise ValueError("Complete the frozen data spot-check first")
    coverage = json.loads((coverage_root / "coverage.json").read_text())
    inputs = dict(coverage["inputs_sha256"])
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Coverage inputs changed; repeat the audit in a new directory")
    if (any(coverage["new_candidate_rejections"].values())
            or any(source["provenance"]["issues"] for source in coverage["candidates_by_source"].values())):
        raise ValueError("Resolve candidate audit flags before preparing training")
    inputs[str(coverage_root / "coverage.json")] = sha256(coverage_root / "coverage.json")
    inputs[str(coverage_root / "asr/results.json")] = sha256(coverage_root / "asr/results.json")
    paths = sorted(set(DATA_PROCESSED.glob("*/train.jsonl")) | set(DATA_PROCESSED.glob("*/val.jsonl")) |
                   set(DATA_RAW.glob("*/manifest.jsonl")) | set(ROOT.glob("eval_cases/*.jsonl")))
    excluded = set()
    for path in paths:
        inputs[str(path)] = sha256(path)
        excluded.update(evaluation._norm(row["text"]) for row in _read_manifest(path) if row.get("text"))
    for path in sorted(evaluation.EVAL_DIR.glob("*.json")):
        inputs[str(path)] = sha256(path)
        excluded.update(evaluation._norm(row["text"])
                        for row in json.loads(path.read_text()).get("items", []) if row.get("text"))
    source_path = CHECKPOINT_DIR.parent / "holdout_texts_20260917/sources/data/fil_ph/test.tsv"
    source_plan_path = CHECKPOINT_DIR.parent / "holdout_texts_20260917/plan.json"
    source_plan = json.loads(source_plan_path.read_text())
    if (source_plan["source_revision"] != REVISION
            or sha256(source_path) != source_plan["inputs_sha256"][str(source_path)]):
        raise ValueError("Cached official text source changed")
    inputs[str(source_plan_path)] = sha256(source_plan_path)
    inputs[str(source_path)] = sha256(source_path)
    texts = {evaluation._norm(row["text"]): row for row in parse_tsv(source_path.read_text())
             if 10 <= len(row["text"]) <= 80 and not evaluation._is_numeric(row)
             and evaluation._norm(row["text"]) not in excluded}
    if not 5 <= len(texts) <= 30:
        raise ValueError(f"Unexpected short-text inventory: {len(texts)}")
    original_cases = _read_manifest(ROOT / "eval_cases/fleurs_test_holdout_20260917.jsonl")
    references = {lang: sorted({row["ref_audio"] for row in original_cases if row["ref_lang"] == lang})
                  for lang in ("zh", "en", "tl")}
    short_cases = []
    for position, row in enumerate(sorted(texts.values(), key=lambda row: int(row["source_sentence_id"]))):
        for ref_lang, paths in references.items():
            reference = paths[position % len(paths)]
            inputs[reference] = sha256(reference)
            short_cases.append({
                **row, "case_id": f"r11_tl_{row['source_sentence_id']}_{ref_lang}", "lang": "tl",
                "ref_audio": reference, "ref_lang": ref_lang, "numeric": False,
                "source_dataset": "google/fleurs", "source_revision": REVISION,
                "source_split": "test", "source_license": "CC-BY-4.0",
                "note": "Short read-speech text pilot, not dialogue or verified <=8s audio. "
                        "Repeated refs/seeds share a source sentence; not independent text samples.",
            })
    controls = []
    for lang in ("th", "tl", "vi", "id", "ms"):
        for ref_lang in ("zh", "en", "tl"):
            choices = sorted((row for row in original_cases if row["lang"] == lang
                              and row["ref_lang"] == ref_lang), key=lambda row: row["case_id"])
            controls.append(random.Random(f"r11:{lang}:{ref_lang}").choice(choices))
    recipe_path = DATA_PROCESSED / "joint_omni8/mix.json"
    inputs[str(recipe_path)] = sha256(recipe_path)
    parts = json.loads(recipe_path.read_text())["parts"]
    if sum(weight for _, weight in parts) != 100 or dict(parts).get("fleurs_tl") != 17:
        raise ValueError("Unexpected r8 source recipe")
    pool_root = CHECKPOINT_DIR.parent / "training_pool_20260917"
    pools = {source: [without_ref(row) for row in _read_manifest(pool_root / source / "targets.jsonl")]
             for source, _ in parts}
    reserved = {evaluation._norm(row["text"]) for row in short_cases + controls}
    if any(evaluation._norm(row["text"]) in reserved for rows in pools.values() for row in rows):
        raise ValueError("Evaluation text entered pilot pool")
    arms = make_arms(pools, parts)
    steps = min(len(rows) // 16 for rows in arms.values())
    validations = [without_ref(row) for row in _read_manifest(DATA_PROCESSED / "joint_omni8/val.jsonl")]
    protected = memberships(validations)
    audio_hashes = {}
    for rows in arms.values():
        if memberships(rows) & protected:
            raise ValueError("Pilot/validation membership overlap")
        for row in rows + validations:
            if row["audio"] in audio_hashes:
                continue
            info = sf.info(row["audio"])
            if (info.samplerate != 16000 or info.channels != 1 or not 3 <= info.duration <= 30
                    or abs(info.duration - row["duration"]) > 0.03):
                raise ValueError(f"Audio header mismatch: {row['audio']}")
            if len(row["text"].encode()) + math.ceil(row["duration"] * 50) + 4 > 4096:
                raise ValueError("Possible implicit token-budget filtering")
            audio_hashes[row["audio"]] = sha256(row["audio"])
    base_config = ROOT / "configs/lora_omni5_r8.yaml"
    inputs[str(base_config)] = sha256(base_config)
    base_path = yaml.safe_load(base_config.read_text())["pretrained_path"]
    inputs.update({str(path): sha256(path) for path in Path(base_path).iterdir() if path.is_file()})
    inputs.update({str(path): sha256(path) for path in (VOXCPM_REPO / "src/voxcpm/training").glob("*.py")})
    inputs[str(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py")] = sha256(
        VOXCPM_REPO / "scripts/train_voxcpm_finetune.py")
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("An audited input changed")
    if any((CHECKPOINT_DIR / name).exists() or (ROOT / "configs" / f"{name}.yaml").exists()
           for name in ARMS.values()):
        raise FileExistsError("Existing r11 run/config; no implicit overwrite or resume")
    work.mkdir()
    _write_jsonl(short_cases, work / "short_cases.jsonl")
    _write_jsonl(controls, work / "control_cases.jsonl")
    _write_jsonl(validations, work / "validation.jsonl")
    configs = {}
    for arm, rows in arms.items():
        manifest = work / arm / "train.jsonl"
        _write_jsonl(rows, manifest)
        summary = dataset_summary(rows)
        shares = {lang: {"requested": 0.1 if lang == "zh" else 0.05 if lang == "en" else 0.17,
                         "actual": hours / summary["hours"], "hours": hours}
                  for lang, hours in summary["language_hours"].items()}
        write_json(manifest.parent / "mix.json", {
            "parts": parts, "basis": "duration", "max_repeat": 1, "train": summary,
            "val": dataset_summary(validations), "language_shares": shares,
            "intervention": "TL short-duration share 25%" if arm == "short" else "unbiased source sampling",
        })
        config_path = build_yaml(ARMS[arm], base_path, str(manifest), str(work / "validation.jsonl"),
                                 overrides={"num_iters": steps, "warmup_steps": max(1, steps // 10)})
        issues = preflight(config_path, gpus=1)
        if any(not issue.startswith("警告") for issue in issues):
            raise ValueError("\n".join(issues))
        configs[arm] = {
            "path": str(config_path), "sha256": sha256(config_path), "warnings": issues,
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "summary": summary, "language_shares": shares,
            "tl": describe([row for row in rows if row["lang"] == "tl"]),
            "nominal_epochs": steps * 16 / len(rows),
        }
    comparable = [{key: value for key, value in yaml.safe_load(Path(entry["path"]).read_text()).items()
                   if key not in ("train_manifest", "save_path", "tensorboard")}
                  for entry in configs.values()]
    if comparable[0] != comparable[1]:
        raise ValueError("A/B hyperparameters differ")
    plan = {
        "configs": configs, "steps": steps, "training_seed": 42, "evaluation_seeds": SEEDS,
        "source_parts": parts, "base_path": base_path, "inputs_sha256": inputs,
        "audio_sha256": audio_hashes,
        "frozen_files": {str(path): sha256(path) for path in work.rglob("*") if path.is_file()},
        "source_sha256": {str(path): sha256(path) for path in
                          (Path(__file__), ROOT / "scripts/ref_ab.py", ROOT / "scripts/training_coverage.py",
                           ROOT / "scripts/holdout_texts.py", ROOT / "src/voxft/data/pipeline.py",
                           ROOT / "src/voxft/train/yaml_builder.py", ROOT / "src/voxft/train/launcher.py",
                           ROOT / "src/voxft/eval.py", ROOT / "src/voxft/infer.py")},
        "short_texts": len(texts), "short_conditions": len(short_cases), "control_cases": len(controls),
        "note": "Fresh base, fixed optimizer updates and LR schedule, at most one nominal epoch. "
                "Only TL within-FLEURS duration sampling changes; other source rows identical. "
                "Both manifests target 12h and original duration shares; actual consumed hours/order "
                "are NOT guaranteed identical with shuffled incomplete epochs. "
                "One training seed; sampling-policy pilot, not proof of a pure length causal effect. "
                "No ref/projection changes. Inventory <=1 exposure, language quality unverified. "
                "Short-case analysis must cluster by source_sentence_id across refs/seeds. "
                "15 five-language controls are a smoke screen, not final acceptance. "
                "Only a promising pilot earns full five-language/replay/pressure validation; no promotion.",
    }
    write_json(work / "plan.json", plan)
    print(json.dumps({key: value for key, value in plan.items()
                      if key not in ("inputs_sha256", "audio_sha256")}, ensure_ascii=False, indent=2), flush=True)


def run(work):
    import yaml

    plan = json.loads((work / "plan.json").read_text())
    for group in ("inputs_sha256", "audio_sha256", "frozen_files", "source_sha256"):
        if any(sha256(path) != digest for path, digest in plan[group].items()):
            raise ValueError(f"Frozen inputs changed: {group}")
    if (work / "status.json").exists():
        raise FileExistsError("Pilot already started; refusing implicit restart")
    runtime_environment(work)
    try:
        for arm, entry in plan["configs"].items():
            if sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("Config changed")
            config = yaml.safe_load(Path(entry["path"]).read_text())
            save_path = Path(config["save_path"])
            if save_path.exists() and (not save_path.is_dir() or any(save_path.iterdir())):
                raise FileExistsError(save_path)
            if shutil.disk_usage(work).free < 3 * 1024**3:
                raise RuntimeError("Need at least 3GiB free before each arm")
            write_json(work / "status.json", {"stage": f"train_{arm}", "time": time.time()})
            with (work / f"train_{arm}.log").open("x") as log, \
                    (work / f"wandb_{arm}.log").open("x") as bridge_log:
                bridge = subprocess.Popen([sys.executable, "-u", "-m", "voxft.train.tb_wandb_bridge",
                                           config["tensorboard"], ARMS[arm]], cwd=ROOT,
                                          stdout=bridge_log, stderr=subprocess.STDOUT)
                try:
                    subprocess.run([sys.executable, "-u",
                                    str(VOXCPM_REPO / "scripts/train_voxcpm_finetune.py"),
                                    "--config_path", entry["path"]], cwd=VOXCPM_REPO,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                finally:
                    bridge.terminate()
                    try:
                        bridge.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        bridge.kill()
                        bridge.wait()
            state = json.loads((save_path / "latest/training_state.json").read_text())
            if state["step"] != plan["steps"]:
                raise ValueError("Training stopped before the frozen budget")
            write_json(work / f"train_{arm}.done.json", state)
            for checkpoint in save_path.glob("step_*"):
                if checkpoint.is_dir() and not checkpoint.is_symlink():
                    shutil.rmtree(checkpoint)
        for label in ("base", "r8", *ARMS):
            for cohort in (("short",) if label in ("base", "r8") else ("short", "control")):
                if shutil.disk_usage(work).free < 1024**3:
                    raise RuntimeError("Need at least 1GiB free before evaluation")
                write_json(work / "status.json", {"stage": f"eval_{label}_{cohort}", "time": time.time()})
                with (work / f"eval_{label}_{cohort}.log").open("x") as log:
                    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "eval",
                                    "--work", str(work), "--label", label, "--cohort", cohort],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(work / "status.json", {"stage": "pilot_audio_complete_not_accepted", "time": time.time()})
        (work / "done").write_text("Training and pilot generation complete; analysis/acceptance pending.\n")
    except Exception as error:
        write_json(work / "status.json", {"stage": "failed", "error": str(error), "time": time.time()})
        raise


def evaluate(work, label, cohort):
    plan = json.loads((work / "plan.json").read_text())
    path = work / f"{cohort}_cases.jsonl"
    if sha256(path) != plan["frozen_files"][str(path)]:
        raise ValueError("Evaluation cases changed")
    target = ("base" if label == "base" else str(CHECKPOINT_DIR /
              ("lora_omni5_r8" if label == "r8" else ARMS[label]) / "latest"))
    report = evaluation.evaluate(target, "tl", _read_manifest(path), base=plan["base_path"],
                                 seeds=SEEDS, cfg_value=1.8, inference_timesteps=20, progress=print)
    write_json(work / f"{label}_{cohort}_report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "eval"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "short_cue_ab_20260918")
    parser.add_argument("--label", choices=("base", "r8", *ARMS))
    parser.add_argument("--cohort", choices=("short", "control"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.work)
    elif args.action == "run":
        run(args.work)
    else:
        if args.label is None or args.cohort is None:
            parser.error("eval requires --label and --cohort")
        evaluate(args.work, args.label, args.cohort)
