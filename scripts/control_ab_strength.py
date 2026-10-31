"""Frozen on-LoRA strength diagnostic; reuse endpoints, never train or promote."""
import argparse
from collections import Counter
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import control_ab_fast_eval as fast
import control_ab_review as review
from voxft.data.pipeline import _write_jsonl

frozen = fast.frozen
STRENGTHS = (0.5, 0.75)
TRAINED_ON_SHA256 = "6459e546860eec41beb04f9a7b28f1843bec363bc31a09822b3e3f524cdcbd9f"
METRICS = ("cer", "suspected_truncation", "over_read", "head_silence_sec",
           "tail_silence_sec", "speech_ratio", "speaker_sim",
           "rms_dbfs", "p90_frame_rms_dbfs")


def tasks(cases, index=0):
    strengths = STRENGTHS if index % 2 == 0 else STRENGTHS[::-1]
    for strength in strengths:
        for position, case in enumerate(cases):
            for seed in frozen.SEEDS:
                yield f"s{strength:g}_{position:02d}_{seed}", strength, case, seed


def check_inputs(work):
    marker = work / "run.started"
    if marker.exists() and marker.read_text(encoding="utf-8") != frozen.sha256(work / "plan.json"):
        raise ValueError("Run plan changed")
    return frozen.check_inputs(work)


def verify_report(report, plan, cases, strength, seeds=None):
    seeds = frozen.SEEDS if seeds is None else seeds
    expected = {(case["case_id"], seed): case for case in cases for seed in seeds}
    actual = {(row["case_id"], row["seed"]): row for row in report["items"]}
    settings = {"target": "base" if strength == 0 else plan["target"],
                "base": plan["base_path"], "lora_strength": strength, "cfg_value": 1.8,
                "inference_timesteps": 20, "retry_badcase": False, "asr_model": "large-v3",
                "asr_auto_detect_langs": ["tl"]}
    if len(actual) != len(report["items"]) or actual.keys() != expected.keys():
        raise ValueError("Incomplete or duplicated case/seed results")
    if any(report.get(key) != value for key, value in settings.items()):
        raise ValueError("Evaluation/strength settings changed")
    for key, case in expected.items():
        if any(actual[key].get(field) != value for field, value in case.items()):
            raise ValueError(f"Case changed: {key}")
    hashes = report.get("wav_sha256", {})
    if hashes.keys() != {row["wav"] for row in report["items"]}:
        raise ValueError("Incomplete WAV hash coverage")
    for path, digest in hashes.items():
        review.checked(path, digest)


def prepare(source, pack, output):
    if output.exists():
        raise FileExistsError(output)
    old_plan, all_cases, reports, _ = review.load_sources(source)
    package = review.read_json(pack / "package.json")
    if not package["complete"] or package["package_id"] != "acoustic-5b297d423cc5":
        raise ValueError("Wrong or incomplete acoustic package")
    for relative, digest in package["artifacts_sha256"].items():
        review.checked(pack / relative, digest)
    for path, digest in package["source_sha256"].items():
        review.checked(path, digest)
    selection = review.read_json(pack / "selection.json")["items"]
    by_id = {case["case_id"]: case for case in all_cases}
    cases = [by_id[row["case_id"]] for row in selection]
    expected = Counter({(lang, group): count for lang in frozen.LANGS
                        for group, count in (("random", 4), ("tail_risk", 1),
                                             ("historical_risk", 1))})
    actual = Counter((case["lang"], row["stratum"]) for case, row in zip(cases, selection))
    if actual != expected or len({case["sentence_id"] for case in cases}) != 18:
        raise ValueError("Expected the original 18 distinct reviewed texts and strata")
    marker = review.read_json(review.checked(source / "trained_on.json", TRAINED_ON_SHA256))
    if marker["step"] != old_plan["num_iters"]:
        raise ValueError("Incomplete on checkpoint")
    profile_path = source / "fast_eval_profile.json"
    profile = review.read_json(profile_path)
    if not profile["metrics_exactly_equal"] or profile["source_sha256"] != frozen.sha256(fast.__file__):
        raise ValueError("Cached evaluator differs from validated implementation")
    config = old_plan["configs"]["on"]
    review.checked(config, old_plan["inputs_sha256"].get(config))
    plan = {"base_path": old_plan["base_path"], "target": frozen.target_path(old_plan, "on"),
            "seeds": frozen.SEEDS, "strengths": list(STRENGTHS), "new_outputs": 180,
            "reused_outputs": 180, "max_workers": 4, "selection": selection,
            "source": str(source), "package": str(pack), "production_changed": False,
            "training_steps": 0, "authorization_date": "2026-10-22"}
    inputs = {path: digest for path, digest in old_plan["inputs_sha256"].items()
              if path.endswith((".py", "/uv.lock", "/pyproject.toml"))
              or path.startswith(old_plan["base_path"] + "/")}
    inputs.update(marker["inputs_sha256"])
    inputs[config] = old_plan["inputs_sha256"][config]
    for case in cases:
        inputs[case["ref_audio"]] = old_plan["inputs_sha256"][case["ref_audio"]]
    inputs.update(package["source_sha256"])
    inputs.update({str(pack / relative): digest
                   for relative, digest in package["artifacts_sha256"].items()})
    for path in (Path(__file__).resolve(), Path(inspect.getfile(review.level_metrics)),
                 Path(fast.__file__), source / "trained_on.json", profile_path,
                 pack / "package.json", pack / "feedback_review/mapped_feedback.json",
                 pack / "feedback_review/submitted.txt"):
        inputs[str(path)] = frozen.sha256(path)
    subsets = {}
    selected_ids = {case["case_id"] for case in cases}
    for name, strength in (("base", 0), ("on", 1)):
        report = deepcopy(reports[name])
        report["items"] = [row for row in report["items"] if row["case_id"] in selected_ids]
        original_hashes = review.wav_hashes(reports[name])
        report["wav_sha256"] = {row["wav"]: original_hashes[row["wav"]] for row in report["items"]}
        verify_report(report, plan, cases, strength)
        inputs.update(report["wav_sha256"])
        report.update(frozen.evaluation._report_metrics(report["items"]))
        report.update(source_report=str(source / f"{name}_report.json"),
                      report_path=str(output / f"{name}_report.json"), reused_without_rescoring=True)
        subsets[name] = report
    for path, digest in inputs.items():
        review.checked(path, digest)
    if shutil.disk_usage(output.parent).free < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB free")
    output.mkdir()
    _write_jsonl(cases, output / "cases.jsonl")
    for name, report in subsets.items():
        frozen.write_json(output / f"{name}_report.json", report)
    for name in ("cases.jsonl", "base_report.json", "on_report.json"):
        inputs[str(output / name)] = frozen.sha256(output / name)
    plan.update(inputs_sha256=inputs, note=(
        "Post-feedback development diagnostic, not independent acceptance. Separate random and "
        "outcome-enriched strata. No endpoint regeneration, training, retry, cropping or normalization. "
        "RMS is amplitude, not LUFS; energy boundaries are not word alignment. Native quality unverified."))
    frozen.write_json(output / "plan.json", plan)
    frozen.write_json(output / "status.json", {"stage": "prepared", "time": time.time()})
    print("PREPARED 18 cases x 5 seeds x 2 strengths; 180 endpoints reused", flush=True)


def check_scaling(model, *, actual_strength=None):
    modules = list(model.tts_model._iter_lora_modules())
    if not modules or any(not torch.equal(module.scaling, torch.full_like(
            module.scaling, module._base_scaling)) for module in modules):
        raise ValueError("Expected enabled, unscaled loaded LoRA")
    originals = [module.scaling.detach().clone() for module in modules]
    strengths = ((0, 1, *STRENGTHS) if actual_strength is None else
                 tuple(dict.fromkeys((0, 1, *STRENGTHS, actual_strength))))
    for strength in strengths:
        with frozen.evaluation.infer.scaled_lora(model, strength):
            if any(not torch.equal(module.scaling, torch.full_like(
                    module.scaling, module._base_scaling * strength)) for module in modules):
                raise ValueError("Effective LoRA scaling differs")
        if any(not torch.equal(module.scaling, original)
               for module, original in zip(modules, originals)):
            raise ValueError("LoRA scaling was not restored")
    return {"modules": len(modules), "checked_strengths": list(strengths),
            "restored": True, "output_endpoint_equivalence": "not_regenerated"}


def worker(work, index):
    frozen.runtime_environment(work / f"runtime{index}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    plan = check_inputs(work)
    cases = frozen._read_manifest(work / "cases.jsonl")
    frozen.evaluation.EVAL_DIR = work / "raw_reports"
    frozen.evaluation.infer.CHECKPOINT_DIR = work
    model = frozen.evaluation.infer.get_model(plan["base_path"], plan["target"])
    frozen.write_json(work / f"scaling{index}.json", check_scaling(model))
    count = 0
    with fast.cached_services(cases):
        for identifier, strength, case, seed in tasks(cases, index):
            if not fast.claim(work / "claims", identifier):
                continue
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError("Less than 4 GiB free")
            review.checked(case["ref_audio"], plan["inputs_sha256"][case["ref_audio"]])
            start = time.monotonic()
            report = frozen.evaluation.evaluate(
                plan["target"], "th", [case], base=plan["base_path"], seeds=[seed],
                cfg_value=1.8, inference_timesteps=20, lora_strength=strength)
            report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"]) for row in report["items"]}
            verify_report(report, plan, [case], strength, [seed])
            frozen.write_json(work / "items" / f"{identifier}.json", report)
            count += 1
            frozen.write_json(work / f"worker{index}_progress.json", {
                "completed": count, "last_task": identifier, "time": time.time(),
                "wall_sec": round(time.monotonic() - start, 3)})
            print(f"DONE {identifier} worker={index} count={count}", flush=True)
    check_inputs(work)


def collect(work):
    if (work / "comparison.json").exists():
        raise FileExistsError(work / "comparison.json")
    plan = check_inputs(work)
    cases = frozen._read_manifest(work / "cases.jsonl")
    reports = {name: review.read_json(work / f"{name}_report.json") for name in ("base", "on")}
    for name, strength in (("base", 0), ("on", 1)):
        verify_report(reports[name], plan, cases, strength)
    frozen.evaluation.EVAL_DIR = work / "raw_reports"
    for strength in STRENGTHS:
        paths = []
        for identifier, value, case, seed in tasks(cases):
            if value == strength:
                path = work / "items" / f"{identifier}.json"
                verify_report(review.read_json(path), plan, [case], strength, [seed])
                paths.append(str(path))
        name = f"s{strength:g}"
        destination = work / f"{name}_report.json"
        if destination.exists():
            raise FileExistsError(destination)
        report = frozen.evaluation.merge_reports(paths)
        report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"]) for row in report["items"]}
        verify_report(report, plan, cases, strength)
        frozen.write_json(destination, report)
        reports[name] = report
    levels = {row["wav"]: review.level_metrics(row["wav"])
              for report in reports.values() for row in report["items"]}
    frozen.write_json(work / "levels.json", levels)
    indices = {name: {(row["case_id"], row["seed"]): {**row, **levels[row["wav"]]}
                      for row in report["items"]} for name, report in reports.items()}
    strata = {row["case_id"]: row["stratum"] for row in plan["selection"]}
    groups = {}
    for stratum in sorted(set(strata.values())):
        groups[stratum] = {}
        for lang in sorted(frozen.LANGS):
            group = {}
            for name, rows in indices.items():
                items = [row for row in rows.values()
                         if row["lang"] == lang and strata[row["case_id"]] == stratum]
                group[name] = frozen.evaluation._report_metrics(items)
                group[name]["levels"] = {
                    metric: float(np.mean([row[metric] for row in items]))
                    for metric in ("rms_dbfs", "p90_frame_rms_dbfs")}
            groups[stratum][lang] = group
    paired = [{"case_id": case["case_id"], "lang": case["lang"],
               "stratum": strata[case["case_id"]], "seed": seed,
               "models": {name: {metric: index[case["case_id"], seed].get(metric)
                                 for metric in (*METRICS, "wav")}
                          for name, index in indices.items()}}
              for case in cases for seed in frozen.SEEDS]
    reviewed = review.read_json(Path(plan["package"]) / "feedback_review/mapped_feedback.json")
    focused = [{**feedback, "comparison": next(row for row in paired
                if (row["case_id"], row["seed"]) == (feedback["case_id"], feedback["seed"]))}
               for feedback in reviewed if feedback["human_review"]["acoustic_status"] == "abnormal"]
    frozen.write_json(work / "comparison.json", {
        "by_stratum": groups, "paired": paired, "reviewed_abnormal": focused,
        "inputs_sha256": {str(work / filename): frozen.sha256(work / filename)
                         for filename in ("plan.json", "levels.json",
                                          *(f"{name}_report.json" for name in reports))},
        "production_changed": False, "language_quality": "unverified", "new_audio_human_review": "pending",
        "note": plan["note"]})
    check_inputs(work)


def run(work):
    if (work / "run.started").exists():
        raise FileExistsError("Refusing overwrite or implicit resume")
    plan = check_inputs(work)
    if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU already has compute jobs")
    with (work / "run.started").open("x", encoding="utf-8") as stream:
        stream.write(frozen.sha256(work / "plan.json"))
    for directory in ("claims", "items"):
        (work / directory).mkdir()
    processes = []
    try:
        for index in range(plan["max_workers"]):
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).strip())
            if free < 12 * 1024:
                break
            with (work / f"worker{index}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                     "worker", "--work", str(work), "--index", str(index)],
                    cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            while not (work / f"worker{index}_progress.json").exists() and process.poll() is None:
                if any(child.poll() not in (None, 0) for child in processes):
                    raise RuntimeError("Worker failed during startup")
                time.sleep(2)
            if process.poll() not in (None, 0):
                raise RuntimeError("Worker failed during startup")
        if not processes:
            raise RuntimeError("Insufficient free GPU memory")
        while True:
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Worker failed; preserve completed items, no automatic retry")
            frozen.write_json(work / "status.json", {
                "stage": "evaluating", "completed": len(list((work / "items").glob("*.json"))),
                "expected": plan["new_outputs"], "time": time.time(),
                "pids": [process.pid for process in processes if process.poll() is None]})
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(10)
        collect(work)
        frozen.write_json(work / "status.json", {
            "stage": "complete", "completed": plan["new_outputs"], "time": time.time(),
            "production_changed": False, "training_steps": 0})
    except BaseException as error:
        frozen.write_json(work / "status.json", {
            "stage": "failed", "error": str(error), "time": time.time()})
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "collect"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--index", type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is None or args.package is None:
            parser.error("prepare requires --source and --package")
        prepare(args.source, args.package, args.work)
    elif args.action == "worker":
        if args.index is None:
            parser.error("worker requires --index")
        worker(args.work, args.index)
    else:
        {"run": run, "collect": collect}[args.action](args.work)
