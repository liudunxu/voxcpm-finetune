"""Evaluate the frozen control A/B one case/seed at a time, with a shared work queue."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import control_ab_run as frozen
from voxft.data import pipeline
from voxft.qc import audio as qc


def pcm_key(waveform, sample_rate):
    return int(sample_rate), hashlib.sha256(
        np.asarray(waveform, dtype=np.float32).reshape(-1).tobytes()).hexdigest()


@contextmanager
def cached_services(cases):
    references = {pcm_key(*pipeline.load_wav_mono(path)) for path in
                  {row.get("ref_audio_original") or row.get("ref_audio") for row in cases}
                  if path}
    original_embedding = qc.speaker_embedding
    original_acoustics = frozen.evaluation._acoustics
    original_whisper = pipeline._whisper_model
    embeddings, reference_audio = {}, {}

    def embedding(waveform, sample_rate):
        key = pcm_key(waveform, sample_rate)
        if key not in references:
            return original_embedding(waveform, sample_rate)
        if key not in embeddings:
            embeddings[key] = original_embedding(waveform, sample_rate)
        return embeddings[key]

    def acoustics(wav_path, text, ref_path, ref_cache):
        return original_acoustics(wav_path, text, ref_path, reference_audio)

    qc.speaker_embedding = embedding
    frozen.evaluation._acoustics = acoustics
    pipeline._whisper_model = lru_cache(maxsize=1)(original_whisper)
    try:
        yield
    finally:
        qc.speaker_embedding = original_embedding
        frozen.evaluation._acoustics = original_acoustics
        pipeline._whisper_model = original_whisper


def jobs(cases, preferred="off"):
    for model in (preferred, "on" if preferred == "off" else "off"):
        for position, case in enumerate(cases):
            for seed in frozen.SEEDS:
                yield f"{model}_{position:04d}_{seed}", model, case, seed


def claim(directory, identifier):
    try:
        with (directory / f"{identifier}.json").open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "time": time.time()}, stream)
    except FileExistsError:
        return False
    return True


def profile(work):
    output = work / "fast_eval_profile.json"
    if output.exists():
        raise FileExistsError(output)
    frozen.check_inputs(work)
    report_path = work / "base_shard0.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = [next(row for row in report["items"] if row["lang"] == lang
                     and row.get("cohort") == "independent_text") for lang in frozen.LANGS]
    if any(frozen.sha256(row["wav"]) != report["wav_sha256"][row["wav"]] for row in selected):
        raise ValueError("Profiling WAV changed")
    frozen.runtime_environment(work / "profile_runtime")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    frozen.evaluation._acoustics(selected[0]["wav"], selected[0]["text"], selected[0]["ref_audio"], {})
    results = {}
    for mode in ("original", "cached", "cached_repeat", "original_repeat"):
        timings, metrics = [], []

        def measure():
            cache = {}
            for _ in range(2):
                for row in selected:
                    start = time.monotonic()
                    metrics.append(frozen.evaluation._acoustics(
                        row["wav"], row["text"], row["ref_audio"], cache))
                    timings.append(time.monotonic() - start)

        if mode.startswith("cached"):
            with cached_services(selected):
                measure()
        else:
            measure()
        results[mode] = {"seconds": timings, "metrics": metrics}
    equal = (all(row.get("speaker_sim") is not None for row in results["original"]["metrics"])
             and all(result["metrics"] == results["original"]["metrics"] for result in results.values()))
    frozen.write_json(output, {
        "source_sha256": frozen.sha256(__file__), "report_sha256": frozen.sha256(report_path),
        "wav_sha256": {row["wav"]: report["wav_sha256"][row["wav"]] for row in selected},
        "metrics_exactly_equal": equal, "runs": results,
        "note": "CPU-only timing and cache equivalence check; no TTS/ASR or historical score replacement.",
    })
    if not equal:
        raise ValueError("Cached acoustic metrics differ; do not enable optimized evaluation")
    print(json.dumps({mode: round(sum(result["seconds"]), 3) for mode, result in results.items()}))


def verify_run(work, directory):
    plan = frozen.check_inputs(work)
    run_plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if any(frozen.sha256(path) != digest for path, digest in run_plan["inputs_sha256"].items()):
        raise ValueError("Optimized evaluation input changed")
    return plan


def worker(work, directory, index):
    frozen.runtime_environment(directory / f"worker{index}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    plan = verify_run(work, directory)
    cases = frozen._read_manifest(work / "cases.jsonl")
    frozen.evaluation.EVAL_DIR = directory / "raw_reports"
    frozen.evaluation.infer.CHECKPOINT_DIR = directory
    count = 0
    with cached_services(cases):
        for identifier, model, case, seed in jobs(cases, "off" if index % 2 == 0 else "on"):
            if not claim(directory / "claims", identifier):
                continue
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError("Less than 4 GiB free")
            start = time.monotonic()
            report = frozen.evaluation.evaluate(
                frozen.target_path(plan, model), "th", [case], base=plan["base_path"],
                seeds=[seed], cfg_value=plan["cfg_value"],
                inference_timesteps=plan["inference_timesteps"])
            if len(report["items"]) != 1 or report["items"][0]["seed"] != seed:
                raise ValueError("Expected exactly one case/seed output")
            report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"]) for row in report["items"]}
            frozen.write_json(directory / "items" / f"{identifier}.json", report)
            count += 1
            frozen.write_json(directory / f"worker{index}_progress.json", {
                "completed": count, "last_task": identifier, "time": time.time(),
                "wall_sec": round(time.monotonic() - start, 3),
                "gen_sec": report["items"][0]["gen_sec"],
            })
            print(f"DONE {identifier} count={count}", flush=True)
    verify_run(work, directory)


def run(work):
    directory = work / "fast_eval"
    if directory.exists() or any((work / f"{model}_report.json").exists() for model in ("off", "on")):
        raise FileExistsError("Refusing to overwrite or implicitly resume evaluation")
    plan = frozen.check_inputs(work)
    if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("Wait for training to release the GPU")
    profile_path = work / "fast_eval_profile.json"
    profiling = json.loads(profile_path.read_text(encoding="utf-8"))
    if not profiling["metrics_exactly_equal"] or profiling["source_sha256"] != frozen.sha256(__file__):
        raise ValueError("Need a matching cache-equivalence check")
    inputs = {str(path): frozen.sha256(path) for path in
              (Path(__file__).resolve(), work / "plan.json", profile_path)}
    for model in ("off", "on"):
        marker = work / f"trained_{model}.json"
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state["step"] != plan["num_iters"]:
            raise ValueError(f"Incomplete training: {model}")
        inputs[str(marker)] = frozen.sha256(marker)
        inputs.update(state["inputs_sha256"])
    if any(frozen.sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Trained checkpoint changed")
    directory.mkdir()
    for name in ("claims", "items"):
        (directory / name).mkdir()
    frozen.write_json(directory / "plan.json", {
        "inputs_sha256": inputs, "max_workers": 4, "items": plan["cases"] * len(frozen.SEEDS) * 2,
        "note": "Same frozen evaluator, cases, five seeds, CFG, timesteps, ASR and QC. "
                "One-item reports; cached identical reference embeddings and persistent Whisper. "
                "Exclusive task claims, no automatic retry or production promotion. "
                "ASR stochastic fallback remains unchanged, not a bitwise reproducibility guarantee.",
    })
    processes = []
    try:
        for index in range(4):
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).strip())
            if free < 12 * 1024:
                break
            with (directory / f"worker{index}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "worker",
                     "--work", str(work), "--index", str(index)],
                    cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            while not (directory / f"worker{index}_progress.json").exists() and process.poll() is None:
                if any(child.poll() not in (None, 0) for child in processes):
                    raise RuntimeError("Evaluation worker failed during startup")
                time.sleep(2)
            if process.poll() not in (None, 0):
                raise RuntimeError(f"Worker {index} failed during startup")
        if not processes:
            raise RuntimeError("Insufficient free GPU memory")
        while True:
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Evaluation worker failed; preserve completed item reports")
            frozen.write_json(work / "status.json", {
                "stage": "evaluate_dynamic", "pids": [process.pid for process in processes
                                                   if process.poll() is None],
                "completed_items": len(list((directory / "items").glob("*.json"))),
                "expected_items": plan["cases"] * len(frozen.SEEDS) * 2,
                "time": time.time(),
            })
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(10)
        verify_run(work, directory)
        cases = frozen._read_manifest(work / "cases.jsonl")
        frozen.evaluation.EVAL_DIR = directory / "raw_reports"
        for model in ("off", "on"):
            paths = []
            for identifier, target, case, seed in jobs(cases):
                if target != model:
                    continue
                path = directory / "items" / f"{identifier}.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                if len(report["items"]) != 1:
                    raise ValueError(f"Unexpected task size: {identifier}")
                row = report["items"][0]
                if row["seed"] != seed or any(row.get(key) != value for key, value in case.items()):
                    raise ValueError(f"Task conditions changed: {identifier}")
                if report["wav_sha256"] != {row["wav"]: frozen.sha256(row["wav"])}:
                    raise ValueError(f"Task WAV changed: {identifier}")
                paths.append(str(path))
            report = frozen.evaluation.merge_reports(paths)
            frozen.verify_report(report, plan, cases, model)
            report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"]) for row in report["items"]}
            frozen.write_json(work / f"{model}_report.json", report)
        frozen.collect(work, ("base", "r23", "off", "on"), work / "comparison.json")
        frozen.write_json(work / "status.json", {
            "stage": "evaluate_complete", "time": time.time(), "production_changed": False,
            "scheduler": str(directory),
        })
    except BaseException as error:
        frozen.write_json(work / "status.json", {
            "stage": "evaluate_failed", "time": time.time(), "error": str(error),
        })
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
    parser.add_argument("action", choices=("profile", "run", "worker"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=range(4))
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.work, args.work / "fast_eval", args.index)
    else:
        {"profile": profile, "run": run}[args.action](args.work)
