"""Frozen October 28 base/r26 evaluation only; ID first, no retries or training."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import control_ab_run as frozen
from control_ab_fast_eval import cached_services, claim
from voxft.data import pipeline
from voxft.qc import audio as qc

DATA = Path("/root/autodl-tmp")
CPU_AUDIT = DATA / "finetune_readiness_20261028/cpu"
HUB = DATA / "hf_home/hub"
BASE = HUB / "models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915"
WHISPER = HUB / "models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
SPEAKER = HUB / "models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"
R26 = DATA / "voxft_ckpt/lora_omni5_r26_r18_inoc10/latest"
CASES = frozen.ROOT / "eval_cases/fresh_text_holdout_20261028.jsonl"
CASES_SHA = "10566052db1254a68d8c2e295248b0283cde4e22b410e4762b9231339df55267"
MODELS = ("base", "r26")
SETTINGS = {
    "evaluation_seeds": [42, 43, 44, 45, 46], "cfg_value": 1.8,
    "inference_timesteps": 20, "retry_badcase": False, "asr_model": "large-v3",
}
WEIGHTS = {
    BASE / "model.safetensors": "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d",
    BASE / "audiovae.pth": "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1",
    R26 / "lora_weights.safetensors": "65d79023a7e74487b5ba0f81a633f1dd5e0ca9cde721b6e9ed75a31d0357d333",
    R26 / "lora_config.json": "bb0d33fede7c0f40960f493da0c30d455415ccd5b72a0aed659b0d024e79b456",
    R26 / "training_state.json": "ae78b7ca42d46f9c404451521b20036360ef3284653611647e1f93622e8a546d",
}


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def prepare(work):
    if work.exists():
        raise FileExistsError(f"Refusing to overwrite or implicitly resume: {work}")
    summary_path = CPU_AUDIT / "summary.json"
    audio_path = CPU_AUDIT / "audio_audit.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audio = json.loads(audio_path.read_text(encoding="utf-8"))["files"]
    if (frozen.sha256(CASES) != CASES_SHA
            or summary["fresh_evaluation"]["cases_sha256"] != CASES_SHA
            or any(summary.get(key) != value for key, value in SETTINGS.items())):
        raise ValueError("Frozen CPU evaluation plan changed")
    cases = frozen._read_manifest(CASES)
    references = {row["ref_audio"] for row in cases}
    expected = {**WEIGHTS, **{Path(path): audio[path]["file_sha256"] for path in references}}
    for path, digest in expected.items():
        if frozen.sha256(path) != digest:
            raise ValueError(f"Frozen weight/reference changed: {path}")
    paths = {CASES, summary_path, audio_path, *expected}
    for directory in (BASE, WHISPER, SPEAKER):
        paths.update(path for path in directory.iterdir() if path.is_file())
    for directory in (frozen.ROOT / "src", frozen.ROOT / "scripts", frozen.VOXCPM_REPO / "src"):
        paths.update(directory.rglob("*.py"))
        paths.update(directory.rglob("*.sh"))
    inputs = {str(path): frozen.sha256(path) for path in sorted(paths)}
    work.mkdir(parents=True)
    shutil.copyfile(CASES, work / "cases.jsonl")
    inputs[str(work / "cases.jsonl")] = CASES_SHA
    save_new(work / "plan.json", {
        **SETTINGS, "base_path": str(BASE), "targets": {"base": "base", "r26": str(R26)},
        "cases": len(cases), "generation_budget": 1800, "max_workers": 2,
        "asr_path": str(WHISPER), "speaker_model_path": str(SPEAKER),
        "inputs_sha256": inputs, "training_authorized": False, "production_changed": False,
        "note": "Historical r26, not a model trained on quarantined candidates. "
                "ID-first scheduling preserves case bytes and all five seeds. "
                "Read-speech holdout only; native quality and reference identity unverified. "
                "No downloads, additional regression suites, automatic retries or promotion.",
    })
    checked(work)
    print(f"PREPARED {work}: 1800 outputs", flush=True)


def checked(work):
    plan = frozen.check_inputs(work)
    if (any(plan.get(key) != value for key, value in SETTINGS.items())
            or plan["targets"] != {"base": "base", "r26": str(R26)}
            or plan["base_path"] != str(BASE)
            or plan["asr_path"] != str(WHISPER) or plan["speaker_model_path"] != str(SPEAKER)
            or plan["cases"] != 180 or plan["generation_budget"] != 1800
            or plan["max_workers"] != 2 or plan["training_authorized"] is not False
            or frozen.sha256(work / "cases.jsonl") != CASES_SHA
            or SETTINGS["evaluation_seeds"] != frozen.SEEDS):
        raise ValueError("Evaluation scope/settings changed")
    return plan


def tasks(cases, model):
    if model not in MODELS:
        raise ValueError(f"Unapproved model: {model}")
    for position, case in sorted(enumerate(cases), key=lambda pair: pair[1]["lang"] != "id"):
        for seed in SETTINGS["evaluation_seeds"]:
            yield f"{model}_{position:04d}_{seed}", case, seed


def gpu_services(plan, cases, model):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
    whisper = pipeline._whisper_model("th", "large-v3")
    if whisper.model.device != "cuda":
        raise RuntimeError("Whisper is not using CUDA")
    for path in sorted({row["ref_audio"] for row in cases}):
        if qc.speaker_embedding(*pipeline.load_wav_mono(path)) is None:
            raise RuntimeError(f"Reference speaker embedding unavailable: {path}")
    loaded = frozen.evaluation.infer.get_model(
        plan["base_path"], None if model == "base" else plan["targets"][model])
    if next(loaded.tts_model.parameters()).device.type != "cuda":
        raise RuntimeError("TTS is not using CUDA")
    return {
        "target": plan["targets"][model], "base": plan["base_path"],
        "gpu": torch.cuda.get_device_name(), "asr_device": whisper.model.device,
        "asr_compute_type": whisper.model.compute_type, "tts_device": "cuda",
        "lora_parameter_tensors": sum("lora_" in name for name, _ in loaded.tts_model.named_parameters()),
        "environment": {key: os.environ.get(key) for key in (
            "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VOXFT_WHISPER_MODEL_LARGE",
            "VOXFT_SPK_EMB_MODEL", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "faster-whisper", "ctranslate2", "transformers", "soundfile", "numpy")},
    }


def worker(work, model):
    plan = checked(work)
    cases = frozen._read_manifest(work / "cases.jsonl")
    runtime = work / f"runtime_{model}"
    frozen.runtime_environment(runtime)
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VOXFT_WHISPER_MODEL_LARGE=plan["asr_path"], VOXFT_SPK_EMB_MODEL=plan["speaker_model_path"],
        TORCHINDUCTOR_CACHE_DIR=str(runtime / "inductor"))
    frozen.evaluation.EVAL_DIR = work / "raw_reports"
    frozen.evaluation.infer.CHECKPOINT_DIR = work
    with cached_services(cases):
        save_new(work / f"{model}_runtime.json", gpu_services(plan, cases, model))
        for count, (identifier, case, seed) in enumerate(tasks(cases, model), 1):
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError("Less than 4 GiB free; preserve outputs")
            if not claim(work / "claims", identifier):
                raise RuntimeError(f"Task already claimed; no automatic retry: {identifier}")
            started = time.monotonic()
            report = frozen.evaluation.evaluate(
                plan["targets"][model], "th", [case], base=plan["base_path"],
                seeds=[seed], cfg_value=plan["cfg_value"],
                inference_timesteps=plan["inference_timesteps"])
            report["wav_sha256"] = {row["wav"]: frozen.sha256(row["wav"]) for row in report["items"]}
            save_new(work / "items" / f"{identifier}.json", report)
            frozen.write_json(work / f"{model}_progress.json", {
                "completed": count, "last_task": identifier, "lang": case["lang"],
                "time": time.time(), "wall_sec": round(time.monotonic() - started, 3),
            })
            print(f"DONE {identifier} lang={case['lang']} completed={count}/900", flush=True)
    checked(work)


def collect(work):
    plan = checked(work)
    cases = frozen._read_manifest(work / "cases.jsonl")
    expected = {f"{identifier}.json" for model in MODELS for identifier, _, _ in tasks(cases, model)}
    if {path.name for path in (work / "items").glob("*.json")} != expected:
        raise ValueError("Incomplete or unexpected item reports")
    if any((work / f"{model}_report.json").exists() for model in MODELS):
        raise FileExistsError("Refusing to replace a merged report")
    frozen.evaluation.EVAL_DIR = work / "raw_reports"
    for model in MODELS:
        paths = []
        for identifier, case, seed in tasks(cases, model):
            path = work / "items" / f"{identifier}.json"
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
        save_new(work / f"{model}_report.json", report)
    frozen.collect(work, MODELS, work / "comparison.json")


def run(work):
    if (work / "started.json").exists():
        raise FileExistsError("Refusing to overwrite or implicitly resume evaluation")
    plan = checked(work)
    if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is busy; leave existing processes alone")
    if shutil.disk_usage(work).free < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB free")
    if not claim(work, "started"):
        raise FileExistsError("Evaluation already started")
    processes = []
    try:
        for directory in ("items", "claims"):
            (work / directory).mkdir()
        for model in MODELS:
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).strip())
            if free < 12 * 1024:
                raise RuntimeError("Insufficient GPU memory; do not change evaluation conditions")
            with (work / f"{model}.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()),
                     "worker", "--work", str(work), "--model", model],
                    cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            while not (work / f"{model}_progress.json").exists() and process.poll() is None:
                if any(child.poll() not in (None, 0) for child in processes):
                    raise RuntimeError("Worker failed during startup")
                time.sleep(2)
            if process.poll() not in (None, 0):
                raise RuntimeError(f"{model} failed during startup")
        while True:
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("Worker failed; preserve outputs, no automatic retry")
            frozen.write_json(work / "status.json", {
                "stage": "evaluating", "time": time.time(),
                "pids": [process.pid for process in processes if process.poll() is None],
                "completed_items": len(list((work / "items").glob("*.json"))),
                "expected_items": plan["generation_budget"],
            })
            if all(process.poll() is not None for process in processes):
                break
            time.sleep(10)
        collect(work)
        frozen.write_json(work / "status.json", {
            "stage": "complete", "completed_items": plan["generation_budget"], "time": time.time(),
            "training_started": False, "production_changed": False,
        })
    except BaseException as error:
        frozen.write_json(work / "status.json", {
            "stage": "failed", "error": str(error), "time": time.time(),
            "completed_items": len(list((work / "items").glob("*.json"))),
            "training_started": False, "production_changed": False,
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
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.action == "worker":
        if args.model is None:
            parser.error("worker requires --model")
        worker(args.work, args.model)
    else:
        {"prepare": prepare, "run": run}[args.action](args.work)
