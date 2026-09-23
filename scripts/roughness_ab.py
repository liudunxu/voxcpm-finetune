"""Pinned base/r8 acoustic diagnostic; isolated Linux GPU only, no training."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from voxft import infer
from voxft.paths import ROOT, VOXCPM_REPO

from quality_probe import write_json
from ref_ab import runtime_environment, sha256


def check_inputs(bundle, base, lora):
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    cases = plan["cases"]
    if (len(cases) != 2 or len({case["case_id"] for case in cases}) != 2
            or plan["generation_count"] != 12
            or plan["retry_badcase"] is not False):
        raise ValueError("Expected two frozen cases and twelve single generations")
    for case in cases:
        if (case["seeds"] != [case["seed"] + offset for offset in (0, 1, 7)]
                or not case["case_id"].replace("_", "").isalnum()):
            raise ValueError("Invalid case identity or adjacent seeds")
    references = {case["reference"]: case["reference_sha256"] for case in cases}
    for directory, manifest in ((bundle, references), (base, plan["base_sha256"]),
                                (lora, plan["lora_sha256"])):
        for name, expected in manifest.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid input member path")
            if sha256(directory / name) != expected:
                raise ValueError(f"Frozen input changed: {directory / name}")
    if not plan["base_sha256"] or not plan["lora_sha256"]:
        raise ValueError("Model fingerprints are required")
    return plan


def run(bundle, base, lora, output):
    plan = check_inputs(bundle, base, lora)
    if sys.platform != "linux" or not infer.torch.cuda.is_available():
        raise RuntimeError("Use an isolated Linux CUDA worker, not the local workstation")
    output.mkdir(parents=True, exist_ok=False)
    runtime_environment(output)
    shutil.copy2(bundle / "plan.json", output / "plan.json")
    report = {
        "status": "running", "scope": plan["scope"],
        "plan_sha256": sha256(bundle / "plan.json"),
        "model_files_verified": True, "production_identity_verified": False,
        "normalize": False, "denoise": False, "retry_badcase": False,
        "language_quality": "unverified", "items": [],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__).resolve(), ROOT / "src/voxft/infer.py",
                         ROOT / "src/voxft/lora/merge.py",
                         *sorted((VOXCPM_REPO / "src").rglob("*.py")))
        },
    }
    write_json(output / "report.json", report)
    previous_output = infer.CHECKPOINT_DIR
    infer.CHECKPOINT_DIR = output
    try:
        for case in plan["cases"]:
            for seed in case["seeds"]:
                base_take, r8_take, loading = infer.synthesize_ab(
                    case["text"], str(base), str(lora),
                    ref_audio=str(bundle / case["reference"]),
                    cfg_value=case["cfg_value"],
                    inference_timesteps=case["inference_timesteps"],
                    seed=seed, control=case["control"],
                )
                for arm, (filename, seconds) in (("base", base_take), ("r8", r8_take)):
                    import soundfile as sf

                    audio = sf.info(filename)
                    if audio.frames <= 0:
                        raise ValueError("Empty model output")
                    destination = output / "auditions" / f"{case['case_id']}_{seed}_{arm}.wav"
                    shutil.move(filename, destination)
                    report["items"].append({
                        "case_id": case["case_id"], "arm": arm, "seed": seed,
                        "wav": str(destination.relative_to(output)),
                        "sha256": sha256(destination), "seconds": seconds,
                        "duration": audio.duration, "sample_rate": audio.samplerate,
                        "loading": loading,
                    })
                    write_json(output / "report.json", report)
                print(case["case_id"], seed, "paired", flush=True)
        check_inputs(bundle, base, lora)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        infer.CHECKPOINT_DIR = previous_output
        write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.check_only:
        check_inputs(arguments.bundle, arguments.base, arguments.lora)
        print("Frozen model and reference files verified; no generation")
    else:
        run(arguments.bundle, arguments.base, arguments.lora, arguments.output)
