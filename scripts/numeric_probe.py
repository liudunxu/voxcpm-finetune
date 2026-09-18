"""Frozen r8 input-representation diagnostic; no training or promotion."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
from statistics import mean
import time

from voxft import eval as evaluation
from voxft.paths import ROOT, VOXCPM_REPO

from holdout_eval import check_inputs
from quality_probe import write_json
from ref_ab import runtime_environment, sha256


def common_scores(row, verbalize):
    expected = verbalize(row["source_text"], row["lang"])
    actual = verbalize(row["hyp"], row["lang"])
    return {
        "common_expected": expected, "common_hyp": actual,
        "common_cer": evaluation._error_rate(evaluation._norm(actual), evaluation._norm(expected)),
        "common_suspected_truncation": evaluation._is_truncated(actual, expected),
        "common_over_read": evaluation._over_read(actual, expected),
    }


def collect(work):
    plan = check_inputs(work)
    spec = importlib.util.spec_from_file_location("frozen_locale_text", work / "locale_text.py")
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    reports = {}
    for arm, cases in plan["arms"].items():
        report = json.loads((work / f"{arm}_report.json").read_text())
        expected = {(case["case_id"], seed): case for case in cases for seed in plan["seeds"]}
        rows = {(row["case_id"], row["seed"]): row for row in report["items"]}
        if len(rows) != len(report["items"]) or rows.keys() != expected.keys():
            raise ValueError(f"Incomplete report: {arm}")
        for key in ("target", "base", "cfg_value", "inference_timesteps", "retry_badcase"):
            if report[key] != plan[key]:
                raise ValueError(f"Changed {key}: {arm}")
        for key, row in rows.items():
            if any(row.get(field) != value for field, value in expected[key].items()):
                raise ValueError(f"Changed input: {arm}/{key}")
            row.update(common_scores(row, normalizer.verbalize_locale_numbers))
            row["wav_sha256"] = sha256(row["wav"])
        reports[arm] = rows
    pairs = []
    for key, raw in reports["raw"].items():
        words = reports["words"][key]
        pairs.append({
            "case_id": key[0], "seed": key[1], "lang": raw["lang"], "numeric": raw["numeric"],
            "common_cer_raw": raw["common_cer"], "common_cer_words": words["common_cer"],
            "delta_cer": words["common_cer"] - raw["common_cer"],
            "delta_audio_sec": words["audio_sec"] - raw["audio_sec"],
            "same_input": raw["text"] == words["text"],
            "same_wav": raw["wav_sha256"] == words["wav_sha256"],
        })
    summaries = {}
    for lang in ("th", "vi", "id", "ms"):
        rows = [row for row in pairs if row["lang"] == lang and row["numeric"]]
        summaries[lang] = {
            "texts": len({row["case_id"] for row in rows}), "pairs": len(rows),
            "common_cer_raw": mean(row["common_cer_raw"] for row in rows),
            "common_cer_words": mean(row["common_cer_words"] for row in rows),
            "delta_audio_sec": mean(row["delta_audio_sec"] for row in rows),
        }
    write_json(work / "comparison.json", {
        "by_lang_numeric": summaries, "pairs": pairs,
        "items": {arm: list(rows.values()) for arm, rows in reports.items()},
        "note": "Development diagnostic, not native-language acceptance. Shared numeric canonicalization "
                "can collapse or misread number spellings; raw hypotheses and original scores remain intact. "
                "Do not compare original arm-specific CER, or treat five seeds as five independent texts.",
    })
    return all(row["same_input"] and row["same_wav"] for row in pairs if not row["numeric"])


def run(work):
    if (work / "plan.json").exists():
        raise FileExistsError("Run already frozen; inspect status instead of restarting")
    specification = json.loads((work / "inputs.json").read_text())
    if (specification["seeds"] != [42, 43, 44, 45, 49]
            or {arm: len(cases) for arm, cases in specification["arms"].items()}
            != {"raw": 12, "words": 12, "legacy_ms": 1}):
        raise ValueError("Expected the frozen 125-output pilot")
    if shutil.disk_usage(work).free < 2 * 1024**3:
        raise RuntimeError("Need at least 2GiB free")
    runtime_environment(work / "runtime")
    source = json.loads((work / "source_identity.json").read_text())
    for filename, digest in source["artifact_sha256"].items():
        if sha256(work / filename) != digest:
            raise ValueError(f"Transferred artifact changed: {filename}")
    paths = {work / name for name in source["artifact_sha256"]} | {work / "source_identity.json"}
    paths |= {Path(case["ref_audio"]) for cases in specification["arms"].values() for case in cases}
    paths |= {path for directory in (specification["base"], specification["target"])
              for path in Path(directory).rglob("*") if path.is_file()}
    paths |= set((ROOT / "src").rglob("*.py")) | set((VOXCPM_REPO / "src").rglob("*.py"))
    paths |= {Path(__file__).resolve(), ROOT / "scripts/ref_ab.py",
              ROOT / "scripts/holdout_eval.py", ROOT / "scripts/quality_probe.py", ROOT / "uv.lock"}
    write_json(work / "plan.json", {
        **specification, "inputs_sha256": {str(path): sha256(path) for path in sorted(paths)},
    })
    completed = 0
    try:
        for arm, cases in specification["arms"].items():
            check_inputs(work)

            def progress(message):
                print(f"{arm}: {message}", flush=True)
                if shutil.disk_usage(work).free < 1024**3:
                    raise RuntimeError("Less than 1GiB free")
                match = re.match(r"\[(\d+)/(\d+)\]", message)
                if match:
                    write_json(work / "status.json", {
                        "stage": "evaluation", "arm": arm, "completed": completed + int(match[1]),
                        "total": 125, "message": message, "time": time.time(),
                    })

            report = evaluation.evaluate(
                specification["target"], "th", cases, base=specification["base"],
                seeds=specification["seeds"], cfg_value=specification["cfg_value"],
                inference_timesteps=specification["inference_timesteps"], progress=progress,
            )
            write_json(work / f"{arm}_report.json", report)
            completed += len(report["items"])
        controls_match = collect(work)
        write_json(work / "status.json", {
            "stage": "done" if controls_match else "control_mismatch",
            "completed": completed, "total": 125, "controls_identical": controls_match,
            "time": time.time(),
        })
        if controls_match:
            (work / "done").write_text("125 outputs and common-reference diagnostics complete\n")
    except Exception as error:
        write_json(work / "failure.json", {"error": repr(error), "time": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "collect"))
    parser.add_argument("--work", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "run":
        run(arguments.work)
    else:
        collect(arguments.work)
