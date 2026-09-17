"""Compare default and zero-temperature ASR on six frozen WAVs; never regenerate TTS."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import time

from voxft import eval as evaluation
from voxft.data.pipeline import _whisper_model
from voxft.paths import CHECKPOINT_DIR

from quality_probe import write_json
from ref_ab import runtime_environment, sha256

PROFILES = {"default": {}, "temperature_zero": {"temperature": 0.0}}


def summarize(items, samples):
    summaries = {}
    for profile in PROFILES:
        rows = [row for row in items if row["profile"] == profile]
        expected = {(sample["model"], sample["case_id"], sample["seed"], repeat)
                    for sample in samples for repeat in range(3)}
        actual = {(row["model"], row["case_id"], row["seed"], row["repeat"]) for row in rows}
        if len(rows) != len(expected) or actual != expected:
            raise ValueError(f"Incomplete or duplicate repeatability results: {profile}")
        groups = []
        for sample in samples:
            group = [row for row in rows if all(row[key] == sample[key]
                     for key in ("model", "case_id", "seed"))]
            errors = [row["cer"] for row in group]
            groups.append({
                **{key: sample[key] for key in ("model", "case_id", "seed", "original_cer")},
                "unique_hypotheses": len({row["hyp"] for row in group}),
                "unique_normalized_hypotheses": len({evaluation._norm(row["hyp"]) for row in group}),
                "cer_min": min(errors), "cer_max": max(errors),
                "cer_range": round(max(errors) - min(errors), 6),
                "reported_temperatures": sorted({segment["temperature"]
                                                for row in group for segment in row["segments"]}),
            })
        summaries[profile] = groups
    return summaries


def prepare(work):
    if not (work / "done").is_file():
        raise ValueError("Wait for the original evaluation to finish")
    previous = json.loads((work / "diagnostic_sample/plan.json").read_text())
    selected = [row for row in previous["samples"]["diagnostic"]
                if any(reason["metric"] == "cer" for reason in row["reasons"])][:3]
    if len(selected) != 3:
        raise ValueError("Expected three frozen CER diagnostic cases")
    reports, indices = evaluation._review_reports(str(work / "base_report.json"),
                                                 str(work / "r8_report.json"))
    samples = []
    for who, model in (("a", "base"), ("b", "r8")):
        path = work / f"{model}_report.json"
        if sha256(path) != previous["source_reports_sha256"][model]:
            raise ValueError(f"Original report changed: {model}")
        if reports[who]["asr_model"] != "large-v3":
            raise ValueError("Expected original large-v3 ASR")
        for selection in selected:
            row = indices[who][selection["case_id"], selection["seed"]]
            samples.append({
                **{key: row[key] for key in ("case_id", "seed", "lang", "text", "wav")},
                "model": model, "original_cer": row["cer"], "audio_sha256": sha256(row["wav"]),
            })
    return {
        "samples": samples, "profiles": PROFILES, "repeats": 3,
        "source_reports_sha256": previous["source_reports_sha256"],
        "note": "Six outcome-selected WAVs, not a random quality sample. Same model/VAD/language; "
                "only temperature differs. Repeat order forward/reverse/forward. No TTS, "
                "no training, no score replacement or automatic ASR-default change. "
                "Three identical decodes do not establish accuracy or universal determinism. "
                "Segment temperature is the library's reported value, not always the selected "
                "attempt's temperature when all fallback attempts fail.",
    }


def run(work):
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    output = work / "asr_repeatability"
    if output.exists():
        raise FileExistsError(output)
    plan = prepare(work)
    if os.environ.get("VOXFT_WHISPER_MODEL_LARGE"):
        raise ValueError("Custom ASR override needs a separate frozen model audit")
    model_path = Path(download_model("large-v3", local_files_only=True))
    plan["asr_model_path"] = str(model_path)
    plan["asr_model_sha256"] = {str(path): sha256(path) for path in model_path.iterdir() if path.is_file()}
    source_paths = {Path(__file__), Path(evaluation.__file__)}
    source_paths.update(Path(inspect.getfile(function))
                        for function in (_whisper_model, write_json, sha256, WhisperModel.transcribe))
    plan["source_sha256"] = {str(path): sha256(path) for path in sorted(source_paths)}
    plan["versions"] = {name: version(name) for name in ("faster-whisper", "ctranslate2", "torch", "av")}
    output.mkdir()
    write_json(output / "plan.json", plan)
    shutil.copyfile(__file__, output / "source.py")
    runtime_environment(output)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
    result = {"items": [], "complete": False, "plan_sha256": sha256(output / "plan.json")}
    try:
        write_json(output / "status.json", {"stage": "loading_asr", "time": time.time()})
        print("Frozen six WAVs; loading cached ASR for 36 decodes", flush=True)
        whisper = _whisper_model("th", str(model_path))
        result["device"] = getattr(whisper.model, "device", None)
        result["compute_type"] = getattr(whisper.model, "compute_type", None)
        for profile, overrides in PROFILES.items():
            write_json(output / "status.json", {"stage": profile, "time": time.time()})
            for repeat in range(3):
                ordered = plan["samples"] if repeat % 2 == 0 else list(reversed(plan["samples"]))
                for sample in ordered:
                    if sha256(sample["wav"]) != sample["audio_sha256"]:
                        raise ValueError(f"Frozen audio changed: {sample['wav']}")
                    language = None if sample["lang"] in evaluation.AUTO_DETECT_LANGS else sample["lang"]
                    started = time.monotonic()
                    print(f"Decoding {profile}/{repeat} {sample['model']} "
                          f"{sample['case_id']}/{sample['seed']}", flush=True)
                    segments, info = whisper.transcribe(sample["wav"], language=language,
                                                         vad_filter=True, **overrides)
                    segments = [asdict(segment) for segment in segments]
                    hyp = " ".join(segment["text"].strip() for segment in segments)
                    result["items"].append({
                        **sample, "profile": profile, "repeat": repeat, "hyp": hyp, "segments": segments,
                        "cer": round(evaluation._error_rate(evaluation._norm(hyp),
                                                           evaluation._norm(sample["text"])), 6),
                        "transcription_options": asdict(info.transcription_options),
                        "vad_options": asdict(info.vad_options) if info.vad_options is not None else None,
                        "duration_after_vad": info.duration_after_vad,
                        "seconds": round(time.monotonic() - started, 3),
                    })
                    write_json(output / "results.json", result)
                    print(f"ASR repeatability {len(result['items'])}/36 {profile} "
                          f"{sample['model']} {sample['case_id']} repeat={repeat}", flush=True)
        for path, expected in {**plan["source_sha256"], **plan["asr_model_sha256"]}.items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen code/model changed: {path}")
        result["summary"] = summarize(result["items"], plan["samples"])
        result["complete"] = True
        write_json(output / "results.json", result)
        write_json(output / "status.json", {"stage": "complete", "time": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"stage": "failed", "error": str(error), "time": time.time()})
        raise


def align_vi(work):
    import soundfile

    source = work / "asr_repeatability/results.json"
    previous = json.loads(source.read_text())
    plan = json.loads((source.parent / "plan.json").read_text())
    if not previous["complete"]:
        raise ValueError("Repeatability probe is not complete")
    selected = [row for row in previous["items"] if row["profile"] == "temperature_zero"
                and row["repeat"] == 0 and row["case_id"] == "fleurs_test_vi_1695"]
    if len(selected) != 2 or {row["model"] for row in selected} != {"base", "r8"}:
        raise ValueError("Expected the frozen base/r8 Vietnamese pair")
    for path, expected in plan["asr_model_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"ASR model changed: {path}")
    output = work / "asr_alignment"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "source.py")
    runtime_environment(output)
    result = {
        "source_results_sha256": sha256(source), "source_code_sha256": sha256(__file__),
        "asr_model_sha256": plan["asr_model_sha256"], "complete": False, "items": [],
        "note": "Same two WAVs and zero-temperature settings; only word_timestamps=True added. "
                "No manual text deletion, no TTS generation, no original report overwrite. "
                "ASR alignment sensitivity is not native confirmation or model improvement.",
    }
    write_json(output / "results.json", result)
    whisper = _whisper_model("vi", plan["asr_model_path"])
    for row in selected:
        if sha256(row["wav"]) != row["audio_sha256"]:
            raise ValueError(f"Frozen audio changed: {row['wav']}")
        duration = soundfile.info(row["wav"]).duration
        segments, info = whisper.transcribe(row["wav"], language="vi", vad_filter=True,
                                            temperature=0.0, word_timestamps=True)
        segments = [asdict(segment) for segment in segments]
        hyp = " ".join(segment["text"].strip() for segment in segments)
        baseline_end = max((segment["end"] for segment in row["segments"]), default=0)
        aligned_end = max((segment["end"] for segment in segments), default=0)
        result["items"].append({
            **{key: row[key] for key in ("model", "case_id", "seed", "wav", "audio_sha256")},
            "source_audio_sec": duration, "baseline_cer": row["cer"], "baseline_hyp": row["hyp"],
            "baseline_end_sec": baseline_end, "aligned_end_sec": aligned_end,
            "baseline_timestamp_overrun_sec": max(0, baseline_end - duration),
            "aligned_timestamp_overrun_sec": max(0, aligned_end - duration),
            "hyp": hyp, "segments": segments, "transcription_options": asdict(info.transcription_options),
            "cer": round(evaluation._error_rate(evaluation._norm(hyp),
                                               evaluation._norm(row["text"])), 6),
        })
        write_json(output / "results.json", result)
        print(f"Word alignment {len(result['items'])}/2 {row['model']}", flush=True)
    result["complete"] = True
    result["finished"] = time.time()
    write_json(output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "holdout_eval_20260917")
    parser.add_argument("--align-vi", action="store_true")
    args = parser.parse_args()
    (align_vi if args.align_vi else run)(args.work)
