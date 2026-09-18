"""Reuse saved audio for paired diagnostics and review; never regenerate TTS."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from voxft import eval as evaluation
from voxft.data.pipeline import _read_manifest
from voxft.paths import CHECKPOINT_DIR

from acoustic_review import select_cases
from quality_probe import write_json
from ref_ab import runtime_environment, sha256

VOICE_CASES = ("fleurs_test_th_1739", "fleurs_test_th_1707",
               "fleurs_test_vi_1856", "fleurs_test_vi_1874")
VOICE_SEEDS = (42, 43, 44, 45, 49)


def voice_window_bounds(sample_count, sample_rate, edges, position):
    if sample_rate <= 0 or sample_count <= 0 or any(
            value is None or not math.isfinite(value) or value < 0 for value in edges.values()):
        raise ValueError("Invalid audio length/rate or edge measurements")
    start = round(edges["head_silence_sec"] * sample_rate)
    stop = sample_count - round(edges["tail_silence_sec"] * sample_rate)
    width = round(3.0 * sample_rate)
    if width <= 0 or stop - start < width:
        raise ValueError("Less than three seconds inside speech boundaries")
    starts = {"start": start, "middle": (start + stop - width) // 2, "end": stop - width}
    if position not in starts:
        raise ValueError(f"Unknown voice-window position: {position}")
    return starts[position], starts[position] + width


def select_samples(cases, paired_cases, indices):
    controls = [{"case_id": case_id, "seed": 42} for case_id in select_cases(cases)]
    selected = {}
    for metric, direction, limit in (("cer", 1, 10), ("speaker_sim", -1, 5)):
        ranked = sorted(
            (row for row in paired_cases if row[metric] is not None
             and direction * (row[metric][1] - row[metric][0]) > 0),
            key=lambda row: (-direction * (row[metric][1] - row[metric][0]), row["case_id"]),
        )
        for row in ranked[:limit]:
            case_id = row["case_id"]
            if case_id not in selected:
                seeds = sorted(seed for identifier, seed in indices["a"] if identifier == case_id)
                seed = max(seeds, key=lambda seed: direction * (
                    indices["b"][case_id, seed][metric] - indices["a"][case_id, seed][metric]))
                selected[case_id] = {"case_id": case_id, "seed": seed, "reasons": []}
            selected[case_id]["reasons"].append({
                "metric": metric, "mean_base": row[metric][0], "mean_r8": row[metric][1],
            })
    return {"diagnostic": list(selected.values()), "random_control": controls}


def collect(work):
    output = work / "diagnostic_sample"
    if output.exists():
        raise FileExistsError(output)
    comparison = json.loads((work / "comparison.json").read_text())
    plan = json.loads((work / "plan.json").read_text())
    cases_path = work / "cases.jsonl"
    if sha256(cases_path) != plan["inputs_sha256"][str(cases_path)]:
        raise ValueError("Frozen cases changed")
    sources = {model: work / f"{model}_report.json" for model in ("base", "r8")}
    for model, source in sources.items():
        if sha256(source) != comparison["inputs_sha256"][model]:
            raise ValueError(f"Collected report changed: {model}")
    reports, indices = evaluation._review_reports(str(sources["base"]), str(sources["r8"]))
    cases = _read_manifest(cases_path)
    expected = {(case["case_id"], seed) for case in cases for seed in plan["evaluation_seeds"]}
    if expected != indices["a"].keys():
        raise ValueError("Incomplete frozen case/seed coverage")
    samples = select_samples(cases, comparison["r8_minus_base"]["by_case"], indices)
    metadata = {
        "selection_seed": 20260917, "samples": samples,
        "source_cases_sha256": sha256(cases_path),
        "source_reports_sha256": comparison["inputs_sha256"],
        "comparison_sha256": sha256(work / "comparison.json"),
        "script_sha256": sha256(__file__),
        "overlap_case_ids": sorted({row["case_id"] for row in samples["diagnostic"]}
                                   & {row["case_id"] for row in samples["random_control"]}),
        "note": "Diagnostics: top 10 positive case-mean CER deltas plus top 5 ref-similarity "
                "drops; one worst paired seed for the first selecting metric. Not confirmed "
                "acoustic failures. Controls: one case per target/ref-language stratum, "
                "RNG 20260917 and seed 42; chosen after results exist but algorithm uses "
                "metadata only, never excludes diagnostic overlaps. Separate reports, no "
                "online badcase-rate estimate. Full five-seed evidence stays in originals. "
                "No native-quality validation. Samples used for tuning become development cases.",
    }
    exports = {}
    for group, selections in samples.items():
        if not selections:
            raise ValueError(f"Empty review sample: {group}")
        for who, model in (("a", "base"), ("b", "r8")):
            report = deepcopy(reports[who])
            items = [deepcopy(indices[who][row["case_id"], row["seed"]]) for row in selections]
            for item in items:
                if not Path(item["wav"]).is_file():
                    raise FileNotFoundError(item["wav"])
            report.update(items=items, label=f"holdout_{group}_{model}",
                          source_report=str(sources[model]),
                          source_report_sha256=comparison["inputs_sha256"][model],
                          diagnostic_sample=metadata)
            for field in ("cohorts", "human_review"):
                report.pop(field, None)
            report.update(evaluation._report_metrics(items))
            exports[group, model] = report
    output.mkdir()
    write_json(output / "plan.json", metadata)
    saved = {}
    for group in samples:
        saved[group] = {model: evaluation._save_report(exports[group, model])["report_path"]
                        for model in sources}
        evaluation.review_session(saved[group]["base"], saved[group]["r8"])
    write_json(output / "reports.json", saved)
    print(f"Saved review pairs: { {group: len(rows) for group, rows in samples.items()} }", flush=True)


def recheck_asr(work):
    from voxft.data.pipeline import _whisper_model

    output = work / "asr_recheck"
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((work / "diagnostic_sample/plan.json").read_text())
    reports = {}
    for model, expected in plan["source_reports_sha256"].items():
        source = work / f"{model}_report.json"
        if sha256(source) != expected:
            raise ValueError(f"Collected report changed: {model}")
        reports[model] = {(row["case_id"], row["seed"]): row
                          for row in json.loads(source.read_text())["items"]}
    cases = [row for row in plan["samples"]["diagnostic"]
             if any(reason["metric"] == "cer" for reason in row["reasons"])][:3]
    keys = sorted({(row["case_id"], seed) for row in cases for seed in (row["seed"], 49)})
    if not keys:
        raise ValueError("No CER diagnostic cases")
    audio_hashes = {row["wav"]: sha256(row["wav"])
                    for index in reports.values() for key in keys for row in [index[key]]}
    output.mkdir()
    result = {
        "source_reports_sha256": plan["source_reports_sha256"], "audio_sha256": audio_hashes,
        "script_sha256": sha256(__file__), "keys": keys, "asr_model": "large-v3",
        "note": "Same saved audio: repeat original VAD-on ASR, then change only vad_filter=False. "
                "Top 3 CER case regressions, selected worst seed and seed49 within-case control. "
                "Decoder sensitivity only; not native confirmation or new acceptance scores. "
                "Original reports and production remain unchanged.",
        "items": [], "complete": False,
    }
    write_json(output / "results.json", result)
    runtime_environment(work)
    whisper = _whisper_model("th", "large-v3")
    for model, index in reports.items():
        for key in keys:
            row = index[key]
            for vad_filter in (True, False):
                segments, info = whisper.transcribe(
                    row["wav"], language=None if row["lang"] in evaluation.AUTO_DETECT_LANGS
                    else row["lang"], vad_filter=vad_filter)
                segments = [{"start": segment.start, "end": segment.end, "text": segment.text}
                            for segment in segments]
                hyp = " ".join(segment["text"].strip() for segment in segments)
                result["items"].append({
                    "model": model, "case_id": key[0], "seed": key[1], "vad_filter": vad_filter,
                    "wav": row["wav"], "original_cer": row["cer"], "original_hyp": row["hyp"],
                    "hyp": hyp, "segments": segments,
                    "duration_after_vad": getattr(info, "duration_after_vad", None),
                    "cer": evaluation._error_rate(evaluation._norm(hyp), evaluation._norm(row["text"])),
                    "suspected_truncation": evaluation._is_truncated(hyp, row["text"]),
                    "over_read": evaluation._over_read(hyp, row["text"]),
                })
                write_json(output / "results.json", result)
                print(f"ASR recheck {len(result['items'])}/{len(keys) * len(reports) * 2}", flush=True)
    result["complete"] = True
    write_json(output / "results.json", result)


def localize_voice(work, windows=False):
    import inspect
    from importlib.metadata import version
    import os
    import shutil
    import time

    import numpy as np
    from huggingface_hub import snapshot_download
    from voxft.data.pipeline import load_wav_mono
    from voxft.qc import audio as quality
    from voice_stability import summarize_embeddings

    output = work / ("voice_windows" if windows else "local_voice")
    if output.exists():
        raise FileExistsError(output)
    if not (work / "done").is_file():
        raise ValueError("Wait for the original evaluation to finish")
    selection = json.loads((work / "diagnostic_sample/plan.json").read_text())
    reports, indices = evaluation._review_reports(str(work / "base_report.json"),
                                                  str(work / "r8_report.json"))
    keys = [(case_id, seed) for case_id in VOICE_CASES for seed in VOICE_SEEDS]
    samples = {model: [indices[who][key] for key in keys] for who, model in (("a", "base"), ("b", "r8"))}
    inputs, historical = {}, {}
    for model, items in samples.items():
        source = work / f"{model}_report.json"
        if sha256(source) != selection["source_reports_sha256"][model]:
            raise ValueError(f"Original report changed: {model}")
        inputs[str(source)] = sha256(source)
        voice_path = work / "voice_stability" / source.name
        historical[model] = json.loads(voice_path.read_text())["cross_cue"]["groups"]
        inputs[str(voice_path)] = sha256(voice_path)
        for item in items:
            for field in ("wav", "ref_audio"):
                inputs[item[field]] = sha256(item[field])
    name = os.environ.get("VOXFT_SPK_EMB_MODEL", quality.SPK_EMB_MODEL_DEFAULT)
    model_path = Path(name) if Path(name).is_dir() else Path(snapshot_download(name, local_files_only=True))
    os.environ["VOXFT_SPK_EMB_MODEL"] = str(model_path)
    model_hashes = {str(path): sha256(path) for path in model_path.iterdir() if path.is_file()}
    versions = {name: version(name) for name in ("torch", "transformers")}
    full_vectors = {}
    if windows:
        previous_dir = work / "local_voice"
        previous = json.loads((previous_dir / "plan.json").read_text())
        previous_result = json.loads((previous_dir / "results.json").read_text())
        if (not previous_result["complete"]
                or previous_result["plan_sha256"] != sha256(previous_dir / "plan.json")
                or previous["model_sha256"] != model_hashes or previous["versions"] != versions):
            raise ValueError("Completed local-voice baseline/model/version does not match")
        if previous["inputs_sha256"] != inputs:
            raise ValueError("Local-voice baseline inputs changed")
        for path, expected in previous["source_sha256"].items():
            if Path(path) != Path(__file__) and sha256(path) != expected:
                raise ValueError(f"Local-voice shared source changed: {path}")
        for filename in ("plan.json", "results.json", "embeddings.npz"):
            path = previous_dir / filename
            inputs[str(path)] = sha256(path)
        with np.load(previous_dir / "embeddings.npz", allow_pickle=False) as arrays:
            for model, items in samples.items():
                if [(row["case_id"], row["seed"], row["wav"]) for row in items] != [
                        (row["case_id"], row["seed"], row["wav"]) for row in previous["samples"][model]]:
                    raise ValueError("Local-voice baseline sample order does not match")
                full_vectors[model] = arrays[f"edge_trim_{model}"]
                if len(full_vectors[model]) != len(items) or not np.isfinite(full_vectors[model]).all():
                    raise ValueError("Invalid cached full-utterance embeddings")
    sources = {Path(__file__), Path(evaluation.__file__), Path(quality.__file__),
               *(Path(inspect.getfile(function)) for function in
                 (load_wav_mono, summarize_embeddings, write_json, sha256))}
    plan = {
        "samples": samples, "inputs_sha256": inputs, "model_sha256": model_hashes,
        "source_sha256": {str(path): sha256(path) for path in sources},
        "versions": versions,
        "profiles": ["start", "middle", "end"] if windows else ["raw", "edge_trim"],
        "raw_tolerance": 0.00001,
        "note": "Four preidentified TH/VI cues x five seeds x base/r8. Raw versus in-memory "
                "edge_silence trimming only; internal pauses and reference audio unchanged. "
                "No gain/EQ/pitch changes, no WAV writes or TTS. Reproduce all ten raw cross-cue "
                "groups per model before interpreting trim sensitivity. Outcome-selected "
                "diagnostics, not native judgement or verified speaker-identity labels.",
    }
    if windows:
        plan["window_sec"] = 3.0
        plan["note"] = (
            "Same forty saved WAVs as local_voice; cached full-utterance embeddings reused. "
            "Three-second start/middle/end windows inside measured speech boundaries; "
            "windows may overlap, retain internal pauses, and are NOT word/phoneme aligned. "
            "No TTS, resynthesis, new ASR, waveform writes or production changes. "
            "Window-versus-full cosine is duration/content sensitive, not an identity gate. "
            "Full-cue review exports keep original audio/metrics and unverified human fields.")
    output.mkdir()
    write_json(output / "plan.json", plan)
    shutil.copyfile(__file__, output / "source.py")
    runtime_environment(output)
    loaded = quality._spk_model()
    if loaded is None or loaded[1].training:
        raise RuntimeError("Speaker model unavailable or not in evaluation mode")
    result = {
        "complete": False, "plan_sha256": sha256(output / "plan.json"),
        "device": str(loaded[1].device), "torch_threads": loaded[2].get_num_threads(),
        "summaries": {}, "items": [],
    }
    write_json(output / "results.json", result)
    embeddings = {}
    total = sum(len(items) for items in samples.values()) * len(plan["profiles"])
    for profile in plan["profiles"]:
        result["summaries"][profile] = {}
        for model, items in samples.items():
            vectors = []
            for index, item in enumerate(items):
                if sha256(item["wav"]) != inputs[item["wav"]]:
                    raise ValueError(f"Frozen audio changed: {item['wav']}")
                wav, sample_rate = load_wav_mono(item["wav"])
                edges = quality.edge_silence(wav, sample_rate)
                if any(value is None or not np.isfinite(value) or value < 0 for value in edges.values()):
                    raise ValueError(f"Unusable edge measurements: {item['wav']}")
                start, stop = 0, len(wav)
                if windows:
                    start, stop = voice_window_bounds(len(wav), sample_rate, edges, profile)
                elif profile == "edge_trim":
                    start = round(edges["head_silence_sec"] * sample_rate)
                    stop = len(wav) - round(edges["tail_silence_sec"] * sample_rate)
                wav = wav[start:stop]
                vector = quality.speaker_embedding(wav, sample_rate)
                if vector is None or not np.isfinite(vector).all():
                    raise ValueError(f"Missing/invalid speaker embedding: {item['wav']}")
                vectors.append(vector)
                result["items"].append({
                    "profile": profile, "model": model,
                    **{key: item[key] for key in ("case_id", "seed", "wav")},
                    **edges, "embedded_audio_sec": len(wav) / sample_rate,
                    "start_sec": start / sample_rate, "end_sec": stop / sample_rate,
                    **({"cosine_to_own_full": float(vector @ full_vectors[model][index]),
                        "speech_ratio": quality.speech_ratio(wav, sample_rate),
                        "band_ratio_2_8k": quality.band_ratio_2_8k(wav, sample_rate)}
                       if windows else {}),
                })
                write_json(output / "results.json", result)
                print(f"Local voice {len(result['items'])}/{total} {profile} {model}", flush=True)
            embeddings[f"{profile}_{model}"] = np.stack(vectors)
            summary = summarize_embeddings(items, embeddings[f"{profile}_{model}"])
            groups = summary["cross_cue"]["groups"]
            if len(groups) != 10 or any(row["items"] != 2 for row in groups):
                raise ValueError("Expected ten same-reference, same-seed two-cue groups")
            if profile == "raw":
                old = {(row["lang"], row["ref_audio"], row["unit"]): row for row in historical[model]}
                errors = [abs(row["mean_cosine"] - old[row["lang"], row["ref_audio"], row["unit"]]["mean_cosine"])
                          for row in groups]
                if max(errors) > plan["raw_tolerance"]:
                    raise ValueError(f"Raw speaker scores failed reproduction: {model}, {max(errors)}")
                summary["max_raw_reproduction_error"] = max(errors)
            result["summaries"][profile][model] = summary
            write_json(output / "results.json", result)
    for path, expected in {**inputs, **model_hashes, **plan["source_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    np.savez(output / "embeddings.npz", **embeddings)
    if windows:
        saved = {}
        for who, model in (("a", "base"), ("b", "r8")):
            report = deepcopy(reports[who])
            items = sorted(deepcopy(samples[model]), key=lambda row: (row["lang"], row["seed"], row["case_id"]))
            report.update(items=items, label=f"holdout_local_voice_{model}",
                          source_report=str(work / f"{model}_report.json"),
                          source_report_sha256=selection["source_reports_sha256"][model],
                          diagnostic_note=plan["note"])
            for field in ("cohorts", "human_review"):
                report.pop(field, None)
            report.update(evaluation._report_metrics(items))
            saved[model] = evaluation._save_report(report)["report_path"]
        evaluation.review_session(saved["base"], saved["r8"])
        write_json(output / "reports.json", saved)
        result["review_reports"] = saved
    result.update(complete=True, finished=time.time())
    write_json(output / "results.json", result)


def import_offline_feedback(work, feedback_path):
    import shutil

    directory = work / "dual_cue"
    output = directory / "feedback_review"
    if output.exists():
        raise FileExistsError(output)
    key_path = directory / "offline_key.json"
    key = json.loads(key_path.read_text())
    feedback = feedback_path.read_text(encoding="utf-8-sig")
    lines = [line.strip() for line in feedback.splitlines() if line.strip()]
    if not lines or lines[0] != f"试听包 {key['package_id']}":
        raise ValueError("Feedback package does not match the fixed mapping")
    records = {row["code"]: row for row in key["items"]}
    if len(records) != len(key["items"]):
        raise ValueError("Duplicate offline codes")
    statuses = {"无明显异常": "clear", "有明确异常": "abnormal",
                "不确定": None, "未评": None, "未指定判断": None}
    submitted = {}
    for line in lines[1:]:
        if line == "其余条目未评；语言质量未验证。":
            continue
        code, delimiter, description = line.partition("：")
        verdict, _, notes = description.partition("；")
        if not delimiter or code not in records or verdict not in statuses or code in submitted:
            raise ValueError(f"Invalid or duplicate feedback line: {line}")
        submitted[code] = {"submitted_verdict": verdict,
                           "human_review": {"acoustic_status": statuses[verdict],
                                            "acoustic_types": [], "notes": notes}}
    if not submitted:
        raise ValueError("No submitted feedback")
    if sha256(directory / "results.json") != key["source_results_sha256"]:
        raise ValueError("Changed source results")
    saved = json.loads((directory / "reports.json").read_text())
    paths = [directory / Path(saved[model]).name for model in ("base", "r8")]
    for model, path in zip(("base", "r8"), paths):
        if sha256(path) != key["reports_sha256"][saved[model]]:
            raise ValueError(f"Changed frozen report: {path.name}")
    reports, indices = evaluation._review_reports(*(str(path) for path in paths))
    if any(not report.get("review_only") for report in reports.values()):
        raise ValueError("Expected acoustic-only reports")
    session = evaluation.review_session(*(str(path) for path in paths), seed=key["shuffle_seed"])
    ratings, mapped, checked = {}, [], set()
    for number, pair in enumerate(session, 1):
        entry = {}
        for slot, side in (("1", "A"), ("2", "B")):
            code = f"D{number:02d}-{side}"
            record = records.get(code, {})
            who = pair[f"who_{slot}"]
            item = indices[who][pair["case_id"], pair["seed"]]
            if (record.get("model") != {"a": "base", "b": "r8"}[who]
                    or any(record.get(field) != item[field]
                           for field in ("case_id", "seed", "wav", "components"))):
                raise ValueError(f"Fixed offline mapping mismatch: {code}")
            audio_root = (directory / "offline").resolve()
            local_audio = (audio_root / record["local_audio"]).resolve()
            if (not local_audio.is_relative_to(audio_root)
                    or sha256(local_audio) != record["audio_sha256"]):
                raise ValueError(f"Changed offline audio: {code}")
            checked.add(code)
            if code in submitted:
                entry[f"s{slot}"] = submitted[code]["human_review"]
                mapped.append({**record, "lang": pair["lang"], **submitted[code]})
        ratings[evaluation.review_key(pair)] = entry
    if checked != records.keys():
        raise ValueError("Offline code coverage mismatch")
    provenance = {"package_id": key["package_id"], "mapping_sha256": sha256(key_path),
                  "feedback_sha256": sha256(feedback_path), "submitted": len(submitted),
                  "model_arms": {"a": "base", "b": "r8"},
                  "note": "Acoustic-only two-cue group feedback. Notes are verbatim, "
                          "not inferred noise/cutoff/native ratings or per-cue labels."}
    output.mkdir()
    destinations = []
    for who, model in (("a", "base"), ("b", "r8")):
        destination = output / f"reviewed_{model}.json"
        reports[who].update(report_path=str(destination.resolve()),
                            offline_feedback=provenance)
        write_json(destination, reports[who])
        destinations.append(str(destination.resolve()))
    summary = evaluation.save_reviews(*destinations, session, ratings)
    shutil.copyfile(feedback_path, output / "submitted.txt")
    write_json(output / "mapped_feedback.json", {**provenance, "items": mapped})
    write_json(output / "summary.json", {**provenance, **summary})
    print(f"Imported {len(submitted)} acoustic judgments into report copies: {output}")
    return summary


def offline_review(work):
    import secrets
    import wave
    from html import escape
    from urllib.parse import quote
    from uuid import uuid4

    directory = work / "dual_cue"
    output = directory / "offline"
    index_path = output / "index.html"
    key_path = directory / "offline_key.json"
    if index_path.exists() or key_path.exists():
        raise FileExistsError("Offline review already exists; keep its codes and feedback unchanged")
    result = json.loads((directory / "results.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    if (not result["complete"] or not verification["complete"]
            or result["plan_sha256"] != sha256(directory / "plan.json")):
        raise ValueError("Incomplete or changed dual-cue inputs")
    saved = json.loads((directory / "reports.json").read_text())
    report_paths = [directory / Path(saved[model]).name for model in ("base", "r8")]
    for model, path in zip(("base", "r8"), report_paths):
        if sha256(path) != verification["report_sha256"][saved[model]]:
            raise ValueError(f"Changed review report: {path.name}")
    reports, indices = evaluation._review_reports(*(str(path) for path in report_paths))
    if any(not report.get("review_only") for report in reports.values()):
        raise ValueError("Expected acoustic-only dual-cue reports")
    for field in ("base", "cfg_value", "inference_timesteps", "retry_badcase"):
        if reports["a"].get(field) != reports["b"].get(field):
            raise ValueError(f"Different generation settings: {field}")
    audio_records = {row["wav"]: row for row in result["items"]}
    expected_count = sum(len(report["items"]) for report in reports.values())
    if len(audio_records) != len(result["items"]) or len(audio_records) != expected_count:
        raise ValueError("Missing or duplicate audio records")
    for who, report in reports.items():
        for item in report["items"]:
            record = audio_records.get(item["wav"], {})
            local_audio = output / "audio" / Path(item["wav"]).name
            if (not item.get("review_only") or record.get("components") != item["components"]
                    or record.get("model") != {"a": "base", "b": "r8"}[who]
                    or not record.get("pcm_samples_verified")
                    or sha256(local_audio) != record.get("audio_sha256")):
                raise ValueError(f"Changed or mismatched audio: {local_audio.name}")
            with wave.open(str(local_audio)) as audio:
                if ((audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
                        != (48000, 1, 2) or len(item["components"]) != 2
                        or audio.getnframes() != item["components"][-1]["end_sample"]):
                    raise ValueError(f"Invalid dual-cue WAV: {local_audio.name}")
    package_id = f"dual-{uuid4().hex[:12]}"
    shuffle_seed = secrets.randbits(32)
    session = evaluation.review_session(*(str(path) for path in report_paths), seed=shuffle_seed)
    key = {"package_id": package_id, "shuffle_seed": shuffle_seed,
           "source_results_sha256": sha256(directory / "results.json"),
           "reports_sha256": verification["report_sha256"], "items": []}
    cards = []
    for number, pair in enumerate(session, 1):
        group_code = f"D{number:02d}"
        language = escape({"th": "泰语", "vi": "越南语"}.get(pair["lang"], pair["lang"]))
        players = []
        for slot, side in (("1", "A"), ("2", "B")):
            item = indices[pair[f"who_{slot}"]][pair["case_id"], pair["seed"]]
            record = audio_records[item["wav"]]
            code = f"{group_code}-{side}"
            filename = Path(item["wav"]).name
            second_start = item["components"][1]["start_sample"] / item["sample_rate"]
            key["items"].append({"code": code, **record, "local_audio": f"audio/{filename}"})
            players.append(f"""
<div class="take" data-code="{code}">
  <h3>{code}</h3><p class="timing">第二句从 {second_start:.2f} 秒开始</p>
  <audio controls preload="metadata" aria-label="{code} 双句音频" src="audio/{escape(quote(filename))}"></audio>
  <label>声学判断
    <select aria-label="{code} 声学判断">
      <option value="">未评</option><option>无明显异常</option><option>有明确异常</option><option>不确定</option>
    </select>
  </label>
  <label>异常时间点与说明
    <input aria-label="{code} 备注" placeholder="例如 6.2秒爆音；第二句明显娃娃音">
  </label>
</div>""")
        cards.append(f'<section><h2>{group_code} <span>{language}</span></h2>'
                     f'<div class="pair">{"".join(players)}</div></section>')
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>双句声学盲听 · 离线版</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f7;color:#17202a;font:16px/1.6 system-ui,sans-serif}
main{max-width:1000px;margin:auto;padding:28px 20px 60px}h1{font-size:30px;line-height:1.2}
h2{font-size:20px;margin:0 0 16px}h2 span{font-size:14px;font-weight:400;color:#52606d;margin-left:12px}
h3{margin:0;font-size:18px}header p{margin:10px 0}.badge{color:#205d49;font-weight:700}
section,.feedback{background:#fff;border:1px solid #dce1e6;border-radius:12px;padding:22px;margin-top:20px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:28px}.take{min-width:0}.timing{font-size:13px;color:#52606d;margin:4px 0 10px}
audio{width:100%;display:block;margin-bottom:16px}label{display:block;font-size:14px;margin:12px 0}
select,input,textarea,button{font:inherit}select,input,textarea{width:100%;border:1px solid #a8b2bd;border-radius:6px;padding:9px;background:white;color:#17202a}
input{margin-top:4px}button{border:0;border-radius:6px;background:#205d49;color:white;padding:10px 18px;cursor:pointer}
:focus-visible{outline:3px solid #326ec7;outline-offset:3px}textarea{min-height:160px;resize:vertical;margin:12px 0}
.hint{font-size:14px;color:#52606d}footer{font-size:12px;color:#52606d;margin-top:24px}
@media(max-width:650px){.pair{grid-template-columns:1fr;gap:22px}main{padding:18px 12px}section{padding:18px}}
</style></head><body><main>
<header><div class="badge">本地 WAV · 无需联网 · 不改变音频</div><h1>双句声学盲听</h1>
<p>每个代号内有两句，使用同一模型、参考音与 seed。比较同组 A / B；不同组的 A 不保证来自同一模型。</p>
<p>只标爆音、金属感、异常娃娃音、明显音色跳变等；听不懂语言不影响。没有把握选“不确定”，没听过保持“未评”。</p>
<p class="hint">句间另加了 0.5 秒静音；疑似拼接点问题需回原单句确认。保持 1 倍速，不需要填发音或自然度评分。</p>
<p class="hint">可以直接回复“D03-B，8.2秒金属感”。也可在页面填写后导出反馈；关闭页面前请导出，填写内容不会自动保存。</p>
</header>
{{cards}}
<div class="feedback"><h2>导出反馈</h2>
<button id="export" type="button">生成并下载已填反馈</button>
<p id="status" role="status" aria-live="polite" class="hint"></p>
<label for="feedback-text">反馈文本（也可选中复制给我）</label>
<textarea id="feedback-text" readonly placeholder="只导出主动填写的条目；未评不算无异常。"></textarea>
</div><footer>试听包 {{package}} · 模型映射单独保存，不在本页展示。请保持 index.html 与 audio 文件夹相邻。</footer>
</main><script>
const players = Array.from(document.querySelectorAll('audio'));
players.forEach(player => {
  player.addEventListener('play', () => players.forEach(other => { if (other !== player) other.pause(); }));
  player.addEventListener('error', () => {
    document.getElementById('status').textContent = '本地音频加载失败，请检查 audio 文件夹是否与 index.html 放在一起。';
  });
});
document.getElementById('export').addEventListener('click', () => {
  const rows = Array.from(document.querySelectorAll('[data-code]')).flatMap(take => {
    const verdict = take.querySelector('select').value;
    const notes = take.querySelector('input').value.trim();
    return verdict || notes ? [`${take.dataset.code}：${verdict || '未指定判断'}${notes ? '；' + notes : ''}`] : [];
  });
  const status = document.getElementById('status');
  if (!rows.length) { status.textContent = '尚未填写反馈；未评条目不会当作无异常。'; return; }
  const text = '试听包 {{package}}\\n' + rows.join('\\n') + '\\n其余条目未评；语言质量未验证。\\n';
  document.getElementById('feedback-text').value = text;
  const downloadUrl = URL.createObjectURL(new Blob([text], {type:'text/plain;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = '{{package}}-feedback.txt';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
  status.textContent = `已生成 ${rows.length} 条反馈；可发送下载的文本文件或复制文本。`;
});
</script></body></html>
"""
    page = page.replace("{{cards}}", "\n".join(cards)).replace("{{package}}", package_id)
    write_json(key_path, key)
    index_path.write_text(page, encoding="utf-8")
    print(f"Offline review: {len(session)} pairs / {expected_count} verified WAVs; {index_path}")
    return index_path


def dual_cue_review(work):
    import shutil
    import time
    from uuid import uuid4

    import numpy as np
    import soundfile as sf

    output = work / "dual_cue"
    if output.exists():
        raise FileExistsError(output)
    previous_dir = work / "voice_windows"
    previous = json.loads((previous_dir / "plan.json").read_text())
    previous_result = json.loads((previous_dir / "results.json").read_text())
    if (not previous_result["complete"]
            or previous_result["plan_sha256"] != sha256(previous_dir / "plan.json")):
        raise ValueError("Completed voice-window plan does not match")
    inputs = {**previous["inputs_sha256"],
              **{str(previous_dir / name): sha256(previous_dir / name)
                 for name in ("plan.json", "results.json")}}
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    reports, indices = evaluation._review_reports(str(work / "base_report.json"),
                                                  str(work / "r8_report.json"))
    for field in ("base", "cfg_value", "inference_timesteps", "retry_badcase"):
        if reports["a"].get(field) != reports["b"].get(field):
            raise ValueError(f"Different generation settings: {field}")
    expected_keys = {(case_id, seed) for case_id in VOICE_CASES for seed in VOICE_SEEDS}
    groups = {}
    for who, model in (("a", "base"), ("b", "r8")):
        samples = previous["samples"][model]
        if (len(samples) != len(expected_keys)
                or {(row["case_id"], row["seed"]) for row in samples} != expected_keys
                or any(row != indices[who][row["case_id"], row["seed"]] for row in samples)):
            raise ValueError(f"Frozen voice samples do not match: {model}")
        groups[model] = []
        for first_case, second_case in (VOICE_CASES[:2], VOICE_CASES[2:]):
            for seed in VOICE_SEEDS:
                members = [indices[who][case_id, seed] for case_id in (first_case, second_case)]
                if not members[0].get("ref_audio") or any(
                        members[0].get(field) != members[1].get(field)
                        for field in ("lang", "ref_audio", "ref_lang", "control", "speaker", "seed")):
                    raise ValueError("Two cues must share language/reference/control/speaker/seed")
                for member in members:
                    header = sf.info(member["wav"])
                    if ((header.format, header.subtype, header.samplerate, header.channels)
                            != ("WAV", "PCM_16", 48000, 1) or header.frames <= 0):
                        raise ValueError(f"Expected nonempty mono 48kHz PCM_16 WAV: {member['wav']}")
                    if inputs.get(member["wav"]) != sha256(member["wav"]):
                        raise ValueError(f"Unfrozen source audio: {member['wav']}")
                groups[model].append(members)
    audio_dir = evaluation.EVAL_DIR / f"dual_cue_audio_{uuid4().hex}"
    plan = {
        "groups": groups, "gap_samples": 24000, "sample_rate": 48000,
        "audio_dir": str(audio_dir),
        "inputs_sha256": inputs,
        "source_sha256": {str(path): sha256(path) for path in
                          (Path(__file__), Path(evaluation.__file__))},
        "versions": {"numpy": np.__version__, "soundfile": sf.__version__},
        "note": "Acoustic-only two-cue composites, not new TTS or quantitative evaluation. "
                "Each file keeps one model/reference/seed; cue order matches across A/B. "
                "Original PCM samples and edge pauses preserved, plus 0.5s digital silence. "
                "No resampling, gain, trimming, fades, ASR or embeddings. Verify suspected "
                "splice-boundary artifacts against original WAVs. Random A/B mapping is fixed "
                "inside each composite, NOT across review items. Outcome-selected TH/VI "
                "diagnostics, not an online badcase rate or native-quality acceptance. "
                "New group ratings start unfilled; original per-cue ratings remain untouched.",
    }
    output.mkdir()
    write_json(output / "plan.json", plan)
    shutil.copyfile(__file__, output / "source.py")
    audio_dir.mkdir(parents=True, exist_ok=False)
    result = {"complete": False, "plan_sha256": sha256(output / "plan.json"), "items": []}
    write_json(output / "results.json", result)
    exports = {}
    for who, model in (("a", "base"), ("b", "r8")):
        items = []
        for members in groups[model]:
            waves = [sf.read(member["wav"], dtype="int16")[0] for member in members]
            joined = np.concatenate([waves[0], np.zeros(plan["gap_samples"], dtype=np.int16), waves[1]])
            destination = audio_dir / f"{uuid4().hex}.wav"
            sf.write(destination, joined, plan["sample_rate"], subtype="PCM_16")
            restored, sample_rate = sf.read(destination, dtype="int16")
            if sample_rate != plan["sample_rate"] or not np.array_equal(restored, joined):
                raise ValueError(f"Composite PCM samples changed: {destination}")
            starts = [0, len(waves[0]) + plan["gap_samples"]]
            components = [{
                "case_id": member["case_id"], "seed": member["seed"], "wav": member["wav"],
                "audio_sha256": inputs[member["wav"]],
                "start_sample": start, "end_sample": start + len(wave),
            } for member, wave, start in zip(members, waves, starts)]
            item = {
                "case_id": "dual_" + "__".join(member["case_id"] for member in members),
                **{key: members[0].get(key) for key in
                   ("lang", "seed", "ref_audio", "ref_lang", "control", "speaker")},
                "text": "双句声学复核：每段内同模型、同ref、同seed；中间另加0.5秒静音。"
                        "只标明确声学异常；语言维度不懂留空。甲乙只在本条内固定，不跨条。\n\n"
                        f"第1句：{members[0]['text']}\n\n第2句：{members[1]['text']}",
                "numeric": any(evaluation._is_numeric(member) for member in members),
                "wav": str(destination), "review_only": True, "components": components,
                "sample_rate": sample_rate, "gap_samples": plan["gap_samples"],
                "playback_duration_sec": len(joined) / sample_rate,
                "human_review": {"acoustic_status": None, "acoustic_types": [], "notes": ""},
            }
            items.append(item)
            result["items"].append({"model": model, "case_id": item["case_id"], "seed": item["seed"],
                                    "wav": str(destination), "audio_sha256": sha256(destination),
                                    "components": components, "pcm_samples_verified": True})
            write_json(output / "results.json", result)
            print(f"Dual cue {len(result['items'])}/20 {model}", flush=True)
        exports[model] = {
            **{key: reports[who][key] for key in
               ("target", "base", "cfg_value", "inference_timesteps", "retry_badcase", "lora_strength")
               if key in reports[who]},
            "label": f"holdout_dual_cue_{model}", "review_only": True,
            "review_unit": "two_cues_same_ref_seed", "items": items, "note": plan["note"],
            "source_report": str(work / f"{model}_report.json"),
            "source_report_sha256": inputs[str(work / f"{model}_report.json")],
        }
    for path, expected in {**inputs, **plan["source_sha256"]}.items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    saved = {model: evaluation._save_report(report)["report_path"] for model, report in exports.items()}
    if len(evaluation.review_session(saved["base"], saved["r8"])) != 10:
        raise ValueError("Expected ten paired two-cue review items")
    write_json(output / "reports.json", saved)
    result.update(complete=True, finished=time.time(), review_reports=saved)
    write_json(output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "holdout_eval_20260917")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recheck-asr", action="store_true")
    mode.add_argument("--localize-voice", action="store_true")
    mode.add_argument("--voice-windows", action="store_true")
    mode.add_argument("--dual-cue-review", action="store_true")
    mode.add_argument("--offline-review", action="store_true")
    mode.add_argument("--import-offline-feedback", type=Path)
    args = parser.parse_args()
    if args.import_offline_feedback:
        import_offline_feedback(args.work, args.import_offline_feedback)
    elif args.offline_review:
        offline_review(args.work)
    elif args.dual_cue_review:
        dual_cue_review(args.work)
    elif args.voice_windows:
        localize_voice(args.work, windows=True)
    else:
        (localize_voice if args.localize_voice else recheck_asr if args.recheck_asr else collect)(args.work)
