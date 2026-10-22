"""Existing-WAV acoustic review and CPU-only tail diagnostics for 2026-10-22."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from html import escape
import inspect
import json
from pathlib import Path
import random
import secrets
import shutil
from uuid import uuid4

import numpy as np

from voxft import eval as evaluation, infer
from voxft.data.pipeline import _read_manifest, _write_jsonl, load_wav_mono
from voxft.qc.audio import edge_silence

from control_ab_run import LANGS, SEEDS, verify_report
from quality_probe import write_json
from ref_ab import sha256


FROZEN = {
    "plan.json": "b4d32700c0507f274b68cb7b750e44bd6ff5fd8b7cc572f9f824dc2fbe76b118",
    "comparison.json": "b6d74a4ef12d527fcaba1c28de3d3e3da3ed2881795812ecad799cc99e9e7ca1",
}
STATUSES = {"无明显异常": "clear", "有明确异常": "abnormal",
            "不确定": None, "未评": None, "未指定判断": None}


def checked(path, digest):
    if not digest or sha256(path) != digest:
        raise ValueError(f"Frozen file changed or missing hash: {path}")
    return Path(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sources(work):
    for filename, digest in FROZEN.items():
        checked(work / filename, digest)
    plan, comparison = read_json(work / "plan.json"), read_json(work / "comparison.json")
    for helper in (verify_report, evaluation._report_metrics, infer._gen_kwargs,
                   load_wav_mono, edge_silence):
        path = inspect.getfile(helper)
        checked(path, plan["inputs_sha256"].get(path))
    cases_path = checked(work / "cases.jsonl", plan["inputs_sha256"].get(str(work / "cases.jsonl")))
    cases = _read_manifest(cases_path)
    reports = {}
    for name in ("base", "r23", "off", "on"):
        path = work / f"{name}_report.json"
        report = read_json(checked(path, comparison["inputs_sha256"].get(str(path))))
        verify_report(report, plan, cases, name)
        reports[name] = report
    return plan, cases, reports, comparison["inputs_sha256"]


def random_selection(cases):
    rng, selected, used = random.Random(20261022), [], set()
    for lang in sorted(LANGS):
        for ref_lang, condition in (("zh", "plain"), ("zh", "control"),
                                    ("zh", "plain"), ("en", "control")):
            pool = sorted((row for row in cases if row["cohort"] == "independent_text"
                           and row["lang"] == lang and row["ref_lang"] == ref_lang
                           and row["condition"] == condition and row["sentence_id"] not in used),
                          key=lambda row: row["case_id"])
            if not pool:
                raise ValueError(f"Insufficient metadata-only random stratum: {lang}/{ref_lang}")
            row = rng.choice(pool)
            used.add(row["sentence_id"])
            selected.append({"case_id": row["case_id"], "seed": rng.choice(SEEDS),
                             "sentence_id": row["sentence_id"], "stratum": "random"})
    return selected


def select_pairs(cases, reports):
    selected = random_selection(cases)
    used = {row["sentence_id"] for row in selected}
    base = {(row["case_id"], row["seed"]): row for row in reports["base"]["items"]}
    for lang in sorted(LANGS):
        for stratum, cohort in (("tail_risk", "independent_text"),
                                ("historical_risk", "historical_regression")):
            pool = sorted((row for row in reports["on"]["items"]
                           if row["lang"] == lang and row["cohort"] == cohort
                           and row["sentence_id"] not in used),
                          key=lambda row: (row["case_id"], row["seed"]))

            def priority(row):
                other = base[row["case_id"], row["seed"]]
                tail_growth = row["tail_silence_sec"] - other["tail_silence_sec"]
                if stratum == "tail_risk":
                    return (tail_growth,)
                return (int(row["suspected_truncation"]) - int(other["suspected_truncation"]),
                        other["speech_ratio"] - row["speech_ratio"], tail_growth)

            if not pool:
                raise ValueError(f"Empty risk stratum: {lang}/{stratum}")
            row = max(pool, key=priority)
            used.add(row["sentence_id"])
            selected.append({"case_id": row["case_id"], "seed": row["seed"],
                             "sentence_id": row["sentence_id"], "stratum": stratum,
                             "selection_priority": priority(row)})
    return selected


def wav_hashes(report):
    if report.get("wav_sha256"):
        hashes = report["wav_sha256"]
    else:
        hashes, shard_rows = {}, {}
        for path in report["merged_from"]:
            shard = read_json(path)
            for row in shard["items"]:
                key = row["case_id"], row["seed"]
                if key in shard_rows:
                    raise ValueError("Duplicate shard result")
                shard_rows[key] = row
                digest = shard["wav_sha256"].get(row["wav"])
                if not digest:
                    raise ValueError("Missing original shard WAV hash")
                hashes[row["wav"]] = digest
        merged_rows = {(row["case_id"], row["seed"]): row for row in report["items"]}
        if shard_rows != merged_rows:
            raise ValueError("Shard items differ from frozen merged report")
    if hashes.keys() != {row["wav"] for row in report["items"]}:
        raise ValueError("Incomplete WAV hash coverage")
    return hashes


def tail_summary(rows):
    values = sorted(row["tail_silence_sec"] for row in rows
                    if row.get("tail_silence_sec") is not None)
    return {"rows": len(rows), "measured": len(values), "unknown": len(rows) - len(values),
            "mean_sec": float(np.mean(values)) if values else None,
            "p90_sec": values[min(len(values) - 1, int(len(values) * .9))] if values else None,
            "over_0_5s": sum(value > .5 for value in values),
            "max_sec": max(values) if values else None}


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    return [{**dict(zip(fields, key)), **tail_summary(items)} for key, items in sorted(groups.items())]


def diagnose_reports(reports):
    result = {}
    baseline = {(row["case_id"], row["seed"]): row for row in reports["base"]["items"]}
    for model, report in reports.items():
        changes = []
        for row in report["items"]:
            other = baseline[row["case_id"], row["seed"]]
            changes.append({key: row[key] for key in ("case_id", "seed", "lang", "cohort",
                                                      "sentence_id", "condition", "ref_lang")})
            changes[-1].update(tail_growth_sec=row["tail_silence_sec"] - other["tail_silence_sec"],
                               tail_silence_sec=row["tail_silence_sec"],
                               base_tail_sec=other["tail_silence_sec"],
                               cer_delta=row["cer"] - other["cer"],
                               suspected_truncation=row["suspected_truncation"],
                               speech_ratio_delta=row["speech_ratio"] - other["speech_ratio"])
        result[model] = {
            "by_condition": grouped(report["items"], ("cohort", "lang", "condition")),
            "by_reference": grouped(report["items"], ("cohort", "lang", "ref_lang")),
            "by_seed": grouped(report["items"], ("cohort", "lang", "seed")),
            "top_tail_growth": sorted(changes, key=lambda row: row["tail_growth_sec"], reverse=True)[:20],
        }
    return result


def inspect_audio(path):
    audio, sample_rate = load_wav_mono(path)
    edges = {key: float(value) if value is not None else None
             for key, value in edge_silence(audio, sample_rate).items()}
    tail_samples = int((edges["tail_silence_sec"] or 0) * sample_rate)
    tail = audio[-tail_samples:] if tail_samples else np.array([])
    return {**edges, "sample_rate": sample_rate, "audio_sec": len(audio) / sample_rate,
            "tail_rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(tail ** 2))), 1e-7)))
            if tail.size else None}


def diagnose_training(work, plan):
    path = work / "on.jsonl"
    rows = _read_manifest(checked(path, plan["inputs_sha256"].get(str(path))))
    paths = sorted({row["audio"] for row in rows})

    def inspect(path):
        checked(path, plan["inputs_sha256"].get(path))
        return inspect_audio(path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        metrics = dict(zip(paths, executor.map(inspect, paths)))
    weighted = [{**row, **metrics[row["audio"]]} for row in rows]
    unique = list({row["audio"]: row for row in weighted}.values())
    return {"manifest_sha256": sha256(path), "unique_targets": len(unique),
            "by_lang_row_weighted": grouped(weighted, ("lang",)),
            "by_source_unique": grouped(unique, ("lang", "source_id")),
            "items": [{key: row.get(key) for key in
                       ("audio", "lang", "source_id", "head_silence_sec", "tail_silence_sec", "audio_sec")}
                      for row in unique]}


def prepare(work, output):
    if output.exists():
        raise FileExistsError(output)
    plan, cases, reports, report_hashes = load_sources(work)
    selected = select_pairs(cases, reports)
    indices = {name: {(row["case_id"], row["seed"]): row for row in reports[name]["items"]}
               for name in ("base", "on")}
    hashes = {name: wav_hashes(reports[name]) for name in indices}
    for selection in selected:
        for name in indices:
            row = indices[name][selection["case_id"], selection["seed"]]
            checked(row["wav"], hashes[name].get(row["wav"]))
            checked(row["ref_audio"], plan["inputs_sha256"].get(row["ref_audio"]))
    output.mkdir(parents=True)
    for folder in ("public/audio", "references", "reports"):
        (output / folder).mkdir(parents=True)
    write_json(output / "selection.json", {
        "selection_seed": 20261022, "items": selected,
        "note": "12 metadata-only random pairs, 6 outcome-enriched diagnostic pairs. "
                "Post-experiment development review, not a new independent validation set. "
                "Selection does not assert a confirmed acoustic or content defect."})
    saved = {}
    for name in indices:
        subset = deepcopy(reports[name])
        subset["items"] = [deepcopy(indices[name][row["case_id"], row["seed"]]) for row in selected]
        for row in subset["items"]:
            row.pop("human_review", None)
        subset.update(evaluation._report_metrics(subset["items"]), review_only=True)
        saved[name] = output / "reports" / f"{name}.json"
        subset["report_path"] = str(saved[name])
        write_json(saved[name], subset)
    shuffle_seed = secrets.randbits(32)
    session = evaluation.review_session(str(saved["base"]), str(saved["on"]), seed=shuffle_seed)
    package_id = f"acoustic-{uuid4().hex[:12]}"
    mapping, cards, observations = [], [], []
    counters = Counter()
    for pair, selection in zip(session, selected):
        stratum = selection["stratum"]
        prefix = "R" if stratum == "random" else "D"
        counters[prefix] += 1
        group_code = f"{prefix}{counters[prefix]:02}"
        players = []
        for slot, side in (("1", "A"), ("2", "B")):
            model = {"a": "base", "b": "on"}[pair[f"who_{slot}"]]
            row = indices[model][pair["case_id"], pair["seed"]]
            code = f"{group_code}-{side}"
            audio_path = output / "public" / "audio" / f"{code}.wav"
            shutil.copyfile(row["wav"], audio_path)
            checked(audio_path, hashes[model][row["wav"]])
            ref_digest = plan["inputs_sha256"][row["ref_audio"]]
            ref_path = output / "references" / f"{ref_digest}.wav"
            if not ref_path.exists():
                shutil.copyfile(row["ref_audio"], ref_path)
            checked(ref_path, ref_digest)
            metrics = inspect_audio(audio_path)
            if any(metrics[field] != row[field] for field in ("head_silence_sec", "tail_silence_sec")):
                raise ValueError(f"Frozen edge-silence metric mismatch: {code}")
            observations.append({"code": code, "model": model, **selection, **metrics})
            parameters = infer._gen_kwargs(row["text"], row["ref_audio"], None, 1.8, 20,
                                           row["seed"], row["control"])
            parameters["retry_badcase"] = False
            mapping.append({
                "code": code, "model": model, "who": pair[f"who_{slot}"], "slot": slot,
                **selection, "lang": row["lang"], "source_wav": row["wav"],
                "audio": str(audio_path.relative_to(output)), "audio_sha256": hashes[model][row["wav"]],
                "reference": {"status": "available", "clip": str(ref_path.relative_to(output)),
                              "original_path": row["ref_audio"], "audio_sha256": ref_digest},
                "prompt": {"status": "not_supplied"}, "reference_lang": row["ref_lang"],
                "effective_parameters": parameters, "original_qc": row,
                "reference_identity_verified": False, "gpu_preprocessed_audio_persisted": False,
                "training_eligible": False, "split": "development_only"})
            players.append(f'<div class="take" data-code="{code}"><h3>{code}</h3>'
                           f'<audio controls preload="none" aria-label="{code} 完整原音频" '
                           f'src="audio/{code}.wav"></audio>'
                           f'<label>声学判断<select aria-label="{code} 声学判断"><option value="">未评</option>'
                           '<option>无明显异常</option><option>有明确异常</option><option>不确定</option>'
                           f'</select></label><label>时间点与说明<input aria-label="{code} 备注" '
                           'placeholder="如 3.2秒沙哑/爆音；末尾明显空白"></label></div>')
        title = "随机对照" if stratum == "random" else "诊断样本"
        language = {"id": "印尼语", "th": "泰语", "tl": "菲律宾语 / Tagalog"}[pair["lang"]]
        cards.append(f'<section><h2>{group_code} · {escape(language)} · {title}</h2>'
                     f'<div class="pair">{"".join(players)}</div></section>')
    key_path = output / "mapping.json"
    write_json(key_path, {"package_id": package_id, "shuffle_seed": shuffle_seed, "items": mapping,
                         "reports_sha256": {str(path.relative_to(output)): sha256(path)
                                            for path in saved.values()},
                         "source_reports_sha256": report_hashes})
    (output / "conditioning.jsonl").write_text(
        "".join(json.dumps({"pack_id": package_id, "audition_code": row["code"], **row},
                           ensure_ascii=False) + "\n" for row in mapping), encoding="utf-8")
    print("Selected WAV/ref hashes verified; scanning existing training targets on CPU.", flush=True)
    sentences = {row["sentence_id"] for row in selected}
    diagnostics = {"models": diagnose_reports(reports), "selected_wav_recheck": observations,
                   "selected_sentence_all_seeds": {
                       name: [row for row in report["items"] if row["sentence_id"] in sentences]
                       for name, report in reports.items()},
                   "training": diagnose_training(work, plan),
                   "note": "Energy boundaries are not verified speech/word boundaries. No trimming, "
                           "new TTS/ASR, rescoring replacement or causal/native quality conclusion."}
    write_json(output / "diagnostics.json", diagnostics)
    template = Path(__file__).with_suffix(".html")
    page = template.read_text(encoding="utf-8").replace("{{cards}}", "\n".join(cards))
    (output / "public" / "index.html").write_text(page.replace("{{package}}", package_id), encoding="utf-8")
    provenance = {str(work / filename): digest for filename, digest in FROZEN.items()}
    provenance.update(report_hashes)
    provenance.update({path: sha256(path) for name in indices if not reports[name].get("wav_sha256")
                       for path in reports[name]["merged_from"]})
    provenance.update({str(Path(__file__).resolve()): sha256(__file__), str(template): sha256(template)})
    artifacts = {str(path.relative_to(output)): sha256(path)
                 for path in output.rglob("*") if path.is_file()}
    write_json(output / "package.json", {
        "complete": True, "package_id": package_id, "source_sha256": provenance,
        "artifacts_sha256": artifacts, "pairs": len(selected), "audio_count": len(mapping),
        "total_audio_seconds": sum(row["audio_sec"] for row in observations),
        "training_eligible": False, "language_quality": "unverified",
        "production_changed": False, "review_received": False})
    print(f"Ready: {output / 'public/index.html'} ({len(selected)} pairs)", flush=True)


def import_feedback(output, feedback_path):
    destination = output / "feedback_review"
    if destination.exists():
        raise FileExistsError(destination)
    package = read_json(output / "package.json")
    if not package["complete"]:
        raise ValueError("Incomplete package")
    for filename, digest in package["artifacts_sha256"].items():
        path = (output / filename).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Artifact path escapes package")
        checked(path, digest)
    key = read_json(output / "mapping.json")
    lines = [line.strip() for line in feedback_path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if key["package_id"] != package["package_id"] or not lines or lines[0] != f"试听包 {key['package_id']}":
        raise ValueError("Feedback package mismatch")
    records = {row["code"]: row for row in key["items"]}
    submitted = {}
    for line in lines[1:]:
        if line == "其余条目未评；语言质量未验证。":
            continue
        code, delimiter, description = line.partition("：")
        verdict, _, notes = description.partition("；")
        if not delimiter or code not in records or code in submitted or verdict not in STATUSES:
            raise ValueError(f"Invalid/duplicate feedback: {line}")
        submitted[code] = {"submitted_verdict": verdict,
                           "human_review": {"acoustic_status": STATUSES[verdict], "notes": notes,
                                            "acoustic_types": []}}
    if not submitted:
        raise ValueError("No feedback supplied")
    saved = [output / "reports" / f"{model}.json" for model in ("base", "on")]
    session = evaluation.review_session(*(str(path) for path in saved), seed=key["shuffle_seed"])
    ratings, strata = {}, defaultdict(list)
    for pair in session:
        entry, pair_strata = {}, set()
        for slot in ("1", "2"):
            matches = [row for row in records.values() if row["case_id"] == pair["case_id"]
                       and row["seed"] == pair["seed"] and row["slot"] == slot]
            if len(matches) != 1 or matches[0]["who"] != pair[f"who_{slot}"]:
                raise ValueError("Changed blind mapping")
            record = matches[0]
            pair_strata.add(record["stratum"])
            if record["code"] in submitted:
                entry[f"s{slot}"] = submitted[record["code"]]["human_review"]
        if len(pair_strata) != 1:
            raise ValueError("Mismatched pair strata")
        strata[pair_strata.pop()].append(pair)
        ratings[evaluation.review_key(pair)] = entry
    destination.mkdir()
    copies = []
    for path in saved:
        copied = destination / path.name
        document = read_json(path)
        document.update(report_path=str(copied), offline_feedback={
            "package_id": package["package_id"], "feedback_sha256": sha256(feedback_path),
            "mapping_sha256": sha256(output / "mapping.json"), "language_quality": "unverified"})
        write_json(copied, document)
        copies.append(str(copied))
    summaries = {stratum: evaluation.save_reviews(*copies, pairs, ratings)
                 for stratum, pairs in strata.items()}
    write_json(destination / "summary.json", {"by_stratum": summaries,
               "note": "Unknown/unrated are not passes. Do not pool risk-selected and random samples."})
    write_json(destination / "mapped_feedback.json",
               [{**records[code], **feedback} for code, feedback in submitted.items()])
    shutil.copyfile(feedback_path, destination / "submitted.txt")
    return summaries


def level_metrics(path):
    audio, sample_rate = load_wav_mono(path)
    audio = np.asarray(audio, dtype=np.float64)
    frame = int(.04 * sample_rate)
    if len(audio) < frame:
        raise ValueError("Audio too short for level diagnostics")
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame)[::frame // 2]
    frame_db = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-7))
    return {"rms_dbfs": float(20 * np.log10(max(float(np.sqrt(np.mean(audio ** 2))), 1e-7))),
            "peak_dbfs": float(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-7))),
            "p90_frame_rms_dbfs": float(np.percentile(frame_db, 90))}


def archive_feedback(pack, output):
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to((pack / "public").resolve()):
        raise ValueError("Feedback archive must remain private")
    package = read_json(pack / "package.json")
    if not package["complete"]:
        raise ValueError("Incomplete package")
    for filename, digest in package["artifacts_sha256"].items():
        path = (pack / filename).resolve()
        if not path.is_relative_to(pack.resolve()):
            raise ValueError("Artifact path escapes package")
        checked(path, digest)
    key = read_json(pack / "mapping.json")
    records = {row["code"]: row for row in key["items"]}
    reviewed = read_json(pack / "feedback_review/mapped_feedback.json")
    feedback_path = pack / "feedback_review/submitted.txt"
    lines = [line.strip() for line in feedback_path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    expected = [f"试听包 {package['package_id']}"] + [
        f"{row['code']}：{row['submitted_verdict']}"
        + (f"；{row['human_review']['notes']}" if row["human_review"]["notes"] else "")
        for row in reviewed]
    if lines and lines[-1] == "其余条目未评；语言质量未验证。":
        lines.pop()
    if lines != expected or len({row["code"] for row in reviewed}) != len(reviewed):
        raise ValueError("Mapped feedback differs from verbatim submission")
    for row in reviewed:
        original = records.get(row["code"])
        if (original is None or any(row.get(field) != value for field, value in original.items())
                or row["human_review"]["acoustic_status"] != STATUSES[row["submitted_verdict"]]):
            raise ValueError("Mapped feedback differs from frozen package")
    source_files = [pack / filename for filename in ("package.json", "mapping.json", "conditioning.jsonl")]
    source_files += sorted((pack / "feedback_review").glob("*"))
    provenance = {"package_id": package["package_id"], "feedback_sha256": sha256(feedback_path),
                  "inputs_sha256": {str(path): sha256(path) for path in source_files if path.is_file()}}
    output.mkdir(parents=True)
    for directory in ("audio", "conditioning", "evidence"):
        (output / directory).mkdir()
    for path in source_files:
        if path.is_file():
            shutil.copyfile(path, output / "evidence" / path.name)
    annotations, conditioning, regressions, levels = [], [], [], []
    for row in reviewed:
        status = row["human_review"]["acoustic_status"]
        if status not in ("clear", "abnormal"):
            continue
        clip = f"audio/{row['audio_sha256']}.wav"
        reference = {**row["reference"],
                     "clip": f"conditioning/{row['reference']['audio_sha256']}.wav"}
        for source, destination, digest in (
            (pack / row["audio"], output / clip, row["audio_sha256"]),
            (pack / row["reference"]["clip"], output / reference["clip"], reference["audio_sha256"]),
        ):
            if not destination.exists():
                shutil.copyfile(source, destination)
            checked(destination, digest)
        levels.append({"code": row["code"], "model": row["model"], "lang": row["lang"],
                       "audio_sha256": row["audio_sha256"], **level_metrics(output / clip)})
        common = {**row, "pack_id": package["package_id"], "audition_code": row["code"],
                  "clip": clip, "reference": reference, "feedback_sha256": provenance["feedback_sha256"],
                  "source_kind": "synthetic_tts", "training_eligible": False,
                  "split": "development_only", "language_quality": "unverified",
                  "transcript_verified": False, "training_authorization": "unverified"}
        conditioning.append({**common, "stage": "offline_inference",
                             "replay_complete": False, "gpu_preprocessed_audio_persisted": False})
        if status == "clear":
            annotations.append({**common, "target_text": row["original_qc"]["text"],
                                "candidate_status": "acoustic_clear_content_unverified"})
        else:
            case = {field: row["original_qc"][field] for field in
                    ("lang", "text", "control", "ref_lang", "numeric", "condition")}
            regressions.append({
                **case, "case_id": f"{package['package_id']}_{row['code']}",
                "source_case_id": row["case_id"], "seed": row["seed"],
                "ref_audio": str((output / reference["clip"]).resolve()),
                "ref_audio_original_path": row["reference"]["original_path"],
                "ref_audio_sha256": reference["audio_sha256"],
                "cohort": "human_acoustic_development_regression",
                "feedback_package": package["package_id"], "feedback_sha256": provenance["feedback_sha256"],
                "audition_code": row["code"], "observed_model": row["model"],
                "observed_audio": str((output / clip).resolve()), "audio_sha256": row["audio_sha256"],
                "reported_acoustic_status": "abnormal", "reported_notes": row["human_review"]["notes"],
                "training_eligible": False, "language_quality": "unverified",
                "note": "用户声学盲听确认：" + row["human_review"]["notes"]
                        + " 保留原条件与原seed；这是结果富集的开发回归，不是独立验收或母语内容判断。",
            })
    for filename, rows in (("annotations.jsonl", annotations), ("conditioning.jsonl", conditioning),
                           ("regressions.jsonl", regressions)):
        _write_jsonl(rows, output / filename)
    write_json(output / "levels.json", {
        "items": levels, "note": "Read-only amplitude diagnostics; not LUFS, trusted speech bounds, "
                                "native assessment or replacement of frozen QC scores."})
    write_json(output / "provenance.json", {
        **provenance, "complete": True, "clear_candidates": len(annotations),
        "regressions": len(regressions), "training_rows": 0,
        "artifacts_sha256": {str(path.relative_to(output)): sha256(path)
                             for path in output.rglob("*") if path.is_file()}})
    return {"clear_candidates": len(annotations), "regressions": len(regressions), "training_rows": 0}


def serve(output):
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from starlette.staticfiles import StaticFiles

    package = read_json(output / "package.json")
    if not package["complete"]:
        raise ValueError("Incomplete package")
    public = (output / "public").resolve()
    for filename, digest in package["artifacts_sha256"].items():
        if filename.startswith("public/"):
            path = (output / filename).resolve()
            if not path.is_relative_to(public):
                raise ValueError("Public asset escapes package")
            checked(path, digest)
    app = Starlette(routes=[Mount("/", app=StaticFiles(directory=public, html=True))])
    uvicorn.run(app, host="127.0.0.1", port=16006)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--work", type=Path)
    action.add_argument("--feedback", type=Path)
    action.add_argument("--archive", type=Path)
    action.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve(args.output.resolve())
    elif args.archive:
        print(archive_feedback(args.output.resolve(), args.archive.resolve()))
    elif args.work:
        prepare(args.work.resolve(), args.output.resolve())
    else:
        import_feedback(args.output.resolve(), args.feedback.resolve())
