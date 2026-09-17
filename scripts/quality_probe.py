"""2026-09-17 r8 强度实验：prepare → worker 0/2、1/2 → collect。仅在 GPU 机器执行。"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from voxft import eval as evaluation, infer
from voxft.paths import CHECKPOINT_DIR

SOURCES = {
    "base": "base_8de890a3af74491b948c1ae0449967b0.json",
    "r8": "lora_omni5_r8_latest_f1315988633448b1bd151f4fd999323b.json",
    "r9": "lora_omni5_r9_latest_a2f9e99f614245f0842c37a076c381bf.json",
}
STRENGTHS = (0, 0.5, 0.75, 1)
SEEDS = [42, 43, 44, 45, 46]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(work):
    work.mkdir(parents=True, exist_ok=False)
    reports = {name: json.loads((evaluation.EVAL_DIR / filename).read_text())
               for name, filename in SOURCES.items()}
    for report in reports.values():
        assert report["cfg_value"] == 1.8 and report["inference_timesteps"] == 20
        assert report["retry_badcase"] is False
    base = {(item["case_id"], item["seed"]): item for item in reports["base"]["items"]}
    rows = {item["case_id"]: item for item in reports["r8"]["items"]}
    for item in reports["r8"]["items"]:
        other = base[item["case_id"], item["seed"]]
        assert all(item.get(key) == other.get(key) for key in
                   ("text", "lang", "ref_audio", "control", "ref_lang"))
    rng = random.Random(20260917)
    selected = {}
    for lang in ("th", "tl", "vi", "id", "ms", "zh", "en"):
        pool = sorted((row for row in rows.values() if row["lang"] == lang),
                      key=lambda row: row["case_id"])
        if lang in ("zh", "en"):
            for row in rng.sample(pool, 2):
                selected[row["case_id"]] = "random_replay"
            continue
        for ref_lang in ("zh", "en", "tl"):
            row = rng.choice([row for row in pool if row["ref_lang"] == ref_lang])
            selected[row["case_id"]] = "random_control"
        risk = max((item for item in reports["r8"]["items"]
                    if item["lang"] == lang and item["case_id"] not in selected),
                   key=lambda item: item["cer"] - base[item["case_id"], item["seed"]]["cer"])
        selected[risk["case_id"]] = "cer_regression_diagnostic_not_confirmed_badcase"
    keys = ("case_id", "text", "lang", "ref_audio", "ref_lang", "control",
            "speaker", "numeric", "note")
    cases = [{**{key: rows[case_id][key] for key in keys if key in rows[case_id]},
              "cohort": cohort} for case_id, cohort in selected.items()]
    (work / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases))
    plan = {
        "source_reports": SOURCES, "seeds": SEEDS, "strengths": STRENGTHS,
        "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
        "selection_seed": 20260917, "selected": selected,
        "held_out_case_ids": sorted(set(rows) - set(selected)),
        "note": "pilot 不作通过验收；随机对照与诊断富集分开汇总；未覆盖 seed+7。",
        "sha256": {name: hashlib.sha256((evaluation.EVAL_DIR / filename).read_bytes()).hexdigest()
                   for name, filename in SOURCES.items()},
    }
    write_json(work / "plan.json", plan)
    for name, report in reports.items():
        report["seed_stability"] = evaluation.seed_stability(report["items"])
        write_json(work / f"{name}_stability.json", report["seed_stability"])
        report["items"] = [item for item in report["items"] if item["case_id"] in selected]
        report.update(evaluation._report_metrics(report["items"]))
        write_json(work / f"{name}_historical_subset.json", report)
    print(f"Prepared {len(cases)} cases; {len(plan['held_out_case_ids'])} held out", flush=True)


def worker(work, shard):
    import numpy as np
    import soundfile as sf

    cases = [json.loads(line) for line in (work / "cases.jsonl").read_text().splitlines()]
    target = str(CHECKPOINT_DIR / "lora_omni5_r8/latest")
    model = infer.get_model(lora_dir=target)
    first = cases[shard[0]]
    kwargs = infer._gen_kwargs(first["text"], first["ref_audio"], None, 1.8, 20, 42)
    kwargs["retry_badcase"] = False
    endpoint_checks = []
    for strength in (0, 1):
        model.set_lora_enabled(bool(strength))
        original, _ = infer._run(model, kwargs)
        model.set_lora_enabled(True)
        scaled, _ = infer._run(model, kwargs, lora_strength=strength)
        original_wav, original_sr = sf.read(original)
        scaled_wav, scaled_sr = sf.read(scaled)
        same = original_sr == scaled_sr and np.array_equal(original_wav, scaled_wav)
        endpoint_checks.append({"strength": strength, "identical": same,
                                "toggle_wav": original, "scaled_wav": scaled})
        write_json(work / f"endpoints_{shard[0]}.json", endpoint_checks)
        if not same:
            raise RuntimeError(f"Endpoint strength={strength} differs from official toggle")
    for strength in STRENGTHS:
        destination = work / f"strength_{strength:g}_shard_{shard[0]}.json"
        if destination.exists():
            raise FileExistsError(destination)
        report = evaluation.evaluate(target, "th", cases, seeds=SEEDS,
                                     cfg_value=1.8, inference_timesteps=20,
                                     lora_strength=strength, shard=shard, progress=print)
        write_json(destination, report)
        print(f"Finished strength={strength} shard={shard[0]}: {report['report_path']}", flush=True)


def collect(work, shards):
    summaries = {}
    for strength in STRENGTHS:
        report = evaluation.merge_reports([
            str(work / f"strength_{strength:g}_shard_{index}.json") for index in range(shards)])
        report["cohorts"] = {
            cohort: evaluation._report_metrics([item for item in report["items"]
                                                if item["cohort"] == cohort])
            for cohort in sorted({item["cohort"] for item in report["items"]})}
        write_json(work / f"strength_{strength:g}.json", report)
        summaries[strength] = {key: report[key] for key in
                              ("report_path", "by_lang", "seed_stability", "cohorts")}
    write_json(work / "comparison.json", summaries)
    print("Collected all strengths; production unchanged.", flush=True)


def followup(work, shard):
    plan = json.loads((work / "plan.json").read_text())
    pilot = [json.loads(line) for line in (work / "cases.jsonl").read_text().splitlines()]
    cases = []
    for lang in ("th", "tl", "vi", "id", "ms", "zh", "en"):
        cases.append(min((case for case in pilot if case["lang"] == lang
                          and case["cohort"].startswith("random")),
                         key=lambda case: len(case["text"])))
    historical = json.loads((evaluation.EVAL_DIR /
                            "base_87eed5ce0e16475a9da9c1b9ea43b8b7.json").read_text())
    keys = ("case_id", "text", "lang", "ref_audio", "ref_lang", "control", "numeric")
    for item in historical["items"]:
        review = item.get("human_review") or {}
        if review.get("noise") is True or review.get("cutoff") is True:
            cases.append({**{key: item[key] for key in keys if key in item},
                          "cohort": "historical_acoustic_badcase"})
    full_r8 = json.loads((evaluation.EVAL_DIR / SOURCES["r8"]).read_text())
    drift_ids = {"id_nat_11", "id_nat_24", "tl_nat_09", "tl_nat_22"}
    drift_rows = {item["case_id"]: item for item in full_r8["items"]
                  if item["case_id"] in drift_ids}
    if set(drift_rows) != drift_ids:
        raise ValueError("Missing fixed cross-cue diagnostic cases")
    for case_id, item in sorted(drift_rows.items()):
        if not any(case["case_id"] == case_id for case in cases):
            cases.append({**{key: item[key] for key in keys if key in item},
                          "cohort": "speaker_drift_diagnostic_not_confirmed_badcase"})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Follow-up cases contain duplicate IDs")
    seeds = [42, 43, 44, 45, 49]
    settings = [(1.8, 20), (1.6, 20), (1.8, 28)]
    write_json(work / f"followup_plan_{shard[0]}.json", {
        "cases": cases, "seeds": seeds, "settings": settings, "strength": 1,
        "pilot_selection_seed": plan["selection_seed"],
        "note": "固定生产 r8 strength=1，分别改 CFG 或步数，不混调强度；覆盖 42→43/49。"
                "4 条跨 cue 诊断来自既有全量报告，不是人工确认的娃娃音。"})
    for cfg, steps in settings:
        destination = work / f"followup_cfg{cfg}_steps{steps}_shard{shard[0]}.json"
        if destination.exists():
            raise FileExistsError(destination)
        report = evaluation.evaluate(str(CHECKPOINT_DIR / "lora_omni5_r8/latest"), "th",
                                     cases, seeds=seeds, cfg_value=cfg,
                                     inference_timesteps=steps, shard=shard, progress=print)
        write_json(destination, report)
        print(f"Finished followup cfg={cfg} steps={steps} shard={shard[0]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "collect", "followup"))
    parser.add_argument("--work", type=Path, default=CHECKPOINT_DIR.parent / "quality_20260917")
    parser.add_argument("--shard", default="0/2")
    args = parser.parse_args()
    shard = tuple(map(int, args.shard.split("/")))
    if len(shard) != 2 or not 0 <= shard[0] < shard[1]:
        parser.error("--shard must be K/N with 0 <= K < N")
    if args.action == "prepare":
        prepare(args.work)
    elif args.action == "worker":
        worker(args.work, shard)
    elif args.action == "followup":
        followup(args.work, shard)
    else:
        collect(args.work, shard[1])
