# -*- coding: utf-8 -*-
"""refrob 阶段 2/3 验收分析：v2 干净红线 + 降质套件对照。

输入：
- /root/autodl-tmp/refrob_20261020/report_map_stage2.jsonl（tag=v2clean 的 4 份 +
  tag=refrob_<cond> 的 12 份合并报告）
- /root/autodl-tmp/refrob_20261020/report_map.jsonl（阶段 1 的 base/r8 降质报告，配对用）

口径：同 case/seed 配对，case 内先对 3 seed 取均值，bootstrap(10000, seed 42)
按 case 重采样出 95% 区间。红线：th/tl/id 分语种 ΔCER>0.05（相对 base）判退化。
输出：stdout 表格 + 独立复核报告（拒绝覆盖），不改历史 stage2_analysis.json。
"""
import argparse
import json
import math
import random
from pathlib import Path

from ref_ab import sha256

WORK = Path("/root/autodl-tmp/refrob_20261020")
CONDS = ["clean", "babble_sir15", "babble_sir8", "lowpass4k", "lowpass3k",
         "combo_babble8_lp3k"]
RED_LINE = 0.05
BOOT_N, BOOT_SEED = 10000, 42


def load_map(path, inputs=None):
    out = {}
    if inputs is not None:
        inputs[str(path)] = sha256(path)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if inputs is not None:
            inputs[m["merged"]] = sha256(m["merged"])
        r = json.loads(Path(m["merged"]).read_text(encoding="utf-8"))
        key = (m.get("tag") or m["condition"], m["target"])
        if key in out:
            raise ValueError(f"Duplicate report mapping: {key}")
        out[key] = r
    return out


def case_means(report, key, langs=None):
    by_case: dict[str, list[float]] = {}
    for i in report["items"]:
        v = i.get(key)
        if v is None:
            continue
        if langs and i.get("lang") not in langs:
            continue
        by_case.setdefault(i["case_id"], []).append(float(v))
    return {c: sum(v) / len(v) for c, v in by_case.items()}


def boot_ci(deltas: dict, n=BOOT_N, seed=BOOT_SEED):
    vals = list(deltas.values())
    if not vals:
        return None
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals)
                   for _ in range(n))
    return (sum(vals) / len(vals), means[int(0.025 * n)], means[int(0.975 * n)])


def delta(a, b):  # a - b 按 case 配对
    if a.keys() != b.keys():
        raise ValueError("Incomplete paired case sets")
    return {c: a[c] - b[c] for c in a}


def paired_changes(candidate, baseline, metric, lang=None):
    for field in ("cfg_value", "inference_timesteps", "retry_badcase", "asr_model",
                  "asr_auto_detect_langs"):
        if candidate.get(field) != baseline.get(field):
            raise ValueError(f"Different evaluation configuration: {field}")
    indices = [{(row["case_id"], row["seed"]): row for row in report["items"]}
               for report in (candidate, baseline)]
    if any(len(index) != len(report["items"])
           for index, report in zip(indices, (candidate, baseline))):
        raise ValueError("Duplicate case/seed")
    if indices[0].keys() != indices[1].keys():
        raise ValueError("Incomplete paired case/seed sets")
    grouped = {}
    for key, row in indices[0].items():
        other = indices[1][key]
        for field in ("text", "lang", "control", "ref_audio", "ref_lang",
                      "ref_audio_original", "ref_degradation"):
            if row.get(field) != other.get(field):
                raise ValueError(f"Different paired condition: {key}/{field}")
        if lang is not None and row["lang"] != lang:
            continue
        values = (row.get(metric), other.get(metric))
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Missing/nonfinite paired metric: {key}/{metric}")
        grouped.setdefault(key[0], []).append(float(values[0]) - float(values[1]))
    return {case: sum(values) / len(values) for case, values in grouped.items()}


def fmt_ci(ci, nd=4):
    if not ci:
        return "-"
    return f"{ci[0]:+.{nd}f} [{ci[1]:+.{nd}f},{ci[2]:+.{nd}f}]"


def main(work=WORK, output=None):
    output = output or work / "stage2_analysis_review_20261022.json"
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    s2 = load_map(work / "report_map_stage2.jsonl", inputs)
    s1 = load_map(work / "report_map.jsonl", inputs)
    names = {"base": "base", "lora_omni5_r8_latest": "r8",
             "lora_omni5_r12a_refrob_clean_latest": "r12a",
             "lora_omni5_r12b_refrob_deg50_latest": "r12b"}
    v2 = {names[t]: r for (tag, t), r in s2.items() if tag == "v2clean"}
    out: dict = {"v2clean": {}, "refrob": {}}

    print("===== 阶段 A：v2 全量（118 case × 3 seed，cfg1.8/20步/禁重试）=====")
    print(f'{"model":<6}{"CER":>8}{"th":>7}{"tl":>7}{"id":>7}{"zh":>7}{"en":>7}'
          f'{"SIM":>7}{"漏尾":>6}{"多读":>6}{"尾静音":>7}')
    for m in ("base", "r8", "r12a", "r12b"):
        r = v2[m]
        bl = {k: v.get("mean_cer") for k, v in r.get("by_lang", {}).items()}
        print(f'{m:<6}{r["mean_cer"]:>8.4f}'
              + "".join(f'{bl.get(l, 0):>7.3f}' for l in ("th", "tl", "id", "zh", "en"))
              + f'{r.get("mean_speaker_sim", 0):>7.3f}'
              f'{r.get("suspected_truncation_rate", 0):>6.3f}'
              f'{r.get("over_read_rate", 0):>6.3f}{r.get("mean_tail_silence", 0):>7.3f}')
        out["v2clean"][m] = {
            "mean_cer": r["mean_cer"], "by_lang_cer": bl,
            "mean_speaker_sim": r.get("mean_speaker_sim"),
            "truncation_rate": r.get("suspected_truncation_rate"),
            "over_read_rate": r.get("over_read_rate"),
            "mean_tail_silence": r.get("mean_tail_silence"),
        }

    print("\n--- 红线：相对 base 的分语种 ΔCER（正=变差；>0.05 判退化）---")
    redline = {}
    for m in ("r8", "r12a", "r12b"):
        row = {}
        for lang in ("th", "tl", "id"):
            d = paired_changes(v2[m], v2["base"], "cer", lang)
            ci = boot_ci(d)
            row[lang] = fmt_ci(ci, 4)
        redline[m] = row
        print(f"{m:<6}" + "  ".join(f"{l}: {row[l]}" for l in ("th", "tl", "id")))
    out["v2clean"]["redline_vs_base"] = redline

    print("\n===== 阶段 B：降质套件（r12a/r12b 与阶段 1 的 base/r8 同条件配对）=====")
    print("Δ = 该降质条件 − 同模型 clean（case 内 3 seed 均值后配对）")
    rob = {}
    # 阶段 1 报告的 target 键是全名，阶段 2 同理；统一映射到短名查表
    key1 = {"base": "base", "r8": "lora_omni5_r8_latest"}
    key2 = {"r12a": "lora_omni5_r12a_refrob_clean_latest",
            "r12b": "lora_omni5_r12b_refrob_deg50_latest"}
    for m in ("base", "r8", "r12a", "r12b"):
        clean_r = s1[("clean", key1[m])] if m in key1 else s2[("refrob_clean", key2[m])]
        for metric in ("speaker_sim", "cer", "audio_sec"):
            for cond in CONDS[1:]:
                r = s1[(cond, key1[m])] if m in key1 else s2[(f"refrob_{cond}", key2[m])]
                d = delta(case_means(r, metric), case_means(clean_r, metric))
                ci = boot_ci(d)
                rob.setdefault(m, {}).setdefault(metric, {})[cond] = {
                    "n": len(d), "mean": round(ci[0], 4) if ci else None,
                    "ci95": [round(ci[1], 4), round(ci[2], 4)] if ci else None}
    for metric in ("speaker_sim", "cer", "audio_sec"):
        print(f"\n-- Δ{metric} --")
        print(f'{"condition":<22}' + "".join(f'{m:>28}' for m in ("base", "r8", "r12a", "r12b")))
        for cond in CONDS[1:]:
            cells = []
            for m in ("base", "r8", "r12a", "r12b"):
                o = rob[m][metric][cond]
                cells.append(fmt_ci((o["mean"], *o["ci95"]), 3) if o["ci95"] else "-")
            print(f"{cond:<22}" + "".join(f"{c:>28}" for c in cells))
    out["refrob"] = rob
    out["direct_b_minus_a"] = {}
    out["degradation_effect_b_minus_a"] = {}
    metrics = ("cer", "suspected_truncation", "over_read", "tail_silence_sec",
               "speech_ratio", "speaker_sim")
    for metric in metrics:
        for lang in (None, "th", "tl", "id"):
            group = lang or "all"
            clean_delta = paired_changes(s2[("refrob_clean", key2["r12b"])],
                                         s2[("refrob_clean", key2["r12a"])], metric, lang)
            for cond in CONDS:
                direct = paired_changes(s2[(f"refrob_{cond}", key2["r12b"])],
                                        s2[(f"refrob_{cond}", key2["r12a"])], metric, lang)
                effect = delta(direct, clean_delta)
                for name, values in (("direct_b_minus_a", direct),
                                     ("degradation_effect_b_minus_a", effect)):
                    ci = boot_ci(values)
                    out[name].setdefault(metric, {}).setdefault(group, {})[cond] = {
                        "cases": len(values), "mean": ci[0] if ci else None,
                        "ci95": list(ci[1:]) if ci else None,
                    }
    out["interpretation"] = {
        "primary": ["cer", "suspected_truncation", "over_read", "acoustic_review"],
        "degraded_speaker_sim": "reported_only_not_a_veto",
        "acoustic_review": "unverified",
        "promotion": "not_determined_by_this_reanalysis",
        "note": "Direct B-A and difference-in-differences are separate; per-language n=6. "
                "Historical reports, metrics, audio and verdicts remain unchanged.",
    }
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("Historical input changed during reanalysis")
    out["inputs_sha256"] = inputs
    with output.open("x", encoding="utf-8") as stream:
        json.dump(out, stream, ensure_ascii=False, indent=2)
    print(f"\n已写独立复核报告 {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.work, args.output)
