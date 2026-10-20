# -*- coding: utf-8 -*-
"""refrob 阶段 2/3 验收分析：v2 干净红线 + 降质套件对照。

输入：
- /root/autodl-tmp/refrob_20261020/report_map_stage2.jsonl（tag=v2clean 的 4 份 +
  tag=refrob_<cond> 的 12 份合并报告）
- /root/autodl-tmp/refrob_20261020/report_map.jsonl（阶段 1 的 base/r8 降质报告，配对用）

口径：同 case/seed 配对，case 内先对 3 seed 取均值，bootstrap(10000, seed 42)
按 case 重采样出 95% 区间。红线：th/tl/id 分语种 ΔCER>0.05（相对 base）判退化。
输出：stdout 表格 + /root/autodl-tmp/refrob_20261020/stage2_analysis.json。
"""
import json
import random
from pathlib import Path

WORK = Path("/root/autodl-tmp/refrob_20261020")
CONDS = ["clean", "babble_sir15", "babble_sir8", "lowpass4k", "lowpass3k",
         "combo_babble8_lp3k"]
RED_LINE = 0.05
BOOT_N, BOOT_SEED = 10000, 42


def load_map(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        r = json.loads(Path(m["merged"]).read_text(encoding="utf-8"))
        key = (m.get("tag") or m["condition"], m["target"])
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
    return {c: a[c] - b[c] for c in a if c in b}


def fmt_ci(ci, nd=4):
    if not ci:
        return "-"
    return f"{ci[0]:+.{nd}f} [{ci[1]:+.{nd}f},{ci[2]:+.{nd}f}]"


def main():
    s2 = load_map(WORK / "report_map_stage2.jsonl")
    s1 = load_map(WORK / "report_map.jsonl")
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
            d = delta(case_means(v2[m], "cer", {lang}), case_means(v2["base"], "cer", {lang}))
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

    (WORK / "stage2_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 stage2_analysis.json")


if __name__ == "__main__":
    main()
