# -*- coding: utf-8 -*-
"""短 cue 门分析：base/r8/r12a 在 117 条平行短文本上的对照。

配对口径与前几轮一致：case 内先对 3 seed 取均值，bootstrap(10000, seed 42) 按 case 重采样。
分语种每格 39 case（13 句 × 3 ref_lang）。
输出 stdout 表格 + /root/autodl-tmp/refrob_20261020/short_gate_analysis.json。
"""
import json
import random
from pathlib import Path

WORK = Path("/root/autodl-tmp/refrob_20261020")
BOOT_N, BOOT_SEED = 10000, 42


def load():
    out = {}
    for line in (WORK / "report_map_short_gate.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        r = json.loads(Path(m["merged"]).read_text(encoding="utf-8"))
        name = {"base": "base", "lora_omni5_r8_latest": "r8",
                "lora_omni5_r12a_refrob_clean_latest": "r12a"}[m["target"]]
        out[name] = r
    return out


def case_means(report, key, lang=None):
    by_case: dict[str, list[float]] = {}
    for i in report["items"]:
        v = i.get(key)
        if v is None or (lang and i.get("lang") != lang):
            continue
        by_case.setdefault(i["case_id"], []).append(float(v))
    return {c: sum(v) / len(v) for c, v in by_case.items()}


def rate(report, key, lang=None):
    vals = [i[key] for i in report["items"]
            if i.get(key) is not None and (not lang or i.get("lang") == lang)]
    return sum(bool(v) for v in vals) / len(vals) if vals else None


def boot_ci(deltas, n=BOOT_N, seed=BOOT_SEED):
    vals = list(deltas.values())
    if not vals:
        return None
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals)
                   for _ in range(n))
    return (sum(vals) / len(vals), means[int(0.025 * n)], means[int(0.975 * n)])


def main():
    reps = load()
    out = {"per_model_lang": {}, "paired_vs_base": {}}
    print("===== 短 cue 门：每模型 × 语种（CER / 漏尾率 / 多读率 / 时长 / SIM）=====")
    for lang in ("th", "tl", "id"):
        print(f"-- {lang} --")
        for m in ("base", "r8", "r12a"):
            r = reps[m]
            cer = case_means(r, "cer", lang)
            sim = case_means(r, "speaker_sim", lang)
            sec = case_means(r, "audio_sec", lang)
            row = {"cer": round(sum(cer.values()) / len(cer), 4),
                   "trunc": round(rate(r, "suspected_truncation", lang), 4),
                   "over": round(rate(r, "over_read", lang), 4),
                   "audio_sec": round(sum(sec.values()) / len(sec), 3),
                   "sim": round(sum(sim.values()) / len(sim), 4)}
            out["per_model_lang"][f"{m}/{lang}"] = row
            print(f'  {m:<5} CER {row["cer"]:.4f}  漏尾 {row["trunc"]:.3f}  '
                  f'多读 {row["over"]:.3f}  时长 {row["audio_sec"]:.2f}s  SIM {row["sim"]:.3f}')
    print("\n===== 配对 Δ vs base（负=更好；区间跨零=不可辨）=====")
    for m in ("r8", "r12a"):
        for lang in ("th", "tl", "id"):
            d = {c: a - b for c, a in case_means(reps[m], "cer", lang).items()
                 if c in (b_ := case_means(reps["base"], "cer", lang)) for b in [b_[c]]}
            ci = boot_ci(d)
            out["paired_vs_base"][f"{m}/{lang}"] = {
                "n": len(d), "mean": round(ci[0], 4),
                "ci95": [round(ci[1], 4), round(ci[2], 4)]}
            print(f'  {m}/{lang}: ΔCER {ci[0]:+.4f} [{ci[1]:+.4f},{ci[2]:+.4f}] (n={len(d)})')
    (WORK / "short_gate_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 short_gate_analysis.json")


if __name__ == "__main__":
    main()
