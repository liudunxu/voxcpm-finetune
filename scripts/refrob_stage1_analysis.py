"""refrob 阶段 1 基线分析：各降质条件 vs clean 的配对差值 + bootstrap 区间。

输入：/root/autodl-tmp/refrob_20261020/report_map.jsonl（12 份合并报告）。
口径：按 (case_id, seed) 配对，先在 case 内对 3 seed 取均值，再对 18 个 case
做 10000 次 bootstrap 重采样出 95% 区间（沿用 r11 配对统计口径：缺失不补零，
不把 54 条生成当 54 条独立文本）。分语种每格仅 6 case，只给均值不给区间。
输出：stdout 表格 + /root/autodl-tmp/refrob_20261020/paired_analysis.json。
"""
import json
import random
from pathlib import Path

WORK = Path("/root/autodl-tmp/refrob_20261020")
CONDS = ["clean", "babble_sir15", "babble_sir8", "lowpass4k", "lowpass3k",
         "combo_babble8_lp3k"]
BOOT_N = 10000
BOOT_SEED = 42


def load():
    data = {}  # (cond, tgt) -> {(case_id, seed): item}
    meta = {}
    for line in (WORK / "report_map.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        r = json.loads(Path(m["merged"]).read_text(encoding="utf-8"))
        tgt = "base" if m["target"] == "base" else "r8"
        data[(m["condition"], tgt)] = {(i["case_id"], i["seed"]): i for i in r["items"]}
        meta[(m["condition"], tgt)] = {"merged": m["merged"], "shards": m["shards"],
                                       "n_items": len(r["items"])}
    return data, meta


def case_means(items, key):
    """(case_id,seed) -> item 压成 case_id -> 均值。"""
    by_case: dict[str, list[float]] = {}
    for (cid, _seed), i in items.items():
        v = i.get(key)
        if v is not None:
            by_case.setdefault(cid, []).append(float(v))
    return {c: sum(v) / len(v) for c, v in by_case.items()}


def paired_delta(base_cases, cond_cases):
    """按 case 配对求差，返回 {case_id: delta}（两边都有的 case 才计入）。"""
    return {c: cond_cases[c] - base_cases[c] for c in base_cases if c in cond_cases}


def bootstrap_ci(deltas: dict, n=BOOT_N, seed=BOOT_SEED):
    vals = list(deltas.values())
    if not vals:
        return None
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    return (sum(vals) / len(vals), lo, hi)


def main():
    data, meta = load()
    out = {"boot_n": BOOT_N, "boot_seed": BOOT_SEED, "reports": {
        f"{c}/{t}": m for (c, t), m in meta.items()}, "targets": {}}
    for tgt in ("base", "r8"):
        clean = data[("clean", tgt)]
        tgt_out = {}
        for metric in ("cer", "speaker_sim", "tail_silence_sec", "audio_sec"):
            clean_cm = case_means(clean, metric)
            metric_out = {}
            for cond in CONDS[1:]:
                cond_cm = case_means(data[(cond, tgt)], metric)
                deltas = paired_delta(clean_cm, cond_cm)
                ci = bootstrap_ci(deltas)
                by_lang = {}
                for lang in ("th", "tl", "id"):
                    ld = [d for c, d in deltas.items() if c.startswith(f"{lang}_")]
                    if ld:
                        by_lang[lang] = round(sum(ld) / len(ld), 4)
                metric_out[cond] = {
                    "n_cases": len(deltas),
                    "mean_delta": round(ci[0], 4) if ci else None,
                    "ci95": [round(ci[1], 4), round(ci[2], 4)] if ci else None,
                    "by_lang_mean_delta": by_lang,
                }
            tgt_out[metric] = metric_out
        out["targets"][tgt] = tgt_out

    (WORK / "paired_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for tgt in ("base", "r8"):
        print(f"\n===== {tgt}：降质条件 − clean（case 内先对 3 seed 取均值）=====")
        print(f'{"metric":<14}{"condition":<22}{"Δ均值":>9}  {"95%区间":>18}  分语种Δ')
        for metric in ("cer", "speaker_sim", "tail_silence_sec", "audio_sec"):
            for cond in CONDS[1:]:
                o = out["targets"][tgt][metric][cond]
                ci = o["ci95"]
                ci_s = f"[{ci[0]:+.4f},{ci[1]:+.4f}]" if ci else "-"
                langs = " ".join(f"{k}:{v:+.3f}" for k, v in o["by_lang_mean_delta"].items())
                m = o["mean_delta"]
                print(f"{metric:<14}{cond:<22}{(f'{m:+.4f}' if m is not None else '-'):>9}  {ci_s:>18}  {langs}")


if __name__ == "__main__":
    main()
