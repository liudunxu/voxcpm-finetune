# -*- coding: utf-8 -*-
"""r13 v2 验收分析：r13 vs base/r8/r12a 分语种配对（门槛：th/tl/id 须比 base 强）。"""
import json
import random
from pathlib import Path

EVAL_DIR = Path("/root/autodl-tmp/voxft_ckpt/eval")
MAP = Path("/root/autodl-tmp/refrob_20261020/report_map_stage2.jsonl")
BOOT_N, BOOT_SEED = 10000, 42

# 今天的 v2clean 基线（阶段 2 验收沉淀）
BASELINES = {}
for line in MAP.read_text(encoding="utf-8").splitlines():
    m = json.loads(line)
    if m["tag"] == "v2clean":
        name = {"base": "base", "lora_omni5_r8_latest": "r8",
                "lora_omni5_r12a_refrob_clean_latest": "r12a"}.get(m["target"])
        if name:
            BASELINES[name] = json.loads(Path(m["merged"]).read_text(encoding="utf-8"))


def case_means(report, key, lang=None):
    by_case: dict[str, list[float]] = {}
    for i in report["items"]:
        v = i.get(key)
        if v is None or (lang and i.get("lang") != lang):
            continue
        by_case.setdefault(i["case_id"], []).append(float(v))
    return {c: sum(v) / len(v) for c, v in by_case.items()}


def boot_ci(deltas):
    vals = list(deltas.values())
    if not vals:
        return None
    rng = random.Random(BOOT_SEED)
    means = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals)
                   for _ in range(BOOT_N))
    return (sum(vals) / len(vals), means[int(0.025 * BOOT_N)], means[int(0.975 * BOOT_N)])


def main():
    r13_file = max(EVAL_DIR.glob("lora_omni5_r13_idemph_latest_*.json"), key=lambda p: p.stat().st_mtime)
    r13 = json.loads(r13_file.read_text(encoding="utf-8"))
    print("r13 报告:", r13_file.name, len(r13["items"]), "条")
    bl = {k: v.get("mean_cer") for k, v in r13["by_lang"].items()}
    print("r13 总体 CER {:.4f} | th {:.3f} tl {:.3f} id {:.3f} zh {:.3f} en {:.3f} | SIM {:.3f} | 尾静音 {:.3f}".format(
        r13["mean_cer"], bl.get("th", 0), bl.get("tl", 0), bl.get("id", 0),
        bl.get("zh", 0), bl.get("en", 0), r13.get("mean_speaker_sim") or 0,
        r13.get("mean_tail_silence") or 0))

    print("\n===== 配对 ΔCER（负=更好；晋级口径：th/tl/id 相对 base 区间整体 <0）=====")
    out = {}
    for other_name, other in [("base", BASELINES["base"]), ("r8", BASELINES["r8"]),
                              ("r12a", BASELINES["r12a"])]:
        for lang in ("th", "tl", "id"):
            a, b = case_means(r13, "cer", lang), case_means(other, "cer", lang)
            d = {c: a[c] - b[c] for c in a if c in b}
            ci = boot_ci(d)
            out[f"r13-{other_name}/{lang}"] = {"n": len(d), "mean": round(ci[0], 4),
                                               "ci95": [round(ci[1], 4), round(ci[2], 4)]}
            print("  r13-{} {:<3} ΔCER {:+.4f} [{:+.4f},{:+.4f}] (n={})".format(
                other_name, lang, ci[0], ci[1], ci[2], len(d)))
    Path("/root/autodl-tmp/refrob_20261020/r13_v2_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
