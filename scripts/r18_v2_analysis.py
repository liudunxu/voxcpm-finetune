# -*- coding: utf-8 -*-
"""r18 v2 验收分析：r18 vs base / r13x 分语种配对 bootstrap（晋级口径见 docs/r18_plan_20261021.md）。"""
import json
import random
from pathlib import Path

EVAL_DIR = Path("/root/autodl-tmp/voxft_ckpt/eval")
BOOT_N, BOOT_SEED = 10000, 42

BASE = EVAL_DIR / "base_b34db7c942ec4464ae04543c02bb8f9a.json"
R13X = EVAL_DIR / "lora_omni5_r13x_idemph_3ep_latest_2bbd5a05aa8b4ed4a55073b011871c95.json"
R18 = EVAL_DIR / "lora_omni5_r18_idaccent_3ep_latest_14e9edd24d5a441288a77e1614b8e547.json"


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
    base = json.loads(BASE.read_text(encoding="utf-8"))
    r13x = json.loads(R13X.read_text(encoding="utf-8"))
    r18 = json.loads(R18.read_text(encoding="utf-8"))
    print(f"items: base {len(base['items'])} / r13x {len(r13x['items'])} / r18 {len(r18['items'])}")

    print("\n===== 总体指标 =====")
    for name, rep in [("base", base), ("r13x", r13x), ("r18", r18)]:
        bl = {k: v.get("mean_cer") for k, v in rep["by_lang"].items()}
        print("{} 总体 {:.4f} | th {:.4f} tl {:.4f} id {:.4f} zh {:.4f} en {:.4f} | SIM {:.4f} | 漏尾 {:.4f}".format(
            name, rep["mean_cer"], bl.get("th", 0), bl.get("tl", 0), bl.get("id", 0),
            bl.get("zh", 0), bl.get("en", 0), rep.get("mean_speaker_sim") or 0,
            rep.get("suspected_truncation_rate") or 0))

    print("\n===== 配对 ΔCER（负=r18 更好）=====")
    out = {}
    for other_name, other in [("base", base), ("r13x", r13x)]:
        for lang in ("th", "tl", "id", "zh", "en"):
            a, b = case_means(r18, "cer", lang), case_means(other, "cer", lang)
            d = {c: a[c] - b[c] for c in a if c in b}
            ci = boot_ci(d)
            sig = "显著" if ci[1] > 0 or ci[2] < 0 else "跨零"
            out[f"r18-{other_name}/{lang}"] = {"n": len(d), "mean": round(ci[0], 4),
                                               "ci95": [round(ci[1], 4), round(ci[2], 4)], "sig": sig}
            print("  r18-{} {:<3} ΔCER {:+.4f} [{:+.4f},{:+.4f}] {} (n={})".format(
                other_name, lang, ci[0], ci[1], ci[2], sig, len(d)))
    Path("/root/autodl-tmp/logs/r18_v2_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
