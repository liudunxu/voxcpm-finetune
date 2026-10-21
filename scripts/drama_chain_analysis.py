# -*- coding: utf-8 -*-
"""drama_chain 2×2 对齐验收分析：ref 长短 × 情绪前缀有无，base vs r18。

从两份报告（base / ckpt）的 items 按 case_id 解析条件（*_nref_plain / *_nref_emo /
*_sref_plain / *_sref_emo），输出：
  1. 每模型 × ref × 情绪 的 mean_cer / speaker_sim / 尾静音 / speech_ratio
  2. 情绪前缀效应：同 ref 条件下 emo − plain 的 Δsim / ΔCER（音色漂移直接读数）
  3. 短 ref 效应：同情绪条件下 sref − nref 的 Δsim / ΔCER
  4. 分语种 CER 对照（r18 vs base，分条件）

用法：uv run python scripts/drama_chain_analysis.py base.json r18.json
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

COND = re.compile(r"_(nref|sref)_(plain|emo)$")


def load(path: str) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    out = []
    for it in items:
        m = COND.search(it["case_id"])
        if not m:
            continue
        out.append({**it, "ref_cond": m.group(1), "ctrl_cond": m.group(2),
                    "text_key": COND.sub("", it["case_id"])})
    return out


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def by_cond(items):
    g = defaultdict(list)
    for it in items:
        g[(it["ref_cond"], it["ctrl_cond"])].append(it)
    return g


def main():
    base_items, r18_items = load(sys.argv[1]), load(sys.argv[2])
    print(f"base n={len(base_items)}  r18 n={len(r18_items)}\n")

    print("== 1. 每模型 × ref × 情绪：CER / speaker_sim / 尾静音 ==")
    print(f"{'模型':<6}{'ref':<6}{'情绪':<7}{'CER':>8}{'SIM':>8}{'尾静音':>8}{'漏尾率':>8}")
    for name, items in (("base", base_items), ("r18", r18_items)):
        for (rc, cc), g in sorted(by_cond(items).items()):
            trunc = mean([bool(i["suspected_truncation"]) for i in g])
            print(f"{name:<6}{rc:<6}{cc:<7}{mean([i['cer'] for i in g]):>8.4f}"
                  f"{mean([i['speaker_sim'] for i in g]):>8.4f}"
                  f"{mean([i['tail_silence_sec'] for i in g]):>8.3f}{trunc:>8.3f}")

    print("\n== 2. 情绪前缀效应（emo − plain，同 ref 同 case 逐对再平均）==")
    for name, items in (("base", base_items), ("r18", r18_items)):
        idx = {(i["text_key"], i["ref_cond"], i["ctrl_cond"], i["seed"]): i for i in items}
        for rc in ("nref", "sref"):
            dsim, dcer = [], []
            for (tk, r, c, s), it in idx.items():
                if r == rc and c == "emo" and (tk, r, "plain", s) in idx:
                    p = idx[(tk, r, "plain", s)]
                    if it["speaker_sim"] is not None and p["speaker_sim"] is not None:
                        dsim.append(it["speaker_sim"] - p["speaker_sim"])
                    dcer.append(it["cer"] - p["cer"])
            print(f"  {name} {rc}: Δsim={mean(dsim):+.4f} (n={len(dsim)})  Δcer={mean(dcer):+.4f}")

    print("\n== 3. 短 ref 效应（sref − nref，同情绪同 case）==")
    for name, items in (("base", base_items), ("r18", r18_items)):
        idx = {(i["text_key"], i["ref_cond"], i["ctrl_cond"], i["seed"]): i for i in items}
        for cc in ("plain", "emo"):
            dsim, dcer = [], []
            for (tk, r, c, s), it in idx.items():
                if r == "sref" and c == cc and (tk, "nref", c, s) in idx:
                    p = idx[(tk, "nref", c, s)]
                    if it["speaker_sim"] is not None and p["speaker_sim"] is not None:
                        dsim.append(it["speaker_sim"] - p["speaker_sim"])
                    dcer.append(it["cer"] - p["cer"])
            print(f"  {name} {cc}: Δsim={mean(dsim):+.4f} (n={len(dsim)})  Δcer={mean(dcer):+.4f}")

    print("\n== 4. 分语种 CER（按条件）==")
    for rc in ("nref", "sref"):
        for cc in ("plain", "emo"):
            line = f"  {rc}/{cc}: "
            for lang in ("th", "id", "tl", "en"):
                b = mean([i["cer"] for i in base_items if i["lang"] == lang
                          and i["ref_cond"] == rc and i["ctrl_cond"] == cc])
                r = mean([i["cer"] for i in r18_items if i["lang"] == lang
                          and i["ref_cond"] == rc and i["ctrl_cond"] == cc])
                line += f"{lang} {b:.4f}→{r:.4f}   "
            print(line)


if __name__ == "__main__":
    main()
