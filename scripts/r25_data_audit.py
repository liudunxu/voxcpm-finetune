"""阶段 A：跑飞的数据侧归因审计（只读）。

对比 r18（垫尾仅 +0.14s、id 探针 0.0013、冷启动 1390 步）与 r23（id p90 尾静音 1.02s、
跑飞 3/96、续训 2909 步）两份训练清单，逐源拆四个维度：目标时长、rate（文本-时长错配）、
情绪前缀行、源构成。判据在 docs/finetune_playbook.md 已写死，本脚本只出数字不下结论。

严格只读：只打开清单，不写任何数据目录。
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/autodl-tmp/voxft_data/processed")
RUNS = {"r18": ROOT / "r18_idaccent" / "train.jsonl",
        "r23": ROOT / "r23_thtlboost" / "train.jsonl"}
TARGET_LANGS = ("id", "th", "tl")


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


def has_prefix(text):
    return bool(text) and text.lstrip().startswith("(")


def main():
    data = {name: load(p) for name, p in RUNS.items()}
    for name, rows in data.items():
        hours = sum(r.get("duration") or 0 for r in rows) / 3600
        print(f"[{name}] {RUNS[name]}  rows={len(rows)}  hours={hours:.3f}")

    # ---------- 1) 源构成 ----------
    print("\n=== 1) 源构成（rows / hours），r23 − r18 ===")
    comp = {}
    for name, rows in data.items():
        c = defaultdict(lambda: [0, 0.0])
        for r in rows:
            key = (r.get("lang"), r.get("source_id"))
            c[key][0] += 1
            c[key][1] += (r.get("duration") or 0) / 3600
        comp[name] = c
    keys = sorted(set(comp["r18"]) | set(comp["r23"]),
                  key=lambda k: (k[0] not in TARGET_LANGS, k[0] or "", k[1] or ""))
    print(f"  {'lang':<5}{'source_id':<22}{'r18 rows':>9}{'r23 rows':>9}{'Δrows':>8}"
          f"{'r18 h':>8}{'r23 h':>8}{'Δh':>8}")
    for k in keys:
        a = comp["r18"].get(k, [0, 0.0])
        b = comp["r23"].get(k, [0, 0.0])
        if a[0] == b[0] and abs(a[1] - b[1]) < 0.005:
            continue
        print(f"  {str(k[0]):<5}{str(k[1]):<22}{a[0]:>9}{b[0]:>9}{b[0] - a[0]:>+8}"
              f"{a[1]:>8.3f}{b[1]:>8.3f}{b[1] - a[1]:>+8.3f}")

    # ---------- 2) 目标时长分布 ----------
    print("\n=== 2) 目标时长分布（秒）===")
    for lang in TARGET_LANGS:
        print(f"  [{lang}]")
        print(f"    {'run':<5}{'n':>6}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>8}"
              f"{'>10s':>7}{'>15s':>7}{'>20s':>7}{'占比>15s':>10}")
        for name, rows in data.items():
            d = [r.get("duration") or 0 for r in rows if r.get("lang") == lang]
            if not d:
                continue
            n = len(d)
            print(f"    {name:<5}{n:>6}{pct(d, .5):>7.2f}{pct(d, .9):>7.2f}{pct(d, .99):>7.2f}"
                  f"{max(d):>8.2f}{sum(1 for x in d if x > 10):>7}{sum(1 for x in d if x > 15):>7}"
                  f"{sum(1 for x in d if x > 20):>7}{sum(1 for x in d if x > 15) / n:>10.2%}")

    print("\n  >15s 与 >20s 的行按源拆分（r23）：")
    for thr in (15, 20):
        c = Counter((r.get("lang"), r.get("source_id")) for r in data["r23"]
                    if (r.get("duration") or 0) > thr)
        c18 = Counter((r.get("lang"), r.get("source_id")) for r in data["r18"]
                      if (r.get("duration") or 0) > thr)
        print(f"    >{thr}s: " + ", ".join(f"{k[0]}/{k[1]} {v}(r18 {c18.get(k, 0)})"
                                            for k, v in c.most_common(8)) or f"    >{thr}s: 无")

    # ---------- 3) rate（文本-时长错配）----------
    print("\n=== 3) rate 分布（量纲按语种，不横比；低 rate = 长停顿/慢速）===")
    for lang in TARGET_LANGS:
        print(f"  [{lang}]")
        print(f"    {'run':<5}{'n':>6}{'p5':>8}{'p10':>8}{'p50':>8}{'p90':>8}"
              f"{'<p10(r23)行数':>14}{'占比':>8}")
        base_p10 = None
        for name, rows in data.items():
            v = [r.get("rate") for r in rows if r.get("lang") == lang and r.get("rate") is not None]
            if not v:
                continue
            p10 = pct(v, .10)
            if name == "r23":
                base_p10 = p10
            print(f"    {name:<5}{len(v):>6}{pct(v, .05):>8.2f}{p10:>8.2f}{pct(v, .5):>8.2f}"
                  f"{pct(v, .9):>8.2f}", end="")
            if base_p10 is not None:
                low = sum(1 for x in v if x < base_p10)
                print(f"{low:>14}{low / len(v):>8.2%}")
            else:
                print(f"{'-':>14}{'-':>8}")

    print("\n  「长音频 + 低 rate」错配行（duration>10s 且 rate < 该语种 r23 的 p10）：")
    for lang in TARGET_LANGS:
        v23 = [r.get("rate") for r in data["r23"] if r.get("lang") == lang and r.get("rate") is not None]
        v18 = [r.get("rate") for r in data["r18"] if r.get("lang") == lang and r.get("rate") is not None]
        if not v23 or not v18:
            continue
        t23, t18 = pct(v23, .10), pct(v18, .10)
        n23 = [r for r in data["r23"] if r.get("lang") == lang and (r.get("duration") or 0) > 10
               and (r.get("rate") or 99) < t23]
        n18 = [r for r in data["r18"] if r.get("lang") == lang and (r.get("duration") or 0) > 10
               and (r.get("rate") or 99) < t18]
        src23 = Counter(r.get("source_id") for r in n23)
        print(f"    {lang}: r18 {len(n18)}/{len(data['r18'])} ({len(n18)/len(data['r18']):.2%})  "
              f"r23 {len(n23)}/{len(data['r23'])} ({len(n23)/len(data['r23']):.2%})  "
              f"r23 按源={dict(src23.most_common(5))}")

    # ---------- 4) 情绪前缀行 ----------
    print("\n=== 4) 情绪/控制前缀行（text 以 '(' 开头）===")
    for name, rows in data.items():
        pre = [r for r in rows if has_prefix(r.get("text"))]
        bylang = Counter(r.get("lang") for r in pre)
        bysrc = Counter(r.get("source_id") for r in pre)
        d = [r.get("duration") or 0 for r in pre]
        v = [r.get("rate") for r in pre if r.get("rate") is not None]
        print(f"  [{name}] 前缀行 {len(pre)}/{len(rows)} ({len(pre)/len(rows):.2%}) 语种={dict(bylang)}")
        print(f"         源={dict(bysrc.most_common(5))}")
        if d:
            print(f"         duration p50={pct(d,.5):.2f} p90={pct(d,.9):.2f} max={max(d):.2f}"
                  f"{'  |  rate p10=' + format(pct(v, .10), '.2f') + ' p50=' + format(pct(v, .5), '.2f') if v else ''}")
        ev = Counter((r.get("lang"), bool(r.get("emotion_verified"))) for r in rows)
        print(f"         emotion_verified=true 行数={sum(v2 for k, v2 in ev.items() if k[1])}")

    # ---------- 5) ref / 身份覆盖 ----------
    print("\n=== 5) ref 与身份覆盖（与音色一致性相关）===")
    for name, rows in data.items():
        with_ref = sum(1 for r in rows if r.get("ref_audio"))
        ver = sum(1 for r in rows if r.get("speaker_verified"))
        cross = sum(1 for r in rows if r.get("ref_audio") and r.get("ref_lang")
                    and r.get("ref_lang") != r.get("lang"))
        print(f"  [{name}] 带 ref {with_ref}/{len(rows)} ({with_ref/len(rows):.2%})  "
              f"speaker_verified {ver} ({ver/len(rows):.2%})  跨语种 ref {cross}")
        per_lang = defaultdict(lambda: [0, 0])
        for r in rows:
            if r.get("lang") in TARGET_LANGS:
                per_lang[r["lang"]][1] += 1
                if r.get("ref_audio"):
                    per_lang[r["lang"]][0] += 1
        print("         分语种带 ref: " + ", ".join(
            f"{k} {v[0]}/{v[1]} ({v[0]/v[1]:.1%})" for k, v in sorted(per_lang.items())))


if __name__ == "__main__":
    main()
