#!/usr/bin/env python3
"""构建 r20 混合：r19_emotion 底 − thai_ser 减至 300 行 + id/tl 中性前缀接种。

依据（r19 验收归因）：
- r19 的 254 条英文情绪前缀全部落在 th，id/tl 对「英文前缀」输入形态 OOD；
- thai_ser 511 行把 th 份额从 27.7% 推到 32.9%，id 从 37.4% 稀释到 34.7%，
  弱条件（短 zh ref）下 id 解码稳定性被挤占（8/96 跑飞，r13x 同集 0/96）；
- r19 的情绪音色漂移改善（Δsim −0.024→−0.011）真实存在，前缀训练要保留，
  但 thai_ser 对 th 的 CER 收益≈0（r19 th 0.078 ≈ r13x 0.077），故减量。

做法：
- thai_ser：保留全部 254 条带前缀行 + 从其余行确定性抽 46 行 → 共 300 行
- id/tl：各抽 10% 行加中性英文前缀（轮换 4 个「无表演」短语，不伪造情绪标签，
  标记 control_inoculation=true 以便审计）
- 其余行原样保留；val.jsonl 原样拷贝；mix.json 重算 language_shares。
"""
import json
import random
import sys
from pathlib import Path

SRC = Path("/root/autodl-tmp/voxft_data/processed/r19_emotion")
DST = Path("/root/autodl-tmp/voxft_data/processed/r20_emofix")
NEUTRAL_PHRASES = [
    "neutral, natural tone",
    "calm and steady",
    "speak naturally and clearly",
    "relaxed and natural",
]
ID_TL_INOC_RATE = 0.10
THAI_SER_KEEP_EXTRA = 46  # 254 带前缀 + 46 = 300


def main() -> None:
    rows = [json.loads(x) for x in (SRC / "train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    thai_extra = [r for r in rows if r["source_id"] == "thai_ser" and not r["text"].startswith("(")]
    rng = random.Random(42)
    keep_extra = set(id(r) for r in rng.sample(thai_extra, THAI_SER_KEEP_EXTRA))
    dropped = 0
    for r in rows:
        if r["source_id"] == "thai_ser" and not r["text"].startswith("(") and id(r) not in keep_extra:
            dropped += 1
            continue
        out.append(r)
    print(f"thai_ser 裁掉 {dropped} 行，保留 {sum(1 for r in out if r['source_id'] == 'thai_ser')} 行")

    rng_id = random.Random(43)
    rng_tl = random.Random(44)
    inoc = {"id": 0, "tl": 0}
    for r in out:
        lg = r["lang"]
        if lg not in ("id", "tl") or r["text"].startswith("("):
            continue
        rng = rng_id if lg == "id" else rng_tl
        if rng.random() < ID_TL_INOC_RATE:
            phrase = NEUTRAL_PHRASES[inoc[lg] % len(NEUTRAL_PHRASES)]
            r["text"] = f"({phrase}){r['text']}"
            r["control"] = phrase
            r["control_inoculation"] = True
            inoc[lg] += 1
    print(f"中性前缀接种：id {inoc['id']} 行、tl {inoc['tl']} 行")

    DST.mkdir(parents=True, exist_ok=True)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DST / "val.jsonl").write_bytes((SRC / "val.jsonl").read_bytes())

    hours: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in out:
        hours[r["lang"]] = hours.get(r["lang"], 0.0) + r.get("duration", 0.0) / 3600
        counts[r["lang"]] = counts.get(r["lang"], 0) + 1
    total_h = sum(hours.values())
    shares = {lg: {"actual": round(h / total_h, 4), "hours": round(h, 3), "rows": counts[lg]}
              for lg, h in sorted(hours.items())}
    (DST / "mix.json").write_text(json.dumps({
        "derived_from": "r19_emotion（thai_ser 减至 300 行 + id/tl 10% 中性前缀接种）",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares,
        "neutral_inoculation": inoc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总计 {len(out)} 行 / {total_h:.2f}h")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
