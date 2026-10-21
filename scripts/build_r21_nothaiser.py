#!/usr/bin/env python3
"""构建 r21 续训数据：r20_emofix 去掉全部 thai_ser 行（保留 id/tl 中性前缀接种）。

依据：r20 证明 300 行 thai_ser 仍留 5/96 id 跑飞（r13x 为 0），thai_ser 是失稳源；
r20→r21 从 r20 checkpoint 续训（LR 接续余弦尾部 ~1.4e-5），数据切到去 thai_ser 的
纠正混合，让 id 份额回到 r13x 的 ~37.4%。
"""
import json
import collections
import shutil
from pathlib import Path

SRC = Path("/root/autodl-tmp/voxft_data/processed/r20_emofix")
DST = Path("/root/autodl-tmp/voxft_data/processed/r21_nothaiser")


def main() -> None:
    rows = [json.loads(x) for x in (SRC / "train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = [r for r in rows if r["source_id"] != "thai_ser"]
    hours: dict[str, float] = {}
    counts: dict[str, int] = {}
    inoc = 0
    for r in out:
        hours[r["lang"]] = hours.get(r["lang"], 0.0) + r.get("duration", 0.0) / 3600
        counts[r["lang"]] = counts.get(r["lang"], 0) + 1
        inoc += 1 if r.get("control_inoculation") else 0
    total_h = sum(hours.values())
    DST.mkdir(parents=True, exist_ok=True)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(SRC / "val.jsonl", DST / "val.jsonl")
    shares = {lg: {"actual": round(h / total_h, 4), "hours": round(h, 3), "rows": counts[lg]}
              for lg, h in sorted(hours.items())}
    (DST / "mix.json").write_text(json.dumps({
        "derived_from": "r20_emofix 去掉全部 thai_ser（保留 id/tl 中性前缀接种）",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares, "neutral_inoculation_rows": inoc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(out)} 行 / {total_h:.2f}h, 接种行 {inoc}")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")


if __name__ == "__main__":
    main()
