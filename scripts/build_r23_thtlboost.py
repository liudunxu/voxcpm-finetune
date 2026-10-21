#!/usr/bin/env python3
"""构建 r23 续训混合：r22_idboost 底 + th/tl 增量（音色地道性优先口径）。

- th +800：gigaspeech2_th 500（自然口语锚点）+ cv22_th 300（朗读补充）
- tl +500：fleurs_tl 剩余未用行（tl 唯一已加工源；fleurs 行长，800 行会把 tl 时长份额推过 id，裁到 500）
- 排除 r22 train/val 已用 audio 路径（防重复曝光与验证集泄漏）；
  新行对齐 r22 train 列集（HF datasets schema 一致），ref_audio=None
- 确定性抽样 seed=46
"""
import json
import random
import shutil
from pathlib import Path

R22 = Path("/root/autodl-tmp/voxft_data/processed/r22_idboost")
PROC = Path("/root/autodl-tmp/voxft_data/processed")
DST = Path("/root/autodl-tmp/voxft_data/processed/r23_thtlboost")
PLAN = [("gigaspeech2_th", 500), ("cv22_th", 300), ("fleurs_tl", 500)]


def load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    train = load(R22 / "train.jsonl")
    val = load(R22 / "val.jsonl")
    used = {r["audio"] for r in train} | {r["audio"] for r in val}
    cols = set()
    for r in train:
        cols.update(r.keys())

    rng = random.Random(46)
    add_all = []
    for src, want in PLAN:
        pool = [r for r in load(PROC / src / "train.jsonl") if r.get("audio") and r["audio"] not in used]
        take = rng.sample(pool, min(want, len(pool)))
        print(f"{src}: 可用 {len(pool)} 行，取 {len(take)} 行")
        for r in take:
            for k in list(r.keys()):
                if k not in cols:
                    del r[k]
            for k in cols - set(r.keys()):
                r[k] = None
            r["source_id"] = f"{src}_r23boost"
        add_all.extend(take)
    out = train + add_all

    DST.mkdir(parents=True, exist_ok=True)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(R22 / "val.jsonl", DST / "val.jsonl")

    hours: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in out:
        lg = r["lang"]
        hours[lg] = hours.get(lg, 0.0) + (r.get("duration") or 0.0) / 3600
        counts[lg] = counts.get(lg, 0) + 1
    total_h = sum(hours.values())
    shares = {lg: {"actual": round(h / total_h, 4), "hours": round(h, 3), "rows": counts[lg]}
              for lg, h in sorted(hours.items())}
    (DST / "mix.json").write_text(json.dumps({
        "derived_from": "r22_idboost + th 800 行（gigaspeech2_th 500 + cv22_th 300）+ tl fleurs_tl 剩余全取",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总计 {len(out)} 行 / {total_h:.2f}h")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")


if __name__ == "__main__":
    main()
