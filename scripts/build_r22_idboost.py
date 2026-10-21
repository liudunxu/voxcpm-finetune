#!/usr/bin/env python3
"""构建 r22 续训混合：r21_nothaiser 底 + gigaspeech2_id 补 800 行自然口语（id 音色地道性优先）。

约束：
- 只从 processed/gigaspeech2_id 里挑「r21 train 未使用、且不在 val.jsonl」的行
  （按 audio 路径排除，防验证集泄漏与重复曝光）
- 新行对齐 r21 train 的列集（HF datasets 要求 train/val schema 一致）；
  源级加工未做 ref 配对，新增行 ref_audio=None（混合里 30-50% 带 ref 即可，
  r21 已有配对行，新增行当无 ref 样本）
- 确定性抽样 seed=45
"""
import json
import random
from pathlib import Path

R21 = Path("/root/autodl-tmp/voxft_data/processed/r21_nothaiser")
GIG = Path("/root/autodl-tmp/voxft_data/processed/gigaspeech2_id")
DST = Path("/root/autodl-tmp/voxft_data/processed/r22_idboost")
ADD = 800


def load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    train = load(R21 / "train.jsonl")
    val = load(R21 / "val.jsonl")
    used = {r["audio"] for r in train} | {r["audio"] for r in val}
    cols = set()
    for r in train:
        cols.update(r.keys())

    pool = [r for r in load(GIG / "train.jsonl") if r.get("audio") and r["audio"] not in used]
    print(f"gigaspeech2_id 可用池 {len(pool)} 行")
    rng = random.Random(45)
    add = rng.sample(pool, ADD)
    for r in add:
        for k in list(r.keys()):
            if k not in cols:
                del r[k]
        for k in cols - set(r.keys()):
            r[k] = None
        r["source_id"] = "gigaspeech2_id_r22boost"
    out = train + add

    DST.mkdir(parents=True, exist_ok=True)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import shutil
    shutil.copy(R21 / "val.jsonl", DST / "val.jsonl")

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
        "derived_from": "r21_nothaiser + gigaspeech2_id 800 行（音色地道性优先，跑飞容忍）",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares, "added_id_rows": ADD,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总计 {len(out)} 行 / {total_h:.2f}h")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")


if __name__ == "__main__":
    main()
