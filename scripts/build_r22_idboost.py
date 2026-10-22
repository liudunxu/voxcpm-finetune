#!/usr/bin/env python3
"""构建 r22 续训混合：r21_nothaiser 底 + gigaspeech2_id 补 800 行自然口语（id 音色地道性优先）。

约束：
- 只取未用目标音频/原音频，排除评测文本及验证音频/身份/session，保持 3-20s
- 新行对齐 r21 train 的列集（HF datasets 要求 train/val schema 一致）；
  保留原 source_id 与可信 ref，实际覆盖率写入 mix.json，不伪造配对
- 已存在输出拒绝覆盖；历史 r22 清单不重新构建
- 确定性抽样 seed=45
"""
import json
import random
from pathlib import Path

from incremental_mix import align_addition, describe_mix, eligible_additions, load_exclusions, validate_mix

R21 = Path("/root/autodl-tmp/voxft_data/processed/r21_nothaiser")
GIG = Path("/root/autodl-tmp/voxft_data/processed/gigaspeech2_id")
DST = Path("/root/autodl-tmp/voxft_data/processed/r22_idboost")
ADD = 800


def load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"Frozen output already exists: {DST}")
    train = load(R21 / "train.jsonl")
    val = load(R21 / "val.jsonl")
    excluded, eval_hashes = load_exclusions()
    validate_mix(train, val, excluded)
    cols = set()
    for r in train:
        cols.update(r.keys())

    pool, rejected = eligible_additions(load(GIG / "train.jsonl"), train, val, excluded)
    print(f"gigaspeech2_id 可用池 {len(pool)} 行；排除 {rejected}")
    if len(pool) < ADD:
        raise ValueError(f"Insufficient safe additions: {len(pool)} < {ADD}")
    rng = random.Random(45)
    add = [align_addition(record, cols) for record in rng.sample(pool, ADD)]
    out = train + add
    audit = validate_mix(out, val, excluded)

    DST.mkdir(parents=True, exist_ok=False)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import shutil
    shutil.copy(R21 / "val.jsonl", DST / "val.jsonl")

    summary, shares = describe_mix(out)
    total_h = summary["hours"]
    (DST / "mix.json").write_text(json.dumps({
        "derived_from": "r21_nothaiser + gigaspeech2_id 800 行（音色地道性优先，跑飞容忍）",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares, "added_id_rows": ADD,
        "train": summary, "metadata_audit": audit, "evaluation_sha256": eval_hashes,
        "requested_share_note": "Incremental row counts, not a fixed duration-share recipe.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总计 {len(out)} 行 / {total_h:.2f}h")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")


if __name__ == "__main__":
    main()
