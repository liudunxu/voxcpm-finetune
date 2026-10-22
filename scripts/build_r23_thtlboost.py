#!/usr/bin/env python3
"""构建 r23 续训混合：r22_idboost 底 + th/tl 增量（音色地道性优先口径）。

- th +800：gigaspeech2_th 500（自然口语锚点）+ cv22_th 300（朗读补充）
- tl +500：fleurs_tl 剩余未用行（tl 唯一已加工源；fleurs 行长，800 行会把 tl 时长份额推过 id，裁到 500）
- 排除已用目标音频/原音频、评测文本、验证音频/身份/session；保持 3-20s
- 新行对齐 r22 train 列集，保留原 source_id 与可信 ref，报告实际覆盖率
- 已存在输出拒绝覆盖；历史 r23 清单不重新构建
- 确定性抽样 seed=46
"""
import json
import random
import shutil
from pathlib import Path

from incremental_mix import align_addition, describe_mix, eligible_additions, load_exclusions, validate_mix

R22 = Path("/root/autodl-tmp/voxft_data/processed/r22_idboost")
PROC = Path("/root/autodl-tmp/voxft_data/processed")
DST = Path("/root/autodl-tmp/voxft_data/processed/r23_thtlboost")
PLAN = [("gigaspeech2_th", 500), ("cv22_th", 300), ("fleurs_tl", 500)]


def load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"Frozen output already exists: {DST}")
    train = load(R22 / "train.jsonl")
    val = load(R22 / "val.jsonl")
    excluded, eval_hashes = load_exclusions()
    validate_mix(train, val, excluded)
    cols = set()
    for r in train:
        cols.update(r.keys())

    rng = random.Random(46)
    add_all = []
    for src, want in PLAN:
        pool, rejected = eligible_additions(load(PROC / src / "train.jsonl"),
                                           train + add_all, val, excluded)
        if len(pool) < want:
            raise ValueError(f"{src}: insufficient safe additions {len(pool)} < {want}")
        take = rng.sample(pool, want)
        print(f"{src}: 可用 {len(pool)} 行，取 {len(take)} 行")
        print(f"排除 {rejected}")
        add_all.extend(align_addition(record, cols) for record in take)
    out = train + add_all
    audit = validate_mix(out, val, excluded)

    DST.mkdir(parents=True, exist_ok=False)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(R22 / "val.jsonl", DST / "val.jsonl")

    summary, shares = describe_mix(out)
    total_h = summary["hours"]
    (DST / "mix.json").write_text(json.dumps({
        "derived_from": "r22_idboost + th 800 行（gigaspeech2_th 500 + cv22_th 300）+ tl fleurs_tl 500 行",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares,
        "train": summary, "metadata_audit": audit, "evaluation_sha256": eval_hashes,
        "requested_share_note": "Incremental row counts, not a fixed duration-share recipe.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"总计 {len(out)} 行 / {total_h:.2f}h")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")


if __name__ == "__main__":
    main()
