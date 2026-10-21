#!/usr/bin/env python3
"""修复 r21_nothaiser 的 train/val 列不一致（HF datasets 要求两文件 schema 完全相同）。

train 比 val 多 control_inoculation（接种标记），val 比 train 多 9 个 thai_ser 专有列
（r20 把 thai_ser 行留在 train 里所以两边列集相同能过；r21 从 train 删掉 thai_ser 后
schema 就错开了）。修法：以 train 的列集为准，val 删多余列、补缺失列为 None。
"""
import json
from pathlib import Path

DST = Path("/root/autodl-tmp/voxft_data/processed/r21_nothaiser")


def load(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    train = load(DST / "train.jsonl")
    val = load(DST / "val.jsonl")
    cols = set()
    for r in train:
        cols.update(r.keys())
    fixed = 0
    for r in val:
        extra = set(r.keys()) - cols
        for k in extra:
            del r[k]
        missing = cols - set(r.keys())
        for k in missing:
            r[k] = None
        if extra or missing:
            fixed += 1
    with open(DST / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"val.jsonl 修正 {fixed}/{len(val)} 行，列数对齐 train 的 {len(cols)} 列")


if __name__ == "__main__":
    main()
