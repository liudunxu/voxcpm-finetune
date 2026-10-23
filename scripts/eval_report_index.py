#!/usr/bin/env python3
"""扫描评测报告目录，按「模型 × 验收集」分类并输出矩阵与索引。

为什么要有这个：报告文件名是 `<label>_<uuid>.json`，uuid 与内容无关，
一个模型在多套 case 集上各有一份、还有 `--shard` 的分片与 `--merge` 的合并件，
人工靠文件名根本认不出来（2026-10-23 一天内因此认错三次：把 base 的 drama_chain
当成 drama_chain_prod、把 r18 的 336 条当成 dramaprod）。分类只信报告**内容**：
case_id 前缀 + 语种集合 + 条数 + seed 组，不看文件名。

只读，不改任何报告。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# case_id 前缀 → 验收集名。顺序有意义：先匹配更具体的前缀。
SET_RULES = (
    ("drama_chain_prod", "drama_prod_"),
    ("drama_chain", "drama_"),
    ("id_prosody_probe", "id_ms_"),
    ("runaway_regression", "runaway_"),
    ("fleurs_holdout", "fleurs_test_"),
    ("refrob", "refrob_"),
    ("acoustic_feedback", "acoustic-"),
)
DEFAULT_SET = "omnivoice_prod_v2"


def classify(case_ids) -> str:
    for name, prefix in SET_RULES:
        if any(str(c).startswith(prefix) for c in case_ids):
            return name
    return DEFAULT_SET


def inspect(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        return None
    case_ids = [i.get("case_id") for i in items]
    shard = "shard" in path.name
    return {
        "path": str(path),
        "name": path.name,
        "label": doc.get("label") or "?",
        "target": doc.get("target"),
        "set": classify(case_ids),
        "items": len(items),
        "cases": len(set(case_ids)),
        "seeds": sorted({i.get("seed") for i in items if i.get("seed") is not None}),
        "langs": sorted({i.get("lang") for i in items if i.get("lang")}),
        "cfg": doc.get("cfg_value"),
        "steps": doc.get("inference_timesteps"),
        "strength": doc.get("lora_strength"),
        "shard": shard,
        "merged_from": len(doc.get("merged_from") or []) if not shard else 0,
    }


def main(directory=Path("work/rescore"), show_shards: bool = False) -> int:
    directory = Path(directory)
    if not directory.is_dir():
        print(f"目录不存在：{directory}", file=sys.stderr)
        return 1
    rows = [r for r in (inspect(p) for p in sorted(directory.glob("*.json"))) if r]
    if not rows:
        print(f"{directory} 下没有可识别的评测报告", file=sys.stderr)
        return 1
    merged = [r for r in rows if not r["shard"]]
    shards = [r for r in rows if r["shard"]]

    sets = sorted({r["set"] for r in merged})
    labels = sorted({r["label"] for r in merged})
    print(f"# 评测报告索引：{directory}（合并件 {len(merged)} 份，分片 {len(shards)} 份）\n")
    print("分类只依据报告内容（case_id 前缀 / 语种 / 条数 / seed），不依据文件名。\n")
    print("| 模型 label | " + " | ".join(sets) + " |")
    print("|---|" + "---|" * len(sets))
    for label in labels:
        cells = []
        for s in sets:
            hit = [r for r in merged if r["label"] == label and r["set"] == s]
            if not hit:
                cells.append("—")
                continue
            r = max(hit, key=lambda x: Path(x["path"]).stat().st_mtime)
            cells.append(f"{r['items']}条/{len(r['seeds'])}seed `{Path(r['path']).name[:28]}…`"
                         + (f"（另有 {len(hit) - 1} 份）" if len(hit) > 1 else ""))
        print(f"| `{label}` | " + " | ".join(cells) + " |")

    print("\n## 逐份明细（合并件）\n")
    print("| 文件 | label | 验收集 | items | cases | seeds | 语种 | cfg/steps |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(merged, key=lambda x: (x["set"], x["label"], x["name"])):
        print(f"| `{r['name']}` | {r['label']} | {r['set']} | {r['items']} | {r['cases']} | "
              f"{','.join(str(s) for s in r['seeds'])} | {'/'.join(r['langs'])} | "
              f"{r['cfg']}/{r['steps']}" + (f" str{r['strength']}" if r["strength"] not in (None, 1.0) else "") + " |")

    odd = [r for r in merged if r["cfg"] != 1.8 or r["steps"] != 20]
    if odd:
        print("\n⚠️ 口径不是 cfg1.8/20步 的报告（不能与生产口径横比）：")
        for r in odd:
            print(f"  - `{r['name']}` cfg={r['cfg']} steps={r['steps']}")
    dupes = [k for k, n in Counter((r["label"], r["set"]) for r in merged).items() if n > 1]
    if dupes:
        print("\n⚠️ 同一 (label, 验收集) 有多份合并报告（引用前必须确认用哪一份）：")
        for label, s in sorted(dupes):
            names = [r["name"] for r in merged if r["label"] == label and r["set"] == s]
            print(f"  - {label} / {s}: {names}")
    if show_shards and shards:
        print(f"\n分片报告 {len(shards)} 份（未合并，一般不直接引用）：")
        for r in shards:
            print(f"  - `{r['name']}` {r['label']} / {r['set']} / {r['items']}条")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", type=Path, default=Path("work/rescore"))
    ap.add_argument("--show-shards", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.directory, a.show_shards))
