# -*- coding: utf-8 -*-
"""refrob 阶段 2/3：构建 A 组训练混合（r12a_refrob_clean），B 组由 degrade_train_refs 派生。

设计（2026-10-20 用户确认，修订自 docs/ref_robustness_plan_20261012.md 的
「r8 同源配比」——r8 是五语种配比且 vi/ms 数据已删，物理上不可重建）：

- 配比：th/tl/id 各 28.3 分 + zh 10 + en 5（按有效音频时长，max_repeat=3，
  12h 训练预算，seed 42），语种内部沿用 r8/r10 的源结构：
  th = fleurs:yodas:gs2:cv22 = 2:3:9:3，id = fleurs:gs2:cv22 = 4:10:3，
  tl = fleurs（无其它已核验源），zh/en = fleurs。
- **ref 来源只有 cv22 可信配对**：已实测 r8 混合（joint_omni8）32307 行 **0 条 ref**，
  不配 cv22 的话 B 臂「50% ref 降质」就是空操作。cv22 只覆盖 th/id，
  tl 无已验证同人 ref（不为凑比例伪造身份）。
- 评测文本隔离：r10 的 *_eval_clean 别名已剔除 v2 全部 182 文本（118 是其子集，
  仍安全）；本脚本再剔 tl_short_holdout_20260918 与 fleurs_test_holdout_20260917
  （FLEURS 句库跨 split 平行，test 文本可能落在 train 里），并在混合后断言
  训练集不含任何评测文本。剔除产生的新别名以 r12_ 前缀落盘，不改旧目录。
- 只构建数据集，不改任何源音频；GPU 训练在后续步骤单独进行。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, _write_jsonl, mix_manifests
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED

ROOT = Path(__file__).resolve().parents[1]
OUT_A = "r12a_refrob_clean"
TRAIN_HOURS = 12
SEED = 42

# (源数据集, 权重)；th/id/tl 各合计 28.3，zh 10，en 5
PARTS = [
    ("r10_fleurs_th_eval_clean", 28.3 * 2 / 17),
    ("r10_yodas_th_eval_clean", 28.3 * 3 / 17),
    ("gigaspeech2_th", 28.3 * 9 / 17),
    ("r10_cv22_th_pairs", 28.3 * 3 / 17),
    ("r10_fleurs_id_eval_clean", 28.3 * 4 / 17),
    ("gigaspeech2_id", 28.3 * 10 / 17),
    ("r10_cv22_id_pairs", 28.3 * 3 / 17),
    ("r10_fleurs_tl_eval_clean", 28.3),
    ("r10_fleurs_zh_eval_clean", 10.0),
    ("r10_fleurs_en_eval_clean", 5.0),
]

EVAL_TEXT_FILES = [
    "omnivoice_prod_v2.jsonl",
    "tl_short_holdout_20260918.jsonl",
    "fleurs_test_holdout_20260917.jsonl",
]


def eval_texts() -> set[str]:
    texts = set()
    for name in EVAL_TEXT_FILES:
        path = ROOT / "eval_cases" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.add(_norm(json.loads(line)["text"]))
    return texts


def band_shares(rows):
    """3-8s 时长占比（短 cue 方向的记录口径；按有效时长）。"""
    total = sum(r["duration"] for r in rows)
    band = sum(r["duration"] for r in rows if 3 <= r["duration"] <= 8)
    return round(band / total, 4) if total else 0.0


def main():
    excluded = eval_texts()
    print(f"评测文本排除集：{len(excluded)} 条（{', '.join(EVAL_TEXT_FILES)}）")
    if (DATA_PROCESSED / OUT_A).exists():
        raise FileExistsError(f"{OUT_A} 已存在，不覆盖旧产物")

    parts, removals = [], {}
    for name, weight in PARTS:
        src = DATA_PROCESSED / name
        rows = _read_manifest(src / "train.jsonl")
        clean = [r for r in rows if _norm(r["text"]) not in excluded]
        removed = len(rows) - len(clean)
        removals[name] = removed
        if removed:
            alias = f"r12_{name}_clean"
            out = DATA_PROCESSED / alias
            if out.exists():
                raise FileExistsError(f"{alias} 已存在，不覆盖")
            _write_jsonl(clean, out / "train.jsonl")
            _write_jsonl(_read_manifest(src / "val.jsonl"), out / "val.jsonl")
            name = alias
        parts.append((name, weight))
    print("逐源剔除评测文本行数：", {k: v for k, v in removals.items() if v} or "全部 0")

    result = mix_manifests(parts, OUT_A, seed=SEED, train_hours=TRAIN_HOURS, progress=print)
    train = _read_manifest(DATA_PROCESSED / OUT_A / "train.jsonl")
    leaked = [r for r in train if _norm(r["text"]) in excluded]
    if leaked:
        raise ValueError(f"混合后仍含评测文本 {len(leaked)} 行：{leaked[0]['text'][:40]}")

    mix = json.loads((DATA_PROCESSED / OUT_A / "mix.json").read_text(encoding="utf-8"))
    by_lang_rows: dict[str, list] = {}
    for r in train:
        by_lang_rows.setdefault(r.get("lang", "unknown"), []).append(r)
    print(f"\n=== {OUT_A} ===")
    print(f"train {len(train)} 行 / {mix['train']['hours']}h；ref 行 "
          f"{mix['train']['with_ref_audio']}（跨语言 ref {mix['train']['cross_language_refs']}）")
    for lang, s in sorted(mix["language_shares"].items()):
        rows = by_lang_rows.get(lang, [])
        print(f"  {lang}: 请求 {s['requested']:.1%} 实际 {s['actual']:.1%} "
              f"({s['hours']}h)，3-8s 时长占比 {band_shares(rows):.1%}")
    (DATA_PROCESSED / OUT_A / "r12_build_notes.json").write_text(json.dumps({
        "excluded_text_files": EVAL_TEXT_FILES, "excluded_texts": len(excluded),
        "removed_rows_by_source": removals,
        "lang_3_8s_duration_share": {k: band_shares(v) for k, v in by_lang_rows.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("构建记录写入 r12_build_notes.json")
    return result


if __name__ == "__main__":
    sys.exit(main())
