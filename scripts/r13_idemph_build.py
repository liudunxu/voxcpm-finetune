# -*- coding: utf-8 -*-
"""r13：id 加量轮数据集构建（单变量 = id 份额 28.3→38.3，从 zh/en 回放扣，th/tl 不动）。

依据（2026-10-20 用户确认）：id 的 MY 腔是**基座预训练自带**（id 母语者实听基座合成
样本反馈），不是 r8 的 ms 训练数据；杠杆 = 用更多地道 id 数据把韵律往 id 拉。
其余全部沿用 r12a 配方（源结构/预算 12h/seed 42/评测文本三重排除），便于与 r12a 配对。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, _write_jsonl, mix_manifests
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED

ROOT = Path(__file__).resolve().parents[1]
OUT = "r13_idemph"
TRAIN_HOURS = 12
SEED = 42

# r12a 结构，唯一变化：id 28.3→38.3，zh 10→5、en 5→2.5（回放削半，th/tl 不动）
ID, TH, TL, ZH, EN = 38.3, 28.3, 28.3, 5.0, 2.5
PARTS = [
    ("r10_fleurs_th_eval_clean", TH * 2 / 17),
    ("r10_yodas_th_eval_clean", TH * 3 / 17),
    ("gigaspeech2_th", TH * 9 / 17),
    ("r10_cv22_th_pairs", TH * 3 / 17),
    ("r10_fleurs_id_eval_clean", ID * 4 / 17),
    ("gigaspeech2_id", ID * 10 / 17),
    ("r10_cv22_id_pairs", ID * 3 / 17),
    ("r10_fleurs_tl_eval_clean", TL),
    ("r10_fleurs_zh_eval_clean", ZH),
    ("r10_fleurs_en_eval_clean", EN),
]

EVAL_TEXT_FILES = [
    "omnivoice_prod_v2.jsonl",
    "tl_short_holdout_20260918.jsonl",
    "fleurs_test_holdout_20260917.jsonl",
]


def main():
    excluded = set()
    for name in EVAL_TEXT_FILES:
        for line in (ROOT / "eval_cases" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                excluded.add(_norm(json.loads(line)["text"]))
    if (DATA_PROCESSED / OUT).exists():
        raise FileExistsError(f"{OUT} 已存在，不覆盖旧产物")
    result = mix_manifests(PARTS, OUT, seed=SEED, train_hours=TRAIN_HOURS, progress=print)
    train = _read_manifest(DATA_PROCESSED / OUT / "train.jsonl")
    leaked = [r for r in train if _norm(r["text"]) in excluded]
    if leaked:
        raise ValueError(f"混合后仍含评测文本 {len(leaked)} 行")
    mix = json.loads((DATA_PROCESSED / OUT / "mix.json").read_text(encoding="utf-8"))
    print(f"\n=== {OUT} ===")
    print(f"train {len(train)} 行 / {mix['train']['hours']}h；ref 行 {mix['train']['with_ref_audio']}")
    for lang, s in sorted(mix["language_shares"].items()):
        print(f"  {lang}: 请求 {s['requested']:.1%} 实际 {s['actual']:.1%} ({s['hours']}h)")
    return result


if __name__ == "__main__":
    sys.exit(main())
