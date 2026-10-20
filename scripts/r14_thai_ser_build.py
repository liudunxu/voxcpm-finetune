# -*- coding: utf-8 -*-
"""r14：thai_ser 泰语情感语料入混（基于当前冠军 r13x 配方）。

单变量 = th 内部加入 thai_ser 份额（fleurs:yodas:gs2:cv22:thai_ser = 2:3:9:3:3）。
修复性变更（非假设变量）：zh/en 回放恢复到 10/5（r13 砍半致 zh CER 0.234 的教训，
zh/en 不作门禁但与三语种归因无关）；沿用 r13x 已验证的 3 epoch 与 id 38.3% 加量。
评测文本三重排除同前，构建后断言零泄漏。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, mix_manifests
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED

ROOT = Path(__file__).resolve().parents[1]
OUT = "r14_thai_ser"
TRAIN_HOURS = 12
SEED = 42

ID, TL = 38.3, 28.3
TH = 28.3  # th 内部 5 源共 20 份
PARTS = [
    ("r10_fleurs_th_eval_clean", TH * 2 / 20),
    ("r10_yodas_th_eval_clean", TH * 3 / 20),
    ("gigaspeech2_th", TH * 9 / 20),
    ("r10_cv22_th_pairs", TH * 3 / 20),
    ("thai_ser", TH * 3 / 20),
    ("r10_fleurs_id_eval_clean", ID * 4 / 17),
    ("gigaspeech2_id", ID * 10 / 17),
    ("r10_cv22_id_pairs", ID * 3 / 17),
    ("r10_fleurs_tl_eval_clean", TL),
    ("r10_fleurs_zh_eval_clean", 10.0),
    ("r10_fleurs_en_eval_clean", 5.0),
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
    mix_manifests(PARTS, OUT, seed=SEED, train_hours=TRAIN_HOURS, progress=print)
    train = _read_manifest(DATA_PROCESSED / OUT / "train.jsonl")
    leaked = [r for r in train if _norm(r["text"]) in excluded]
    if leaked:
        raise ValueError(f"混合后仍含评测文本 {len(leaked)} 行")
    mix = json.loads((DATA_PROCESSED / OUT / "mix.json").read_text(encoding="utf-8"))
    print(f"\n=== {OUT} ===")
    t = mix["train"]
    print(f"train {len(train)} 行 / {t['hours']}h；ref 行 {t['with_ref_audio']}；"
          f"带控制前缀 {t['with_control']}；情绪分布 {t['emotions']}")
    for lang, s in sorted(mix["language_shares"].items()):
        print(f"  {lang}: 请求 {s['requested']:.1%} 实际 {s['actual']:.1%} ({s['hours']}h)")


if __name__ == "__main__":
    sys.exit(main())
