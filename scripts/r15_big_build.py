# -*- coding: utf-8 -*-
"""r15_big：放大轮数据集——全部已核验库存源，不设 12h 预算上限。

配方 = r14 的源结构与语种权重（th 内部 fleurs:yodas:gs2:cv22:thai_ser=2:3:9:3:3，
id=fleurs:gs2:cv22=4:10:3 占 38.3，tl=fleurs，zh 10，en 5），但不传 train_hours：
mix_manifests 按 max_repeat=3 用尽各源可用时长（小语种缺口不重分配，language_shares
如实记录实际占比）。这是 r8（75h/2020 步）之后第一个全预算轮。
评测文本三重排除断言同前。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, mix_manifests
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED

ROOT = Path(__file__).resolve().parents[1]
OUT = "r15_big"
SEED = 42

ID, TL = 38.3, 28.3
TH = 28.3
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
    mix_manifests(PARTS, OUT, seed=SEED, train_hours=None, progress=print)
    train = _read_manifest(DATA_PROCESSED / OUT / "train.jsonl")
    leaked = [r for r in train if _norm(r["text"]) in excluded]
    if leaked:
        raise ValueError(f"混合后仍含评测文本 {len(leaked)} 行")
    mix = json.loads((DATA_PROCESSED / OUT / "mix.json").read_text(encoding="utf-8"))
    t = mix["train"]
    print(f"\n=== {OUT} ===")
    print(f"train {len(train)} 行 / {t['hours']}h；ref 行 {t['with_ref_audio']}；"
          f"带控制前缀 {t['with_control']}")
    for lang, s in sorted(mix["language_shares"].items()):
        print(f"  {lang}: 请求 {s['requested']:.1%} 实际 {s['actual']:.1%} ({s['hours']}h)")


if __name__ == "__main__":
    sys.exit(main())
