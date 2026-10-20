# -*- coding: utf-8 -*-
"""r16_big：全预算 + Atika88 印尼语源（在 r15_big 配方上加一个变量）。

r15_big 由用户决定于 2026-10-20 中止（不等验收，直接并入 Atika88 开大轮）。
唯一新增量 = id 内部加入 atika88_id 份额（fleurs:gs2:cv22:atika88 = 4:10:3:5）。
其余同 r15_big：id 40.3、th 内部 2:3:9:3:3（含 thai_ser）、tl 30.3、
zh/en 回放减到 6/3（证据：r13x 在 zh 5% × 3 epoch 下 zh CER 正常 0.067，
r13 的 zh 崩坏是 1 epoch 欠训而非回放减半），
不设 12h 上限（max_repeat=3 用尽各源）。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, mix_manifests
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED

ROOT = Path(__file__).resolve().parents[1]
OUT = "r16_big"
SEED = 42

ID, TL = 40.3, 30.3
TH = 30.3
PARTS = [
    ("r10_fleurs_th_eval_clean", TH * 2 / 20),
    ("r10_yodas_th_eval_clean", TH * 3 / 20),
    ("gigaspeech2_th", TH * 9 / 20),
    ("r10_cv22_th_pairs", TH * 3 / 20),
    ("thai_ser", TH * 3 / 20),
    ("r10_fleurs_id_eval_clean", ID * 4 / 22),
    ("gigaspeech2_id", ID * 10 / 22),
    ("r10_cv22_id_pairs", ID * 3 / 22),
    ("atika88_id", ID * 5 / 22),
    ("r10_fleurs_tl_eval_clean", TL),
    ("r10_fleurs_zh_eval_clean", 6.0),
    ("r10_fleurs_en_eval_clean", 3.0),
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
          f"控制前缀 {t['with_control']}")
    for lang, s in sorted(mix["language_shares"].items()):
        print(f"  {lang}: 请求 {s['requested']:.1%} 实际 {s['actual']:.1%} ({s['hours']}h)")


if __name__ == "__main__":
    sys.exit(main())
