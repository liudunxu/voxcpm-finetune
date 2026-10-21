# -*- coding: utf-8 -*-
"""r19：情绪前缀接种轮（r13x 配方 + thai_ser 情绪数据，单变量 = 控制前缀训练信号）。

依据（2026-10-21 drama_chain 三模型对齐验收，docs 见 execution_log）：
- 生产链路 = 中文 ref + 自动情绪 prompt + 短 cue；用户优先级：口音 > 无 badcase > 音色一致 > 情绪
- r13x 在 zh-ref 链路全面优于 r18 且无跑飞；但两个 LoRA 训练 control 前缀均为 0%，
  情绪跟随完全靠基座残留；情绪前缀导致的音色漂移 base −0.048，r13x −0.021
- thai_ser 是唯一可信情绪标签源（CC-BY-SA，含 SA 红线：r19 不对外分发，仅 API 交付）
- r14 已证 thai_ser 3/20 份额对 CER 无可测增量——本轮目标不是 CER，是情绪前缀覆盖

配方 = r13_idemph 原样（ID 38.3 / TH 28.3 / TL 28.3 / ZH 5 / EN 2.5，12h，seed 42）
+ thai_ser 8 份（~0.94h，~5 情绪类 × 可信标签，经管线 apply_control_prefixes 自动加前缀，
  朗读源无 verified 标签不加——覆盖率守纪律，不伪造标签）。
总量 ~12.9h，3ep ≈ 1180 步，守 r13x 甜点位。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, _write_jsonl, mix_manifests
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED

ROOT = Path(__file__).resolve().parents[1]
OUT = "r19_emotion"
TRAIN_HOURS = 12.94   # r13 的 12h × 110.4/102.4 份，保持非 thai_ser 各源绝对时长不变
SEED = 42

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
    # thai_ser 按 speaker 分 4 片并发加工的产物（全局 train/val 不串，合计视为一个源）
    ("thai_ser_s0", 2.0),
    ("thai_ser_s1", 2.0),
    ("thai_ser_s2", 2.0),
    ("thai_ser_s3", 2.0),
]

EVAL_TEXT_FILES = [
    "omnivoice_prod_v2.jsonl",
    "tl_short_holdout_20260918.jsonl",
    "fleurs_test_holdout_20260917.jsonl",
    "id_prosody_probe_20261020.jsonl",
    "drama_chain_20261021.jsonl",
]


def main():
    excluded = set()
    for name in EVAL_TEXT_FILES:
        path = ROOT / "eval_cases" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                excluded.add(_norm(json.loads(line)["text"]))
    if (DATA_PROCESSED / OUT).exists():
        raise FileExistsError(f"{OUT} 已存在，不覆盖旧产物")
    result = mix_manifests(PARTS, OUT, seed=SEED, train_hours=TRAIN_HOURS, progress=print)
    train = _read_manifest(DATA_PROCESSED / OUT / "train.jsonl")
    leaked = [r for r in train if _norm(r["text"]) in excluded]
    if leaked:
        raise ValueError(f"混合后仍含评测文本 {len(leaked)} 行")
    n_control = sum(1 for r in train if r.get("control"))
    mix = json.loads((DATA_PROCESSED / OUT / "mix.json").read_text(encoding="utf-8"))
    print(f"\n=== {OUT} ===")
    print(f"train {len(train)} 行 / {mix['train']['hours']}h；ref 行 {mix['train']['with_ref_audio']}；"
          f"带控制前缀 {n_control} 行（{n_control / len(train):.1%}）")
    for lang, s in sorted(mix["language_shares"].items()):
        print(f"  {lang}: 请求 {s['requested']:.1%} 实际 {s['actual']:.1%} ({s['hours']}h)")
    return result


if __name__ == "__main__":
    sys.exit(main())
