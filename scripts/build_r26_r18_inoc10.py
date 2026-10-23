#!/usr/bin/env python3
"""构建 r26（= r25 的 B 臂）：r18 清单 + 按 r20 既有约定对 id/tl 做 10% 中性前缀接种。

为什么要有 B 臂（r25 的 dramaprod 结果，见 docs/finetune_playbook.md）：
- Δsim(emo−plain) 出现单调剂量-反应：前缀行 0（r18）−0.0190 → 233 行（r25）−0.0145 →
  373 行冷启动（r24）−0.0129 → 373 行 + 续训（r23）−0.0127；id 子项 −0.0246 → −0.0163 → −0.0085 → −0.0089。
  r25 的剂量只有 id 4.2% / tl 7.2%，只拿到约一半收益 ⇒ B 臂把剂量提到 r20 约定的 10%。
- r25 同时证明**前缀不是时长失稳源**（id p90 尾静音 0.46s，为微调轮最好，优于 r18 的 0.48 与 r23 的 1.02），
  所以加大剂量不需要担心把 r19 那套垫尾带进来——但这仍是要复测的门禁，不是已知结论。

与 build_r25_r18_prefix_graft.py 的区别：A 臂移植 r23 已选定的 233 行；B 臂按
build_r20_emofix.py 的原样逻辑（NEUTRAL_PHRASES 轮换、id 用 Random(43)、tl 用 Random(44)、
命中率 10%）在 r18 自己的行上重新接种 ⇒ 行数更多、剂量与 r23 相当。
两个脚本刻意不合并：A 臂那个已用于产出 r25 的训练数据、SHA 记进了文档，不能再改。

守卫（任一不满足即失败退出，不写半个产物）：目标目录已存在 → 拒绝；行数与 audio 序列不变；
只有被接种行的 text/control/control_inoculation 变化；val.jsonl 与 r18 逐字节相同（断言 SHA）。
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

DEFAULT_R18 = Path("/root/autodl-tmp/voxft_data/processed/r18_idaccent")
DEFAULT_DST = Path("/root/autodl-tmp/voxft_data/processed/r26_r18_inoc10")
DEFAULT_EVAL_CASES = (
    "/root/voxcpm-finetune/eval_cases/omnivoice_prod_v2.jsonl",
    "/root/voxcpm-finetune/eval_cases/drama_chain_prod_20261021.jsonl",
    "/root/voxcpm-finetune/eval_cases/id_prosody_probe_20261020.jsonl",
)
DEFAULT_A_ARM = Path("/root/autodl-tmp/voxft_data/processed/r25_r18_prefix_graft")

# 与 build_r20_emofix.py 逐字相同，保证"同一种接种"这件事可比
NEUTRAL_PHRASES = [
    "neutral, natural tone",
    "calm and steady",
    "speak naturally and clearly",
    "relaxed and natural",
]
ID_TL_INOC_RATE = 0.10
PREFIX_FIELDS = ("text", "control", "control_inoculation")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return "".join((text or "").split()).lower()


def main(src_r18: Path = DEFAULT_R18, dst: Path = DEFAULT_DST, eval_cases=DEFAULT_EVAL_CASES,
         a_arm: Path | None = DEFAULT_A_ARM) -> int:
    src_r18, dst = Path(src_r18), Path(dst)
    if dst.exists():
        print(f"拒绝覆盖已存在的目标目录：{dst}", file=sys.stderr)
        return 1

    rows = load(src_r18 / "train.jsonl")
    out = [dict(r) for r in rows]

    rng = {"id": random.Random(43), "tl": random.Random(44)}
    inoc = Counter()
    phrases = Counter()
    touched: list[int] = []
    for i, r in enumerate(out):
        lg = r.get("lang")
        if lg not in rng or (r.get("text") or "").startswith("("):
            continue
        if rng[lg].random() < ID_TL_INOC_RATE:
            phrase = NEUTRAL_PHRASES[inoc[lg] % len(NEUTRAL_PHRASES)]
            r["text"] = f"({phrase}){r['text']}"
            r["control"] = phrase
            r["control_inoculation"] = True
            inoc[lg] += 1
            phrases[phrase] += 1
            touched.append(i)

    if not touched:
        print("没有任何行被接种，停止（不产出与 r18 相同的清单冒充新轮）", file=sys.stderr)
        return 1

    # 守卫：除接种行的三个字段外，一切不变
    assert len(out) == len(rows), "行数变了"
    assert [r["audio"] for r in out] == [r["audio"] for r in rows], "audio 序列变了"
    changed = 0
    for a, b in zip(rows, out):
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if diff:
            assert diff <= set(PREFIX_FIELDS), f"意外改动了字段 {diff}"
            changed += 1
    assert changed == len(touched), f"改动行数 {changed} != 接种行数 {len(touched)}"

    # 与 A 臂的重叠（用于解释剂量-反应，不影响构建）
    overlap = None
    if a_arm and (Path(a_arm) / "train.jsonl").exists():
        a_rows = load(Path(a_arm) / "train.jsonl")
        a_pref = {r["audio"] for r in a_rows if (r.get("text") or "").startswith("(")}
        overlap = sum(1 for i in touched if out[i]["audio"] in a_pref)

    eval_texts = set()
    for p in eval_cases:
        path = Path(p)
        if path.exists():
            eval_texts |= {norm(c.get("text")) for c in load(path)}
    exposed_inoc = sum(1 for i in touched if norm(rows[i]["text"]) in eval_texts)
    exposed_all = sum(1 for r in rows if norm(r["text"]) in eval_texts)

    dst.mkdir(parents=True)
    with open(dst / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (dst / "val.jsonl").write_bytes((src_r18 / "val.jsonl").read_bytes())
    assert sha256(dst / "val.jsonl") == sha256(src_r18 / "val.jsonl"), "val.jsonl 未逐字节拷贝"

    hours: dict[str, float] = {}
    counts: Counter = Counter()
    for r in out:
        hours[r["lang"]] = hours.get(r["lang"], 0.0) + r.get("duration", 0.0) / 3600
        counts[r["lang"]] += 1
    total_h = sum(hours.values())
    shares = {lg: {"actual": round(h / total_h, 4), "hours": round(h, 3), "rows": counts[lg]}
              for lg, h in sorted(hours.items())}
    lang_rows = {lg: sum(1 for r in out if r.get("lang") == lg) for lg in ("id", "tl")}
    (dst / "mix.json").write_text(json.dumps({
        "derived_from": "r18_idaccent 清单逐行原样 + 按 build_r20_emofix.py 的约定对 id/tl 做 10% 中性前缀接种"
                        "（id Random(43)、tl Random(44)、NEUTRAL_PHRASES 轮换）；不增删行、不改音频",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares,
        "neutral_inoculation": dict(inoc),
        "inoc_rate_per_lang": {lg: round(inoc[lg] / lang_rows[lg], 4) for lg in lang_rows if lang_rows[lg]},
        "prefix_phrases": dict(phrases),
        "overlap_with_r25_graft": overlap,
        "val_copied_byte_identical_from": str(src_r18 / "val.jsonl"),
        "eval_text_exposure": {
            "inoculated_rows_matching_eval_text": exposed_inoc,
            "r18_rows_matching_eval_text": exposed_all,
            "note": "接种不改裸文本，故文本暴露与 r18 相同；前缀只会让精确匹配更少。",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (dst / "build_audit.json").write_text(json.dumps({
        "src_r18_train_sha256": sha256(src_r18 / "train.jsonl"),
        "dst_train_sha256": sha256(dst / "train.jsonl"),
        "val_sha256": sha256(dst / "val.jsonl"),
        "rows": len(out), "inoculated_rows": len(touched),
        "inoc_by_lang": dict(inoc), "prefix_phrases": dict(phrases),
        "overlap_with_r25_graft": overlap,
        "inoc_rate": ID_TL_INOC_RATE, "seeds": {"id": 43, "tl": 44},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"输出 {len(out)} 行 / {total_h:.3f}h → {dst}")
    print(f"接种 {len(touched)} 行：{dict(inoc)}（占该语种 "
          + ", ".join(f"{lg} {inoc[lg]}/{lang_rows[lg]}={inoc[lg]/lang_rows[lg]:.1%}" for lg in lang_rows if lang_rows[lg])
          + "）")
    print(f"短语分布 {dict(phrases)}；与 A 臂(r25) 移植行重叠 {overlap}")
    print(f"评测文本暴露：接种行 {exposed_inoc} 条命中（r18 全体 {exposed_all} 条命中）")
    print(f"train.jsonl SHA {sha256(dst / 'train.jsonl')}")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-r18", type=Path, default=DEFAULT_R18)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--eval-cases", nargs="*", default=list(DEFAULT_EVAL_CASES))
    ap.add_argument("--a-arm", type=Path, default=DEFAULT_A_ARM, help="A 臂清单目录，用于统计重叠；不需要时传空目录")
    a = ap.parse_args()
    sys.exit(main(a.src_r18, a.dst, a.eval_cases, a.a_arm))
