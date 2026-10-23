#!/usr/bin/env python3
"""构建 r25：r18 清单 + 把 r23 的中性前缀移植到其中重合的 233 行（唯一变量）。

依据（阶段 A 审计 + r24 结果，见 docs/finetune_playbook.md）：
- r18（冷启动 1390 步、id 份额 47.5%、**0 条前缀行**）在同一份 v2 集上三语种 CER 与尾静音
  全面优于 r23（id 0.0157 vs 0.0177、th 0.0834 vs 0.0877、tl 0.0128 vs 0.0135；
  id/th/tl p90 尾静音 +0.14/+0.14/+0.12 vs r23 的 +0.32/…），且 id 探针 0.0013 为历代最佳；
- r24 证明 **Δsim(emo−plain) 的收益来自数据里的中性前缀行、不来自续训链**（−0.0129 在冷启动下保留）；
- 阶段 A 把"长目标 / 低 rate 错配 / 前缀行时长偏斜"三个靶子全部否掉 ⇒ r19 起的失稳不能归给数据形状。
⇒ 待验证的单变量假设：**把中性前缀接种到 r18 这份干净配方上，能否同时保住 r18 的稳定性与 r23 的情绪音色收益。**

做法（严格移植，不重新抽样）：
- 以 r18 的 train.jsonl 为底，逐行原样输出；
- 只有「r23 里带前缀、且 `audio` 路径与 r18 某行完全相同」的行（实测 233 条）改成 r23 的
  text/control/control_inoculation；其余字段与其余行**逐字段不变**；
- val.jsonl 逐字节拷贝 r18 的（保证 val loss 与 r18 可比、不引入新划分）；
- mix.json 按 r20 的约定重算 language_shares（行数与时长不变，故与 r18 的份额一致），并记移植来源。

守卫（任一不满足即失败退出，不写半个产物）：
- 目标目录已存在 → 拒绝覆盖；
- 行数、audio 路径序列、非移植行的全部字段必须与 r18 完全一致；
- 移植行的裸文本必须严格等于 r23 文本剥掉前缀后的结果（防止把前缀接到另一句上）；
- val.jsonl 的 SHA 必须与 r18 的相同。
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT_R18 = Path("/root/autodl-tmp/voxft_data/processed/r18_idaccent")
DEFAULT_R23 = Path("/root/autodl-tmp/voxft_data/processed/r23_thtlboost")
DEFAULT_DST = Path("/root/autodl-tmp/voxft_data/processed/r25_r18_prefix_graft")
DEFAULT_EVAL_CASES = (
    "/root/voxcpm-finetune/eval_cases/omnivoice_prod_v2.jsonl",
    "/root/voxcpm-finetune/eval_cases/drama_chain_prod_20261021.jsonl",
    "/root/voxcpm-finetune/eval_cases/id_prosody_probe_20261020.jsonl",
)
PREFIX_FIELDS = ("text", "control", "control_inoculation")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_prefix(text: str) -> str:
    t = text.lstrip()
    return t[t.index(")") + 1:] if t.startswith("(") and ")" in t else t


def norm(text: str) -> str:
    return "".join((text or "").split()).lower()


def main(src_r18: Path = DEFAULT_R18, src_r23: Path = DEFAULT_R23, dst: Path = DEFAULT_DST,
         eval_cases=DEFAULT_EVAL_CASES) -> int:
    SRC_R18, SRC_R23, DST = Path(src_r18), Path(src_r23), Path(dst)
    if DST.exists():
        print(f"拒绝覆盖已存在的目标目录：{DST}", file=sys.stderr)
        return 1

    r18 = load(SRC_R18 / "train.jsonl")
    r23 = load(SRC_R23 / "train.jsonl")
    print(f"r18 {len(r18)} 行 / r23 {len(r23)} 行")

    prefixed = [r for r in r23 if (r.get("text") or "").lstrip().startswith("(")]
    by_audio: dict[str, dict] = {}
    for r in prefixed:
        by_audio.setdefault(r["audio"], r)
    index18 = {r["audio"]: i for i, r in enumerate(r18)}
    graft = [(index18[a], src) for a, src in by_audio.items() if a in index18]
    graft.sort()
    print(f"r23 前缀行 {len(prefixed)} 条，其中 audio 路径命中 r18 的 {len(graft)} 条")
    if not graft:
        print("没有可移植的行，停止（不产出空清单）", file=sys.stderr)
        return 1

    # 守卫：移植行的裸文本必须与 r18 原文严格一致
    bad = []
    for i, src in graft:
        if strip_prefix(src["text"]) != r18[i]["text"]:
            bad.append((r18[i]["audio"], r18[i]["text"][:40], strip_prefix(src["text"])[:40]))
    if bad:
        print(f"拒绝：{len(bad)} 行的裸文本与 r18 不一致，前缀会接到另一句上。样例：{bad[:3]}", file=sys.stderr)
        return 1

    out = [dict(r) for r in r18]
    phrases = Counter()
    for i, src in graft:
        for field in PREFIX_FIELDS:
            if field in src:
                out[i][field] = src[field]
            else:
                out[i].pop(field, None)
        phrases[src.get("control") or "?"] += 1

    # 守卫：除移植行的三个字段外，一切不变
    assert len(out) == len(r18), "行数变了"
    assert [r["audio"] for r in out] == [r["audio"] for r in r18], "audio 序列变了"
    changed = 0
    for a, b in zip(r18, out):
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        if diff:
            assert diff <= set(PREFIX_FIELDS), f"意外改动了字段 {diff}"
            changed += 1
    assert changed == len(graft), f"改动行数 {changed} != 移植行数 {len(graft)}"

    # 评测文本暴露审计（只报告，不据此删行）
    eval_texts = set()
    for p in eval_cases:
        path = Path(p)
        if path.exists():
            eval_texts |= {norm(c.get("text")) for c in load(path)}
    exposed = [i for i, _ in graft if norm(strip_prefix(r18[i]["text"])) in eval_texts]
    r18_exposed = sum(1 for r in r18 if norm(r["text"]) in eval_texts)

    DST.mkdir(parents=True)
    with open(DST / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DST / "val.jsonl").write_bytes((SRC_R18 / "val.jsonl").read_bytes())
    assert sha256(DST / "val.jsonl") == sha256(SRC_R18 / "val.jsonl"), "val.jsonl 未逐字节拷贝"

    hours: dict[str, float] = {}
    counts: Counter = Counter()
    for r in out:
        hours[r["lang"]] = hours.get(r["lang"], 0.0) + r.get("duration", 0.0) / 3600
        counts[r["lang"]] += 1
    total_h = sum(hours.values())
    shares = {lg: {"actual": round(h / total_h, 4), "hours": round(h, 3), "rows": counts[lg]}
              for lg, h in sorted(hours.items())}
    graft_by_lang = Counter(r18[i]["lang"] for i, _ in graft)
    (DST / "mix.json").write_text(json.dumps({
        "derived_from": "r18_idaccent 清单逐行原样 + 移植 r23_thtlboost 的中性前缀到 audio 路径重合的 "
                        f"{len(graft)} 行（唯一变量；不增删行、不改音频、不重新抽样）",
        "total_rows": len(out), "total_hours": round(total_h, 3),
        "language_shares": shares,
        "neutral_inoculation": dict(graft_by_lang),
        "prefix_phrases": dict(phrases),
        "val_copied_byte_identical_from": str(SRC_R18 / "val.jsonl"),
        "eval_text_exposure": {
            "grafted_rows_matching_eval_text": len(exposed),
            "r18_rows_matching_eval_text": r18_exposed,
            "note": "移植不改变裸文本，故文本暴露与 r18 完全相同；前缀只会让精确匹配更少。",
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "src_r18_train_sha256": sha256(SRC_R18 / "train.jsonl"),
        "src_r23_train_sha256": sha256(SRC_R23 / "train.jsonl"),
        "dst_train_sha256": sha256(DST / "train.jsonl"),
        "val_sha256": sha256(DST / "val.jsonl"),
        "rows": len(out), "grafted_rows": len(graft),
        "graft_by_lang": dict(graft_by_lang), "prefix_phrases": dict(phrases),
    }
    (DST / "build_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"输出 {len(out)} 行 / {total_h:.3f}h → {DST}")
    print(f"移植 {len(graft)} 行，按语种 {dict(graft_by_lang)}，前缀种类 {dict(phrases)}")
    print(f"评测文本暴露：移植行 {len(exposed)} 条命中（r18 全体 {r18_exposed} 条命中）——移植不改裸文本，暴露与 r18 相同")
    print(f"train.jsonl SHA {audit['dst_train_sha256']}")
    for lg, s in shares.items():
        print(f"  {lg}: {s['rows']} 行 / {s['hours']}h / {s['actual'] * 100:.1f}%")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-r18", type=Path, default=DEFAULT_R18)
    ap.add_argument("--src-r23", type=Path, default=DEFAULT_R23)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--eval-cases", nargs="*", default=list(DEFAULT_EVAL_CASES))
    a = ap.parse_args()
    sys.exit(main(a.src_r18, a.src_r23, a.dst, a.eval_cases))
