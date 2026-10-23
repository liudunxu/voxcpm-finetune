#!/usr/bin/env python3
"""构建 r27：把 r26 的配方按 (语种, 源) 等比放大到 ~25h，3 epoch ⇒ 步数随之放大。

为什么是"等比放大"而不是随便加料：r23/r25/r26 三轮的数据量与步数几乎相同（15–16.5h、1094–2909 步），
轴最优却分属三轮 ⇒ 构成这条前沿已经画完，继续调前缀剂量只是沿前沿移动。要换维度就只能动**规模**。
等比放大保证每个 (语种, 源) 的相对份额与 r26 完全一致，于是唯一变量是总量（和随之而来的步数），
不会顺手把源构成也改了——否则又会变成 r19 那种一次改四件事的组合拳。

做法：
- 继承 r26 的 train.jsonl **逐行原样**（7412 行，含 502 行已接种的中性前缀）；
- 对每个 (lang, source_id)，按 `base_hours × (scale − 1)` 计算目标新增时长，从对应 processed 源池里
  用 `incremental_mix.eligible_additions` 过滤后**确定性**抽取（按 audio 路径排序，不用随机数，
  以便复现与审计）；某源不够就照实少加、**不把缺口重分配给其它语种或源**；
- 新增的 id/tl 行按同一套 NEUTRAL_PHRASES 轮换做 10% 接种（独立 RNG，与 r26 的 43/44 区分），
  使前缀占该语种的比例不被稀释；
- val.jsonl **逐字节拷贝 r26 的**（三轮共用同一验证集，val loss 才可比、也不引入新划分）；
- 出参前跑 `incremental_mix.validate_mix`：任何违规或原音频曝光 >3× 直接抛错，不写产物。

只读既有清单与源池，不改任何旧产物；目标目录已存在则拒绝覆盖。
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from incremental_mix import (  # noqa: E402
    audit_mix, describe_mix, eligible_additions, load_exclusions)

DEFAULT_BASE = Path("/root/autodl-tmp/voxft_data/processed/r26_r18_inoc10")
DEFAULT_SOURCES = Path("/root/autodl-tmp/voxft_data/processed")
DEFAULT_DST = Path("/root/autodl-tmp/voxft_data/processed/r27_scale25h")
NEUTRAL_PHRASES = [
    "neutral, natural tone",
    "calm and steady",
    "speak naturally and clearly",
    "relaxed and natural",
]
INOC_RATE = 0.10
INOC_SEEDS = {"id": 45, "tl": 46}  # 与 r26 的 43/44 区分，避免误认为是同一批行
PREFIX_FIELDS = ("text", "control", "control_inoculation")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hours(rows) -> float:
    return sum(r.get("duration") or 0.0 for r in rows) / 3600


def main(base_dir=DEFAULT_BASE, sources_dir=DEFAULT_SOURCES, dst=DEFAULT_DST,
         target_hours: float = 25.0, excluded=None, dry_run: bool = False,
         max_dur: float = 30.0) -> int:
    base_dir, sources_dir, dst = Path(base_dir), Path(sources_dir), Path(dst)
    if dst.exists() and not dry_run:
        print(f"拒绝覆盖已存在的目标目录：{dst}", file=sys.stderr)
        return 1

    base = load(base_dir / "train.jsonl")
    val = load(base_dir / "val.jsonl")
    if excluded is None:
        excluded, _ = load_exclusions()
    base_h = hours(base)
    scale = target_hours / base_h
    print(f"继承 {len(base)} 行 / {base_h:.3f}h；目标 {target_hours}h ⇒ scale={scale:.4f}")

    # 每个 (lang, source) 的既有时长与目标新增时长
    per_src = defaultdict(list)
    for r in base:
        per_src[(r.get("lang"), r.get("source_id"))].append(r)
    targets = {k: hours(v) * (scale - 1) for k, v in per_src.items()}

    added, rejected_all, exhausted = [], Counter(), []
    for (lang, src), need_h in sorted(targets.items()):
        if need_h <= 0 or not src:
            continue
        pool_path = sources_dir / src / "train.jsonl"
        if not pool_path.exists():
            exhausted.append(f"{lang}/{src}: 源池缺失 {pool_path}")
            continue
        pool = load(pool_path)
        kept, rejected = eligible_additions(pool, base + added, val, excluded, max_dur)
        rejected_all.update(rejected)
        kept.sort(key=lambda r: str(r.get("audio")))  # 确定性抽取，不依赖随机数
        picked, got_h = [], 0.0
        for row in kept:
            if got_h >= need_h:
                break
            picked.append(row)
            got_h += (row.get("duration") or 0.0) / 3600
        added.extend(picked)
        flag = "" if got_h >= need_h * 0.999 else "  ← 源池不足，照实少加（不重分配）"
        if flag:
            exhausted.append(f"{lang}/{src}: 需要 {need_h:.3f}h，只有 {got_h:.3f}h 合格")
        print(f"  {lang:<3}{src:<22} 既有 {hours(per_src[(lang, src)]):6.3f}h → 新增 {got_h:6.3f}h "
              f"({len(picked):>4} 行 / 候选合格 {len(kept):>5}){flag}")

    # 新增 id/tl 行按 10% 接种，保持前缀比例不被稀释
    rng = {lg: random.Random(sd) for lg, sd in INOC_SEEDS.items()}
    inoc, phrases = Counter(), Counter()
    for lg in ("id", "tl"):
        idx = [i for i, r in enumerate(added) if r.get("lang") == lg]
        for i in idx:
            if rng[lg].random() < INOC_RATE:
                phrase = NEUTRAL_PHRASES[inoc[lg] % len(NEUTRAL_PHRASES)]
                added[i]["text"] = f"({phrase}){added[i]['text']}"
                added[i]["control"] = phrase
                added[i]["control_inoculation"] = True
                inoc[lg] += 1
                phrases[phrase] += 1

    out = base + added
    # 守卫：继承部分必须逐字段不变
    assert len(out) == len(base) + len(added), "行数不符"
    for a, b in zip(base, out):
        assert a == b, "继承行被改动了"
    for r in added:
        assert {k for k in r if k in PREFIX_FIELDS} <= set(PREFIX_FIELDS)

    pref = Counter(r["lang"] for r in out if (r.get("text") or "").startswith("("))
    tot = Counter(r["lang"] for r in out)
    audit = audit_mix(out, val, excluded)

    # 泄漏与曝光是硬门禁；时长门只对新增行强制（见下）
    if audit["evaluation_text_rows"]:
        print(f"拒绝：训练清单命中评测文本 {audit['evaluation_text_rows']} 条", file=sys.stderr)
        return 1
    base_audit = audit_mix(base, val, excluded)
    if audit["validation_text_rows"] > base_audit["validation_text_rows"]:
        print(f"拒绝：新增行带来 {audit['validation_text_rows'] - base_audit['validation_text_rows']} 条"
              f"验证集文本命中（继承清单原有 {base_audit['validation_text_rows']} 条）", file=sys.stderr)
        return 1
    if audit["origins_above_3x"]:
        print(f"拒绝：原音频曝光 >3× 共 {len(audit['origins_above_3x'])} 个", file=sys.stderr)
        return 1

    def out_of_range(rows, upper):
        bad = []
        for r in rows:
            try:
                d = float(r.get("duration"))
            except (TypeError, ValueError):
                bad.append(r)
                continue
            if not (3 <= d <= upper):
                bad.append(r)
        return bad

    inherited_bad = out_of_range(base, 20.0)
    added_bad = out_of_range(added, max_dur)
    if added_bad:
        print(f"拒绝：{len(added_bad)} 条新增行时长不在 3–{max_dur:g}s 内", file=sys.stderr)
        return 1

    def dist(rows):
        d = sorted(float(r.get("duration") or 0) for r in rows)
        if not d:
            return "n=0"
        q = lambda p: d[min(len(d) - 1, int(p * len(d)))]
        return (f"n={len(d)} p50={q(.5):.2f} p90={q(.9):.2f} max={d[-1]:.2f} "
                f">15s={sum(1 for x in d if x > 15) / len(d):.1%} >20s={sum(1 for x in d if x > 20) / len(d):.1%}")

    print(f"\n时长分布对照（新增行的上限 = {max_dur:g}s）：")
    print(f"  继承 {len(base)} 行 : {dist(base)}")
    print(f"  新增 {len(added)} 行 : {dist(added)}")
    print(f"  合计 {len(out)} 行 : {dist(out)}")
    # 继承行沿用 r18 时代的清单（20s 上限是 r19 才引入的），本轮唯一变量是规模，
    # 不重新过滤继承行——但必须如实报告，不能假装全清单合规。
    try:
        summary = describe_mix(out)
    except Exception as exc:  # 摘要只用于打印，不该让构建失败
        summary = {"error": str(exc)}

    if dry_run:
        print("\n[dry-run] 不写产物")
    else:
        dst.mkdir(parents=True)
        with open(dst / "train.jsonl", "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (dst / "val.jsonl").write_bytes((base_dir / "val.jsonl").read_bytes())
        assert sha256(dst / "val.jsonl") == sha256(base_dir / "val.jsonl"), "val.jsonl 未逐字节拷贝"
        h = defaultdict(float)
        c = Counter()
        for r in out:
            h[r["lang"]] += (r.get("duration") or 0.0) / 3600
            c[r["lang"]] += 1
        total_h = sum(h.values())
        (dst / "mix.json").write_text(json.dumps({
            "derived_from": f"r26_r18_inoc10 逐行原样 + 按 (语种,源) 等比放大到 {target_hours}h"
                            f"（scale={scale:.4f}）；新增 id/tl 行按 10% 接种同一组中性前缀",
            "total_rows": len(out), "total_hours": round(total_h, 3),
            "inherited_rows": len(base), "added_rows": len(added),
            "language_shares": {lg: {"actual": round(h[lg] / total_h, 4), "hours": round(h[lg], 3), "rows": c[lg]}
                                for lg in sorted(h)},
            "neutral_inoculation_new_rows": dict(inoc),
            "prefix_rows_total": dict(pref),
            "prefix_ratio_by_lang": {lg: round(pref[lg] / c[lg], 4) for lg in ("id", "tl") if c[lg]},
            "prefix_phrases_new": dict(phrases),
            "shortfalls": exhausted,
            "val_copied_byte_identical_from": str(base_dir / "val.jsonl"),
            "audit": {k: audit[k] for k in ("rows", "validation_rows", "violation_counts",
                                            "evaluation_text_rows", "validation_text_rows", "origins_above_3x")},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (dst / "build_audit.json").write_text(json.dumps({
            "base_train_sha256": sha256(base_dir / "train.jsonl"),
            "dst_train_sha256": sha256(dst / "train.jsonl"),
            "val_sha256": sha256(dst / "val.jsonl"),
            "rows": len(out), "inherited": len(base), "added": len(added),
            "added_by_lang": dict(Counter(r["lang"] for r in added)),
            "added_by_source": dict(Counter(r.get("source_id") for r in added)),
            "inoculated_new_rows": dict(inoc), "prefix_phrases_new": dict(phrases),
            "rejected_candidate_reasons": dict(rejected_all),
            "shortfalls": exhausted, "scale": round(scale, 4), "target_hours": target_hours,
            "inoc_rate": INOC_RATE, "inoc_seeds": INOC_SEEDS, "max_dur": max_dur,
            "duration_dist": {"inherited": dist(base), "added": dist(added), "total": dist(out)},
            "audit_note": audit["note"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n合计 {len(out)} 行 / {hours(out):.3f}h（继承 {len(base)} + 新增 {len(added)}）")
    print(f"新增按语种 {dict(Counter(r['lang'] for r in added))}")
    print(f"新增按源 {dict(Counter(r.get('source_id') for r in added))}")
    print(f"新增行接种 {dict(inoc)}，短语 {dict(phrases)}")
    print(f"前缀行合计 {dict(pref)}，占该语种比例 "
          + ", ".join(f"{lg} {pref[lg]}/{tot[lg]}={pref[lg]/tot[lg]:.1%}" for lg in ("id", "tl")))
    print(f"候选被拒原因 {dict(rejected_all)}")
    if exhausted:
        print("源池不足（未重分配）：")
        for e in exhausted:
            print(f"  - {e}")
    print(f"审计：违规计数 {audit['violation_counts']}，评测文本命中 {audit['evaluation_text_rows']}，"
          f"验证文本命中 {audit['validation_text_rows']}，曝光>3× {len(audit['origins_above_3x'])}")
    print(f"时长越界（3–20s 外）：继承 {len(inherited_bad)} 行（r18 时代清单，20s 上限是 r19 才引入的，"
          f"本轮不重新过滤以保持单一变量）/ 新增 {len(added_bad)} 行（按 {max_dur:g}s 上限判定，必须为 0）")
    if not dry_run:
        print(f"train.jsonl SHA {sha256(dst / 'train.jsonl')}")
    if summary:
        print("summary:", json.dumps(summary, ensure_ascii=False, default=str)[:400])
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--target-hours", type=float, default=25.0)
    ap.add_argument("--dry-run", action="store_true", help="只算不写，用于先看各源能加多少")
    ap.add_argument("--max-dur", type=float, default=30.0,
                    help="新增行的时长上限（默认 30=官方 3–30s；继承行最长 29.63s，"
                         "用 20 会让新增数据的时长分布与继承数据不一致）")
    a = ap.parse_args()
    sys.exit(main(a.base, a.sources, a.dst, a.target_hours, dry_run=a.dry_run, max_dur=a.max_dur))
