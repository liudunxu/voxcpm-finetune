"""Build the 6-condition degraded-reference eval suites from omnivoice_prod_v2.jsonl.

按语种分层抽 6 条 × {th,tl,id} = 至多 18 条 case；条件矩阵 = clean（对照）+ 5 个
降质 profile，每个条件写一份独立 JSONL（case_id 与原 case 相同，跨条件靠报告间配对）。
本脚本只写 JSONL，不要求降质 wav 存在——wav 之后在远端用
`python -m voxft.data.ref_degrade` 生成（CPU 即可）。
"""
import argparse
import json
import random
from datetime import date
from pathlib import Path

from voxft.data.ref_degrade import PROFILES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval_cases" / "omnivoice_prod_v2.jsonl"
LANGS = ("th", "tl", "id")
CONDITIONS = ("clean", *PROFILES)
PER_LANG = 6
SEED = 42


def select_cases(rows: list[dict], *, per_lang: int = PER_LANG, seed: int = SEED,
                 progress=None) -> list[dict]:
    """seeded 每语种抽 per_lang 条（不足则全取并警告）；返回保持原文件顺序。"""
    rng = random.Random(seed)
    picked: set[str] = set()
    for lang in LANGS:
        pool = [r for r in rows if r.get("lang") == lang]
        if len(pool) < per_lang and progress:
            progress(f"⚠️ {lang} 只有 {len(pool)} 条，不足 {per_lang}，全取")
        picked.update(r["case_id"] for r in rng.sample(pool, min(per_lang, len(pool))))
    return [r for r in rows if r["case_id"] in picked]


def build_condition_rows(cases: list[dict], condition: str) -> list[dict]:
    """clean：行与原 case 逐字段一致，仅追加 ref_audio_original/ref_degradation=None；
    非 clean：ref_audio 改指 refs_degraded/<condition>/<原 basename>（相对路径），
    其余字段（text/lang/seed/note/numeric 等）原样保留。"""
    rows = []
    for case in cases:
        row = dict(case)
        row["ref_audio_original"] = case["ref_audio"]
        if condition == "clean":
            row["ref_degradation"] = None
        else:
            spec = PROFILES[condition]
            row["ref_audio"] = f"refs_degraded/{condition}/{Path(case['ref_audio']).name}"
            row["ref_degradation"] = {"profile": condition, "severity": spec["severity"]}
        rows.append(row)
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> bool:
    """原子写 JSONL；已存在且内容一致则幂等跳过，内容不同则报错——不覆盖旧产物。"""
    content = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return False
        raise ValueError(f"输出已存在且内容不同，不覆盖旧产物: {path}")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    return True


def build(source: str | Path = SOURCE, out_dir: str | Path | None = None, *,
          stamp: str, per_lang: int = PER_LANG, seed: int = SEED,
          progress=None) -> dict[str, int]:
    """生成 6 个条件文件，返回 {文件名: 条数}。"""
    source = Path(source)
    out_dir = Path(out_dir) if out_dir else source.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cases = select_cases(rows, per_lang=per_lang, seed=seed, progress=progress)
    summary = {}
    for condition in CONDITIONS:
        path = out_dir / f"refrob_{stamp}_{condition}.jsonl"
        written = _write_jsonl(build_condition_rows(cases, condition), path)
        summary[path.name] = len(cases)
        if progress:
            progress(f"{path.name}: {len(cases)} 条（{'写入' if written else '已存在且一致，跳过'}）")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out-dir", default=None, help="默认与 --source 同目录")
    ap.add_argument("--stamp", default=date.today().strftime("%Y%m%d"),
                    help="输出文件名 refrob_<stamp>_<condition>.jsonl 的日期戳")
    ap.add_argument("--per-lang", type=int, default=PER_LANG)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    build(args.source, args.out_dir, stamp=args.stamp, per_lang=args.per_lang,
          seed=args.seed, progress=print)
