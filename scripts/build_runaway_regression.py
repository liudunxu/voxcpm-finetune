#!/usr/bin/env python3
"""从已冻结的评测报告里提取「极端跑飞」item，生成常设开发回归集。

为什么要有这个集合：r18/r23/r24/r25 四轮对照显示，每轮都有 1–5 条极端跑飞
（1–3s 的口语 id 短句生成 6–21s），但**落在不同的 (case, seed) 上**——这是 3s 硬下限外的
结构性出分布问题，换配方消除不了。把它们固定成带 seed 的回归集，未来每轮只需
n×3 条输出就能复查这个失败形态，不必重跑整套验收。

只读已冻结报告，不生成音频、不调 ASR/TTS。输出行标注 observed_*（观测到的模型与数值）
与 training_eligible=false / language_quality=unverified：这是**结果富集的开发回归**，
不是独立验收集，也不能用它的命中率推算线上坏例率。
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_REPORTS = {
    "r18": [
        ("dramaprod", "work/rescore/lora_omni5_r18_idaccent_3ep_latest_744c4228c9a74a2884b8e476c1986349.json"),
        ("v2", "work/rescore/lora_omni5_r18_idaccent_3ep_latest_14e9edd24d5a441288a77e1614b8e547.json"),
        ("idprobe", "work/rescore/lora_omni5_r18_idaccent_3ep_latest_9385e915bf4d46e2b640b549446e5fe8.json"),
    ],
    "r23": [
        ("dramaprod", "work/rescore/r23_dramaprod.json"),
        ("v2", "work/rescore/lora_omni5_r23_thtlboost_cont_latest_4dd161ef2217403d8d415a7eb534c312.json"),
        ("idprobe", "work/rescore/lora_omni5_r23_thtlboost_cont_latest_dd5f5c56755149bebed5e4921e98559f.json"),
    ],
    "r24": [
        ("dramaprod", "work/rescore/lora_omni5_r24_coldstart_latest_9462d1a6a265417bb931a267c69bd78a.json"),
        ("v2", "work/rescore/lora_omni5_r24_coldstart_latest_556be769563640aa9bf96689218edd70.json"),
        ("idprobe", "work/rescore/lora_omni5_r24_coldstart_latest_bec165693e9b4edea8ae40d8c425b5b9.json"),
    ],
    "r25": [
        ("dramaprod", "work/rescore/lora_omni5_r25_r18_prefix_graft_latest_533cc495a29a465dbc7b58ab6a0fc055.json"),
        ("v2", "work/rescore/lora_omni5_r25_r18_prefix_graft_latest_3ee97cda229642bca1c6c8d97e1028c7.json"),
        ("idprobe", "work/rescore/lora_omni5_r25_r18_prefix_graft_latest_f6da45869f394e019b4fb6362aefa35d.json"),
    ],
}
LEN_RATIO_MIN = 2.0  # 生成音频 ≥2× 参考长度才算"极端跑飞"；1.4× 是 over_read 的既有阈值，太松


def collect(reports: dict[str, list[tuple[str, str]]], root: Path, len_ratio_min: float) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for model, entries in reports.items():
        for set_name, rel in entries:
            path = root / rel
            if not path.exists():
                print(f"  跳过缺失报告：{rel}", file=sys.stderr)
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            for it in doc.get("items") or []:
                ratio = it.get("len_ratio")
                if ratio is None or ratio < len_ratio_min:
                    continue
                key = (it.get("case_id"), it.get("seed"))
                row = seen.setdefault(key, {
                    "case_id": f"runaway_{it.get('case_id')}_s{it.get('seed')}",
                    "source_case_id": it.get("case_id"),
                    "seed": it.get("seed"),
                    "lang": it.get("lang"),
                    "text": it.get("text"),
                    "control": it.get("control") or "",
                    "ref_lang": it.get("ref_lang"),
                    "ref_audio": it.get("ref_audio"),
                    "numeric": bool(it.get("numeric")),
                    "condition": it.get("condition") or ("control" if (it.get("control") or "").strip() else "plain"),
                    "cohort": "runaway_development_regression",
                    "observed": [],
                    "training_eligible": False,
                    "language_quality": "unverified",
                })
                row["observed"].append({
                    "model": model, "set": set_name,
                    "audio_sec": it.get("audio_sec"), "len_ratio": ratio,
                    "cer": it.get("cer"), "over_read": bool(it.get("over_read")),
                    "tail_silence_sec": it.get("tail_silence_sec"),
                })
    out = sorted(seen.values(), key=lambda r: (r["lang"] or "", r["source_case_id"] or "", r["seed"] or 0))
    for row in out:
        worst = max(row["observed"], key=lambda o: o["len_ratio"])
        row["note"] = (
            f"极端跑飞开发回归：{worst['model']} 在 {worst['set']} 上生成 {worst['audio_sec']}s"
            f"（len_ratio {worst['len_ratio']}、CER {worst['cer']}）；"
            f"共 {len(row['observed'])} 个模型在此 (case,seed) 上越 {len_ratio_min}×："
            + ", ".join(f"{o['model']} {o['len_ratio']}×" for o in sorted(row["observed"], key=lambda o: -o["len_ratio"]))
            + "。保留原 case 的全部条件与原 seed；结果富集，不是独立验收集，"
              "其命中率不能当线上坏例率，也不代表母语者已确认多读内容。"
        )
    return out


def main(root=Path("."), out=Path("eval_cases/runaway_regression_20261023.jsonl"),
         len_ratio_min: float = LEN_RATIO_MIN, reports: dict | None = None) -> int:
    root, out = Path(root), Path(out)
    rows = collect(reports or DEFAULT_REPORTS, root, len_ratio_min)
    if not rows:
        print("没有命中任何极端跑飞 item，不写空文件", file=sys.stderr)
        return 1
    if out.exists():
        print(f"拒绝覆盖已存在的 {out}；要重建请先显式删除", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {len(rows)} 条 → {out}（阈值 len_ratio ≥ {len_ratio_min}）")
    for r in rows:
        w = max(r["observed"], key=lambda o: o["len_ratio"])
        print(f"  {r['lang']} {r['source_case_id']:<38} seed{r['seed']} "
              f"最差 {w['model']} {w['audio_sec']}s / {w['len_ratio']}× / CER {w['cer']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("eval_cases/runaway_regression_20261023.jsonl"))
    ap.add_argument("--len-ratio-min", type=float, default=LEN_RATIO_MIN)
    a = ap.parse_args()
    sys.exit(main(a.root, a.out, a.len_ratio_min))
