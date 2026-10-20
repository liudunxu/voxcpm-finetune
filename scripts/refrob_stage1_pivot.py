"""refrob 阶段 1 透视：按 condition × target 汇总合并报告（读 report_map.jsonl）。"""
import json
import sys
from pathlib import Path

WORK = Path("/root/autodl-tmp/refrob_20261020")


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"


def main():
    rows = []
    for line in (WORK / "report_map.jsonl").read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        r = json.loads(Path(m["merged"]).read_text(encoding="utf-8"))
        tgt = "base" if m["target"] == "base" else "r8"
        by_lang = r.get("by_lang", {})
        rows.append({
            "cond": m["condition"], "tgt": tgt,
            "cer": r.get("mean_cer"), "cer_nn": r.get("mean_cer_non_numeric"),
            "trunc": r.get("suspected_truncation_rate"), "over": r.get("over_read_rate"),
            "tail": r.get("mean_tail_silence"), "sim": r.get("mean_speaker_sim"),
            "lang_cer": {k: v.get("mean_cer") for k, v in by_lang.items()},
            "lang_sim": {k: v.get("mean_speaker_sim") for k, v in by_lang.items()},
        })
    hdr = f'{"condition":<22}{"mdl":<5}{"CER":>8}{"CERnn":>8}{"漏尾":>7}{"多读":>7}{"尾静音":>7}{"SIM":>7}  分语种CER'
    print(hdr)
    for r in rows:
        langs = " ".join(f"{k}:{fmt(v, 3)}" for k, v in sorted(r["lang_cer"].items()))
        print(f'{r["cond"]:<22}{r["tgt"]:<5}{fmt(r["cer"]):>8}{fmt(r["cer_nn"]):>8}'
              f'{fmt(r["trunc"], 3):>7}{fmt(r["over"], 3):>7}{fmt(r["tail"], 3):>7}'
              f'{fmt(r["sim"], 3):>7}  {langs}')


if __name__ == "__main__":
    sys.exit(main())
