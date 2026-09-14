"""把一轮微调的参数与核心指标抽成结构化记录，追加到 docs/runs.md。

所有字段都从已有产物里读：configs/<run>.yaml + .plan.json、混合数据集的 mix.json、
checkpoint 的 train.log、eval 报告 JSON。不新增状态文件——手抄一定会漏，漏了就回溯不了，
周报也无从汇总。

    uv run python -m voxft.train.runlog --run lora_omni_joint5 \\
        --eval <base报告.json> <ckpt报告.json> --verdict 不通过 --next 补 cue 长度数据
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import yaml

from ..paths import CHECKPOINT_DIR, CONFIG_DIR, ROOT

RUNLOG = ROOT / "docs" / "runs.md"

_HEADER = """# 微调运行记录

每轮一行摘要 + 一段明细，**按时间倒序追加**。字段由 `python -m voxft.train.runlog`
从 configs/、mix.json、train.log、eval 报告自动抽取，不要手改数字。

用途：下一轮微调的起点参照、周报汇总、以及「这个结论是哪一轮、用什么数据得出的」回溯。
"""

# 摘要表列名；新增字段先加这里
_COLS = ("run", "日期", "数据(h)", "步数", "LoRA r/α", "lr", "总体 CER", "退化语种", "结论")


def _read_yaml(run: str) -> dict:
    p = CONFIG_DIR / f"{run}.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def _read_plan(run: str) -> dict:
    p = CONFIG_DIR / f"{run}.plan.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _read_mix(train_manifest: str) -> dict:
    """混合数据集才有 mix.json；单源训练返回空。"""
    p = Path(train_manifest).parent / "mix.json" if train_manifest else None
    if p and p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        return {"language_shares": m.get("language_shares", {}),
                "train": m.get("train", {}), "max_repeat": m.get("max_repeat")}
    return {}


_VAL = re.compile(r"^\[val\] step (\d+): loss/total: ([\d.]+)")


def _val_loss(run: str) -> list[tuple[int, float]]:
    p = CHECKPOINT_DIR / run / "train.log"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _VAL.match(line.strip())
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def _report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _regressed(evals: list[dict], noise: float = 0.005) -> dict[str, dict[str, list[str]]]:
    """以第一份报告为基线，把 CER 变差的语种分成「红线」与「噪声级」两档。

    每语种只有十几条样本，单个字符的差异就能让均值动 0.002-0.003；若用极小阈值，
    几乎每轮都会被判"退化"，红线就失去意义。noise 默认 0.005 ≈ 整个语种子集里
    多错 2-3 个字符，低于它的照实列出但不触发红线。
    """
    if len(evals) < 2:
        return {}
    base = evals[0].get("by_lang", {})
    out = {}
    for e in evals[1:]:
        red, small = [], []
        for lang, v in sorted(e.get("by_lang", {}).items()):
            b = base.get(lang, {}).get("mean_cer")
            c = v.get("mean_cer")
            if b is None or c is None or c <= b:
                continue
            (red if c - b > noise else small).append(f"{lang} {b:.4f}→{c:.4f}")
        out[e.get("label", "?")] = {"red": red, "noise": small}
    return out


def build_record(run: str, eval_paths: list[str], verdict: str = "",
                 next_step: str = "", notes: str = "", noise: float = 0.005) -> str:
    cfg, plan = _read_yaml(run), _read_plan(run)
    mix = _read_mix(cfg.get("train_manifest", ""))
    lora = cfg.get("lora") or {}
    vals = _val_loss(run)
    evals = [_report(p) for p in eval_paths]
    regressed = _regressed(evals, noise)
    train = mix.get("train") or {}
    hours = train.get("hours")

    lines = [f"## {run}", "",
             f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 结论：**{verdict or '（未填）'}**"]
    if next_step:
        lines.append(f"- 下一步：{next_step}")
    lines += ["", "### 超参数", "",
              "| 项 | 值 |", "|---|---|",
              f"| finetune | {'lora' if lora else 'full'} |",
              f"| LoRA r/alpha/dropout | {lora.get('r')}/{lora.get('alpha')}/{lora.get('dropout')} |",
              f"| enable_lm/dit/proj | {lora.get('enable_lm')}/{lora.get('enable_dit')}/{lora.get('enable_proj')} |",
              f"| learning_rate | {cfg.get('learning_rate')} |",
              f"| batch × 累积 × GPU | {cfg.get('batch_size')} × {cfg.get('grad_accum_steps')} × {plan.get('gpus')} = {plan.get('effective_batch')} |",
              f"| epochs / num_iters | {plan.get('epochs')} / {cfg.get('num_iters')} |",
              f"| warmup / weight_decay / max_grad_norm | {cfg.get('warmup_steps')} / {cfg.get('weight_decay')} / {cfg.get('max_grad_norm')} |",
              f"| save/valid_interval | {cfg.get('save_interval')}/{cfg.get('valid_interval')} |",
              f"| max_batch_tokens | {cfg.get('max_batch_tokens')} |",
              f"| 基座 | {cfg.get('pretrained_path')} |",
              f"| 训练清单 | {cfg.get('train_manifest')} |",
              f"| 样本数 / 语种条数 | {plan.get('train_samples')} / {json.dumps(plan.get('langs'), ensure_ascii=False)} |"]
    if vals:
        lines.append(f"| val loss 首→末 | {vals[0][1]:.4f} (step {vals[0][0]}) → "
                     f"{vals[-1][1]:.4f} (step {vals[-1][0]})，最优 "
                     f"{min(v for _, v in vals):.4f} |")

    if mix:
        lines += ["", "### 数据配比（按时长）", "",
                  f"- 总时长 **{hours} h** / {train.get('rows')} 条，max_repeat={mix.get('max_repeat')}",
                  f"- 带 ref_audio：{train.get('with_ref_audio')}，带控制前缀：{train.get('with_control')}",
                  f"- max_exposure：{train.get('max_exposure')}",
                  "", "| 语种 | requested | actual | hours |", "|---|---|---|---|"]
        for lang, v in sorted((mix.get("language_shares") or {}).items()):
            lines.append(f"| {lang} | {v.get('requested')} | {v.get('actual')} | {v.get('hours')} |")

    if evals:
        langs = sorted({l for e in evals for l in e.get("by_lang", {})})
        e0 = evals[0]
        n_items = len(e0.get("items", []))
        n_cases = len({i["case_id"] for i in e0.get("items", [])})
        lines += ["", "### 验收（离线轨）", "",
                  f"- case 集与口径：cfg={e0.get('cfg_value')}、steps={e0.get('inference_timesteps')}、"
                  f"retry_badcase={e0.get('retry_badcase')}、ASR={e0.get('asr_model')}、"
                  f"{n_cases} 个 case × seed = {n_items} 条样本",
                  "", "| 语种 | " + " | ".join(e.get("label", "?") + " CER" for e in evals) + " |",
                  "|---|" + "---|" * len(evals)]
        for lang in langs:
            row = [lang]
            for e in evals:
                v = e.get("by_lang", {}).get(lang, {})
                row.append(f"{v.get('mean_cer')}"
                           + (f" (WER {v['mean_wer']})" if v.get("mean_wer") is not None else ""))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("| **总体** | " + " | ".join(str(e.get("mean_cer")) for e in evals) + " |")
        lines.append("| 疑似漏尾 | " + " | ".join(str(e.get("suspected_truncation_rate"))
                                                 for e in evals) + " |")
        lines += ["", f"退化语种（红线阈值 ΔCER > {noise}；任一语种触发即整轮不通过）："]
        for label, d in regressed.items():
            lines.append(f"- `{label}` 红线：" + ("；".join(d["red"]) if d["red"] else "无"))
            if d["noise"]:
                lines.append("  - 噪声级（未触发红线，照实记录）：" + "；".join(d["noise"]))
        lines += ["", "报告文件：" + "、".join(f"`{Path(p).name}`" for p in eval_paths)]

    lines += ["", "### 人工盲听", "",
              "（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；"
              "**指标与盲听冲突时以盲听为准**）", ""]
    if notes:
        lines += [notes, ""]
    lines.append("---")
    return "\n".join(lines) + "\n"


def append_record(run: str, eval_paths: list[str], verdict: str = "",
                  next_step: str = "", notes: str = "", noise: float = 0.005) -> Path:
    record = build_record(run, eval_paths, verdict, next_step, notes, noise)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    old = RUNLOG.read_text(encoding="utf-8") if RUNLOG.exists() else _HEADER
    # 新记录插在表头之后、旧记录之前：最近一轮永远在最上面
    RUNLOG.write_text(old.rstrip("\n") + "\n\n" + record, encoding="utf-8")
    return RUNLOG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="configs/<run>.yaml 的 run 名")
    ap.add_argument("--eval", nargs="*", default=[], metavar="REPORT.json",
                    help="eval 报告；第一份当基线，其余与它比退化")
    ap.add_argument("--verdict", default="", help="通过 / 不通过 + 一句原因")
    ap.add_argument("--next", dest="next_step", default="", help="下一轮要改什么")
    ap.add_argument("--notes", default="", help="自由文本（盲听结论、踩坑）")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="ΔCER 低于此值算噪声级波动，不触发红线（默认 0.005）")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="只打印不写入")
    args = ap.parse_args()
    text = build_record(args.run, args.eval, args.verdict, args.next_step, args.notes,
                        args.noise)
    if args.do_print:
        print(text)
    else:
        path = append_record(args.run, args.eval, args.verdict, args.next_step,
                             args.notes, args.noise)
        print(f"已追加 → {path}")


if __name__ == "__main__":
    main()
