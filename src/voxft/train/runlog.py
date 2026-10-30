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
CER_NOISE = 0.05

_HEADER = """# 微调运行记录

每轮一行摘要 + 一段明细，**按时间倒序追加**。字段由 `python -m voxft.train.runlog`
从 configs/、mix.json、train.log、eval 报告自动抽取，不要手改数字。

用途：下一轮微调的起点参照、周报汇总、以及「这个结论是哪一轮、用什么数据得出的」回溯。
"""

# 摘要表列名；新增字段先加这里
_COLS = ("run", "日期", "数据(h)", "步数", "LoRA r/α", "lr", "总体 CER", "退化语种", "结论")


def _read_yaml(run: str) -> dict:
    """缺配置必须硬失败：静默返回 {} 会把 LoRA 轮记成 finetune=full、其余超参全记成 None。"""
    p = CONFIG_DIR / f"{run}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"缺少训练配置 {p}；runlog 不从缺失配置推断超参，请先从 GPU 机取回该 yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


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


def _metric_delta(current: float, baseline: float) -> float:
    """去除远小于报告精度的浮点尾差，避免相等边界被误判为严格超过。"""
    return round(current - baseline, 12)


def _regressed(evals: list[dict], noise: float = CER_NOISE) -> dict[str, dict[str, list[str]]]:
    """以第一份报告为基线，把 CER 变差的语种分成「红线」与「噪声级」两档。

    默认门槛 0.05 是实测出来的，不是拍的：同一份配比（`fleurs_ms=17`，其余不变）跑两轮，
    ms 的 CER 是 0.0630 与 0.1053，**纯 run 间方差就有 0.042**（混合用共享 RNG，改任一权重
    都会挪动后续所有部分的抽样，再叠加训练非确定性）。门槛低于这个数必然天天误报，
    红线喊多了就等于没有红线。

    ⚠️ 正确的修法是提统计功效（每语种 30+ case、5 seed），不是继续调这个门槛；
    在功效提上来之前，红线只能当"值得去看一眼"的提示，不能当自动否决。
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
            (red if _metric_delta(c, b) > noise else small).append(f"{lang} {b:.4f}→{c:.4f}")
        out[e.get("label", "?")] = {"red": red, "noise": small}
    return out


# 时长类门禁阈值，标定依据见 _duration_gates docstring
DURATION_INFLATION_RATIO = 0.10    # mean audio_sec 相对涨幅上限
DURATION_INFLATION_CER_TOL = 0.01  # |Δmean_cer_non_numeric| ≤ 此值才算「变长但内容没变」
TAIL_SILENCE_P90_DELTA = 0.10      # p90 尾静音绝对增量上限（秒）
TAIL_SILENCE_P90_MAX = 0.5         # checkpoint p90 尾静音绝对上限（秒），对齐官方训练数据尾静音上限
SPEECH_RATIO_DROP = 0.05           # mean speech_ratio 下降上限

# 只用于 _seed_severity_lines 的参考分解，不是门禁阈值。2.0× 与常设回归集
# eval_cases/runaway_regression_*.jsonl 的提取阈值同一个数，便于两边对读。
OVER_READ_CATASTROPHIC = 2.0


def _duration_gates(evals: list[dict], *, tail_policy: str = "legacy") -> dict[str, dict[str, list[str]]]:
    """时长类红线：以第一份报告为基线，overall 与 by_lang 都查。

    三条门禁都是已被盲听交叉验证的失败形态（r2 vs base，84 条生产口径 case）：
    盲听 8 条「B 更差」里 6 条是「B 音频明显变长而 CER 没变」（最极端 ms_manglish
    1.76s→3.52s、CER 反而 0.094→0.000）；自动测量同向——尾部静音均值 0.084s→0.292s
    （p90 0.42s，0/84 越过 0.5s 的官方训练数据尾静音上限）、speech_ratio 0.94→0.84
    （缺陷 -0.10，门槛取一半）、总时长 +12.3% 而有声段字/秒只差 1.7%。⇒ 「变长但
    内容没变」（duration_inflation）、尾部垫静音（tail_silence）、有声占比下降
    （speech_ratio）不需要母语者就能判，可以进自动门禁。

    duration 与 CER 必须同取非数字样本，否则数字跑飞会被误判为「内容没变」。
    旧报告没有这些聚合字段时对应检查跳过、不误报；可用 eval --summarize 另存重算。
    tail_silence 的绝对上限
    （p90 > 0.5s）只依赖 checkpoint 自己的值，base 缺字段时仍然生效。
    absolute 策略（用户 2026-10-30 授权）仅以绝对上限否决；增量另记观察项。
    默认 legacy 保留历史调用的语义，新实验须显式记录所用策略。
    """
    if tail_policy not in ("legacy", "absolute"):
        raise ValueError(f"Unknown tail policy: {tail_policy}")
    if len(evals) < 2:
        return {}
    base, base_lang = evals[0], evals[0].get("by_lang", {})
    out = {}
    for e in evals[1:]:
        gates: dict[str, list[str]] = {"duration_inflation": [], "tail_silence": [],
                                       "speech_ratio": []}
        if tail_policy == "absolute":
            gates["tail_silence_observations"] = []
        scopes = [("overall", base, e)]
        scopes += [(lang, base_lang.get(lang, {}), v)
                   for lang, v in sorted(e.get("by_lang", {}).items())]
        for name, b, c in scopes:
            ba, ca = b.get("mean_audio_sec_non_numeric"), c.get("mean_audio_sec_non_numeric")
            bc, cc = b.get("mean_cer_non_numeric"), c.get("mean_cer_non_numeric")
            if (ba and ca and bc is not None and cc is not None
                    and _metric_delta(ca / ba, 1) > DURATION_INFLATION_RATIO
                    and abs(_metric_delta(cc, bc)) <= DURATION_INFLATION_CER_TOL):
                gates["duration_inflation"].append(
                    f"{name} 非数字 audio_sec {ba:.2f}→{ca:.2f}s（+{(ca / ba - 1) * 100:.0f}%），"
                    f"ΔCER非数字 {cc - bc:+.4f}——非数字音频变长但 CER 近似不变")
            bp, cp = b.get("p90_tail_silence"), c.get("p90_tail_silence")
            if cp is not None:
                if bp is not None and _metric_delta(cp, bp) > TAIL_SILENCE_P90_DELTA:
                    key = "tail_silence" if tail_policy == "legacy" else "tail_silence_observations"
                    gates[key].append(
                        f"{name} p90尾静音 {bp:.3f}→{cp:.3f}s（+{cp - bp:.3f}s）")
                if cp > TAIL_SILENCE_P90_MAX:
                    gates["tail_silence"].append(
                        f"{name} p90尾静音 {cp:.3f}s 越过 {TAIL_SILENCE_P90_MAX}s 上限")
            bs, cs = b.get("mean_speech_ratio"), c.get("mean_speech_ratio")
            if bs is not None and cs is not None and _metric_delta(bs, cs) > SPEECH_RATIO_DROP:
                gates["speech_ratio"].append(
                    f"{name} speech_ratio {bs:.3f}→{cs:.3f}（{cs - bs:+.3f}）")
        out[e.get("label", "?")] = gates
    return out


def _seed_severity_lines(evals: list[dict]) -> list[str]:
    """逐 seed 与严重度分解：**只作参考，不改任何门禁的语义与阈值**（改了就和历史轮次不可比）。

    为什么要这两张分解（2026-10-23 r26 的教训，标定见 docs/qc_gates.md）：
    - 极端跑飞按 seed 高度聚集：5 模型 × 3 套 × id 246 条 = 1230 条里 `len_ratio≥2` 的 10 条，
      seed44 占 8 条（均匀零假设单侧 p=0.0034）、seed43 占 2、seed42 为 0 ⇒ 只看均值/总数
      会把 seed 运气读成模型差异。
    - 「多读条数」不分严重度会给出相反结论：r26 在 dramaprod 是 4/96（越 ≤3/96 门），
      但 4 条全是 1.50–2.00× 的中等级；r25 只有 2/96，其中一条 7.97×（11.84s 对 2.5s 台词）。
      按条数 r26 更差、按严重度 r26 更好。
    """
    lines = ["", f"逐 seed 与严重度分解（参考、不作门禁；`len_ratio` ≥{OVER_READ_CATASTROPHIC} 记灾难级，"
                 f"其余多读记中等级）："]
    for e in evals:
        items = e.get("items") or []
        if not items:
            continue
        lines.append(f"- `{e.get('label', '?')}`")
        for lang in sorted({i.get("lang") for i in items if i.get("lang")}):
            rows = [i for i in items if i.get("lang") == lang]
            mod = sum(1 for i in rows if i.get("over_read")
                      and (i.get("len_ratio") or 0) < OVER_READ_CATASTROPHIC)
            cat = sum(1 for i in rows if (i.get("len_ratio") or 0) >= OVER_READ_CATASTROPHIC)
            per = []
            for s in sorted({i.get("seed") for i in rows}):
                sub = [i for i in rows if i.get("seed") == s]
                ratios = [i.get("len_ratio") or 0 for i in sub]
                tails = sorted(i["tail_silence_sec"] for i in sub
                               if i.get("tail_silence_sec") is not None)
                p90 = tails[min(len(tails) - 1, int(0.9 * len(tails)))] if tails else None
                per.append(f"s{s} 多读{sum(1 for i in sub if i.get('over_read'))}/{len(sub)}"
                           f" max{max(ratios):.2f}×" + (f" p90尾{p90:.2f}s" if p90 is not None else ""))
            lines.append(f"  - {lang}（{len(rows)} 条）中等 {mod} / 灾难 {cat} ｜ " + " ｜ ".join(per))
    return lines


def build_record(run: str, eval_paths: list[str], verdict: str = "",
                 next_step: str = "", notes: str = "", noise: float = CER_NOISE,
                 *, tail_policy: str = "legacy") -> str:
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
        # 官方 validate() 只跑 val_loader 的前 max_val_batches=10 个 batch，而 build_dataloader
        # 对 val 也是 shuffle=True ⇒ 这是约 20 条随机样本的估计，跨轮不可比，更不能用来选 checkpoint。
        lines.append(f"| val loss 首→末 | {vals[0][1]:.4f} (step {vals[0][0]}) → "
                     f"{vals[-1][1]:.4f} (step {vals[-1][0]})，最优 "
                     f"{min(v for _, v in vals):.4f}"
                     f"（≤10 batch≈20 条随机子集，跨轮不可比、不作选点依据） |")

    if mix:
        lines += ["", "### 数据配比（按时长）", "",
                  f"- 总时长 **{hours} h** / {train.get('rows')} 条，max_repeat={mix.get('max_repeat')}",
                  f"- 带 ref_audio：{train.get('with_ref_audio')}，带控制前缀：{train.get('with_control')}",
                  f"- max_exposure：{train.get('max_exposure')}",
                  "", "| 语种 | requested | actual | hours |", "|---|---|---|---|"]
        for lang, v in sorted((mix.get("language_shares") or {}).items()):
            lines.append(f"| {lang} | {v.get('requested')} | {v.get('actual')} | {v.get('hours')} |")

    if evals:
        langs = sorted({lang for e in evals for lang in e.get("by_lang", {})})
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
        dur = _duration_gates(evals, tail_policy=tail_policy)
        tail_rule = (f"增量 >{TAIL_SILENCE_P90_DELTA}s 或绝对值 >{TAIL_SILENCE_P90_MAX}s"
                     if tail_policy == "legacy" else
                     f"绝对值 >{TAIL_SILENCE_P90_MAX}s（增量仅观察）")
        lines += ["", f"时长类门禁（非数字 audio_sec 涨幅 >{DURATION_INFLATION_RATIO:.0%} 且 "
                  f"|ΔCER非数字| ≤{DURATION_INFLATION_CER_TOL}；尾静音策略={tail_policy}，p90尾静音"
                  f"{tail_rule}；speech_ratio 降 "
                  f">{SPEECH_RATIO_DROP}；标定依据见 docs/qc_gates.md）："]
        for label, g in dur.items():
            hits = [x for k in ("duration_inflation", "tail_silence", "speech_ratio")
                    for x in g[k]]
            lines.append(f"- `{label}`：" + ("；".join(hits) if hits else "无"))
            if g.get("tail_silence_observations"):
                lines.append("  - 尾静音观察（不否决）：" + "；".join(g["tail_silence_observations"]))
        if any(e.get("mean_audio_sec_non_numeric") is None for e in evals):
            lines.append("非数字时长字段缺失/无适用样本：该项未评，不代表通过；"
                         "旧报告先用 `voxft.eval --summarize` 另存重算。")
        lines += _seed_severity_lines(evals)
        lines += ["", "报告文件：" + "、".join(f"`{Path(p).name}`" for p in eval_paths)]

    lines += ["", "### 人工盲听", "",
              "（声学异常与母语评分分开记录；未评的口音/语言自然度/情绪保持未验证，"
              "工程检查通过不等于完整语言验收通过）", ""]
    if notes:
        lines += [notes, ""]
    lines.append("---")
    return "\n".join(lines) + "\n"


def append_record(run: str, eval_paths: list[str], verdict: str = "",
                  next_step: str = "", notes: str = "", noise: float = CER_NOISE,
                  *, tail_policy: str = "legacy") -> Path:
    """把新一轮记录插到表头之后、旧记录之前——最近一轮永远在最上面。"""
    record = build_record(run, eval_paths, verdict, next_step, notes, noise, tail_policy=tail_policy)
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    old = RUNLOG.read_text(encoding="utf-8") if RUNLOG.exists() else ""
    if old.startswith("# 微调运行记录"):
        head, sep, rest = old.partition("\n## ")
        new = head.rstrip("\n") + "\n\n" + record.rstrip("\n") + "\n" + (sep + rest if sep else "")
    else:
        new = (old or _HEADER).rstrip("\n") + "\n\n" + record
    RUNLOG.write_text(new, encoding="utf-8")
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
    ap.add_argument("--noise", type=float, default=CER_NOISE,
                    help=f"ΔCER 低于此值不触发红线，不代表已证明无退化（默认 {CER_NOISE}）")
    ap.add_argument("--tail-policy", choices=("legacy", "absolute"), default="legacy",
                    help="新实验显式用 absolute：p90>0.5s 否决，增量仅观察；legacy 供历史复现")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="只打印不写入")
    args = ap.parse_args()
    text = build_record(args.run, args.eval, args.verdict, args.next_step, args.notes,
                        args.noise, tail_policy=args.tail_policy)
    if args.do_print:
        print(text)
        # 只打印时明确说出来：曾把 --print 的输出当成已落盘，结果那一轮记录根本没写进 runs.md
        print("\n（--print：只打印，未写入 docs/runs.md）")
    else:
        path = append_record(args.run, args.eval, args.verdict, args.next_step,
                             args.notes, args.noise, tail_policy=args.tail_policy)
        print(f"已追加 → {path}")


if __name__ == "__main__":
    main()
