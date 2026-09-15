"""微调离线验收：逐 case 固定 ref/control/seed，ASR 诊断 + 母语盲听。

ASR 误差不等于发音错误，疑似漏尾不等于真实截断，F0 不是越高越好。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from . import infer
from .paths import CHECKPOINT_DIR

SAMPLE_BY_LANG = infer.SAMPLE_TEXTS
EVAL_DIR = CHECKPOINT_DIR / "eval"

# Taglish 句内英文多，强制单一语言解码会给出失真的转写；th/vi/id/ms 都是 Whisper
# 标准语种，强制解码让 CER 在不同 checkpoint 之间可比。
AUTO_DETECT_LANGS = frozenset({"tl"})

# 按空格切词有意义的语种才算 WER。vi 正字法以音节为单位空格分隔，它的 WER 是音节级
# 错误率，与 tl/en/id/ms 的词级不同量纲，别横向比；th 词间根本没有空格，只有 CER。
WER_LANGS = frozenset({"tl", "en", "vi", "id", "ms"})


def _transcribe(model, wav_path: str, lang: str) -> str:
    segs, _ = model.transcribe(wav_path,
                               language=None if lang in AUTO_DETECT_LANGS else lang,
                               vad_filter=True)
    return " ".join(s.text.strip() for s in segs)


def _norm(text: str, keep_spaces: bool = False) -> str:
    # Thai 的声调/元音组合符属于 M 类，isalnum 会把它们误删。
    text = unicodedata.normalize("NFC", text.lower())
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNM"
                   or (keep_spaces and ch.isspace()))


def _error_rate(hyp, ref) -> float:
    """Levenshtein，CER/WER 可大于 1（插入错误），不截断。"""
    previous = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, 1):
        current = [i]
        for j, actual in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (actual != expected)))
        previous = current
    return previous[-1] / max(1, len(ref))


def _is_truncated(hyp: str, ref: str, tail: int = 8) -> bool:
    """仅定位疑似漏尾（= 少读）；ASR、同义转写也可能触发，必须听音确认。"""
    h, r = _norm(hyp), _norm(ref)
    return bool(r) and (len(h) < 0.6 * len(r) or
                       SequenceMatcher(None, h[-tail:], r[-tail:]).ratio() < 0.5)


def _over_read(hyp: str, ref: str) -> bool:
    """多读/跑飞：归一化文本比参考长出 40% 以上。门限取 OmniVoice 的 overread 判定
    （api.py:6529-6536，长文本档就是 1.4×）。"""
    h, r = _norm(hyp), _norm(ref)
    return bool(r) and len(h) > 1.4 * len(r)


def _is_numeric(case: dict) -> bool:
    """数字类 case 要单独汇总：Whisper 自己会把口播数字词归一成阿拉伯数字或货币符号，
    实测 "isang libo't limang daan pesos" 被转写成 "1,500 pesos"，CER 因此虚高到 0.588
    而 base 与 checkpoint 完全相同——这一类的 CER 差异不能当作 TTS 质量差异。"""
    if "numeric" in case:
        return bool(case["numeric"])
    return bool(re.search(r"\d", case.get("text") or ""))


def _acoustics(wav_path: str, text: str, ref_path: str | None, ref_cache: dict) -> dict:
    """韵律 + 质检指标；音频只解码一次，参考音频按路径缓存。"""
    from .data.pipeline import audio_metrics, load_wav_mono
    from .qc import audio as qc

    wav, sr = load_wav_mono(wav_path)
    out = dict(audio_metrics(wav, sr, text))
    ref = None
    if ref_path:
        if ref_path not in ref_cache:
            ref_cache[ref_path] = load_wav_mono(ref_path)
        ref = ref_cache[ref_path]
    out.update(qc.analyze(wav, sr, ref[0] if ref else None, ref[1] if ref else None))
    ref_chars = len(_norm(text))
    out["chars_per_sec"] = round(ref_chars / out["audio_sec"], 3) if out["audio_sec"] else None
    return out


def _agg(g: list[dict]) -> dict:
    def mean(key, subset=None):
        vals = [i[key] for i in (g if subset is None else subset) if i.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def rate(key):
        vals = [i[key] for i in g if i.get(key) is not None]
        return round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None

    def p90(key):
        vals = sorted(i[key] for i in g if i.get(key) is not None)
        return round(vals[min(len(vals) - 1, int(0.9 * len(vals)))], 4) if vals else None

    out = {
        "cases": len(g),
        "mean_cer": mean("cer"),
        "mean_similarity": mean("similarity"),
        "suspected_truncation_rate": rate("suspected_truncation"),  # 少读 / 漏尾
        "over_read_rate": rate("over_read"),                        # 多读 / 跑飞
        "metallic_rate": rate("metallic"),
        "low_snr_rate": rate("low_snr"),
        "mean_speaker_sim": mean("speaker_sim"),
        "mean_chars_per_sec": mean("chars_per_sec"),
        "mean_speech_ratio": mean("speech_ratio"),
        "mean_audio_sec": mean("audio_sec"),
        "mean_head_silence": mean("head_silence_sec"),
        "mean_tail_silence": mean("tail_silence_sec"),
        "p90_tail_silence": p90("tail_silence_sec"),
        "mean_spectral_rolloff_99": mean("spectral_rolloff_99"),
        "mean_band_ratio_2_8k": mean("band_ratio_2_8k"),
    }
    plain = [i for i in g if not i["numeric"]]
    if plain:
        # 数字类单列：Whisper 会把口播数字词归一成阿拉伯数字或货币符号，
        # 那一类的 CER 差异不代表 TTS 质量差异，混进总均值会把结论带偏
        out["non_numeric_cases"] = len(plain)
        out["mean_cer_non_numeric"] = mean("cer", plain)
    wers = [i["wer"] for i in g if i["wer"] is not None]
    if wers:
        out["mean_wer"] = round(sum(wers) / len(wers), 4)
    return out


_REPORT_NOTE = (
    "ASR/漏尾均为诊断；F0 不作通过门限。按语言、ref 语言、角色、情绪分组做母语盲听。"
    "rate 的量纲随语种不同（th 字符/秒、vi 音节/秒、tl/en/id/ms 词级），不横向比。"
    "suspected_truncation=少读/漏尾，over_read=多读/跑飞（>1.4× 参考长度）。"
    "mean_cer_non_numeric 剔除了含阿拉伯数字的 case——那一类的 CER 会被 Whisper "
    "自身的数字归一化污染，只能靠盲听。metallic 与 low_snr 阈值移植自 OmniVoice "
    "生产口径，但 168 条盲听标注样本标定定案：metallic_score 对人工 noise 标注 "
    "AUC=0.060（反相关），low_snr AUC=0.509（纯随机）且 28.7% 误报（主因是参考音频"
    "噪底）——两者永久只作参考值，不作通过门限。speaker_sim 是 WavLM X-vector 余弦，"
    "只做同一 ref 下 base 与 checkpoint 的相对比较，与生产 ERes2NetV2 门限刻度不可"
    "互换。各指标口径与门禁阈值详见 docs/qc_gates.md。"
)


def _report_metrics(items: list[dict]) -> dict:
    """top-level 聚合 + by_lang；evaluate 与 merge_reports 共用，保证口径一致。"""
    overall = _agg(items)
    by_lang = {lang: _agg([i for i in items if i["lang"] == lang])
               for lang in sorted({i["lang"] for i in items})}
    out = {k: overall.get(k) for k in (
        "mean_similarity", "mean_cer", "mean_cer_non_numeric", "suspected_truncation_rate",
        "over_read_rate", "metallic_rate", "low_snr_rate", "mean_speaker_sim",
        "mean_chars_per_sec", "mean_speech_ratio", "mean_audio_sec", "mean_head_silence",
        "mean_tail_silence", "p90_tail_silence", "mean_spectral_rolloff_99",
        "mean_band_ratio_2_8k")}
    out["mean_f0_std"] = round(sum(i["f0_std_st"] for i in items) / len(items), 2)
    out["by_lang"] = by_lang
    return out


def evaluate(target: str, lang: str, texts: list[str | dict],
             base: str | None = None, ref_audio: str | None = None,
             control: str | None = None, seed: int = 42, *,
             seeds: list[int] | None = None, cfg_value: float = 2.0,
             inference_timesteps: int = 20,
             shard: tuple[int, int] | None = None) -> dict:
    """JSONL case 可覆盖 text/lang/ref_audio/ref_lang/control/seed，其他标签原样保留。

    shard=(k, n) 时只跑原始序号 i % n == k 的 case（case_id 保持原始序号），
    用于多进程并行跑同一份 case 集，事后用 merge_reports 合并。
    """
    from .data.pipeline import _whisper_model

    if not texts or (seeds is not None and not seeds):
        raise ValueError("评测台词和种子不能为空")
    if shard is not None:
        k, n = shard
        if not (0 <= k < n):
            raise ValueError(f"shard 须满足 0 <= k < n，收到 {k}/{n}")
    cases = []
    for i, raw in enumerate(texts):
        case = {"case_id": str(i), "lang": lang, "ref_audio": ref_audio,
                "control": control or "", **({"text": raw} if isinstance(raw, str) else raw)}
        if not isinstance(case.get("text"), str) or not _norm(case["text"]):
            raise ValueError(f"case {i} 缺少有效文本")
        if case["lang"] not in SAMPLE_BY_LANG:
            raise ValueError(f"case {i} 不支持语言 {case['lang']}")
        if case.get("ref_audio") and not Path(case["ref_audio"]).is_file():
            raise ValueError(f"case {i} 参考音频不存在: {case['ref_audio']}")
        cases.append(case)
    if shard is not None:
        cases = [c for c in cases if int(c["case_id"]) % shard[1] == shard[0]]
        if not cases:
            raise ValueError(f"shard {shard[0]}/{shard[1]} 没有分到任何 case")

    whisper = _whisper_model(lang, "large-v3")
    lora = None if target == "base" else target
    model = infer.get_model(base, lora)
    label = Path(target).parent.name + "_" + Path(target).name if lora else "base"
    if shard is not None:
        label += f"_shard{shard[0]}of{shard[1]}"
    items = []
    ref_cache: dict = {}
    for case in cases:
        for requested_seed in seeds if seeds is not None else [int(case.get("seed", seed))]:
            kw = infer._gen_kwargs(case["text"], case.get("ref_audio"), None,
                                   cfg_value, inference_timesteps, requested_seed, case["control"])
            kw["retry_badcase"] = False  # 离线禁用换种子重试，避免 A/B 条件不一致
            wav_path, gen_sec = infer._run(model, kw)
            hyp = _transcribe(whisper, wav_path, case["lang"])
            h, r = _norm(hyp), _norm(case["text"])
            items.append({
                **case, "seed": requested_seed, "hyp": hyp,
                "similarity": round(SequenceMatcher(None, h, r).ratio(), 4),
                "cer": round(_error_rate(h, r), 4),
                "wer": round(_error_rate(_norm(hyp, True).split(),
                                          _norm(case["text"], True).split()), 4)
                       if case["lang"] in WER_LANGS else None,
                "numeric": _is_numeric(case),
                "suspected_truncation": _is_truncated(hyp, case["text"]),   # = 少读/漏尾
                "over_read": _over_read(hyp, case["text"]),                  # = 多读/跑飞
                "len_ratio": round(len(h) / len(r), 3) if r else None,
                **_acoustics(wav_path, case["text"], case.get("ref_audio"), ref_cache),
                "wav": wav_path, "gen_sec": gen_sec,
                "human_review": {"naturalness_1_5": None, "emotion_fit_1_5": None,
                                 "speaker_similarity_1_5": None, "intelligibility_1_5": None,
                                 "cutoff": None, "noise": None, "notes": ""},
            })

    report = {
        "target": target, "base": infer._resolve_base(base), "label": label,
        "cfg_value": cfg_value, "inference_timesteps": inference_timesteps,
        "retry_badcase": False, "asr_model": "large-v3",
        "asr_auto_detect_langs": sorted(AUTO_DETECT_LANGS),
        **({"shard": f"{shard[0]}/{shard[1]}"} if shard is not None else {}),
        **_report_metrics(items),
        "note": _REPORT_NOTE,
        "items": items,
    }
    out_dir = EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_{uuid4().hex}.json"
    report["report_path"] = str(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def merge_reports(paths: list[str]) -> dict:
    """合并同一 target 的分片报告：items 取并集，聚合指标用 _report_metrics 重算。

    分片报告的 case_id 是原始序号（见 evaluate 的 shard 参数），所以合出来的
    报告与一次性整跑完全同口径，可以直接与整跑的基线报告做盲听配对。
    """
    if len(paths) < 2:
        raise ValueError("合并至少需要两份分片报告")
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    first = docs[0]
    for d in docs[1:]:
        for key in ("target", "base", "cfg_value", "inference_timesteps", "asr_model"):
            if d.get(key) != first.get(key):
                raise ValueError(f"分片报告口径不一致（{key}）：{first.get(key)} vs {d.get(key)}")
    seen: set[tuple[str, int]] = set()
    items = []
    for d in docs:
        for i in d["items"]:
            k = (i["case_id"], i["seed"])
            if k in seen:
                raise ValueError(f"case_id={k[0]} seed={k[1]} 在多份分片里重复出现")
            seen.add(k)
            items.append(i)
    items.sort(key=lambda i: (int(i["case_id"]), i["seed"]))
    label = re.sub(r"_shard\d+of\d+$", "", first["label"])
    report = {
        "target": first["target"], "base": first["base"], "label": label,
        "cfg_value": first["cfg_value"], "inference_timesteps": first["inference_timesteps"],
        "retry_badcase": first.get("retry_badcase", False),
        "asr_model": first.get("asr_model"),
        "asr_auto_detect_langs": first.get("asr_auto_detect_langs", []),
        "merged_from": [str(p) for p in paths],
        **_report_metrics(items),
        "note": _REPORT_NOTE,
        "items": items,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"{label}_{uuid4().hex}.json"
    report["report_path"] = str(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# 与 evaluate() 写进 item 的 human_review 字段名保持一致
_REVIEW_FIELDS = ("intelligibility_1_5", "naturalness_1_5", "speaker_similarity_1_5",
                  "emotion_fit_1_5", "cutoff", "noise", "notes")


def list_reports() -> list[str]:
    return sorted(p.name for p in EVAL_DIR.glob("*.json")) if EVAL_DIR.exists() else []


def review_key(pair: dict) -> str:
    return f"{pair['case_id']}|{pair['seed']}"


def _mean(xs: list[float]):
    return round(sum(xs) / len(xs), 3) if xs else None


def review_session(report_a: str, report_b: str, seed: int = 0) -> list[dict]:
    """按 (case_id, seed) 把两份报告配成盲听条目，甲/乙顺序随机。

    who_1/who_2 只在服务端用于回写，页面不显示——评分者一旦知道哪条是 base，
    就会朝"微调应该更好"的方向偏，盲听也就失去意义。
    """
    a = json.loads((EVAL_DIR / report_a).read_text(encoding="utf-8"))
    b = json.loads((EVAL_DIR / report_b).read_text(encoding="utf-8"))
    idx = {(i["case_id"], i["seed"]): i for i in b["items"]}
    rng = random.Random(seed)
    out = []
    for ia in a["items"]:
        ib = idx.get((ia["case_id"], ia["seed"]))
        if ib is None:
            continue
        swap = rng.random() < 0.5
        first, second = (ib, ia) if swap else (ia, ib)
        out.append({"case_id": ia["case_id"], "seed": ia["seed"], "lang": ia["lang"],
                    "text": ia["text"], "wav_1": first["wav"], "wav_2": second["wav"],
                    "who_1": "b" if swap else "a", "who_2": "a" if swap else "b"})
    if not out:
        raise ValueError(f"{report_a} 与 {report_b} 配不出任何 (case_id, seed)；"
                         "两份报告必须用同一份 case 集与同一组 seed 跑出来")
    return out


BLIND_LOSS_MARGIN = 2     # 负 要比 胜 多这么多条才算退化
BLIND_MIN_LOSSES = 3      # 且负本身至少这么多条，否则单条差异就会否决整轮


def save_reviews(report_a: str, report_b: str, session: list[dict],
                 ratings: dict | None) -> dict:
    """把人工评分写回两份报告的 human_review，返回分语种 A/B 汇总。

    ratings: {review_key(pair): {"s1": {...}, "s2": {...}}}，字段名取 _REVIEW_FIELDS。
    汇总的 win/tie/loss 以自然度比较 B 相对 A（B 一般是 checkpoint）。

    退化判据不是简单的「负 > 胜」：实测 id 拿到 0胜/14平/1负、zh 拿到 0胜/2平/1负，
    按「负 > 胜」两条都算退化，但那显然是一两条听感的偶然波动。所以要求
    `负 - 胜 >= BLIND_LOSS_MARGIN` 且 `负 >= BLIND_MIN_LOSSES`，
    在 12-18 条的量级上这个门槛刚好能挡住单条噪声、又不会放过成片的退化。
    """
    docs = {who: json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))
            for who, name in (("a", report_a), ("b", report_b))}
    index = {who: {(i["case_id"], i["seed"]): i for i in d["items"]}
             for who, d in docs.items()}
    nat_by_lang: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"a": [], "b": []})
    verdict: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "tie": 0, "loss": 0, "unrated": 0})
    rated = 0
    for pair in session:
        entry = (ratings or {}).get(review_key(pair)) or {}
        nat: dict[str, float | None] = {}
        for slot in ("1", "2"):
            r = entry.get(f"s{slot}") or {}
            who = pair[f"who_{slot}"]
            item = index[who].get((pair["case_id"], pair["seed"]))
            if item is None:
                continue
            hr = item.setdefault("human_review", {})
            for f in _REVIEW_FIELDS:
                if f in r:
                    hr[f] = r[f]
            hr["paired_report"] = report_b if who == "a" else report_a
            v = r.get("naturalness_1_5")
            nat[who] = float(v) if isinstance(v, (int, float)) and v > 0 else None
        lang = pair["lang"]
        for who in ("a", "b"):
            if nat.get(who) is not None:
                nat_by_lang[lang][who].append(nat[who])
        if nat.get("a") is not None and nat.get("b") is not None:
            rated += 1
            d = nat["b"] - nat["a"]
            verdict[lang]["win" if d > 0 else "loss" if d < 0 else "tie"] += 1
        else:
            verdict[lang]["unrated"] += 1
    for who, name in (("a", report_a), ("b", report_b)):
        p = EVAL_DIR / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(docs[who], ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(p)
    by_lang = {}
    for lang, v in sorted(nat_by_lang.items()):
        d = verdict[lang]
        regressed = (d["loss"] - d["win"] >= BLIND_LOSS_MARGIN
                     and d["loss"] >= BLIND_MIN_LOSSES)
        by_lang[lang] = {"mean_naturalness_a": _mean(v["a"]),
                         "mean_naturalness_b": _mean(v["b"]), **d,
                         "regressed": regressed}
    return {"pairs": len(session), "rated": rated, "by_lang": by_lang,
            "regressed": [k for k, v in by_lang.items() if v["regressed"]],
            "written": [report_a, report_b]}


def _row(name: str, d: dict) -> str:
    return (f"{name:<40} {d.get('mean_cer'):>7} {str(d.get('mean_cer_non_numeric')):>9} "
            f"{str(d.get('suspected_truncation_rate')):>6} {str(d.get('over_read_rate')):>6} "
            f"{str(d.get('metallic_rate')):>6} {str(d.get('mean_speaker_sim')):>7} "
            f"{str(d.get('mean_chars_per_sec')):>7}")


def print_compare(reports: list[dict]) -> None:
    print(f"{'checkpoint':<40} {'CER↓':>7} {'CER非数字':>9} {'少读':>6} {'多读':>6} "
          f"{'金属音':>6} {'SIM':>7} {'字/秒':>7}")
    for r in reports:
        print(_row(r["target"], r))
        if len(r.get("by_lang", {})) > 1:
            for lang, s in r["by_lang"].items():
                print(_row("  " + lang, s))
    print("\n不能凭以上指标自动通过；请做母语盲听，检查情绪、音色、自然度和真实截断。")
    print("数字类 case 的 CER 会被 Whisper 自身的数字归一化污染，结论看「CER非数字」那一列；")
    print("SIM 是 WavLM X-vector 余弦，只在同一 ref 下做 base 与 checkpoint 的相对比较；")
    print("metallic/low_snr 已标定定案为永久参考值（AUC 0.060 反相关 / 0.509 纯随机），不作门限。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="*", help='"base" 或若干 LoRA checkpoint 目录')
    ap.add_argument("--lang", default="th", choices=list(SAMPLE_BY_LANG))
    ap.add_argument("--base", default=None)
    ap.add_argument("--texts-file", default=None, help="逐 case JSONL；相对 ref 路径按此文件目录解析")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, help="覆盖每个 case 的 seed，推荐 42 43 44")
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--inference-timesteps", type=int, default=20)
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="只跑第 K 片（0 起）共 N 片，多进程并行跑同一份 case 集；"
                         "case_id 保持原始序号，跑完用 --merge 合并")
    ap.add_argument("--merge", nargs="+", metavar="REPORT",
                    help="合并若干分片报告（voxft_ckpt/eval 下的文件名或路径）")
    args = ap.parse_args()
    if args.merge:
        paths = [p if Path(p).is_file() else str(EVAL_DIR / p) for p in args.merge]
        report = merge_reports(paths)
        print_compare([report])
        print(f"\n合并报告: {report['report_path']}")
        return
    if not args.targets:
        ap.error("需要至少一个 target（或改用 --merge）")
    shard = None
    if args.shard:
        m = re.fullmatch(r"(\d+)/(\d+)", args.shard)
        if not m:
            ap.error("--shard 格式是 K/N（如 0/3）")
        shard = (int(m.group(1)), int(m.group(2)))
    if args.texts_file:
        path = Path(args.texts_file).resolve()
        texts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for case in texts:
            if case.get("ref_audio"):
                case["ref_audio"] = str((path.parent / case["ref_audio"]).resolve())
    else:
        texts = [SAMPLE_BY_LANG[args.lang]]
        print("提示：内置单句仅为 smoke test，不能用于微调验收。")
    reports = [evaluate(t, args.lang, texts, args.base, args.ref_audio, args.control,
                        args.seed, seeds=args.seeds, cfg_value=args.cfg_value,
                        inference_timesteps=args.inference_timesteps,
                        shard=shard) for t in args.targets]
    print_compare(reports)
    print(f"\n详细报告: {CHECKPOINT_DIR / 'eval'}/")


if __name__ == "__main__":
    main()
