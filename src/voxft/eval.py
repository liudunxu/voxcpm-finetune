"""微调离线验收：逐 case 固定 ref/control/seed，ASR 诊断 + 母语盲听。

ASR 误差不等于发音错误，疑似漏尾不等于真实截断，F0 不是越高越好。
"""
from __future__ import annotations

import argparse
import json
import random
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
    """仅定位疑似漏尾；ASR、同义转写也可能触发，必须听音确认。"""
    h, r = _norm(hyp), _norm(ref)
    return bool(r) and (len(h) < 0.6 * len(r) or
                       SequenceMatcher(None, h[-tail:], r[-tail:]).ratio() < 0.5)


def _prosody(wav_path: str, text: str) -> dict:
    from .data.pipeline import audio_metrics, load_wav_mono
    wav, sr = load_wav_mono(wav_path)
    return audio_metrics(wav, sr, text)


def evaluate(target: str, lang: str, texts: list[str | dict],
             base: str | None = None, ref_audio: str | None = None,
             control: str | None = None, seed: int = 42, *,
             seeds: list[int] | None = None, cfg_value: float = 2.0,
             inference_timesteps: int = 20) -> dict:
    """JSONL case 可覆盖 text/lang/ref_audio/ref_lang/control/seed，其他标签原样保留。"""
    from .data.pipeline import _whisper_model

    if not texts or (seeds is not None and not seeds):
        raise ValueError("评测台词和种子不能为空")
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

    whisper = _whisper_model(lang, "large-v3")
    lora = None if target == "base" else target
    model = infer.get_model(base, lora)
    label = Path(target).parent.name + "_" + Path(target).name if lora else "base"
    items = []
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
                "suspected_truncation": _is_truncated(hyp, case["text"]),
                **_prosody(wav_path, case["text"]), "wav": wav_path, "gen_sec": gen_sec,
                "human_review": {"naturalness_1_5": None, "emotion_fit_1_5": None,
                                 "speaker_similarity_1_5": None, "intelligibility_1_5": None,
                                 "cutoff": None, "noise": None, "notes": ""},
            })
    by_lang = {}
    for lang in sorted({i["lang"] for i in items}):
        g = [i for i in items if i["lang"] == lang]
        wers = [i["wer"] for i in g if i["wer"] is not None]
        by_lang[lang] = {
            "cases": len(g),
            "mean_cer": round(sum(i["cer"] for i in g) / len(g), 4),
            "mean_similarity": round(sum(i["similarity"] for i in g) / len(g), 4),
            "suspected_truncation_rate": round(sum(i["suspected_truncation"] for i in g) / len(g), 4),
            **({"mean_wer": round(sum(wers) / len(wers), 4)} if wers else {}),
        }
    report = {
        "target": target, "base": infer._resolve_base(base), "label": label,
        "cfg_value": cfg_value, "inference_timesteps": inference_timesteps,
        "retry_badcase": False, "asr_model": "large-v3",
        "asr_auto_detect_langs": sorted(AUTO_DETECT_LANGS),
        "mean_similarity": round(sum(i["similarity"] for i in items) / len(items), 4),
        "mean_cer": round(sum(i["cer"] for i in items) / len(items), 4),
        "suspected_truncation_rate": round(sum(i["suspected_truncation"] for i in items) / len(items), 4),
        "mean_f0_std": round(sum(i["f0_std_st"] for i in items) / len(items), 2),
        "by_lang": by_lang,
        "note": "ASR/漏尾均为诊断；F0 不作通过门限。按语言、ref 语言、角色、情绪分组做母语盲听。"
                "rate 的量纲随语种不同（th 字符/秒、vi 音节/秒、tl/en/id/ms 词/秒），不横向比。",
        "items": items,
    }
    out_dir = EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_{uuid4().hex}.json"
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


def save_reviews(report_a: str, report_b: str, session: list[dict],
                 ratings: dict | None) -> dict:
    """把人工评分写回两份报告的 human_review，返回分语种 A/B 汇总。

    ratings: {review_key(pair): {"s1": {...}, "s2": {...}}}，字段名取 _REVIEW_FIELDS。
    汇总的 win/tie/loss 以自然度比较 B 相对 A（B 一般是 checkpoint）。
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
    return {"pairs": len(session), "rated": rated,
            "by_lang": {k: {"mean_naturalness_a": _mean(v["a"]),
                            "mean_naturalness_b": _mean(v["b"]), **verdict[k]}
                        for k, v in sorted(nat_by_lang.items())},
            "written": [report_a, report_b]}


def print_compare(reports: list[dict]) -> None:
    print(f"{'checkpoint':<44} {'CER↓':>8} {'疑似漏尾':>8} {'F0(描述)':>9}")
    for r in reports:
        print(f"{r['target']:<44} {r['mean_cer']:>8} "
              f"{r['suspected_truncation_rate']:>8} {r['mean_f0_std']:>9}")
        if len(r.get("by_lang", {})) > 1:
            for lang, s in r["by_lang"].items():
                print(f"  {lang:<42} {s['mean_cer']:>8} "
                      f"{s['suspected_truncation_rate']:>8}")
    print("\n不能凭以上指标自动通过；请做母语盲听，检查情绪、音色、自然度和真实截断。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+", help='"base" 或若干 LoRA checkpoint 目录')
    ap.add_argument("--lang", default="th", choices=list(SAMPLE_BY_LANG))
    ap.add_argument("--base", default=None)
    ap.add_argument("--texts-file", default=None, help="逐 case JSONL；相对 ref 路径按此文件目录解析")
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, help="覆盖每个 case 的 seed，推荐 42 43 44")
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--inference-timesteps", type=int, default=20)
    args = ap.parse_args()
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
                        inference_timesteps=args.inference_timesteps) for t in args.targets]
    print_compare(reports)
    print(f"\n详细报告: {CHECKPOINT_DIR / 'eval'}/")


if __name__ == "__main__":
    main()
