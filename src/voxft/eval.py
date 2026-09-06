"""微调离线验收：逐 case 固定 ref/control/seed，ASR 诊断 + 母语盲听。

ASR 误差不等于发音错误，疑似漏尾不等于真实截断，F0 不是越高越好。
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from . import infer
from .paths import CHECKPOINT_DIR

SAMPLE_BY_LANG = {
    "th": infer.SAMPLE_TEXTS["泰语"], "tl": infer.SAMPLE_TEXTS["Tagalog"],
    "zh": infer.SAMPLE_TEXTS["中文"], "en": "Are you okay? I was worried about you.",
}


def _transcribe(model, wav_path: str, lang: str) -> str:
    # Taglish 可被识别为英语；不强制单一语言解码。
    segs, _ = model.transcribe(wav_path, language=None if lang == "tl" else lang,
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


def _prosody(wav_path: str) -> dict:
    from .data.pipeline import audio_metrics, load_wav_mono
    wav, sr = load_wav_mono(wav_path)
    return audio_metrics(wav, sr, "")


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
                       if case["lang"] in ("tl", "en") else None,
                "suspected_truncation": _is_truncated(hyp, case["text"]),
                **_prosody(wav_path), "wav": wav_path, "gen_sec": gen_sec,
                "human_review": {"naturalness_1_5": None, "emotion_fit_1_5": None,
                                 "speaker_similarity_1_5": None, "intelligibility_1_5": None,
                                 "cutoff": None, "noise": None, "notes": ""},
            })
    report = {
        "target": target, "base": infer._resolve_base(base), "label": label,
        "cfg_value": cfg_value, "inference_timesteps": inference_timesteps,
        "retry_badcase": False, "asr_model": "large-v3", "asr_tl_language": "auto",
        "mean_similarity": round(sum(i["similarity"] for i in items) / len(items), 4),
        "mean_cer": round(sum(i["cer"] for i in items) / len(items), 4),
        "suspected_truncation_rate": round(sum(i["suspected_truncation"] for i in items) / len(items), 4),
        "mean_f0_std": round(sum(i["f0_std_st"] for i in items) / len(items), 2),
        "note": "ASR/漏尾均为诊断；F0 不作通过门限。按语言、ref 语言、角色、情绪分组做母语盲听。",
        "items": items,
    }
    out_dir = CHECKPOINT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_{uuid4().hex}.json"
    report["report_path"] = str(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_compare(reports: list[dict]) -> None:
    print(f"{'checkpoint':<44} {'CER↓':>8} {'疑似漏尾':>8} {'F0(描述)':>9}")
    for r in reports:
        print(f"{r['target']:<44} {r['mean_cer']:>8} "
              f"{r['suspected_truncation_rate']:>8} {r['mean_f0_std']:>9}")
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
