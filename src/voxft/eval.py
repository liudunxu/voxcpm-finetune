"""微调效果客观评估：批量合成 → Whisper 转写 → 贴合度/截断/韵律报告。

三个指标对应三类线上反馈：
- mean_similarity  文本贴合度（发音错、念错词）
- truncation_rate  截断率（"词被吃掉""听不清他说了什么"）
- mean_f0_std      语调起伏，半音（"robotic""没有情绪"）——越高越有起伏

用法（需 --group qc 安装 faster-whisper）：
    uv run python -m voxft.eval base checkpoints/lora_xxx/latest --lang th
    uv run python -m voxft.eval checkpoints/a --lang tl --texts-file my_texts.jsonl
    uv run python -m voxft.eval base checkpoints/a --lang tl --control "愤怒地，语速快"
"""
from __future__ import annotations

import argparse
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

from . import infer
from .paths import CHECKPOINT_DIR

SAMPLE_BY_LANG = {
    "th": infer.SAMPLE_TEXTS["泰语"],
    "tl": infer.SAMPLE_TEXTS["Tagalog"],
    "zh": infer.SAMPLE_TEXTS["中文"],
}


def _transcribe(model, wav_path: str, lang: str) -> str:
    segs, _ = model.transcribe(wav_path, language=lang, vad_filter=True)
    return "".join(s.text for s in segs)


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _is_truncated(hyp: str, ref: str, tail: int = 8) -> bool:
    """判断生成是否被截断：整体明显变短，或结尾对不上参考文本的结尾。

    线上反馈里的 "word was cut" / "cannot understand what she said" 多是这一类，
    单看平均相似度会被前半句的高分冲淡，所以单独统计。
    """
    h, r = _norm(hyp), _norm(ref)
    if not r:
        return False
    if len(h) < 0.6 * len(r):
        return True
    return SequenceMatcher(None, h[-tail:], r[-tail:]).ratio() < 0.5


def _prosody(wav_path: str) -> dict:
    from .data.pipeline import audio_metrics, load_wav_mono
    wav, sr = load_wav_mono(wav_path)
    return audio_metrics(wav, sr, "")


def evaluate(target: str, lang: str, texts: list[str],
             base: str | None = None, ref_audio: str | None = None,
             control: str | None = None, seed: int = 42) -> dict:
    """target: "base"（纯基座）或 LoRA checkpoint 目录。

    固定 seed + 固定参考音频，否则两次评测之间的差异分不清是模型还是随机性。
    """
    from faster_whisper import WhisperModel

    whisper = WhisperModel("medium", device="auto", compute_type="auto")
    lora = None if target == "base" else target
    label = Path(target).parent.name + "_" + Path(target).name if lora else "base"
    items = []
    for text in texts:
        t0 = time.time()
        wav_path, _ = infer.synthesize(text, base, lora, ref_audio=ref_audio,
                                       seed=seed, control=control)
        hyp = _transcribe(whisper, wav_path, lang)
        sim = SequenceMatcher(
            None, hyp.lower().replace(" ", ""), text.lower().replace(" ", "")
        ).ratio()
        items.append({"text": text, "hyp": hyp, "similarity": round(sim, 3),
                      "truncated": _is_truncated(hyp, text),
                      **_prosody(wav_path),
                      "wav": wav_path, "gen_sec": round(time.time() - t0, 1)})
    report = {
        "target": target, "label": label, "lang": lang, "control": control or "",
        "mean_similarity": round(sum(i["similarity"] for i in items) / len(items), 3),
        "truncation_rate": round(sum(i["truncated"] for i in items) / len(items), 3),
        "mean_f0_std": round(sum(i["f0_std_st"] for i in items) / len(items), 2),
        "items": items,
    }
    out_dir = CHECKPOINT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_{lang}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_compare(reports: list[dict]) -> None:
    print(f"{'checkpoint':<44} {'贴合度':>8} {'截断率':>8} {'语调起伏':>9}")
    print("-" * 72)
    for r in sorted(reports, key=lambda x: -x["mean_similarity"]):
        print(f"{r['target']:<44} {r['mean_similarity']:>8} "
              f"{r['truncation_rate']:>8} {r['mean_f0_std']:>9}")
    print("\n贴合度↑ 截断率↓ 语调起伏↑ 才算通过；语调起伏只在同一参考音频下可比")


def main() -> None:
    ap = argparse.ArgumentParser(description="微调效果客观评估（Whisper 文本贴合度）")
    ap.add_argument("targets", nargs="+",
                    help='"base" 或若干 LoRA checkpoint 目录')
    ap.add_argument("--lang", default="th", choices=list(SAMPLE_BY_LANG))
    ap.add_argument("--base", default=None, help="基座目录（默认 env/官方仓库）")
    ap.add_argument("--texts-file", default=None,
                    help='JSONL，每行 {"text": "..."}；默认用内置示例文本')
    ap.add_argument("--ref-audio", default=None,
                    help="参考音频（固定同一条，否则各 checkpoint 不可比）")
    ap.add_argument("--control", default=None,
                    help='情绪/语气 prompt，如 "愤怒地，语速快" / "sad, slow"')
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.texts_file:
        texts = [json.loads(l)["text"] for l in
                 Path(args.texts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        texts = [SAMPLE_BY_LANG[args.lang]]
    reports = [evaluate(t, args.lang, texts, args.base, args.ref_audio,
                        args.control, args.seed) for t in args.targets]
    print_compare(reports)
    print(f"\n详细报告: {CHECKPOINT_DIR / 'eval'}/")


if __name__ == "__main__":
    main()
