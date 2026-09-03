"""微调效果客观评估：批量合成 → Whisper 转写 → 文本贴合度报告。

用法（需 --group qc 安装 faster-whisper）：
    uv run python -m voxft.eval base checkpoints/lora_xxx/latest --lang th
    uv run python -m voxft.eval checkpoints/a --lang tl --texts-file my_texts.jsonl
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


def evaluate(target: str, lang: str, texts: list[str],
             base: str | None = None) -> dict:
    """target: "base"（纯基座）或 LoRA checkpoint 目录。"""
    from faster_whisper import WhisperModel

    whisper = WhisperModel("medium", device="auto", compute_type="auto")
    lora = None if target == "base" else target
    label = Path(target).parent.name + "_" + Path(target).name if lora else "base"
    items = []
    for text in texts:
        t0 = time.time()
        wav_path, _ = infer.synthesize(text, base, lora)
        hyp = _transcribe(whisper, wav_path, lang)
        sim = SequenceMatcher(
            None, hyp.lower().replace(" ", ""), text.lower().replace(" ", "")
        ).ratio()
        items.append({"text": text, "hyp": hyp, "similarity": round(sim, 3),
                      "wav": wav_path, "gen_sec": round(time.time() - t0, 1)})
    report = {
        "target": target, "label": label, "lang": lang,
        "mean_similarity": round(sum(i["similarity"] for i in items) / len(items), 3),
        "items": items,
    }
    out_dir = CHECKPOINT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_{lang}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_compare(reports: list[dict]) -> None:
    print(f"{'checkpoint':<50} {'平均文本贴合度':>10}")
    print("-" * 64)
    for r in sorted(reports, key=lambda x: -x["mean_similarity"]):
        print(f"{r['target']:<50} {r['mean_similarity']:>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description="微调效果客观评估（Whisper 文本贴合度）")
    ap.add_argument("targets", nargs="+",
                    help='"base" 或若干 LoRA checkpoint 目录')
    ap.add_argument("--lang", default="th", choices=list(SAMPLE_BY_LANG))
    ap.add_argument("--base", default=None, help="基座目录（默认 env/官方仓库）")
    ap.add_argument("--texts-file", default=None,
                    help='JSONL，每行 {"text": "..."}；默认用内置示例文本')
    args = ap.parse_args()

    if args.texts_file:
        texts = [json.loads(l)["text"] for l in
                 Path(args.texts_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        texts = [SAMPLE_BY_LANG[args.lang]]
    reports = [evaluate(t, args.lang, texts, args.base) for t in args.targets]
    print_compare(reports)
    print(f"\n详细报告: {CHECKPOINT_DIR / 'eval'}/")


if __name__ == "__main__":
    main()
