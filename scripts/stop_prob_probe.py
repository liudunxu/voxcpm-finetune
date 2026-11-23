"""Zero-training stop-probability diagnostic (docs/tts_finetune_deepdive_20261123.md §5 item 2).

VoxCPM2 stops when stop_head(...).argmax() == 1, i.e. a fixed p_stop > 0.5
(third_party/VoxCPM/src/voxcpm/model/voxcpm2.py:1113), with
max_len = text_tokens * retry_badcase_ratio_threshold + 10 patches (:654).
LoRA moves the LM hidden state but the stop head stays frozen. This probe logs p_stop
per generated patch via a forward hook (the submodule stays untouched) to tell apart:
  never_rises  - hit the patch cap and p_stop never reached LOW
  missed_peak  - hit the cap although p_stop reached LOW..0.5 after min_len
  late_stop    - stopped, but p_stop first reached LOW >= LATE patches before the stop
  clean_stop   - stopped without an earlier near-miss
Only a pattern that differs between base and r50 on severe outputs justifies a
threshold / length-prior A/B. Production defaults are not changed by this script.

  select: python scripts/stop_prob_probe.py select --reports-dir work/prod_review_20261103/reports \
            --model r50 --out cases.jsonl
  run:    python -X utf8 -u scripts/stop_prob_probe.py run --cases cases.jsonl --label r50 \
            --lora <r50 lora dir> --out r50_trace.jsonl      (repeat with --label base, no --lora)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prod_review_20261103 import load_items, severe_flags  # noqa: E402

LOW = 0.3
LATE = 3  # patches


def classify(trace: list[float], min_len: int = 2, low: float = LOW, late: int = LATE) -> dict:
    """trace[i] = p_stop at loop step i; generation stopped at the last step iff p_stop > 0.5 there."""
    if not trace:
        raise ValueError("empty p_stop trace")
    stopped = trace[-1] > 0.5 and len(trace) - 1 > min_len
    eligible = [i for i, p in enumerate(trace) if i > min_len]
    first_low = next((i for i in eligible if trace[i] >= low), None)
    peak = max((trace[i] for i in eligible), default=0.0)
    if stopped:
        label = "late_stop" if first_low is not None and len(trace) - 1 - first_low >= late else "clean_stop"
    else:
        label = "missed_peak" if peak >= low else "never_rises"
    return {"steps": len(trace), "stopped": stopped, "peak_p_stop": round(peak, 4),
            "first_low_step": first_low, "stop_pattern": label}


def select(reports_dir: Path, model: str, out: Path, controls_per_severe: int = 1) -> int:
    reports = {name: load_items(reports_dir / f"short_{name}.json") for name in ("base", model)}
    flags = severe_flags(reports, "short")
    items = reports[model]
    severe = sorted(k for k, f in flags[model].items() if f["severe"])
    clean = sorted(k for k, f in flags[model].items() if not f["severe"] and not flags["base"][k]["severe"])
    picked, seen = [], set()
    for key in severe:
        # control: a clean output of the same sentence (different ref arm or seed), so text is held fixed
        sid = items[key]["sentence_id"]
        same = [k for k in clean if items[k]["sentence_id"] == sid and k not in seen][:controls_per_severe]
        for k, role in [(key, "severe"), *[(c, "control") for c in same]]:
            seen.add(k)
            it = items[k]
            picked.append({f: it[f] for f in ("case_id", "sentence_id", "lang", "text", "ref_audio",
                                               "ref_arm", "cfg_value", "inference_timesteps", "seed")}
                          | {"role": role, "base_severe": flags["base"][k]["severe"],
                             "source_audio_sec": it.get("audio_sec")})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    return len(picked)


def run(cases: Path, label: str, lora: str | None, out: Path, base: str | None = None) -> None:
    import soundfile as sf
    import torch
    from voxft import infer

    model = infer.get_model(base, lora)
    tts = model.tts_model
    trace: list[float] = []
    hook = tts.stop_head.register_forward_hook(
        lambda _m, _i, o: trace.append(float(torch.softmax(o.float(), dim=-1)[0, 1])))
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] + f"/{json.loads(line)['seed']}" for line in out.open(encoding="utf-8")}
    wav_dir = out.parent / f"{out.stem}_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("a", encoding="utf-8") as f:
            for line in cases.open(encoding="utf-8"):
                case = json.loads(line)
                key = f"{case['case_id']}/{case['seed']}"
                if key in done:
                    continue
                kw = infer._gen_kwargs(case["text"], case["ref_audio"], None, case["cfg_value"],
                                       case["inference_timesteps"], case["seed"])
                kw["retry_badcase"] = False  # same as production / prod_review: single attempt
                trace.clear()
                wav = model.generate(**kw)
                wav_path = wav_dir / f"{case['case_id']}_{case['seed']}.wav"
                sf.write(wav_path, wav, tts.sample_rate)
                text_tokens = len(tts.text_tokenizer(case["text"]))
                f.write(json.dumps(case | classify(trace) | {
                    "label": label, "lora": lora, "p_stop": [round(p, 4) for p in trace],
                    "max_len": int(text_tokens * kw["retry_badcase_ratio_threshold"] + 10),
                    "audio_sec": round(len(wav) / tts.sample_rate, 3), "wav": str(wav_path),
                }, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        hook.remove()


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--reports-dir", type=Path, required=True)
    s.add_argument("--model", default="r50")
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--lora")
    r.add_argument("--base")
    r.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "select":
        print(f"{select(a.reports_dir, a.model, a.out)} cases -> {a.out}")
    else:
        run(a.cases, a.label, a.lora, a.out, a.base)


if __name__ == "__main__":
    main()
