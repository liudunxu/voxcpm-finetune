from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stop_prob_probe import classify, select


def test_classify_patterns():
    assert classify([0.0, 0.0, 0.0, 0.1, 0.9])["stop_pattern"] == "clean_stop"
    assert classify([0.0, 0.0, 0.0, 0.35, 0.1, 0.1, 0.2, 0.8])["stop_pattern"] == "late_stop"
    assert classify([0.0, 0.0, 0.0, 0.45, 0.1, 0.1])["stop_pattern"] == "missed_peak"
    assert classify([0.0, 0.0, 0.0, 0.1, 0.1])["stop_pattern"] == "never_rises"
    # steps <= min_len never count: argmax there is ignored by the generator
    assert classify([0.9, 0.9, 0.9, 0.1])["stop_pattern"] == "never_rises"
    with pytest.raises(ValueError):
        classify([])


def _report(path, rows):
    path.write_text(json.dumps({"items": rows}), encoding="utf-8")


def test_select_pairs_severe_with_same_sentence_control(tmp_path):
    def row(cid, seed, hyp, sec):
        return {"case_id": cid, "sentence_id": cid.split("_ref")[0], "lang": "tl", "text": "Sige!",
                "ref_audio": "r.wav", "ref_arm": "zh", "cfg_value": 1.8, "inference_timesteps": 24,
                "seed": seed, "hyp": hyp, "cer": 0.0, "audio_sec": sec}
    base = [row("s1_refa", 42, "sige", 0.8), row("s1_refb", 42, "sige", 0.8)]
    cand = [row("s1_refa", 42, "", 0.8), row("s1_refb", 42, "sige", 0.8)]
    _report(tmp_path / "short_base.json", base)
    _report(tmp_path / "short_r50.json", cand)
    out = tmp_path / "cases.jsonl"
    assert select(tmp_path, "r50", out) == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["role"] for r in rows] == ["severe", "control"]
    assert rows[1]["case_id"] == "s1_refb" and rows[1]["base_severe"] is False


def test_summarize_frozen_rule(tmp_path):
    from stop_prob_probe import summarize

    def rows(label, patterns):
        return [{"label": label, "role": "severe", "lang": "th", "audio_sec": 2.0, "source_audio_sec": 2.0,
                 "stop_pattern": p} for p in patterns] + [
            {"label": label, "role": "severe", "lang": "th", "audio_sec": 5.0, "source_audio_sec": 2.0,
             "stop_pattern": "never_rises"}]  # not reproduced: ignored
    for label, pats in (("base", ["missed_peak", "clean_stop", "clean_stop"]),
                        ("r50", ["missed_peak", "late_stop", "clean_stop"])):
        (tmp_path / f"{label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows(label, pats)))
    out = summarize([tmp_path / "base.jsonl", tmp_path / "r50.jsonl"])
    assert out["base"]["severe_verdict"] == "not_stop_timing"
    assert out["r50"]["severe"]["reproduced"] == 3 and out["r50"]["severe_verdict"] == "near_miss"
    assert out["threshold_ab_justified"] is True
