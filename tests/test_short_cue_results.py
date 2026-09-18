from copy import deepcopy
from pathlib import Path

import pytest


def test_sentence_cluster_weighting_and_guards(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from short_cue_results import sentence_summary

    report = {"cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
              "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
              "items": [{"case_id": f"{sentence}_{reference}", "source_sentence_id": sentence,
                         "text": f"Sentence {sentence}", "lang": "tl", "numeric": False,
                         "ref_audio": f"/{reference}.wav", "seed": seed, "cer": 0.0,
                         "audio_sec": 2.0}
                        for sentence, references in (("first", range(3)), ("second", range(1)))
                        for reference in references for seed in (42, 43, 44, 45, 49)]}
    changed = deepcopy(report)
    for row in changed["items"]:
        row["cer"] = 0.2 if row["source_sentence_id"] == "first" else 0.0
    summary = sentence_summary(report, changed)
    metric = summary["by_lang"]["tl"]["cer"]
    assert metric["sentences"] == metric["total_sentences"] == 2
    assert metric["delta_b_minus_a"] == pytest.approx(0.1)
    assert metric["sentence_bootstrap_95"] == pytest.approx([0.0, 0.2])
    assert summary == sentence_summary(report, {**changed, "items": list(reversed(changed["items"]))})
    assert summary["by_lang"]["tl"]["speaker_sim"]["sentences"] == 0
    missing = deepcopy(changed)
    missing["items"][0]["audio_sec"] = None
    incomplete = sentence_summary(report, missing)["by_lang"]["tl"]
    assert incomplete["cer"]["sentences"] == incomplete["audio_sec"]["sentences"] == 1
    assert incomplete["cer"]["missing_sentence_ids"] == ["first"]
    assert incomplete["cer"]["sentence_bootstrap_95"] is None
    for field, value in (("source_sentence_id", "changed"), ("source_sentence_id", None),
                         ("text", "changed"), ("numeric", True)):
        invalid = deepcopy(changed)
        invalid["items"][0][field] = value
        with pytest.raises(ValueError):
            sentence_summary(report, invalid)
    with pytest.raises(ValueError):
        sentence_summary(report, {**changed, "items": changed["items"][:-1]})
