import importlib.util
from pathlib import Path

import pytest


def test_coverage_uses_duration_and_reproducible_nondigit_review(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("training_coverage", scripts / "training_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    short = {"duration": 4, "origin_audio": "/original.wav", "text": "Hello!", "audio": "/short.wav",
             "source_id": "fleurs_tl", "lang": "tl"}
    long = {**short, "duration": 12, "origin_audio": "/long.wav", "audio": "/long.wav"}
    coverage = module.describe([short, short, long])
    assert coverage["short_duration_share"] == pytest.approx(0.4)
    assert coverage["short_records"] == 2 and coverage["max_origin_exposure"] == 2
    assert coverage["unique_origins"] == 2 and coverage["unique_texts"] == 1
    assert module.describe([])["short_duration_share"] is None
    for invalid in (float("nan"), 2.9, 30.1):
        with pytest.raises(ValueError, match="duration"):
            module.describe([{**short, "duration": invalid}])
    raw = {("fleurs_tl", "/original.wav"): {"text": "Hello!", "lang": "tl"}}
    counts = module.provenance([short, {**short, "text": "HELLO"}, long,
                               {**short, "text": "Different"}, {**short, "lang": "ms"}], raw)["counts"]
    assert counts == {"exact_text_match": 1, "normalization_only_difference": 1,
                      "missing_raw_origin": 1, "raw_text_mismatch": 1, "raw_language_mismatch": 1}
    rows = [{**short, "lang": lang, "duration": duration, "audio": f"/{lang}/{duration}/{index}.wav"}
            for lang in ("tl", "ms") for duration in (3, 5, 6, 8, 9, 30) for index in range(6)]
    sample = module.review_sample(rows)
    assert len(sample) == len({row["audio"] for row in sample}) == 30
    assert sample == module.review_sample(list(reversed(rows)))
    assert sample == module.review_sample(rows + [{**short, "text": "123"}])
    assert {stratum: sum(row["review_stratum"] == stratum for row in sample)
            for stratum in {row["review_stratum"] for row in sample}} == {
                f"{lang}/{bounds}s": 5 for lang in ("tl", "ms") for bounds in ("3-5", "5-8", "8-30")}
    with pytest.raises(ValueError, match="Insufficient"):
        module.review_sample([])
