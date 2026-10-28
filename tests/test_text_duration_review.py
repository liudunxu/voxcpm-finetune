import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxft.data import pipeline


def sample(**changes):
    return {"audio": "/processed/example.wav", "origin_audio": "/raw/example.wav",
            "text": "Aku tidak tahu.", "duration": 5.0, "lang": "id", **changes}


def approve(row):
    review = pipeline.text_duration_review(row)
    assert review is not None
    return {**row, "text_duration_review": {
        "approved": True, "key": review["review_key"], "note": "Confirmed deliberate slow speech."}}


def test_sparse_text_is_reviewable_not_a_confirmed_audio_failure():
    row = sample()
    original = dict(row)
    review = pipeline.text_duration_review(row)
    assert review["reason"] == "text_duration_review_pending"
    assert review["word_count"] == 3
    assert review["words_per_second"] == 0.6
    assert row == original
    accepted = approve(row)
    assert pipeline.text_duration_review(accepted) is None
    for changed in ({"text": "Aku tahu."}, {"duration": 6.0}, {"origin_audio": "/other.wav"}):
        assert pipeline.text_duration_review({**accepted, **changed}) is not None
    for approval in (True, "true", {"approved": True},
                     {**accepted["text_duration_review"], "note": ""}):
        assert pipeline.text_duration_review({**row, "text_duration_review": approval}) is not None


@pytest.mark.parametrize("row", [
    sample(duration=3), sample(text="Aku sama sekali tidak tahu tentang hal itu."),
    sample(lang="th", text="ฉันไม่รู้เรื่องนี้เลย"),
    sample(lang="zh", text="我完全不知道这件事"),
    sample(lang="id", text="我完全不知道"),
    sample(reference_only=True, text=""), sample(reference_only=True), sample(duration=float("nan")),
])
def test_normal_or_unmeasurable_rows_are_not_word_rate_rejected(row):
    assert pipeline.text_duration_review(row) is None


def test_control_prefix_does_not_hide_sparse_target():
    row = sample()
    controlled = {**row, "text": "(with clear natural delivery)" + row["text"],
                  "control": "with clear natural delivery"}
    assert pipeline.text_duration_review(controlled) == pipeline.text_duration_review(row)


def test_all_candidate_gates_use_the_same_pending_review(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    incremental = importlib.import_module("incremental_mix")
    pool = importlib.import_module("training_pool")
    row = sample()
    assert incremental.row_issues(row, set(), set()) == ["text_duration_review_pending"]
    audit = incremental.audit_mix([row], [], set())
    assert len(audit["violations"]) == 1
    assert audit["violations"][0]["text_duration_diagnostic"] == pipeline.text_duration_review(row)
    assert pool.metadata_rejection(row, set(), set()) == "text_duration_review_pending"
    kept, rejected = incremental.eligible_additions([row], [], [], set())
    assert kept == [] and rejected == {"text_duration_review_pending": 1}
    accepted = approve(row)
    assert incremental.row_issues(accepted, set(), set()) == []
    assert pool.metadata_rejection(accepted, set(), set()) is None


def test_pipeline_quarantines_after_trim_without_extra_asr(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA_RAW", tmp_path / "raw")
    monkeypatch.setattr(pipeline, "DATA_PROCESSED", tmp_path / "processed")
    source = tmp_path / "raw" / "drama_id"
    source.mkdir(parents=True)
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(80000) / 16000)).astype(np.float32)
    rows = []
    for name, text in (("sparse", "Aku tidak tahu."), ("normal", "Aku sama sekali tidak tahu tentang hal itu.")):
        path = source / f"{name}.wav"
        sf.write(path, tone, 16000)
        rows.append({"audio": str(path), "text": text, "lang": "id"})
    manifest = source / "manifest.jsonl"
    pipeline._write_jsonl(rows, manifest)
    before = manifest.read_bytes()
    monkeypatch.setattr(pipeline, "_whisper_model", lambda *args: pytest.fail("extra ASR"))
    result = pipeline.process_dataset("drama_id", opts=pipeline.Options(metrics=False, val_ratio=0))
    assert result["kept"] == 1 and result["review_text_duration"] == 1
    directory = tmp_path / "processed" / "drama_id"
    reviews = pipeline._read_manifest(directory / "text_duration_review.jsonl")
    assert len(reviews) == 1 and reviews[0]["text"] == "Aku tidak tahu."
    assert Path(reviews[0]["audio"]).is_file()
    assert reviews[0]["text_duration_diagnostic"]["word_count"] == 3
    assert manifest.read_bytes() == before
    assert {row["text"] for row in pipeline._read_manifest(directory / "train.jsonl")} == {rows[1]["text"]}


def test_mix_cannot_bypass_review_using_old_processed_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA_PROCESSED", tmp_path)
    original = sample()
    pipeline._write_jsonl([original], tmp_path / "source" / "train.jsonl")
    pipeline._write_jsonl([], tmp_path / "source" / "val.jsonl")
    frozen = (tmp_path / "source" / "train.jsonl").read_bytes()
    with pytest.raises(ValueError, match="text_duration_review"):
        pipeline.mix_manifests([("source", 1)], "new")
    assert not (tmp_path / "new" / "train.jsonl").exists()
    reviews = pipeline._read_manifest(tmp_path / "new" / "text_duration_review.jsonl")
    assert len(reviews) == 1 and reviews[0]["text"] == original["text"]
    assert (tmp_path / "source" / "train.jsonl").read_bytes() == frozen
    accepted = approve(original)
    pipeline._write_jsonl([accepted], tmp_path / "source" / "train.jsonl")
    pipeline.mix_manifests([("source", 1)], "new")
    assert pipeline._read_manifest(tmp_path / "new" / "train.jsonl") == [accepted]
    assert pipeline._read_manifest(tmp_path / "new" / "text_duration_review.jsonl") == []


def test_mix_does_not_review_rows_already_excluded_by_duration(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "DATA_PROCESSED", tmp_path)
    normal = sample(audio="/normal.wav", origin_audio="/normal.wav", duration=4.0)
    pipeline._write_jsonl([sample(duration=25.0), normal], tmp_path / "source" / "train.jsonl")
    pipeline._write_jsonl([], tmp_path / "source" / "val.jsonl")
    pipeline.mix_manifests([("source", 1)], "new", max_sample_dur=20)
    assert pipeline._read_manifest(tmp_path / "new" / "train.jsonl") == [normal]


def test_training_preflight_cannot_bypass_review_or_rewrite_validation(monkeypatch, tmp_path):
    from voxft.train import launcher

    monkeypatch.setattr(launcher, "TRAIN_SCRIPT", Path(__file__))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "target.wav"
    target.write_bytes(b"audio")
    validation_audio = tmp_path / "validation.wav"
    validation_audio.write_bytes(b"validation")
    row = sample(audio=str(target), origin_audio=str(target))
    train, validation = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    pipeline._write_jsonl([row], train)
    pipeline._write_jsonl([sample(audio=str(validation_audio), origin_audio=str(validation_audio))], validation)
    frozen = validation.read_bytes()
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"pretrained_path": str(base), "train_manifest": str(train),
                                  "val_manifest": str(validation), "save_path": str(tmp_path / "ckpt")}),
                      encoding="utf-8")
    issues = launcher.preflight(config)
    review = pipeline.text_duration_review(row)
    assert any(review["reason"] in issue and review["review_key"] in issue for issue in issues)
    pipeline._write_jsonl([approve(row)], train)
    assert all(issue.startswith("警告") for issue in launcher.preflight(config))
    assert validation.read_bytes() == frozen
