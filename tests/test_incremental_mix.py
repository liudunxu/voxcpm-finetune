import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("incremental_mix")


def row(name="target", **fields):
    return {"audio": f"/audio/{name}.wav", "origin_audio": f"/raw/{name}.wav",
            "text": name, "duration": 4.0, "lang": "th", "source_id": "cv22_th",
            **fields}


def test_guards_text_identity_origin_duration_and_provenance(module):
    validation = row("validation", speaker_verified=True, speaker="held-out", session="studio")
    candidates = [
        row("safe"), row("long", duration=20.01), row("nan", duration=float("nan")),
        row("different-path", origin_audio="/raw/used.wav"),
        row("prefix", text="(angry)Frozen sentence!", control="angry"),
        row("speaker", speaker_verified=True, speaker="held-out"),
        row("session", source_id="cv22_th_r23boost", session="studio"),
        row("ref", speaker_verified=True, speaker="new", ref_speaker="new",
            ref_audio=validation["audio"], ref_origin_audio=validation["origin_audio"]),
    ]
    original = [dict(candidate) for candidate in candidates]
    kept, rejected = module.eligible_additions(candidates, [row("used")], [validation],
                                               {"frozensentence"})
    assert kept == [row("safe")]
    assert rejected["duration_outside_3_20s"] == 2
    assert rejected["validation_audio_speaker_or_session"] == 3
    assert rejected["evaluation_or_validation_text"] == 1
    assert rejected["existing_or_duplicate_target"] == 1
    assert candidates == original
    module.validate_mix(kept, [validation], {"frozensentence"})
    with pytest.raises(ValueError, match="Unsafe"):
        module.validate_mix([row("repeat")] * 4, [validation], set())
    with pytest.raises(ValueError, match="Unsafe"):
        module.validate_mix([row("validation")], [validation], set())


def test_reference_requires_verified_different_origin(module):
    invalid = row(ref_audio="/ref.wav", ref_origin_audio="/raw/target.wav")
    assert "unverified_or_same_origin_reference" in module.row_issues(invalid, set(), set())
    valid = row(speaker_verified=True, speaker="actor", ref_speaker="actor",
                ref_audio="/ref.wav", ref_origin_audio="/raw/ref.wav")
    assert module.row_issues(valid, set(), set()) == []
    aliased = {**valid, "ref_origin_audio": "/raw/./target.wav"}
    assert "unverified_or_same_origin_reference" in module.row_issues(aliased, set(), set())
    summary, shares = module.describe_mix([valid, row("id", lang="id", duration=8)])
    assert summary["with_ref_audio"] == 1
    assert shares["id"]["actual"] == pytest.approx(2 / 3)
    assert shares["id"]["requested"] is None


def test_audit_preserves_inputs_and_requires_exclusion_files(module, tmp_path):
    eval_dir = tmp_path / "eval_cases"
    eval_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        module.load_exclusions(eval_dir)
    for name in module.REQUIRED_EVAL_FILES:
        (eval_dir / name).write_text(json.dumps({"text": "held out"}) + "\n", encoding="utf-8")
    for split, sample in (("train", row("train")), ("val", row("val"))):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
    result = module.audit_directory(tmp_path, eval_dir)
    assert result["violation_counts"] == {}
    assert len(result["inputs_sha256"]) == len(before)
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("script", ["build_r22_idboost", "build_r23_thtlboost"])
def test_builders_refuse_frozen_output_before_reading_inputs(module, monkeypatch, tmp_path, script):
    builder = importlib.import_module(script)
    monkeypatch.setattr(builder, "DST", tmp_path)
    sentinel = tmp_path / "train.jsonl"
    sentinel.write_bytes(b"frozen")
    with pytest.raises(FileExistsError, match="Frozen"):
        builder.main()
    assert sentinel.read_bytes() == b"frozen"


@pytest.mark.parametrize("script", ["build_r22_idboost", "build_r23_thtlboost"])
def test_builders_keep_provenance_and_actual_ref_summary(module, monkeypatch, tmp_path, script):
    builder = importlib.import_module(script)
    parent = tmp_path / "parent"
    source = tmp_path / "cv22_th"
    parent.mkdir()
    source.mkdir()
    for directory, split, rows in (
        (parent, "train", [row("old", speaker_verified=None, speaker=None, ref_speaker=None,
                               ref_audio=None, ref_origin_audio=None, ref_duration=None)]),
        (parent, "val", [row("validation")]),
        (source, "train", [
            row("bad", text="excluded"), row("long", duration=25),
            row("good", speaker_verified=True, speaker="actor", ref_speaker="actor",
                ref_audio="/ref.wav", ref_origin_audio="/raw/ref.wav", ref_duration=4),
        ]),
    ):
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")
    if script == "build_r22_idboost":
        monkeypatch.setattr(builder, "R21", parent)
        monkeypatch.setattr(builder, "GIG", source)
        monkeypatch.setattr(builder, "ADD", 1)
    else:
        monkeypatch.setattr(builder, "R22", parent)
        monkeypatch.setattr(builder, "PROC", tmp_path)
        monkeypatch.setattr(builder, "PLAN", [("cv22_th", 1)])
    destination = tmp_path / "new"
    monkeypatch.setattr(builder, "DST", destination)
    monkeypatch.setattr(builder, "load_exclusions", lambda: ({"excluded"}, {}))
    builder.main()
    mixed = builder.load(destination / "train.jsonl")
    assert len(mixed) == 2 and mixed[-1]["source_id"] == "cv22_th"
    assert mixed[-1]["ref_audio"] == "/ref.wav"
    assert (destination / "val.jsonl").read_bytes() == (parent / "val.jsonl").read_bytes()
    summary = json.loads((destination / "mix.json").read_text(encoding="utf-8"))
    assert summary["train"]["with_ref_audio"] == 1
    assert summary["metadata_audit"]["violation_counts"] == {}


def test_new_identity_metadata_cannot_disappear(module):
    with pytest.raises(ValueError, match="discard"):
        module.align_addition(row(speaker_verified=True, speaker="actor"), set(row()))


def test_max_dur_is_parameterisable_and_defaults_to_20(module):
    """r27 需要 3–30s（继承的 r18 系清单本来就含 >20s 的行，用 20s 会让新增数据的时长分布
    与继承数据不一致），但默认值必须仍是 20s，否则既有调用方会被静默放宽。"""
    validation = row("validation", speaker_verified=True, speaker="held-out", session="studio")
    candidates = [row("mid", duration=25.0), row("ok", duration=8.0), row("huge", duration=30.01)]

    kept, rejected = module.eligible_additions(candidates, [], [validation], set())
    assert [r["text"] for r in kept] == ["ok"]
    assert rejected["duration_outside_3_20s"] == 2

    kept30, rejected30 = module.eligible_additions(candidates, [], [validation], set(), 30.0)
    assert [r["text"] for r in kept30] == ["mid", "ok"]
    assert rejected30["duration_outside_3_30s"] == 1
    assert module.row_issues(row("mid", duration=25.0), set(), set(), 30.0) == []
