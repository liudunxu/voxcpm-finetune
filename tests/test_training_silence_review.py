import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("review_training_silence")


@pytest.mark.parametrize("lang,hint", [("id", "id"), ("th", "th"), ("tl", None)])
def test_asr_is_unprompted_and_does_not_reuse_vad_failure(module, lang, hint):
    calls = []

    def transcribe(audio, **kwargs):
        calls.append((audio, kwargs))
        segment = SimpleNamespace(start=0., end=3., text="A different sentence",
                                  avg_logprob=-.2, no_speech_prob=.01)
        return iter([segment]), SimpleNamespace(language=lang, language_probability=.99)

    result = module.transcribe(SimpleNamespace(transcribe=transcribe), "original.wav", lang,
                               ["The expected sentence"])
    assert calls[0][1]["language"] == hint
    assert calls[0][1]["initial_prompt"] is None
    assert calls[0][1]["vad_filter"] is False
    assert calls[0][1]["condition_on_previous_text"] is False
    assert "The expected sentence" not in repr(calls)
    assert result["comparisons"][0]["cer"] > 0
    assert result["hyp"] == "A different sentence"


def test_review_will_not_overwrite_existing_results(module, tmp_path):
    (tmp_path / "summary.json").write_text("frozen", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.review(tmp_path / "missing", tmp_path)
    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == "frozen"
    with pytest.raises(FileExistsError):
        module.quarantine(tmp_path / "missing", tmp_path)


def test_all_suspicions_and_reference_pcm_origin_reuse_are_excluded(module):
    rows = [
        {"audio": "/a.wav", "origin_audio": "/raw/a.wav"},
        {"audio": "/keep.wav", "origin_audio": "/raw/keep.wav"},
        {"audio": "/b.wav", "ref_audio": "/alias.wav"},
        {"audio": "/c.wav", "origin_audio": "/raw/./a.wav"},
        {"audio": "/d.wav", "ref_audio": "/ref.wav", "ref_origin_audio": "/raw/a.wav"},
    ]
    before = repr(rows)
    audio = {row["audio"]: {"pcm_sha256": row["audio"]} for row in rows}
    audio["/alias.wav"] = {"pcm_sha256": "/a.wav"}
    audio["/ref.wav"] = {"pcm_sha256": "different"}
    flagged = [{"audio": "/a.wav", "pcm_sha256": "/a.wav",
                "uses": [{"origin_audio": "/raw/a.wav"}],
                "review_reasons": ["vad_tail_over_0_5s"]}]
    positions, removed = module.exclude_flagged(rows, flagged, audio)
    assert positions == [2]
    assert [row["row"] for row in removed] == [1, 3, 4, 5]
    assert all(row["status"] == "quarantined_suspect_not_confirmed_bad_audio" for row in removed)
    assert repr(rows) == before
    assert module.exclude_flagged([rows[1]], flagged, audio)[1] == []


def test_quarantine_preserves_rows_validation_and_frozen_reports(module, monkeypatch, tmp_path):
    run, audit, cpu = (tmp_path / part for part in ("run", "audit", "cpu"))
    for folder in (run, audit, cpu):
        folder.mkdir()
    samples, audio = [], {}
    for index, name in enumerate(("suspect", "retain", "validation")):
        path = run / f"{name}.wav"
        path.write_bytes(f"frozen audio {name}".encode())
        sample = {"audio": str(path), "origin_audio": str(tmp_path / "raw" / path.name),
                  "duration": 4., "lang": "id", "source_id": "cv22_id",
                  "text": f"{name} is a sufficiently long distinct sentence for a test"}
        samples.append(sample)
        audio[str(path)] = {"pcm_sha256": name, "file_sha256": module.sha256(path)}
    train, val = run / "train.jsonl", run / "val.jsonl"
    raw = [(json.dumps(row, separators=(",", ":")) + "\r\n").encode() for row in samples]
    train.write_bytes(raw[0] + raw[1])
    val.write_bytes(raw[2])
    pcm_path = cpu / "audio_audit.json"
    pcm_path.write_text(json.dumps({"files": audio}), encoding="utf-8")
    inputs = {str(path): module.sha256(path) for path in [train, val, pcm_path]}
    inputs.update({row["audio"]: audio[row["audio"]]["file_sha256"] for row in samples})
    parent = run / "plan.json"
    parent.write_text(json.dumps({"inputs_sha256": inputs,
                                  "language_shares": {"id": {"actual": 1.}}}), encoding="utf-8")
    (audit / "plan.json").write_text(json.dumps({"run": str(run), "inputs_sha256": {
        **inputs, str(parent): module.sha256(parent)}}), encoding="utf-8")
    (audit / "summary.json").write_text('{"status":"complete"}', encoding="utf-8")
    measured = [{"audio": row["audio"], **audio[row["audio"]],
                 "uses": [{"origin_audio": row["origin_audio"]}],
                 "review_reasons": ["no_vad_speech"] if index == 0 else []}
                for index, row in enumerate(samples[:2])]
    (audit / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in measured),
                                             encoding="utf-8")
    (audit / "flagged.json").write_text(json.dumps(measured[:1]), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(module, "load_exclusions", lambda path: (set(), {}))
    output = tmp_path / "candidate"
    result = module.quarantine(audit, output)
    assert result["excluded_rows"] == 1 and result["training_started"] is False
    assert not any(result["pcm"].values()) and not result["checks"]["violations"]
    assert (output / "train.jsonl").read_bytes() == raw[1]
    assert (output / "val.jsonl").read_bytes() == raw[2]
    assert all(path.read_bytes() == content for path, content in before.items())
