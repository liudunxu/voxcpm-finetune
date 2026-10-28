from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("quarantine_finetune_candidates")


def row(name, **fields):
    return {"audio": f"/audio/{name}.wav", "origin_audio": f"/raw/{name}.wav",
            "text": name, "duration": 4.0, "lang": "id", "source_id": "cv22_id", **fields}


def test_quarantine_covers_reference_pcm_and_origin_without_repairing_rows(module):
    slow = row("slow", text="Sani sani sani", duration=7.0)
    reference = row("reference", speaker_verified=True, speaker="actor", ref_speaker="actor",
                    ref_audio="/alias.wav", ref_origin_audio="/renamed.wav")
    alias = row("alias", origin_audio="/raw/./slow.wav")
    exposed = row("exposed", text="(neutral)Held out", control="neutral",
                  control_inoculation=True)
    retained = row("keep", text="A sufficiently long complete sentence for this example",
                   private_metadata={"keep": True})
    thai = row("thai", lang="th", duration=25.0, text="ภาษาไทย")
    train, validation = [slow, reference, alias, exposed, retained, thai], [row("val")]
    original = deepcopy(train)
    audio = {sample["audio"]: {"pcm_sha256": sample["audio"]} for sample in train + validation}
    audio["/alias.wav"] = audio[slow["audio"]]
    positions, rejected, audit = module.quarantine_rows(train, validation, {"heldout"}, audio)
    assert positions == [5, 6]
    assert rejected[0]["issues"] == ["text_duration_review_pending"]
    assert rejected[1]["issues"] == ["quarantined_ref_audio"]
    assert rejected[2]["issues"] == ["quarantined_audio"]
    assert rejected[3]["issues"] == ["evaluation_or_validation_text"]
    assert not audit["violations"] and not any(audit["pcm"].values())
    assert train == original


def test_remaining_pcm_leak_cannot_produce_candidate(module):
    train, validation = [row("train")], [row("val")]
    audio = {sample["audio"]: {"pcm_sha256": "same"} for sample in train + validation}
    with pytest.raises(ValueError, match="PCM"):
        module.quarantine_rows(train, validation, set(), audio)


def test_vad_does_not_approve_or_modify_slow_audio(module, tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.full(96000, 0.1, dtype=np.float32), 16000)
    sample = row("slow", audio=str(path), origin_audio=str(tmp_path / "missing.wav"),
                 duration=6.0, text="Sani sani sani")
    original = path.read_bytes()
    result = module.duration_diagnostic(
        sample, lambda waveform, rate: [{"start": 16000, "end": 48000}])
    assert result["vad_speech_seconds"] == 2
    assert result["vad_words_per_second"] == 1.5
    assert result["vad_head_seconds"] == 1 and result["vad_tail_seconds"] == 3
    assert result["review_status"] == "unreviewed_not_confirmed_bad_audio"
    assert result["transcript_consistency"] == "unverified"
    assert result["raw_audio_available"] is False
    assert "text_duration_review" not in result["sample"]
    assert path.read_bytes() == original
    empty = module.duration_diagnostic(sample, lambda waveform, rate: [])
    assert empty["vad_words_per_second"] is None
    assert empty["vad_head_seconds"] is None
    with pytest.raises(ValueError, match="VAD"):
        module.duration_diagnostic(sample, lambda waveform, rate: [{"start": 0, "end": 96001}])


def test_prepare_refuses_existing_output_before_reading_inputs(module, tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_bytes(b"frozen")
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, tmp_path, tmp_path)
    assert sentinel.read_bytes() == b"frozen"


def test_prepare_preserves_parent_and_retained_bytes(module, monkeypatch, tmp_path):
    parent, audit_dir, eval_dir = (tmp_path / name for name in ("parent", "audit", "eval"))
    for directory in (parent, audit_dir, eval_dir):
        directory.mkdir()
    for name in importlib.import_module("incremental_mix").REQUIRED_EVAL_FILES:
        (eval_dir / name).write_text('{"text":"evaluation"}\n', encoding="utf-8")
    samples, audio = [], {}
    for position, name in enumerate(("keep", "slow", "val"), 1):
        path = parent / f"{name}.wav"
        duration = 6 if name == "slow" else 4
        sf.write(path, np.full(duration * 16000, position * 0.1, dtype=np.float32), 16000)
        sample = row(name, audio=str(path), duration=duration, _private="unchanged")
        samples.append(sample)
        audio[str(path)] = {**module.inspect_audio(sample), "file_sha256": module.sha256(path)}
    lines = [(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\r\n").encode("utf-8")
             for sample in samples]
    (parent / "train.jsonl").write_bytes(lines[0] + lines[1])
    (parent / "val.jsonl").write_bytes(lines[2])
    hashes = {str(parent / name): module.sha256(parent / name)
              for name in ("train.jsonl", "val.jsonl")}
    (audit_dir / "r26_metadata.json").write_text(
        json.dumps({"inputs_sha256": hashes}), encoding="utf-8")
    (audit_dir / "audio_audit.json").write_text(
        json.dumps({"files": audio, "failures": []}), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(module, "RUNS", {"r26": "parent"})
    fake_vad = SimpleNamespace(
        VadOptions=lambda **kwargs: SimpleNamespace(threshold=0.5, **kwargs),
        get_speech_timestamps=lambda waveform, options, sampling_rate: [],
        get_vad_model=lambda: SimpleNamespace(
            session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", fake_vad)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils",
                        SimpleNamespace(get_assets_path=lambda: str(tmp_path)))
    output = tmp_path / "candidate"
    result = module.prepare(audit_dir, eval_dir, output)
    assert (output / "r26/train.jsonl").read_bytes() == lines[0]
    assert (output / "r26/val.jsonl").read_bytes() == lines[2]
    assert result["runs"]["r26"]["excluded_rows"] == 1
    assert result["runs"]["r26"]["training_approved"] is False
    assert result["raw_audio_missing"] == 1
    assert all(path.read_bytes() == contents for path, contents in before.items())
    samples[0]["text"] = "changed"
    (parent / "train.jsonl").write_text(json.dumps(samples[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        module.prepare(audit_dir, eval_dir, tmp_path / "stale")
