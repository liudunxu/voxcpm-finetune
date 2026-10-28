import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_finetune_readiness")


def test_audio_checks_references_and_preserves_bytes(module, tmp_path):
    target, reference = tmp_path / "target.wav", tmp_path / "reference.wav"
    waveform = np.linspace(-0.1, 0.1, 48000, dtype=np.float32)
    sf.write(target, waveform, 16000)
    sf.write(reference, waveform * 0.5, 16000)
    before = {path: path.read_bytes() for path in (target, reference)}
    rows = [{"audio": str(target), "duration": 3, "ref_audio": str(reference), "ref_duration": 3}]
    files, failures, hashes = module.check_audio(rows + rows)
    assert not failures
    assert set(files) == set(hashes) == {str(target), str(reference)}
    assert files[str(target)]["pcm_sha256"] != files[str(reference)]["pcm_sha256"]
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_missing_or_inconsistent_audio_cannot_pass(module, tmp_path):
    target = tmp_path / "target.wav"
    sf.write(target, np.full(48000, 0.1, dtype=np.float32), 16000)
    _, failures, _ = module.check_audio([
        {"audio": str(target), "duration": 3},
        {"audio": str(target), "duration": 4},
        {"audio": str(tmp_path / "missing.wav"), "duration": 3},
    ])
    assert len(failures) == 2
    assert failures[0]["reason"] == "conflicting_manifest_duration"


def test_blocked_reference_stays_blocked_even_if_file_exists(module, tmp_path):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"not yet approved")
    rows = [
        {"case_id": "cross", "ref_audio": "ref.wav", "reference_status": "must_verify_registered"},
        {"case_id": "native", "ref_audio": "ref.wav",
         "reference_status": "blocked_missing_native_reference"},
        {"case_id": "missing", "ref_audio": "missing.wav"},
    ]
    ready, blocked = module.resolve_probes(rows, tmp_path)
    assert [row["case_id"] for row in ready] == ["cross"]
    assert [row["case_id"] for row in blocked] == ["native", "missing"]
    assert rows[0]["ref_audio"] == "ref.wav"


def test_inoculation_does_not_become_verified_control(module):
    rows = [{"audio": "/audio.wav", "origin_audio": "/original.wav", "text": "some text",
             "duration": 5, "lang": "id", "source_id": "cv22_id",
             "control": "neutral", "control_inoculation": True}]
    result = module.coverage(rows)
    assert result["control_inoculation_rows"] == 1
    assert result["verified_control_rows"] == 0


def test_existing_output_is_never_overwritten(module, tmp_path):
    sentinel = tmp_path / "summary.json"
    sentinel.write_bytes(b"frozen")
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, tmp_path, tmp_path)
    assert sentinel.read_bytes() == b"frozen"


def test_fresh_holdout_is_frozen_and_excluded_from_training(module):
    directory = Path(__file__).resolve().parents[1] / "eval_cases"
    path = directory / "fresh_text_holdout_20261028.jsonl"
    assert module.sha256(path) == "10566052db1254a68d8c2e295248b0283cde4e22b410e4762b9231339df55267"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({row["case_id"] for row in rows}) == 180
    excluded, _ = importlib.import_module("incremental_mix").load_exclusions(directory)
    for lang in ("th", "tl", "id"):
        group = [row for row in rows if row["lang"] == lang]
        texts = {module.evaluation._norm(row["text"]) for row in group}
        assert len(group) == 60 and len(texts) == 30
        assert texts <= excluded
        for text in texts:
            pair = [row for row in group if module.evaluation._norm(row["text"]) == text]
            assert {row["condition"] for row in pair} == {"plain", "control"}
            assert len({row["ref_audio"] for row in pair}) == 1
            assert all(row["ref_audio"] and not row["numeric"] for row in pair)
