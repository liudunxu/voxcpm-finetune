from copy import deepcopy
from html.parser import HTMLParser
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("control_ab_review")


def cases_fixture():
    return [{"case_id": f"{lang}_{number}_{condition}", "sentence_id": f"{lang}_{number}",
             "condition": condition, "lang": lang, "ref_lang": "zh" if number < 6 else "en",
             "cohort": "independent_text" if number < 8 else "historical_regression",
             "control": "" if condition == "plain" else "sad", "text": f"{lang} text {number}",
             "numeric": False}
            for lang in ("id", "th", "tl") for number in range(9)
            for condition in ("plain", "control")]


def test_random_selection_is_metadata_only_and_disjoint(module):
    cases = cases_fixture()
    selected = module.random_selection(cases)
    assert selected == module.random_selection(list(reversed(cases)))
    assert len(selected) == len({row["sentence_id"] for row in selected}) == 12
    lookup = {row["case_id"]: row for row in cases}
    assert sum(lookup[row["case_id"]]["ref_lang"] == "zh" for row in selected) == 9
    assert sum(lookup[row["case_id"]]["condition"] == "control" for row in selected) == 6
    assert {row["seed"] for row in selected} <= set(module.SEEDS)
    for case in cases:
        case["cer"] = 99
        case["tail_silence_sec"] = 500
    assert module.random_selection(cases) == selected
    with pytest.raises(ValueError, match="Insufficient"):
        module.random_selection([row for row in cases if row["ref_lang"] == "zh"])


@pytest.fixture
def source(module, tmp_path, monkeypatch):
    wav = tmp_path / "original.wav"
    sf.write(wav, np.concatenate([np.sin(np.arange(8000) * .1) * .2, np.zeros(1600)]), 16000)
    ref = tmp_path / "reference.wav"
    ref.write_bytes(wav.read_bytes())
    metrics = module.inspect_audio(wav)
    cases = [{**row, "ref_audio": str(ref)} for row in cases_fixture()]
    items = [{**row, "seed": seed, "wav": str(wav), **metrics, "f0_std_st": 1.0,
              "cer": .01, "wer": None, "suspected_truncation": False, "speech_ratio": .8,
              "human_review": {"notes": "old judgment must not be shown"}}
             for row in cases for seed in module.SEEDS]
    reports = {name: {"items": deepcopy(items), "label": name, "base": "/remote/base",
                      "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
                      "wav_sha256": {str(wav): module.sha256(wav)}}
               for name in ("base", "r23", "off", "on")}
    plan = {"inputs_sha256": {str(ref): module.sha256(ref)}}
    monkeypatch.setattr(module, "load_sources", lambda work: (plan, cases, reports, {}))
    monkeypatch.setattr(module, "diagnose_training", lambda work, plan: {"test_only": True})
    return wav, ref, cases, reports


def test_risk_selection_does_not_replace_random_or_duplicate_texts(module, source):
    _, _, cases, reports = source
    selected = module.select_pairs(cases, reports)
    assert selected[:12] == module.random_selection(cases)
    assert len(selected) == len({row["sentence_id"] for row in selected}) == 18
    assert [row["stratum"] for row in selected].count("historical_risk") == 3
    random_before = selected[:12]
    reports["on"]["items"][-1]["suspected_truncation"] = True
    assert module.select_pairs(cases, reports)[:12] == random_before
    assert module.select_pairs(cases, reports)[-1]["seed"] == module.SEEDS[-1]


def test_frozen_wav_changed_prevents_export(module, source, tmp_path):
    wav, _, _, _ = source
    wav.write_bytes(wav.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen file"):
        module.prepare(tmp_path, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_package_is_blind_and_preserves_bytes_and_unknown_feedback(module, source, tmp_path):
    wav, ref, _, _ = source
    originals = {path: path.read_bytes() for path in (wav, ref)}
    output = tmp_path / "review"
    module.prepare(tmp_path, output)
    package = module.read_json(output / "package.json")
    key = module.read_json(output / "mapping.json")
    page = (output / "public/index.html").read_text()
    assert package["pairs"] == 18 and package["audio_count"] == 36
    assert package["production_changed"] is False and package["training_eligible"] is False
    assert "old judgment" not in page and "/remote/" not in page
    assert "mapping.json" not in page and "base.json" not in page and "on.json" not in page

    class AudioSources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sources = []

        def handle_starttag(self, tag, attrs):
            if tag == "audio":
                self.sources.append(dict(attrs)["src"])

    parser = AudioSources()
    parser.feed(page)
    assert len(parser.sources) == 36
    assert all((output / "public" / path).read_bytes() == originals[wav] for path in parser.sources)
    assert len(list((output / "references").glob("*.wav"))) == 1
    assert all(row["effective_parameters"]["retry_badcase"] is False for row in key["items"])
    assert all(row["prompt"]["status"] == "not_supplied" for row in key["items"])
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, output)
    feedback = tmp_path / "feedback.txt"
    for text in ("试听包 wrong\nR01-A：无明显异常",
                 f"试听包 {key['package_id']}\nR99-A：无明显异常",
                 f"试听包 {key['package_id']}\nR01-A：无明显异常\nR01-A：不确定"):
        feedback.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            module.import_feedback(output, feedback)
        assert not (output / "feedback_review").exists()
    frozen = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    feedback.write_text(f"试听包 {key['package_id']}\nR01-A：有明确异常；3秒沙哑\nR01-B：不确定\n"
                        "D01-A：无明显异常\n其余条目未评；语言质量未验证。\n", encoding="utf-8")
    summary = module.import_feedback(output, feedback)
    assert set(summary) == {"random", "tail_risk", "historical_risk"}
    assert all(value["rated"] == 0 for value in summary.values())
    assert summary["random"]["acoustic_by_lang"]["id"]["paired"] == 0
    mapped = module.read_json(output / "feedback_review/mapped_feedback.json")
    statuses = {row["code"]: row["human_review"]["acoustic_status"] for row in mapped}
    assert statuses == {"R01-A": "abnormal", "R01-B": None, "D01-A": "clear"}
    assert all(path.read_bytes() == content for path, content in {**originals, **frozen}.items())
    with pytest.raises(FileExistsError):
        module.import_feedback(output, feedback)


def test_tampered_export_cannot_receive_feedback(module, source, tmp_path):
    output = tmp_path / "review"
    module.prepare(tmp_path, output)
    (output / "public/audio/R01-A.wav").write_bytes(b"changed")
    feedback = tmp_path / "feedback.txt"
    feedback.write_text("not evaluated", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen file"):
        module.import_feedback(output, feedback)
    assert not (output / "feedback_review").exists()


def test_shard_hashes_require_original_items_and_complete_coverage(module, tmp_path):
    path = tmp_path / "shard.json"
    row = {"case_id": "one", "seed": 42, "wav": "/original.wav"}
    module.write_json(path, {"items": [row], "wav_sha256": {"/original.wav": "hash"}})
    report = {"items": [row], "merged_from": [str(path)]}
    assert module.wav_hashes(report) == {"/original.wav": "hash"}
    report["items"] = [{**row, "seed": 43}]
    with pytest.raises(ValueError, match="Shard items"):
        module.wav_hashes(report)
    with pytest.raises(ValueError, match="coverage"):
        module.wav_hashes({"items": [row], "wav_sha256": {"/wrong.wav": "hash"}})


def test_tail_summary_unknown_is_not_zero_and_uses_frozen_quantile(module):
    result = module.tail_summary([{"tail_silence_sec": number / 10} for number in range(10)]
                                 + [{"tail_silence_sec": None}])
    assert result["unknown"] == 1 and result["p90_sec"] == .9 and result["over_0_5s"] == 4
    assert module.tail_summary([{}])["p90_sec"] is None


def test_training_scan_separates_exposure_from_unique_audio(module, tmp_path):
    wav = tmp_path / "target.wav"
    sf.write(wav, np.concatenate([np.sin(np.arange(8000) * .1) * .2, np.zeros(1600)]), 16000)
    row = {"audio": str(wav), "lang": "id", "source_id": "fixture"}
    manifest = tmp_path / "on.jsonl"
    manifest.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    plan = {"inputs_sha256": {str(path): module.sha256(path) for path in (wav, manifest)}}
    report = module.diagnose_training(tmp_path, plan)
    assert report["unique_targets"] == 1
    assert report["by_lang_row_weighted"][0]["rows"] == 2
    assert report["by_source_unique"][0]["rows"] == 1
    assert report["by_source_unique"][0]["source_id"] == "fixture"
    wav.write_bytes(wav.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Frozen file"):
        module.diagnose_training(tmp_path, plan)


def test_private_server_supports_ranges_and_hides_mapping(module, source, tmp_path, monkeypatch):
    import uvicorn
    from starlette.testclient import TestClient

    output = tmp_path / "review"
    module.prepare(tmp_path, output)

    def verify_server(app, **kwargs):
        assert kwargs == {"host": "127.0.0.1", "port": 16006}
        client = TestClient(app)
        assert client.get("/").status_code == 200
        assert client.get("/mapping.json").status_code == 404
        assert client.get("/references/").status_code == 404
        assert client.get("/audio/").status_code == 404
        response = client.get("/audio/R01-A.wav", headers={"Range": "bytes=0-1"})
        assert response.status_code == 206 and response.content == b"RI"
        client.close()

    monkeypatch.setattr(uvicorn, "run", verify_server)
    module.serve(output)
    (output / "public/audio/R01-A.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen file"):
        module.serve(output)


def test_feedback_archive_keeps_clear_and_regression_roles_separate(module, source, tmp_path):
    wav, ref, _, _ = source
    pack = tmp_path / "pack"
    module.prepare(tmp_path, pack)
    package = module.read_json(pack / "package.json")
    feedback = tmp_path / "user.txt"
    feedback.write_text(f"试听包 {package['package_id']}\nR01-A：无明显异常\n"
                        "R01-B：有明确异常；开头有些空白\nR02-A：不确定\n"
                        "其余条目未评；语言质量未验证。\n", encoding="utf-8")
    module.import_feedback(pack, feedback)
    original = {path: path.read_bytes() for path in pack.rglob("*") if path.is_file()}
    mapped_path = pack / "feedback_review/mapped_feedback.json"
    mapped = module.read_json(mapped_path)
    mapped[1]["human_review"]["notes"] = "changed"
    module.write_json(mapped_path, mapped)
    output = tmp_path / "archive"
    with pytest.raises(ValueError, match="verbatim"):
        module.archive_feedback(pack, output)
    assert not output.exists()
    mapped_path.write_bytes(original[mapped_path])
    with pytest.raises(ValueError, match="private"):
        module.archive_feedback(pack, pack / "public/archive")
    result = module.archive_feedback(pack, output)
    assert result == {"clear_candidates": 1, "regressions": 1, "training_rows": 0}
    annotations = module._read_manifest(output / "annotations.jsonl")
    conditioning = module._read_manifest(output / "conditioning.jsonl")
    regressions = module._read_manifest(output / "regressions.jsonl")
    assert annotations[0]["code"] == "R01-A"
    assert annotations[0]["candidate_status"] == "acoustic_clear_content_unverified"
    assert len(conditioning) == 2 and all(row["training_eligible"] is False for row in conditioning)
    assert all((output / row["clip"]).read_bytes() == wav.read_bytes() for row in conditioning)
    assert all((output / row["reference"]["clip"]).read_bytes() == ref.read_bytes()
               for row in conditioning)
    assert regressions[0]["reported_notes"] == "开头有些空白"
    assert regressions[0]["language_quality"] == "unverified"
    assert Path(regressions[0]["ref_audio"]).read_bytes() == ref.read_bytes()
    assert all(path.read_bytes() == content for path, content in original.items())
    provenance = module.read_json(output / "provenance.json")
    assert provenance["complete"] is True
    for filename, digest in provenance["artifacts_sha256"].items():
        assert module.sha256(output / filename) == digest
    with pytest.raises(FileExistsError):
        module.archive_feedback(pack, output)
    (pack / "public/audio/R01-A.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen file"):
        module.archive_feedback(pack, tmp_path / "other_archive")


def test_level_probe_distinguishes_amplitude_from_silence(module, tmp_path):
    audio = np.sin(np.arange(16000) * .1) * .2
    paths = {name: tmp_path / f"{name}.wav" for name in ("full", "half", "padded")}
    for name, samples in (("full", audio), ("half", audio * .5),
                          ("padded", np.concatenate([audio, np.zeros(16000)]))):
        sf.write(paths[name], samples, 16000, subtype="FLOAT")
    levels = {name: module.level_metrics(path) for name, path in paths.items()}
    for metric in ("rms_dbfs", "peak_dbfs", "p90_frame_rms_dbfs"):
        assert levels["half"][metric] - levels["full"][metric] == pytest.approx(-6.0206, abs=.001)
    assert levels["padded"]["rms_dbfs"] - levels["full"]["rms_dbfs"] == pytest.approx(-3.0103, abs=.001)
    assert abs(levels["padded"]["p90_frame_rms_dbfs"] - levels["full"]["p90_frame_rms_dbfs"]) < .05


def test_registered_feedback_keeps_original_inputs_and_labels(module):
    directory = Path(__file__).resolve().parents[1] / "eval_cases"
    source = {row["case_id"]: row for row in module._read_manifest(directory / "control_ab_20261022.jsonl")}
    cases = module._read_manifest(directory / "acoustic_feedback_20261022.jsonl")
    assert {row["audition_code"] for row in cases} == {"D01-B", "D02-A", "D05-A", "D06-A"}
    assert len({row["case_id"] for row in cases}) == 4
    for row in cases:
        original = source[row["source_case_id"]]
        assert all(row[field] == original[field] for field in
                   ("lang", "text", "control", "ref_lang", "numeric", "condition"))
        assert row["ref_audio_original_path"] == original["ref_audio"]
        assert row["seed"] in module.SEEDS
        assert row["training_eligible"] is False and row["language_quality"] == "unverified"
    assert next(row for row in cases if row["audition_code"] == "D05-A")["reported_notes"] == "开头有些空白"
