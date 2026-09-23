import importlib.util
import base64
import json
from pathlib import Path

import pytest
import soundfile as sf


def test_ab_candidates_preserve_evidence_and_never_become_training_rows(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/collect_ab_candidates.py"
    spec = importlib.util.spec_from_file_location("collect_ab_candidates", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack, output = tmp_path / "pack", tmp_path / "pool"
    (pack / "audio").mkdir(parents=True)
    (pack / "evidence").mkdir()
    feedback = {"pack_id": "demo-123", "cases": [], "frozen_artifacts_sha256": {}}
    statuses = [("无明显异常", "无明显异常"), ("有明确异常", "不确定"), ("未评", "无明显异常")]
    for index, pair in enumerate(statuses, 1):
        case = {"code": f"C{index:02}", "submitted_preference": "A 更好", "raw_note": "原反馈", "takes": {}}
        for arm, status in zip(("A", "B"), pair):
            code = f"{case['code']}-{arm}"
            audio = pack / "audio" / f"{code}.wav"
            sf.write(audio, [0.1, -0.1] * 8, 16000)
            request = pack / "evidence" / f"{code}-request.json"
            request.write_text(json.dumps({"text": "Bukas na natin ito pag-usapan.", "language": "tl",
                                           "api_key": "DO_NOT_EXPORT",
                                           "reference_audio_base64": base64.b64encode(audio.read_bytes()).decode(),
                                           "prompt_audio_base64": base64.b64encode(audio.read_bytes()).decode(),
                                           "prompt_text": "原参考转写", "control_instruction": "calm"}),
                               encoding="utf-8")
            for path in (audio, request):
                feedback["frozen_artifacts_sha256"][str(path.relative_to(pack))] = module.sha256(path)
            case["takes"][arm] = {"status": status, "audio_sha256": module.sha256(audio),
                                  "automatic_text_qc": {"status": "pass"},
                                  "automatic_severe_issues": [] if arm == "A" else ["output_vocalization"]}
        feedback["cases"].append(case)
    feedback_path = pack / "feedback.json"
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    rows = module.collect(pack, output)
    assert [row["audition_code"] for row in rows] == ["C01-A", "C01-B", "C03-B"]
    assert all(row["training_eligible"] is False and row["source_kind"] == "synthetic_tts" for row in rows)
    assert all(row["language_quality"] == "unverified" and "audio" not in row and "text" not in row for row in rows)
    assert [row["candidate_status"] for row in rows] == ["acoustics_only", "needs_qc_review", "needs_qc_review"]
    assert rows[2]["pair_preference"] == "A 更好"
    assert len(list(output.rglob("annotations.jsonl"))) == 1
    assert len(list((output / "audio").glob("*.wav"))) == 1
    assert len(list((output / "conditioning").glob("*.wav"))) == 1
    conditions = [json.loads(line) for line in next(output.rglob("conditioning.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert len(conditions) == len(rows)
    assert all(row["reference"] == row["prompt"] and row["training_eligible"] is False for row in conditions)
    assert all(row["gpu_preprocessed_audio_persisted"] is False and row["replay_complete"] is False for row in conditions)
    assert conditions[0]["requested_parameters"]["prompt_text"] == "原参考转写"
    assert (output / conditions[0]["reference"]["clip"]).read_bytes() == (pack / "audio/C01-A.wav").read_bytes()
    assert "DO_NOT_EXPORT" not in json.dumps(conditions) and "audio_base64" not in json.dumps(conditions)
    assert "DO_NOT_EXPORT" not in json.dumps(rows)
    assert module.collect(pack, output) == rows
    request = pack / "evidence/C01-A-request.json"
    request.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen artifact missing or changed"):
        module.collect(pack, tmp_path / "bad-input")
    assert not (tmp_path / "bad-input").exists()
    feedback["cases"][0]["code"] = "../../outside"
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        module.collect(pack, output)


def test_voice_id_only_does_not_fabricate_a_reference(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/collect_ab_candidates.py"
    spec = importlib.util.spec_from_file_location("collect_ab_candidates", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = tmp_path / "pack"
    (pack / "audio").mkdir(parents=True)
    (pack / "evidence").mkdir()
    audio = pack / "audio/C01-A.wav"
    sf.write(audio, [0.1, -0.1] * 8, 16000)
    request = pack / "evidence/C01-A-request.json"
    request.write_text(json.dumps({"text": "Hi", "language": "en", "voice_id": "expired-cache-key"}))
    feedback = {"pack_id": "missing", "frozen_artifacts_sha256": {
        str(path.relative_to(pack)): module.sha256(path) for path in (audio, request)},
        "cases": [{"code": "C01", "takes": {"A": {"status": "无明显异常", "audio_sha256": module.sha256(audio)}}}]}
    (pack / "feedback.json").write_text(json.dumps(feedback))
    module.collect(pack, tmp_path / "out")
    record = json.loads(next((tmp_path / "out").rglob("conditioning.jsonl")).read_text())
    assert record["reference"]["status"] == record["prompt"]["status"] == "voice_id_only_not_restored"
    assert not (tmp_path / "out/conditioning").exists()
