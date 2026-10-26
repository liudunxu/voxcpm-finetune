import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_language_reading_probes import TEXTS, build_cases, validate_cases
from voxft.eval import WER_LANGS


def test_probe_manifest_is_bounded_and_preserves_language_specific_evidence():
    path = Path(__file__).resolve().parents[1] / "eval_cases" / "language_reading_probes_20261026.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == validate_cases(build_cases())
    assert len(rows) == 72
    assert {row["lang"] for row in rows} == {"id", "th", "tl"}
    assert all(row["reference_required"] for row in rows)
    assert sum(row["reference_status"] == "blocked_missing_native_reference" for row in rows) == 24
    by_id = {row["case_id"]: row for row in rows}
    assert "Rp 1.500.000,50" in by_id["reading_id_currency_id"]["text"]
    assert "campaign Instagram" in by_id["reading_id_code_switch_en"]["text"]
    assert by_id["reading_id_short_command_zh"]["text"] == "Jangan!"
    assert "หนึ่งพันห้าร้อยบาท" in by_id["reading_th_currency_words_th"]["text"]
    assert by_id["reading_th_currency_words_th"]["offline_alignment"] == "mfa_optional_not_executed"
    assert "th" not in WER_LANGS
    assert by_id["reading_tl_contraction_di_tl"]["text"].startswith("'Di")
    assert "N’yo" in by_id["reading_tl_contraction_nyo_tl"]["text"]
    assert all(row["review_required"] == "native_blind_listening" for row in rows)


def test_probe_texts_do_not_silently_enable_no_reference_eval():
    for lang, cases in TEXTS.items():
        assert len(cases) == 8
        for row in build_cases():
            assert row["ref_audio"]
