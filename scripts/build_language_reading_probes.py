#!/usr/bin/env python3
"""Build bounded id/th/tl reading probes without synthesis, ASR or network access."""
import argparse
import json
from pathlib import Path

from voxft.eval import SAMPLE_BY_LANG, WER_LANGS

TEXTS = {
    "id": [
        ("currency", "Total tagihannya Rp 1.500.000,50.", True),
        ("percent", "Diskon 20% menjadi Rp 250.000.", True),
        ("date", "Rapat tanggal 12 Maret 2026 pukul 09.30.", True),
        ("phone", "Hubungi saya di 0812-3456-7890.", True),
        ("code_switch", "Besok kita review budget Rp 2.000.000 untuk campaign Instagram dan TikTok.", True),
        ("mixed_currency", "Harga tiketnya US$ 25, bukan Rp 25.000.", True),
        ("short_negative", "Bukan kamu.", False),
        ("short_command", "Jangan!", False),
    ],
    "th": [
        ("currency_digits", "ราคารวม 1,500.50 บาท", True),
        ("currency_words", "ราคารวมหนึ่งพันห้าร้อยบาทห้าสิบสตางค์", True),
        ("percent_words", "ส่วนลดยี่สิบเปอร์เซ็นต์", True),
        ("date", "วันที่ 12 มีนาคม 2026 เวลา 09:30 น.", True),
        ("phone", "โทรหาฉันที่ 081-234-5678", True),
        ("question", "คุณไม่เข้าใจหรือครับ", False),
        ("negative", "ไม่ใช่คุณ", False),
        ("short_answer", "ได้", False),
    ],
    "tl": [
        ("taglish", "Bukas natin i-review ang budget para sa Instagram campaign.", False),
        ("contraction_di", "'Di ako pupunta bukas.", False),
        ("contraction_nyo", "N’yo bang kunin ang ₱1,500.50?", True),
        ("currency", "₱1,500.50 ang babayaran mo bukas.", True),
        ("phone", "Tawagan mo ako sa 0917-123-4567.", True),
        ("date", "Petsa: Marso 12, 2026.", True),
        ("percent", "20% off ang lahat ng items sa checkout.", True),
        ("short_negative", "Hindi.", False),
    ],
}
REFS = (
    ("zh", "refs/ref_zh_01.wav", "must_verify_registered_cross_language_reference"),
    ("en", "refs/ref_en_01.wav", "must_verify_registered_cross_language_reference"),
)


def build_cases():
    rows = []
    for lang, cases in TEXTS.items():
        variants = [*REFS, (lang, f"refs/native_{lang}_01.wav", "blocked_missing_native_reference")]
        for kind, text, numeric in cases:
            for ref_lang, ref_audio, reference_status in variants:
                rows.append({
                    "case_id": f"reading_{lang}_{kind}_{ref_lang}",
                    "lang": lang, "text": text, "numeric": numeric,
                    "ref_audio": ref_audio, "ref_lang": ref_lang,
                    "reference_required": True, "reference_status": reference_status,
                    "review_required": "native_blind_listening",
                    "note": "仅离线探针；读音、自然度和情绪以对应母语盲听为准，自动指标不能替代人耳。",
                    **({"offline_alignment": "mfa_optional_not_executed"} if lang == "th" else {}),
                })
    return rows


def validate_cases(rows):
    ids = set()
    for row in rows:
        if row["case_id"] in ids or row["lang"] not in SAMPLE_BY_LANG or not row["text"].strip():
            raise ValueError(f"invalid probe case: {row}")
        ids.add(row["case_id"])
        if not row["ref_audio"] or not row["reference_required"]:
            raise ValueError(f"probe must not fall back to no-reference mode: {row['case_id']}")
        if row["ref_lang"] == row["lang"] and row["reference_status"] != "blocked_missing_native_reference":
            raise ValueError(f"native reference must be explicit: {row['case_id']}")
        if row["lang"] == "th" and row["lang"] in WER_LANGS:
            raise ValueError("Thai must not use whitespace WER")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("eval_cases/language_reading_probes_20261026.jsonl"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = validate_cases(build_cases())
    counts = {lang: sum(row["lang"] == lang for row in rows) for lang in TEXTS}
    blocked = sum(row["reference_status"] == "blocked_missing_native_reference" for row in rows)
    print(f"validated={len(rows)} by_lang={counts} blocked_native_reference={blocked}; no requests sent")
    if not args.write:
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
