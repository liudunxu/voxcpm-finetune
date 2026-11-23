from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from r55_ratio_trim_run import text_units, trim


def test_text_units_by_script():
    assert text_units("(calm)Kumusta ka po?", "tl") == 3  # control prefix stripped, words
    assert text_units("สวัสดี ครับ", "th") == 10  # letters + marks, no spaces
    assert text_units("你好，世界", "zh") == 4


def test_trim_per_language_tails():
    rows = [{"lang": "id", "text": "a b", "duration": float(d)} for d in range(1, 101)]
    rows += [{"lang": "th", "text": "กข", "duration": 3.0}] * 10
    kept, dropped = trim(rows)
    # id: 100 rows -> drop 1 lowest + 5 highest ratio; th: 10 rows -> int(0.1)=0, int(0.5)=0 -> nothing dropped
    assert sorted(r["duration"] for r in dropped) == [1.0, 96.0, 97.0, 98.0, 99.0, 100.0]
    assert len(kept) == 104 and sum(r["lang"] == "th" for r in kept) == 10
