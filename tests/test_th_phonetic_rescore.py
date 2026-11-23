from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from th_phonetic_rescore import rescore_item

FAKE = {"ไม่": "mai", "ม่าย": "mai", "อย่านะ": "yana", "อย่างนั้น": "yangnan"}.get


def test_spelling_variant_is_not_an_error():
    it = {"lang": "th", "text": "ไม่!", "hyp": "ม่าย"}
    assert rescore_item(it, romanize=FAKE)["cer"] == 0


def test_real_misread_stays_an_error_and_other_langs_untouched():
    it = {"lang": "th", "text": "อย่านะ!", "hyp": "อย่างนั้น"}
    assert rescore_item(it, romanize=FAKE)["cer"] > 0.3
    tl = {"lang": "tl", "text": "Sige!", "hyp": "x", "cer": 1.0}
    assert rescore_item(tl) is tl
