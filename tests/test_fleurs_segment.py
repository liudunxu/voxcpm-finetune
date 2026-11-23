from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fleurs_segment_20261103 import split_words


def test_latin_split_unchanged():
    surface, words, joiner = split_words("Kumusta ka, Maria?", "tl")
    assert surface == ["Kumusta", "ka,", "Maria?"] and words == ["kumusta", "ka", "maria"] and joiner == " "


def test_thai_split_keeps_original_text_and_romanizes():
    pytest.importorskip("pythainlp")
    fake = {"ผม": "pm", "จะ": "cha", "ไป": "pai", "ตลาด": "tlaat", "ครับ": "khrap"}
    text = "ผมจะไปตลาด, ครับ"
    surface, words, joiner = split_words(text, "th", romanize=lambda t: fake.get(t, t))
    assert joiner.join(surface) == text  # segment text is always an exact substring
    assert words == ["pm", "cha", "pai", "tlaat", "khrap"]
    assert len(surface) == len(words)
