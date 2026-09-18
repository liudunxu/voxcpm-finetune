from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from numeric_probe import common_scores


def test_numeric_probe_uses_same_source_reference_for_both_inputs():
    def verbalize(text, language):
        assert language == "ms"
        return text.replace("RM1,200", "seribu dua ratus ringgit")

    source = "Harganya RM1,200."
    expected = verbalize(source, "ms")
    raw = common_scores({"lang": "ms", "source_text": source, "text": source, "hyp": source}, verbalize)
    words = common_scores({"lang": "ms", "source_text": source, "text": expected, "hyp": expected}, verbalize)
    duplicate = common_scores({
        "lang": "ms", "source_text": source,
        "text": expected.replace(".", " ringgit."), "hyp": expected.replace(".", " ringgit."),
    }, verbalize)
    assert raw["common_expected"] == words["common_expected"] == duplicate["common_expected"]
    assert raw["common_cer"] == words["common_cer"] == 0
    assert duplicate["common_cer"] > 0
    with pytest.raises(KeyError):
        common_scores({"lang": "ms", "text": source, "hyp": source}, verbalize)
