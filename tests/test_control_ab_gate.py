from copy import deepcopy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("control_ab_gate")


def test_arms_reject_nonprefix_changes(module):
    controlled = {"audio": "/target.wav", "text": "(angry)hello", "control": "angry",
                  "emotion_verified": True}
    plain = {**controlled, "text": "hello", "control": ""}
    module.verify_arms([plain], [controlled])
    for changed in ({**plain, "audio": "/different.wav"}, {**plain, "text": "different"}):
        with pytest.raises(ValueError):
            module.verify_arms([changed], [controlled])
    with pytest.raises(ValueError):
        module.verify_arms([], [controlled])
    unverified = {**controlled, "emotion_verified": False}
    with pytest.raises(ValueError, match="Unverified"):
        module.verify_arms([{**unverified, "text": "hello", "control": ""}], [unverified])


def test_pcm_isolation_includes_references_and_exposure(module):
    audio = {name: {"pcm_sha256": digest}
             for name, digest in (("train", "train_pcm"), ("ref", "ref_pcm"),
                                  ("val", "val_pcm"), ("copy", "ref_pcm"))}
    train = [{"audio": "train", "ref_audio": "ref"}]
    assert not any(module.pcm_conflicts(train, [{"audio": "val"}], audio).values())
    assert module.pcm_conflicts(train, [{"audio": "copy"}], audio)["train_validation_pcm"]
    assert module.pcm_conflicts([{"audio": "ref", "ref_audio": "copy"}], [], audio)[
        "identical_target_reference"]
    assert module.pcm_conflicts([{"audio": "train"}] * 4, [], audio)["pcm_exposure_above_3"]


def test_historical_source_alias_does_not_change_verified_identity(module):
    row = {"source_id": "cv22_th_r23boost", "audio": "/target.wav",
           "origin_audio": "/raw/ref.wav", "speaker_verified": True, "speaker": "cv22_th:actor"}
    raw = {"/raw/ref.wav": {"speaker": "actor"}}
    module.verify_provenance(row, raw)
    with pytest.raises(ValueError, match="speaker differs"):
        module.verify_provenance({**row, "speaker": "cv22_th:other"}, raw)
    with pytest.raises(ValueError, match="Missing raw"):
        module.verify_provenance(row, {})


def test_replay_reduction_keeps_all_targets_and_order(module):
    targets = [{"audio": f"/target{index}", "lang": module.LANGS[index % 3], "duration": 10}
               for index in range(95)]
    replay = [{"audio": f"/{lang}{index}", "lang": lang, "duration": 10}
              for lang in ("zh", "en") for index in range(10)]
    rows = targets + replay
    selected, summary = module.trim_replay(rows)
    assert selected[:95] == targets and len(selected) == 100
    assert summary["actual_shares"]["zh"] == pytest.approx(0.04)
    assert summary["actual_shares"]["en"] == pytest.approx(0.01)
    assert module.trim_replay(rows) == (selected, summary)
    assert rows == targets + replay


def test_reference_selection_rejects_training_pcm_and_metadata(module):
    rows = [{"audio": f"/ref{index}", "duration": 5} for index in range(6)]
    audio = {row["audio"]: {"pcm_sha256": str(index)} for index, row in enumerate(rows)}
    chosen = module.choose_references(rows, {("audio", "/ref0")}, {"1", "2"}, audio)
    assert chosen == rows[3:]
    with pytest.raises(ValueError, match="three traceable"):
        module.choose_references(rows, set(), {"0", "1", "2", "3"}, audio)


def test_cases_pair_control_without_inflating_text_count(module):
    rows = {lang: [{"source_sentence_id": str(index), "text": f"{lang} sentence {index}",
                    "source_audio_filename": f"{index}.wav"} for index in range(30)]
            for lang in module.LANGS}
    references = {lang: [f"/{lang}.wav"] for lang in ("zh", "en", "tl")}
    cases = module.build_cases(rows, references)
    assert len(cases) == 180 and len({row["sentence_id"] for row in cases}) == 90
    assert sum(row["ref_lang"] == "zh" for row in cases) == 144
    assert sum(row["ref_lang"] == "en" for row in cases) == 36
    for plain, controlled in zip(cases[::2], cases[1::2]):
        assert plain["control"] == "" and controlled["control"]
        assert all(plain[field] == controlled[field] for field in
                   ("text", "ref_audio", "sentence_id", "lang", "ref_lang"))
    changed = deepcopy(rows)
    changed["tl"].pop()
    with pytest.raises(ValueError, match="30 independent"):
        module.build_cases(changed, references)


def test_historical_regressions_use_source_languages_and_shared_seeds(module):
    directory = module.ROOT / "eval_cases"
    manifests = {directory / name: module._read_manifest(directory / name)
                 for name in ("omnivoice_prod_v2.jsonl", "short_cue_probe_20261020.jsonl",
                              "id_prosody_probe_20261020.jsonl")}
    cases = module.regression_cases(manifests)
    assert len(cases) == 20 and len({row["sentence_id"] for row in cases}) == 10
    assert all(row["ref_lang"] in {"zh", "en"} and "seed" not in row for row in cases)
    assert all(row["cohort"] == "historical_regression" for row in cases)
    assert all(Path(row["ref_audio"]).is_absolute() for row in cases)
    assert any(row["historical_case_id"] == "id_ms_colloquial_en" for row in cases)
