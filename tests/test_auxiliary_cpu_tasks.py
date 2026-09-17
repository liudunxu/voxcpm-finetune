import importlib.util
import json
from pathlib import Path

import pytest


def test_holdout_parser_and_sampling_never_normalize_training_text(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("holdout_texts", scripts / "holdout_texts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = '"Hello, World!"'
    rows = module.parse_tsv(f"1\ta.wav\t{raw}\thello world\tletters\t64000\tFEMALE\n")
    assert rows[0]["text"] == raw
    selected, summary = module.select_texts(rows + rows, set(), "tl")
    assert len(selected) == 1 and summary["excluded"]["duplicate_text"] == 1
    assert summary["shortfall"] == 29
    assert module.select_texts(rows, {"helloworld"}, "tl")[0] == []
    assert module.select_texts([{**rows[0], "text": "There are 10 words here."}], set(), "tl")[0] == []
    with pytest.raises(ValueError, match="schema"):
        module.parse_tsv("1\ta.wav\ttext\n")
    with pytest.raises(ValueError, match="mismatch"):
        module.parse_tsv("1\ta.wav\twrong\tother\tletters\t64000\tFEMALE\n")


def test_acoustic_sample_is_fixed_and_does_not_rewrite_reports(monkeypatch, tmp_path):
    from voxft import eval as evaluation

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("acoustic_review", scripts / "acoustic_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "reports")
    cases = [{"case_id": f"{lang}_{reference}", "text": "test", "lang": lang,
              "ref_lang": reference, "ref_audio": f"/{reference}.wav", "numeric": False}
             for lang in module.TARGET_LANGS for reference in ("zh", "en", "tl")]
    assert module.select_cases(cases) == module.select_cases(list(reversed(cases)))
    with pytest.raises(ValueError, match="sample"):
        module.select_cases(cases[:-1])
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    module.prepare(tmp_path)
    originals = {}
    for arm in ("off", "on"):
        audio = tmp_path / f"{arm}.wav"
        audio.touch()
        report = {"cohorts": {"stale_full_report_counts": 75},
                  "items": [{**case, "seed": seed, "wav": str(audio), "cer": 0.1, "wer": 0.1,
                             "audio_sec": 4.0, "f0_std_st": 1.0}
                            for case in cases for seed in (42, 43, 44, 45, 46)]}
        source = tmp_path / f"{arm}_report.json"
        source.write_text(json.dumps(report))
        originals[source] = source.read_bytes()
    module.collect(tmp_path)
    saved = json.loads((tmp_path / "acoustic_sample/reports.json").read_text())
    assert all("cohorts" not in json.loads(Path(path).read_text()) for path in saved.values())
    session = evaluation.review_session(saved["off"], saved["on"])
    assert len(session) == 30 and {row["seed"] for row in session} == {42, 43}
    assert all(not row["review_1"] and not row["review_2"] for row in session)
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    with pytest.raises(FileExistsError):
        module.collect(tmp_path)


def test_holdout_preparation_preserves_multiple_existing_references(monkeypatch, tmp_path):
    import huggingface_hub

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("holdout_prepare", scripts / "holdout_texts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("DATA_PROCESSED", "DATA_RAW", "ROOT", "CHECKPOINT_DIR"):
        monkeypatch.setattr(module, name, tmp_path / name)
    monkeypatch.setattr(module.evaluation, "EVAL_DIR", tmp_path / "reports")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("---\nlicense: [cc-by-4.0]\n---\nTest")
    for lang in module.TARGET_LANGS:
        directory = source / "data" / module.get_source(f"fleurs_{lang}").config
        directory.mkdir(parents=True)
        lines = []
        for position in range(60):
            text = f"{lang.upper()} example sentence {chr(97 + position // 26)}{chr(97 + position % 26)}."
            lines.append(f"{position}\tclip.wav\t{text}\t{text.lower()}\tletters\t64000\tFEMALE\n")
        (directory / "test.tsv").write_text("".join(lines))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda filename, **kwargs: str(source / filename))
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    references = []
    for lang, count in (("zh", 4), ("en", 4), ("tl", 5)):
        for position in range(count):
            audio = tmp_path / f"{lang}_{position}.wav"
            audio.touch()
            references.append({"text": "existing text", "ref_lang": lang, "ref_audio": str(audio)})
    (experiment / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in references))
    module.prepare(tmp_path / "output", experiment)
    plan = json.loads((tmp_path / "output/plan.json").read_text())
    cases = module._read_manifest(tmp_path / "output/cases.jsonl")
    assert len(cases) == 150 and all(row["selected"] == 30 for row in plan["by_lang"].values())
    assert {row["ref_audio"] for row in cases} == {row["ref_audio"] for row in references}
    assert plan["evaluated"] is False and plan["evaluation_seeds"][-1] == 49
