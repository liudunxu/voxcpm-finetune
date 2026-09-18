import importlib.util
import json
from pathlib import Path

import pytest


def test_offline_review_verifies_downloads_and_keeps_blind_codes(monkeypatch, tmp_path):
    import wave
    from html.parser import HTMLParser

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("holdout_diagnostics", scripts / "holdout_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "dual_cue"
    audio_dir = directory / "offline" / "audio"
    audio_dir.mkdir(parents=True)
    module.write_json(directory / "plan.json", {"test": True})
    records, saved, report_hashes = [], {}, {}
    for number, model in enumerate(("base", "r8"), 1):
        filename = f"{number:032x}.wav"
        audio_path = audio_dir / filename
        with wave.open(str(audio_path), "wb") as audio:
            audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 24032)
        components = [{"case_id": "first", "start_sample": 0, "end_sample": 16},
                      {"case_id": "second", "start_sample": 24016, "end_sample": 24032}]
        item = {"case_id": "dual_test", "seed": 42, "lang": "th", "text": "same text",
                "wav": f"/remote/{filename}", "ref_audio": "/remote/ref.wav",
                "components": components, "sample_rate": 48000, "review_only": True,
                "human_review": {"notes": "Do not inherit earlier labels"}}
        report_path = directory / f"{model}.json"
        module.write_json(report_path, {"review_only": True, "label": model, "items": [item]})
        saved[model] = f"/remote/{model}.json"
        report_hashes[saved[model]] = module.sha256(report_path)
        records.append({"model": model, "case_id": item["case_id"], "seed": 42,
                        "wav": item["wav"], "components": components,
                        "audio_sha256": module.sha256(audio_path), "pcm_samples_verified": True})
    module.write_json(directory / "reports.json", saved)
    module.write_json(directory / "results.json", {
        "complete": True, "plan_sha256": module.sha256(directory / "plan.json"), "items": records,
    })
    module.write_json(directory / "verification.json", {
        "complete": True, "report_sha256": report_hashes,
    })
    original_audio = audio_path.read_bytes()
    audio_path.write_bytes(original_audio + b"changed")
    with pytest.raises(ValueError, match="audio"):
        module.offline_review(tmp_path)
    assert not (directory / "offline_key.json").exists()
    audio_path.write_bytes(original_audio)
    index_path = module.offline_review(tmp_path)
    page = index_path.read_text()
    key_path = directory / "offline_key.json"
    mapping = json.loads(key_path.read_text())
    assert {item["code"] for item in mapping["items"]} == {"D01-A", "D01-B"}
    assert {item["model"] for item in mapping["items"]} == {"base", "r8"}
    assert all(item["audio_sha256"] == module.sha256(index_path.parent / item["local_audio"])
               for item in mapping["items"])

    class AudioSources(HTMLParser):
        sources = []

        def handle_starttag(self, tag, attrs):
            if tag == "audio":
                self.sources.append(dict(attrs)["src"])

    parser = AudioSources()
    parser.feed(page)
    assert len(parser.sources) == 2
    assert all(source.startswith("audio/") and (index_path.parent / source).is_file()
               for source in parser.sources)
    assert "fetch(" not in page and "https://" not in page and "/remote/" not in page
    assert "base.json" not in page and "r8.json" not in page
    assert "Do not inherit earlier labels" not in page
    assert page.count('<option value="">未评</option>') == 2
    frozen = {path: path.read_bytes() for path in (index_path, key_path, audio_path)}
    with pytest.raises(FileExistsError):
        module.offline_review(tmp_path)
    assert all(path.read_bytes() == contents for path, contents in frozen.items())
    feedback_path = tmp_path / "feedback.txt"
    for text in ("试听包 wrong\nD01-A：无明显异常",
                 f"试听包 {mapping['package_id']}\nD99-A：无明显异常",
                 f"试听包 {mapping['package_id']}\nD01-A：无明显异常\nD01-A：不确定"):
        feedback_path.write_text(text)
        with pytest.raises(ValueError):
            module.import_offline_feedback(tmp_path, feedback_path)
        assert not (directory / "feedback_review").exists()
    feedback_path.write_text(
        f"试听包 {mapping['package_id']}\nD01-A：有明确异常；模糊\nD01-B：不确定；有些模糊\n"
        "其余条目未评；语言质量未验证。\n"
    )
    original_reports = {path: path.read_bytes() for path in directory.glob("*.json")}
    summary = module.import_offline_feedback(tmp_path, feedback_path)
    assert summary["rated"] == 0 and summary["by_lang"]["th"]["regressed"] is None
    assert summary["acoustic_by_lang"]["th"]["paired"] == 0
    imported = json.loads((directory / "feedback_review" / "mapped_feedback.json").read_text())
    by_code = {item["code"]: item for item in imported["items"]}
    assert by_code["D01-A"]["human_review"]["acoustic_status"] == "abnormal"
    assert by_code["D01-A"]["human_review"]["notes"] == "模糊"
    assert by_code["D01-B"]["human_review"]["acoustic_status"] is None
    for item in imported["items"]:
        report = json.loads((directory / "feedback_review" / f"reviewed_{item['model']}.json").read_text())
        assert report["items"][0]["human_review"]["notes"] == item["human_review"]["notes"]
        assert "naturalness_1_5" not in report["items"][0]["human_review"]
        assert report["items"][0]["components"] == item["components"]
    assert all(path.read_bytes() == contents for path, contents in original_reports.items())
    assert all(path.read_bytes() == contents for path, contents in frozen.items())
    with pytest.raises(FileExistsError):
        module.import_offline_feedback(tmp_path, feedback_path)


def test_diagnostic_export_preserves_sources_and_independent_controls(monkeypatch, tmp_path):
    from voxft import eval as evaluation

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("holdout_diagnostics", scripts / "holdout_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path / "reports")
    cases = [{"case_id": f"{lang}_{reference}", "text": "Test sentence.", "lang": lang,
              "ref_lang": reference, "ref_audio": f"/{reference}.wav", "numeric": False}
             for lang in ("th", "tl", "vi", "id", "ms") for reference in ("zh", "en", "tl")]
    seeds = [42, 43, 44, 45, 49]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    module.write_json(tmp_path / "plan.json", {
        "evaluation_seeds": seeds, "inputs_sha256": {str(cases_path): module.sha256(cases_path)},
    })
    originals = {}
    for model in ("base", "r8"):
        audio = tmp_path / f"{model}.wav"
        audio.touch()
        rows = [{**case, "seed": seed, "wav": str(audio), "cer": 0.1, "wer": 0.1, "hyp": "test",
                 "speaker_sim": 0.9, "audio_sec": 4.0, "f0_std_st": 1.0,
                 "human_review": {"acoustic_status": None}}
                for case in cases for seed in seeds]
        if model == "r8":
            for row in rows:
                if row["seed"] == 49:
                    row.update(cer=0.6, speaker_sim=0.5)
        source = tmp_path / f"{model}_report.json"
        module.write_json(source, {"items": rows, "cohorts": {"stale": 75}, "label": model})
        originals[source] = source.read_bytes()
    hashes = {model: module.sha256(tmp_path / f"{model}_report.json") for model in ("base", "r8")}
    paired = [{"case_id": case["case_id"], "cer": [0.1, 0.2], "speaker_sim": [0.9, 0.82]}
              for case in cases]
    module.write_json(tmp_path / "comparison.json", {
        "inputs_sha256": hashes, "r8_minus_base": {"by_case": paired},
    })
    _, indices = evaluation._review_reports(str(tmp_path / "base_report.json"),
                                            str(tmp_path / "r8_report.json"))
    selected = module.select_samples(cases, paired, indices)
    assert selected == module.select_samples(list(reversed(cases)), list(reversed(paired)), indices)
    altered = [{**row, "cer": [0.9, 0.1], "speaker_sim": [0.4, 0.9]} for row in paired]
    assert selected["random_control"] == module.select_samples(cases, altered, indices)["random_control"]
    assert len(selected["diagnostic"]) == 10
    assert {row["seed"] for row in selected["diagnostic"]} == {49}
    assert {row["seed"] for row in selected["random_control"]} == {42}
    missing = tmp_path / "r8.wav"
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        module.collect(tmp_path)
    assert not (tmp_path / "diagnostic_sample").exists()
    missing.touch()
    source = tmp_path / "base_report.json"
    source.write_text("changed")
    with pytest.raises(ValueError, match="report changed"):
        module.collect(tmp_path)
    source.write_bytes(originals[source])
    module.collect(tmp_path)
    saved = json.loads((tmp_path / "diagnostic_sample/reports.json").read_text())
    metadata = json.loads((tmp_path / "diagnostic_sample/plan.json").read_text())
    assert len(metadata["overlap_case_ids"]) == 10
    for group, paths in saved.items():
        session = evaluation.review_session(paths["base"], paths["r8"])
        assert len(session) == len(selected[group])
        for path in paths.values():
            report = json.loads(Path(path).read_text())
            assert "cohorts" not in report
            assert sum(row["cases"] for row in report["by_lang"].values()) == len(selected[group])
            assert all(row["human_review"]["acoustic_status"] is None for row in report["items"])
    assert all(path.read_bytes() == data for path, data in originals.items())
    with pytest.raises(FileExistsError):
        module.collect(tmp_path)
    from types import SimpleNamespace
    from voxft.data import pipeline

    calls = []

    class Whisper:
        def transcribe(self, wav, **kwargs):
            calls.append((wav, kwargs))
            return [SimpleNamespace(start=0, end=4, text="Test sentence.")], SimpleNamespace()

    monkeypatch.setattr(module, "runtime_environment", lambda work: None)
    monkeypatch.setattr(pipeline, "_whisper_model", lambda *args: Whisper())
    module.recheck_asr(tmp_path)
    result = json.loads((tmp_path / "asr_recheck/results.json").read_text())
    assert result["complete"] is True and len(result["items"]) == 12
    assert all(row["cer"] == 0 for row in result["items"])
    for original, variant in zip(calls[::2], calls[1::2]):
        assert original[0] == variant[0]
        assert original[1]["vad_filter"] is True and variant[1]["vad_filter"] is False
        assert {**original[1], "vad_filter": False} == variant[1]
    assert all(path.read_bytes() == data for path, data in originals.items())
    with pytest.raises(FileExistsError):
        module.recheck_asr(tmp_path)


def test_local_voice_probe_reproduces_raw_and_preserves_source_audio(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from voxft.data import pipeline
    from voxft.qc import audio as quality

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("local_voice", scripts / "holdout_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from voice_stability import summarize_embeddings

    monkeypatch.setattr(module.evaluation, "EVAL_DIR", tmp_path / "review_reports")
    (tmp_path / "done").touch()
    (tmp_path / "diagnostic_sample").mkdir()
    (tmp_path / "voice_stability").mkdir()
    model_dir = tmp_path / "speaker_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setenv("VOXFT_SPK_EMB_MODEL", str(model_dir))
    audio = tmp_path / "source.wav"
    audio.write_bytes(b"untouched")
    hashes, originals = {}, {audio: audio.read_bytes()}
    for model in ("base", "r8"):
        rows = [{"case_id": case_id, "seed": seed, "lang": case_id.split("_")[2],
                 "text": case_id, "wav": str(audio), "ref_audio": str(audio),
                 "f0_std_st": 1., "numeric": False, "wer": None, "human_review": {}}
                for case_id in module.VOICE_CASES for seed in (42, 43, 44, 45, 49)]
        path = tmp_path / f"{model}_report.json"
        module.write_json(path, {"items": rows})
        hashes[model] = module.sha256(path)
        originals[path] = path.read_bytes()
        path = tmp_path / "voice_stability" / path.name
        module.write_json(path, summarize_embeddings(rows, np.tile([1., 0.], (20, 1))))
        originals[path] = path.read_bytes()
    module.write_json(tmp_path / "diagnostic_sample/plan.json", {"source_reports_sha256": hashes})
    monkeypatch.setattr(module, "runtime_environment", lambda work: None)
    monkeypatch.setattr(quality, "_spk_model", lambda: (
        None, SimpleNamespace(training=False, device="cpu"), SimpleNamespace(get_num_threads=lambda: 1), "test"))
    waveform = np.arange(1000, dtype=float)
    monkeypatch.setattr(pipeline, "load_wav_mono", lambda path: (waveform.copy(), 100))
    monkeypatch.setattr(quality, "edge_silence", lambda wav, sr: {
        "head_silence_sec": 1., "tail_silence_sec": 2.})
    lengths = []

    def embedding(wav, sample_rate):
        lengths.append(len(wav))
        if len(wav) == 300:
            assert any(np.array_equal(wav, waveform[start:start + 300]) for start in (100, 300, 500))
        else:
            assert np.array_equal(wav, waveform if len(wav) == 1000 else waveform[100:800])
        return np.array([1., 0.])

    monkeypatch.setattr(quality, "speaker_embedding", embedding)
    module.localize_voice(tmp_path)
    result = json.loads((tmp_path / "local_voice/results.json").read_text())
    assert result["complete"] and len(result["items"]) == 80
    assert lengths == [1000] * 40 + [700] * 40
    assert result["summaries"]["raw"]["r8"]["max_raw_reproduction_error"] == 0
    assert (tmp_path / "local_voice/embeddings.npz").is_file()
    assert all(path.read_bytes() == original for path, original in originals.items())
    with pytest.raises(FileExistsError):
        module.localize_voice(tmp_path)
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="baseline inputs changed"):
        module.localize_voice(tmp_path, windows=True)
    assert not (tmp_path / "voice_windows").exists()
    audio.write_bytes(originals[audio])
    module.localize_voice(tmp_path, windows=True)
    windows = json.loads((tmp_path / "voice_windows/results.json").read_text())
    window_plan = json.loads((tmp_path / "voice_windows/plan.json").read_text())
    assert "NOT word/phoneme aligned" in window_plan["note"]
    assert windows["complete"] and len(windows["items"]) == 120
    assert lengths == [1000] * 40 + [700] * 40 + [300] * 120
    assert all(row["embedded_audio_sec"] == 3 and row["cosine_to_own_full"] == 1
               for row in windows["items"])
    saved = windows["review_reports"]
    assert len(module.evaluation.review_session(saved["base"], saved["r8"])) == 20
    for path in saved.values():
        report = json.loads(Path(path).read_text())
        assert all(row["wav"] == str(audio) and row["human_review"] == {} for row in report["items"])
        assert report["items"] == sorted(report["items"], key=lambda row: (row["lang"], row["seed"], row["case_id"]))
    assert all(path.read_bytes() == original for path, original in originals.items())
    with pytest.raises(FileExistsError):
        module.localize_voice(tmp_path, windows=True)


def test_voice_windows_are_bounded_and_do_not_claim_word_alignment(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from holdout_diagnostics import voice_window_bounds
    edges = {"head_silence_sec": 0.1, "tail_silence_sec": 0.2}
    assert [voice_window_bounds(1000, 100, edges, position) for position in ("start", "middle", "end")] == [
        (10, 310), (345, 645), (680, 980)]
    for duration in (None, float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError):
            voice_window_bounds(1000, 100, {**edges, "tail_silence_sec": duration}, "start")
    with pytest.raises(ValueError, match="three seconds"):
        voice_window_bounds(300, 100, edges, "start")
    with pytest.raises(ValueError, match="Unknown"):
        voice_window_bounds(1000, 100, edges, "unknown")


def test_dual_cue_review_preserves_pcm_model_mapping_and_original_ratings(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf
    from voxft.ui.app import do_review_load

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import holdout_diagnostics as module

    monkeypatch.setattr(module.evaluation, "EVAL_DIR", tmp_path / "reports")
    previous = tmp_path / "voice_windows"
    previous.mkdir()
    samples, inputs, originals = {}, {}, {}
    for model_index, model in enumerate(("base", "r8")):
        rows = []
        for case_index, case_id in enumerate(module.VOICE_CASES):
            for seed in module.VOICE_SEEDS:
                audio = tmp_path / f"{model}_{case_id}_{seed}.wav"
                wave = np.arange(48 + seed, dtype=np.int16) + 1000 * (model_index + case_index)
                wave[0], wave[-1] = -32768, 32767
                sf.write(audio, wave, 48000, subtype="PCM_16")
                inputs[str(audio)] = module.sha256(audio)
                originals[audio] = audio.read_bytes()
                rows.append({"case_id": case_id, "seed": seed, "lang": case_id.split("_")[2],
                             "text": case_id, "wav": str(audio), "ref_audio": "same-reference",
                             "numeric": False, "cer": 0.2, "f0_std_st": 1.,
                             "human_review": {"acoustic_status": "abnormal"}})
        samples[model] = rows
        report = tmp_path / f"{model}_report.json"
        module.write_json(report, {"items": rows, "target": model, "cfg_value": 1.8,
                                  "inference_timesteps": 20, "retry_badcase": False})
        inputs[str(report)] = module.sha256(report)
        originals[report] = report.read_bytes()
    module.write_json(previous / "plan.json", {"samples": samples, "inputs_sha256": inputs})
    module.write_json(previous / "results.json", {
        "complete": True, "plan_sha256": module.sha256(previous / "plan.json")})
    source_audio = next(path for path in originals if path.suffix == ".wav")
    source_audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        module.dual_cue_review(tmp_path)
    assert not (tmp_path / "dual_cue").exists()
    source_audio.write_bytes(originals[source_audio])
    with monkeypatch.context() as patch:
        patch.setattr(sf, "info", lambda path: SimpleNamespace(
            format="WAV", subtype="PCM_16", samplerate=16000, channels=1, frames=100))
        with pytest.raises(ValueError, match="48kHz PCM_16"):
            module.dual_cue_review(tmp_path)
    assert not (tmp_path / "dual_cue").exists()
    module.dual_cue_review(tmp_path)
    result = json.loads((tmp_path / "dual_cue/results.json").read_text())
    assert result["complete"] and len(result["items"]) == 20
    assert all(row["pcm_samples_verified"] for row in result["items"])
    saved = result["review_reports"]
    for model, path in saved.items():
        report = json.loads(Path(path).read_text())
        assert report["review_only"] and "mean_cer" not in report and "seed_stability" not in report
        assert len(report["items"]) == 10
        with pytest.raises(ValueError, match="仅供试听"):
            module.evaluation.summarize_report(path)
        for item in report["items"]:
            assert item["human_review"]["acoustic_status"] is None and "cer" not in item
            assert model not in Path(item["wav"]).stem
            assert Path(item["wav"]).is_relative_to(module.evaluation.EVAL_DIR)
            actual, sample_rate = sf.read(item["wav"], dtype="int16")
            assert sample_rate == 48000
            assert sf.info(item["wav"]).subtype == "PCM_16"
            assert item["playback_duration_sec"] == len(actual) / 48000
            for component in item["components"]:
                assert Path(component["wav"]).name.startswith(model + "_")
                assert component["seed"] == item["seed"]
                source, _ = sf.read(component["wav"], dtype="int16")
                assert np.array_equal(actual[component["start_sample"]:component["end_sample"]], source)
            first, second = item["components"]
            assert second["start_sample"] - first["end_sample"] == 24000
            assert not actual[first["end_sample"]:second["start_sample"]].any()
    loaded = do_review_load(saved["base"], saved["r8"], 0)
    assert len(loaded[0]) == 10 and "双句声学复核" in loaded[3]
    reports, indices = module.evaluation._review_reports(saved["base"], saved["r8"])
    for blind_seed in (0, 1):
        session = module.evaluation.review_session(saved["base"], saved["r8"], blind_seed)
        for pair in session:
            for slot in ("1", "2"):
                who = pair[f"who_{slot}"]
                item = indices[who][pair["case_id"], pair["seed"]]
                assert pair[f"wav_{slot}"] == item["wav"]
                assert all(Path(part["wav"]).name.startswith(reports[who]["target"] + "_")
                           for part in item["components"])
    summary = module.evaluation.save_reviews(saved["base"], saved["r8"], loaded[0], {})
    assert summary["rated"] == 0 and all(
        value["a"]["unknown"] == value["b"]["unknown"] == 5
        for value in summary["acoustic_by_lang"].values())
    assert all(path.read_bytes() == original for path, original in originals.items())
    with pytest.raises(FileExistsError):
        module.dual_cue_review(tmp_path)
