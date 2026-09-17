import importlib.util
import inspect
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from voxft import eval as evaluation, infer
from voxft.train import runlog


def test_lora_strength_endpoints_and_restore(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "test_lora_layer", infer.VOXCPM_REPO / "src/voxcpm/modules/layers/lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layer = module.LoRALinear(torch.nn.Linear(3, 2, bias=False), r=2, alpha=4, dropout=0)
    with torch.no_grad():
        layer.lora_A.fill_(0.2)
        layer.lora_B.fill_(0.3)
    inputs = torch.ones(1, 3)
    original = layer(inputs).detach().clone()
    model = SimpleNamespace(tts_model=SimpleNamespace(_iter_lora_modules=lambda: iter([layer])))
    layer.set_enabled(False)
    base = layer(inputs).detach().clone()
    for strength in (0, 0.5, 0.75, 1):
        with infer.scaled_lora(model, strength):
            assert torch.allclose(layer(inputs), base + strength * (original - base))
        assert layer.scaling.item() == 0
    layer.set_enabled(True)
    def fail(**kwargs):
        assert layer.scaling.item() == 1
        raise RuntimeError("generation failed")
    model.generate = fail
    monkeypatch.setattr(infer, "CHECKPOINT_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="generation failed"):
        infer._run(model, {}, lora_strength=0.5)
    assert torch.equal(layer(inputs), original)
    for strength in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            with infer.scaled_lora(model, strength):
                pytest.fail("invalid strength accepted")
    model.tts_model._iter_lora_modules = lambda: iter([])
    with pytest.raises(ValueError, match="LoRA"):
        with infer.scaled_lora(model, 0):
            pytest.fail("empty adapter accepted")


def test_seed_stability_missing_values_and_case_guards():
    items = [{"case_id": "sample", "lang": "th", "text": "text", "seed": seed,
              "speaker_sim": value, "cer": value, "audio_sec": value * 10}
             for seed, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], 42)]
    summary = evaluation.seed_stability(items)
    assert summary["overall"]["cases_with_5_seeds"] == 1
    assert summary["by_case"][0]["speaker_sim"] == {
        "n": 5, "std": pytest.approx(0.158114), "range": 0.4}
    assert summary == evaluation.seed_stability(list(reversed(items)))
    missing = [dict(items[0], speaker_sim=None), dict(items[1], speaker_sim=float("nan"))]
    assert evaluation.seed_stability(missing)["overall"]["speaker_sim"]["mean_std"] is None
    assert evaluation.seed_stability(items[:1])["by_case"][0]["audio_sec"]["range"] is None
    with pytest.raises(ValueError, match="重复"):
        evaluation.seed_stability(items + items[:1])
    with pytest.raises(ValueError, match="条件不一致"):
        evaluation.seed_stability(items + [dict(items[0], seed=99, ref_audio="different.wav")])


def test_strength_reports_merge_and_reaggregate(tmp_path, monkeypatch):
    from voxft.data import pipeline
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_whisper_model", lambda *args: object())
    monkeypatch.setattr(infer, "get_model", lambda *args: object())
    monkeypatch.setattr(infer, "_resolve_base", lambda *args: "base-model")
    calls = []
    monkeypatch.setattr(infer, "_run", lambda model, kwargs, **extra:
                        calls.append((kwargs, extra)) or ("fake.wav", 0.1))
    monkeypatch.setattr(evaluation, "_transcribe", lambda *args: "test")
    monkeypatch.setattr(evaluation, "_acoustics", lambda *args: {
        "f0_std_st": 1, "audio_sec": 2, "speaker_sim": 0.9})
    cases = [{"case_id": name, "text": "test", "lang": "ms"} for name in ("one", "two")]
    full = evaluation.evaluate("/r8/latest", "ms", cases, seeds=[42, 43],
                               cfg_value=1.8, lora_strength=0.5)
    assert full["lora_strength"] == 0.5 and "strength0.5" in full["label"]
    assert all(extra == {"lora_strength": 0.5} and not kwargs["retry_badcase"]
               and kwargs["cfg_value"] == 1.8 for kwargs, extra in calls)
    shards = [evaluation.evaluate("/r8/latest", "ms", cases, seeds=[42, 43],
                                 cfg_value=1.8, lora_strength=0.5, shard=(index, 2))
              for index in range(2)]
    merged = evaluation.merge_reports([report["report_path"] for report in shards])
    assert merged["seed_stability"] == full["seed_stability"]
    shards[1]["lora_strength"] = 0.75
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(shards[1]))
    with pytest.raises(ValueError, match="lora_strength"):
        evaluation.merge_reports([shards[0]["report_path"], str(bad)])
    source = tmp_path / "reviewed.json"
    full["items"][0]["human_review"]["notes"] = "keep this annotation"
    full["note"] = "legacy measurement description"
    source.write_text(json.dumps(full))
    before = source.read_bytes()
    summarized = evaluation.summarize_report(str(source))
    assert source.read_bytes() == before
    assert summarized["report_path"] != str(source)
    assert summarized["items"][0]["human_review"]["notes"] == "keep this annotation"
    assert summarized["note"] == "legacy measurement description"
    assert summarized["seed_stability"] == full["seed_stability"]
    assert evaluation.evaluate("base", "ms", cases)["lora_strength"] == 0
    with pytest.raises(ValueError, match="base"):
        evaluation.evaluate("base", "ms", cases, lora_strength=0.5)


def test_runlog_cer_default_and_cli_override(monkeypatch):
    for function in (runlog._regressed, runlog.build_record, runlog.append_record):
        assert inspect.signature(function).parameters["noise"].default == 0.05
    reports = [{"by_lang": {"th": {"mean_cer": 0.1}}},
               {"label": "r8", "by_lang": {"th": {"mean_cer": 0.13}}}]
    assert not runlog._regressed(reports)["r8"]["red"]
    assert runlog._regressed(reports, noise=0.005)["r8"]["red"]
    seen = []
    monkeypatch.setattr(runlog, "build_record", lambda *args: seen.append(args[-1]) or "")
    for options in ([], ["--noise", "0.02"]):
        monkeypatch.setattr(sys, "argv", ["runlog", "--run", "test", "--print", *options])
        runlog.main()
    assert seen == [0.05, 0.02]


def test_probe_selection_keeps_controls_and_holdout(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "quality_probe", infer.VOXCPM_REPO.parents[1] / "scripts/quality_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    items = [{"case_id": f"{lang}_{ref}_{index}", "lang": lang, "ref_lang": ref,
              "ref_audio": f"{ref}.wav", "text": "test", "seed": seed,
              "cer": index / 10, "wer": None, "numeric": False, "f0_std_st": 1}
             for lang in ("th", "tl", "vi", "id", "ms", "zh", "en")
             for ref in ("zh", "en", "tl") for index in range(2) for seed in probe.SEEDS]
    report = {"items": items, "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False}
    for filename in probe.SOURCES.values():
        (tmp_path / filename).write_text(json.dumps(report))
    probe.prepare(tmp_path / "first")
    probe.prepare(tmp_path / "second")
    plan = json.loads((tmp_path / "first/plan.json").read_text())
    assert plan == json.loads((tmp_path / "second/plan.json").read_text())
    assert len(plan["selected"]) == 24
    assert not set(plan["selected"]) & set(plan["held_out_case_ids"])
    assert set(plan["selected"]) | set(plan["held_out_case_ids"]) == {
        item["case_id"] for item in items}
    assert list(plan["selected"].values()).count("random_control") == 15
    with pytest.raises(FileExistsError):
        probe.prepare(tmp_path / "first")
    history = {"items": [dict(items[0], case_id="known_noise",
                              human_review={"noise": True})]}
    (tmp_path / "base_87eed5ce0e16475a9da9c1b9ea43b8b7.json").write_text(json.dumps(history))
    report["items"] += [dict(items[0], case_id=case_id) for case_id in
                        ("id_nat_11", "id_nat_24", "tl_nat_09", "tl_nat_22")]
    (tmp_path / probe.SOURCES["r8"]).write_text(json.dumps(report))
    requests = []
    monkeypatch.setattr(evaluation, "evaluate", lambda *args, **kwargs:
                        requests.append((args, kwargs)) or {"checked": True})
    probe.followup(tmp_path / "first", (0, 2))
    assert [(kwargs["cfg_value"], kwargs["inference_timesteps"]) for _, kwargs in requests] == [
        (1.8, 20), (1.6, 20), (1.8, 28)]
    assert all(kwargs["seeds"] == [42, 43, 44, 45, 49] for _, kwargs in requests)
    assert len(requests[0][0][2]) == 12


def test_acoustic_review_preserves_unknown_and_existing_ratings(tmp_path, monkeypatch):
    from voxft.ui import app
    monkeypatch.setattr(evaluation, "EVAL_DIR", tmp_path)
    for name in ("a", "b"):
        report = {"items": [{"case_id": "case", "lang": "th", "text": "test",
                             "seed": 42, "wav": f"{name}.wav",
                             "human_review": {"notes": "old", "cutoff": None}}]}
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    session = evaluation.review_session("a.json", "b.json")
    controls = app._rv_restore({}, None)
    assert controls[2:5] == [app._RV_UNKNOWN] * 3
    rating = app._rv_collect(*controls)
    assert rating["s1"]["cutoff"] is None and rating["s1"]["acoustic_status"] is None
    assert "未验证（没有成对评分）" in app.do_review_save(
        "a.json", "b.json", session, {}, 0, *controls)
    rating["s1"].update(acoustic_status="abnormal", acoustic_types=["爆音"], notes="甲 0.8s")
    rating["s2"]["acoustic_status"] = "clear"
    result = evaluation.save_reviews("a.json", "b.json", session,
                                     {evaluation.review_key(session[0]): rating})
    assert result["rated"] == 0
    assert result["by_lang"]["th"]["regressed"] is None
    assert result["by_lang"]["th"]["unrated"] == 1
    assert result["acoustic_by_lang"]["th"]["paired"] == 1
    restored = app.do_review_load("a.json", "b.json", 0)
    assert restored[10] == "有明确异常" and restored[11] == ["爆音"]
    repeated = evaluation.save_reviews("a.json", "b.json", session, {})
    assert repeated["acoustic_by_lang"] == result["acoustic_by_lang"]
    doc = json.loads((tmp_path / "a.json").read_text())
    doc["items"][0]["human_review"]["naturalness_1_5"] = 4
    (tmp_path / "a.json").write_text(json.dumps(doc))
    evaluation.save_reviews("a.json", "b.json", session, {})
    assert json.loads((tmp_path / "a.json").read_text())["items"][0]["human_review"][
        "naturalness_1_5"] == 4


def test_pairwise_voice_stability_separates_seeds_cues_and_refs():
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "voice_stability", infer.VOXCPM_REPO.parents[1] / "scripts/voice_stability.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = [{"case_id": case_id, "text": case_id, "seed": seed, "ref_audio": "ref.wav",
              "lang": "th"} for case_id in ("one", "two") for seed in (42, 43)]
    vectors = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    report = module.summarize_embeddings(items, vectors)
    assert report["cross_seed"]["by_lang"]["th"]["mean_cosine"] == 1
    assert report["cross_cue"]["by_lang"]["th"]["mean_cosine"] == 0
    for item in items[2:]:
        item["ref_audio"] = "different.wav"
    assert not module.summarize_embeddings(items, vectors)["cross_cue"]["groups"]


def test_duration_gate_uses_the_same_non_numeric_cohort_as_cer():
    items = [{"numeric": False, "audio_sec": 2, "cer": 0, "wer": None},
             {"numeric": True, "audio_sec": 4, "cer": 0, "wer": None}]
    base = {"label": "base", **evaluation._agg(items)}
    items[1].update(audio_sec=40, cer=4.7)
    changed = {"label": "changed", **evaluation._agg(items)}
    assert changed["mean_audio_sec"] == 21
    assert changed["mean_audio_sec_non_numeric"] == 2
    assert not runlog._duration_gates([base, changed])["changed"]["duration_inflation"]
    items[0]["audio_sec"] = 3
    stretched = {"label": "stretched", **evaluation._agg(items)}
    assert runlog._duration_gates([base, stretched])["stretched"]["duration_inflation"]
    del stretched["mean_audio_sec_non_numeric"]
    assert not runlog._duration_gates([base, stretched])["stretched"]["duration_inflation"]
