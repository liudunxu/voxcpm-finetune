from contextlib import nullcontext
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("control_ab_strength")


def test_tasks_are_only_two_new_strengths_and_preserve_five_seeds(module):
    cases = [{"case_id": str(index)} for index in range(18)]
    jobs = list(module.tasks(cases))
    assert len(jobs) == len({identifier for identifier, *_ in jobs}) == 180
    assert {strength for _, strength, *_ in jobs} == {.5, .75}
    assert {seed for *_, seed in jobs} == {42, 43, 44, 45, 46}
    assert {task[0] for task in module.tasks(cases, 1)} == {task[0] for task in jobs}
    assert next(module.tasks(cases, 1))[1] == .75


def test_scaling_checks_real_tensor_values_and_restoration(module, monkeypatch):
    layers = [SimpleNamespace(scaling=torch.tensor(value), _base_scaling=value)
              for value in (1., 2.)]
    model = SimpleNamespace(tts_model=SimpleNamespace(_iter_lora_modules=lambda: iter(layers)))
    result = module.check_scaling(model)
    assert result["modules"] == 2 and result["restored"]
    assert [layer.scaling.item() for layer in layers] == [1., 2.]
    monkeypatch.setattr(module.frozen.evaluation.infer, "scaled_lora", lambda *args: nullcontext())
    with pytest.raises(ValueError, match="Effective"):
        module.check_scaling(model)
    layers[0].scaling.fill_(0)
    with pytest.raises(ValueError, match="unscaled"):
        module.check_scaling(model)


@pytest.fixture
def experiment(module, tmp_path, monkeypatch):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"original reference")
    cases = [{"case_id": f"{lang}_{group}", "sentence_id": f"{lang}_{group}", "lang": lang,
              "text": "hello", "ref_audio": str(reference), "ref_lang": "zh",
              "control": "", "cohort": "independent_text"}
             for lang in ("id", "th", "tl") for group in ("random", "tail_risk", "historical_risk")]
    selection = [{"case_id": case["case_id"], "sentence_id": case["sentence_id"],
                  "stratum": case["case_id"].split("_", 1)[1], "seed": 42} for case in cases]
    module._write_jsonl(cases, tmp_path / "cases.jsonl")
    plan = {"base_path": "/base", "target": "/on/latest", "package": str(tmp_path / "package"),
            "selection": selection, "max_workers": 4, "new_outputs": len(cases) * 10,
            "note": "No native review; development only",
            "inputs_sha256": {str(reference): module.frozen.sha256(reference)}}
    module.frozen.write_json(tmp_path / "plan.json", plan)
    for name in ("claims", "items", "package/feedback_review"):
        (tmp_path / name).mkdir(parents=True)
    module.frozen.write_json(tmp_path / "package/feedback_review/mapped_feedback.json", [{
        "code": "D05-A", "case_id": "tl_tail_risk", "seed": 42, "model": "on",
        "human_review": {"acoustic_status": "abnormal", "notes": "开头有些空白"}}])
    monkeypatch.setattr(module.frozen, "runtime_environment", lambda path: None)
    monkeypatch.setattr(module.fast, "cached_services", lambda cases: nullcontext())
    monkeypatch.setattr(module, "check_scaling", lambda model: {"restored": True})
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    monkeypatch.setattr(module.frozen.evaluation, "EVAL_DIR", tmp_path / "raw_reports")
    monkeypatch.setattr(module.frozen.evaluation.infer, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(module.frozen.evaluation.infer, "get_model", lambda *args: object())
    monkeypatch.setattr(module.fast.pipeline, "_whisper_model", lambda *args: object())
    monkeypatch.setattr(module.frozen.evaluation, "_transcribe", lambda *args: "hello")
    monkeypatch.setattr(module.frozen.evaluation, "_acoustics", lambda *args: {
        "audio_sec": 2, "speaker_sim": .9, "head_silence_sec": .1,
        "tail_silence_sec": .2, "speech_ratio": .8, "f0_std_st": 1.})
    monkeypatch.setattr(module.review, "level_metrics", lambda path: {
        "rms_dbfs": -30., "peak_dbfs": -20., "p90_frame_rms_dbfs": -25.})
    generated = []

    def generate(model, kwargs, *, lora_strength=None):
        assert kwargs["retry_badcase"] is False
        assert kwargs["cfg_value"] == 1.8 and kwargs["inference_timesteps"] == 20
        assert lora_strength in (.5, .75)
        generated.append((kwargs["text"], kwargs["seed"], lora_strength))
        path = tmp_path / f"output{len(generated)}.wav"
        path.write_bytes(b"original target")
        return str(path), .1

    monkeypatch.setattr(module.frozen.evaluation.infer, "_run", generate)
    return cases, plan, generated


def test_worker_persists_per_item_and_does_not_retry_claims(
        module, tmp_path, monkeypatch, experiment):
    cases, plan, generated = experiment
    evaluate = module.frozen.evaluation.evaluate

    def fail_second(*args, **kwargs):
        if generated:
            raise RuntimeError("deliberate")
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(module.frozen.evaluation, "evaluate", fail_second)
    with pytest.raises(RuntimeError, match="deliberate"):
        module.worker(tmp_path, 0)
    saved = tmp_path / "items/s0.5_00_42.json"
    before = saved.read_bytes()
    report = module.review.read_json(saved)
    module.verify_report(report, plan, cases[:1], .5, [42])
    monkeypatch.setattr(module.frozen.evaluation, "evaluate", evaluate)
    module.worker(tmp_path, 1)
    assert saved.read_bytes() == before
    assert len(generated) == len(cases) * 10 - 1
    assert (tmp_path / "claims/s0.5_00_43.json").exists()
    assert not (tmp_path / "items/s0.5_00_43.json").exists()


@pytest.mark.parametrize("fault", ["strength", "target", "text", "seed", "duplicate", "wav", "hash"])
def test_reject_changed_reports_or_audio(module, tmp_path, monkeypatch, experiment, fault):
    cases, plan, _ = experiment
    monkeypatch.setattr(module, "tasks", lambda cases, index=0:
                        iter([("single", .5, cases[0], 42)]))
    module.worker(tmp_path, 0)
    report = module.review.read_json(tmp_path / "items/single.json")
    if fault in ("strength", "target"):
        report["lora_strength" if fault == "strength" else "target"] = 0
    elif fault in ("text", "seed"):
        report["items"][0][fault] = "changed"
    elif fault == "duplicate":
        report["items"].append(deepcopy(report["items"][0]))
    elif fault == "wav":
        Path(report["items"][0]["wav"]).write_bytes(b"changed")
    else:
        report["wav_sha256"] = {}
    with pytest.raises(ValueError):
        module.verify_report(report, plan, cases[:1], .5, [42])


def test_plan_hash_and_completed_run_prevent_mutation_and_resume(module, tmp_path, experiment):
    (tmp_path / "run.started").write_text(module.frozen.sha256(tmp_path / "plan.json"))
    module.check_inputs(tmp_path)
    with pytest.raises(FileExistsError, match="implicit resume"):
        module.run(tmp_path)
    plan = module.review.read_json(tmp_path / "plan.json")
    plan["target"] = "/different"
    module.frozen.write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="Run plan"):
        module.check_inputs(tmp_path)


def test_collect_keeps_strata_old_reports_and_head_silence_feedback(
        module, tmp_path, experiment):
    cases, plan, generated = experiment
    module.worker(tmp_path, 0)
    snapshots = {}
    for name, strength in (("base", 0), ("on", 1)):
        parts = [module.review.read_json(tmp_path / "items" / f"{identifier}.json")
                 for identifier, value, _, _ in module.tasks(cases) if value == .5]
        report = deepcopy(parts[0])
        report.update(target="base" if strength == 0 else plan["target"], lora_strength=strength,
                      items=[part["items"][0] for part in parts])
        report["wav_sha256"] = {row["wav"]: module.frozen.sha256(row["wav"])
                                for row in report["items"]}
        path = tmp_path / f"{name}_report.json"
        module.frozen.write_json(path, report)
        snapshots[path] = path.read_bytes()
    module.collect(tmp_path)
    assert len(generated) == len(cases) * 10
    assert all(path.read_bytes() == data for path, data in snapshots.items())
    comparison = module.review.read_json(tmp_path / "comparison.json")
    assert set(comparison["by_stratum"]) == {"random", "tail_risk", "historical_risk"}
    assert len(comparison["paired"]) == len(cases) * 5
    for groups in comparison["by_stratum"].values():
        assert set(groups) == {"id", "th", "tl"}
        assert all(set(models) == {"base", "on", "s0.5", "s0.75"} for models in groups.values())
    assert comparison["reviewed_abnormal"][0]["human_review"]["notes"] == "开头有些空白"
    assert comparison["language_quality"] == "unverified"
    assert comparison["new_audio_human_review"] == "pending"
    with pytest.raises(FileExistsError):
        module.collect(tmp_path)


def test_prepare_never_overwrites(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path / "source", tmp_path / "package", tmp_path)


@pytest.mark.parametrize("fault", ["none", "reference", "missing_result", "cache"])
def test_prepare_freezes_exact_review_inputs_and_reuses_endpoints(
        module, tmp_path, monkeypatch, fault):
    source, package, output = (tmp_path / name for name in ("source", "package", "new"))
    source.mkdir()
    (package / "feedback_review").mkdir(parents=True)
    reference, audio, config, weight = (source / name for name in
                                        ("ref.wav", "target.wav", "on.yaml", "weights"))
    for path in (reference, audio, config, weight):
        path.write_bytes(b"frozen")
    cases = [{"case_id": f"{lang}_{index}", "sentence_id": f"{lang}_{index}",
              "lang": lang, "text": "hello", "control": "", "ref_audio": str(reference)}
             for lang in ("id", "th", "tl") for index in range(6)]
    selection = [{"case_id": case["case_id"], "sentence_id": case["sentence_id"],
                  "seed": 42, "stratum": "random" if index % 6 < 4
                  else ("tail_risk" if index % 6 == 4 else "historical_risk")}
                 for index, case in enumerate(cases)]
    module.frozen.write_json(package / "selection.json", {"items": selection})
    module.frozen.write_json(package / "package.json", {
        "complete": True, "package_id": "acoustic-5b297d423cc5", "source_sha256": {},
        "artifacts_sha256": {"selection.json": module.frozen.sha256(package / "selection.json")}})
    for name in ("mapped_feedback.json", "submitted.txt"):
        (package / "feedback_review" / name).write_text("[]")
    module.frozen.write_json(source / "trained_on.json", {
        "step": 393, "inputs_sha256": {str(weight): module.frozen.sha256(weight)}})
    monkeypatch.setattr(module, "TRAINED_ON_SHA256", module.frozen.sha256(source / "trained_on.json"))
    module.frozen.write_json(source / "fast_eval_profile.json", {
        "metrics_exactly_equal": fault != "cache", "source_sha256": module.frozen.sha256(module.fast.__file__)})
    old_plan = {"base_path": "/base", "num_iters": 393, "configs": {"on": str(config)},
                "targets": {"on": "/on/latest"},
                "inputs_sha256": {str(path): module.frozen.sha256(path) for path in (reference, config)}}
    reports = {}
    for name, strength in (("base", 0), ("on", 1)):
        report = {"target": "base" if strength == 0 else "/on/latest", "base": "/base",
                  "lora_strength": strength, "cfg_value": 1.8, "inference_timesteps": 20,
                  "retry_badcase": False, "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
                  "items": [{**case, "seed": seed, "numeric": False, "wer": 0, "cer": 0, "f0_std_st": 1.,
                             "wav": str(audio)} for case in cases for seed in module.frozen.SEEDS],
                  "wav_sha256": {str(audio): module.frozen.sha256(audio)}}
        reports[name] = report
        module.frozen.write_json(source / f"{name}_report.json", report)
    originals = {path: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setattr(module.review, "load_sources", lambda work: (old_plan, cases, reports, {}))
    if fault == "reference":
        reference.write_bytes(b"changed")
    elif fault == "missing_result":
        reports["on"]["items"].pop()
    if fault != "none":
        with pytest.raises(ValueError):
            module.prepare(source, package, output)
        assert not output.exists()
        return
    module.prepare(source, package, output)
    plan = module.check_inputs(output)
    assert plan["new_outputs"] == plan["reused_outputs"] == 180
    assert plan["training_steps"] == 0 and plan["production_changed"] is False
    assert module.frozen._read_manifest(output / "cases.jsonl") == cases
    for name in ("base", "on"):
        assert module.review.read_json(output / f"{name}_report.json")["items"] == reports[name]["items"]
    assert all(path.read_bytes() == data for path, data in originals.items())
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen input"):
        module.check_inputs(output)
