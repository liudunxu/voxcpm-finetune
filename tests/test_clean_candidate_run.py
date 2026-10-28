import copy
import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("clean_candidate_run")


def row(name, **extra):
    return {"audio": f"/{name}.wav", "origin_audio": f"/raw/{name}.wav",
            "text": "A complete natural sentence", "lang": "id", **extra}


def test_encoding_quarantine_propagates_to_pcm_alias_and_reference(module):
    bad = row("bad", text="An event costs Â¥2,500")
    alias = row("alias")
    reference = row("reference", ref_audio="/alias.wav")
    clean = row("keep", text="São Paulo")
    validation = [row("validation")]
    train = [bad, alias, reference, clean]
    original = copy.deepcopy(train)
    audio = {r["audio"]: {"pcm_sha256": r["audio"]} for r in train + validation}
    audio[alias["audio"]] = audio[bad["audio"]]
    result, removed, prefixes = module.clean_rows(train, validation, audio)
    assert result == {"train": [clean], "val": validation}
    assert len(removed) == 3 and "quarantined_ref_audio" in removed[2]["reasons"]
    assert not prefixes and train == original


def test_only_unverified_control_prefix_is_removed(module):
    unverified = row("auto", text="(calm)Keep (the literal parentheses).", control="calm",
                     control_inoculation=True, emotion_verified=False, metadata="preserve")
    verified = row("verified", text="(sad)The verified words.", control="sad", control_verified=True)
    validation = [row("validation")]
    audio = {r["audio"]: {"pcm_sha256": r["audio"]} for r in [unverified, verified, *validation]}
    result, removed, prefixes = module.clean_rows([unverified, verified], validation, audio)
    assert result["train"][0] == {k: ("Keep (the literal parentheses)." if k == "text" else
                                    "" if k == "control" else v)
                                  for k, v in unverified.items() if k != "control_inoculation"}
    assert result["train"][1] == verified and len(prefixes) == 1 and not removed
    with pytest.raises(ValueError, match="mismatch"):
        module.clean_rows([row("mismatch", control="sad")], validation,
                          {**audio, "/mismatch.wav": {"pcm_sha256": "mismatch"}})


def test_validation_encoding_is_excluded_but_not_rewritten(module):
    train, val = [row("train")], [row("bad", text="For Â£27 million"), row("good")]
    original = copy.deepcopy(val)
    audio = {r["audio"]: {"pcm_sha256": r["audio"]} for r in train + val}
    result, removed, _ = module.clean_rows(train, val, audio)
    assert result["val"] == [val[1]] and removed[0]["split"] == "val"
    assert val == original


def test_one_epoch_recipe_and_fresh_output_guards(module, tmp_path):
    template = Path(__file__).resolve().parents[1] / "configs/lora_omni5_r26_r18_inoc10.yaml"
    (tmp_path / "train.jsonl").write_text("{}\n" * 7333, encoding="utf-8")
    config = module.candidate_config(template, tmp_path)
    old = yaml.safe_load(template.read_text(encoding="utf-8"))
    assert config["num_iters"] == config["max_steps"] == 459
    assert config["warmup_steps"] == 45
    assert config["lora"] == old["lora"] and not config["lora"]["enable_proj"]
    assert config["pretrained_path"] == old["pretrained_path"]
    assert config["learning_rate"] == 1e-4 and config["batch_size"] * config["grad_accum_steps"] == 16
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
    (tmp_path / "checkpoint").mkdir()
    with pytest.raises(FileExistsError):
        module.frozen.require_fresh_checkpoint(config)


def test_four_shards_preserve_all_conditions_and_seeds(module):
    cases = [{"case_id": f"{lang}_{i}", "lang": lang} for lang in ("th", "tl", "id") for i in range(60)]
    jobs = module.tasks(cases)
    shards = [jobs[i::4] for i in range(4)]
    assert len(jobs) == 900 and all(len(shard) == 225 for shard in shards)
    assert len({job[0] for shard in shards for job in shard}) == 900
    assert all(case["lang"] == "id" for _, case, _ in jobs[:300])
    assert {seed for _, _, seed in jobs} == {42, 43, 44, 45, 46}


def test_changed_inputs_or_incomplete_checkpoint_are_rejected(module, tmp_path):
    protected = tmp_path / "input"
    protected.write_text("frozen", encoding="utf-8")
    plan = {"inputs_sha256": {str(protected): module.frozen.sha256(protected)},
            "training_authorized": True, "epochs": 1, "generation_budget": 900, "num_iters": 459}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "trained.json").write_text(json.dumps({"step": 458}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        module.checked(tmp_path, trained=True)
    protected.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        module.checked(tmp_path)
