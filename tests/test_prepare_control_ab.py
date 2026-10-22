import importlib
import json
from pathlib import Path

import pytest
import yaml


def test_control_ab_changes_only_verified_prefixes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("prepare_control_ab")
    source = tmp_path / "source"
    source.mkdir()
    common = {"source_id": "thai_ser", "lang": "th", "duration": 4}
    controlled = {**common, "audio": "/target.wav", "origin_audio": "/raw_target.wav",
                  "text": "(angry)hello", "control": "angry", "emotion_verified": True}
    excluded = {**common, "audio": "/excluded.wav", "origin_audio": "/raw_excluded.wav",
                "text": "excluded"}
    validation = {**common, "audio": "/validation.wav", "origin_audio": "/raw_validation.wav",
                  "text": "validation"}
    (source / "train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [controlled, excluded]), encoding="utf-8")
    (source / "val.jsonl").write_text(json.dumps(validation), encoding="utf-8")
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"lora": {"enable_proj": False}, "batch_size": 2,
                                       "grad_accum_steps": 8, "learning_rate": 1e-4,
                                       "warmup_steps": 120}), encoding="utf-8")
    monkeypatch.setattr(module, "load_exclusions", lambda directory: ({"excluded"}, {}))
    output = tmp_path / "prepared"
    plan = module.prepare(source, output, template, tmp_path)
    on = module._read_manifest(output / "on.jsonl")
    off = module._read_manifest(output / "off.jsonl")
    assert on == [controlled]
    assert off == [{**controlled, "text": "hello", "control": ""}]
    assert plan["training_started"] is False and len(plan["pending_gates"]) == 3
    assert len(plan["removed_source_rows"]) == 1
    assert (output / "validation.jsonl").read_bytes() == (source / "val.jsonl").read_bytes()
    configs = [yaml.safe_load((output / f"{arm}.yaml").read_text()) for arm in ("off", "on")]
    paths = {"train_manifest", "save_path", "tensorboard"}
    assert {key: value for key, value in configs[0].items() if key not in paths} == {
        key: value for key, value in configs[1].items() if key not in paths}
    with pytest.raises(FileExistsError):
        module.prepare(source, output, template, tmp_path)
    controlled["emotion_verified"] = False
    (source / "train.jsonl").write_text(json.dumps(controlled), encoding="utf-8")
    with pytest.raises(ValueError, match="verified"):
        module.prepare(source, tmp_path / "rejected", template, tmp_path)
    assert not (tmp_path / "rejected").exists()
