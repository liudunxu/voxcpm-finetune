import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_short_pilot_changes_only_tl_and_never_repeats_audio(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("short_cue_ab", scripts / "short_cue_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pools = {
        source: [{"source_id": source, "duration": duration, "origin_audio": f"/{source}/{index}.wav"}
                 for index in range(7000) for duration in [4 if index < 600 else 12]]
        for source in ("fleurs_tl", "other")
    }
    arms = module.make_arms(pools, [("fleurs_tl", 17), ("other", 83)])
    assert arms == module.make_arms(pools, [("fleurs_tl", 17), ("other", 83)])
    common = lambda arm: sorted(row["origin_audio"] for row in arm if row["source_id"] != "fleurs_tl")
    assert common(arms["control"]) == common(arms["short"])
    shares = {}
    for name, rows in arms.items():
        assert len(rows) == len({row["origin_audio"] for row in rows})
        assert sum(row["duration"] for row in rows) / 3600 == pytest.approx(12, abs=0.01)
        tl = [row for row in rows if row["source_id"] == "fleurs_tl"]
        shares[name] = sum(row["duration"] for row in tl if row["duration"] <= 8) / sum(
            row["duration"] for row in tl)
    assert shares["short"] == pytest.approx(0.25, abs=0.001)
    assert shares["control"] < shares["short"]
    with pytest.raises(ValueError, match="Insufficient"):
        module.take_seconds(pools["fleurs_tl"][:1], 100, "test")
    with pytest.raises(ValueError, match="Duplicate"):
        module.make_arms({name: [{**row, "origin_audio": "/shared.wav"} for row in rows]
                          for name, rows in pools.items()}, [("fleurs_tl", 17), ("other", 83)])


def test_training_failure_stops_before_second_arm(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("short_cue_failure", scripts / "short_cue_ab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"save_path": str(tmp_path / "checkpoint"), "tensorboard": "/unused"}))
    (tmp_path / "checkpoint").mkdir()
    plan = {group: {} for group in ("inputs_sha256", "audio_sha256", "frozen_files", "source_sha256")}
    plan["configs"] = {arm: {"path": str(config), "sha256": module.sha256(config)} for arm in module.ARMS}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    calls = []
    monkeypatch.setattr(module, "runtime_environment", lambda work: None)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda work: SimpleNamespace(free=10 * 1024**3))
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(
        terminate=lambda: calls.append("bridge_terminated"), wait=lambda **kwargs: 0))

    def fail(*args, **kwargs):
        calls.append("train")
        raise module.subprocess.CalledProcessError(17, args[0])

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(module.subprocess.CalledProcessError):
        module.run(tmp_path)
    assert calls == ["train", "bridge_terminated"]
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "failed"
    assert not (tmp_path / "train_short.log").exists()
    with pytest.raises(FileExistsError, match="already started"):
        module.run(tmp_path)
