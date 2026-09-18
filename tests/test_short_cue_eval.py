import importlib.util
import json
from pathlib import Path

import pytest


def load_module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("short_cue_eval", scripts / "short_cue_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_requires_complete_unchanged_conditions(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    cases = tmp_path / "cases.jsonl"
    case = {"case_id": "short_tl", "source_sentence_id": "14", "text": "Kumusta ka?",
            "lang": "tl", "ref_audio": "/ref.wav"}
    cases.write_text(json.dumps(case) + "\n")
    wav = tmp_path / "output.wav"
    wav.write_bytes(b"audio")
    plan = {"evaluation_seeds": module.SEEDS, "base_path": "/base",
            "cfg_value": 1.8, "inference_timesteps": 20}
    job = {"target": "base", "cases": str(cases)}
    report = {"target": "base", "base": "/base", "lora_strength": 0,
              "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
              "asr_model": "large-v3", "asr_auto_detect_langs": ["tl"],
              "items": [{**case, "seed": seed, "wav": str(wav)} for seed in module.SEEDS]}
    module.validate_report(plan, job, report)
    for change in ("missing", "duplicate", "seed", "text", "ref_audio", "source_sentence_id",
                   "cfg_value", "target", "wav"):
        changed = json.loads(json.dumps(report))
        if change == "missing":
            changed["items"].pop()
        elif change == "duplicate":
            changed["items"].append(changed["items"][0])
        elif change in ("cfg_value", "target"):
            changed[change] = "changed"
        else:
            changed["items"][0][change] = "changed"
        with pytest.raises(ValueError):
            module.validate_report(plan, job, changed)
    with pytest.raises(ValueError, match="Only one or two"):
        module.prepare(tmp_path, tmp_path / "invalid", 3)
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, tmp_path, 2)


def test_worker_failure_stops_sibling_and_never_starts_next_wave(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    processes = []
    monkeypatch.setattr(module, "prepare", lambda *args: {"jobs": dict.fromkeys(
        ("base_short", "r8_short", "control_short", "short_short"))})
    monkeypatch.setattr(module, "runtime_environment", lambda path: None)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(processes)
            self.returncode = None if not processes else -11
            self.terminated = False
            processes.append(self)
            if len(processes) == 1:
                (tmp_path / "base_short_progress.json").write_text('{"completed": 1}')

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="Evaluation worker failed"):
        module.run(tmp_path, tmp_path, 2)
    assert len(processes) == 2 and processes[0].terminated
    assert not processes[1].terminated
    assert not (tmp_path / "control_short.log").exists()
    assert not (tmp_path / "done").exists()
    assert json.loads((tmp_path / "status.json").read_text())["stage"] == "failed"
