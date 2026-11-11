import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("better,worse,stop", [
    (False, False, True), (True, True, True), (True, False, False),
])
def test_short_screen_gates_later_gpu_work(monkeypatch, tmp_path, better, worse, stop):
    script = Path(__file__).resolve().parents[1] / "scripts/r52_stage_20261104.py"
    spec = importlib.util.spec_from_file_location("r52_stage", script)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    monkeypatch.setattr(stage.pr, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(stage, "run", lambda *args: calls.append(("run", *args)))
    for name in ("gen_stage", "asr_stage_async", "finalize"):
        monkeypatch.setattr(stage.pr, name, lambda *args, name=name: calls.append((name, *args)))
    monkeypatch.setattr(stage.pr, "log", lambda *_: None)
    monkeypatch.setattr(stage.cand, "ctx", lambda *_: ({}, {}))
    monkeypatch.setattr(stage.pr, "analyze_phase", lambda *_: {"_totals": {}})

    def compare(*_args, **_kwargs):
        calls.append(("compare",))
        return {
            lang: {"better": better and lang == "th", "worse": worse and lang == "id",
                   "base_severe": 2, "cand_severe": 1}
            for lang in ("th", "tl", "id")
        }

    monkeypatch.setattr(stage.pr, "compare", compare)
    stage.main()
    stages = [call for call in calls if call[0] != "run"]
    assert stages[:4] == [
        ("gen_stage", "short", "r52"), ("asr_stage_async", "short", "r52"),
        ("finalize", "short", "r52"), ("compare",),
    ]
    result = json.loads((tmp_path / "r52_stage.json").read_text(encoding="utf-8"))
    if stop:
        assert len(stages) == 4
        assert result["stopped"]
        assert not any("prod_review_lean_final" in str(call) for call in calls)
    else:
        assert ("gen_stage", "th_codeswitch", "r52") in stages
        assert ("finalize", "dev_b", "r52") in stages
        assert result["candidates_for_final"] == ["r51", "soup", "r52"]
        assert calls[-1] == ("run", "scripts/prod_review_lean_final_20261104.py", "r51", "soup", "r52")
