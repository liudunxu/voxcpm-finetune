"""极端跑飞回归集提取逻辑自测。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_runaway_regression")


def item(case, seed, ratio, lang="id", **kw):
    return {"case_id": case, "seed": seed, "lang": lang, "len_ratio": ratio, "text": "teks",
            "control": "", "ref_lang": "zh", "ref_audio": "/refs/r.wav", "numeric": False,
            "audio_sec": round(2.5 * ratio, 2), "cer": 0.1, "over_read": ratio > 1.4, **kw}


def write_report(path, items, label="m"):
    path.write_text(json.dumps({"label": label, "items": items}, ensure_ascii=False), encoding="utf-8")


def test_keeps_only_over_threshold_and_merges_same_case_seed(module, tmp_path):
    write_report(tmp_path / "a.json", [item("c1", 44, 7.97), item("c2", 42, 1.10)])
    write_report(tmp_path / "b.json", [item("c1", 44, 3.00), item("c3", 43, 1.90)], label="m2")
    rows = module.collect({"rA": [("dramaprod", "a.json")], "rB": [("dramaprod", "b.json")]},
                          tmp_path, 2.0)
    assert [r["source_case_id"] for r in rows] == ["c1"], "1.90× 在阈值下，不该被收进来"
    row = rows[0]
    assert row["case_id"] == "runaway_c1_s44"
    assert row["seed"] == 44 and row["training_eligible"] is False
    assert row["language_quality"] == "unverified"
    assert {o["model"] for o in row["observed"]} == {"rA", "rB"}, "同 (case,seed) 跨模型应合并成一条"
    assert "rA" in row["note"] and "7.97" in row["note"]


def test_missing_report_is_skipped_not_fatal(module, tmp_path, capsys):
    write_report(tmp_path / "a.json", [item("c1", 44, 4.2)])
    rows = module.collect({"rA": [("dramaprod", "a.json"), ("v2", "nope.json")]}, tmp_path, 2.0)
    assert len(rows) == 1
    assert "跳过缺失报告" in capsys.readouterr().err


def test_main_refuses_to_overwrite_existing_output(module, tmp_path, monkeypatch):
    write_report(tmp_path / "a.json", [item("c1", 44, 4.2)])
    monkeypatch.setattr(module, "DEFAULT_REPORTS", {"rA": [("dramaprod", "a.json")]})
    out = tmp_path / "regression.jsonl"
    assert module.main(root=tmp_path, out=out, len_ratio_min=2.0) == 0
    first = out.read_text(encoding="utf-8")
    assert module.main(root=tmp_path, out=out, len_ratio_min=2.0) == 1, "已有产物必须拒绝覆盖"
    assert out.read_text(encoding="utf-8") == first


def test_main_writes_nothing_when_no_hit(module, tmp_path, monkeypatch):
    write_report(tmp_path / "a.json", [item("c1", 44, 1.05)])
    monkeypatch.setattr(module, "DEFAULT_REPORTS", {"rA": [("dramaprod", "a.json")]})
    out = tmp_path / "regression.jsonl"
    assert module.main(root=tmp_path, out=out, len_ratio_min=2.0) == 1
    assert not out.exists()
