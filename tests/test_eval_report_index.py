"""评测报告索引/分类工具自测。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("eval_report_index")


def report(label, case_prefix, n_cases, langs=("id",), cfg=1.8, steps=20, shard=False):
    items = [{"case_id": f"{case_prefix}{i}", "seed": 42 + (i % 3), "lang": langs[i % len(langs)],
              "cer": 0.01 * i, "audio_sec": 3.0} for i in range(n_cases)]
    tag = case_prefix.rstrip("_").replace("_", "-")  # 文件名必须唯一，否则会互相覆盖
    name = f"{label}_{tag}_shard0of3_abc.json" if shard else f"{label}_{tag}_abc123.json"
    return name, {"label": label, "items": items, "cfg_value": cfg, "inference_timesteps": steps,
                  "lora_strength": 0.0 if label == "base" else 1.0}


def write(directory, entries):
    directory.mkdir(parents=True, exist_ok=True)
    for name, doc in entries:
        (directory / name).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def test_classifies_by_content_not_filename(module, tmp_path):
    """文件名里的 uuid 与内容无关；drama_chain 与 drama_chain_prod 只靠 case_id 前缀区分。"""
    d = tmp_path / "reports"
    write(d, [
        report("base", "drama_prod_id_nat_", 112, ("id", "th", "tl", "en")),
        report("base", "drama_id_nat_", 112, ("id", "th", "tl", "en")),
        report("base", "id_ms_shared_", 18),
        report("base", "id_nat_", 118, ("id", "th", "tl", "zh", "en")),
    ])
    rows = {r["set"]: r for r in (module.inspect(p) for p in sorted(d.glob("*.json")))}
    assert set(rows) == {"drama_chain_prod", "drama_chain", "id_prosody_probe", "omnivoice_prod_v2"}
    assert rows["drama_chain_prod"]["cases"] == 112
    assert rows["id_prosody_probe"]["langs"] == ["id"]


def test_shards_are_separated_from_merged(module, tmp_path):
    d = tmp_path / "reports"
    write(d, [report("ck", "id_nat_", 118), report("ck", "id_nat_", 40, shard=True)])
    rows = [module.inspect(p) for p in sorted(d.glob("*.json"))]
    assert sum(1 for r in rows if r["shard"]) == 1
    assert sum(1 for r in rows if not r["shard"]) == 1


def test_flags_duplicates_and_off_spec_reports(module, tmp_path, capsys):
    """同 (label, 验收集) 多份、以及非 cfg1.8/20步 的报告都必须被显式标出来。"""
    d = tmp_path / "reports"
    write(d, [
        report("ck", "id_nat_", 118),
        ("ck_renamed.json", json.loads(json.dumps(report("ck", "id_nat_", 118)[1]))),
        report("odd", "id_nat_", 118, cfg=2.0, steps=32),
    ])
    assert module.main(d) == 0
    out = capsys.readouterr().out
    assert "有多份合并报告" in out and "ck_renamed.json" in out
    assert "口径不是 cfg1.8/20步" in out and "cfg=2.0 steps=32" in out


def test_missing_directory_and_no_reports(module, tmp_path, capsys):
    assert module.main(tmp_path / "nope") == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert module.main(empty) == 1
    assert "没有可识别的评测报告" in capsys.readouterr().err
