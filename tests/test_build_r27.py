"""r27（等比放大）构建脚本的守卫自测。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_r27_scale25h")


def row(name, lang, src, dur, text=None, **kw):
    """dur 的单位是秒（清单口径），有效范围 3–30s。"""
    return {"audio": f"/audio/{name}.wav", "origin_audio": f"/raw/{name}.wav",
            "text": text if text is not None else f"teks {name}" + " kata" * int(dur), "duration": dur,
            "lang": lang, "source_id": src, "speaker": f"spk_{name}", "session": f"ses_{name}",
            "rate": 2.0, **kw}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


BASE_ID = row("id0", "id", "srcA", 30.0)
BASE_TL = row("tl0", "tl", "srcT", 30.0)
BASE_HOURS = 60 / 3600  # 继承清单共 60s


@pytest.fixture
def pools(tmp_path):
    """base 共 60s（id/srcA 30s + tl/srcT 30s）；每个源池另有 2 条 15s 新行（合计 30s）。"""
    base = tmp_path / "base"
    write_jsonl(base / "train.jsonl", [BASE_ID, BASE_TL])
    (base / "val.jsonl").write_bytes(b'{"audio": "/audio/v.wav", "text": "val"}\n')

    src = tmp_path / "sources"
    write_jsonl(src / "srcA" / "train.jsonl",
                [BASE_ID, row("idN0", "id", "srcA", 15.0), row("idN1", "id", "srcA", 15.0)])
    write_jsonl(src / "srcT" / "train.jsonl", [
        BASE_TL, row("tlN0", "tl", "srcT", 15.0), row("tlN1", "tl", "srcT", 15.0),
        row("tlBAD", "tl", "srcT", 10.0, text="FROZEN SENTENCE"),
        row("tlLONG", "tl", "srcT", 30.5),
    ])
    # 注意：_norm 会去掉空格与标点，排除集必须按归一化后的形态给（既有 test_incremental_mix 同理）
    return {"base": base, "src": src, "excluded": {"frozensentence"}}


def read_out(dst):
    return [json.loads(x) for x in (dst / "train.jsonl").read_text(encoding="utf-8").splitlines()]


def test_scales_proportionally_and_applies_guards(module, pools, tmp_path):
    dst = tmp_path / "r27"
    assert module.main(pools["base"], pools["src"], dst, 2 * BASE_HOURS, pools["excluded"]) == 0
    out = read_out(dst)
    base = read_out(pools["base"])
    assert out[:len(base)] == base, "继承行必须逐字段原样、且排在最前"
    added = out[len(base):]
    by_src = {}
    for r in added:
        by_src[r["source_id"]] = by_src.get(r["source_id"], 0) + 1
    assert by_src == {"srcA": 2, "srcT": 2}, by_src
    assert all(r["text"] != "FROZEN SENTENCE" for r in added), "命中评测文本的行必须被挡"
    assert all(3 <= r["duration"] <= 30 for r in added), "新增行必须落在 3–30s"

    audit = json.loads((dst / "build_audit.json").read_text(encoding="utf-8"))
    assert audit["inherited"] == 2 and audit["added"] == 4
    assert audit["rejected_candidate_reasons"]["existing_or_duplicate_target"] == 2
    assert audit["rejected_candidate_reasons"]["evaluation_or_validation_text"] == 1
    assert audit["rejected_candidate_reasons"]["duration_outside_3_30s"] == 1
    assert audit["max_dur"] == 30.0
    assert (dst / "val.jsonl").read_bytes() == (pools["base"] / "val.jsonl").read_bytes()

    mix = json.loads((dst / "mix.json").read_text(encoding="utf-8"))
    assert mix["inherited_rows"] == 2 and mix["added_rows"] == 4
    assert mix["shortfalls"] == []
    assert abs(mix["total_hours"] - 2 * BASE_HOURS) < 1e-3, "scale=2 ⇒ 总量应是继承的两倍"
    # 等比放大的定义：份额与继承清单一致
    assert abs(mix["language_shares"]["id"]["actual"] - 0.5) < 0.02
    assert abs(mix["language_shares"]["tl"]["actual"] - 0.5) < 0.02


def test_inoculation_keeps_prefix_shape(module, pools, tmp_path):
    dst = tmp_path / "r27"
    module.main(pools["base"], pools["src"], dst, 2 * BASE_HOURS, pools["excluded"])
    out = read_out(dst)
    mix = json.loads((dst / "mix.json").read_text(encoding="utf-8"))
    prefixed = [r for r in out if r["text"].startswith("(")]
    assert sum(mix["prefix_rows_total"].values()) == len(prefixed)
    for r in prefixed:
        assert r["control"] in module.NEUTRAL_PHRASES
        assert r["control_inoculation"] is True
        expected = row(Path(r["audio"]).stem, r["lang"], r["source_id"], r["duration"])["text"]
        assert r["text"] == f"({r['control']}){expected}"
    assert len(prefixed) <= 2, "10% 接种不该把整批都加前缀"


def test_shortfall_is_reported_not_redistributed(module, pools, tmp_path):
    """源池不够时照实少加并记录缺口，绝不用别的源/语种把总量凑上去（AGENTS.md：缺口不重分配）。"""
    dst = tmp_path / "r27"
    target = 4 * BASE_HOURS  # scale=4 ⇒ 每源需 +90s，但池里只有 30s
    assert module.main(pools["base"], pools["src"], dst, target, pools["excluded"]) == 0
    mix = json.loads((dst / "mix.json").read_text(encoding="utf-8"))
    assert len(mix["shortfalls"]) == 2, mix["shortfalls"]
    assert mix["total_hours"] < target - 1e-6, "没凑够就是没凑够，不许用别的源补"
    assert abs(mix["language_shares"]["id"]["actual"] - 0.5) < 0.02, "份额不能被挪动"
    assert abs(mix["language_shares"]["tl"]["actual"] - 0.5) < 0.02


def test_refuses_overwrite_and_dry_run_writes_nothing(module, pools, tmp_path):
    dst = tmp_path / "r27"
    dst.mkdir()
    assert module.main(pools["base"], pools["src"], dst, 2 * BASE_HOURS, pools["excluded"]) == 1
    assert not (dst / "train.jsonl").exists()

    dry = tmp_path / "dry"
    assert module.main(pools["base"], pools["src"], dry, 2 * BASE_HOURS, pools["excluded"],
                       dry_run=True) == 0
    assert not dry.exists()
