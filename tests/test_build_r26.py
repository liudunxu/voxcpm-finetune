"""r26（r25 的 B 臂，10% 中性前缀接种）构建脚本的守卫自测。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_r26_r18_inoc10")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def r18(tmp_path):
    """40 条 id + 20 条 tl + 5 条 th：够让 10% 接种稳定命中，也够验证 th 不参与。"""
    rows = []
    for i in range(40):
        rows.append({"audio": f"/audio/id{i}.wav", "origin_audio": f"/raw/id{i}.wav",
                     "text": f"kalimat nomor {i}", "duration": 5.0, "lang": "id",
                     "source_id": "gigaspeech2_id", "rate": 2.1})
    for i in range(20):
        rows.append({"audio": f"/audio/tl{i}.wav", "origin_audio": f"/raw/tl{i}.wav",
                     "text": f"pangungusap {i}", "duration": 11.0, "lang": "tl",
                     "source_id": "fleurs_tl", "rate": 2.0})
    for i in range(5):
        rows.append({"audio": f"/audio/th{i}.wav", "origin_audio": f"/raw/th{i}.wav",
                     "text": f"ประโยค {i}", "duration": 4.0, "lang": "th",
                     "source_id": "gigaspeech2_th", "rate": 12.0})
    d = tmp_path / "r18"
    write_jsonl(d / "train.jsonl", rows)
    (d / "val.jsonl").write_bytes(b'{"audio": "/audio/v.wav", "text": "val"}\n')
    cases = tmp_path / "cases.jsonl"
    write_jsonl(cases, [{"case_id": "x", "lang": "id", "text": "kalimat nomor 3"}])
    return {"src": d, "cases": [str(cases)], "rows": rows}


def read_out(dst):
    return [json.loads(x) for x in (dst / "train.jsonl").read_text(encoding="utf-8").splitlines()]


def test_inoculation_touches_only_prefix_fields(module, r18, tmp_path):
    dst = tmp_path / "r26"
    assert module.main(r18["src"], dst, r18["cases"], tmp_path / "nope") == 0
    out = read_out(dst)
    assert len(out) == len(r18["rows"])
    assert [r["audio"] for r in out] == [r["audio"] for r in r18["rows"]]
    touched = 0
    for before, after in zip(r18["rows"], out):
        diff = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        if diff:
            assert diff == {"text", "control", "control_inoculation"}, diff
            assert after["lang"] in ("id", "tl"), "th 不该被接种"
            assert after["text"] == f"({after['control']}){before['text']}"
            touched += 1
    assert touched, "10% 接种在 60 条 id/tl 上不可能一条都不命中"
    mix = json.loads((dst / "mix.json").read_text(encoding="utf-8"))
    assert mix["total_rows"] == len(r18["rows"])
    assert set(mix["neutral_inoculation"]) <= {"id", "tl"}
    for lang, rate in mix["inoc_rate_per_lang"].items():
        assert 0.0 <= rate <= 0.3, f"{lang} 接种率 {rate} 偏离 10% 目标太远"
    assert sum(mix["prefix_phrases"].values()) == touched


def test_val_byte_identical_and_no_eval_text_leak(module, r18, tmp_path):
    dst = tmp_path / "r26"
    module.main(r18["src"], dst, r18["cases"], tmp_path / "nope")
    assert (dst / "val.jsonl").read_bytes() == (r18["src"] / "val.jsonl").read_bytes()
    mix = json.loads((dst / "mix.json").read_text(encoding="utf-8"))
    # r18 清单里有一条文本与评测 case 相同，接种不改裸文本 ⇒ 暴露数不变、且只报告不删行
    assert mix["eval_text_exposure"]["r18_rows_matching_eval_text"] == 1
    assert mix["eval_text_exposure"]["inoculated_rows_matching_eval_text"] in (0, 1)
    assert mix["total_rows"] == len(r18["rows"])


def test_selection_is_deterministic(module, r18, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert module.main(r18["src"], a, r18["cases"], tmp_path / "nope") == 0
    assert module.main(r18["src"], b, r18["cases"], tmp_path / "nope") == 0
    ha = json.loads((a / "build_audit.json").read_text(encoding="utf-8"))["dst_train_sha256"]
    hb = json.loads((b / "build_audit.json").read_text(encoding="utf-8"))["dst_train_sha256"]
    assert ha == hb, "同 seed 同清单必须逐字节可复现"


def test_refuses_to_overwrite(module, r18, tmp_path):
    dst = tmp_path / "r26"
    dst.mkdir()
    assert module.main(r18["src"], dst, r18["cases"], tmp_path / "nope") == 1
    assert not (dst / "train.jsonl").exists()


def test_records_overlap_with_a_arm(module, r18, tmp_path):
    """A 臂清单在场时要报重叠数，用于解释剂量-反应；不在场时记 null 而不是猜。"""
    a = tmp_path / "r25"
    a_rows = [dict(r) for r in r18["rows"]]
    a_rows[0]["text"] = "(calm and steady)" + a_rows[0]["text"]
    a_rows[0]["control"] = "calm and steady"
    write_jsonl(a / "train.jsonl", a_rows)
    dst = tmp_path / "r26"
    assert module.main(r18["src"], dst, r18["cases"], a) == 0
    mix = json.loads((dst / "mix.json").read_text(encoding="utf-8"))
    assert mix["overlap_with_r25_graft"] in (0, 1)
    dst2 = tmp_path / "r26_noa"
    assert module.main(r18["src"], dst2, r18["cases"], tmp_path / "nope") == 0
    assert json.loads((dst2 / "mix.json").read_text(encoding="utf-8"))["overlap_with_r25_graft"] is None
