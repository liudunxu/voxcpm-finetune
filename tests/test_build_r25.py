"""r25 构建脚本的守卫自测：移植必须是纯文本级单变量。"""
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_r25_r18_prefix_graft")


def row(name, lang="id", text=None, dur=4.0, source="gigaspeech2_id", **extra):
    return {"audio": f"/audio/{name}.wav", "origin_audio": f"/raw/{name}.wav",
            "text": text if text is not None else f"kalimat {name}", "duration": dur,
            "lang": lang, "source_id": source, "rate": 2.1, "speaker": f"spk_{name}",
            "session": f"ses_{name}", **extra}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def layouts(tmp_path):
    r18 = tmp_path / "r18"
    r23 = tmp_path / "r23"
    base = [row("a"), row("b", lang="tl", source="fleurs_tl"), row("c", lang="th", source="gigaspeech2_th")]
    write_jsonl(r18 / "train.jsonl", base)
    (r18 / "val.jsonl").write_bytes(b'{"audio": "/audio/v.wav", "text": "val"}\n')
    # r23：两条前缀行命中 r18 的 audio；一条前缀行的 audio 不在 r18 里（应被忽略）
    write_jsonl(r23 / "train.jsonl", [
        row("a", text="(calm and steady)kalimat a", control="calm and steady", control_inoculation=True),
        row("b", lang="tl", source="fleurs_tl", text="(relaxed and natural)kalimat b",
            control="relaxed and natural", control_inoculation=True),
        row("zzz", text="(neutral, natural tone)kalimat zzz", control="neutral, natural tone",
            control_inoculation=True),
        row("c", lang="th", source="gigaspeech2_th"),
    ])
    cases = tmp_path / "cases.jsonl"
    write_jsonl(cases, [{"case_id": "x", "lang": "id", "text": "kalimat a"}])
    return {"r18": r18, "r23": r23, "dst": tmp_path / "r25", "eval": [str(cases)], "base": base}


def test_graft_is_text_only_and_keeps_every_row(module, layouts):
    assert module.main(layouts["r18"], layouts["r23"], layouts["dst"], layouts["eval"]) == 0
    out = [json.loads(x) for x in (layouts["dst"] / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(out) == len(layouts["base"])
    assert [r["audio"] for r in out] == [r["audio"] for r in layouts["base"]]
    for before, after in zip(layouts["base"], out):
        diff = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        if after["audio"].endswith("a.wav") or after["audio"].endswith("b.wav"):
            assert diff == {"text", "control", "control_inoculation"}, diff
        else:
            assert diff == set(), f"未移植的行被改动了：{diff}"
    assert out[0]["text"].startswith("(calm and steady)")
    assert out[2]["text"] == "kalimat c"          # th 行不参与移植


def test_val_is_byte_identical_and_mix_records_the_graft(module, layouts):
    module.main(layouts["r18"], layouts["r23"], layouts["dst"], layouts["eval"])
    assert (layouts["dst"] / "val.jsonl").read_bytes() == (layouts["r18"] / "val.jsonl").read_bytes()
    mix = json.loads((layouts["dst"] / "mix.json").read_text(encoding="utf-8"))
    assert mix["total_rows"] == 3
    assert mix["neutral_inoculation"] == {"id": 1, "tl": 1}
    assert mix["prefix_phrases"] == {"calm and steady": 1, "relaxed and natural": 1}
    assert mix["eval_text_exposure"]["grafted_rows_matching_eval_text"] == 1
    assert sum(s["rows"] for s in mix["language_shares"].values()) == 3


def test_refuses_to_overwrite_existing_destination(module, layouts):
    layouts["dst"].mkdir(parents=True)
    assert module.main(layouts["r18"], layouts["r23"], layouts["dst"], layouts["eval"]) == 1
    assert not (layouts["dst"] / "train.jsonl").exists()


def test_refuses_when_bare_text_differs(module, layouts):
    """前缀接到另一句上就等于悄悄换了训练文本，必须硬失败。"""
    rows = [json.loads(x) for x in (layouts["r23"] / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = "(calm and steady)kalimat yang berbeda"
    write_jsonl(layouts["r23"] / "train.jsonl", rows)
    assert module.main(layouts["r18"], layouts["r23"], layouts["dst"], layouts["eval"]) == 1
    assert not layouts["dst"].exists()


def test_stops_when_nothing_to_graft(module, layouts, tmp_path):
    empty = tmp_path / "r23_none"
    write_jsonl(empty / "train.jsonl", [row("q"), row("r", text="(calm and steady)kalimat r")])
    assert module.main(layouts["r18"], empty, layouts["dst"], layouts["eval"]) == 1
    assert not layouts["dst"].exists()
