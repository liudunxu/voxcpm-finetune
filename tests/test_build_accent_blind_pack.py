"""id 口音盲听包构建脚本自测：盲化、完整性与泄漏守卫。"""
import importlib
import json
import wave
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_accent_blind_pack")


def make_wav(path: Path, seconds: float, sr: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))


def fake_report(path: Path, label: str, cases=("c1", "c2"), seed=42, dur=0.10):
    # 台词不含 case_id：真实探针文本是印尼语句子，而台词本来就要显示给听者
    items = [{"case_id": c, "seed": seed, "lang": "id", "ref_lang": "zh", "text": "Tolong aku.",
              "control": "", "ref_audio": "/refs/ref_zh_01.wav", "cer": 0.01, "len_ratio": 1.0,
              "over_read": False, "audio_sec": dur, "wav": f"/remote/{label}_{c}.wav"}
             for c in cases]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": label, "items": items}, ensure_ascii=False), encoding="utf-8")


def plan_for(module, tmp_path, models=("m1", "m2"), dur=0.10):
    reports = {}
    for m in models:
        p = tmp_path / f"{m}.json"
        fake_report(p, m, dur=dur)
        reports[m] = p
    entries = module.select(reports, "zh", 42)
    rows = [{"key": f"{e['case_id']}__{m}", "remote": info["wav"], "model": m,
             "case_id": e["case_id"], "seed": 42, "audio_sec": info["audio_sec"],
             "text": e["text"], "ref_lang": "zh", "ref_audio": e["ref_audio"]}
            for e in entries for m, info in e["models"].items()]
    plan = {"ref_lang": "zh", "seed": 42, "models": list(models), "cases": entries, "fetch": rows}
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return pp, rows, dur


def stage_wavs(raw: Path, rows, dur):
    for r in rows:
        make_wav(raw / f"{r['key']}.wav", dur)


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_resolve_models_rejects_unknown_and_too_many(module):
    got = module.resolve_models("base,r26")
    assert list(got) == ["base", "r26"]
    with pytest.raises(SystemExit, match="未知模型"):
        module.resolve_models("base,r99")
    with pytest.raises(SystemExit, match="最多"):
        module.resolve_models(",".join(sorted(module.DEFAULT_MODELS) + ["base"]))  # 8 个 > 7 个代码


def test_select_keeps_only_cases_present_in_every_model(module, tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    fake_report(a, "a", cases=("c1", "c2"))
    fake_report(b, "b", cases=("c1",))          # 少一条 ⇒ 交集只剩 c1
    entries = module.select({"a": a, "b": b}, "zh", 42)
    assert [e["case_id"] for e in entries] == ["c1"]
    assert set(entries[0]["models"]) == {"a", "b"}


def test_build_blinds_codes_and_verifies_durations(module, tmp_path):
    pp, rows, dur = plan_for(module, tmp_path, models=("m1", "m2", "m3"))
    raw = tmp_path / "raw"
    stage_wavs(raw, rows, dur)
    dst = tmp_path / "pack"
    assert module.cmd_build(Args(plan=pp, raw=raw, dst=dst, shuffle_seed=7)) == 0

    pub, priv = dst / "public", dst / "private"
    html = (pub / "index.html").read_text(encoding="utf-8")
    mapping = json.loads((priv / "mapping.json").read_text(encoding="utf-8"))
    assert len(mapping["entries"]) == 6                      # 2 组 × 3 模型
    assert sorted(p.name for p in (pub / "audio").glob("*.wav")) == [
        "g1_a.wav", "g1_b.wav", "g1_c.wav", "g2_a.wav", "g2_b.wav", "g2_c.wav"]
    # 公开侧不得出现模型名或 case_id
    for banned in ("m1", "m2", "m3", "lora_omni5"):
        assert banned not in html, f"public/ 泄漏了模型身份 {banned}"
    # 私有侧必须能还原
    assert {e["code"] for e in mapping["entries"] if e["group"] == 1} == {"A", "B", "C"}
    assert all(e["reported_audio_sec"] == pytest.approx(e["actual_duration_sec"], abs=0.05)
               for e in mapping["entries"])


def test_build_refuses_wrong_duration_missing_file_and_existing_dir(module, tmp_path):
    pp, rows, dur = plan_for(module, tmp_path, models=("m1", "m2"))
    raw = tmp_path / "raw"
    stage_wavs(raw, rows, dur)
    # 时长不符：把一条换成 0.30s（报告写的是 0.10s）
    make_wav(raw / f"{rows[0]['key']}.wav", 0.30)
    assert module.cmd_build(Args(plan=pp, raw=raw, dst=tmp_path / "p1", shuffle_seed=1)) == 1
    assert not (tmp_path / "p1").exists()

    stage_wavs(raw, rows, dur)                                  # 恢复正确时长
    (raw / f"{rows[1]['key']}.wav").unlink()                    # 缺一条
    assert module.cmd_build(Args(plan=pp, raw=raw, dst=tmp_path / "p2", shuffle_seed=1)) == 1

    stage_wavs(raw, rows, dur)
    assert module.cmd_build(Args(plan=pp, raw=raw, dst=tmp_path / "p3", shuffle_seed=1)) == 0
    assert module.cmd_build(Args(plan=pp, raw=raw, dst=tmp_path / "p3", shuffle_seed=1)) == 1  # 拒绝覆盖


def test_shuffle_is_reproducible_and_changes_with_seed(module, tmp_path):
    pp, rows, _ = plan_for(module, tmp_path, models=("m1", "m2", "m3", "m4"))
    raw = tmp_path / "raw"
    stage_wavs(raw, rows, 0.10)
    outs = []
    for i, seed in enumerate((11, 11, 22)):
        dst = tmp_path / f"pack{i}"
        assert module.cmd_build(Args(plan=pp, raw=raw, dst=dst, shuffle_seed=seed)) == 0
        m = json.loads((dst / "private" / "mapping.json").read_text(encoding="utf-8"))
        outs.append([(e["group"], e["code"], e["model"]) for e in m["entries"]])
    assert outs[0] == outs[1], "同 shuffle_seed 必须可复现"
    assert outs[0] != outs[2], "不同 shuffle_seed 应当给出不同顺序（否则等于没盲化）"


def test_html_renders_only_the_given_codes(module):
    groups = [{"group": 1, "case_id": "cx", "text": "Tolong aku.", "seed": 42, "ref_lang": "zh",
               "ref_audio": "/r.wav", "codes": {"A": "m1", "B": "m2"}}]
    html = module.render_html(groups, "AB")
    assert "1A" in html and "1B" in html
    assert "1C" not in html
    assert "codes_per_group=2" in html
    assert "Tolong aku." in html
