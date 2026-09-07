"""成片导入链路自测：切分分组、标注校验、追加去重/替换、holdout 端到端。

不依赖 PyAV 与 faster-whisper（本地默认没装 qc 组），解码与转写全部 monkeypatch。
"""
import json
import sys

import numpy as np
import pytest
import soundfile as sf

from voxft.data import ingest
from voxft.data import pipeline as pl
from voxft.data.ingest import (ALL, _split_long, append_to_source, clip_label,
                               group_regions, ingest_file, list_batches,
                               load_candidates, pin_holdout, save_candidates,
                               validate_annotation)
from voxft.data.pipeline import TARGET_SR, Options, process_dataset

SR = TARGET_SR


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "DATA_RAW", tmp_path / "raw")
    monkeypatch.setattr(pl, "DATA_PROCESSED", tmp_path / "processed")


class _Seg:
    def __init__(self, start, end, logprob=-0.2):
        self.start, self.end = start, end
        self.text = "kumusta ka na ba"
        self.avg_logprob, self.no_speech_prob = logprob, 0.05


class _FakeModel:
    """边界趟与逐条转写趟共用同一个假模型；calls 用来验证断点续跑。"""

    def __init__(self, regions, logprob=-0.2):
        self.regions, self.logprob, self.calls = regions, logprob, 0

    def transcribe(self, wav, vad_filter=True):
        self.calls += 1
        segs = [_Seg(s, e, self.logprob) for s, e in self.regions]
        return iter(segs), type("Info", (), {"language": "tl"})()


def _stub_decode(monkeypatch, seconds=12.0):
    def decode(src, dst, progress=None):
        dst.parent.mkdir(parents=True, exist_ok=True)
        tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * SR)) / SR)
        sf.write(dst, tone.astype(np.float32), SR, subtype="PCM_16")
        return dst
    monkeypatch.setattr(ingest, "decode_to_wav", decode)


def _stub_whisper(monkeypatch, regions, logprob=-0.2):
    model = _FakeModel(regions, logprob)
    monkeypatch.setattr(pl, "_whisper_model", lambda *a, **k: model)
    return model


def _fake_video(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"not really a video")   # 解码被 stub 掉，内容无所谓
    return p


def test_group_regions_merges_splits_and_drops_short():
    wav = np.zeros(int(60 * SR), np.float32)
    wav[int(10 * SR):int(50 * SR)] = 0.3
    regions = [(0.0, 2.0), (2.3, 4.0), (6.0, 6.5), (10.0, 50.0)]
    clips, too_short = group_regions(regions, wav, 3.0, 30.0)
    assert (round(clips[0][0], 2), round(clips[0][1], 2)) == (0.0, 4.15), \
        "隔 0.3s 的两段应并成一条完整语流"
    assert too_short == 1, "0.5s 的碎句应被丢弃并计数"
    assert all(e - s <= 30.0 for s, e in clips), "40s 的超长区间没切开"

    far = [(0.0, 4.0), (9.0, 13.0)]
    assert len(group_regions(far, wav, 3.0, 30.0)[0]) == 2, "隔 5s 不该合并"


def test_split_long_cuts_at_the_quietest_frame():
    """超长台词要在停顿处切开，不能切在词中间。"""
    wav = np.full(int(40 * SR), 0.3, np.float32)
    wav[int(14 * SR):int(15 * SR)] = 0.0          # 14-15s 是窗口里唯一的停顿
    pieces = _split_long(wav, 0.0, 40.0, 30.0)
    assert len(pieces) == 2 and all(e - s <= 30.0 for s, e in pieces)
    assert 13.0 <= pieces[1][0] <= 16.0, f"没切在停顿处: {pieces}"


def test_safe_id_blocks_path_traversal():
    """素材 ID 会变成 DATA_RAW 下的目录名，必须挡掉路径穿越。"""
    assert ingest._safe_id("ep01 / 第一集") == "ep01_第一集"
    for bad in ("", "  ", ".", ".."):
        with pytest.raises(ValueError, match="素材 ID 非法"):
            ingest._safe_id(bad)
    assert "/" not in ingest._safe_id("../../etc/passwd")


def test_decode_without_pyav_gives_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(RuntimeError, match="uv sync --group qc"):
        ingest.decode_to_wav(tmp_path / "ep.mp4", tmp_path / "out.wav")


def test_ingest_file_writes_candidates_and_resumes(monkeypatch, tmp_path):
    _stub_decode(monkeypatch)
    model = _stub_whisper(monkeypatch, [(0.0, 9.0)])

    summary = ingest_file(_fake_video(tmp_path, "ep01.mp4"), "drama_tl")
    assert summary["clips"] == 1 and summary["transcribed"] == 1, summary
    rows = load_candidates("drama_tl", "ep01")
    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "kumusta ka na ba"
    assert row["session"] == row["ingest_video"] == "ep01", \
        "session 必须是素材 ID，否则同一集的切片会各自成组、跨 train/val 泄漏"
    assert row["verdict"] == "待定"
    assert float(row["end"]) - float(row["start"]) >= 3.0

    calls = model.calls
    summary2 = ingest_file(_fake_video(tmp_path, "ep01.mp4"), "drama_tl")
    assert model.calls == calls + 1, "断点续跑失效：重跑又逐条转写了一遍"
    assert summary2["transcribed"] == 1
    assert list_batches("drama_tl") == ["ep01"]


def test_ingest_passes_the_intelligibility_gate_down(monkeypatch, tmp_path):
    """drama_tl 的加工阶段不跑 ASR 质检（qc=none），含糊音必须在切分阶段就挡住。"""
    _stub_decode(monkeypatch)
    _stub_whisper(monkeypatch, [(0.0, 9.0)], logprob=-2.5)
    monkeypatch.setattr(pl, "options_for", lambda sid, **kw: Options(asr_min_logprob=-1.0))

    summary = ingest_file(_fake_video(tmp_path, "ep03.mp4"), "drama_tl")
    assert summary["transcribed"] == 0 and summary["dropped"] == 1, summary
    row = load_candidates("drama_tl", "ep03")[0]
    assert row["transcribe_error"].startswith("unintelligible"), row
    assert not row.get("text"), "听不清的不能拿到训练文本"
    assert append_to_source("drama_tl", [dict(row, verdict="保留")])["appended"] == 0


def test_validate_annotation_enforces_the_same_rules_as_processing():
    ok = validate_annotation("Hindi mo alam.", "actor01", True, "surprised", True,
                             "惊讶地，语气克制", "surprised, restrained", True, "保留")
    assert ok["speaker_verified"] is True and ok["speaker"] == "actor01"
    assert ok["control_zh"] == "惊讶地，语气克制" and ok["emotion_verified"] is True

    with pytest.raises(ValueError, match="必须有台词文本"):
        validate_annotation("", "", False, "", False, "", "", False, "保留")
    with pytest.raises(ValueError, match="裸台词"):
        validate_annotation("(愤怒地)Ano ka!", "", False, "", False, "", "", False, "保留")
    with pytest.raises(ValueError, match="真实 speaker ID"):
        validate_annotation("Ano ka!", "", True, "", False, "", "", False, "保留")
    with pytest.raises(ValueError, match="只能使用中英文"):
        validate_annotation("Ano ka!", "a1", True, "", False, "โกรธ", "", True, "保留")
    with pytest.raises(ValueError, match="verdict 必须是"):
        validate_annotation("Ano ka!", "", False, "", False, "", "", False, "maybe")

    # 没核实身份就不能偷偷写 speaker_verified=true：那会污染 ref 配对与响度对齐
    anon = validate_annotation("Ano ka!", "", False, "", False, "", "", False, "待定")
    assert anon["speaker_verified"] is False and anon["speaker"] == ""


def test_append_filters_replaces_and_pins(tmp_path):
    def row(name, **kw):
        return {"audio": str(tmp_path / name), "text": "t", "session": "ep01",
                "ingest_video": "ep01", "verdict": "保留", "start": 0.0, "end": 4.0, **kw}

    rows = [row("a.wav", speaker="actor01", speaker_verified=True),
            row("b.wav", text=""),
            row("c.wav", verdict="丢弃"),
            row("d.wav", transcribe_error="unintelligible:logprob=-2.50")]
    res = append_to_source("drama_tl", rows)
    assert res["appended"] == 1 and res["total"] == 1, res
    manifest = pl.DATA_RAW / "drama_tl" / "manifest.jsonl"
    saved = pl._read_manifest(manifest)
    assert "verdict" not in saved[0], "工作流状态不该进训练清单"
    assert saved[0]["start"] == 0.0, "起止时间是溯源信息，要留着"

    before = pl._read_manifest(manifest)
    assert append_to_source("drama_tl", [dict(rows[2])])["appended"] == 0
    assert pl._read_manifest(manifest) == before, "没有可追加的行时不该改写清单"

    rows[0]["audio"] = str(tmp_path / "a2.wav")   # 同一素材重切 → 替换而不是叠加
    res2 = append_to_source("drama_tl", rows, holdout=("ep01",))
    assert res2["total"] == 1 and res2["appended"] == 1, res2
    assert [r["audio"] for r in pl._read_manifest(manifest)] == [str(tmp_path / "a2.wav")]
    assert pl._load_holdout(manifest) == ("ep01",), "holdout 没钉住"


def test_clip_labels_are_unique_across_batches():
    rows = [{"ingest_video": "ep01", "start": 0.0, "end": 4.2, "verdict": "保留",
             "speaker": "actor01"},
            {"ingest_video": "ep02", "start": 0.0, "end": 3.0,
             "transcribe_error": "unintelligible:logprob=-2.50"}]
    labels = [clip_label(i, r) for i, r in enumerate(rows)]
    assert len(set(labels)) == len(labels), "下拉框靠 label 反查序号，重复就会点错条"
    assert "actor01" in labels[0] and "听不清" in labels[1]


def test_end_to_end_append_then_process_keeps_holdout_out_of_train(monkeypatch, tmp_path):
    """整链路：切两集 → 标注保留 → 追加 → 钉住第一集 → 加工 → 钉住的只能在验证集。"""
    _stub_decode(monkeypatch)
    for vid, text in (("ep01", "una"), ("ep02", "pangalawa")):
        _stub_whisper(monkeypatch, [(0.0, 4.0), (4.5, 9.0)])
        ingest_file(_fake_video(tmp_path, f"{vid}.mp4"), "drama_tl", vid)
        rows = load_candidates("drama_tl", vid)
        for r in rows:
            r.update(validate_annotation(text, "", False, "", False, "", "", False, "保留"))
        save_candidates("drama_tl", rows)

    all_rows = load_candidates("drama_tl", ALL)
    assert len(all_rows) == 2, "两集应各切出一条候选"
    assert append_to_source("drama_tl", all_rows)["appended"] == 2
    assert pin_holdout("drama_tl", ["ep01"]) == ("ep01",)

    stats = process_dataset("drama_tl", opts=pl.options_for("drama_tl", val_ratio=0.5))
    assert stats["val"] == 1 and stats["train"] == 1, stats
    assert stats["holdout_pinned_records"] == 1
    out = pl.DATA_PROCESSED / "drama_tl"
    assert {r["session"] for r in pl._read_manifest(out / "val.jsonl")} == {"ep01"}, \
        "钉住的素材没留在验证集"
    assert {r["session"] for r in pl._read_manifest(out / "train.jsonl")} == {"ep02"}, \
        "钉住的素材进了训练集"
    assert json.loads((pl.DATA_RAW / "drama_tl" / "holdout.json").read_text()) == \
        {"sessions": ["ep01"]}
