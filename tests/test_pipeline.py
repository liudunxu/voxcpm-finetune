"""数据管线核心逻辑自测：裁静音/归一化/加工/混合。"""
import json
import numpy as np
import pytest
import soundfile as sf

from voxft.data.pipeline import (
    Options, apply_control_prefixes, apply_speaker_gain,
    audio_metrics, cluster_pseudo_speakers, mix_manifests, peak_normalize,
    process_dataset, rms_dbfs, trim_silence,
)
from voxft.data.registry import get_source, row_passes
from voxft.paths import DATA_PROCESSED, DATA_RAW


@pytest.fixture(autouse=True)
def isolated_data(monkeypatch, tmp_path):
    from voxft.data import pipeline as pl
    monkeypatch.setattr(pl, "DATA_RAW", tmp_path / "raw")
    monkeypatch.setattr(pl, "DATA_PROCESSED", tmp_path / "processed")
    monkeypatch.setitem(globals(), "DATA_RAW", pl.DATA_RAW)
    monkeypatch.setitem(globals(), "DATA_PROCESSED", pl.DATA_PROCESSED)


def test_trim_and_normalize():
    sr = 16000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)  # 1s 正弦
    wav = np.concatenate([tone, np.zeros(int(2.5 * sr))])  # 尾随 2.5s 静音
    trimmed = trim_silence(wav, sr)
    tail = len(trimmed) / sr - 1.0
    assert 0 <= tail <= 0.35, f"尾静音未裁到 <0.5s: {tail}"
    normed = peak_normalize(trimmed)
    assert abs(float(np.abs(normed).max()) - 0.95) < 1e-6


def _make_source(tmp_path, name, n_spk=2, per_spk=6, dur=4.0):
    src = DATA_RAW / name
    audio = src / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    sr = 22050  # 故意用非目标采样率，验证重采样
    rows = []
    for spk in range(n_spk):
        for i in range(per_spk):
            tone = 0.3 * np.sin(2 * np.pi * (300 + 50 * i)
                                * np.arange(int(dur * sr)) / sr)
            p = audio / f"spk{spk}_{i}.wav"
            sf.write(p, tone, sr)
            rows.append({"audio": str(p), "text": f"文本{spk}-{i}",
                         "speaker": f"spk{spk}", "speaker_verified": True})
    with (src / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return src


def test_process_and_mix():
    _make_source(None, "t_main")
    _make_source(None, "t_mix", n_spk=1, per_spk=8)
    stats = process_dataset("t_main", opts=Options(val_ratio=0.25))
    assert stats["kept"] == 12
    assert stats["val"] >= 1 and stats["train"] + stats["val"] == 12
    train = [json.loads(l) for l in
             (DATA_PROCESSED / "t_main" / "train.jsonl").read_text().splitlines()]
    assert all(3.9 <= r["duration"] <= 4.1 for r in train)
    assert 0 < stats["with_ref_audio"] <= stats["train"]
    # ref_audio 必须是同说话人的另一条
    for r in train:
        if "ref_audio" in r:
            assert r["ref_audio"] != r["audio"]

    process_dataset("t_mix", opts=Options(val_ratio=0.15))
    mix_manifests([("t_main", 0.85), ("t_mix", 0.15)], "t_mixed")
    mixed = (DATA_PROCESSED / "t_mixed" / "train.jsonl").read_text().splitlines()
    assert len(mixed) > 0
    assert (DATA_PROCESSED / "t_mixed" / "mix.json").exists()


# ---------------------------------------------------------------- 去念稿感相关

def _tone(freq, dur, sr=16000, amp=0.3):
    return (amp * np.sin(2 * np.pi * freq
                         * np.arange(int(dur * sr)) / sr)).astype(np.float32)


def test_short_words_are_not_concatenated():
    _make_source(None, "words", dur=0.6)
    with pytest.raises(RuntimeError, match="过滤后无剩余样本"):
        process_dataset("words", opts=Options(metrics=False))


def test_speaker_gain_preserves_relative_dynamics():
    """按说话人整体增益：喊叫与耳语的相对强弱必须保留（逐条归一会抹平它）。"""
    import soundfile as sf
    d = DATA_PROCESSED / "t_gain"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for name, amp in (("loud", 0.6), ("quiet", 0.06)):
        w = _tone(220, 1.0, amp=amp)
        p = d / f"{name}.wav"
        sf.write(p, w, 16000)
        recs.append({"audio": str(p), "speaker": "spk", "speaker_verified": True,
                     "_rms_dbfs": rms_dbfs(w)})
    before = recs[0]["_rms_dbfs"] - recs[1]["_rms_dbfs"]
    apply_speaker_gain(recs, target_dbfs=-24.0)
    after = recs[0]["_rms_dbfs"] - recs[1]["_rms_dbfs"]
    assert abs(before - after) < 0.5, f"动态被压掉了: {before} -> {after}"
    assert abs(np.mean([r["_rms_dbfs"] for r in recs]) + 24.0) < 6.0


def test_control_prefix_from_emotion():
    """情绪标签必须变成 (控制指令) 前缀，否则 LoRA 会冲掉基座的情绪 prompt 能力。"""
    import random
    recs = [{"text": "ako ay masaya", "emotion": "angry", "emotion_verified": True, "rate": 3.0,
             "_rms_dbfs": -20.0} for _ in range(20)]
    n = apply_control_prefixes(recs, Options(control_ratio=1.0), random.Random(1))
    assert n == 20
    assert all(r["text"].startswith("(") and ")" in r["text"] for r in recs)
    assert all(r["control"] for r in recs)
    # 中英文前缀都要出现（线上 prompt 就是中英文写的）
    langs = {any("\u4e00" <= ch <= "\u9fff" for ch in r["control"]) for r in recs}
    assert langs == {True, False}


def test_control_prefix_left_off_for_some_samples():
    """一半样本保持裸文本，保住无前缀推理路径。"""
    import random
    recs = [{"text": f"t{i}", "emotion": "sad", "emotion_verified": True,
             "rate": 3.0, "_rms_dbfs": -20.0}
            for i in range(200)]
    n = apply_control_prefixes(recs, Options(control_ratio=0.5), random.Random(7))
    assert 60 < n < 140, n


def test_pseudo_speaker_clustering_separates_timbres():
    embs = []
    from voxft.data.pipeline import _embed
    for f in (150, 150, 150, 900, 900, 900):
        embs.append(_embed(_tone(f, 1.0), 16000))
    labels = cluster_pseudo_speakers(embs, threshold=0.86)
    assert len(set(labels)) >= 2
    assert labels[0] == labels[1] and labels[3] == labels[4]


def test_metrics_flag_flat_vs_varied():
    """只验证 F0 统计能描述不同波形，不把它当成自然度分数。"""
    sr = 16000
    flat = _tone(200, 2.0)
    t = np.arange(int(2.0 * sr)) / sr
    varied = (0.3 * np.sin(2 * np.pi * (200 + 60 * np.sin(2 * np.pi * 1.5 * t))
                           * t)).astype(np.float32)
    assert audio_metrics(varied, sr, "abc")["f0_std_st"] > \
        audio_metrics(flat, sr, "abc")["f0_std_st"]


def test_mix_caps_repetition():
    """小语料被 tile 十几倍会直接训过拟合，重复必须封顶。"""
    _make_source(None, "t_big", n_spk=2, per_spk=20)
    _make_source(None, "t_small", n_spk=1, per_spk=4)
    process_dataset("t_big", opts=Options(val_ratio=0.1))
    process_dataset("t_small", opts=Options(val_ratio=0.1))
    res = mix_manifests([("t_big", 0.2), ("t_small", 0.8)], "t_capped",
                        max_repeat=2.0)
    small_rows = len((DATA_PROCESSED / "t_small" / "train.jsonl")
                     .read_text().splitlines())
    assert res["t_small/train"] <= small_rows * 2
    assert "t_small/train_capped_from" in res


def test_row_filters_drop_word_lists():
    """filipinospeechcorpus 的 machine / 单词条目必须在下载阶段就被挡掉。"""
    src = get_source("filipino_speech")
    assert not row_passes(src, {"speech_type": "machine", "num_words": 9}.get)
    assert not row_passes(src, {"speech_type": "read", "num_words": 1}.get)
    assert row_passes(src, {"speech_type": "spontaneous", "num_words": 7}.get)
    assert not row_passes(src, {}.get)


def test_thai_ser_column_mapping():
    """THAI-SER 没有名为 audio 的列，必须走 registry 的列映射，否则整个源被跳过。"""
    src = get_source("thai_ser")
    cols = ["audio_id", "mic_clip", "mic_con", "mic_zoom", "script_sent",
            "actor_id", "majority_emo", "agreement"]
    assert src.audio_column(cols) == "mic_con"     # 不能选 mic_zoom
    assert src.audio_column(["mic_clip", "mic_zoom"]) == "mic_clip"
    assert src.audio_column(["mic_zoom"]) is None
    assert not row_passes(src, {"agreement": 0.4}.get)
    assert row_passes(src, {"agreement": 0.9, "turn_type": "impro"}.get)
    assert not row_passes(src, {"agreement": 0.9, "turn_type": "script"}.get)


def test_taglish_languages_accept_english():
    """Tagalog 源必须放行 en：只认 tl 会把句内英文多的 code-switch 样本全部误杀。"""
    assert set(get_source("filipino_emotion").languages()) == {"tl", "en"}
    assert set(get_source("filswitch").languages()) == {"tl", "en"}
    assert get_source("thai_ser").languages() == ("th",)
    assert get_source("aishell3").languages() == ("zh",)


def test_filswitch_is_anchor_not_expressive():
    """FilSwitch 是口播风格，教句内英文词怎么念，不该被当情感主力（控制前缀比例低）。"""
    from voxft.data.pipeline import options_for
    src = get_source("filswitch")
    assert not src.expressive and not src.pseudo_speaker and not src.has_speaker
    assert src.qc == "none"          # 开语种过滤会误杀 code-switch 样本
    o = options_for("filswitch")
    assert o.control_ratio == 0.25 and o.whisper_lang is None


def test_preferred_source_per_lang_and_role():
    """每个 (语种, 角色) 槽位有且只有一个首选，页面与混合建议都依赖它。"""
    from voxft.data.registry import SOURCES, preferred_sources
    pref = preferred_sources()
    assert set(pref) == {("th", "expressive"), ("th", "anchor"),
                         ("tl", "expressive"), ("tl", "anchor"),
                         ("zh", "antiforget")}
    slots = [(s.lang, s.role) for s in SOURCES if s.preferred]
    assert len(slots) == len(set(slots)), "同一槽位出现多个首选"


def test_sources_are_sorted_by_quality():
    from voxft.data.registry import sources_by_quality

    grouped = {}
    for source in sources_by_quality():
        grouped.setdefault(source.lang, []).append(source)
    for sources in grouped.values():
        assert [s.quality for s in sources] == sorted(
            (s.quality for s in sources), reverse=True)
    assert [s.id for s in grouped["th"]] == [
        "drama_th", "thai_ser", "yodas_th", "porjai_th", "fleurs_th",
        "thai20k", "cv22_th"]
    assert [s.id for s in grouped["tl"]] == [
        "drama_tl", "fleurs_tl", "filipino_speech", "filswitch",
        "filipino_emotion", "tagalog_tts"]


def test_yodas_th_session_from_utt_id():
    """完整视频 ID 用于 holdout，不当作真实说话人 ID。"""
    src = get_source("yodas_th")
    assert src.session_of("LxhKGbH7YP0-00233-00065479-00065723") == "LxhKGbH7YP0"
    assert src.session_of("ZLOvTh-VZvA-00078-00022899-00023178") == "ZLOvTh-VZvA"
    assert not src.has_speaker
    assert not row_passes(src, {"grade_avg": "A", "dnsmos_overall": 3.5}.get)
    assert not row_passes(src, {"grade_avg": "S+", "dnsmos_overall": 3.0}.get)
    assert row_passes(src, {"grade_avg": "S+", "dnsmos_overall": 3.4}.get)


def test_display_label_round_trips_to_source_id():
    """下拉框显示文本必须能反解回 id——加【首选】标记时踩过这个坑。"""
    from voxft.data.registry import SOURCES, source_id_from_display
    for src in SOURCES:
        assert source_id_from_display(src.display()) == src.id
        assert source_id_from_display(src.id) == src.id
    import pytest
    with pytest.raises(KeyError):
        source_id_from_display("不存在的源 — 说明 [x]")


def test_transcribe_checkpoints_so_a_crash_does_not_lose_work(monkeypatch, tmp_path):
    """万级转写中途挂掉不能白跑：已转好的必须已经落回清单，重跑跳过。"""
    from voxft.data import pipeline as pl

    class _Seg:
        text = "kumusta"

    class _Info:
        language = "tl"

    class _FakeModel:
        def __init__(self):
            self.n = 0

        def transcribe(self, wav, vad_filter=True):
            assert len(wav) == 1600  # 非 16k 原录音必须先重采样；签名不接收 batched
            self.n += 1
            if self.n == 5:
                raise RuntimeError("boom")   # 模拟中途崩
            return [_Seg()], _Info()

    monkeypatch.setattr(pl, "_whisper_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(pl, "load_wav_mono", lambda p: (np.zeros(2205, np.float32), 22050))
    rows = [{"audio": f"{i}.wav", "text": ""} for i in range(8)]
    saved: list[list[dict]] = []
    with pytest.raises(RuntimeError, match="转写失败"):
        pl._transcribe_manifest(rows, "tl", ("tl", "en"),
                                checkpoint=saved.append, checkpoint_every=2)
    assert saved, "从未落盘"
    mid = saved[0]
    assert len(mid) == len(rows), "落盘丢了未处理的行，重跑会漏数据"
    assert sum(1 for r in mid if r["text"]) == 2   # 前两条已转好
    # 用落盘结果重跑：已有文本的直接放行，不会重复转写
    assert sum(bool(r["text"]) for r in saved[-1]) == 4
    again, bad = pl._transcribe_manifest(saved[-1], "tl", ("tl", "en"))
    assert len(again) == 8 and bad == 0
    assert all(r["text"] == "kumusta" for r in again)


def test_verified_controls_and_reference_quadrants():
    import random
    from voxft.data.pipeline import build_control, pair_references, split_records
    opts = Options(val_ratio=0.2)
    rows = []
    for speaker in range(5):
        for i in range(21):
            rows.append({"audio": f"{speaker}_{i}.wav", "origin_audio": f"raw/{speaker}_{i}.wav",
                         "speaker": f"actor{speaker}", "speaker_verified": True,
                         "duration": 4.0, "lang": "en" if i == 20 else "tl",
                         "reference_only": i == 20, "text": "test dialogue",
                         "emotion": "angry", "emotion_verified": True})
    train, val = split_records(rows, opts)
    assert {r["speaker"] for r in train}.isdisjoint(r["speaker"] for r in val)
    for group in (train, val):
        rng = random.Random(42)
        apply_control_prefixes(group, opts, rng)
        result = pair_references(group, opts, rng)
        assert sum(bool(r.get("control")) for r in result) == len(result) * 0.5
        assert sum(bool(r.get("ref_audio")) for r in result) == len(result) * 0.5
        assert sum(bool(r.get("control") and r.get("ref_audio")) for r in result) == len(result) * 0.3
        for r in result:
            if r.get("ref_audio"):
                assert r["ref_speaker"] == r["speaker"] and r["ref_lang"] == "en"
                assert r["ref_audio"] in {x["audio"] for x in group}
    unknown = [{**r, "speaker_verified": False} for r in rows]
    for r in unknown:
        for key in ("ref_audio", "ref_lang", "ref_speaker"):
            r.pop(key, None)
    assert all(not r.get("ref_audio") for r in pair_references(unknown, opts, random.Random(42)))
    assert build_control({"emotion": "angry", "rate": 20, "_rms_dbfs": -10}, random.Random(1), 1) == ""
    assert build_control({"emotion": "angry", "emotion_verified": True}, random.Random(1), 1) == "愤怒地"


def test_sessions_are_indivisible_and_split_is_control_independent():
    from voxft.data.pipeline import split_records
    rows = [{"audio": str(i), "origin_audio": str(i), "session": f"video{i // 3}",
             "source_id": "test", "speaker": str(i), "speaker_verified": False} for i in range(12)]
    train, val = split_records(rows, Options(val_ratio=0.1, val_max=1))
    assert len(val) == 3  # val_max 是软目标，不截断会话
    assert {r["session"] for r in train}.isdisjoint(r["session"] for r in val)
    assert split_records(rows, Options(val_ratio=0.1, val_max=1, control_ratio=0)) == (train, val)


def test_atomic_write_and_transcribe_qc_signature(tmp_path):
    from types import SimpleNamespace
    from voxft.data.pipeline import _write_jsonl, _whisper_similarity
    p = tmp_path / "raw.jsonl"
    _write_jsonl([{"text": "original"}], p)
    before = p.read_bytes()
    with pytest.raises(ValueError):
        _write_jsonl([{"score": float("nan")}], p)
    assert p.read_bytes() == before

    class Model:
        def transcribe(self, wav, vad_filter=True):
            assert len(wav) == 1600
            return [SimpleNamespace(text="hello")], SimpleNamespace(language="en")
    assert _whisper_similarity(Model(), np.zeros(2205), 22050, "hello") == (1.0, "en")


def test_gain_peak_backoff_is_shared_by_speaker(tmp_path):
    recs = []
    for i, amp in enumerate((0.9, 0.002, 0.004)):
        wav = _tone(200, 1, amp=amp)
        p = tmp_path / f"{i}.wav"
        sf.write(p, wav, 16000, subtype="FLOAT")
        recs.append({"audio": str(p), "speaker": "verified", "speaker_verified": True,
                     "_rms_dbfs": rms_dbfs(wav)})
    before = [r["_rms_dbfs"] for r in recs]
    apply_speaker_gain(recs, -24)
    gains = [r["_rms_dbfs"] - level for r, level in zip(recs, before)]
    assert max(gains) - min(gains) < 0.01
    assert max(np.abs(sf.read(r["audio"])[0]).max() for r in recs) <= 0.971


def test_mix_duration_exposure_and_leak_detection():
    from voxft.data.pipeline import _write_jsonl
    for name, duration in (("short", 3), ("long", 9)):
        rows = [{"audio": f"/{name}/{i}.wav", "duration": duration, "text": "test",
                 "lang": "tl"} for i in range(10)]
        _write_jsonl(rows, DATA_PROCESSED / name / "train.jsonl")
        _write_jsonl([{**rows[0], "audio": f"/{name}/val.wav"}], DATA_PROCESSED / name / "val.jsonl")
    mix_manifests([("short", 0.5), ("long", 0.5)], "duration_mix")
    info = json.loads((DATA_PROCESSED / "duration_mix" / "mix.json").read_text())
    assert info["datasets"]["short/train"]["seconds"] == 60
    assert info["datasets"]["long/train"]["seconds"] == 63
    assert info["val"]["rows"] == 2 and info["val"]["max_exposure"] == 1
    mix_manifests([("duration_mix", 1)], "nested")
    nested = json.loads((DATA_PROCESSED / "nested" / "mix.json").read_text())
    assert nested["train"]["max_exposure"] <= 3
    _write_jsonl([{"audio": "/short/0.wav", "duration": 3}], DATA_PROCESSED / "short" / "val.jsonl")
    with pytest.raises(ValueError, match="共享音频"):
        mix_manifests([("short", 1)], "leaked")


def test_curated_import_cross_language_refs_and_safe_reprocessing():
    from voxft.data.pipeline import _write_jsonl
    raw = _make_source(None, "drama_tl", n_spk=2, per_spk=21)
    rows = [json.loads(line) for line in (raw / "manifest.jsonl").read_text().splitlines()]
    for i, row in enumerate(rows):
        row.update(lang="en" if i % 21 == 20 else "tl", reference_only=i % 21 == 20,
                   emotion="angry", emotion_verified=True)
        if row["reference_only"]:
            row["text"] = ""  # ref-only 不需 ASR 或训练正文
    _write_jsonl(rows, raw / "manifest.jsonl")
    opts = Options(metrics=False, val_ratio=0.2)
    result = process_dataset("drama_tl", opts=opts, manifest_path=raw / "manifest.jsonl")
    assert result["cross_language_refs"] == 10 and result["with_ref_control"] == 6
    train_path = DATA_PROCESSED / "drama_tl" / "train.jsonl"
    first = [json.loads(line) for line in train_path.read_text().splitlines()]
    before = {r["audio"]: sf.read(r["audio"])[0] for r in first}
    process_dataset("drama_tl", opts=opts, manifest_path=raw / "manifest.jsonl")
    second = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert {r["audio"] for r in first}.isdisjoint(r["audio"] for r in second)
    for path, wav in before.items():
        assert np.array_equal(sf.read(path)[0], wav)


def test_process_keeps_rejected_and_unprocessed_raw_rows(monkeypatch):
    from types import SimpleNamespace
    from voxft.data import pipeline as pl
    raw = _make_source(None, "drama_tl", n_spk=1, per_spk=4)
    manifest = raw / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["text"] = ""
    pl._write_jsonl(rows, manifest)
    class Model:
        calls = 0
        def transcribe(self, wav, vad_filter=True):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("interrupted")
            return [SimpleNamespace(text="kumusta")], SimpleNamespace(
                language="ja" if self.calls == 2 else "tl")
    model = Model()
    monkeypatch.setattr(pl, "_whisper_model", lambda *args: model)
    with pytest.raises(RuntimeError, match="转写失败"):
        process_dataset("drama_tl", opts=Options(metrics=False), manifest_path=manifest)
    saved = pl._read_manifest(manifest)
    assert len(saved) == 4 and saved[0]["text"] == "kumusta"
    assert saved[1]["text"] == "" and "transcribe_error" in saved[1]
    assert saved[2]["text"] == saved[3]["text"] == ""
    assert [r["audio"] for r in saved] == [r["audio"] for r in rows]
