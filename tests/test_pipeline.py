"""数据管线核心逻辑自测：裁静音/归一化/加工/混合。"""
import json
import numpy as np
import soundfile as sf

from voxft.data.pipeline import (
    Clip, Options, _concatenated, apply_control_prefixes, apply_speaker_gain,
    audio_metrics, cluster_pseudo_speakers, mix_manifests, peak_normalize,
    process_dataset, rms_dbfs, trim_silence,
)
from voxft.data.registry import get_source, row_passes
from voxft.paths import DATA_PROCESSED, DATA_RAW


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
                         "speaker": f"spk{spk}"})
    with (src / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return src


def test_process_and_mix():
    _make_source(None, "t_main")
    _make_source(None, "t_mix", n_spk=1, per_spk=8)
    stats = process_dataset("t_main", opts=Options(min_dur=1.0, val_ratio=0.25))
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

    process_dataset("t_mix", opts=Options(min_dur=1.0, val_ratio=0.15))
    out = mix_manifests([("t_main", 0.85), ("t_mix", 0.15)], "t_mixed")
    mixed = (DATA_PROCESSED / "t_mixed" / "train.jsonl").read_text().splitlines()
    assert len(mixed) > 0
    assert (DATA_PROCESSED / "t_mixed" / "mix.json").exists()


# ---------------------------------------------------------------- 去念稿感相关

def _tone(freq, dur, sr=16000, amp=0.3):
    return (amp * np.sin(2 * np.pi * freq
                         * np.arange(int(dur * sr)) / sr)).astype(np.float32)


def test_concat_stays_in_session_and_adds_punctuation():
    """拼接必须限定在同一次录音内，且句间补标点——否则训出报菜名式念稿声。"""
    import random
    clips = [Clip(_tone(300, 1.0), "alpha", "spk0", session="s1"),
             Clip(_tone(310, 1.0), "beta", "spk0", session="s1"),
             Clip(_tone(320, 1.0), "gamma", "spk0", session="s2")]
    out = list(_concatenated(iter(clips), target=1.8, max_dur=30.0,
                             sep=". ", rng=random.Random(0)))
    texts = [c.text for c in out]
    assert any("alpha. beta" in t for t in texts), texts
    assert all("gamma" not in t or t.startswith("gamma") for t in texts), texts
    # 段间停顿必须抖动，不能是固定 0.3s
    joined = next(c for c in out if "alpha" in c.text)
    assert len(joined.wav) / 16000 > 2.0


def test_thai_concat_keeps_space_not_period():
    """泰语不用句点，句间是空格；给泰语补 '.' 会让训练文本偏离真实分布。"""
    import random
    clips = [Clip(_tone(300, 1.0), "ก", "s", session="x"),
             Clip(_tone(300, 1.0), "ข", "s", session="x")]
    out = list(_concatenated(iter(clips), 1.8, 30.0, " ", random.Random(0)))
    assert "." not in out[0].text and "ก ข" in out[0].text


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
        recs.append({"audio": str(p), "speaker": "spk", "_rms_dbfs": rms_dbfs(w)})
    before = recs[0]["_rms_dbfs"] - recs[1]["_rms_dbfs"]
    apply_speaker_gain(recs, target_dbfs=-24.0)
    after = recs[0]["_rms_dbfs"] - recs[1]["_rms_dbfs"]
    assert abs(before - after) < 0.5, f"动态被压掉了: {before} -> {after}"
    assert abs(np.mean([r["_rms_dbfs"] for r in recs]) + 24.0) < 6.0


def test_control_prefix_from_emotion():
    """情绪标签必须变成 (控制指令) 前缀，否则 LoRA 会冲掉基座的情绪 prompt 能力。"""
    import random
    recs = [{"text": "ako ay masaya", "emotion": "angry", "rate": 3.0,
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
    recs = [{"text": f"t{i}", "emotion": "sad", "rate": 3.0, "_rms_dbfs": -20.0}
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
    """f0 起伏指标要能区分平读与有起伏——它是"robotic"的量化抓手。"""
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
    process_dataset("t_big", opts=Options(min_dur=1.0, val_ratio=0.1))
    process_dataset("t_small", opts=Options(min_dur=1.0, val_ratio=0.1))
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
    assert row_passes(src, {}.get)  # 缺列时不拦


def test_thai_ser_column_mapping():
    """THAI-SER 没有名为 audio 的列，必须走 registry 的列映射，否则整个源被跳过。"""
    src = get_source("thai_ser")
    cols = ["audio_id", "mic_clip", "mic_con", "mic_zoom", "script_sent",
            "actor_id", "majority_emo", "agreement"]
    assert src.audio_column(cols) == "mic_con"     # 不能选 mic_zoom
    assert src.audio_column(["mic_clip", "mic_zoom"]) == "mic_clip"
    assert src.audio_column(["mic_zoom"]) is None
    assert src.separator() == " "                  # 泰语不加句点
    assert not row_passes(src, {"agreement": 0.4}.get)
    assert row_passes(src, {"agreement": 0.9}.get)


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
    assert not src.expressive and src.pseudo_speaker
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


def test_yodas_th_session_from_utt_id():
    """YODAS 的会话是 utt_id 里的 YouTube video id；拼接必须只在同一视频内做。"""
    src = get_source("yodas_th")
    assert src.session_of("LxhKGbH7YP0-00233-00065479-00065723") == "LxhKGbH7YP0"
    assert src.concat_target == 8.0 and src.has_speaker
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

        def transcribe(self, wav, **kw):
            self.n += 1
            if self.n == 5:
                raise RuntimeError("boom")   # 模拟中途崩
            return [_Seg()], _Info()

    monkeypatch.setattr(pl, "_whisper_model", lambda *a, **k: _FakeModel())
    monkeypatch.setattr(pl, "load_wav_mono", lambda p: (np.zeros(1600, np.float32), 16000))
    rows = [{"audio": f"{i}.wav", "text": ""} for i in range(8)]
    saved: list[list[dict]] = []
    out, bad = pl._transcribe_manifest(rows, "tl", ("tl", "en"),
                                       checkpoint=saved.append, checkpoint_every=2)
    assert len(out) == 7 and bad == 1
    assert saved, "从未落盘"
    mid = saved[0]
    assert len(mid) == len(rows), "落盘丢了未处理的行，重跑会漏数据"
    assert sum(1 for r in mid if r["text"]) == 2   # 前两条已转好
    # 用落盘结果重跑：已有文本的直接放行，不会重复转写
    again, _ = pl._transcribe_manifest(mid, "tl", ("tl", "en"))
    assert sum(1 for r in again if r["text"] == "kumusta") >= 2
