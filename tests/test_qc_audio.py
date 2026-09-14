"""质检指标移植自 OmniVoice 生产口径，阈值必须与线上一致，所以逐条钉住。"""
import numpy as np
import pytest

from voxft.qc.audio import (FLOOR_SEPARATION_MIN_DB, floor_separation_db,
                            metallic_resonance, speaker_sim)

SR = 48000


def _tone(freq, seconds, sr=SR, amp=0.3):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_metallic_resonance_flags_sustained_narrowband_peak():
    """判据是 3-10kHz 内单帧单点吃掉 ≥20% 能量且持续，不是谱质心也不是 HNR。"""
    assert metallic_resonance(_tone(5000, 1.0), SR)["metallic"] is True
    noisy = np.random.default_rng(0).normal(0, 0.3, SR).astype(np.float32)
    assert metallic_resonance(noisy, SR)["metallic"] is False
    # 低于 3kHz 不该触发：1.8kHz 会撞上普通 F2/F3 共振峰，把干净人声误判成金属音
    assert metallic_resonance(_tone(1500, 1.0), SR)["metallic"] is False


def test_metallic_resonance_returns_none_when_it_cannot_decide():
    """时长或采样率不足时给 None，不能给 False——False 会被下游当成"检测通过"。"""
    assert metallic_resonance(_tone(5000, 0.2), SR)["metallic"] is None
    assert metallic_resonance(_tone(5000, 1.0, sr=8000), 8000)["metallic"] is None


def test_floor_separation_orders_clean_above_noisy():
    sr = 16000
    t = np.arange(sr * 2) / sr
    clean = np.where(t < 1.0, 0.3 * np.sin(2 * np.pi * 200 * t), 0.0).astype(np.float32)
    noisy = (clean + np.random.default_rng(1).normal(0, 0.05, t.size)).astype(np.float32)
    sep_clean, sep_noisy = floor_separation_db(clean, sr), floor_separation_db(noisy, sr)
    assert sep_clean > sep_noisy
    assert sep_clean >= FLOOR_SEPARATION_MIN_DB > sep_noisy
    # 不足 4 帧长（0.2s @16k）时不给结论，而不是给一个假的 dB 值
    assert floor_separation_db(clean[:sr // 8], sr) is None


def test_speaker_sim_uses_embeddings_and_never_fakes_a_number(monkeypatch):
    """MFCC 余弦实测全部饱和在 0.985-0.996，做不了优化目标，已换成 WavLM X-vector。
    模型不可用时必须返回 None 并带原因——编一个数会让"没算"被当成"算出来很好"。"""
    from voxft.qc import audio as qc
    sr = 16000
    wav = np.zeros(sr, dtype=np.float32)

    monkeypatch.setattr(qc, "_spk_model", lambda: None)
    monkeypatch.setattr(qc, "_SPK_ERR", "OfflineMode: 连不上权重仓库")
    out = qc.analyze(wav, sr, wav, sr)
    assert out["speaker_sim"] is None
    assert out["speaker_sim_backend"] is None
    assert "连不上" in out["speaker_sim_error"]

    # 有嵌入时：都已 L2 归一化，点积即余弦（正交=0，同向=1）
    monkeypatch.setattr(qc, "speaker_embedding",
                        lambda w, s: np.array([1.0, 0.0]) if w is wav else np.array([0.0, 1.0]))
    other = np.zeros(sr, dtype=np.float32)
    assert qc.speaker_sim(wav, sr, other, sr) == 0.0
    monkeypatch.setattr(qc, "speaker_embedding", lambda w, s: np.array([1.0, 0.0]))
    assert qc.speaker_sim(wav, sr, other, sr) == 1.0


def test_speech_ratio_separates_padding_from_slow_speech():
    """同样 2 秒音频：一段全程有声、一半是尾部静音，speech_ratio 必须能分开，
    否则 chars_per_sec 变慢就分不清是模型说慢了还是多垫了静音。"""
    sr = 16000
    t = np.arange(sr * 2) / sr
    voiced = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    padded = np.concatenate([voiced[:sr], np.zeros(sr, dtype=np.float32)])
    from voxft.qc.audio import speech_ratio
    assert speech_ratio(voiced, sr) > 0.9
    assert 0.35 < speech_ratio(padded, sr) < 0.65
    assert speech_ratio(voiced[:sr // 8], sr) is None


def test_numeric_flag_covers_verbalized_numbers():
    """上游把数字 verbalize 之后文本里没有阿拉伯数字，但 Whisper 会把音频转写回
    "1,500 pesos" 这种符号形式，CER 照样虚高——所以必须支持显式标注。"""
    from voxft.eval import _is_numeric
    assert _is_numeric({"text": "Giá vé là 250.000 đồng."}) is True
    assert _is_numeric({"text": "Bayad ko ay isang libo't limang daan pesos."}) is False
    assert _is_numeric({"text": "isang libo't limang daan pesos", "numeric": True}) is True
    assert _is_numeric({"text": "Hindi ko inaasahan na babalik ka pa."}) is False


def test_over_read_uses_the_production_threshold():
    from voxft.eval import _over_read
    assert _over_read("a" * 15, "abcdefghij") is True      # 1.5× > 1.4× 门限
    assert _over_read("a" * 14, "abcdefghij") is False     # 1.4× 正好不触发
    assert _over_read("abc", "abcdefghij") is False        # 少读不归它管
    assert _over_read("anything", "") is False


@pytest.mark.parametrize("text,expected",
                         [("Harganya Rp250.000.", True), ("Saya tak sangka.", False)])
def test_numeric_autodetect(text, expected):
    from voxft.eval import _is_numeric
    assert _is_numeric({"text": text}) is expected
