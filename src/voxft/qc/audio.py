"""合成音频的自动质检指标。

移植自 OmniVoice（/Users/dunxu.liu/workspace/others/OmniVoice/api.py），阈值原样保留，
不自己发明——线上就是按这些门限判 badcase 的，评测口径与生产对齐才有意义：
  metallic_resonance  api.py:5362-5442（_narrowband_resonance_metrics + 判定）
  floor_separation_db api.py:1254-1272（_frame_floor_separation_db），门限 18.0dB
                      来自 api_contract.py:15-17 REFERENCE_FLOOR_SEPARATION_MIN_DB
  speech_ratio        api.py:1843-1895（_waveform_loudness_profile 的 gate）

⚠️ **metallic_resonance 在本项目的用法下没有区分力，只能当参考值、不能当门禁**（实测校正）：
拿 84 条 48kHz 原始模型输出对照人工盲听，自动检出 4 条（base 1 / checkpoint 3），
**人工对这 4 条全部判 noise=False、自然度 5/5**；反过来人工唯一标了 noise=True 的那条
score 只有 0.0694，远低于门限，**漏报**。4 误报 1 漏报 0 命中。
原因是那套阈值（peak_ratio≥0.20、连续≥5帧、占比均值≥0.28）是在 OmniVoice
**后处理过**的音频上标定的——它上线前有 peak ceiling 0.94、level match、可选 noise gate，
频谱形态与这里的裸输出不同；4 条误报的 score 全挤在 0.29-0.33，刚好压线，也说明门限
对这个分布太松。**没有人工标注量之前不要重新标定，也不要拿它否决任何一轮微调。**

刻意**没有**移植的：
  duration_off_reference —— 在 OmniVoice 里是死代码，5 个调用点全部传 ref_duration=None
    （api.py:9037-9040 的理由：VoxCPM 的参考音频只是音色锚，其时长与期望输出长度无关）。
  RuleDurationEstimator —— 200 行 / 600 语种的 unicode 权重表（Apache-2.0）。这里不需要
    绝对期望时长：A/B 比的是 checkpoint 与 base 的相对值，用 chars_per_sec 对照即可，
    省掉一次大移植。真要绝对门限再引入。

speaker_sim 用 librosa MFCC 余弦，是 OmniVoice 自己的 mfcc_v1 回退档（api.py:3614-3642），
它把这一档标为 reliability=low_mfcc_fallback。真正的说话人嵌入是 modelscope 的
ERes2NetV2（iic/speech_eres2netv2_sv_zh-cn_16k-common），本项目不引 modelscope。
"""
from __future__ import annotations

import numpy as np

# 与 OmniVoice 的环境变量默认值一致（api.py:5362-5370）
METALLIC_PEAK_RATIO_MIN = 0.20
METALLIC_SUSTAINED_MIN_RUN = 5
METALLIC_SUSTAINED_MIN_RATIO = 0.28
METALLIC_MIN_FREQ_HZ = 3000.0
METALLIC_MIN_BAND_ENERGY_RATIO = 0.01
# 噪底与有声电平的分隔低于此值判 low_snr
FLOOR_SEPARATION_MIN_DB = 18.0


def metallic_resonance(wav: np.ndarray, sr: int) -> dict:
    """3-10kHz 内单峰持续共振 = 金属音/玻璃音。

    不是谱质心、也不是谐波噪声比：判据是「单个频点吃掉该频段 20% 以上能量」且
    「3-10kHz 占 100Hz-10kHz 总能量 1% 以上」，在有声帧上连续 ≥5 帧、且占比均值 ≥0.28。
    下限取 3kHz 而非 1.8kHz，因为 1.8kHz 会撞上普通 F2/F3 共振峰、把干净女声误判成金属音。
    """
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size < max(0.45, sr * 0.45) or sr < 12000:
        return {"metallic": None, "metallic_score": None, "metallic_max_run": None}
    frame = max(256, int(0.040 * sr))
    hop = max(128, int(0.020 * sr))
    if arr.size < frame:
        return {"metallic": None, "metallic_score": None, "metallic_max_run": None}
    window = np.hanning(frame).astype(np.float32)
    spectra, rms_values = [], []
    for start in range(0, arr.size - frame + 1, hop):
        block = arr[start:start + frame]
        rms_values.append(float(np.sqrt(np.mean(np.square(block), dtype=np.float64))))
        spectra.append(np.abs(np.fft.rfft(block * window)) ** 2)
    if len(spectra) < 8:
        return {"metallic": None, "metallic_score": None, "metallic_max_run": None}
    rms = np.asarray(rms_values, dtype=np.float64)
    active = rms >= max(0.008, float(np.percentile(rms, 70)) * 0.22)
    if int(active.sum()) < 6:
        return {"metallic": None, "metallic_score": None, "metallic_max_run": None}
    spec = np.asarray(spectra, dtype=np.float64)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    nyq = min(10000.0, sr / 2.0)
    band = (freqs >= METALLIC_MIN_FREQ_HZ) & (freqs <= nyq)
    if not np.any(band):
        return {"metallic": None, "metallic_score": None, "metallic_max_run": None}
    band_spec = spec[:, band]
    speech = (freqs >= 100.0) & (freqs <= nyq)
    high_ratio = np.sum(band_spec, axis=1) / np.maximum(np.sum(spec[:, speech], axis=1), 1e-12)
    peak_ratio = np.max(band_spec, axis=1) / np.maximum(np.sum(band_spec, axis=1), 1e-12)
    sustained = ((peak_ratio >= METALLIC_PEAK_RATIO_MIN)
                 & (high_ratio >= METALLIC_MIN_BAND_ENERGY_RATIO) & active)
    run = max_run = 0
    for value in sustained.tolist():
        run = run + 1 if value else 0
        max_run = max(max_run, run)
    score = float(np.mean(sustained[active]))
    return {"metallic": bool(max_run >= METALLIC_SUSTAINED_MIN_RUN
                             and score >= METALLIC_SUSTAINED_MIN_RATIO),
            "metallic_score": round(score, 4), "metallic_max_run": max_run}


def floor_separation_db(wav: np.ndarray, sr: int) -> float | None:
    """有声电平 p85 与噪底 p15 的分隔（dB）。低于 18dB 判 low_snr。

    这是能量分位差，不是真实 SNR——与 pipeline.audio_metrics 的 energy_range_db 同类，
    只是帧长与分位点取了线上那一套，便于和生产的 quality_issues 对照。
    """
    y = np.asarray(wav, dtype=np.float64).reshape(-1)
    frame = int(0.05 * sr)
    if frame <= 0 or y.size < frame * 4:
        return None
    hop = max(1, frame // 2)
    count = (y.size - frame) // hop + 1
    if count < 8:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop][:count]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    floor_db, speech_db = np.percentile(db, [15.0, 85.0])
    return round(float(speech_db - floor_db), 2)


def speaker_sim(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float | None:
    """MFCC 均值向量的余弦相似度（OmniVoice 的 mfcc_v1 回退档，reliability=low）。

    只能用于同一 ref 下 base 与 checkpoint 的**相对**比较，绝对值没有校准意义。
    短于 0.5s 不给分。
    """
    import librosa

    def feat(wav, sr):
        w = np.asarray(wav, dtype=np.float32).reshape(-1)
        if w.size / max(sr, 1) < 0.5:
            return None
        if sr != 16000:
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)
        m = librosa.feature.mfcc(y=w, sr=16000, n_mfcc=20)
        c = librosa.feature.spectral_centroid(y=w, sr=16000)[0]
        z = librosa.feature.zero_crossing_rate(w)[0]
        return np.concatenate([m.mean(axis=1), m.std(axis=1), [c.mean(), c.std(),
                                                               z.mean(), z.std()]])

    fa, fb = feat(a, sr_a), feat(b, sr_b)
    if fa is None or fb is None:
        return None
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
    return round(float(np.dot(fa, fb) / denom), 4) if denom > 1e-12 else 0.0


def speech_ratio(wav: np.ndarray, sr: int) -> float | None:
    """有声帧占比。门限取 OmniVoice `_waveform_loudness_profile` 那一套
    （gate = max(-52, p90-32) dB，帧 40ms，api.py:1843-1895）。

    用来把「语速变慢」和「尾部多垫静音」分开：两者的 audio_sec 都会变大、
    chars_per_sec 都会变小，但前者是模型行为、后者裁掉就行，处置完全相反。
    """
    y = np.asarray(wav, dtype=np.float64).reshape(-1)
    frame = int(0.04 * sr)
    if frame <= 0 or y.size < frame * 4:
        return None
    hop = max(1, frame // 2)
    count = (y.size - frame) // hop + 1
    if count < 4:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop][:count]
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    gate = max(-52.0, float(np.percentile(db, 90)) - 32.0)
    return round(float(np.mean(db >= gate)), 4)


def analyze(wav: np.ndarray, sr: int, ref_wav: np.ndarray | None = None,
            ref_sr: int | None = None) -> dict:
    """一次装齐全部质检指标；音频只加载一次。"""
    out = metallic_resonance(wav, sr)
    sep = floor_separation_db(wav, sr)
    out["floor_separation_db"] = sep
    out["low_snr"] = None if sep is None else bool(sep < FLOOR_SEPARATION_MIN_DB)
    out["audio_sec"] = round(len(np.asarray(wav).reshape(-1)) / max(sr, 1), 3)
    out["speech_ratio"] = speech_ratio(wav, sr)
    out["speaker_sim"] = (speaker_sim(wav, sr, ref_wav, ref_sr)
                          if ref_wav is not None else None)
    return out
