"""数据管线核心逻辑自测：裁静音/归一化/加工/混合。"""
import json
import numpy as np
import soundfile as sf

from voxft.data.pipeline import (
    Options, mix_manifests, peak_normalize, process_dataset, trim_silence,
)
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
