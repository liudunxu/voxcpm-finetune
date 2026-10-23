"""Is speaker cluster `review_4f50387a` actually one person?

Reads only what was already extracted from the frozen MP4 (per-cue client refs)
plus the embedded metadata. Computes pairwise WavLM cosine across that speaker's
10 distinct references and prints DIS's own pitch/gender readings alongside.
No TTS, no ASR, no remote calls.
"""

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from voxft.qc.audio import speaker_embedding, speaker_sim

ROOT = Path("work/video_review/20261022_43_108")
COND = ROOT / "review-references" / "conditioning"
TARGET = "review_4f50387a"
SR = 16000


def read(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def f0_median(wav, sr=SR):
    import librosa

    frame = 2 ** math.ceil(math.log2(2 * sr / 60))
    if len(wav) < frame * 2:
        return None
    f0 = librosa.yin(wav, fmin=50, fmax=500, sr=sr, frame_length=frame)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else None


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    meta = json.loads((ROOT / "embedded_meta.json").read_text(encoding="utf-8"))["metadata"]
    refs_manifest = json.loads((ROOT / "review-references" / "review_references.json").read_text(encoding="utf-8"))
    by_index = {c["index"]: c for c in refs_manifest["cues"]}

    # speaker -> {sha: [cue indices]}
    groups = defaultdict(lambda: defaultdict(list))
    for cue in meta["cues"]:
        sp = cue.get("speaker")
        sha = ((cue.get("conditioning_audio") or {}).get("reference") or {}).get("sha256")
        if sp and sha:
            groups[sp][sha].append(cue["index"])

    for speaker, shas in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{'=' * 78}\nspeaker {speaker}: {len(shas)} 条不同参考 / "
              f"{sum(len(v) for v in shas.values())} 条 cue")
        if len(shas) < 2:
            print("  只有一条参考字节，无法做池内互证（跳过）")
            continue

        embs, rows = {}, []
        for sha, cues in sorted(shas.items(), key=lambda kv: kv[1][0]):
            path = COND / f"{sha}.wav"
            if not path.exists():
                print(f"  cue{cues} 参考缺失：{path.name}")
                continue
            wav = read(path)
            emb = speaker_embedding(wav, SR)
            if emb is None:
                print(f"  cue{cues} 嵌入失败")
                continue
            embs[sha[:8]] = emb
            q = next((c.get("reference_quality") or {} for c in meta["cues"]
                      if ((c.get("conditioning_audio") or {}).get("reference") or {}).get("sha256") == sha), {})
            rows.append({
                "sha": sha[:8], "cues": cues, "dur": len(wav) / SR,
                "f0_local": f0_median(wav), "f0_dis": q.get("median_f0_hz"),
                "gender": q.get("gender"), "gconf": q.get("gender_confidence"),
                "snr": q.get("snr_db"), "active": q.get("active_speech_ratio"),
            })

        print(f"  {'ref':<10}{'cues':<18}{'dur':>6}{'F0本地':>8}{'F0(DIS)':>9}"
              f"{'gender':>10}{'conf':>6}{'SNR':>6}{'有声占比':>9}")
        for r in rows:
            print(f"  {r['sha']:<10}{str(r['cues']):<18}{r['dur']:>6.2f}{r['f0_local']:>8.1f}"
                  f"{str(r['f0_dis']):>9}{str(r['gender']):>10}{str(r['gconf']):>6}"
                  f"{str(r['snr']):>6}{str(r['active']):>9}")

        pairs = []
        print("\n  两两 WavLM cosine（DIS 生产门限刻度 0.45，仅作参照）：")
        for a, b in itertools.combinations(rows, 2):
            c = cos(embs[a["sha"]], embs[b["sha"]])
            pairs.append((c, a, b))
        for c, a, b in sorted(pairs, reverse=True):
            flag = "  ← 低于 0.45" if c < 0.45 else ("  ← 低于 0.60" if c < 0.60 else "")
            print(f"    {a['sha']} (cue{a['cues'][0]:>2}) ↔ {b['sha']} (cue{b['cues'][0]:>2}): {c:.4f}{flag}")
        vals = [c for c, _, _ in pairs]
        print(f"\n  n={len(vals)} min={min(vals):.4f} 中位={sorted(vals)[len(vals) // 2]:.4f} max={max(vals):.4f}")
        low = [(c, a["sha"], b["sha"]) for c, a, b in pairs if c < 0.60]
        print(f"  低于 0.60 的对子：{len(low)}/{len(vals)}"
              + (f" → {[(round(c, 3), x, y) for c, x, y in sorted(low)[:6]]}" if low else ""))


if __name__ == "__main__":
    main()
