from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..paths import DATA_PROCESSED, DATA_RAW

TARGET_SR = 16000
_TERMINAL_PUNCT = ".!?。！？…"


@dataclass
class Options:
    min_dur: float = 3.0
    max_dur: float = 30.0
    ref_audio_ratio: float = 0.4     # 官方建议 30-50% 样本带 ref_audio
    ref_min_dur: float = 3.0         # ref 片段时长约束，对齐线上 5-10s 的参考音频
    ref_max_dur: float = 10.0
    val_ratio: float = 0.02
    val_max: int = 200               # val 上限，防说话人少的源切出巨大验证集
    utmos_min: float | None = None       # 如 3.5；None = 不做 UTMOS 过滤
    whisper_lang: str | None = None      # "th"/"tl"/"zh"；None = 不做转写校验
    whisper_min_sim: float = 0.55
    concat_target: float | None = None   # 短句语料：同会话拼接到约该秒数
    sent_sep: str = ". "                 # 拼接时的句间分隔（泰语传 " "）
    control_ratio: float = 0.5       # 带 (情绪/语速/音量) 控制前缀的样本比例
    control_zh_ratio: float = 0.5    # 前缀用中文的比例，其余用英文（对齐线上 prompt 语言）
    pseudo_speaker: bool = False     # 无说话人列时聚类伪说话人，以启用 ref 配对
    pseudo_speaker_threshold: float = 0.86
    metrics: bool = True             # 计算 f0/能量/语速/SNR：控制前缀与表现力筛选都依赖它
    min_snr_db: float | None = None  # 如 12.0；None = 不按 SNR 过滤
    min_f0_std: float | None = None  # 如 1.5（半音）；只保留有语调起伏的样本
    target_dbfs: float = -24.0       # 按说话人整体增益对齐，保留条内与条间动态
    seed: int = 42


@dataclass
class Clip:
    wav: np.ndarray
    text: str
    speaker: str = "default"
    emotion: str = ""
    session: str = ""


# ---------------------------------------------------------------- 音频基础处理

def load_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(path, dtype="float32", always_2d=True)
    return arr.mean(axis=1), sr


def _frame_rms(wav: np.ndarray, sr: int, win: float = 0.025) -> np.ndarray:
    n = max(256, int(win * sr))
    return librosa.feature.rms(y=wav, frame_length=n, hop_length=n // 2)[0]


def trim_silence(wav: np.ndarray, sr: int, floor: float = 1e-3,
                 tail_keep: float = 0.3) -> np.ndarray:
    """裁掉首尾静音；尾部最多保留 tail_keep 秒（官方要求 <0.5s，防生成失控）。

    阈值取"峰值 × floor"与"实测底噪 × 3"的较大者：众包语料底噪高，
    只用相对峰值的固定门限（-60dB）经常整条裁不动。
    """
    peak = float(np.abs(wav).max())
    if peak < 1e-8:
        return wav
    rms = _frame_rms(wav, sr)
    noise = float(np.percentile(rms, 10)) if rms.size else 0.0
    thr = max(peak * floor, noise * 3.0)
    nz = np.nonzero(np.abs(wav) > thr)[0]
    if len(nz) == 0:
        return wav
    start = max(0, nz[0] - int(0.05 * sr))
    end = min(len(wav), nz[-1] + 1 + int(tail_keep * sr))
    return wav[start:end]


def peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.abs(wav).max())
    return wav * (peak / m) if m > 1e-8 else wav


def rms_dbfs(wav: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    return 20.0 * math.log10(max(r, 1e-8))


def apply_speaker_gain(records: list[dict], target_dbfs: float,
                       progress=None) -> int:
    """按说话人整体增益对齐响度，而不是逐条峰值归一。

    逐条归一会把喊叫和耳语拉到同一响度，抹掉"音量=情绪强度"这条线索——
    这正是情感语料训完仍然平淡的原因之一。这里对同一说话人施加同一个增益，
    使其响度中位数落在 target_dbfs，条与条之间的相对强弱完整保留。
    """
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_spk[rec["speaker"]].append(rec)
    changed = 0
    for spk, recs in by_spk.items():
        med = float(np.median([r["_rms_dbfs"] for r in recs]))
        gain_db = target_dbfs - med
        gain = 10.0 ** (gain_db / 20.0)
        for rec in recs:
            wav, sr = load_wav_mono(rec["audio"])
            out = wav * gain
            m = float(np.abs(out).max())
            if m > 0.97:  # 削波保护：整个说话人一起回退，保持组内相对关系
                out = out * (0.97 / m)
            sf.write(rec["audio"], out, sr)
            rec["_rms_dbfs"] = rms_dbfs(out)
            changed += 1
            if progress and changed % 500 == 0:
                progress(f"响度对齐 {changed}/{len(records)} 条")
    if progress:
        progress(f"响度对齐完成：{changed} 条 / {len(by_spk)} 个说话人")
    return changed


# ---------------------------------------------------------------- 表现力指标

def audio_metrics(wav: np.ndarray, sr: int, text: str) -> dict:
    """算出用于"表现力筛选 + 控制前缀"的客观指标。

    f0_std_st  语调起伏（半音标准差）——念稿声这项很低
    energy_std_db  音量起伏
    rate       语速（有空格按词/秒，否则按字/秒）
    snr_db     粗略信噪比（帧能量 p90 / p10）
    """
    dur = len(wav) / sr
    rms = _frame_rms(wav, sr)
    voiced = rms > max(float(np.percentile(rms, 40)), 1e-5)
    snr = 20.0 * math.log10(max(float(np.percentile(rms, 90)), 1e-8)
                            / max(float(np.percentile(rms, 10)), 1e-8))
    energy_std = float(np.std(20.0 * np.log10(np.maximum(rms[voiced], 1e-8)))) \
        if voiced.any() else 0.0
    f0_std = 0.0
    try:
        f0 = librosa.yin(wav, fmin=60, fmax=400, sr=sr,
                         frame_length=1024, hop_length=256)
        f0 = f0[np.isfinite(f0) & (f0 > 60) & (f0 < 400)]
        if f0.size > 5:
            st = 12.0 * np.log2(f0 / float(np.median(f0)))
            f0_std = float(np.std(st))
    except Exception:
        pass
    units = len(text.split()) if " " in text.strip() else len(text)
    return {
        "f0_std_st": round(f0_std, 2),
        "energy_std_db": round(energy_std, 2),
        "rate": round(units / dur, 2) if dur > 0 else 0.0,
        "snr_db": round(snr, 1),
    }


# ---------------------------------------------------------------- 伪说话人聚类

def _embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """MFCC 均值+标准差做粗粒度音色向量（够用于把同一个人的片段聚到一起）。

    丢掉 c0：它只反映整体能量，留着会让所有样本的余弦相似度都 >0.95，聚不出人。
    """
    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20)[1:]
    v = np.concatenate([m.mean(axis=1), m.std(axis=1)]).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def cluster_pseudo_speakers(embs: list[np.ndarray], threshold: float = 0.6,
                            max_clusters: int = 2000) -> list[str]:
    """在线 leader 聚类：与已有质心余弦相似度超阈值就并入，否则开新簇。

    先减掉全语料均值（等价于语料级倒谱均值归一），把录音通道这类共性成分去掉，
    余弦才有区分度。无说话人列的源（FLEURS / 情感语料）靠它拿到 ref 配对能力；
    跨说话人乱配会损害克隆，所以宁可簇分得细一点——阈值调高只会让配对变少，不会配错。
    """
    if not embs:
        return []
    mat = np.stack(embs)
    mat = mat - mat.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    embs = list(mat / np.maximum(norms, 1e-8))
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[str] = []
    for e in embs:
        if centroids:
            sims = np.array([float(np.dot(e, c)) for c in centroids])
            j = int(sims.argmax())
            if sims[j] >= threshold:
                centroids[j] = (centroids[j] * counts[j] + e) / (counts[j] + 1)
                nrm = float(np.linalg.norm(centroids[j]))
                if nrm > 1e-8:
                    centroids[j] /= nrm
                counts[j] += 1
                labels.append(f"c{j:04d}")
                continue
        if len(centroids) >= max_clusters:
            labels.append("c_rest")
            continue
        centroids.append(e.copy())
        counts.append(1)
        labels.append(f"c{len(centroids) - 1:04d}")
    return labels


# ---------------------------------------------------------------- 控制前缀

# VoxCPM2 的情绪/风格控制就是文本前缀 "(控制指令)正文"（见官方 cli.build_final_text）。
# 训练文本里完全没有前缀，LoRA 会把基座的指令跟随能力冲掉，情绪 prompt 越调越不灵；
# 所以这里按情绪标签 + 实测语速/音量，给一部分样本自动生成前缀，另一部分保持裸文本。
_EMO_PHRASES: dict[str, dict[str, list[str]]] = {
    "neutral": {"zh": ["平静地", "语气自然", "正常语调"],
                "en": ["neutral tone", "calm and natural", "plain delivery"]},
    "angry": {"zh": ["愤怒地", "生气，语气冲", "暴怒，音量大"],
              "en": ["angry", "furious tone", "shouting in anger"]},
    "anger": {"zh": ["愤怒地", "生气，语气冲", "暴怒，音量大"],
              "en": ["angry", "furious tone", "shouting in anger"]},
    "happy": {"zh": ["开心地", "语气愉快", "兴奋，语调上扬"],
              "en": ["happy", "cheerful tone", "excited and upbeat"]},
    "happiness": {"zh": ["开心地", "语气愉快", "兴奋，语调上扬"],
                  "en": ["happy", "cheerful tone", "excited and upbeat"]},
    "sad": {"zh": ["伤心地", "低落，声音发闷", "哽咽，语速慢"],
            "en": ["sad", "sorrowful tone", "downcast, slow"]},
    "sadness": {"zh": ["伤心地", "低落，声音发闷", "哽咽，语速慢"],
                "en": ["sad", "sorrowful tone", "downcast, slow"]},
    "fearful": {"zh": ["害怕地", "紧张发抖", "惊恐，声音发紧"],
                "en": ["fearful", "scared and trembling", "terrified"]},
    "fear": {"zh": ["害怕地", "紧张发抖", "惊恐，声音发紧"],
             "en": ["fearful", "scared and trembling", "terrified"]},
    "surprised": {"zh": ["惊讶地", "吃惊，音调上扬", "难以置信"],
                  "en": ["surprised", "astonished", "in disbelief"]},
    "frustration": {"zh": ["烦躁地", "无奈又不耐烦", "压着火气"],
                    "en": ["frustrated", "exasperated tone", "holding back anger"]},
    "frustrated": {"zh": ["烦躁地", "无奈又不耐烦", "压着火气"],
                   "en": ["frustrated", "exasperated tone", "holding back anger"]},
    "disgust": {"zh": ["厌恶地", "嫌弃的语气"],
                "en": ["disgusted", "contemptuous tone"]},
}
_RATE_PHRASES = {"slow": {"zh": ["语速慢"], "en": ["slow paced"]},
                 "fast": {"zh": ["语速快"], "en": ["fast paced"]}}
_VOL_PHRASES = {"quiet": {"zh": ["轻声", "音量小"], "en": ["soft voice", "quiet"]},
                "loud": {"zh": ["音量大"], "en": ["loud"]}}


def _bucket(value: float, lo: float, hi: float, names: tuple[str, str, str]) -> str:
    return names[0] if value < lo else (names[2] if value > hi else names[1])


def build_control(rec: dict, rng: random.Random, zh_ratio: float,
                  rate_lo: float, rate_hi: float,
                  vol_lo: float, vol_hi: float) -> str:
    """生成 "(愤怒地，语速快，音量大)" 这样的控制前缀；无可用信息时返回空串。

    前缀语言只用中英文——线上的情绪 prompt 就是中英文写的，用目标语言写反而对不上。
    """
    lang = "zh" if rng.random() < zh_ratio else "en"
    parts: list[str] = []
    emo = (rec.get("emotion") or "").lower()
    pool = _EMO_PHRASES.get(emo)
    if pool:
        parts.append(rng.choice(pool[lang]))
    rate = _bucket(rec.get("rate", 0.0), rate_lo, rate_hi,
                   ("slow", "normal", "fast"))
    if rate != "normal" and rng.random() < 0.6:
        parts.append(rng.choice(_RATE_PHRASES[rate][lang]))
    vol = _bucket(rec.get("_rms_dbfs", -24.0), vol_lo, vol_hi,
                  ("quiet", "normal", "loud"))
    if vol != "normal" and rng.random() < 0.4:
        parts.append(rng.choice(_VOL_PHRASES[vol][lang]))
    if not parts:
        return ""
    sep = "，" if lang == "zh" else ", "
    return sep.join(parts)


def apply_control_prefixes(records: list[dict], opts: Options,
                           rng: random.Random) -> int:
    """给 control_ratio 比例的样本加控制前缀；其余保持裸文本，保住无前缀推理路径。"""
    if opts.control_ratio <= 0 or not records:
        return 0
    rates = [r.get("rate", 0.0) for r in records]
    vols = [r.get("_rms_dbfs", -24.0) for r in records]
    rate_lo, rate_hi = float(np.percentile(rates, 25)), float(np.percentile(rates, 75))
    vol_lo, vol_hi = float(np.percentile(vols, 25)), float(np.percentile(vols, 75))
    n = 0
    for rec in records:
        if rng.random() >= opts.control_ratio:
            continue
        ctrl = build_control(rec, rng, opts.control_zh_ratio,
                             rate_lo, rate_hi, vol_lo, vol_hi)
        if not ctrl:
            continue
        rec["control"] = ctrl
        rec["text"] = f"({ctrl}){rec['text']}"
        n += 1
    return n


# ---------------------------------------------------------------- 转写 / 质检

def _whisper_model(lang: str, size: str = "medium"):
    from faster_whisper import WhisperModel  # 可选依赖：uv sync --group qc
    return WhisperModel(size, device="auto", compute_type="auto")


def _whisper_similarity(model, wav: np.ndarray, sr: int,
                        ref_text: str) -> tuple[float, str]:
    """转写并与参考文本比相似度；返回 (相似度, 检测语种)。不指定语言以便检测混入语种。"""
    import torch
    batched = torch.cuda.is_available()  # GPU 批式转写约快 3-4 倍
    segs, info = model.transcribe(wav, vad_filter=True, batched=batched)
    hyp = "".join(s.text for s in segs).lower().replace(" ", "")
    ref = ref_text.lower().replace(" ", "")
    return SequenceMatcher(None, hyp, ref).ratio(), getattr(info, "language", "")


def _read_manifest(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _transcribe_manifest(rows: list[dict], lang: str,
                         progress=None) -> tuple[list[dict], int]:
    """给无文本的条目批量 Whisper 转写；语种不符或空转写直接丢弃。

    用 large-v3：转写结果直接成为训练标签，准确率优先（质检用 medium 即可）。
    保留标点——标点是文本 TTS 里唯一的韵律控制信号，去掉就等于教模型念平。
    """
    import torch
    model = _whisper_model(lang, "large-v3")
    batched = torch.cuda.is_available()
    out, bad = [], 0
    for i, row in enumerate(rows):
        if progress and i % 200 == 0:
            progress(f"转写 {i}/{len(rows)}")
        if row.get("text"):
            out.append(row)
            continue
        try:
            wav, sr = load_wav_mono(row["audio"])
        except Exception:
            bad += 1
            continue
        try:
            segs, info = model.transcribe(wav, vad_filter=True, batched=batched)
            text = " ".join(s.text.strip() for s in segs).strip()
        except Exception:
            bad += 1
            continue
        det = getattr(info, "language", "")
        if not text or (det and det.split("-")[0] != lang):
            bad += 1
            continue
        out.append({**row, "text": text})
    return out, bad


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: v for k, v in rec.items()
                                if not k.startswith("_")},
                               ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 解码 / 拼接

def _decoded_clips(rows: list[dict], stats: dict):
    """逐条解码 → 16k → 裁静音（不做逐条响度归一，留给按说话人的增益对齐）。"""
    for row in rows:
        try:
            wav, sr = load_wav_mono(row["audio"])
        except Exception:
            stats["drop_decode"] += 1
            continue
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        yield Clip(trim_silence(wav, TARGET_SR), str(row["text"]).strip(),
                   row.get("speaker", "default"), row.get("emotion", ""),
                   row.get("session", ""))


def _punctuated(text: str, sep: str) -> str:
    """拼接前给分句补上句末标点；泰语传 sep=" " 时按其书写习惯不加句点。"""
    t = text.strip()
    if not t or not sep.strip():
        return t
    return t if t[-1] in _TERMINAL_PUNCT else t + sep.strip()[0]


def _concatenated(clips, target: float, max_dur: float, sep: str,
                  rng: random.Random):
    """短句拼接：只在"同说话人 + 同一次录音"内按原始顺序拼，句间补标点、间隔随机。

    早先的实现按说话人把彼此无关的孤立词等间隔粘成长样本，训出来就是
    "报菜名"式的匀速无起伏念稿声。这里三点都改了：分组限定到同一次录音、
    文本补句末标点、停顿在 0.15-0.5s 之间抖动（真人停顿本来就不等长）。
    """
    # _punctuated 已把句末标点补在每段结尾，段间只需要空白，否则会出现 "alpha.. beta"
    sep_ws = "".join(ch for ch in sep if ch.isspace()) or " "
    acc: dict[tuple, list] = {}

    def flush(key):
        wavs, texts = acc.pop(key)
        return Clip(np.concatenate(wavs), "".join(texts).strip(), key[0],
                    "", key[1])

    for clip in clips:
        dur = len(clip.wav) / TARGET_SR
        key = (clip.speaker, clip.session)
        if dur >= target:  # 本身已够长，直接放行
            yield clip
            continue
        wavs, texts = acc.get(key, ([], []))
        if wavs and (sum(len(w) for w in wavs) / TARGET_SR + dur) > max_dur:
            acc[key] = [wavs, texts]
            yield flush(key)
            wavs, texts = [], []
        if wavs:
            gap = np.zeros(int(rng.uniform(0.15, 0.5) * TARGET_SR), dtype=np.float32)
            wavs.append(gap)
            texts.append(sep_ws)
        wavs.append(clip.wav)
        texts.append(_punctuated(clip.text, sep))
        acc[key] = [wavs, texts]
        if sum(len(w) for w in wavs) / TARGET_SR >= target:
            yield flush(key)
    for key in list(acc):
        wavs, _ = acc[key]
        if sum(len(w) for w in wavs) / TARGET_SR >= 3.0:  # 尾部：够最短时长才保留
            yield flush(key)
        else:
            acc.pop(key)


# ---------------------------------------------------------------- 主流程

def options_for(source_id: str, **overrides) -> Options:
    """按 registry 的数据源策略生成加工参数（UI 与 CLI 共用，避免两处各写一份）。"""
    from .registry import get_source
    src = get_source(source_id)
    opts = Options(
        ref_audio_ratio=0.4 if (src.has_speaker or src.pseudo_speaker) else 0.0,
        utmos_min=3.5 if src.qc == "full" else None,
        whisper_lang=src.lang if src.qc in ("whisper", "full") else None,
        concat_target=src.concat_target or None,
        sent_sep=src.separator(),
        pseudo_speaker=src.pseudo_speaker,
        control_ratio=0.5 if src.expressive else 0.25,
    )
    for k, v in overrides.items():
        if v is not None:
            setattr(opts, k, v)
    return opts


def _source_lang(source_id: str) -> str:
    from .registry import get_source
    try:
        return get_source(source_id).lang
    except KeyError:
        return "zh"


def process_dataset(source_id: str, out_name: str | None = None,
                    opts: Options | None = None, max_items: int | None = None,
                    progress=None) -> dict:
    """加工 data/raw/<source_id>/manifest.jsonl → data/processed/<out_name>/

    16k 重采样 → 裁静音 → 时长过滤 → 质检 → 表现力指标 → 伪说话人聚类 →
    按说话人响度对齐 → 控制前缀 → ref_audio 配对 → 分层切分 → train/val JSONL。
    """
    opts = opts or Options()
    rng = random.Random(opts.seed)
    src = DATA_RAW / source_id
    manifest = src / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"先运行下载：python -m voxft.data.download --source {source_id}")
    out = DATA_PROCESSED / (out_name or source_id)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest)
    truncated = bool(max_items)
    if max_items:
        rows = rows[:max_items]

    if any(not r.get("text") for r in rows):  # 缺文本的条目：转写 + 语种过滤
        lang = _source_lang(source_id)
        n_missing = sum(1 for r in rows if not r.get("text"))
        if progress:
            progress(f"{source_id}: {n_missing} 条缺文本，自动 Whisper 转写 + 语种过滤（{lang}）")
        rows, n_bad = _transcribe_manifest(rows, lang, progress)
        if not truncated:
            _write_jsonl(rows, manifest)  # 写回，下次加工直接复用
        if progress:
            progress(f"转写完成：保留 {len(rows)} 条，丢弃 {n_bad} 条（语种不符/空）")
        if not rows:
            raise RuntimeError(f"{source_id}: 转写后无可用样本")

    whisper = None
    if opts.whisper_lang:
        if progress:
            progress(f"加载 Whisper 质检模型（{opts.whisper_lang}，首次需下载）...")
        whisper = _whisper_model(opts.whisper_lang)
        if progress:
            progress("Whisper 质检模型就绪")
    score_wav = None
    if opts.utmos_min is not None:
        from ..qc.utmos import get_scorer, last_error
        if progress:
            progress("加载 UTMOS 评分器（首次需下载权重 ~1.2GB，可能需几分钟）...")
        if get_scorer() is None:
            if progress:
                progress(f"警告: UTMOS 评分器不可用（{last_error()}），"
                         "跳过 UTMOS 质检，仅用其余质检项继续")
        else:
            from ..qc.utmos import score_wav

    kept: list[dict] = []
    embs: list[np.ndarray] = []
    stats = {"total": len(rows), "drop_decode": 0, "drop_duration": 0,
             "drop_lang": 0, "drop_whisper": 0, "drop_utmos": 0,
             "drop_snr": 0, "drop_flat": 0}
    samples = _decoded_clips(rows, stats)
    if opts.concat_target:
        if progress:
            progress(f"短句语料：同说话人+同会话拼接到约 {opts.concat_target}s"
                     f"（句末补标点，间隔 0.15-0.5s 抖动）")
        samples = _concatenated(samples, opts.concat_target, opts.max_dur,
                                opts.sent_sep, rng)
    for i, clip in enumerate(samples):
        if progress and i % 50 == 0:
            progress(f"加工 {source_id}: 已产出 {i} 条样本")
        dur = len(clip.wav) / TARGET_SR
        if not (opts.min_dur <= dur <= opts.max_dur):
            stats["drop_duration"] += 1
            continue
        if whisper is not None:
            try:
                sim, det = _whisper_similarity(whisper, clip.wav, TARGET_SR, clip.text)
            except Exception:
                sim, det = 0.0, ""
            if det and det.split("-")[0] != opts.whisper_lang:
                stats["drop_lang"] += 1
                continue
            if sim < opts.whisper_min_sim:
                stats["drop_whisper"] += 1
                continue
        if score_wav is not None:
            mos = score_wav(clip.wav, TARGET_SR)
            if mos is None or mos < opts.utmos_min:
                stats["drop_utmos"] += 1
                continue
        met = audio_metrics(clip.wav, TARGET_SR, clip.text) if opts.metrics else {}
        if opts.min_snr_db is not None and met.get("snr_db", 99) < opts.min_snr_db:
            stats["drop_snr"] += 1
            continue
        if opts.min_f0_std is not None and met.get("f0_std_st", 99) < opts.min_f0_std:
            stats["drop_flat"] += 1
            continue
        dst = audio_dir / f"{i:07d}.wav"
        sf.write(dst, clip.wav, TARGET_SR)
        rec = {"audio": str(dst), "text": clip.text, "speaker": clip.speaker,
               "duration": round(dur, 2), "_rms_dbfs": rms_dbfs(clip.wav)}
        if clip.emotion:
            rec["emotion"] = clip.emotion
        rec.update(met)
        kept.append(rec)
        if opts.pseudo_speaker:
            embs.append(_embed(clip.wav, TARGET_SR))

    stats["kept"] = len(kept)
    if not kept:
        raise RuntimeError(f"{source_id}: 过滤后无剩余样本，请放宽质检参数")

    if opts.pseudo_speaker and embs:
        if progress:
            progress("无说话人列：聚类伪说话人（用于同说话人 ref 配对）...")
        labels = cluster_pseudo_speakers(embs, opts.pseudo_speaker_threshold)
        for rec, lab in zip(kept, labels):
            rec["speaker"] = lab
        n_clusters = len(set(labels))
        stats["pseudo_speakers"] = n_clusters
        if progress:
            progress(f"伪说话人聚类完成：{n_clusters} 簇 / {len(kept)} 条"
                     f"（阈值 {opts.pseudo_speaker_threshold}）")
            if n_clusters > 0.4 * len(kept):
                progress("警告: 簇数接近样本数，几乎配不出 ref_audio —— "
                         "把 pseudo_speaker_threshold 调低（如 0.45）再跑一次")

    if progress:
        progress(f"按说话人对齐响度到 {opts.target_dbfs} dBFS（保留条间动态）...")
    apply_speaker_gain(kept, opts.target_dbfs, progress)

    n_ctrl = apply_control_prefixes(kept, opts, rng)
    stats["with_control"] = n_ctrl
    if progress:
        progress(f"控制前缀：{n_ctrl}/{len(kept)} 条带 (情绪/语速/音量) 前缀，"
                 f"其余保持裸文本")

    # ref_audio：同说话人随机配另一条（官方建议 30-50%）；
    # 候选限定在 ref_min_dur~ref_max_dur，对齐线上参考音频的时长分布，
    # 也避免超长 ref 把序列撑爆后被训练脚本静默丢弃。
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for rec in kept:
        by_spk[rec["speaker"]].append(rec)
    for rec in kept:
        pool = [r for r in by_spk[rec["speaker"]]
                if r is not rec and opts.ref_min_dur <= r["duration"] <= opts.ref_max_dur]
        if pool and rng.random() < opts.ref_audio_ratio:
            # 同说话人里挑信噪比最好的几条之一，别拿噪声样本当参考
            pool.sort(key=lambda r: -r.get("snr_db", 0.0))
            rec["ref_audio"] = rng.choice(pool[:max(1, len(pool) // 4)])["audio"]

    # val：说话人内分层抽样 + 上限，避免把整个说话人从 train 里挖走
    val: list[dict] = []
    target = min(opts.val_max, max(1, int(len(kept) * opts.val_ratio)))
    for recs in by_spk.values():
        if len(val) >= target:
            break
        k = max(1, round(len(recs) * opts.val_ratio)) if len(recs) > 4 else 0
        if k:
            val.extend(rng.sample(recs, min(k, len(recs) - 1)))
    if not val:
        pick = rng.sample(kept, min(target, max(1, len(kept) // 10)))
        val = pick if len(pick) < len(kept) else pick[:1]
    val = val[:target]
    val_ids = {id(r) for r in val}
    train = [r for r in kept if id(r) not in val_ids]
    rng.shuffle(train)

    _write_jsonl(train, out / "train.jsonl")
    _write_jsonl(val, out / "val.jsonl")
    expressive = [r.get("f0_std_st", 0.0) for r in train]
    stats.update({
        "train": len(train), "val": len(val), "speakers": len(by_spk),
        "with_ref_audio": sum(1 for r in train if "ref_audio" in r),
        "f0_std_median": round(float(np.median(expressive)), 2) if expressive else 0.0,
        "output": str(out),
    })
    (out / "stats.json").write_text(
        json.dumps({"source_id": source_id, "options": asdict(opts), **stats},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def mix_manifests(parts: list[tuple[str, float]], out_name: str,
                  seed: int = 42, max_repeat: float = 3.0) -> dict:
    """按权重混合多个已加工数据集。

    parts: [(processed_name, weight), ...]，如 [("thai_ser", 0.45), ("fleurs_zh", 0.1)]
    小语料被 tile 到高权重时重复因子会很大（十几倍），直接把它训过拟合；
    max_repeat 给重复次数封顶，够不到的权重会按实际条数缩水并在 mix.json 里记下来。
    """
    if not parts:
        raise ValueError("parts 不能为空")
    rng = random.Random(seed)
    total_w = sum(w for _, w in parts)
    out = DATA_PROCESSED / out_name
    summary: dict = {}
    for split in ("train", "val"):
        rows_by_part = []
        for name, _w in parts:
            p = DATA_PROCESSED / name / f"{split}.jsonl"
            if not p.exists():
                raise FileNotFoundError(p)
            rows_by_part.append(_read_manifest(p))
        n_total = sum(len(c) for c in rows_by_part)
        mixed: list[dict] = []
        for (name, w), rows in zip(parts, rows_by_part):
            if not rows:
                continue
            want = max(1, round(w / total_w * n_total))
            cap = int(len(rows) * max_repeat)
            target = min(want, cap)
            tiled = (rows * ((target // len(rows)) + 1))[:target]
            summary[f"{name}/{split}"] = target
            if target < want:
                summary[f"{name}/{split}_capped_from"] = want
            mixed.extend(tiled)
        rng.shuffle(mixed)
        _write_jsonl(mixed, out / f"{split}.jsonl")
    (out / "mix.json").write_text(
        json.dumps({"parts": parts, "max_repeat": max_repeat, "counts": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), **summary}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--utmos-min", type=float, default=None)
    ap.add_argument("--whisper-lang", default=None)
    ap.add_argument("--control-ratio", type=float, default=None)
    ap.add_argument("--min-snr-db", type=float, default=None)
    args = ap.parse_args()
    o = options_for(args.source, utmos_min=args.utmos_min,
                    whisper_lang=args.whisper_lang,
                    control_ratio=args.control_ratio,
                    min_snr_db=args.min_snr_db)
    print(json.dumps(process_dataset(args.source, args.out, o, args.max_items),
                     ensure_ascii=False, indent=2))
