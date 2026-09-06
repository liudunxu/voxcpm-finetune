from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from ..paths import DATA_PROCESSED, DATA_RAW

TARGET_SR = 16000


@dataclass
class Options:
    min_dur: float = 3.0
    max_dur: float = 30.0
    ref_audio_ratio: float = 0.5     # 只给身份可靠的样本配对，不强凑比例
    ref_control_ratio: float = 0.3  # 全体目标中 ref + control 的期望占比
    ref_min_dur: float = 3.0         # ref 片段时长约束，对齐线上 5-10s 的参考音频
    ref_max_dur: float = 10.0
    val_ratio: float = 0.02
    val_max: int = 200               # 按组切分的软目标，不拆说话人来满足上限
    utmos_min: float | None = None       # 如 3.5；None = 不做 UTMOS 过滤
    whisper_lang: str | None = None      # "th"/"tl"/"zh"；None = 不做转写校验
    accept_langs: tuple[str, ...] = ()   # 允许的检测语种（留空=只认 whisper_lang）
    whisper_min_sim: float = 0.55
    control_ratio: float = 0.5       # 带 (情绪/语速/音量) 控制前缀的样本比例
    control_zh_ratio: float = 0.5    # 前缀用中文的比例，其余用英文（对齐线上 prompt 语言）
    pseudo_speaker: bool = False     # 可选的审计分组，不会赋予 ref 身份可信度
    pseudo_speaker_threshold: float = 0.86
    metrics: bool = True             # 声学描述，仅用于诊断，不作为表现力通过条件
    min_snr_db: float | None = None  # 仅兼容旧参数；启用会报错，能量分位差不是 SNR
    min_f0_std: float | None = None  # 仅兼容旧参数；不能按音高起伏硬筛
    target_dbfs: float = -24.0       # 按说话人整体增益对齐，保留条内与条间动态
    seed: int = 42


@dataclass
class Clip:
    wav: np.ndarray
    text: str
    speaker: str = "default"
    emotion: str = ""
    session: str = ""
    metadata: dict = field(default_factory=dict)


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
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    if peak < 1e-8:
        return wav
    rms = _frame_rms(wav, sr)
    noise = float(np.percentile(rms, 10)) if rms.size else 0.0
    # 无明显静音时，p10 可能是轻声语音；限制门限，避免把弱辅音当底噪裁掉。
    thr = max(peak * floor, min(noise * 3.0, peak * 0.01))
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
        # 未验证身份的组可能混人；不施加推测的说话人增益。
        if not all(r.get("speaker_verified", False) for r in recs):
            continue
        med = float(np.median([r["_rms_dbfs"] for r in recs]))
        gain_db = target_dbfs - med
        gain = 10.0 ** (gain_db / 20.0)
        peak = max(float(np.abs(load_wav_mono(r["audio"])[0]).max()) for r in recs)
        gain = min(gain, 0.97 / max(peak, 1e-8))
        for rec in recs:
            wav, sr = load_wav_mono(rec["audio"])
            out = wav * gain
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
    """声学描述；不用于硬筛平读或生成控制指令。

    f0_std_st  音高起伏（半音标准差），含声调、清浊音误差，不能代表自然度
    energy_std_db  音量起伏
    rate       语速（有空格按词/秒，否则按字/秒）
    energy_range_db  帧能量 p90 / p10，不是真实 SNR
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
        "energy_range_db": round(snr, 1),
    }


# ---------------------------------------------------------------- 伪说话人聚类

def _embed(wav: np.ndarray, sr: int) -> np.ndarray:
    """MFCC 均值+标准差做审计向量，不能证明同一说话人。

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
    余弦才有区分度。结果仅用于人工审计，不能直接拿来配 ref。
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
    for i, e in enumerate(embs):
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
            labels.append(f"unverified_{i}")
            continue
        centroids.append(e.copy())
        counts.append(1)
        labels.append(f"c{len(centroids) - 1:04d}")
    return labels


# ---------------------------------------------------------------- 控制前缀

# VoxCPM2 的情绪/风格控制就是文本前缀 "(控制指令)正文"（见官方 cli.build_final_text）。
# 训练文本里完全没有前缀，LoRA 会把基座的指令跟随能力冲掉，情绪 prompt 越调越不灵；
# 只按可信标签给一部分样本加前缀，另一部分保持裸文本。
_EMO_PHRASES: dict[str, dict[str, list[str]]] = {
    "neutral": {"zh": ["平静地", "语气自然", "正常语调"],
                "en": ["neutral tone", "calm and natural", "plain delivery"]},
    "angry": {"zh": ["愤怒地"], "en": ["angry"]},
    "anger": {"zh": ["愤怒地"], "en": ["angry"]},
    "happy": {"zh": ["开心地"], "en": ["happy"]},
    "happiness": {"zh": ["开心地"], "en": ["happy"]},
    "sad": {"zh": ["伤心地"], "en": ["sad"]},
    "sadness": {"zh": ["伤心地"], "en": ["sad"]},
    "fearful": {"zh": ["害怕地"], "en": ["fearful"]},
    "fear": {"zh": ["害怕地"], "en": ["fearful"]},
    "surprised": {"zh": ["惊讶地"], "en": ["surprised"]},
    "frustration": {"zh": ["烦躁地"], "en": ["frustrated"]},
    "frustrated": {"zh": ["烦躁地"], "en": ["frustrated"]},
    "disgust": {"zh": ["厌恶地"], "en": ["disgusted"]},
}
_RATE_PHRASES = {"slow": {"zh": ["语速慢"], "en": ["slow paced"]},
                 "fast": {"zh": ["语速快"], "en": ["fast paced"]}}
_VOL_PHRASES = {"quiet": {"zh": ["轻声", "音量小"], "en": ["soft voice", "quiet"]},
                "loud": {"zh": ["音量大"], "en": ["loud"]}}


def build_control(rec: dict, rng: random.Random, zh_ratio: float) -> str:
    """只使用经审核的中英文指令/标签；不从音量、空格数猜测表演方式。"""
    lang = "zh" if rng.random() < zh_ratio else "en"
    if rec.get("control_verified") is True:
        control = rec.get(f"control_{lang}") or rec.get("control_zh") or rec.get("control_en")
        if control:
            control = re.sub(r"[()（）]", "", str(control)).strip()
            if re.search(r"[\u0e00-\u0e7f]", control):
                raise ValueError("控制前缀只能使用中英文")
            return control
    parts: list[str] = []
    emo = (rec.get("emotion") or "").lower()
    pool = _EMO_PHRASES.get(emo) if rec.get("emotion_verified") is True else None
    if pool:
        parts.append(rng.choice(pool[lang]))
    if rec.get("control_verified") is True:
        for key, phrases in (("rate_label", _RATE_PHRASES), ("volume_label", _VOL_PHRASES)):
            if rec.get(key) in phrases:
                parts.append(rng.choice(phrases[rec[key]][lang]))
    if not parts:
        return ""
    sep = "，" if lang == "zh" else ", "
    return sep.join(parts)


def apply_control_prefixes(records: list[dict], opts: Options,
                           rng: random.Random) -> int:
    """给 control_ratio 比例的样本加控制前缀；其余保持裸文本，保住无前缀推理路径。"""
    if opts.control_ratio <= 0 or not records:
        return 0
    candidates = []
    for rec in records:
        if rec.get("reference_only"):
            continue
        ctrl = build_control(rec, rng, opts.control_zh_ratio)
        if ctrl:
            candidates.append((rec, ctrl))
    n_targets = sum(not r.get("reference_only") for r in records)
    selected = rng.sample(candidates, min(len(candidates), round(n_targets * opts.control_ratio)))
    for rec, ctrl in selected:
        rec["control"] = ctrl
        rec["text"] = f"({ctrl}){rec['text']}"
    return len(selected)


# ---------------------------------------------------------------- 转写 / 质检

# 转写标签会直接变成训练文本，所以用 large-v3；质检只是打分，medium 足够。
# 权重可用 VOXFT_WHISPER_MODEL / VOXFT_WHISPER_MODEL_LARGE 覆盖成本地目录或镜像仓库。
_WHISPER_ENV = {"medium": "VOXFT_WHISPER_MODEL",
                "large-v3": "VOXFT_WHISPER_MODEL_LARGE"}


def _whisper_model(lang: str, size: str = "medium", progress=None):
    """加载 faster-whisper 权重，带重试。

    large-v3 约 3GB，国内直连 huggingface.co 经常在 SSL 握手就超时；
    voxft 默认把 HF_ENDPOINT 指到 hf-mirror.com（`paths.load_dotenv`），
    这里再补上重试与一条能照着做的报错，别让 27 秒的握手超时把整批转写打回去。
    """
    import os
    import time as _time
    from faster_whisper import WhisperModel  # 可选依赖：uv sync --group qc

    name = os.environ.get(_WHISPER_ENV.get(size, ""), "") or size
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    last: Exception | None = None
    for attempt in range(3):
        try:
            if progress:
                progress(f"加载 Whisper {name}（endpoint={endpoint}，"
                         f"第 {attempt + 1}/3 次）...")
            return WhisperModel(name, device="auto", compute_type="auto")
        except Exception as exc:
            last = exc
            if progress:
                progress(f"加载失败（{exc}），20 秒后重试")
            _time.sleep(20)
    raise RuntimeError(
        f"Whisper 权重加载失败（{name}，endpoint={endpoint}）：{last}\n"
        f"排查顺序：\n"
        f"1) 确认 .env 里 HF_ENDPOINT=https://hf-mirror.com，且 HF_HOME 指向大盘\n"
        f"2) 命令行预下载（hf 不读项目 .env，要手动 export；\n"
        f"   走镜像必须关 xet，否则 reconstruction 阶段直连 CAS 会 401）：\n"
        f"   export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1\n"
        f"   hf download Systran/faster-whisper-{name}\n"
        f"3) 已有本地权重目录时：在 .env 设 "
        f"{_WHISPER_ENV.get(size, 'VOXFT_WHISPER_MODEL')}=/path/to/model"
    ) from last


def _whisper_similarity(model, wav: np.ndarray, sr: int,
                        ref_text: str) -> tuple[float, str]:
    """转写并与参考文本比相似度；返回 (相似度, 检测语种)。不指定语言以便检测混入语种。"""
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    segs, info = model.transcribe(wav, vad_filter=True)
    hyp = "".join(s.text for s in segs).lower().replace(" ", "")
    ref = ref_text.lower().replace(" ", "")
    return SequenceMatcher(None, hyp, ref).ratio(), getattr(info, "language", "")


def _read_manifest(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"JSONL 每行必须为对象: {manifest}")
    return rows


def _transcribe_manifest(rows: list[dict], lang: str, accept: tuple[str, ...] = (),
                         progress=None, checkpoint=None,
                         checkpoint_every: int = 300) -> tuple[list[dict], int]:
    """每 300 条及退出时保存完整原清单；坏例仅从本次输出排除，不删除原记录。"""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every 必须大于 0")
    working = [dict(r) for r in rows]
    todo = sum(1 for r in rows if not r.get("text") and not r.get("reference_only"))
    if not todo:
        return working, 0
    model = _whisper_model(lang, "large-v3", progress)
    if progress:
        progress(f"Whisper large-v3 就绪，待转写 {todo} 条")
    out, bad, done = [], 0, 0

    def _flush():
        if checkpoint:
            checkpoint([dict(r) for r in working])

    try:
        for row in working:
            if row.get("text") or row.get("reference_only"):
                out.append(row)
                continue
            try:
                wav, sr = load_wav_mono(row["audio"])
            except (OSError, ValueError, RuntimeError) as exc:
                row["transcribe_error"] = f"decode: {exc}"
                bad += 1
                continue
            try:
                if sr != TARGET_SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
                segs, info = model.transcribe(wav, vad_filter=True)
                text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as exc:
                raise RuntimeError(f"转写失败，已保留原记录与进度：{row['audio']}: {exc}") from exc
            det = getattr(info, "language", "")
            ok = accept or (lang,)
            done += 1
            if not text or (det and det.split("-")[0] not in ok):
                row["transcribe_error"] = f"empty_text_or_language:{det}"
                bad += 1
            else:
                row.update(text=text, transcript_source="whisper-large-v3")
                row.pop("transcribe_error", None)
                out.append(row)
            if progress and done % 100 == 0:
                progress(f"转写 {done}/{todo}（本次排除 {bad} 条，原记录保留）")
            if done % checkpoint_every == 0:
                _flush()
    finally:
        _flush()
    return out, bad


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            tmp = Path(f.name)
            for rec in records:
                f.write(json.dumps({k: v for k, v in rec.items() if not k.startswith("_")},
                                   ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- 解码

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
        if not wav.size or not np.isfinite(wav).all() or np.abs(wav).max() < 1e-8:
            stats["drop_decode"] += 1
            continue
        yield Clip(trim_silence(wav, TARGET_SR), str(row["text"]).strip(),
                   row.get("speaker", "default"), row.get("emotion", ""),
                   row.get("session", ""), dict(row))


# ---------------------------------------------------------------- 主流程

def options_for(source_id: str, **overrides) -> Options:
    """按 registry 的数据源策略生成加工参数（UI 与 CLI 共用，避免两处各写一份）。"""
    from .registry import get_source
    src = get_source(source_id)
    opts = Options(
        ref_audio_ratio=0.5,
        utmos_min=3.5 if src.qc == "full" else None,
        whisper_lang=src.lang if src.qc in ("whisper", "full") else None,
        pseudo_speaker=src.pseudo_speaker,
        accept_langs=src.languages(),
        control_ratio=0.5 if src.expressive else 0.25,
    )
    for k, v in overrides.items():
        if v is not None:
            setattr(opts, k, v)
    if overrides.get("ref_control_ratio") is None:
        opts.ref_control_ratio = min(opts.ref_control_ratio, opts.ref_audio_ratio)
    return opts


def _source_lang(source_id: str) -> str:
    from .registry import get_source
    try:
        return get_source(source_id).lang
    except KeyError:
        return "zh"


def split_records(records: list[dict], opts: Options) -> tuple[list[dict], list[dict]]:
    """按身份/会话/原音频连通分组，固定划分后才配 ref；不截断组来凑 val_max。"""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, r in enumerate(records):
        keys = [("audio", r.get("origin_audio", r["audio"]))]
        if r.get("speaker_verified") is True:
            keys.append(("speaker", r["speaker"]))
        if r.get("session"):
            keys.append(("session", r.get("source_id", ""), r["session"]))
        for key in keys:
            if key in owners:
                parent[root(i)] = root(owners[key])
            owners[key] = i
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[root(i)].append(r)
    # 按稳定 ID 排序和独立种子分组，改变控制前缀比例不会改变验证集。
    groups = sorted(groups.values(), key=lambda rs: min(r.get("origin_audio", r["audio"]) for r in rs))
    random.Random(opts.seed).shuffle(groups)
    if opts.val_ratio <= 0 or len(groups) < 2:
        return records, []
    target = min(opts.val_max, max(1, round(sum(not r.get("reference_only") for r in records) * opts.val_ratio)))
    val, train = [], []
    remaining = sum(any(not r.get("reference_only") for r in g) for g in groups)
    val_targets = 0
    for group in groups:
        n = sum(not r.get("reference_only") for r in group)
        if val_targets < target and n and remaining > 1:
            val.extend(group)
            val_targets += n
        else:
            train.extend(group)
        remaining -= bool(n)
    return train, val


def pair_references(records: list[dict], opts: Options, rng: random.Random) -> list[dict]:
    """仅在本 split 内按已验证身份配对；有同人中/英文候选时优先跨语言。"""
    by_spk = defaultdict(list)
    targets = [r for r in records if not r.get("reference_only")]
    for r in records:
        if r.get("speaker_verified") is True and opts.ref_min_dur <= r["duration"] <= opts.ref_max_dur:
            by_spk[r["speaker"]].append(r)
    candidates = {True: [], False: []}
    for rec in targets:
        if rec.get("speaker_verified") is True and any(
                r["origin_audio"] != rec["origin_audio"] for r in by_spk[rec["speaker"]]):
            candidates[bool(rec.get("control"))].append(rec)
    total = round(len(targets) * opts.ref_audio_ratio)
    n_control = min(total, round(len(targets) * opts.ref_control_ratio))
    selected = rng.sample(candidates[True], min(n_control, len(candidates[True])))
    selected += rng.sample(candidates[False], min(total - len(selected), len(candidates[False])))
    # 缺少可信样本时报告缺口，不用错误身份或未审核标签补足配比。
    # ponytail: 每人候选线性扫描；单人万级语料成为瓶颈时再按语言建索引。
    for rec in selected:
        pool = [r for r in by_spk[rec["speaker"]] if r["origin_audio"] != rec["origin_audio"]]
        cross = [r for r in pool if r.get("lang") in ("zh", "en") and r.get("lang") != rec.get("lang")]
        ref = rng.choice(cross or pool)
        rec.update(ref_audio=ref["audio"], ref_duration=ref["duration"],
                   ref_lang=ref.get("lang", ""), ref_speaker=ref["speaker"],
                   ref_origin_audio=ref["origin_audio"])
    return targets


def dataset_summary(records: list[dict]) -> dict:
    """实际样本曝光统计；声学指标仅作描述，不声称自然度已达标。"""
    from collections import Counter
    exposures = Counter(r.get("origin_audio", r["audio"]) for r in records)
    ref_exposures = Counter(r.get("ref_origin_audio", r["ref_audio"])
                            for r in records if r.get("ref_audio"))
    return {
        "rows": len(records), "seconds": round(sum(r["duration"] for r in records), 4),
        "hours": round(sum(r["duration"] for r in records) / 3600, 4),
        "speakers": len({r["speaker"] for r in records if r.get("speaker_verified") is True}),
        "languages": dict(Counter(r.get("lang", "unknown") for r in records)),
        "emotions": dict(Counter(r.get("emotion") or "unlabeled" for r in records)),
        "with_control": sum(bool(r.get("control")) for r in records),
        "with_ref_audio": sum(bool(r.get("ref_audio")) for r in records),
        "with_ref_control": sum(bool(r.get("ref_audio") and r.get("control")) for r in records),
        "cross_language_refs": sum(bool(r.get("ref_lang") and r["ref_lang"] != r.get("lang")) for r in records),
        "unique_audio": len(exposures), "max_exposure": max(exposures.values(), default=0),
        "unique_ref_audio": len(ref_exposures), "max_ref_exposure": max(ref_exposures.values(), default=0),
    }


def process_dataset(source_id: str, out_name: str | None = None,
                    opts: Options | None = None, max_items: int | None = None,
                    progress=None, manifest_path: str | Path | None = None) -> dict:
    """加工 data/raw/<source_id>/manifest.jsonl → data/processed/<out_name>/

    16k 重采样 → 裁静音 → 时长过滤 → 质检 → 声学描述 →
    身份/会话隔离切分 → 按说话人响度对齐 → 可信控制前缀 → split 内 ref 配对。
    """
    opts = opts or Options()
    for name in (source_id, out_name or source_id):
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("source_id/out_name 必须是单个数据集名称")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items 必须大于 0")
    if not (3 <= opts.min_dur <= opts.max_dur <= 30):
        raise ValueError("时长范围必须满足 3 <= min_dur <= max_dur <= 30")
    if not (3 <= opts.ref_min_dur <= opts.ref_max_dur <= 10):
        raise ValueError("参考时长必须满足 3 <= ref_min_dur <= ref_max_dur <= 10")
    if not (0 <= opts.ref_control_ratio <= opts.ref_audio_ratio <= 0.5):
        raise ValueError("ref_control_ratio <= ref_audio_ratio <= 0.5")
    if not (0 <= opts.control_ratio <= 1 and 0 <= opts.control_zh_ratio <= 1
            and 0 <= opts.val_ratio < 1 and opts.val_max >= 1):
        raise ValueError("控制比例/验证集比例或 val_max 无效")
    if opts.min_snr_db is not None or opts.min_f0_std is not None:
        raise ValueError("不再按能量分位差或 F0 起伏硬筛；请使用真实音质检查与母语试听")
    rng = random.Random(opts.seed)
    src = DATA_RAW / source_id
    manifest = Path(manifest_path).resolve() if manifest_path else src / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"先运行下载：python -m voxft.data.download --source {source_id}")
    out = DATA_PROCESSED / (out_name or source_id)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    # 每次加工写新音频目录，失败也不破坏旧 manifest 正在引用的音频。
    audio_dir = Path(tempfile.mkdtemp(prefix="run_", dir=audio_dir))

    rows = _read_manifest(manifest)
    original_rows = rows
    from .registry import get_source
    from .download import _aishell_text, _clean_text
    try:
        source = get_source(source_id)
    except KeyError:
        source = None
    seen = set()
    normalized = []
    for row in rows:
        if not isinstance(row.get("audio"), str) or not row["audio"].strip():
            raise ValueError("原始 JSONL 每行必须提供 audio 路径")
        audio = Path(row["audio"])
        if not audio.is_absolute():
            audio = manifest.parent / audio
        row = {**row, "audio": str(audio.resolve()), "text": _clean_text(row.get("text"))}
        if row.get("ref_audio"):
            raise ValueError("原始 ref 请作为独立 reference_only 行导入，由 split 内验证身份后配对")
        if row["text"].startswith(("(", "（")):
            raise ValueError("原始 text 请提供裸台词，控制指令放 control_zh/control_en 列")
        if source_id == "aishell3":
            row["text"] = _aishell_text(row["text"])
        if source_id == "thai_ser":
            if not row.get("turn_type"):
                raise ValueError("旧 THAI-SER manifest 缺 turn_type；请在远程重新下载以保留元数据")
            if row["turn_type"] != "impro":
                continue
        if row["audio"] in seen:
            continue
        seen.add(row["audio"])
        normalized.append(row)
    rows = normalized
    truncated = bool(max_items)
    if max_items:
        rows = rows[:max_items]

    if any(not r.get("text") and not r.get("reference_only") for r in rows):
        lang = _source_lang(source_id)
        n_missing = sum(1 for r in rows if not r.get("text"))
        if progress:
            progress(f"{source_id}: {n_missing} 条缺文本，自动 Whisper 转写 + 语种过滤（{lang}）")
        def save_transcripts(updated):
            by_audio = {r["audio"]: r for r in updated}
            saved = []
            for original in original_rows:
                path = Path(original["audio"])
                path = path if path.is_absolute() else manifest.parent / path
                update = by_audio.get(str(path.resolve()), {})
                rec = dict(original)
                for key in ("text", "transcript_source", "transcribe_error"):
                    if key in update:
                        rec[key] = update[key]
                if update.get("text") and "transcribe_error" not in update:
                    rec.pop("transcribe_error", None)
                saved.append(rec)
            _write_jsonl(saved, manifest)
        ckpt = None if truncated else save_transcripts
        rows, n_bad = _transcribe_manifest(rows, lang, opts.accept_langs,
                                           progress, ckpt)
        if progress:
            progress(f"转写完成：保留 {len(rows)} 条，丢弃 {n_bad} 条（语种不符/空）")
        if not rows:
            raise RuntimeError(f"{source_id}: 转写后无可用样本")

    for row in rows:
        for flag in ("reference_only", "emotion_verified", "control_verified"):
            if flag in row and not isinstance(row[flag], bool):
                raise ValueError(f"{flag} 必须为布尔值")
        speaker = _clean_text(row.get("speaker"))
        default_verified = bool(source and source.has_speaker and source.kind != "local" and speaker)
        verified = row.get("speaker_verified", default_verified)
        if not isinstance(verified, bool) or (verified and speaker in ("", "default")):
            raise ValueError("speaker_verified 必须为布尔值；可信身份必须提供真实 speaker ID")
        row.update(speaker_verified=verified, origin_audio=row["audio"],
                   source_id=source_id, lang=row.get("lang") or _source_lang(source_id),
                   speaker=f"{row.get('speaker_namespace') or source_id}:{speaker or 'unknown'}")
        row.setdefault("emotion_verified", source_id == "thai_ser")

    whisper = None
    if opts.whisper_lang:
        if progress:
            progress(f"加载 Whisper 质检模型（{opts.whisper_lang}，首次需下载）...")
        whisper = _whisper_model(opts.whisper_lang, "medium", progress)
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
             "drop_lang": 0, "drop_whisper": 0, "drop_utmos": 0}
    samples = _decoded_clips(rows, stats)
    for i, clip in enumerate(samples):
        if progress and i % 50 == 0:
            progress(f"加工 {source_id}: 已产出 {i} 条样本")
        dur = len(clip.wav) / TARGET_SR
        lo, hi = (opts.ref_min_dur, opts.ref_max_dur) if clip.metadata.get("reference_only") else (opts.min_dur, opts.max_dur)
        if not (lo <= dur <= hi):
            stats["drop_duration"] += 1
            continue
        if whisper is not None and not clip.metadata.get("reference_only"):
            try:
                sim, det = _whisper_similarity(whisper, clip.wav, TARGET_SR, clip.text)
            except Exception as exc:
                raise RuntimeError(f"Whisper 质检运行失败，停止加工：{clip.metadata['audio']}: {exc}") from exc
            ok = opts.accept_langs or (opts.whisper_lang,)
            if det and det.split("-")[0] not in ok:
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
        dst = audio_dir / f"{i:07d}.wav"
        sf.write(dst, clip.wav, TARGET_SR)
        rec = {**clip.metadata, "audio": str(dst), "text": clip.text, "speaker": clip.speaker,
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
            progress("聚类伪说话人（仅供人工审计，不用于 ref 配对）...")
        labels = cluster_pseudo_speakers(embs, opts.pseudo_speaker_threshold)
        for rec, lab in zip(kept, labels):
            rec["pseudo_speaker"] = lab
        n_clusters = len(set(labels))
        stats["pseudo_speakers"] = n_clusters
        if progress:
            progress(f"伪说话人聚类完成：{n_clusters} 簇 / {len(kept)} 条"
                     f"（阈值 {opts.pseudo_speaker_threshold}）")
            if n_clusters > 0.4 * len(kept):
                progress("伪说话人仅供审计；不会降低阈值或直接用于 ref 配对")

    if progress:
        progress(f"按说话人对齐响度到 {opts.target_dbfs} dBFS（保留条间动态）...")
    train, val = split_records(kept, opts)
    for split in (train, val):
        apply_speaker_gain(split, opts.target_dbfs, progress)
        apply_control_prefixes(split, opts, rng)
    train = pair_references(train, opts, rng)
    val = pair_references(val, opts, rng)
    if not train:
        raise ValueError("切分后没有训练目标；请增加独立说话人/会话或检查 reference_only")
    if not val and progress:
        progress("警告：没有可独立留出的验证组，请另备未见说话人评测集")
    rng.shuffle(train)

    _write_jsonl(train, out / "train.jsonl")
    _write_jsonl(val, out / "val.jsonl")
    expressive = [r.get("f0_std_st", 0.0) for r in train]
    stats.update({
        **dataset_summary(train), "train": len(train), "val": len(val),
        "validation": dataset_summary(val),
        "f0_std_median": round(float(np.median(expressive)), 2) if expressive else 0.0,
        "output": str(out),
    })
    if progress:
        progress(f"训练目标 {len(train)} 条；control={stats['with_control']}，"
                 f"ref={stats['with_ref_audio']}，ref+control={stats['with_ref_control']}，"
                 f"跨语言 ref={stats['cross_language_refs']}；不足配比不强凑")
    (out / "stats.json").write_text(
        json.dumps({"source_id": source_id, "options": asdict(opts), **stats},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def mix_manifests(parts: list[tuple[str, float]], out_name: str,
                  seed: int = 42, max_repeat: float = 3.0) -> dict:
    """按有效音频时长采样；每条原始目标音频全局最多 3×，验证集不重复采样。"""
    from collections import Counter
    if not parts or any(not math.isfinite(w) or w <= 0 for _, w in parts):
        raise ValueError("parts 不能为空，权重须为有限正数")
    if len({n for n, _ in parts}) != len(parts) or out_name in {n for n, _ in parts}:
        raise ValueError("输入数据集不能重复，输出不能覆盖输入")
    if not (1 <= max_repeat <= 3 and float(max_repeat).is_integer()):
        raise ValueError("max_repeat 必须为 1、2 或 3")
    for name in [out_name, *(n for n, _ in parts)]:
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("输入/输出必须是单个数据集名称")
    rng = random.Random(seed)
    total_w = sum(w for _, w in parts)
    out = DATA_PROCESSED / out_name
    summary, mixes, details = {}, {}, {}
    exposures = Counter()
    for split in ("train", "val"):
        rows_by_part = []
        for name, _w in parts:
            p = DATA_PROCESSED / name / f"{split}.jsonl"
            if not p.exists():
                raise FileNotFoundError(p)
            rows_by_part.append(_read_manifest(p))
        for rows in rows_by_part:
            for rec in rows:
                if not math.isfinite(float(rec.get("duration", 0))) or float(rec.get("duration", 0)) <= 0:
                    raise ValueError("混合需要有效 duration，请先重新加工旧数据集")
                rec["duration"] = float(rec["duration"])
        seconds = sum(float(r["duration"]) for rows in rows_by_part for r in rows)
        mixed: list[dict] = []
        val_seen = set()
        for (name, w), rows in zip(parts, rows_by_part):
            if not rows:
                continue
            want = seconds * w / total_w
            selected, consumed = [], 0.0
            for _ in range(int(max_repeat) if split == "train" else 1):
                shuffled = rng.sample(rows, len(rows))
                for rec in shuffled:
                    key = rec.get("origin_audio", rec["audio"])
                    if split == "train":
                        if consumed >= want or exposures[key] >= max_repeat:
                            continue
                        exposures[key] += 1
                    else:
                        if key in val_seen:
                            continue
                        val_seen.add(key)
                    selected.append(rec)
                    consumed += float(rec["duration"])
            summary[f"{name}/{split}"] = len(selected)
            if split == "train" and consumed < want:
                summary[f"{name}/{split}_capped_from"] = round(want / (sum(r["duration"] for r in rows) / len(rows)))
            details[f"{name}/{split}"] = {"requested_hours": round(want / 3600, 4),
                                          **dataset_summary(selected)}
            mixed.extend(selected)
        rng.shuffle(mixed)
        mixes[split] = mixed

    def identities(rows):
        keys = set()
        for r in rows:
            for k in ("audio", "ref_audio", "origin_audio", "ref_origin_audio"):
                if r.get(k):
                    keys.add(("audio", str(Path(r[k]).resolve())))
            if r.get("speaker_verified") is True:
                keys.add(("speaker", r["speaker"]))
            if r.get("session"):
                keys.add(("session", r.get("source_id", ""), r["session"]))
        return keys

    if identities(mixes["train"]) & identities(mixes["val"]):
        raise ValueError("训练/验证存在共享音频、ref、说话人或会话；请先按组重新加工")
    if not mixes["train"]:
        raise ValueError("混合后训练集为空")
    for split, mixed in mixes.items():
        _write_jsonl(mixed, out / f"{split}.jsonl")
    actual_seconds = sum(r["duration"] for r in mixes["train"])
    for key, detail in details.items():
        if key.endswith("/train"):
            detail["actual_duration_share"] = round(detail["seconds"] / actual_seconds, 4)
    (out / "mix.json").write_text(
        json.dumps({"parts": parts, "basis": "duration", "max_repeat": max_repeat,
                    "counts": summary, "datasets": details,
                    "train": dataset_summary(mixes["train"]),
                    "val": dataset_summary(mixes["val"])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), **summary}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--source")
    group.add_argument("--mix", nargs="+", metavar="DATASET=WEIGHT")
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest", default=None, help="已审核原始 JSONL；相对音频路径按此文件所在目录解析")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--utmos-min", type=float, default=None)
    ap.add_argument("--whisper-lang", default=None)
    ap.add_argument("--control-ratio", type=float, default=None)
    ap.add_argument("--min-snr-db", type=float, default=None)
    ap.add_argument("--ref-audio-ratio", type=float, default=None)
    ap.add_argument("--ref-control-ratio", type=float, default=None)
    ap.add_argument("--val-ratio", type=float, default=None)
    args = ap.parse_args()
    if args.mix:
        if not args.out:
            ap.error("--mix 需要 --out")
        parts = [(part.rsplit("=", 1)[0], float(part.rsplit("=", 1)[1])) for part in args.mix]
        print(json.dumps(mix_manifests(parts, args.out), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    o = options_for(args.source, utmos_min=args.utmos_min,
                    whisper_lang=args.whisper_lang,
                    control_ratio=args.control_ratio,
                    min_snr_db=args.min_snr_db, ref_audio_ratio=args.ref_audio_ratio,
                    ref_control_ratio=args.ref_control_ratio, val_ratio=args.val_ratio)
    print(json.dumps(process_dataset(args.source, args.out, o, args.max_items,
                                     progress=print, manifest_path=args.manifest),
                     ensure_ascii=False, indent=2))
