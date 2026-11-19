"""timbre_v1 音色评测臂（OmniVoice docs/timbre_consistency_finetune_plan_20261105.md 阶段 0）。

build：从 DIS 的 UAT 回放取生产角色参考与目标语台词，生成冻结 case 集与 plan.json（闸门写死）。
score：读 voxft.eval 报告，算参考相似度（ERes2NetV2 生产刻度 + WavLM，分开报告）、同参考跨句一致性、
音区比、性别翻转、CER，做按参考/按句 cluster bootstrap，并按 plan.json 判定是否进入阶段 1。
自动声纹不能替代人耳；闸门只决定「要不要训练」，不宣布音色变好。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
TAG = "timbre_v1_20261119"
CASES = ROOT / "eval_cases" / f"{TAG}.jsonl"
LEVERS = ROOT / "eval_cases" / f"{TAG}_levers.jsonl"
PLAN = ROOT / "eval_cases" / f"{TAG}.plan.json"
REF_DIR = ROOT / "work" / TAG / "refs"

LANGS = ("th", "tl", "id")
BUCKETS = ("short", "medium", "long")
SEEDS_PER_REF = 5
CFG = 1.8


def steps_for(duration: float) -> int:
    """TODO.md 冻结口径：步数按参考时长 <2s 28 / 2–5s 24 / ≥5s 22。"""
    return 28 if duration < 2 else 24 if duration < 5 else 22
# 线上 UAT 记录里该情绪×强度最常见的非空 control_instruction 原文
CONTROLS = {
    ("angry", "medium"): "angry and emotionally tense, with clear, natural exclamatory emphasis",
    ("angry", "high"): "openly angry, as a full-voiced outburst",
    ("fearful", "medium"): "fearful and tense, with urgent, brisk delivery",
    ("fearful", "high"): "fearful and tense, with hard-driving urgency",
    ("sad", "medium"): "sad and emotionally weighted, with controlled, restrained delivery",
    ("sad", "high"): "deeply grieved, with a natural, controlled tremble",
}
# (名字, DIS work 下的相对路径, 来源核验方式)。来源核验：字节 SHA 须出现在原请求记录里，或等于试听包 key.json 登记值。
REFS = (
    ("tl123_c_3b9447", "uat_replay/r20/refs/c_3b9447.wav", ("records", "uat_replay/r20/workspace_*.json")),
    ("tl123_c_5c08e5", "uat_replay/r20/refs/c_5c08e5.wav", ("records", "uat_replay/r20/workspace_*.json")),
    ("tl123_c_d4ae78", "uat_replay/r20/refs/c_d4ae78.wav", ("records", "uat_replay/r20/workspace_*.json")),
    ("tl129_c_0ebc41", "uat_replay/r22/refs/c_0ebc41.wav", ("records", "uat_replay/r22/workspace/ep_*.json")),
    ("tl129_c_e9961f", "uat_replay/r22/refs/c_e9961f.wav", ("records", "uat_replay/r22/workspace/ep_*.json")),
    ("tl123_batang", "listening/minor_char_ab_20261106_key/batang_shared_ref.wav",
     ("key", "listening/minor_char_ab_20261106_key/key.json")),
    ("tl123_inay", "listening/minor_char_ab_20261106_key/inay_shared_ref.wav",
     ("key", "listening/minor_char_ab_20261106_key/key.json")),
    ("tl123_kaibigan", "listening/minor_char_ab_20261106_key/kaibigan_shared_ref.wav",
     ("key", "listening/minor_char_ab_20261106_key/key.json")),
)
CROPS = (3.0, 5.0)
GATE = {"min_drop": 0.02, "ci": 0.95, "bootstrap": 2000, "gender_conf": 0.6}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 2**31


def bucket(text: str, lang: str) -> str | None:
    # th 无词间空格，按非空白字符数分桶；其余按词数。两档之间留空隙，避免边界句来回漂。
    n = len(re.sub(r"\s", "", text)) if lang == "th" else len(text.split())
    short, medium, long_ = ((2, 10), (18, 40), (50, 90)) if lang == "th" else ((1, 4), (6, 12), (16, 30))
    for name, (lo, hi) in zip(BUCKETS, (short, medium, long_)):
        if lo <= n <= hi:
            return name
    return None


def text_pools(dis_work: Path) -> dict:
    pools = defaultdict(set)
    files = [*dis_work.glob("uat_replay/*/workspace/ep_*.json"), *dis_work.glob("uat_replay/*/workspace_*.json")]
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8")).get("data") or {}
        lang = (data.get("project") or {}).get("target_language")
        if lang not in LANGS:
            continue
        for cue in data.get("cues") or []:
            text = " ".join((cue.get("target_text") or "").split())
            # 数字走 TN、括号/星号是标签残留，都会把音色指标和文本归一化问题混在一起
            if text and not re.search(r"[\d\[\]()（）*]", text) and (b := bucket(text, lang)):
                pools[(lang, b)].add(text)
    # 按文本哈希排序：确定、与采集顺序无关
    return {k: sorted(v, key=lambda t: hashlib.sha256(t.encode()).hexdigest()) for k, v in pools.items()}


def verify_ref(dis_work: Path, path: Path, check) -> str:
    digest = sha256(path)
    kind, where = check
    if kind == "records":
        if not any(digest in p.read_text(encoding="utf-8") for p in dis_work.glob(where)):
            raise ValueError(f"{path} 的 SHA 不在原请求记录 {where} 中")
    elif digest not in Path(dis_work, where).read_text(encoding="utf-8"):
        raise ValueError(f"{path} 的 SHA 与 {where} 登记值不符")
    return digest


def center_crop(src: Path, dst: Path, seconds: float) -> None:
    wav, sr = sf.read(src, dtype="int16")
    n = int(seconds * sr)
    start = max(0, (len(wav) - n) // 2)
    sf.write(dst, wav[start:start + n], sr, subtype="PCM_16")


def build(dis_root: Path) -> dict:
    dis_work = dis_root / "work"
    REF_DIR.mkdir(parents=True, exist_ok=True)
    pools = text_pools(dis_work)
    need = len(REFS)
    for key in [(lang, b) for lang in LANGS for b in BUCKETS]:
        if len(pools.get(key, ())) < need:
            raise ValueError(f"{key} 台词不足 {need} 句")
    rel = lambda p: str(Path("..") / p.relative_to(ROOT))  # noqa: E731  case 内相对路径按 case 文件目录解析
    refs, main, levers = {}, [], []
    for i, (name, src_rel, check) in enumerate(REFS):
        src = dis_work / src_rel
        digest = verify_ref(dis_work, src, check)
        dst = REF_DIR / f"{name}.wav"
        shutil.copyfile(src, dst)
        info = sf.info(dst)
        crops = {}
        for sec in CROPS:
            if info.duration >= sec + 0.4:
                crop = REF_DIR / f"{name}_crop{sec:g}s.wav"
                center_crop(dst, crop, sec)
                crops[f"{sec:g}s"] = {"path": rel(crop), "sha256": sha256(crop), "steps": steps_for(sec)}
        seeds = [stable_seed(TAG, digest, k) for k in range(SEEDS_PER_REF)]
        refs[name] = {"source": src_rel, "sha256": digest, "provenance": check[0], "sr": info.samplerate,
                      "duration": round(info.duration, 3), "seeds": seeds, "crops": crops}
        texts = {(lang, b): pools[(lang, b)][i] for lang in LANGS for b in BUCKETS}
        base = {"ref": name, "ref_audio": rel(dst), "cfg_value": CFG, "inference_timesteps": steps_for(info.duration)}
        conds = [((lang, b), "", "", "") for lang in LANGS for b in BUCKETS]
        conds += [((LANGS[j % 3], "medium"), emo, level, text)
                  for j, ((emo, level), text) in enumerate(CONTROLS.items())]
        for (lang, b), emo, level, control in conds:
            control_key = f"{emo}-{level}" if emo else "plain"
            case = {**base, "case_id": f"{name}|{lang}|{b}|{control_key}", "lang": lang, "bucket": b,
                    "text": texts[(lang, b)], "control": control, "emotion": emo, "intensity": level or "none",
                    "condition": "main"}
            main += [{**case, "seed": s} for s in seeds]
            if emo:
                continue
            levers += [{**case, "case_id": case["case_id"] + "|cfg1.6", "cfg_value": 1.6,
                        "condition": "cfg1.6", "seed": s} for s in seeds]
            for label, crop in crops.items():
                levers += [{**case, "case_id": case["case_id"] + f"|ref{label}", "ref_audio": crop["path"],
                            "ref_audio_original": rel(dst), "inference_timesteps": crop["steps"],
                            "condition": f"ref{label}", "seed": s}
                           for s in seeds]
    for path, rows in ((CASES, main), (LEVERS, levers)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    plan = {
        "tag": TAG,
        "source_plan": "OmniVoice docs/timbre_consistency_finetune_plan_20261105.md 阶段 0",
        "arms": {
            "base": {"target": "base"},
            "r50": {"target": "checkpoints/r50_3ep_20261103/latest", "production": True,
                    "lora_weights_sha256": sha256(ROOT / "checkpoints/r50_3ep_20261103/latest/lora_weights.safetensors")},
        },
        "generation": {"cfg_value": CFG, "inference_timesteps": "按参考时长 <2s 28 / 2–5s 24 / ≥5s 22（TODO.md 冻结口径）", "retry_badcase": False,
                       "seeds_per_ref": SEEDS_PER_REF, "seed_rule": "stable_seed(tag, ref_sha256, k)，同参考固定，模拟生产按角色稳定 seed"},
        "cases": {"main": {"path": str(CASES.relative_to(ROOT)), "rows": len(main), "sha256": sha256(CASES),
                           "arms": ["base", "r50"]},
                  "levers": {"path": str(LEVERS.relative_to(ROOT)), "rows": len(levers), "sha256": sha256(LEVERS),
                             "arms": ["r50"]}},
        "refs": refs,
        "controls": {f"{e}-{lv}": t for (e, lv), t in CONTROLS.items()},
        "metrics": {
            "sim_eres": "输出↔参考 ERes2NetV2（DIS 本地 ONNX，生产刻度）",
            "sim_wavlm": "输出↔参考 WavLM-SV（报告 speaker_sim）；与 sim_eres 分开报告，不混比",
            "consistency": "同臂同参考同 seed、plain 九句两两余弦均值（两种刻度各一）",
            "register": "输出/参考 F0 p25、p50、p75 之比（pyin 70–450Hz）",
            "gender_flip": f"双方性别置信度均 ≥{GATE['gender_conf']} 且相反（OmniVoice _segment_profile 移植）",
            "intelligibility": "CER 均值、严重=over_read 或 suspected_truncation",
        },
        "statistics": {"bootstrap": GATE["bootstrap"], "ci": GATE["ci"],
                       "clusters": "差值先按簇取均值再等权平均；参考簇、台词簇各做一次，两者 CI 上界都 <0 才算显著"},
        "gate": {
            "min_drop": GATE["min_drop"],
            "1_regression": "r50−base 在 sim 或 consistency 上：两种声纹刻度均 Δ≤−min_drop 且显著",
            "2_slice_gap": ("r50 切片缺口（high 情绪 vs 同句 plain；short vs 同参考同语种 medium）两种刻度均 ≤−min_drop 且显著，"
                            "且零训练杠杆（high→medium 强度上限；cfg1.6）都没能把缺口收回一半"),
            "decision": "任一条成立才进入阶段 1；否则结案，剩余问题回 DIS 管线",
        },
        "deviations": [
            "参考只有 8 条（计划约 12）：UAT 101/103 只有 2 个角色有参考，签名 URL 已过期；改用本地来源可核验的 UAT 123/129 角色参考与任务 123 配角共用参考。Kasama 疑似多人同名剔除；泰语反馈参考源语言为 en 剔除。",
            "去掉 r46u10 对照（用户 2026-11-19 决定，权重本机无副本）：闸门 2 只看 r50 自身缺口与零训练杠杆，不再用模型间差异作方差代理。",
            "FLEURS 对照参考未纳入：文件只在旧 GPU 机上，本机无法冻结 SHA。",
            "台词每语种 24 句（每档 8 句 = 参考数），不是 30 句；全部取自 UAT 目标语台词，过滤数字/括号。",
            "best_of=2 用已生成 seed 两两配对事后择优（按 ERes2NetV2 选、用 WavLM 评，避免拿选择指标自证），不额外生成。",
            "参考音时长杠杆用中心裁剪 3s/5s，与原长参考对比；相似度仍对原长参考计算。",
            "性别失衡：8 条里 7 条女声、1 条男声（c_3b9447，约 150Hz），男声那条在生产估计器下判 unknown，所以性别翻转实际只测得到女→男。",
        ],
    }
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


# ---------------- score ----------------

class Eres2Net:
    """DIS local_audio_models 的 ERes2NetV2 ONNX 前端逐行移植；重采样用 resample_poly，与 DIS 滤波器有微小差异。"""

    def __init__(self, directory: Path):
        import onnxruntime as ort
        with np.load(directory / "frontend.npz", allow_pickle=False) as arrays:
            self.filters, self.window = arrays["filters"].copy(), arrays["window"].copy()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 2
        self.session = ort.InferenceSession(str(directory / "speaker.onnx"), opts, providers=["CPUExecutionProvider"])

    def features(self, samples: np.ndarray) -> np.ndarray:
        from scipy.fft import rfft
        frames = np.lib.stride_tricks.sliding_window_view(samples, 400)[::160].copy()
        frames -= frames.mean(axis=1, keepdims=True)
        frames -= np.float32(0.97) * np.pad(frames, ((0, 0), (1, 0)), mode="edge")[:, :-1]
        power = np.abs(rfft(frames * self.window, n=512, axis=1)) ** 2
        feats = np.log(np.maximum(power @ self.filters.T, np.finfo(np.float32).eps))
        feats -= feats.mean(axis=0, keepdims=True)
        return feats[None].astype(np.float32)

    def run(self, feats: np.ndarray) -> np.ndarray:
        vector = np.asarray(self.session.run(None, {"features": feats})[0], dtype=np.float64).reshape(-1)
        return vector / np.linalg.norm(vector)

    def embed(self, wav16: np.ndarray) -> np.ndarray | None:
        if wav16.size < 8000:
            return None
        limit = 30 * 16000
        start = max(0, (wav16.size - limit) // 2)
        return self.run(self.features(wav16[start:start + limit].astype(np.float32)))


def to16k(path: str) -> np.ndarray:
    from scipy.signal import resample_poly
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    g = math.gcd(sr, 16000)
    return wav if sr == 16000 else resample_poly(wav, 16000 // g, sr // g).astype(np.float32)


def _frame_f0(frame: np.ndarray, sr: int):
    """OmniVoice api._estimate_frame_f0：自相关取峰，取容差内最短 lag 防亚谐波。"""
    x = frame.astype(np.float64) - frame.mean()
    n, energy = x.size, float(x @ x)
    lo, hi = max(1, int(sr / 450.0)), min(n - 2, int(sr / 70.0))
    if energy <= 0 or hi <= lo:
        return None
    lags = np.arange(lo, hi + 1)
    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    left, right = csum[n - lags], energy - csum[lags]
    corr = np.full(lags.size, -1.0)
    for i in np.nonzero((left > 0) & (right > 0))[0]:
        corr[i] = float(x[: n - lags[i]] @ x[lags[i]:]) / math.sqrt(left[i] * right[i])
    best = float(corr.max())
    if best < 0.45:
        return None
    i = int(np.nonzero(corr >= best * 0.95)[0][0])
    return sr / int(lags[i]), float(corr[i])


def omnivoice_gender(wav16: np.ndarray) -> tuple[str, float]:
    """OmniVoice api._segment_profile 全段口径（降到 8kHz、int16 刻度、≤7 帧 80ms），生产 0.6 门限按它定。"""
    y = (np.clip(wav16[::2], -1, 1) * 32767.0).astype(np.int16).astype(np.float64)
    sr, size = 8000, int(0.080 * 8000)
    if y.size < size:
        return "unknown", 0.0
    count = min(7, max(1, int((y.size - size) / size) + 1))
    offsets = [(y.size - size) // 2] if count == 1 else [int(round((y.size - size) * i / (count - 1))) for i in range(count)]
    rms = [math.sqrt(float(np.mean(y[o:o + size] ** 2))) for o in offsets]
    floor = max(120.0, max(rms) * 0.20)
    est = [e for o, r in zip(offsets, rms) if r >= floor and (e := _frame_f0(y[o:o + size], sr))]
    ratio = len(est) / len(offsets)
    if not est or ratio < 0.35:
        return "unknown", 0.0
    f0, strength = float(np.median([e[0] for e in est])), float(np.median([e[1] for e in est]))
    if 155.0 < f0 < 190.0:
        return "unknown", 0.0
    male = f0 <= 155.0
    distance = min(1.0, max(0.0, (165.0 - f0) / 55.0 if male else (f0 - 180.0) / 80.0))
    conf = max(0.0, min(1.0, strength * 0.70 + ratio * 0.20 + distance * 0.10))
    return ("male" if male else "female") if conf >= 0.55 else "unknown", round(conf, 3)


def f0_profile(wav16: np.ndarray) -> dict:
    """音区用 pyin 全帧分位数；性别用 OmniVoice 生产估计器。"""
    import librosa
    f0, voiced, _ = librosa.pyin(wav16, fmin=70, fmax=450, sr=16000, frame_length=1024)
    voiced_f0 = f0[voiced & np.isfinite(f0)]
    gender, conf = omnivoice_gender(wav16)
    out = {"voiced_ratio": round(float(voiced.mean()), 3) if voiced.size else 0.0,
           "gender": gender, "gender_conf": conf}
    if voiced_f0.size < 5:
        return {**out, "p25": None, "p50": None, "p75": None}
    return {**out, **{f"p{q}": round(float(np.percentile(voiced_f0, q)), 1) for q in (25, 50, 75)}}


def cluster_ci(diffs: dict, n: int = GATE["bootstrap"], ci: float = GATE["ci"], seed: int = 0) -> dict:
    """diffs: 簇 -> 差值列表。簇内先平均、簇间等权；按簇有放回重采样。"""
    means = np.array([np.mean(v) for v in diffs.values() if v])
    if means.size == 0:
        return {"point": None, "lo": None, "hi": None, "clusters": 0}
    rng = np.random.default_rng(seed)
    boots = means[rng.integers(0, means.size, (n, means.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"point": round(float(means.mean()), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "clusters": int(means.size)}


def significant_drop(*cis: dict, min_drop: float = GATE["min_drop"]) -> bool:
    return all(c["point"] is not None and c["point"] <= -min_drop and c["hi"] < 0 for c in cis)


def paired(a: list[dict], b: list[dict], metric: str, key, cluster) -> dict:
    """a−b 按 key 配对，按 cluster(item) 分簇。"""
    index = {key(x): x for x in b}
    diffs = defaultdict(list)
    for x in a:
        y = index.get(key(x))
        if y is not None and x.get(metric) is not None and y.get(metric) is not None:
            diffs[cluster(x)].append(x[metric] - y[metric])
    return diffs


def item_key(x):
    return (x["case_id"], x["seed"])


def consistency(items: list[dict], scale: str) -> list[dict]:
    groups = defaultdict(list)
    for x in items:
        if x["control"] == "" and x.get(f"emb_{scale}") is not None:
            groups[(x["ref"], x["seed"])].append(np.asarray(x[f"emb_{scale}"]))
    return [{"ref": r, "seed": s, "case_id": f"{r}|consistency", f"cons_{scale}":
             float(np.mean([a @ b for a, b in combinations(v, 2)]))}
            for (r, s), v in groups.items() if len(v) >= 2]


def _f0_worker(path: str) -> dict:
    return f0_profile(to16k(path))


def _limit_threads() -> None:
    import os
    import torch
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _wavlm_embedder():
    """与 voxft.qc.audio.speaker_embedding 同模型同预处理，只是放到 GPU（有的话）上跑。"""
    from voxft.qc.audio import _spk_model
    proc, model, torch, _ = _spk_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    def embed(wav16):
        if wav16.size < 8000:
            return None
        inputs = proc(wav16[:16000 * 30], sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**inputs).embeddings[0].float().cpu().numpy().astype(np.float64)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 1e-12 else None
    return embed


def annotate(items: list[dict], eres: Eres2Net, cache: dict, workers: int = 6) -> None:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    paths = sorted({p for x in items for p in (x["wav"], x.get("ref_audio_original") or x["ref_audio"])} - cache.keys())
    # cgroup 只给 ~7.7 核：pyin 进程池各 1 线程，主进程做嵌入，避免 torch/onnx 按 64 逻辑核超订
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_limit_threads) as pool:
        f0s = pool.map(_f0_worker, paths, chunksize=8)
        wavlm = _wavlm_embedder()
        embs = {}
        for i, path in enumerate(paths):
            wav = to16k(path)
            embs[path] = (eres.embed(wav), wavlm(wav))
            if i % 200 == 0:
                print(f"embed {i}/{len(paths)}", flush=True)
        for i, (path, f0) in enumerate(zip(paths, f0s)):
            cache[path] = {"eres": embs[path][0], "wavlm": embs[path][1], "f0": f0}
            if i % 200 == 0:
                print(f"f0 {i}/{len(paths)}", flush=True)

    def profile(path):
        return cache[path]

    for x in items:
        out = profile(x["wav"])
        ref = profile(x.get("ref_audio_original") or x["ref_audio"])
        x["emb_eres"], x["emb_wavlm"] = out["eres"], out["wavlm"]
        x["sim_eres"] = float(out["eres"] @ ref["eres"]) if out["eres"] is not None else None
        x["sim_wavlm"] = float(out["wavlm"] @ ref["wavlm"]) if out["wavlm"] is not None else None
        for q in ("p25", "p50", "p75"):
            o, r = out["f0"][q], ref["f0"][q]
            x[f"reg_{q}"] = o / r if o and r else None
        go, gr = out["f0"], ref["f0"]
        both = min(go["gender_conf"], gr["gender_conf"]) >= GATE["gender_conf"]
        x["gender_flip"] = (both and {go["gender"], gr["gender"]} == {"male", "female"}) if both else None
        x["severe"] = bool(x.get("over_read") or x.get("suspected_truncation"))


def arm_summary(items: list[dict]) -> dict:
    def mean(field, rows):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    slices = {"all": items, "plain": [x for x in items if x["control"] == ""],
              **{b: [x for x in items if x["bucket"] == b and x["control"] == ""] for b in BUCKETS},
              **{f"intensity_{lv}": [x for x in items if x["intensity"] == lv] for lv in ("medium", "high")},
              **{f"lang_{lang}": [x for x in items if x["lang"] == lang] for lang in LANGS}}
    out = {}
    for name, rows in slices.items():
        flips = [r["gender_flip"] for r in rows if r.get("gender_flip") is not None]
        out[name] = {"n": len(rows), **{f: mean(f, rows) for f in
                     ("sim_eres", "sim_wavlm", "reg_p25", "reg_p50", "reg_p75", "cer")},
                     "severe_rate": mean("severe", rows),
                     "gender_flip_rate": round(sum(flips) / len(flips), 4) if flips else None,
                     "gender_flip_n": len(flips)}
    for scale in ("eres", "wavlm"):
        cons = consistency(items, scale)
        out["plain"][f"consistency_{scale}"] = mean(f"cons_{scale}", cons)
    return out


def regression_gate(r50, base) -> dict:
    res = {}
    for scale in ("eres", "wavlm"):
        m = f"sim_{scale}"
        res[m] = {"by_ref": cluster_ci(paired(r50, base, m, item_key, lambda x: x["ref"])),
                  "by_text": cluster_ci(paired(r50, base, m, item_key, lambda x: x["text"]))}
        c = f"cons_{scale}"
        res[f"consistency_{scale}"] = {"by_ref": cluster_ci(paired(
            consistency(r50, scale), consistency(base, scale), c, item_key, lambda x: x["ref"]))}
    sim_fires = all(significant_drop(*res[f"sim_{s}"].values()) for s in ("eres", "wavlm"))
    cons_fires = all(significant_drop(*res[f"consistency_{s}"].values()) for s in ("eres", "wavlm"))
    return {"metrics": res, "fires": sim_fires or cons_fires}


def slice_gap(items: list[dict], slice_name: str, metric: str) -> dict:
    """同臂内：high 情绪 vs 同句同 seed plain；short vs 同参考同语种同 seed medium。按参考成簇。"""
    diffs = defaultdict(list)
    if slice_name in ("high", "medium"):
        plain = {(x["ref"], x["text"], x["seed"]): x for x in items if x["control"] == ""}
        for x in items:
            y = plain.get((x["ref"], x["text"], x["seed"]))
            if x["intensity"] == slice_name and y and x.get(metric) is not None and y.get(metric) is not None:
                diffs[x["ref"]].append(x[metric] - y[metric])
    else:
        medium = {(x["ref"], x["lang"], x["seed"]): x for x in items
                  if x["control"] == "" and x["bucket"] == "medium"}
        for x in items:
            y = medium.get((x["ref"], x["lang"], x["seed"]))
            if (x["control"] == "" and x["bucket"] == "short" and y
                    and x.get(metric) is not None and y.get(metric) is not None):
                diffs[x["ref"]].append(x[metric] - y[metric])
    return cluster_ci(diffs)


def gap_gate(r50, cfg16) -> dict:
    out = {}
    for name in ("high", "short"):
        gaps = {s: slice_gap(r50, name, f"sim_{s}") for s in ("eres", "wavlm")}
        stable = significant_drop(*gaps.values())
        levers = {}
        if name == "high":
            levers["cap_medium"] = slice_gap(r50, "medium", "sim_eres")
        if cfg16:
            levers["cfg1.6"] = slice_gap(cfg16, name, "sim_eres") if name == "short" else None
        closed = any(v and v["point"] is not None and gaps["eres"]["point"] is not None
                     and v["point"] >= gaps["eres"]["point"] / 2 for v in levers.values())
        out[name] = {"gap": gaps, "levers": levers, "fires": bool(stable and not closed)}
    return out


def best_of_two(items: list[dict]) -> dict:
    """seed 两两配对，按 sim_eres 择优、用 sim_wavlm 评；对照取同对 WavLM 均值。"""
    groups = defaultdict(list)
    for x in items:
        groups[x["case_id"]].append(x)
    picked, single = [], []
    for rows in groups.values():
        rows = sorted(rows, key=lambda r: r["seed"])
        for a, b in zip(rows[0::2], rows[1::2]):
            if None in (a.get("sim_eres"), b.get("sim_eres"), a.get("sim_wavlm"), b.get("sim_wavlm")):
                continue
            picked.append(max(a, b, key=lambda r: r["sim_eres"])["sim_wavlm"])
            single.append((a["sim_wavlm"] + b["sim_wavlm"]) / 2)
    return {"pairs": len(picked), "wavlm_selected": round(float(np.mean(picked)), 4) if picked else None,
            "wavlm_single": round(float(np.mean(single)), 4) if single else None}


def lever_report(r50: list[dict], levers: list[dict]) -> dict:
    main = {item_key(x): x for x in r50}
    out = {}
    for cond in sorted({x["condition"] for x in levers}):
        rows = [x for x in levers if x["condition"] == cond]
        base_key = lambda x: (x["case_id"].rsplit("|", 1)[0], x["seed"])  # noqa: E731
        res = {}
        for m in ("sim_eres", "sim_wavlm", "cer"):
            diffs = defaultdict(list)
            for x in rows:
                y = main.get(base_key(x))
                if y and x.get(m) is not None and y.get(m) is not None:
                    diffs[x["ref"]].append(x[m] - y[m])
            res[m] = cluster_ci(diffs)
        out[cond] = res
    return out


def score(reports: dict, dis_model_dir: Path, out_path: Path) -> dict:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    eres = Eres2Net(dis_model_dir)
    import torch
    torch.set_num_threads(2)
    cache, arms = {}, {}
    for arm, path in reports.items():
        # 分片报告用逗号连起来传入，同一臂直接拼接 items
        items = [x for p in path.split(",") for x in json.loads(Path(p).read_text(encoding="utf-8"))["items"]]
        annotate(items, eres, cache)
        arms[arm] = items
    result = {"tag": TAG, "plan_sha256": sha256(PLAN), "reports": reports,
              "summary": {arm: arm_summary(items) for arm, items in arms.items()}}
    if {"r50", "base"} <= arms.keys():
        result["gate_1"] = regression_gate(arms["r50"], arms["base"])
    if {"r50", "r50_levers"} <= arms.keys():
        result["gate_2"] = gap_gate(arms["r50"], [x for x in arms["r50_levers"] if x["condition"] == "cfg1.6"])
    if "r50" in arms:
        result["best_of_2"] = best_of_two(arms["r50"])
    if "r50_levers" in arms:
        result["levers"] = lever_report(arms["r50"], arms["r50_levers"])
    complete = {"gate_1", "gate_2"} <= result.keys()
    fires = complete and (result["gate_1"]["fires"] or any(g["fires"] for g in result["gate_2"].values()))
    result["decision"] = ("incomplete" if not complete else "enter_stage_1" if fires else "close_no_training")
    result["note"] = plan["gate"]["decision"] + "；自动声纹不能替代人耳。"
    for items in arms.values():
        for x in items:
            x.pop("emb_eres", None), x.pop("emb_wavlm", None)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dis-root", type=Path, default=ROOT.parent / "dubbing_intelligence_service")
    s = sub.add_parser("score")
    s.add_argument("--report", action="append", required=True, metavar="ARM=PATH",
                   help="arm 取 base / r50 / r50_levers；分片报告用逗号连接")
    s.add_argument("--eres2net-dir", type=Path,
                   default=ROOT.parent / "dubbing_intelligence_service" / "models" / "eres2netv2")
    s.add_argument("--out", type=Path, default=ROOT / "work" / TAG / "score.json")
    args = ap.parse_args()
    if args.cmd == "build":
        plan = build(args.dis_root)
        print(json.dumps({k: plan["cases"][k]["rows"] for k in plan["cases"]}), PLAN)
        return
    reports = dict(r.split("=", 1) for r in args.report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = score(reports, args.eres2net_dir, args.out)
    print(result["decision"], args.out)


if __name__ == "__main__":
    main()
