from __future__ import annotations

import re
import sys
import time
from uuid import uuid4
from pathlib import Path

import torch

from .paths import CHECKPOINT_DIR, VOXCPM_REPO, env

torch.set_float32_matmul_precision("high")  # 启用 TF32，消警告并提速

_MODEL = None
_MODEL_KEY = None

# 试听示例文本（覆盖目标语言）
SAMPLE_TEXTS = {
    "泰语": "สวัสดีค่ะ ยินดีต้อนรับสู่ระบบสังเคราะห์เสียง",
    "Tagalog": "Magandang araw! Maligayang pagdating sa aming sistema.",
    "中文": "你好，欢迎使用语音合成系统。",
}


def _import_voxcpm():
    src = VOXCPM_REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from voxcpm import VoxCPM  # noqa: PLC0415
    return VoxCPM


def _resolve_base(base_path: str | None) -> str:
    return base_path or env("VOXCPM_BASE_PATH") or "openbmb/VoxCPM2"


def get_model(base_path: str | None = None, lora_dir: str | None = None):
    """加载基座（可带 LoRA）；相同配置复用已加载模型。"""
    global _MODEL, _MODEL_KEY
    base = _resolve_base(base_path)
    # latest/ 会被训练覆盖；路径相同也要在权重更新后重载。
    fingerprint = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                        for p in sorted(Path(lora_dir).glob("lora_*")) if p.is_file()) if lora_dir else ()
    key = (base, lora_dir or "", fingerprint)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    VoxCPM = _import_voxcpm()
    kwargs = {"load_denoiser": False}  # 去噪器依赖 modelscope，试听场景不需要
    if lora_dir:
        from voxcpm.model.voxcpm2 import LoRAConfig
        from .lora.merge import read_lora_config
        kwargs["lora_config"] = LoRAConfig(**read_lora_config(lora_dir))
    # 先释放旧模型，避免 A/B 切换时两个完整基座同时占用显存。
    _MODEL = None
    _MODEL_KEY = None
    model = VoxCPM.from_pretrained(base, **kwargs)
    if lora_dir:
        loaded, skipped = model.load_lora(lora_dir)
        expected = {name.replace("._orig_mod.", ".") for name, _ in
                    model.tts_model.named_parameters() if "lora_" in name}
        missing = expected - set(loaded)
        if not loaded or skipped or missing:
            raise RuntimeError(f"LoRA 加载不完整：loaded={len(loaded)}, skipped={skipped}, missing={sorted(missing)}")
    _MODEL = model
    _MODEL_KEY = key
    return _MODEL


def switch_lora(lora_dir: str | None) -> str:
    """按 checkpoint 配置切换；加载失败直接中止，禁止伪装成 LoRA 输出。"""
    if _MODEL is None:
        raise RuntimeError("模型尚未加载")
    get_model(_MODEL_KEY[0] if _MODEL_KEY else None, lora_dir)
    return f"已加载 {lora_dir}" if lora_dir else "纯基座"


def clean_control(control: str | None) -> str:
    """清掉控制指令里的括号，否则会破坏 "(控制指令)正文" 这个前缀格式。"""
    return re.sub(r"[()（）]", "", (control or "")).strip()


def _gen_kwargs(text: str, ref_audio: str | None, ref_text: str | None,
                cfg_value: float, inference_timesteps: int,
                seed: int | None, control: str | None = None) -> dict:
    control = clean_control(control)
    if control:
        # VoxCPM2 的情绪/语气控制没有独立条件通道，就是文本前缀（官方 cli.build_final_text）。
        text = f"({control}){text}"
    kwargs = {
        "text": text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "retry_badcase": True,          # 防失控兜底（官方建议）
        "retry_badcase_max_times": 3,
        "retry_badcase_ratio_threshold": 6.0,
    }
    if ref_audio:
        kwargs["reference_wav_path"] = ref_audio
        # 带控制前缀时只能走 reference-only：combined 模式会拼成
        # prompt_text + "(控制)正文"，前缀跑到句中就失效了（官方 app 同样规避）
        if ref_text and not control:
            kwargs["prompt_wav_path"] = ref_audio
            kwargs["prompt_text"] = ref_text
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


def _run(model, kwargs: dict) -> tuple[str, float]:
    t0 = time.time()
    wav = model.generate(**kwargs)
    out = CHECKPOINT_DIR / "auditions" / f"audition_{uuid4().hex}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sr = getattr(getattr(model, "tts_model", None), "sample_rate", None) or 48000
    sf.write(out, wav, sr)
    return str(out), round(time.time() - t0, 1)


def synthesize(text: str, base_path: str | None = None,
               lora_dir: str | None = None,
               ref_audio: str | None = None, ref_text: str | None = None,
               cfg_value: float = 2.0, inference_timesteps: int = 20,
               seed: int | None = None, control: str | None = None
               ) -> tuple[str, float]:
    """生成语音，返回 (wav 路径, 耗时秒)。control 为情绪/语气 prompt（中英文）。"""
    model = get_model(base_path, lora_dir)
    return _run(model, _gen_kwargs(text, ref_audio, ref_text,
                                   cfg_value, inference_timesteps, seed, control))


def synthesize_ab(text: str, base_path: str | None, lora_dir: str,
                  ref_audio: str | None = None, ref_text: str | None = None,
                  cfg_value: float = 2.0, inference_timesteps: int = 20,
                  seed: int = 42, control: str | None = None):
    """A/B 对比：带 LoRA 结构的同一模型禁用/启用适配器，参数与权重先严格核验。

    返回 ((基座wav, 秒), (LoRA wav, 秒), 切换状态)。固定 seed 保证可比。
    验收微调是否"更听指令"就看这里：同一 control 前缀，基座 vs LoRA。
    """
    if not lora_dir:
        raise ValueError("A/B 需要选择 LoRA checkpoint")
    model = get_model(base_path, lora_dir)
    kwargs = _gen_kwargs(text, ref_audio, ref_text,
                         cfg_value, inference_timesteps, seed, control)
    kwargs["retry_badcase"] = False  # A/B 固定种子，不自动换种子掩盖坏例
    model.set_lora_enabled(False)
    try:
        r_base = _run(model, kwargs)
    finally:
        model.set_lora_enabled(True)
    r_lora = _run(model, kwargs)
    return r_base, r_lora, f"已核验 LoRA：{lora_dir}；基座禁用适配器，LoRA 启用适配器"


def list_lora_dirs() -> list[str]:
    """列出 checkpoints 下所有 LoRA checkpoint 目录（按时间从新到旧）。"""
    from .lora.merge import is_lora_dir
    dirs: list[Path] = []
    if CHECKPOINT_DIR.exists():
        for d in CHECKPOINT_DIR.rglob("step_*"):
            if is_lora_dir(d):
                dirs.append(d)
        for d in CHECKPOINT_DIR.iterdir():
            if d.is_dir() and is_lora_dir(d / "latest"):
                dirs.append(d / "latest")
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [str(d) for d in dirs]
