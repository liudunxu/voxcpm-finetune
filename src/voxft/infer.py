from __future__ import annotations

import sys
import time
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
    key = (base, lora_dir or "")
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    VoxCPM = _import_voxcpm()
    kwargs = {"load_denoiser": False}  # 去噪器依赖 modelscope，试听场景不需要
    if lora_dir:
        kwargs["lora_weights_path"] = lora_dir
    _MODEL = VoxCPM.from_pretrained(base, **kwargs)
    _MODEL_KEY = key
    return _MODEL


def switch_lora(lora_dir: str | None) -> str:
    """热切换当前模型的 LoRA；返回状态描述。"""
    if _MODEL is None:
        return "模型尚未加载"
    try:
        if hasattr(_MODEL, "unload_lora"):
            _MODEL.unload_lora()
        if lora_dir:
            ret = _MODEL.load_lora(lora_dir)
            skipped = getattr(ret, "skipped_keys", None) or (ret[0] if isinstance(ret, tuple) else None)
            if skipped:
                return f"警告: skipped_keys 非空 {skipped}（推理侧配置需与训练一致）"
            return f"已加载 {lora_dir}"
        return "已卸载 LoRA（纯基座）"
    except Exception as exc:
        return f"切换失败: {exc}"


def synthesize(text: str, base_path: str | None = None,
               lora_dir: str | None = None,
               ref_audio: str | None = None, ref_text: str | None = None,
               cfg_value: float = 2.0, inference_timesteps: int = 20,
               seed: int | None = None) -> tuple[str, float]:
    """生成语音，返回 (wav 路径, 耗时秒)。"""
    model = get_model(base_path, lora_dir)
    t0 = time.time()
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
        if ref_text:
            kwargs["prompt_wav_path"] = ref_audio
            kwargs["prompt_text"] = ref_text
    if seed is not None:
        kwargs["seed"] = seed
    wav = model.generate(**kwargs)
    out = CHECKPOINT_DIR / "auditions" / f"audition_{int(time.time())}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sr = getattr(getattr(model, "tts_model", None), "sample_rate", None) or 48000
    sf.write(out, wav, sr)
    return str(out), round(time.time() - t0, 1)


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
