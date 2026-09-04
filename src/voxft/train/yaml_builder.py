from __future__ import annotations

import time
from pathlib import Path

import yaml

from ..paths import CHECKPOINT_DIR, CONFIG_DIR, env

# 官方推荐默认值（VoxCPM 2）
DEFAULTS = {
    "sample_rate": 16000,        # AudioVAE 编码器输入，勿改成输出采样率
    "out_sample_rate": 48000,    # 仅推理用
    "batch_size": 2,             # 官方示例值；音频序列长，激活显存大，勿调大
    "grad_accum_steps": 8,       # 等效 batch = 2 × 8 = 16
    "num_workers": 8,           # 官方 v2 配置值；音频解码是瓶颈，别调小
    "num_iters": 1000,
    "log_interval": 10,
    "valid_interval": 250,
    "save_interval": 250,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "max_batch_tokens": 8192,
    "max_grad_norm": 1.0,        # 官方 v2 配置值；情感语料动态大，更容易出梯度尖峰
    "diff_loss_weight": 1.0,
    "stop_loss_weight": 1.0,
}

# r=64/alpha=64：语言 + 风格双适配（纯说话人适配用 32 就够）；
# dropout 0.05：情感语料体量小（万级），0 容易几百步就过拟合到固定腔调
LORA_PRESET = {"learning_rate": 1e-4, "r": 64, "alpha": 64, "dropout": 0.05}
FULL_PRESET = {"learning_rate": 1e-5}  # 约为 LoRA 的 1/10，防灾难性遗忘


def build_yaml(run_name: str, pretrained_path: str, train_manifest: str,
               val_manifest: str = "", finetune_type: str = "lora",
               overrides: dict | None = None) -> Path:
    """生成官方训练脚本可用的 YAML，写入 configs/<run_name>.yaml。"""
    if finetune_type not in ("lora", "full"):
        raise ValueError("finetune_type 必须是 lora 或 full")
    base = env("VOXCPM_BASE_PATH") or "openbmb/VoxCPM2"
    cfg = {
        "pretrained_path": pretrained_path or base,
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        **DEFAULTS,
    }
    if finetune_type == "lora":
        cfg["learning_rate"] = LORA_PRESET["learning_rate"]
        cfg["lora"] = {
            "enable_lm": True,
            "enable_dit": True,      # 对音质至关重要
            "enable_proj": False,
            "r": LORA_PRESET["r"],
            "alpha": LORA_PRESET["alpha"],
            "dropout": LORA_PRESET["dropout"],
        }
    else:
        cfg["learning_rate"] = FULL_PRESET["learning_rate"]
    cfg["max_steps"] = cfg["num_iters"]
    save_path = CHECKPOINT_DIR / run_name
    cfg["save_path"] = str(save_path)
    cfg["tensorboard"] = str(save_path / "logs")
    cfg["lambdas"] = {
        "loss/diff": cfg.pop("diff_loss_weight"),
        "loss/stop": cfg.pop("stop_loss_weight"),
    }
    if overrides:
        for k, v in overrides.items():
            if k in ("lora", "lambdas") and isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        if "num_iters" in overrides and "max_steps" not in overrides:
            cfg["max_steps"] = cfg["num_iters"]

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def default_run_name(finetune_type: str) -> str:
    return f"{finetune_type}_{time.strftime('%m%d_%H%M%S')}"
