from __future__ import annotations

import json
import math
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


def manifest_langs(train_manifest: str) -> dict[str, int]:
    """清单里各语种的条数；未标 lang 的行归到 unknown（联合微调时它会直接暴露出来）。"""
    langs: dict[str, int] = {}
    for line in Path(train_manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            lang = json.loads(line).get("lang") or "unknown"
            langs[lang] = langs.get(lang, 0) + 1
    return langs


def steps_for_epochs(train_manifest: str, epochs: float, batch_size: int = 2,
                     grad_accum_steps: int = 8, gpus: int = 1) -> int:
    if not (0 < epochs <= 3) or any(not isinstance(n, int) or n < 1
                                   for n in (batch_size, grad_accum_steps, gpus)):
        raise ValueError("epochs 必须在 (0, 3]，batch/累积/GPU 数必须为正整数")
    with Path(train_manifest).open(encoding="utf-8") as f:
        n = sum(bool(line.strip()) for line in f)
    if not n:
        raise ValueError("训练清单为空")
    return max(1, math.ceil(n * epochs / (batch_size * grad_accum_steps * gpus)))


def build_yaml(run_name: str, pretrained_path: str, train_manifest: str,
               val_manifest: str = "", finetune_type: str = "lora",
               overrides: dict | None = None, *, epochs: float | None = None,
               gpus: int = 1) -> Path:
    """生成官方训练脚本可用的 YAML，写入 configs/<run_name>.yaml。"""
    if finetune_type not in ("lora", "full"):
        raise ValueError("finetune_type 必须是 lora 或 full")
    if Path(run_name).name != run_name or run_name in ("", ".", ".."):
        raise ValueError("run_name 必须为单个目录名")
    if overrides and "training_cfg_rate" in overrides:
        raise ValueError("training_cfg_rate 属于基座 dit_config.cfm_config，不是训练 YAML 顶层参数")
    train_manifest = str(Path(train_manifest).resolve())
    val_manifest = str(Path(val_manifest).resolve()) if val_manifest else ""
    if pretrained_path and Path(pretrained_path).is_dir():
        pretrained_path = str(Path(pretrained_path).resolve())
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

    if epochs is not None:
        if overrides and ("num_iters" in overrides or "max_steps" in overrides):
            raise ValueError("epochs 与显式训练步数只能选择一种")
        cfg["num_iters"] = cfg["max_steps"] = steps_for_epochs(
            train_manifest, epochs, cfg["batch_size"], cfg["grad_accum_steps"], gpus)
        cfg["warmup_steps"] = max(1, int(cfg["num_iters"] * 0.1))
    if val_manifest and Path(val_manifest).exists() and not Path(val_manifest).read_text().strip():
        cfg["val_manifest"] = ""
    for key in ("batch_size", "grad_accum_steps", "num_iters", "save_interval", "valid_interval"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    if not isinstance(gpus, int) or gpus < 1 or not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("GPU 数必须为正整数，学习率必须为有限正数")
    langs = manifest_langs(train_manifest)
    train_samples = sum(langs.values())
    if not train_samples:
        raise ValueError("训练清单为空")

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    path.with_suffix(".plan.json").write_text(json.dumps(
        {"gpus": gpus, "epochs": epochs, "num_iters": cfg["num_iters"],
         "effective_batch": cfg["batch_size"] * cfg["grad_accum_steps"] * gpus,
         "train_manifest": train_manifest, "train_samples": train_samples,
         "langs": dict(sorted(langs.items()))},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def default_run_name(finetune_type: str, train_manifest: str = "") -> str:
    """带上语种，联合 run 才分得清训了什么；≥3 个语种用 jointN，免得 run 名过长。"""
    langs = sorted(manifest_langs(train_manifest)) if train_manifest else []
    tag = f"joint{len(langs)}" if len(langs) > 2 else "-".join(langs)
    return f"{finetune_type}_{tag + '_' if tag else ''}{time.strftime('%m%d_%H%M%S')}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="按 1–3 epoch 生成 LoRA 配置；在 GPU 机器执行训练")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--base", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()
    from .launcher import resolve_base_path
    base = resolve_base_path(args.base or env("VOXCPM_BASE_PATH") or "openbmb/VoxCPM2",
                             progress=print)
    print(build_yaml(args.run or default_run_name("lora", args.train), base, args.train,
                     args.val, epochs=args.epochs, gpus=args.gpus))
