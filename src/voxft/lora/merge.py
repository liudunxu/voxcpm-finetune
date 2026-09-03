from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..paths import CHECKPOINT_DIR, env


def find_checkpoints(run_name: str | None = None) -> list[Path]:
    """列出 checkpoint 目录（step_* 与 latest）。"""
    roots = [CHECKPOINT_DIR / run_name] if run_name else [
        d for d in CHECKPOINT_DIR.iterdir() if d.is_dir()
    ] if CHECKPOINT_DIR.exists() else []
    out = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(sorted(root.glob("step_*")))
        if (root / "latest").exists():
            out.append(root / "latest")
    return out


def is_lora_dir(path: Path) -> bool:
    return (path / "lora_weights.safetensors").exists() and (path / "lora_config.json").exists()


def merge_lora(base_path: str, lora_dir: str | Path, out_dir: str | Path) -> Path:
    """把 LoRA 增量合并进基座，导出完整模型目录（与全量 checkpoint 同构）。

    base_path: 基座目录（含 model.safetensors + config.json + audiovae.pth）
    """
    base, lora_dir, out = Path(base_path), Path(lora_dir), Path(out_dir)
    if not is_lora_dir(lora_dir):
        raise ValueError(f"{lora_dir} 缺少 lora_weights.safetensors / lora_config.json")
    lora_cfg = json.loads((lora_dir / "lora_config.json").read_text())
    lora_params = lora_cfg.get("lora", lora_cfg)  # 兼容嵌套/平铺两种结构
    r = float(lora_params.get("r", 32))
    scale = float(lora_params.get("alpha", r)) / r
    lora_weights = load_file(str(lora_dir / "lora_weights.safetensors"))

    # 官方命名：<prefix>.lora_A / <prefix>.lora_B，基座权重在 <prefix>.weight
    pairs: dict[str, tuple[str, str]] = {}
    for key in lora_weights:
        if key.endswith(".lora_A"):
            prefix = key[: -len(".lora_A")]
            b_key = prefix + ".lora_B"
            if b_key in lora_weights:
                pairs[prefix + ".weight"] = (key, b_key)

    out.mkdir(parents=True, exist_ok=True)
    merged_n = 0
    for f in sorted(base.glob("*.safetensors")):
        tensors = load_file(str(f))
        for base_key, (a_key, b_key) in pairs.items():
            if base_key in tensors:
                a, b = lora_weights[a_key].float(), lora_weights[b_key].float()
                tensors[base_key] = tensors[base_key].float().add_(b @ a * scale).to(tensors[base_key].dtype)
                merged_n += 1
        save_file(tensors, str(out / f.name))
    if merged_n != len(pairs):
        print(f"[merge] 警告: {len(pairs)} 个 LoRA 层中 {merged_n} 个在基座中找到匹配；"
              f"请核对基座与训练版本一致")
    for f in base.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy2(f, out / f.name)
    print(f"[merge] 完成：{merged_n} 层合并，缩放 {scale:.3f} → {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="合并 LoRA 到基座模型")
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None,
                    help="基座目录；默认取 VOXCPM_BASE_PATH（本地目录时）")
    args = ap.parse_args()
    base = args.base or env("VOXCPM_BASE_PATH")
    if not base or not Path(base).is_dir():
        raise SystemExit("需要本地基座目录：先下载 openbmb/VoxCPM2 或设置 VOXCPM_BASE_PATH")
    merge_lora(base, args.lora_dir, args.out)
