"""LoRA merge 数学正确性自测（合成矩阵）。"""
import json

import torch
from safetensors.torch import load_file, save_file

from voxft.lora.merge import merge_lora


def test_merge_math(tmp_path):
    torch.manual_seed(0)
    out_f, in_f, r, alpha = 5, 4, 2, 4.0
    W = torch.randn(out_f, in_f)
    A = torch.randn(r, in_f)
    B = torch.randn(out_f, r)

    base = tmp_path / "base"
    base.mkdir()
    save_file({"lm.layer.weight": W}, str(base / "model.safetensors"))
    (base / "config.json").write_text("{}")
    (base / "audiovae.pth").write_bytes(b"x")

    lora = tmp_path / "lora"
    lora.mkdir()
    save_file({"lm.layer.lora_A": A, "lm.layer.lora_B": B},
              str(lora / "lora_weights.safetensors"))
    (lora / "lora_config.json").write_text(json.dumps({"r": r, "alpha": alpha}))

    out = merge_lora(base, lora, tmp_path / "merged")
    merged = load_file(str(out / "model.safetensors"))["lm.layer.weight"]
    expected = W + (B @ A) * (alpha / r)
    assert torch.allclose(merged, expected, atol=1e-6)
    # 非 safetensors 文件（config/audiovae）应被复制
    assert (out / "config.json").exists() and (out / "audiovae.pth").exists()
