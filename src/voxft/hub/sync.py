from __future__ import annotations

import json
from pathlib import Path

from ..paths import env


def _api():
    from huggingface_hub import HfApi
    token = env("HF_TOKEN")
    if not token:
        raise RuntimeError("未配置 HF_TOKEN（.env）")
    return HfApi(token=token)


def _dataset_card(local_dir: Path) -> str:
    meta = {}
    for name in ("stats.json", "mix.json"):
        p = local_dir / name
        if p.exists():
            meta[name] = json.loads(p.read_text(encoding="utf-8"))
    return f"""---
license: other
task_categories: [text-to-speech]
---

# VoxCPM 微调数据集

由 voxft 数据管线加工，用于 VoxCPM 2 微调（JSONL: audio/text/ref_audio/duration）。

- train: `train.jsonl`，val: `val.jsonl`
- 音频：16kHz 单声道 WAV，已裁尾静音（<0.5s）、响度归一化、3–30s 过滤

```json
{json.dumps(meta, ensure_ascii=False, indent=2)}
```

> 使用请遵守上游数据集许可（见上游来源页面）。
"""


def _model_card(local_dir: Path) -> str:
    cfgs = sorted(Path(local_dir).glob("*.yaml"))
    cfg_text = cfgs[0].read_text(encoding="utf-8") if cfgs else "（无训练配置）"
    return f"""---
license: apache-2.0
base_model: openbmb/VoxCPM2
tags: [voxcpm, tts, fine-tune, lora]
---

# VoxCPM 2 微调权重

由 voxft 工作台训练/合并产出。加载方式见仓库内说明（LoRA 目录含
lora_weights.safetensors + lora_config.json；merged 目录为完整模型）。

## 训练配置

```yaml
{cfg_text}
```
"""


def upload_folder(local_dir: str | Path, repo_id: str, kind: str = "model") -> str:
    """上传目录到 HuggingFace。kind: model | dataset。返回仓库链接。"""
    local = Path(local_dir)
    if not local.is_dir():
        raise FileNotFoundError(local)
    api = _api()
    repo_type = "dataset" if kind == "dataset" else "model"
    card = (_dataset_card if kind == "dataset" else _model_card)(local)
    card_path = local / "README.md"
    if not card_path.exists():
        card_path.write_text(card, encoding="utf-8")
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type=repo_type)
    host = "datasets" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{host + '/' if host else ''}{repo_id}"
