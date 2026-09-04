from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip() or default


def load_dotenv() -> None:
    """极简 .env 加载（KEY=VALUE），已存在的环境变量优先。"""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _disable_xet_on_mirror() -> None:
    """走 HF 镜像时关掉 xet 传输。

    hf-mirror 这类镜像只代理 HF API，不代理 xet 的 CAS 服务器
    （cas-server.xethub.hf.co），走 xet 会在 reconstruction 阶段直连并报
    401 Unauthorized。关掉后回落经典 HTTP 下载，镜像能正常代理。
    直连 huggingface.co 时不动它——那种情况下 xet 更快。
    """
    endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if endpoint and "huggingface.co" not in endpoint:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# 必须在下方路径常量读取环境变量之前加载 .env
load_dotenv()
_disable_xet_on_mirror()

# 数据根目录可用 VOXFT_DATA_ROOT 覆盖（GPU 服务器上建议指向 /root/autodl-tmp）
_DATA_ROOT = Path(os.environ.get("VOXFT_DATA_ROOT") or (ROOT / "data"))
DATA_RAW = _DATA_ROOT / "raw"
DATA_PROCESSED = _DATA_ROOT / "processed"
CONFIG_DIR = ROOT / "configs"
CHECKPOINT_DIR = Path(os.environ.get("VOXFT_CKPT_ROOT") or (ROOT / "checkpoints"))
VOXCPM_REPO = ROOT / "third_party" / "VoxCPM"
MODEL_DIR = ROOT / "models"
