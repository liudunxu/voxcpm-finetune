"""权重/模型预取：读项目 .env，自动用镜像并关掉 xet。

`hf` / `huggingface-cli` 不读项目 .env，每次都得手动 export 三个变量，
还容易漏掉 HF_HUB_DISABLE_XET（走镜像不关它会在 reconstruction 阶段 401）。
这个入口导入 voxft 时就把 .env 和 xet 开关处理好了。

    uv run python -m voxft.data.prefetch --whisper large-v3
    uv run python -m voxft.data.prefetch --repo openbmb/VoxCPM2
"""
from __future__ import annotations

import argparse
import os

from ..paths import env  # 导入即触发 load_dotenv + 镜像下自动关 xet

WHISPER_REPO = "Systran/faster-whisper-{size}"


def prefetch(repo: str, progress=None) -> str:
    """下载整个仓库到 HF 缓存，返回本地快照目录。"""
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    xet = os.environ.get("HF_HUB_DISABLE_XET", "0")
    msg = (f"预取 {repo}\n  endpoint={endpoint}\n  "
           f"HF_HUB_DISABLE_XET={xet}\n  HF_HOME={env('HF_HOME', '(默认 ~/.cache)')}")
    print(msg) if progress is None else progress(msg)
    path = snapshot_download(repo_id=repo, token=env("HF_TOKEN") or None)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="预取模型权重（自动用 .env 里的镜像设置）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--whisper", metavar="SIZE",
                   help="faster-whisper 尺寸，如 large-v3 / medium")
    g.add_argument("--repo", help="任意 HF 仓库 id")
    args = ap.parse_args()

    repo = args.repo or WHISPER_REPO.format(size=args.whisper)
    path = prefetch(repo)
    print(f"\n完成 → {path}")
    if args.whisper:
        key = ("VOXFT_WHISPER_MODEL_LARGE" if args.whisper.startswith("large")
               else "VOXFT_WHISPER_MODEL")
        print(f"\n可选：把它固定到 .env，避免以后再走网络\n  echo '{key}={path}' >> .env")


if __name__ == "__main__":
    main()
