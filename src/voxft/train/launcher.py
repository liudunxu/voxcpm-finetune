from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..paths import CHECKPOINT_DIR, VOXCPM_REPO

TRAIN_SCRIPT = VOXCPM_REPO / "scripts" / "train_voxcpm_finetune.py"


def gpu_command(config_path: str | Path, gpus: int = 1,
                cuda_devices: str | None = None) -> str:
    """生成 GPU 机器上的训练命令（本地无 CUDA 时复制到远程执行）。"""
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"官方训练脚本不存在: {TRAIN_SCRIPT}（submodule 未初始化？）")
    if gpus > 1:
        cmd = f"torchrun --nproc_per_node={gpus} {TRAIN_SCRIPT} --config_path {config_path}"
    else:
        cmd = f"python {TRAIN_SCRIPT} --config_path {config_path}"
    cmd = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {cmd}"
    if cuda_devices:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} {cmd}"
    return cmd


def resolve_base_path(path: str, progress=None) -> str:
    """基座路径归一化：本地目录直接用；HF 仓库 ID 则下载快照后返回本地目录。

    官方训练脚本要求 pretrained_path 是含 config.json 的本地目录。
    """
    if not path or Path(path).is_dir():
        return path
    import threading

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    from ..log import _fmt_size
    from ..paths import env
    token = env("HF_TOKEN") or None
    if progress:
        import os
        progress(f"基座模型本地不存在，开始下载 {path}（模型较大，请耐心等待；"
                 f"endpoint={os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'}）")
    cache_dir = Path(HF_HUB_CACHE) / f"models--{path.replace('/', '--')}"

    def _dir_size(d: Path) -> int:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) \
            if d.exists() else 0

    stop = threading.Event()
    total = 0

    def watcher():
        # tqdm_class 不会传给单文件下载，聚合条对超大文件无输出，直接轮询目录大小
        while not stop.wait(5):
            try:
                done = _dir_size(cache_dir)
                if total:
                    progress(f"基座下载中: {_fmt_size(done)}/{_fmt_size(total)}"
                             f" ({min(100, 100 * done / total):.0f}%)")
                else:
                    progress(f"基座下载中: {_fmt_size(done)}")
            except Exception as exc:
                progress(f"进度监控异常（不影响下载）: {exc}")

    t = threading.Thread(target=watcher, daemon=True) if progress else None
    if t:
        progress(f"下载进度监控已启动（每 5 秒更新）")
        t.start()
    try:
        if progress:
            try:
                info = HfApi(token=token).repo_info(path, files_metadata=True)
                total = sum(s.size or 0 for s in info.siblings)
                progress(f"基座总大小: {_fmt_size(total)}")
            except Exception:
                pass
        local = snapshot_download(path, token=token)
    finally:
        stop.set()
    if progress:
        progress(f"基座模型就绪 → {local}")
    return local


def preflight(config_path: str | Path) -> list[str]:
    """训练前预检；返回问题列表（空 = 通过）。把报错提前到启动前。"""
    import json
    import yaml

    issues: list[str] = []
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return [f"配置文件不存在: {cfg_path}"]
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if not TRAIN_SCRIPT.exists():
        issues.append(f"官方训练脚本不存在: {TRAIN_SCRIPT}（执行过 git submodule update --init？）")

    for key, required in (("train_manifest", True), ("val_manifest", False)):
        p = str(cfg.get(key, "") or "")
        if not p:
            if required:
                issues.append(f"{key} 未配置")
            continue
        mp = Path(p)
        if not mp.exists():
            issues.append(f"{key} 不存在: {mp}")
            continue
        with mp.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    issues.append(f"{key} 第 {i + 1} 行不是合法 JSON")
                    continue
                if not Path(rec.get("audio", "")).exists():
                    issues.append(f"{key} 第 {i + 1} 行音频不存在: {rec.get('audio')}")

    pre = str(cfg.get("pretrained_path", "") or "")
    if pre and not Path(pre).is_dir():
        if pre.count("/") == 1:
            issues.append(f"pretrained_path 是 HF 仓库 ID（{pre}），训练脚本要求本地目录："
                          "请在页面重新「生成训练配置」（会自动下载基座），"
                          "或在 .env 设置 VOXCPM_BASE_PATH 指向本地基座目录")
        else:
            issues.append(f"pretrained_path 既非本地目录也非 HF 仓库 ID（owner/name）: {pre}")

    save = Path(cfg.get("save_path", ""))
    try:
        save.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        issues.append(f"save_path 不可写: {save} ({exc})")

    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("警告：本机无 CUDA，训练请在 GPU 服务器执行（本机仅生成命令）")
    except Exception:
        pass
    return issues


_PROC: subprocess.Popen | None = None


def start_local(config_path: str | Path, gpus: int = 1) -> Path:
    """在本机（需有 GPU）以后台子进程启动训练，日志写入 run 目录下 train.log。"""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        raise RuntimeError("已有训练任务在运行；先停止或等待其结束")
    issues = preflight(config_path)
    fatal = [i for i in issues if not i.startswith("警告")]
    if fatal:
        raise RuntimeError("预检未通过：\n" + "\n".join(issues))
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    log_path = Path(cfg["save_path"]) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = gpu_command(config_path, gpus)
    # ponytail: 单实例全局锁（_PROC），多任务并行时改为按 run 字典管理
    _PROC = subprocess.Popen(
        cmd, shell=True, cwd=VOXCPM_REPO,
        stdout=log_path.open("a"), stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    from .tb_wandb_bridge import start_bridge
    start_bridge(cfg.get("tensorboard", ""), Path(config_path).stem)
    return log_path


def stop_local() -> bool:
    global _PROC
    if _PROC is None or _PROC.poll() is not None:
        return False
    _PROC.terminate()
    try:
        _PROC.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _PROC.kill()
    _PROC = None
    return True


def status() -> dict:
    running = _PROC is not None and _PROC.poll() is None
    return {"running": running, "returncode": None if running else
            (_PROC.returncode if _PROC is not None else None)}


def tail_log(log_path: str | Path, lines: int = 30) -> str:
    p = Path(log_path)
    if not p.exists():
        return "（日志尚未生成）"
    return "".join(p.read_text(encoding="utf-8", errors="replace")
                   .splitlines(keepends=True)[-lines:])


def list_runs() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(d.name for d in CHECKPOINT_DIR.iterdir() if d.is_dir())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(gpu_command(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    else:
        print("用法: python -m voxft.train.launcher <config.yaml> [gpus]")
