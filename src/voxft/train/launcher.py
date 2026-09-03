from __future__ import annotations

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
    if cuda_devices:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_devices} {cmd}"
    return cmd


_PROC: subprocess.Popen | None = None


def start_local(config_path: str | Path, gpus: int = 1) -> Path:
    """在本机（需有 GPU）以后台子进程启动训练，日志写入 <save_path 同级>/train.log。"""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        raise RuntimeError("已有训练任务在运行；先停止或等待其结束")
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
