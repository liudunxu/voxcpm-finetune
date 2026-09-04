"""TensorBoard → wandb 旁路桥接。

官方训练脚本只写 tensorboardX 日志；此模块周期读取 event 文件，
把标量与验证音频转发到 wandb，不侵入 submodule 代码。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ..paths import env

AUDIO_SR = 48000  # 官方用 out_sample_rate 写验证音频


def _ensure_wandb(run_name: str):
    if not env("WANDB_API_KEY"):
        return None
    import wandb
    if wandb.run is None:
        wandb.init(project=env("WANDB_PROJECT", "voxcpm-finetune"), name=run_name)
    return wandb


def sync_once(tb_dir: str | Path, run_name: str, state: dict) -> int:
    """增量同步一次；返回本轮转发的标量条数。

    state["_max_step"] 记录已上报的全局最大 step；wandb 要求 step 单调，
    event 文件里验证指标可能晚于更晚的训练步落盘，早于它的直接丢弃。
    """
    wandb = _ensure_wandb(run_name)
    if wandb is None:
        return 0
    from tbparse import SummaryReader

    reader = SummaryReader(str(tb_dir))
    n = 0
    max_step = state.get("_max_step", -1)
    try:
        scalars = reader.scalars
        pending: dict[int, dict[str, float]] = {}
        for _, row in scalars.iterrows():
            tag, step, value = row["tag"], int(row["step"]), float(row["value"])
            if step <= state.get(tag, -1):
                continue
            state[tag] = step
            if step <= max_step:
                continue  # 晚到的旧步数，wandb 会拒收
            pending.setdefault(step, {})[tag] = value
        for step in sorted(pending):
            wandb.log(pending[step], step=step)
            n += len(pending[step])
            max_step = step
    except Exception:
        pass
    state["_max_step"] = max_step
    try:
        tensors = reader.tensors
        for _, row in tensors.iterrows():
            tag, step = row["tag"], int(row["step"])
            key = f"audio:{tag}:{step}"
            if state.get(key) or step <= max_step:
                continue
            state[key] = True
            import numpy as np
            arr = np.asarray(row["value"], dtype=np.float32).reshape(-1)
            wandb.log({tag: wandb.Audio(arr, sample_rate=AUDIO_SR)}, step=step)
            max_step = step
    except Exception:
        pass
    state["_max_step"] = max_step
    return n


def start_bridge(tb_dir: str | Path, run_name: str, interval: float = 30.0,
                 stop_event: threading.Event | None = None) -> threading.Thread | None:
    """后台线程周期同步，直到 stop_event 被置位或 tb_dir 消失。"""
    if not env("WANDB_API_KEY"):
        print("[wandb] 未配置 WANDB_API_KEY，跳过监控桥接")
        return None
    stop_event = stop_event or threading.Event()

    def loop():
        state: dict = {}
        while not stop_event.is_set():
            if Path(tb_dir).exists():
                try:
                    sync_once(tb_dir, run_name, state)
                except Exception as exc:
                    print(f"[wandb] 桥接异常: {exc}")
            stop_event.wait(interval)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python -m voxft.train.tb_wandb_bridge <tb_dir> <run_name>")
        raise SystemExit(1)
    state: dict = {}
    while True:
        n = sync_once(sys.argv[1], sys.argv[2], state)
        if n:
            print(f"已转发 {n} 条指标")
        time.sleep(30)
