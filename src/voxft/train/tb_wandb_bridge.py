"""TensorBoard → wandb/SwanLab 旁路桥接。

官方训练脚本只写 tensorboardX 日志；此模块周期读取 event 文件，
把标量与验证音频转发到配置的监控后端，不侵入 submodule 代码。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ..paths import env

def _ensure_wandb(run_name: str):
    if not env("WANDB_API_KEY"):
        return None
    import wandb
    if wandb.run is None:
        wandb.init(project=env("WANDB_PROJECT", "voxcpm-finetune"), name=run_name)
    return wandb


def _ensure_swanlab(run_name: str):
    if not env("SWANLAB_API_KEY"):
        return None
    import swanlab
    try:
        swanlab.get_run()
    except RuntimeError:
        swanlab.init(project=env("WANDB_PROJECT", "voxcpm-finetune"), name=run_name)
    return swanlab


def _sync_backend(reader, tb_dir, backend, state: dict) -> int:
    n = 0
    max_step = state.get("_max_step", -1)
    scalars = reader.scalars
    pending: dict[int, dict[str, float]] = {}
    for _, row in scalars.iterrows():
        tag, step, value = row["tag"], int(row["step"]), float(row["value"])
        if step <= state.get(tag, -1):
            continue
        if step <= max_step:
            state[tag] = step
            continue
        pending.setdefault(step, {})[tag] = value
    for step in sorted(pending):
        backend.log(pending[step], step=step)
        for tag in pending[step]:
            state[tag] = step
        n += len(pending[step])
        max_step = step
    state["_max_step"] = max_step

    import io
    import soundfile as sf
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    for path in sorted(Path(tb_dir).rglob('events.out.tfevents.*')):
        for event in EventFileLoader(str(path)).Load():
            step = int(event.step)
            for value in event.summary.value:
                if value.WhichOneof('value') != 'tensor' or not value.tensor.string_val:
                    continue
                raw = value.tensor.string_val[0]
                if not raw.startswith(b'RIFF'):
                    continue
                key = f"audio:{value.tag}:{step}"
                if state.get(key) or step < max_step:
                    continue
                arr, sample_rate = sf.read(io.BytesIO(raw), dtype='float32')
                backend.log({value.tag: backend.Audio(arr, sample_rate=sample_rate)}, step=step)
                state[key] = True
                max_step = max(max_step, step)
                state["_max_step"] = max_step
    return n


def sync_once(tb_dir: str | Path, run_name: str, state: dict) -> int:
    """增量同步到各后端；每个后端独立记录全局最大 step 和逐 tag 进度。"""
    backends = []
    errors = {}
    for name, ensure, backend_state in (('wandb', _ensure_wandb, state),
                                        ('swanlab', _ensure_swanlab, state.setdefault('_swanlab', {}))):
        try:
            backend = ensure(run_name)
        except Exception as error:
            errors[name] = str(error)
            continue
        if backend is not None:
            backends.append((name, backend, backend_state))
    state['_backend_errors'] = errors
    if not backends:
        if errors:
            raise RuntimeError('Monitoring backend initialization failed: '
                               + '; '.join(f'{name}: {error}' for name, error in errors.items()))
        return 0
    from tbparse import SummaryReader

    reader = SummaryReader(str(tb_dir))
    n, succeeded = 0, 0
    for name, backend, backend_state in backends:
        try:
            n += _sync_backend(reader, tb_dir, backend, backend_state)
            succeeded += 1
        except Exception as error:
            errors[name] = str(error)
    state['_backend_errors'] = errors
    if errors and not succeeded:
        raise RuntimeError('Monitoring backend synchronization failed: '
                           + '; '.join(f'{name}: {error}' for name, error in errors.items()))
    return n


def start_bridge(tb_dir: str | Path, run_name: str, interval: float = 30.0,
                 stop_event: threading.Event | None = None, progress=None
                 ) -> threading.Thread | None:
    """后台线程周期同步，直到 stop_event 被置位或 tb_dir 消失。"""
    if not env("WANDB_API_KEY") and not env("SWANLAB_API_KEY"):
        if progress:
            progress("[monitor] 未配置 WANDB_API_KEY 或 SWANLAB_API_KEY，跳过监控桥接")
        return None
    stop_event = stop_event or threading.Event()

    def loop():
        state: dict = {}
        while not stop_event.is_set():
            if Path(tb_dir).exists():
                try:
                    sync_once(tb_dir, run_name, state)
                    if progress and state.get('_backend_errors'):
                        progress(f"[monitor] 部分后端异常: {state['_backend_errors']}")
                except Exception as exc:
                    if progress:
                        progress(f"[monitor] 桥接异常: {exc}")
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
        if state.get('_backend_errors'):
            print(f"后端异常: {state['_backend_errors']}", file=sys.stderr, flush=True)
        if n:
            print(f"已转发 {n} 条指标")
        time.sleep(30)
