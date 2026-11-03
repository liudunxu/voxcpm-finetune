"""把已结束训练的 TensorBoard、数据覆盖和阶段门指标回补到 SwanLab。

sync_once 每次从头读取并增量转发，连续三次读到 0 条就退出。
阶段门、标量与验证音频按 step 对齐，避免乱序或漏传较早音频。
"""
import sys
import time

from voxft.train.tb_wandb_bridge import sync_once


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: python scripts/wandb_backfill.py <tb_dir> <run_name>", file=sys.stderr)
        return 1
    tb_dir, run_name = sys.argv[1], sys.argv[2]
    state: dict = {}
    idle = 0
    total = 0
    while idle < 3:
        n = sync_once(tb_dir, run_name, state)
        if state.get('_backend_errors'):
            print(f"backend_errors={state['_backend_errors']}", flush=True)
        total += n
        max_step = state.get('_max_step', -1)
        print(f"forwarded={n} total={total} max_step={max_step}", flush=True)
        idle = idle + 1 if n == 0 else 0
        time.sleep(5)
    print(f"done total={total} max_step={max_step}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
