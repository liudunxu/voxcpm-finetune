"""把已结束训练的 TensorBoard 事件一次性回补到 wandb，然后退出（不留常驻进程）。

用于补桥接漏掉的轮次：sync_once 每次从头读事件目录、按 state["_max_step"] 增量转发，
所以连续三次读到 0 条就说明已经追平，可以退出。
"""
import sys
import time

from voxft.train.tb_wandb_bridge import sync_once


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: python wandb_backfill_once.py <tb_dir> <run_name>", file=sys.stderr)
        return 1
    tb_dir, run_name = sys.argv[1], sys.argv[2]
    state: dict = {}
    idle = 0
    total = 0
    while idle < 3:
        n = sync_once(tb_dir, run_name, state)
        total += n
        print(f"forwarded={n} total={total} max_step={state.get('_max_step')}", flush=True)
        idle = idle + 1 if n == 0 else 0
        time.sleep(5)
    print(f"done total={total} max_step={state.get('_max_step')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
