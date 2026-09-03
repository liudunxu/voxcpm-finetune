"""TB→wandb 桥接增量逻辑自测：合成 TensorBoard event 文件 + 桩 wandb。"""
from pathlib import Path

import pytest


@pytest.fixture
def stub_wandb(monkeypatch):
    class Run:
        def __init__(self):
            self.logged = []

        def log(self, d, step=None):
            self.logged.append((dict(d), step))

    run = Run()
    stub = type("wandb_stub", (), {"run": run, "log": run.log, "Audio": lambda *a, **k: a[0]})
    from voxft.train import tb_wandb_bridge
    monkeypatch.setattr(tb_wandb_bridge, "_ensure_wandb", lambda name: stub)
    return run


def test_bridge_incremental(tmp_path, stub_wandb):
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    w = SummaryWriter(str(tmp_path))
    for i in range(5):
        w.add_scalar("loss/diff", 1.0 / (i + 1), i)
    w.close()

    state: dict = {}
    n1 = sync_once(tmp_path, "test_run", state)
    assert n1 == 5
    assert stub_wandb.logged[-1][0]["loss/diff"] == pytest.approx(0.2)
    # 重复同步不应产生新数据
    assert sync_once(tmp_path, "test_run", state) == 0
    # 新数据写入后仅转发增量
    w = SummaryWriter(str(tmp_path))
    w.add_scalar("loss/diff", 0.1, 5)
    w.close()
    assert sync_once(tmp_path, "test_run", state) == 1
