"""TB→wandb 桥接增量逻辑自测：合成 TensorBoard event 文件 + 桩 wandb。"""

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
    monkeypatch.setattr(tb_wandb_bridge, "_ensure_swanlab", lambda name: None)
    return run


@pytest.fixture
def stub_swanlab(monkeypatch):
    class Run:
        def __init__(self):
            self.logged = []

        def log(self, d, step=None):
            self.logged.append((dict(d), step))

    run = Run()
    stub = type("swanlab_stub", (), {"log": run.log, "Audio": lambda *a, **k: a[0]})
    from voxft.train import tb_wandb_bridge
    monkeypatch.setattr(tb_wandb_bridge, "_ensure_wandb", lambda name: None)
    monkeypatch.setattr(tb_wandb_bridge, "_ensure_swanlab", lambda name: stub)
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


def test_swanlab_bridge_has_independent_incremental_state(tmp_path, stub_swanlab):
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    w = SummaryWriter(str(tmp_path))
    w.add_scalar("loss/diff", 1.0, 0)
    w.close()
    state: dict = {}
    assert sync_once(tmp_path, "test_run", state) == 1
    assert stub_swanlab.logged == [({"loss/diff": 1.0}, 0)]
    assert state["_swanlab"]["_max_step"] == 0
    assert "_max_step" not in state
    assert sync_once(tmp_path, "test_run", state) == 0


def test_swanlab_audio_uses_wav_tensor_and_deduplicates(tmp_path, stub_swanlab):
    import numpy as np
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    w = SummaryWriter(str(tmp_path))
    w.add_scalar('loss/diff', 1.0, 0)
    w.add_audio('valid/sample', np.zeros(480, dtype=np.float32), 0, sample_rate=48000)
    w.close()
    state: dict = {}
    assert sync_once(tmp_path, 'test_run', state) == 1
    assert len(stub_swanlab.logged) == 2
    assert stub_swanlab.logged[1][0]['valid/sample'].shape == (480,)
    assert state['_swanlab']['audio:valid/sample:0'] is True
    assert sync_once(tmp_path, 'test_run', state) == 0


def test_wandb_failure_does_not_block_swanlab(tmp_path, stub_swanlab, monkeypatch):
    from tensorboardX import SummaryWriter
    from voxft.train import tb_wandb_bridge

    def fail(_):
        raise RuntimeError('wandb blocked')

    monkeypatch.setattr(tb_wandb_bridge, '_ensure_wandb', fail)
    w = SummaryWriter(str(tmp_path))
    w.add_scalar('loss/diff', 1.0, 0)
    w.close()
    state: dict = {}
    assert tb_wandb_bridge.sync_once(tmp_path, 'test_run', state) == 1
    assert stub_swanlab.logged == [({'loss/diff': 1.0}, 0)]
    assert state['_backend_errors'] == {'wandb': 'wandb blocked'}


def test_start_bridge_reports_via_progress_not_print(monkeypatch, capsys):
    """库函数不许 print：UI 的 stdout 可能是断开的 pty，Errno 5 会把已成功的启动报成失败。"""
    from voxft.train import tb_wandb_bridge

    monkeypatch.setenv("WANDB_API_KEY", "")
    monkeypatch.setenv("SWANLAB_API_KEY", "")
    seen = []
    assert tb_wandb_bridge.start_bridge("/nonexistent", "run", progress=seen.append) is None
    assert seen and "WANDB_API_KEY" in seen[0]
    assert capsys.readouterr().out == ""


def test_failed_scalar_is_reported_and_retried(tmp_path, stub_swanlab, monkeypatch):
    from tensorboardX import SummaryWriter
    from voxft.train import tb_wandb_bridge

    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar('loss/diff', 1.0, 0)
    writer.close()
    backend = type('Backend', (), {})()
    backend.log = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('upload failed'))
    monkeypatch.setattr(tb_wandb_bridge, '_ensure_swanlab', lambda _: backend)
    state = {}
    with pytest.raises(RuntimeError, match='synchronization failed'):
        tb_wandb_bridge.sync_once(tmp_path, 'test_run', state)
    assert state['_backend_errors'] == {'swanlab': 'upload failed'}
    assert 'loss/diff' not in state['_swanlab']
    backend.log = stub_swanlab.log
    assert tb_wandb_bridge.sync_once(tmp_path, 'test_run', state) == 1
    assert state['_backend_errors'] == {}
    assert stub_swanlab.logged == [({'loss/diff': 1.0}, 0)]


def test_sync_failure_does_not_block_other_backend(tmp_path, stub_swanlab, monkeypatch):
    from tensorboardX import SummaryWriter
    from voxft.train import tb_wandb_bridge

    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar('loss/diff', 1.0, 0)
    writer.close()
    failing = type('Backend', (), {})()
    failing.log = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('wandb upload failed'))
    monkeypatch.setattr(tb_wandb_bridge, '_ensure_wandb', lambda _: failing)
    state = {}
    assert tb_wandb_bridge.sync_once(tmp_path, 'test_run', state) == 1
    assert state['_backend_errors'] == {'wandb': 'wandb upload failed'}
    assert stub_swanlab.logged == [({'loss/diff': 1.0}, 0)]


def test_failed_audio_is_retried_without_duplicating_scalar(tmp_path, stub_swanlab, monkeypatch):
    import numpy as np
    from tensorboardX import SummaryWriter
    from voxft.train import tb_wandb_bridge

    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar('loss/diff', 1.0, 0)
    writer.add_audio('valid/sample', np.zeros(480, dtype=np.float32), 0, sample_rate=48000)
    writer.close()
    backend = type('Backend', (), {})()
    backend.Audio = lambda value, sample_rate: value
    def fail_audio(data, step=None):
        if 'valid/sample' in data:
            raise RuntimeError('audio upload failed')
        return stub_swanlab.log(data, step=step)
    backend.log = fail_audio
    monkeypatch.setattr(tb_wandb_bridge, '_ensure_swanlab', lambda _: backend)
    state = {}
    with pytest.raises(RuntimeError, match='synchronization failed'):
        tb_wandb_bridge.sync_once(tmp_path, 'test_run', state)
    assert state['_swanlab']['loss/diff'] == 0
    assert 'audio:valid/sample:0' not in state['_swanlab']
    backend.log = stub_swanlab.log
    assert tb_wandb_bridge.sync_once(tmp_path, 'test_run', state) == 0
    assert state['_backend_errors'] == {}
    assert len(stub_swanlab.logged) == 2
