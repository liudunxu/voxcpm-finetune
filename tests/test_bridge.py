"""TB→SwanLab 桥接增量逻辑自测：合成 TensorBoard event 文件 + 桩后端。"""

import pytest


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
    monkeypatch.setattr(tb_wandb_bridge, "_ensure_swanlab", lambda name: stub)
    return run


def test_bridge_incremental(tmp_path, stub_swanlab):
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    w = SummaryWriter(str(tmp_path))
    for i in range(5):
        w.add_scalar("loss/diff", 1.0 / (i + 1), i)
    w.close()

    state: dict = {}
    n1 = sync_once(tmp_path, "test_run", state)
    assert n1 == 5
    assert stub_swanlab.logged[-1][0]["loss/diff"] == pytest.approx(0.2)
    assert sync_once(tmp_path, "test_run", state) == 0
    w = SummaryWriter(str(tmp_path))
    w.add_scalar("loss/diff", 0.1, 5)
    w.close()
    assert sync_once(tmp_path, "test_run", state) == 1


def test_swanlab_bridge_incremental_state(tmp_path, stub_swanlab):
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    w = SummaryWriter(str(tmp_path))
    w.add_scalar("loss/diff", 1.0, 0)
    w.close()
    state: dict = {}
    assert sync_once(tmp_path, "test_run", state) == 1
    assert stub_swanlab.logged == [({"loss/diff": 1.0}, 0)]
    assert state["_max_step"] == 0
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
    assert state['audio:valid/sample:0'] is True
    assert sync_once(tmp_path, 'test_run', state) == 0


def test_start_bridge_reports_via_progress_not_print(monkeypatch, capsys):
    """库函数不许 print：UI 的 stdout 可能是断开的 pty，Errno 5 会把已成功的启动报成失败。"""
    from voxft.train import tb_wandb_bridge

    monkeypatch.setenv("SWANLAB_API_KEY", "")
    seen = []
    assert tb_wandb_bridge.start_bridge("/nonexistent", "run", progress=seen.append) is None
    assert seen and "SWANLAB_API_KEY" in seen[0]
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
    assert 'loss/diff' not in state
    backend.log = stub_swanlab.log
    assert tb_wandb_bridge.sync_once(tmp_path, 'test_run', state) == 1
    assert state['_backend_errors'] == {}
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
    assert state['loss/diff'] == 0
    assert 'audio:valid/sample:0' not in state
    backend.log = stub_swanlab.log
    assert tb_wandb_bridge.sync_once(tmp_path, 'test_run', state) == 0
    assert state['_backend_errors'] == {}
    assert len(stub_swanlab.logged) == 2


def test_training_adds_weighted_loss_and_gradient_clipping(tmp_path, stub_swanlab):
    import yaml
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    work = tmp_path / 'run'
    logs = work / 'checkpoint' / 'logs'
    logs.mkdir(parents=True)
    (work / 'train.yaml').write_text(yaml.safe_dump({
        'lambdas': {'loss/diff': 1.0, 'loss/stop': 2.0}, 'max_grad_norm': 1.0}), encoding='utf-8')
    writer = SummaryWriter(str(logs))
    writer.add_scalar('train/loss/diff', 1.0, 0)
    writer.add_scalar('train/loss/stop', 0.25, 0)
    writer.add_scalar('train/grad_norm', 2.0, 0)
    writer.close()
    state = {}
    assert sync_once(logs, 'test_run', state) == 6
    logged = stub_swanlab.logged[0][0]
    assert logged['train/loss/weighted_total'] == pytest.approx(1.5)
    assert logged['train/grad_clip_ratio'] == pytest.approx(0.5)
    assert logged['train/grad_clipped'] == 1.0
    assert sync_once(logs, 'test_run', state) == 0


def test_data_coverage_uses_actual_manifest_and_mix(tmp_path, stub_swanlab):
    import json
    import yaml
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    work = tmp_path / 'run'
    logs = work / 'checkpoint' / 'logs'
    logs.mkdir(parents=True)
    manifest = tmp_path / 'train.jsonl'
    rows = [
        {'lang': 'th', 'duration': 4, 'ref_audio': 'th-ref.wav'},
        {'lang': 'th', 'duration': 10, 'ref_audio': ''},
        {'lang': 'tl', 'duration': 6, 'ref_audio': ''},
    ]
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    (work / 'train.yaml').write_text(yaml.safe_dump({'train_manifest': str(manifest)}), encoding='utf-8')
    (work / 'mix.json').write_text(json.dumps({
        'language_shares': {'th': {'requested': .5, 'actual': .7}, 'tl': {'requested': .5, 'actual': .3}},
        'train': {'with_control': 1, 'max_exposure': 2},
    }), encoding='utf-8')
    writer = SummaryWriter(str(logs))
    writer.add_scalar('train/loss/diff', 1.0, 0)
    writer.close()
    state = {}
    sync_once(logs, 'test_run', state)
    logged = stub_swanlab.logged[0][0]
    assert logged['data/train/rows'] == 3
    assert logged['data/train/ref_rate'] == pytest.approx(1 / 3)
    assert logged['data/train/control_rate'] == pytest.approx(1 / 3)
    assert logged['data/lang/th/duration_p50_sec'] == 7
    assert logged['data/lang/th/short_3_8s_rate'] == .5
    assert logged['data/lang/th/requested_share'] == .5
    assert logged['data/lang/th/actual_share'] == .7
    assert logged['data/lang/tl/ref_rows'] == 0
    assert state['_data'] is True
    assert sync_once(logs, 'test_run', state) == 0


def test_gate_metrics_keep_pairs_languages_and_seeds(tmp_path, stub_swanlab):
    import json
    import numpy as np
    from tensorboardX import SummaryWriter
    from voxft.train.tb_wandb_bridge import sync_once

    def row(case_id, seed, cer, *, over=False, truncation=False, audio=4., speech=.9,
            tail=.1, similarity=.9, hyp='hello', len_ratio=1.):
        return {'case_id': case_id, 'seed': seed, 'lang': 'th', 'text': 'hello', 'hyp': hyp,
                'numeric': False, 'wer': None, 'cer': cer, 'over_read': over,
                'suspected_truncation': truncation, 'audio_sec': audio, 'speech_ratio': speech,
                'tail_silence_sec': tail, 'speaker_sim': similarity, 'len_ratio': len_ratio,
                'rms_dbfs': -30.}

    work = tmp_path / 'run'
    logs = work / 'checkpoint' / 'logs'
    gate = work / 'gate_9'
    logs.mkdir(parents=True)
    gate.mkdir()
    baseline = [row('a', 43, 0.), row('b', 46, .05, audio=5., speech=.8)]
    candidate = [row('a', 43, .3, over=True, audio=10., speech=.6, tail=.2, len_ratio=1.5),
                 row('b', 46, .1, truncation=True, audio=5., speech=.8)]
    (gate / 'checkpoint.json').write_text(json.dumps({'optimizer_updates': 10}), encoding='utf-8')
    (gate / 'result.json').write_text(json.dumps({
        'decision': 'rejected', 'completed': 2, 'skipped': 3,
        'acoustic_failures': {'speech_ratio': ['th']},
        'rows': candidate, 'matched_base': baseline,
    }), encoding='utf-8')
    writer = SummaryWriter(str(logs))
    writer.add_scalar('train/loss/diff', 1.0, 9)
    writer.add_audio('valid/sample', np.zeros(480, dtype=np.float32), 0, sample_rate=48000)
    writer.close()
    state = {}
    sync_once(logs, 'test_run', state)
    assert stub_swanlab.logged[0][1] == 0
    assert stub_swanlab.logged[0][0]['valid/sample'].shape == (480,)
    logged = stub_swanlab.logged[-1][0]
    assert logged['gate/run/optimizer_updates'] == 10
    assert logged['gate/run/large_paired_cer_regressions'] == 1
    assert logged['gate/run/new_overread'] == 1
    assert logged['gate/run/new_suspected_truncation'] == 1
    assert logged['gate/run/paired_extreme_acoustics'] == 1
    assert logged['gate/content/delta/mean_cer'] == pytest.approx(.175)
    assert logged['gate/acoustic/candidate/p90_tail_silence'] == .2
    assert logged['gate/lang/th/cases'] == 2
    assert 'gate/lang/tl/cases' not in logged
    assert logged['gate/seed/s43/large_paired_cer_regressions'] == 1
    assert logged['gate/seed/s43/new_overread'] == 1
    assert logged['gate/seed/s46/large_paired_cer_regressions'] == 0
    assert state['gate:gate_9'] is True
    assert sync_once(logs, 'test_run', state) == 0


def test_swanlab_defines_metric_sections(monkeypatch):
    import sys
    from voxft.train import tb_wandb_bridge

    calls = []
    def no_run():
        raise RuntimeError('no active run')
    stub = type('SwanlabStub', (), {
        'get_run': staticmethod(no_run),
        'init': staticmethod(lambda **kwargs: object()),
        'define_metric': staticmethod(lambda key, section_name: calls.append((key, section_name))),
    })
    monkeypatch.setitem(sys.modules, 'swanlab', stub)
    monkeypatch.setenv('SWANLAB_API_KEY', 'test-key')
    assert tb_wandb_bridge._ensure_swanlab('test_run') is stub
    assert calls == list(tb_wandb_bridge._METRIC_SECTIONS.items())
