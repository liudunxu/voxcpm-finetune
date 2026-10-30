import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('staged_candidate_run')


def test_rank_experiment_preserves_learning_schedule(module, tmp_path):
    parent = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'configs/lora_joint3_r31_granary_consensus_459.yaml').read_text())
    result = module.rank_config(parent, tmp_path, 32)
    assert result['lora'] == {**parent['lora'], 'r': 32, 'alpha': 32}
    assert parent['lora']['r'] == 64
    assert {k for k in result if result[k] != parent[k]} == {
        'lora', 'train_manifest', 'val_manifest', 'save_path', 'tensorboard', 'save_interval'}
    with pytest.raises(ValueError):
        module.rank_config(parent, tmp_path, 16)


@pytest.mark.parametrize('decision', ['pilot_pass', 'rejected'])
def test_gate_waits_on_complete_snapshot_and_rejects_before_more_updates(module, tmp_path, monkeypatch, decision):
    snapshot = tmp_path / 'checkpoint/step_0000150'
    snapshot.mkdir(parents=True)
    (snapshot / 'weights').write_bytes(b'unchanged')
    optimizer = SimpleNamespace(state={0: {'step': torch.tensor(151)}})
    sleeps = []

    def decide(_):
        assert (tmp_path / 'gate_150/ready.json').exists()
        assert module.read(tmp_path / 'gate_150/checkpoint.json')['optimizer_updates'] == 151
        sleeps.append(True)
        module.save(tmp_path / 'gate_150/decision.json', {'decision': decision, 'reason': 'failed_example'})

    monkeypatch.setattr(module.time, 'sleep', decide)
    if decision == 'rejected':
        with pytest.raises(module.GateRejected):
            module.checkpoint_gate(tmp_path, 150, optimizer)
    else:
        module.checkpoint_gate(tmp_path, 150, optimizer)
    assert sleeps == [True]
    assert (snapshot / 'weights').read_bytes() == b'unchanged'
    assert optimizer.state[0]['step'].item() == 151


def test_wrong_optimizer_step_and_existing_run_fail_closed(module, tmp_path):
    with pytest.raises(ValueError, match='Optimizer'):
        module.checkpoint_gate(tmp_path, 150, SimpleNamespace(state={0: {'step': 150}}))
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
