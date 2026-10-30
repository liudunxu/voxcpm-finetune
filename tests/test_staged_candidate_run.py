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
    result = module.candidate_config(parent, tmp_path, 'rank32')
    assert result['lora'] == {**parent['lora'], 'r': 32, 'alpha': 32}
    assert parent['lora']['r'] == 64
    assert {k for k in result if result[k] != parent[k]} == {
        'lora', 'train_manifest', 'val_manifest', 'save_path', 'tensorboard', 'save_interval'}
    with pytest.raises(ValueError):
        module.candidate_config(parent, tmp_path, 'unknown')
    dit = module.candidate_config(parent, tmp_path, 'dit_only')
    assert dit['lora'] == {**parent['lora'], 'enable_lm':False}
    assert dit['learning_rate'] == parent['learning_rate']
    assert dit['num_iters'] == dit['max_steps'] == 459
    lower_lr = module.candidate_config(parent, tmp_path, 'dit_lr5e5')
    assert {k for k in dit if dit[k] != lower_lr[k]} == {'learning_rate'}
    assert lower_lr['learning_rate'] == 5e-5


def test_new_failure_canary_precedes_guards_without_moving_training_inputs(module):
    def case(cid, lang):
        return {'case_id':cid, 'lang':lang, 'text':'unchanged'}
    old = [{'case':case(f'old{i}', 'th'), 'seed':42, 'stage':'risk'} for i in range(10)]
    old += [{'case':case(f'guard{i}', lang), 'seed':42, 'stage':'guard'}
            for i,lang in enumerate(['id']*10 + ['th']*10 + ['tl']*10)]
    old_base = [{**j['case'], 'seed':j['seed']} for j in old]
    canary = case('fresh_th_1844_control', 'th')
    baseline = [{**canary,'seed':seed} for seed in range(42,47)]
    jobs, matched = module.add_failure_canary(old, old_base, [canary], baseline)
    assert jobs[5:] == old and matched[5:] == old_base
    assert [j['seed'] for j in jobs[:5]] == [44,43,45,42,46]
    assert [i for i in range(45) if module.end_of_stage(jobs,i)] == [4,14,24,34,44]
    assert [i for i in range(40) if module.end_of_stage(old,i)] == [9,19,29,39]
    with pytest.raises(ValueError, match='Duplicate'):
        module.add_failure_canary(jobs, matched, [canary], baseline)


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


def test_acoustic_gate_rejects_without_content_regression(module):
    def row(i, lang):
        return {'case_id':str(i), 'seed':42, 'lang':lang, 'cer':.02, 'wer':None,
                'numeric':False, 'f0_std_st':0., 'tail_silence_sec':.1, 'speech_ratio':.99,
                'audio_sec':5., 'text':'unchanged'}
    baseline = [row(i, 'id' if i < 100 else 'th') for i in range(105)]
    index = {module.prior.key(r):r for r in baseline}
    bad_tail = [{**r, 'tail_silence_sec':1.2 if r['lang']=='th' else .1} for r in baseline]
    failed = module.acoustic_gate(bad_tail, index)
    assert any(message.startswith('th ') for message in failed['tail_silence'])
    assert not any(message.startswith('overall ') for message in failed['tail_silence'])
    assert 'speech_ratio' in module.acoustic_gate([{**r,'speech_ratio':.85} for r in baseline], index)
    assert 'duration_inflation' in module.acoustic_gate([{**r,'audio_sec':8.} for r in baseline], index)
    # User relaxed only relative tail changes; absolute under .5 remains allowed.
    assert module.acoustic_gate([{**r,'tail_silence_sec':.34} for r in baseline], index) == {}
