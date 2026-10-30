import importlib
from contextlib import nullcontext
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


def test_paired_candidate_recomputes_schedule_and_retains_optimizer_recipe(module, tmp_path):
    parent = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'configs/lora_joint3_r31_granary_consensus_459.yaml').read_text())
    cfg = module.candidate_config(parent, tmp_path, 'dit_paired', train_rows=1881)
    assert cfg['num_iters'] == cfg['max_steps'] == 118
    assert cfg['warmup_steps'] == 12
    assert cfg['lora'] == {**parent['lora'], 'enable_lm':False}
    for key in ('batch_size', 'grad_accum_steps', 'learning_rate', 'weight_decay', 'max_grad_norm'):
        assert cfg[key] == parent[key]
    gates = module.gate_steps(cfg, 'dit_paired')
    assert gates == (39, 79, 118)
    # Exercise the official save predicate: every intermediate gate must be reachable.
    callbacks = [s for s in range(cfg['num_iters'])
                 if s % cfg['save_interval'] == 0 or s == cfg['num_iters'] - 1] + [cfg['num_iters']]
    assert [s for s in callbacks if s in gates] == list(gates)
    with pytest.raises(ValueError, match='1881'):
        module.candidate_config(parent, tmp_path, 'dit_paired', train_rows=5015)
    assert module.gate_steps(parent, 'dit_only') == (150, 300, 459)
    original = module.candidate_config(parent, tmp_path, 'dit_original', train_rows=1881)
    assert original == cfg
    assert module.gate_steps(original, 'dit_original') == gates
    lower_lr = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5', train_rows=1881)
    assert {k for k in original if original[k] != lower_lr[k]} == {'learning_rate'}
    assert lower_lr['learning_rate'] == 5e-5
    assert module.gate_steps(lower_lr, 'dit_original_lr5e5') == gates


def test_original_lr_control_binds_parent_data_and_single_config_change(module, tmp_path, monkeypatch):
    parent, work = tmp_path / 'r36', tmp_path / 'r37'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R36', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = {'learning_rate':1e-4, 'num_iters':118, 'warmup_steps':12, 'save_path':str(parent)}
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'learning_rate':5e-5, 'save_path':str(work)}
    module.verify_original_lr_control(work, after)
    with pytest.raises(ValueError, match='only learning rate'):
        module.verify_original_lr_control(work, {**after, 'num_iters':459})
    with pytest.raises(ValueError, match='only learning rate'):
        module.verify_original_lr_control(work, {**after, 'learning_rate':1e-4})
    (work / 'train.jsonl').write_text('changed row order or bytes', encoding='utf-8')
    with pytest.raises(ValueError, match='train.jsonl'):
        module.verify_original_lr_control(work, after)


def test_original_control_only_reverses_frozen_paths_and_preserves_order(module):
    first = {'audio':'raw_a','ref_audio':'raw_b','text':'same','origin_audio':'origin_a',
             'ref_origin_audio':'origin_b','speaker':'person_a','source_id':'source','duration':5}
    second = {'audio':'raw_c','text':'second','origin_audio':'origin_c','source_id':'source','duration':4}
    mapping = {'raw_a':{'selected':'vocal_a'}, 'raw_b':{'selected':'vocal_b'}, 'raw_c':{'selected':'raw_c'}}
    derived = {**first, 'audio':'vocal_a','ref_audio':'vocal_b'}
    rows = [derived,second,derived]
    assert module.original_control_rows(rows,mapping,[first,second]) == [first,second,first]
    assert rows[0] == derived
    with pytest.raises(ValueError,match='metadata'):
        module.original_control_rows([{**derived,'text':'changed'}],mapping,[first,second])
    with pytest.raises(ValueError,match='pairing'):
        module.original_control_rows([{**derived,'ref_audio':'raw_c'}],mapping,[first,second])
    with pytest.raises(KeyError):
        module.original_control_rows([{**derived,'audio':'unknown'}],mapping,[first,second])


def test_paired_risks_use_all_seeds_once_before_old_guards(module):
    cases = module.frozen._read_manifest(Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
    selected = [c for c in cases if c['sentence_id'] in ('fresh_tl_1913', 'fresh_id_1808')]
    jobs = [{'case':c, 'seed':s, 'stage':'guard'} for c in selected for s in range(42,47)]
    canary = next(c for c in cases if c['case_id'] == 'fresh_th_1844_control')
    jobs = [{'case':canary, 'seed':s, 'stage':'new_risk'} for s in (44,43,45,42,46)] + jobs
    ordered = module.prioritize_tail_risk(jobs)
    assert ordered[:5] == jobs[:5]
    assert [j['seed'] for j in ordered[5:10]] == [43,42,44,45,46]
    assert all(j['case']['case_id'] == 'fresh_tl_1913_control' for j in ordered[5:10])
    assert module.end_of_stage(ordered, 4) and module.end_of_stage(ordered, 9)
    assert {(j['case']['case_id'],j['seed']) for j in ordered} == {
        (j['case']['case_id'],j['seed']) for j in jobs}
    assert len(ordered) == len(jobs)
    with pytest.raises(ValueError, match='five'):
        module.prioritize_tail_risk([j for j in jobs if not (
            j['case']['case_id'] == 'fresh_tl_1913_control' and j['seed'] == 43)])


@pytest.mark.parametrize('failure,expected,count', [
    ('small_cer',None,5), ('tail','stage_acoustic_regression',5),
    ('overread','new_suspected_overread',1), ('large_cer','two_large_paired_cer_regressions',2)])
def test_new_pilot_observes_cer_but_stops_content_and_acoustic_failures(
        module, tmp_path, monkeypatch, failure, expected, count):
    directory = tmp_path / 'gate_39'
    directory.mkdir()
    module.save(tmp_path / 'plan.json', {'gates':[39,79,118], 'asr_path':'asr',
        'speaker_model_path':'speaker', 'base_path':'base', 'pilot_mean_cer_observe_only':True})
    module.save(directory / 'checkpoint.json', {'target':'candidate', 'inputs_sha256':{}})
    case = {'case_id':'a', 'lang':'th', 'text':'text'}
    jobs = [{'case':case, 'seed':seed, 'stage':'new_risk'} for seed in range(42,47)]
    base = [{**case, 'seed':j['seed'], 'cer':0., 'hyp':'text', 'over_read':False,
             'rms_dbfs':-20, 'speech_ratio':.95, 'audio_sec':5., 'tail_silence_sec':.1,
             'numeric':False, 'f0_std_st':0., 'wer':None} for j in jobs]
    module.save(tmp_path / 'jobs.json', jobs)
    module.save(tmp_path / 'base_items.json', base)
    monkeypatch.setattr(module.frozen, 'runtime_environment', lambda _:None)
    monkeypatch.setattr(module.prior.fast, 'cached_services', lambda _:nullcontext())
    monkeypatch.setattr(module.prior.baseline, 'gpu_services', lambda *_:{})
    monkeypatch.setattr(module.prior, 'verify', lambda *_:None)
    monkeypatch.setattr(module.prior.review, 'level_metrics', lambda _: {})
    monkeypatch.setattr(module.frozen.evaluation, 'EVAL_DIR', tmp_path)
    monkeypatch.setattr(module.frozen.evaluation.infer, 'CHECKPOINT_DIR', tmp_path)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _:SimpleNamespace(free=8*1024**3))

    def evaluate(*args, seeds, **kwargs):
        row = dict(base[seeds[0]-42])
        row.update(cer=.0506 if seeds[0] == 43 else 0.)
        if failure == 'tail': row['tail_silence_sec'] = 1.02
        if failure == 'overread': row['over_read'] = True
        if failure == 'large_cer': row['cer'] = .3
        wav = directory / f'{seeds[0]}.wav'
        wav.write_bytes(b'preserved failed or successful output')
        row['wav'] = str(wav)
        return {'items':[row]}

    monkeypatch.setattr(module.frozen.evaluation, 'evaluate', evaluate)
    module.evaluate_gate(tmp_path, 39)
    result = module.read(directory / 'result.json')
    assert result['reason'] == expected
    assert result['completed'] == count
    assert len(list((directory / 'claims').glob('*'))) == count
    assert len(list(directory.glob('*.wav'))) == count
    assert result['decision'] == ('rejected' if expected else 'pilot_pass')
    assert result['cer_observations']['noise_threshold'] == .05


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
@pytest.mark.parametrize('step,updates,gates', [(150,151,(150,300,459)), (39,40,(39,79,118)),
                                               (79,80,(39,79,118)), (118,118,(39,79,118))])
def test_gate_waits_on_complete_snapshot_and_rejects_before_more_updates(
        module, tmp_path, monkeypatch, decision, step, updates, gates):
    snapshot = tmp_path / f'checkpoint/step_{step:07d}'
    snapshot.mkdir(parents=True)
    (snapshot / 'weights').write_bytes(b'unchanged')
    optimizer = SimpleNamespace(state={0: {'step': torch.tensor(updates)}})
    sleeps = []

    def decide(_):
        assert (tmp_path / f'gate_{step}/ready.json').exists()
        assert module.read(tmp_path / f'gate_{step}/checkpoint.json')['optimizer_updates'] == updates
        sleeps.append(True)
        module.save(tmp_path / f'gate_{step}/decision.json', {'decision': decision, 'reason': 'failed_example'})

    monkeypatch.setattr(module.time, 'sleep', decide)
    if decision == 'rejected':
        with pytest.raises(module.GateRejected):
            module.checkpoint_gate(tmp_path, step, optimizer, gates=gates)
    else:
        module.checkpoint_gate(tmp_path, step, optimizer, gates=gates)
    assert sleeps == [True]
    assert (snapshot / 'weights').read_bytes() == b'unchanged'
    assert optimizer.state[0]['step'].item() == updates


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
