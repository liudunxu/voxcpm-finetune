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
    rank32 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32', train_rows=1881)
    assert rank32 == {**lower_lr, 'lora':{**lower_lr['lora'], 'r':32, 'alpha':32}}
    assert module.gate_steps(rank32, 'dit_original_lr5e5_rank32') == gates
    drop0 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0', train_rows=1881)
    assert drop0 == {**rank32, 'lora':{**rank32['lora'], 'dropout':0.}}
    assert module.gate_steps(drop0, 'dit_original_lr5e5_rank32_drop0') == gates
    proj = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0_proj', train_rows=1881)
    assert proj == {**drop0, 'lora':{**drop0['lora'], 'enable_proj':True}}
    assert module.gate_steps(proj, 'dit_original_lr5e5_rank32_drop0_proj') == gates
    stop2 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0_stop2', train_rows=1881)
    assert stop2 == {**drop0, 'lambdas':{**drop0['lambdas'], 'loss/stop':2.}}
    assert module.gate_steps(stop2, 'dit_original_lr5e5_rank32_drop0_stop2') == gates
    rank16 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank16_drop0', train_rows=1881)
    assert rank16 == {**drop0, 'lora':{**drop0['lora'], 'r':16, 'alpha':16}}
    assert module.gate_steps(rank16, 'dit_original_lr5e5_rank16_drop0') == gates
    seed43 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0_seed43', train_rows=1881)
    assert seed43 == drop0
    assert module.VARIANTS['dit_original_lr5e5_rank32_drop0_seed43']['training_seed'] == 43
    assert module.gate_steps(seed43, 'dit_original_lr5e5_rank32_drop0_seed43') == gates
    early20 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0_early20', train_rows=1881)
    assert early20 == drop0
    assert module.gate_steps(early20, 'dit_original_lr5e5_rank32_drop0_early20') == (9, 19)
    wd01 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0_wd01', train_rows=1881)
    assert wd01 == {**drop0, 'weight_decay':.1}
    assert module.gate_steps(wd01, 'dit_original_lr5e5_rank32_drop0_wd01') == (9, 39, 79, 118)
    micro1 = module.candidate_config(parent, tmp_path, 'dit_original_lr5e5_rank32_drop0_wd01_micro1', train_rows=1881)
    assert micro1 == {**wd01, 'batch_size':1, 'grad_accum_steps':16}
    assert micro1['batch_size'] * micro1['grad_accum_steps'] == wd01['batch_size'] * wd01['grad_accum_steps']
    assert module.gate_steps(micro1, 'dit_original_lr5e5_rank32_drop0_wd01_micro1') == (9, 39, 79, 118)


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
    module.verify_original_config_control(work, after, 'dit_original_lr5e5')
    with pytest.raises(ValueError, match='only learning rate'):
        module.verify_original_config_control(work, {**after, 'num_iters':459}, 'dit_original_lr5e5')
    with pytest.raises(ValueError, match='only learning rate'):
        module.verify_original_config_control(work, {**after, 'learning_rate':1e-4}, 'dit_original_lr5e5')
    (work / 'train.jsonl').write_text('changed row order or bytes', encoding='utf-8')
    with pytest.raises(ValueError, match='train.jsonl'):
        module.verify_original_config_control(work, after, 'dit_original_lr5e5')


@pytest.mark.parametrize('changed_file', [
    'train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'])
def test_original_rank_control_binds_all_parent_inputs(module, tmp_path, monkeypatch, changed_file):
    parent, work = tmp_path / 'r37', tmp_path / 'r38'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R37', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path('configs/lora_joint3_r37_dit_original_lr5e5_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'lora':{**before['lora'], 'r':32, 'alpha':32}, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank32'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'num_iters':459}, {'warmup_steps':45},
                   {'lora':{**after['lora'], 'alpha':64}}, {'lora':{**after['lora'], 'enable_lm':True}}):
        with pytest.raises(ValueError, match='only DiT rank/alpha'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / changed_file).write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match=changed_file):
        module.verify_original_config_control(work, after, variant)


@pytest.mark.parametrize('changed_file', [
    'train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'])
def test_dropout_control_binds_all_parent_inputs(module, tmp_path, monkeypatch, changed_file):
    parent, work = tmp_path / 'r38', tmp_path / 'r39'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R38', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path('configs/lora_joint3_r38_dit_original_lr5e5_rank32_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'lora':{**before['lora'], 'dropout':0.}, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank32_drop0'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'num_iters':459}, {'warmup_steps':45},
                   {'lora':{**after['lora'], 'r':64}}, {'lora':{**after['lora'], 'enable_lm':True}},
                   {'lora':{**after['lora'], 'dropout':.05}}):
        with pytest.raises(ValueError, match='only DiT dropout'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / changed_file).write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match=changed_file):
        module.verify_original_config_control(work, after, variant)


@pytest.mark.parametrize('changed_file', [
    'train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'])
def test_projection_control_binds_r39_and_only_changes_enable_proj(
        module, tmp_path, monkeypatch, changed_file):
    parent, work = tmp_path / 'r39', tmp_path / 'r40'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R39', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path(
        'configs/lora_joint3_r39_dit_original_lr5e5_rank32_drop0_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'lora':{**before['lora'], 'enable_proj':True}, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank32_drop0_proj'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'num_iters':459},
                   {'lora':{**after['lora'], 'enable_lm':True}},
                   {'lora':{**after['lora'], 'enable_proj':False}}):
        with pytest.raises(ValueError, match='only enable_proj'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / changed_file).write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match=changed_file):
        module.verify_original_config_control(work, after, variant)


@pytest.mark.parametrize('changed_file', [
    'train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'])
def test_stop_weight_control_binds_r39_and_only_changes_stop_loss(
        module, tmp_path, monkeypatch, changed_file):
    parent, work = tmp_path / 'r39', tmp_path / 'r41'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R39', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path(
        'configs/lora_joint3_r39_dit_original_lr5e5_rank32_drop0_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'lambdas':{**before['lambdas'], 'loss/stop':2.}, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank32_drop0_stop2'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'num_iters':459},
                   {'lora':{**after['lora'], 'enable_lm':True}},
                   {'lambdas':{**after['lambdas'], 'loss/stop':1.}}):
        with pytest.raises(ValueError, match='only loss/stop'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / changed_file).write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match=changed_file):
        module.verify_original_config_control(work, after, variant)


def test_rank16_control_binds_r39_and_only_changes_rank(module, tmp_path, monkeypatch):
    parent, work = tmp_path / 'r39', tmp_path / 'r42'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R39', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path(
        'configs/lora_joint3_r39_dit_original_lr5e5_rank32_drop0_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'lora':{**before['lora'], 'r':16, 'alpha':16}, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank16_drop0'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'lambdas':{'loss/diff':1., 'loss/stop':2.}},
                   {'lora':{**after['lora'], 'alpha':32}}):
        with pytest.raises(ValueError, match='only DiT rank/alpha'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / 'train.jsonl').write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match='train.jsonl'):
        module.verify_original_config_control(work, after, variant)


@pytest.mark.parametrize('variant', ['dit_original_lr5e5_rank32_drop0_seed43',
                                     'dit_original_lr5e5_rank32_drop0_early20'])
def test_seed_or_early_budget_control_keeps_config_and_sets_official_accelerator_seed(
        module, tmp_path, monkeypatch, variant):
    parent, work = tmp_path / 'r39', tmp_path / 'candidate'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R39', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path(
        'configs/lora_joint3_r39_dit_original_lr5e5_rank32_drop0_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'save_path':str(work)}
    module.verify_original_config_control(work, after, variant)
    with pytest.raises(ValueError, match='identical config'):
        module.verify_original_config_control(work, {**after, 'learning_rate':1e-4}, variant)
    calls = []
    def accelerator(**kwargs):
        calls.append(kwargs)
        return 'accelerator'
    assert module._seeded_accelerator(accelerator, 43)(amp=True) == 'accelerator'
    assert calls == [{'amp':True, 'seed':43}]


def test_weight_decay_control_binds_r39_and_only_changes_regularization(module, tmp_path, monkeypatch):
    parent, work = tmp_path / 'r39', tmp_path / 'r45'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R39', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path(
        'configs/lora_joint3_r39_dit_original_lr5e5_rank32_drop0_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'weight_decay':.1, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank32_drop0_wd01'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'weight_decay':.01}, {'num_iters':20}):
        with pytest.raises(ValueError, match='only weight_decay'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / 'train.jsonl').write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match='train.jsonl'):
        module.verify_original_config_control(work, after, variant)


def test_microbatch_control_binds_r45_and_preserves_effective_batch(module, tmp_path, monkeypatch):
    parent, work = tmp_path / 'r45', tmp_path / 'r46'
    parent.mkdir()
    work.mkdir()
    monkeypatch.setattr(module, 'R45', parent)
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json', 'mix.json', 'jobs.json', 'base_items.json'):
        (parent / name).write_text('frozen identical bytes', encoding='utf-8')
        (work / name).write_bytes((parent / name).read_bytes())
    before = yaml.safe_load(Path(
        'configs/lora_joint3_r45_dit_original_lr5e5_rank32_drop0_wd01_staged.yaml').read_text())
    (parent / 'train.yaml').write_text(yaml.safe_dump(before), encoding='utf-8')
    after = {**before, 'batch_size':1, 'grad_accum_steps':16, 'save_path':str(work)}
    variant = 'dit_original_lr5e5_rank32_drop0_wd01_micro1'
    module.verify_original_config_control(work, after, variant)
    for change in ({'learning_rate':1e-4}, {'batch_size':2}, {'grad_accum_steps':8}, {'num_iters':20}):
        with pytest.raises(ValueError, match='only microbatch'):
            module.verify_original_config_control(work, {**after, **change}, variant)
    (work / 'train.jsonl').write_text('changed bytes', encoding='utf-8')
    with pytest.raises(ValueError, match='train.jsonl'):
        module.verify_original_config_control(work, after, variant)


@pytest.mark.parametrize('duration,ratio,numeric,expected', [
    (10.88,.7901,False,True), (9.6,.7901,False,False),
    (10.88,.854,False,False), (10.88,.7901,True,False)])
def test_extreme_pair_guard_does_not_depend_on_mean_cer(module, duration, ratio, numeric, expected):
    old = {'case_id':'th1844', 'seed':43, 'lang':'th', 'numeric':numeric,
           'audio_sec':4.8, 'speech_ratio':.954, 'cer':0., 'f0_std_st':0., 'wer':None}
    row = {**old, 'audio_sec':duration, 'speech_ratio':ratio, 'cer':.0253}
    base = {module.prior.key(old):old}
    assert 'paired_extreme_acoustic' not in module.acoustic_gate([row], base)
    result = module.acoustic_gate([row], base, paired_extremes=True)
    assert bool(result.get('paired_extreme_acoustic')) == expected


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
    ('extreme','stage_acoustic_regression',5),
    ('overread','new_suspected_overread',1), ('large_cer','two_large_paired_cer_regressions',2)])
def test_new_pilot_observes_cer_but_stops_content_and_acoustic_failures(
        module, tmp_path, monkeypatch, failure, expected, count):
    directory = tmp_path / 'gate_39'
    directory.mkdir()
    module.save(tmp_path / 'plan.json', {'gates':[39,79,118], 'asr_path':'asr',
        'speaker_model_path':'speaker', 'base_path':'base', 'pilot_mean_cer_observe_only':True,
        'paired_extreme_acoustics':failure == 'extreme'})
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
        if failure == 'extreme' and seeds[0] == 43:
            row.update(audio_sec=11., speech_ratio=.79, cer=.08)
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
@pytest.mark.parametrize('step,updates,gates,max_updates', [
    (150,151,(150,300,459),459), (39,40,(39,79,118),118),
    (79,80,(39,79,118),118), (118,118,(39,79,118),118), (19,20,(9,19),118)])
def test_gate_waits_on_complete_snapshot_and_rejects_before_more_updates(
        module, tmp_path, monkeypatch, decision, step, updates, gates, max_updates):
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
            module.checkpoint_gate(tmp_path, step, optimizer, gates=gates, max_updates=max_updates)
    else:
        module.checkpoint_gate(tmp_path, step, optimizer, gates=gates, max_updates=max_updates)
    assert sleeps == [True]
    assert (snapshot / 'weights').read_bytes() == b'unchanged'
    assert optimizer.state[0]['step'].item() == updates


def test_early_budget_stops_after_final_pass_without_changing_lr_schedule(module, tmp_path, monkeypatch):
    module.save(tmp_path / 'train.yaml', {'num_iters':118})
    module.save(tmp_path / 'plan.json', {'variant':'dit_original_lr5e5_rank32_drop0_early20',
        'training_seed':42, 'stop_after_final_gate':True})
    calls = []

    def accelerator(**kwargs):
        return kwargs

    def save_checkpoint(*args):
        pass

    fake = SimpleNamespace(Accelerator=accelerator, save_checkpoint=save_checkpoint)

    def official_train(**cfg):
        assert cfg == {'num_iters':118}
        for step in (19, 20):
            calls.append(step)
            fake.save_checkpoint(None, None, None, None, step)

    fake.train = official_train
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda official: None))
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location', lambda *args: spec)
    monkeypatch.setattr(module.importlib.util, 'module_from_spec', lambda spec: fake)
    updates = []
    monkeypatch.setattr(module, 'checkpoint_gate',
        lambda work, step, optimizer, **kwargs: updates.append(kwargs['max_updates']))
    module.train(tmp_path)
    assert calls == [19]
    assert updates == [118]
    assert fake.save_checkpoint is save_checkpoint
    assert fake.Accelerator is accelerator


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
