from collections import Counter
from contextlib import nullcontext
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('expanded_candidate_eval')


def test_selection_balances_texts_and_keeps_risk_and_pilot_without_score_selection(module):
    cases = module.frozen._read_manifest(Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
    guards = {'fresh_id_1808', 'fresh_th_1689', 'fresh_tl_1913'}
    pilot = [{'case':c, 'stage':'guard'} for c in cases if c['sentence_id'] in guards]
    jobs = module.select_jobs(cases, pilot)
    assert jobs == module.select_jobs(list(reversed(cases)), list(reversed(pilot)))
    assert Counter(j['case']['lang'] for j in jobs) == {'id':100, 'th':100, 'tl':100}
    assert Counter(j['stage'] for j in jobs) == {s:30 for s in range(10)}
    selected = {j['case']['sentence_id'] for j in jobs}
    assert set(module.prior.SENTENCES) | guards <= selected
    assert {j['case']['sentence_id'] for j in jobs if j['stage'] == 0} == guards
    assert all('Ã' not in j['case']['text'] and 'Å' not in j['case']['text'] for j in jobs)
    assert Counter(j['seed'] for j in jobs) == dict.fromkeys(range(42,47),60)


def test_published_results_are_complete_and_cannot_be_replaced(module, tmp_path):
    path = tmp_path / 'result.json'
    module.publish(path, {'text':'ภาษาไทย'})
    original = path.read_bytes()
    assert module.read(path)['text'] == 'ภาษาไทย'
    with pytest.raises(FileExistsError):
        module.publish(path, {'text':'replacement'})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.pending'))


def test_control_update_never_exposes_partial_json(module, tmp_path, monkeypatch):
    path = tmp_path / 'control.json'
    module.update(path, {'stop':False, 'stage':1})
    original_save = module.save

    def writing(pending, value):
        # Even while the new file is only half written, a worker can read the old one.
        pending.write_text('{"stop":', encoding='utf-8')
        assert module.read(path) == {'stop':False, 'stage':1}
        pending.unlink()
        original_save(pending, value)

    monkeypatch.setattr(module, 'save', writing)
    module.update(path, {'stop':True, 'stage':1})
    assert module.read(path) == {'stop':True, 'stage':1}
    assert not list(tmp_path.glob('*.pending'))


def test_worker_finishes_in_flight_result_and_stops_before_another_claim(module, tmp_path, monkeypatch):
    plan = {'asr_path':'asr', 'speaker_model_path':'speaker', 'base_path':'base',
            'targets':{'candidate':'weights'}}
    jobs = [{'id':str(i), 'case':{'case_id':str(i)}, 'seed':42, 'stage':0} for i in range(2)]
    for directory in ('worker0', 'items', 'claims'):
        (tmp_path / directory).mkdir()
    module.save(tmp_path / 'jobs.json', jobs)
    module.save(tmp_path / 'reused.json', [])
    module.save(tmp_path / 'control.json', {'stop':False, 'stage':0})
    monkeypatch.setattr(module.prior.strength, 'check_inputs', lambda _:plan)
    monkeypatch.setattr(module.frozen, 'runtime_environment', lambda _:None)
    monkeypatch.setattr(module.prior.fast, 'cached_services', lambda _:nullcontext())
    monkeypatch.setattr(module.prior.baseline, 'gpu_services', lambda *_:{})
    monkeypatch.setattr(module.prior, 'verify', lambda *_:None)
    monkeypatch.setattr(module.prior.review, 'level_metrics', lambda _: {})
    monkeypatch.setattr(module.frozen.evaluation, 'EVAL_DIR', tmp_path)
    monkeypatch.setattr(module.frozen.evaluation.infer, 'CHECKPOINT_DIR', tmp_path)
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(args)
        module.update(tmp_path / 'control.json', {'stop':True, 'stage':0})
        wav = tmp_path / 'one.wav'
        wav.write_bytes(b'preserve failed audio')
        return {'items':[{'case_id':'0', 'seed':42, 'wav':str(wav)}]}

    monkeypatch.setattr(module.frozen.evaluation, 'evaluate', evaluate)
    module.worker(tmp_path, 0)
    assert len(calls) == 1
    assert {p.stem for p in (tmp_path / 'claims').glob('*.json')} == {'0'}
    assert {p.stem for p in (tmp_path / 'items').glob('*.json')} == {'0'}
    assert (tmp_path / 'one.wav').read_bytes() == b'preserve failed audio'


def test_half_worker_verifies_scaling_and_uses_the_same_strength(module, tmp_path, monkeypatch):
    plan = {'asr_path':'asr', 'speaker_model_path':'speaker', 'base_path':'base',
            'targets':{'candidate':'weights'}, 'lora_strength':.5}
    jobs = [{'id':'0', 'case':{'case_id':'0'}, 'seed':42, 'stage':0}]
    for directory in ('worker0', 'items', 'claims'):
        (tmp_path / directory).mkdir()
    module.save(tmp_path / 'jobs.json', jobs)
    module.save(tmp_path / 'reused.json', [])
    module.save(tmp_path / 'control.json', {'stop':False, 'stage':0})
    monkeypatch.setattr(module.prior.strength, 'check_inputs', lambda _:plan)
    monkeypatch.setattr(module.prior.strength, 'check_scaling', lambda _: {'restored':True})
    monkeypatch.setattr(module.frozen, 'runtime_environment', lambda _:None)
    monkeypatch.setattr(module.prior.fast, 'cached_services', lambda _:nullcontext())
    monkeypatch.setattr(module.prior.baseline, 'gpu_services', lambda *_:{})
    monkeypatch.setattr(module.frozen.evaluation.infer, 'get_model', lambda *_:object())
    monkeypatch.setattr(module.frozen.evaluation, 'EVAL_DIR', tmp_path)
    monkeypatch.setattr(module.frozen.evaluation.infer, 'CHECKPOINT_DIR', tmp_path)
    monkeypatch.setattr(module.prior.review, 'level_metrics', lambda _: {})
    calls = []
    monkeypatch.setattr(module.prior, 'verify', lambda *args: calls.append(args[4]))
    def evaluate(*args, **kwargs):
        calls.append(kwargs['lora_strength'])
        module.update(tmp_path / 'control.json', {'stop':True, 'stage':0})
        wav = tmp_path / 'half.wav'; wav.write_bytes(b'half')
        return {'items':[{'case_id':'0', 'seed':42, 'wav':str(wav)}]}
    monkeypatch.setattr(module.frozen.evaluation, 'evaluate', evaluate)
    module.worker(tmp_path, 0)
    assert calls == [.5, .5]
    assert module.read(tmp_path / 'worker0/scaling.json') == {'restored':True}


def test_expansion_cannot_skip_acoustic_failure(module):
    row = {'case_id':'a','seed':42,'lang':'th','cer':0.,'hyp':'text','text':'text',
           'over_read':False,'rms_dbfs':-20,'speech_ratio':.95,'audio_sec':5.,
           'tail_silence_sec':.1,'numeric':False,'f0_std_st':0,'wer':None}
    candidate = {**row, 'tail_silence_sec':1.2}
    reason, failures = module.decision([candidate], {('a',42):row}, stage_end=True)
    assert reason == 'stage_acoustic_regression'
    assert failures['tail_silence']


def test_diagnostic_preserves_rejection_and_does_not_admit_other_failure(module):
    result = {'decision':'rejected', 'gates':{'150':{
        'reason':'stage_mean_cer_regression_th', 'completed':5}}}
    status = {'stage':'rejected'}
    assert module.parent_gate('r34_diagnostic', status, result) == (150, 151, 5)
    with pytest.raises(ValueError):
        module.parent_gate('r33', status, result)
    result['gates']['150']['reason'] = 'new_suspected_overread'
    with pytest.raises(ValueError):
        module.parent_gate('r34_diagnostic', status, result)
    assert module.terminal_decision(None, True) == 'diagnostic_complete'
    assert module.terminal_decision('near_silence', True) == 'diagnostic_stopped'
    assert module.terminal_decision(None, False) == 'development_pass'
    assert module.parent_gate('r33_early_diagnostic', {'stage':'pilot_complete'},
        {'decision':'development_pilot_pass'}) == (150, 151, 40)


def test_diagnostic_keeps_severe_and_acoustic_stops_but_records_small_cer_changes(module):
    base = [{'case_id':'a', 'seed':s, 'lang':'th', 'cer':0., 'hyp':'text', 'text':'text',
             'over_read':False, 'rms_dbfs':-20, 'speech_ratio':.95, 'audio_sec':5.,
             'tail_silence_sec':.1, 'numeric':False, 'f0_std_st':0, 'wer':None}
            for s in range(42,47)]
    lookup = {module.prior.key(r):r for r in base}
    rows = [{**r, 'cer':.0506 if r['seed'] == 43 else 0.} for r in base]
    assert module.decision(rows, lookup, stage_end=True)[0] == 'stage_mean_cer_regression_th'
    assert module.decision(rows, lookup, stage_end=True, diagnostic=True) == (None, {})
    observed = module.cer_observations(rows, lookup)
    assert observed['noise_threshold'] == .05
    assert not observed['classification']['red'] and observed['classification']['noise']
    rows[-1]['over_read'] = True
    assert module.decision(rows, lookup, diagnostic=True)[0] == 'new_suspected_overread'
    rows[-1]['over_read'] = False
    for row in rows:
        row['tail_silence_sec'] = 1.2
    assert module.decision(rows, lookup, stage_end=True, diagnostic=True)[0] == 'stage_acoustic_regression'
    # Red CER observation still cannot certify or promote a diagnostic run.
    for row in rows:
        row['cer'] = .08
        row['tail_silence_sec'] = .1
    assert module.cer_observations(rows, lookup)['classification']['red']


def test_diagnostic_reuses_all_five_r34_outputs_with_same_preselected_coverage(module):
    cases = module.frozen._read_manifest(Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
    guards = {'fresh_id_1808', 'fresh_th_1689', 'fresh_tl_1913'}
    pilot = [{'case':c, 'stage':'guard'} for c in cases if c['sentence_id'] in guards]
    canary = next(c for c in cases if c['case_id'] == 'fresh_th_1844_control')
    before = module.select_jobs(cases, pilot)
    after = module.select_jobs(cases, [{'case':canary, 'stage':'new_risk'}] + pilot)
    assert after == before
    assert {j['seed'] for j in after if j['case']['case_id'] == canary['case_id']} == set(range(42,47))
    early = module.select_jobs(cases, pilot, canary_first=True)
    assert {j['id'] for j in early} == {j['id'] for j in before}
    assert {j['case']['sentence_id'] for j in early if j['stage'] == 0} == {
        'fresh_id_1808', 'fresh_tl_1913', 'fresh_th_1844'}
    assert Counter(j['stage'] for j in early) == {s:30 for s in range(10)}


def test_r39_diagnostic_requires_exact_stopped_parent_and_reuses_all_45(module):
    gates = {s:{'decision':'pilot_pass', 'reason':None, 'completed':45, 'skipped':0}
             for s in ('39', '79')}
    gates['118'] = {'decision':'rejected', 'reason':'new_suspected_overread',
                    'completed':3, 'skipped':42}
    result = {'decision':'rejected', 'gates':gates}
    assert module.parent_gate('r39_early_diagnostic', {'stage':'rejected'}, result) == (79, 80, 45)
    assert module.parent_gate('r39_u40_diagnostic', {'stage':'rejected'}, result) == (39, 40, 45)
    for invalid in ({'stage':'pilot_complete'}, {'stage':'training'}):
        with pytest.raises(ValueError, match='r39'):
            module.parent_gate('r39_early_diagnostic', invalid, result)
    gates['79']['reason'] = 'stage_acoustic_regression'
    with pytest.raises(ValueError, match='r39'):
        module.parent_gate('r39_early_diagnostic', {'stage':'rejected'}, result)
    with pytest.raises(ValueError, match='r39'):
        module.parent_gate('r39_u40_diagnostic', {'stage':'rejected'}, result)
    cases = module.frozen._read_manifest(Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
    guard_ids = {'fresh_id_1808', 'fresh_th_1689', 'fresh_tl_1913'}
    pilot = [{'case':c, 'seed':s, 'stage':'guard' if c['sentence_id'] in guard_ids else 'risk'}
             for c in cases if c['sentence_id'] in guard_ids or c['case_id'] in {
                 'fresh_th_1844_control', 'fresh_th_1983_control', 'fresh_th_1994_control'}
             for s in module.frozen.SEEDS]
    assert len(pilot) == 45
    jobs = module.select_jobs(cases, pilot, canary_first=True)
    assert {(p['case']['case_id'], p['seed']) for p in pilot} <= {
        (j['case']['case_id'], j['seed']) for j in jobs}
    assert len(jobs) - len(pilot) == 255
    assert {j['case']['sentence_id'] for j in jobs if j['stage'] == 0} == {
        'fresh_id_1808', 'fresh_th_1844', 'fresh_tl_1913'}


def test_u40_fronts_new_id_failure_without_changing_coverage_inputs_or_seed_budget(module):
    cases = module.frozen._read_manifest(Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
    guards = {'fresh_id_1808', 'fresh_th_1689', 'fresh_tl_1913'}
    pilot = [{'case':c, 'stage':'guard'} for c in cases if c['sentence_id'] in guards]
    previous = module.select_jobs(cases, pilot, canary_first=True)
    jobs = module.select_jobs(cases, pilot, canary_first=True, id_canary_first=True)
    assert jobs == module.select_jobs(list(reversed(cases)), pilot, canary_first=True,
                                     id_canary_first=True)
    assert len(jobs) == len({j['id'] for j in jobs}) == 300
    assert {j['id']:(j['case'], j['seed']) for j in jobs} == {
        j['id']:(j['case'], j['seed']) for j in previous}
    assert {j['case']['sentence_id'] for j in jobs if j['stage']==0} == {
        'fresh_id_1973', 'fresh_th_1844', 'fresh_tl_1913'}
    assert [j['case']['condition'] for j in jobs[:10]] == ['plain']*5+['control']*5
    assert [j['seed'] for j in jobs[:10]] == list(range(42,47))*2
    assert Counter(j['stage'] for j in jobs) == {stage:30 for stage in range(10)}


def test_r39_diagnostic_inherits_cer_independent_extreme_acoustic_gate(module):
    base = [{'case_id':f'a{i}', 'seed':42, 'lang':'th', 'cer':0., 'hyp':'text',
             'text':'text', 'over_read':False, 'rms_dbfs':-20, 'speech_ratio':.95,
             'audio_sec':5., 'tail_silence_sec':.1, 'numeric':False, 'f0_std_st':0,
             'wer':None} for i in range(20)]
    lookup = {module.prior.key(r):r for r in base}
    rows = [dict(r) for r in base]
    rows[0].update(audio_sec=10.01, speech_ratio=.849, cer=.15)
    assert module.decision(rows, lookup, stage_end=True, diagnostic=True) == (None, {})
    reason, acoustics = module.decision(rows, lookup, stage_end=True, diagnostic=True,
                                       paired_extremes=True)
    assert reason == 'stage_acoustic_regression'
    assert acoustics['paired_extreme_acoustic']
