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


def test_expansion_cannot_skip_acoustic_failure(module):
    row = {'case_id':'a','seed':42,'lang':'th','cer':0.,'hyp':'text','text':'text',
           'over_read':False,'rms_dbfs':-20,'speech_ratio':.95,'audio_sec':5.,
           'tail_silence_sec':.1,'numeric':False,'f0_std_st':0,'wer':None}
    candidate = {**row, 'tail_silence_sec':1.2}
    reason, failures = module.decision([candidate], {('a',42):row}, stage_end=True)
    assert reason == 'stage_acoustic_regression'
    assert failures['tail_silence']
