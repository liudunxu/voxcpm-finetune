import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('r31_failure_probe')


def test_budget_preserves_endpoints_and_never_combines_interventions(module):
    cases = [{'case_id': f'{sid}_{cond}', 'sentence_id': sid, 'condition': cond,
              'control': 'sad' if cond == 'control' else '',
              'text': 'คำว่า (civilis) text' if i != 1 else 'plain words', 'ref_audio': '/same.wav'}
             for i, sid in enumerate(module.SENTENCES) for cond in ('plain', 'control')]
    complete = {(c['case_id'], seed) for c in cases for seed in module.frozen.SEEDS}
    missing = {(cases[0]['case_id'], 45), (cases[1]['case_id'], 44)}
    original = copy.deepcopy(cases)
    jobs = module.make_jobs(cases, complete - missing)
    assert len(jobs) == len({j['id'] for j in jobs}) == 102
    assert module.Counter(j['view'] for j in jobs) == {
        'original_r31': 2, 'half_r31': 40, 'commas_base': 30, 'commas_r31': 30}
    assert {(j['case']['case_id'], j['seed']) for j in jobs if j['view'] == 'original_r31'} == missing
    assert all(j['strength'] == 1 for j in jobs if j['view'] == 'commas_r31')
    assert all(j['case'] in cases for j in jobs if j['view'] == 'half_r31')
    assert cases == original


def test_punctuation_keeps_spoken_characters_reference_and_control(module):
    case = {'case_id': 'id_plain', 'text': 'Duvall (yang menikah) selesai.',
            'control': 'sad', 'ref_audio': '/actual.wav', 'lang': 'id'}
    result = module.punctuated(case)
    assert result['parent_case_id'] == case['case_id']
    assert result['text'] == 'Duvall , yang menikah, selesai.'
    assert result['control'] == case['control'] and result['ref_audio'] == case['ref_audio']
    assert module.frozen.evaluation._norm(result['text']) == module.frozen.evaluation._norm(case['text'])
    with pytest.raises(ValueError, match='balanced'):
        module.punctuated({**case, 'text': 'bad (unclosed'})


def test_near_silence_and_asr_failure_are_separate_from_tail_silence(module):
    silent = {'hyp': '', 'rms_dbfs': -106, 'speech_ratio': 0., 'over_read': False}
    assert module.flags(silent) == {'empty_asr': True, 'near_silence': True, 'suspected_overread': False}
    assert not module.flags({**silent, 'rms_dbfs': -30, 'speech_ratio': .9})['near_silence']
    assert not module.flags({**silent, 'hyp': 'spoken', 'rms_dbfs': -35, 'speech_ratio': .8})['empty_asr']


def test_prepare_refuses_overwrite(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
