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


def test_r39_punctuation_budget_keeps_words_quotes_and_all_seeds(module):
    cases = [{'case_id':f'fresh_id_1973_{condition}','sentence_id':'fresh_id_1973',
              'condition':condition,'text':'"Namun, panas (selalu) dan matahari (ketika cerah)."',
              'lang':'id','ref_audio':'/same.wav','control':control}
             for condition,control in (('plain',''),('control','sad'))]
    before = copy.deepcopy(cases)
    jobs = module.punctuation_jobs(cases)
    assert len(jobs) == len({j['id'] for j in jobs}) == 20
    assert [j['model'] for j in jobs] == ['base']*10+['candidate']*10
    for model in ('base','candidate'):
        for condition in ('plain','control'):
            group = [j for j in jobs if j['model']==model and j['case']['condition']==condition]
            assert [j['seed'] for j in group] == [42,43,44,45,46]
            assert all(j['case']['text']=='"Namun, panas , selalu, dan matahari , ketika cerah,."' for j in group)
            assert all(j['case']['ref_audio']=='/same.wav' for j in group)
    assert cases == before
    with pytest.raises(ValueError,match='ID1973'):
        module.punctuation_jobs(cases[:1])


def punctuation_row(seed, **extra):
    return {'case_id':'original_commas','seed':seed,'lang':'id','hyp':'spoken words','cer':0.,'wer':0.,
            'over_read':False,'numeric':False,'rms_dbfs':-20.,'speech_ratio':.9,
            'audio_sec':5.,'tail_silence_sec':.1,'f0_std_st':0.,**extra}


def test_punctuation_checks_model_regression_even_if_original_was_already_bad(module):
    base = [punctuation_row(s) for s in (42,43)]
    candidate = [punctuation_row(s,cer=.35) for s in (42,43)]
    reused = {model:[{**r,'case_id':'original','cer':(.4 if model=='candidate' else 0.)} for r in base]
              for model in ('base','candidate')}
    rows = [{'model':'base','item':r} for r in base]+[{'model':'candidate','item':r} for r in candidate]
    reason,_ = module.punctuation_rejection(rows,reused)
    assert reason == 'candidate_vs_commas_base:two_large_paired_cer_regressions'
    assert module.punctuation_rejection([{'model':'base','item':punctuation_row(42,over_read=True)}],reused)[0] == 'base_vs_original:new_suspected_overread'


def test_punctuation_keeps_absolute_acoustic_gate_and_requires_full_group(module):
    original = [punctuation_row(s,case_id='original') for s in module.frozen.SEEDS]
    rows = [{'model':'base','item':punctuation_row(s,tail_silence_sec=.8)} for s in module.frozen.SEEDS]
    reason,acoustics = module.punctuation_rejection(rows,{'base':original},stage_end=True)
    assert reason == 'base_vs_original:stage_acoustic_regression'
    assert acoustics['tail_silence']
    with pytest.raises(ValueError,match='Incomplete condition'):
        module.punctuation_rejection(rows[:-1],{'base':original},stage_end=True)
