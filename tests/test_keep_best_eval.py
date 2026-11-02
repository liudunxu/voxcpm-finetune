import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('keep_best_eval')


def row(case, seed, cer, *, lang='th', hyp='hello', over_read=False, rms=-30, speech=.9):
    return {'case_id': case, 'seed': seed, 'cer': cer, 'lang': lang, 'hyp': hyp,
        'text': 'hello', 'ref_audio': 'ref.wav', 'ref_lang': 'zh', 'control': '',
        'over_read': over_read, 'rms_dbfs': rms, 'speech_ratio': speech,
        'wav': f'{case}-{seed}.wav', 'numeric': False, 'wer': None}


def test_fixed_budget_qc_avoids_bad_seed_and_compares_by_language(module):
    base = [row(case, seed, .1, lang='id' if case == 'id' else 'th')
            for case in ('th', 'id') for seed in module.SEEDS]
    candidate = [row(case, seed, .08 if seed == 43 else .5,
                     lang='id' if case == 'id' else 'th')
                 for case in ('th', 'id') for seed in module.SEEDS]
    candidate[0]['over_read'] = True
    candidate[0]['cer'] = .01
    result = module.compare(base, candidate)
    assert all(row['candidate_seed'] == 43 for row in result['selections'])
    assert result['cases'] == 2
    assert result['by_lang']['th']['candidate_severe_cases'] == 0
    assert result['by_lang']['id']['base_severe_cases'] == 0
    assert result['metrics']['candidate']['by_lang']['th']['mean_cer'] == .08


def test_rejects_missing_seed_and_changed_input(module):
    base = [row('case', seed, .1) for seed in module.SEEDS]
    candidate = [row('case', seed, .1) for seed in module.SEEDS]
    with pytest.raises(ValueError, match='Incomplete'):
        module.compare(base, candidate[:-1])
    candidate[-1]['text'] = 'changed'
    with pytest.raises(ValueError, match='Input mismatch'):
        module.compare(base, candidate)


def test_empty_and_near_silence_remain_severe(module):
    assert module.severe(row('case', 42, 0, hyp=''))
    assert module.severe(row('case', 42, 0, rms=-70, speech=.01))
    assert module.severe(row('case', 42, 0, over_read=True))
