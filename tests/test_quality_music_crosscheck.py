import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('quality_music_crosscheck')


def test_strata_are_disjoint_deterministic_and_source_balanced(module):
    def row(i,score,source):
        return {'audio':str(i),'file_sha256':str(i),'pcm_sha256':str(i),
                'source_id':source,'music_score':score}
    current=[row(i,.75,'id') for i in range(100)]
    current += [row(i,.6,'th' if i%2 else 'id') for i in range(100,160)]
    current += [row(i,.01,str(i%10)) for i in range(160,260)]
    excluded=[row(i,.9,'id') for i in range(260,280)]
    cases=module.select_cases(current,excluded)
    assert cases == module.select_cases(current[::-1],excluded[::-1])
    assert len(cases) == len({r['audio'] for r in cases}) == 176
    low=[r for r in cases if r['role']=='current_low_ast_control']
    assert module.Counter(r['source_id'] for r in low) == dict.fromkeys(map(str,range(10)),4)
    expanded=module.expand_cases(current,cases)
    assert expanded[:176] == cases
    assert len(expanded) == 216
    assert {r['audio'] for r in expanded if r['role'] != 'previously_excluded_control'} >= {
        r['audio'] for r in current if r['music_score'] >= .5}
    assert expanded == module.expand_cases(current[::-1],cases)
    excluded[0]['pcm_sha256']=current[0]['pcm_sha256']
    with pytest.raises(ValueError):
        module.select_cases(current,excluded)


def test_prompt_agreement_needs_all_three_and_windows_cover_tail(module):
    assert module.prompt_margins([.1,.2,.1,.3,.1,.15])['all_three_music_higher']
    assert not module.prompt_margins([.1,.2,.2,.1,.1,.15])['all_three_music_higher']
    assert not module.prompt_margins([.1,.1,.1,.2,.1,.3])['all_three_music_higher']
    with pytest.raises(ValueError):module.prompt_margins([float('nan')]*6)
    assert module.windows(17000,1000)==[(0,10000),(5000,15000),(7000,17000)]
    assert module.windows(3000,1000)==[(0,3000)]
