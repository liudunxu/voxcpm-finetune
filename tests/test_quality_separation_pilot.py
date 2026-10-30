import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('quality_separation_pilot')


@pytest.mark.parametrize('length',[3,10,11,17,30,31])
def test_blended_identity_has_no_missing_endpoints_or_overlap_gain(module,length):
    total=length*100
    source=np.linspace(-.7,.8,total)
    output=np.zeros(total);weights=np.zeros(total)
    for start,end in module.intervals(total,1000,200):
        w=module.blend_weights(start,end,total,200)
        output[start:end]+=source[start:end]*w
        weights[start:end]+=w
    assert np.all(weights>0)
    assert np.allclose(output/weights,source)


def test_sources_and_controls_are_disjoint_and_not_selected_by_separation_scores(module):
    def row(i,lang):return {'audio':str(i),'file_sha256':str(i),'lang':lang}
    suspects=[row(i,'id') for i in range(10)]+[row(i,'th') for i in range(10,15)]
    controls=[row(i,lang) for lang,start in (('id',30),('th',40),('tl',50)) for i in range(start,start+5)]
    current={r['audio'] for r in controls}
    selected=module.select_cases(current,suspects,controls,row(20,'id'))
    assert len(selected)==24
    assert selected==module.select_cases(current,suspects[::-1],controls[::-1],row(20,'id'))
    assert not {r['audio'] for r in selected[:12]} & current
    with pytest.raises(ValueError):module.select_cases(set(),suspects,controls,row(20,'id'))
