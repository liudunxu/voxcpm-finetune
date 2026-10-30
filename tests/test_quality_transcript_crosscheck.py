import importlib
import json
from pathlib import Path

import pytest


def test_selection_exempts_only_numeric_equivalence_and_keeps_mixed_errors(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('quality_transcript_crosscheck')
    def row(i,cer,text,hyp):
        return {'audio':str(i),'file_sha256':str(i),'pcm_sha256':str(i),
                'source_id':'cv22_id','lang':'id','cer':cer,'text':text,'hyp':hyp}
    rows=[row(i,.11,'empat minggu sebelum ini','4 minggu') for i in range(80)]
    rows += [row(i,0.,'tidak ada masalah','tidak ada masalah') for i in range(80,120)]
    rows += [row(120,.11,'empat minggu','4 minggu')]
    cases,enriched=module.select_cases(rows)
    assert {r['audio'] for r in cases} == {str(i) for i in range(120)}
    assert enriched[-1]['numeric_review']['numeric_only_difference']
    assert module.select_cases(rows[::-1])[0] == cases


def test_expanded_collection_rejects_duplicate_review_ids(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('quality_transcript_crosscheck')
    module.save(tmp_path/'plan.json',{'inputs_sha256':{}})
    module.save(tmp_path/'cases.json',[{},{}])
    module.save(tmp_path/'reused.json',{'first':{'review_id':0,'path':'unused','sha256':'unused'}})
    shard=tmp_path/'shard0';(shard/'items').mkdir(parents=True)
    case={'review_id':0}
    output=shard/'items/000.json'
    module.save(output,{'case':case})
    module.save(shard/'cases.json',[case])
    module.save(shard/'summary.json',{'completed':1,'outputs_sha256':{str(output):module.sha(output)}})
    with pytest.raises(ValueError,match='Duplicate'):
        module.collect_expanded(tmp_path)
    assert not (tmp_path/'summary.json').exists()
