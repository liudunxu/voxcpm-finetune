import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('quality_separation_review')


def test_index_rejects_duplicate_missing_and_extra_outcomes(module):
    a={'case':{'audio':'raw'}}
    b={'case':{'audio':'vocals'}}
    assert module.index_results([a,b],{'raw','vocals'})=={'raw':a,'vocals':b}
    for items in ([a,a],[a],[a,b,{'case':{'audio':'unknown'}}]):
        with pytest.raises(ValueError):module.index_results(items,{'raw','vocals'})


@pytest.mark.parametrize('key,value',[
    ('audio','different.wav'),('file_sha256','different bytes'),
    ('text','different target'),('lang','tl'),
])
def test_reused_asr_must_match_actual_audio_and_conditioning(module,key,value):
    case={'audio':'raw','file_sha256':'raw-sha','text':'hello','lang':'en'}
    pair={'case':case,'vocals':'vocals','vocals_sha256':'vocals-sha'}
    outcomes={'raw':{'case':case},'vocals':{'case':{**case,'audio':'vocals','file_sha256':'vocals-sha'}}}
    module.verify_pair_conditioning(pair,outcomes)
    changed=copy.deepcopy(outcomes);changed['vocals']['case'][key]=value
    with pytest.raises(ValueError,match='conditioning'):module.verify_pair_conditioning(pair,changed)


@pytest.mark.parametrize('failure',[None,'duplicate','missing','wrong_vocals','wrong_text'])
def test_full_collection_requires_all_four_frozen_shards(module,tmp_path,failure):
    pairs=[{'case':{'audio':f'raw{i}','text':f'text{i}'},'vocals':f'vocals{i}'} for i in range(4)]
    (tmp_path/'plan.json').write_text(json.dumps({'outputs':4,'inputs_sha256':{},'policy':'test'}))
    (tmp_path/'pairs.json').write_text(json.dumps(pairs))
    for i,pair in enumerate(pairs):
        row={'original_audio':pair['case']['audio'],'vocals':pair['vocals'],
             'text':pair['case']['text'],'review_id':i}
        if i==3 and failure=='duplicate':row['original_audio']='raw0'
        if i==3 and failure=='wrong_vocals':row['vocals']='vocals0'
        if i==3 and failure=='wrong_text':row['text']='different'
        directory=tmp_path/f'shard{i}';directory.mkdir()
        (directory/'comparison.json').write_text(json.dumps({'pairs':[] if i==3 and failure=='missing' else [row],
                                                            'outputs_sha256':{}}))
    if failure:
        with pytest.raises(ValueError):module.collect_expanded(tmp_path)
        assert not (tmp_path/'comparison.json').exists()
    else:
        module.collect_expanded(tmp_path)
        result=json.loads((tmp_path/'comparison.json').read_text())
        assert len(result['pairs'])==4
        assert result['training_eligible'] is False
        assert result['listening_pending'] is True


@pytest.mark.parametrize('failure',[None,'without_eos','changed_audio','nonterminal','raw_outcome_differs'])
def test_readonly_verifier_checks_shards_raw_outcomes_and_audio_sha(module,tmp_path,monkeypatch,failure):
    pairs=[];rows=[]
    for i in range(4):
        raw=tmp_path/f'raw{i}.wav';vocal=tmp_path/f'vocal{i}.wav'
        raw.write_bytes(b'raw');vocal.write_bytes(b'separated')
        pairs.append({'case':{'audio':str(raw),'text':'unchanged','lang':'id','duration':3,
                             'file_sha256':module.sha(raw)},'vocals':str(vocal),
                      'vocals_sha256':module.sha(vocal),'frames':48000,'sample_rate':16000})
        rows.append({'original_audio':str(raw),'vocals':str(vocal),'text':'unchanged',
                     'lang':'id','review_id':i,'qwen_eos':[True,failure!='without_eos']})
        directory=tmp_path/f'shard{i}';directory.mkdir()
        for name in ('plan.json','clap/plan.json','clap/summary.json','qwen/plan.json','qwen/summary.json'):
            path=directory/name;path.parent.mkdir(exist_ok=True)
            path.write_text('{}')
        (directory/'comparison.json').write_text(json.dumps({'pairs':[rows[-1]],'outputs_sha256':{}}))
        for name,marker in (('music.log','MUSIC_COMPLETE'),('asr.log','ASR_COMPLETE'),
                            ('collect.log','PAIRED_REVIEW_COMPLETE 1')):
            (directory/name).write_text(marker+'\n')
    (tmp_path/'pairs.json').write_text(json.dumps(pairs))
    (tmp_path/'plan.json').write_text(json.dumps({'outputs':4,'inputs_sha256':{}}))
    (tmp_path/'comparison.json').write_text(json.dumps({'pairs':rows,'outputs_sha256':{}}))
    monkeypatch.setattr(module,'paired_rows',lambda path,**kw:[rows[int(path.name[-1])]])
    if failure=='changed_audio':Path(pairs[0]['vocals']).write_bytes(b'different')
    if failure=='nonterminal':(tmp_path/'shard1/asr.log').write_text('in progress\n')
    if failure=='raw_outcome_differs':
        monkeypatch.setattr(module,'paired_rows',lambda path,**kw:[{**rows[int(path.name[-1])],'text':'changed'}])
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    if failure not in (None,'without_eos'):
        with pytest.raises(ValueError):module.verify_expanded(tmp_path)
    else:
        verified=module.verify_expanded(tmp_path)
        assert verified['pairs']==verified['unique_sources']==4
        assert verified['raw_outcomes_reconstructed'] is True
        assert sum(s['qwen_without_eos'][1] for s in verified['shards'])==(4 if failure=='without_eos' else 0)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
