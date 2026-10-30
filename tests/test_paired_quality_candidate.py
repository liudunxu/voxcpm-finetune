import copy
import importlib
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('paired_quality_candidate')


def row(text='Ada 12 orang di rumah.', hyp='Ada dua belas orang di rumah.'):
    return {'text':text,'lang':'id','whisper_hyp':[hyp,hyp],'qwen_hyp':[hyp,hyp],'qwen_eos':[True,True],
            'ast_music':[.01,.01],'clap_all_three_music':[False,False],
            'original_metrics':{'rms_dbfs':-20,'speech_ratio':.95},
            'separated_metrics':{'rms_dbfs':-20,'speech_ratio':.95}}


def test_numeric_equivalence_does_not_excuse_value_changes_or_missing_words(module):
    assert module.content_view(row(),0)['reasons']==[]
    for hyp in ('Ada 13 orang di rumah.','Ada dua belas di rumah.'):
        view=module.content_view(row(hyp=hyp),0)
        assert view['reasons']
        assert view['numeric_equivalence']==[False,False]


def test_unicode_equivalence_is_diagnostic_and_never_edits_labels(module):
    value=row('กำลังเดิน','กําลังเดิน');value['lang']='th'
    original=copy.deepcopy(value)
    assert module.content_view(value,0)['reasons']==[]
    assert value==original


def test_shared_low_cer_deletion_remains_a_suspect(module):
    text='Saya melihat semua orang yang sedang berjalan menuju rumah baru itu dengan sangat tenang'
    value=row(text,text.replace(' itu',''))
    result=module.content_view(value,0)
    assert max(result['cer'])<.05
    assert 'shared_asr_suspected_deletion' in result['reasons']


def test_music_suspect_is_not_recovered_by_separation(module):
    for hints in ([True,False],[False,True]):
        value=row();value['clap_all_three_music']=hints
        decision=module.select_version(value,{'reasons':[]},{'reasons':[]})
        assert decision['selected']=='excluded'
        assert decision['status'].startswith('uncertain')


def test_derivative_damage_uses_eligible_original_and_keeps_failed_result(module):
    value=row();original=copy.deepcopy(value)
    assert module.select_version(value,{'reasons':[]},{'reasons':[]})['selected']=='vocals'
    value['separated_metrics']['rms_dbfs']=-80
    value['separated_metrics']['speech_ratio']=0
    value['whisper_hyp'][1]='selamat menikmati';value['qwen_hyp'][1]='Hai'
    decision=module.select_version(value,{'reasons':[]},{'reasons':['no_energy_boundary']})
    assert decision['selected']=='original'
    assert 'rms_loss_gt_6db' in decision['derivative_reasons']
    assert 'asr_worsened_after_separation' in decision['derivative_reasons']
    assert value['whisper_hyp'][1]=='selamat menikmati'
    assert value['text']==original['text']
    assert module.select_version(value,{'reasons':['suspected_clipping']},{'reasons':[]})['selected']=='excluded'


def test_truncated_asr_is_retained_as_a_bad_outcome_without_retry(module):
    value=row();value['qwen_eos'][1]=False
    decision=module.select_version(value,{'reasons':[]},{'reasons':[]})
    assert decision['selected']=='original'
    assert decision['derivative_reasons']==['separated_asr_without_eos']
    value['qwen_eos'][0]=False
    assert module.select_version(value,{'reasons':[]},{'reasons':[]})['selected']=='excluded'


def test_duration_balance_caps_original_exposure_and_preserves_reference_identity(module):
    rows=[{'lang':lang,'duration':3+(i%10),'origin_audio':f'{lang}/{i}',
           'audio':f'derived/{lang}/{i}','speaker':'original-id','ref_audio':None}
          for lang,count in (('id',100),('th',80),('tl',20),('zh',30),('en',20)) for i in range(count)]
    before=copy.deepcopy(rows)
    mixed,budget=module.balanced_mix(rows)
    assert rows==before
    assert module.balanced_mix(rows)==(mixed,budget)
    assert max(Counter(r['origin_audio'] for r in mixed).values())<=2
    assert all(r in rows and not r['ref_audio'] for r in mixed)
    for lang,share in module.SHARES.items():
        seconds=sum(r['duration'] for r in mixed if r['lang']==lang)
        assert budget*share<=seconds<=budget*share+12
    with pytest.raises(ValueError,match='Empty language'):
        module.balanced_mix([r for r in rows if r['lang']!='tl'])
