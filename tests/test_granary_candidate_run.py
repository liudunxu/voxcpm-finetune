import importlib
from pathlib import Path

import pytest


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('granary_candidate_run')


def test_full_source_order_is_required_despite_matching_text(candidate):
    raw = [{'audio':'/raw/0000000.wav','text':'first'},
           {'audio':'/raw/0000001.wav','text':'second'}]
    tsv = [['2','b.wav','second','','','100',''],['1','a.wav','first','','','100','']]
    assert candidate.source_index(raw,tsv) == {'/raw/0000000.wav':'a.wav','/raw/0000001.wav':'b.wav'}
    with pytest.raises(ValueError):
        candidate.source_index(raw[::-1],tsv)
    with pytest.raises(ValueError):
        candidate.source_index(raw,tsv[:1])
    with pytest.raises(ValueError):
        candidate.source_index(raw,[tsv[0],tsv[0]])
    with pytest.raises(ValueError):
        candidate.source_index(raw,[[]])


def test_written_view_keeps_numbers_but_cleaned_view_cannot_certify_alignment(candidate):
    row = {'text':'Um, I have 25 dollars.'}
    label = {'high_quality':True,'tn_raw':'Um, I have twenty five dollars.',
             'itn_raw':'Um, I have 25 dollars.','itn_clean':'I have 25 dollars.'}
    assert candidate.label_reasons(row,label) == ([],['itn_raw'])
    # Do not treat removal of actually spoken fillers as a correct TTS target.
    assert candidate.label_reasons({'text':label['itn_clean']},label)[0] == ['granary_transcript_disagreement']
    assert candidate.label_reasons(row,{**label,'high_quality':False})[0] == ['granary_not_high_quality']
    assert candidate.label_reasons(row,None)[0] == ['missing_granary_label']
    assert candidate.label_reasons(row,{**label,'high_quality':'true'})[0] == ['granary_not_high_quality']


def test_thai_marks_and_language_are_not_flattened(candidate):
    row = {'text':'ป่า'}
    assert candidate.label_reasons(row,{'high_quality':True,'tn_raw':'ปา'})[0]
    assert candidate.asr_pattern_reasons('ภาษาไทย'*8,'ต่างกัน','th',20) == []


def test_asr_hallucination_observations_preserve_actual_repetitions(candidate):
    repeated = 'no '*10
    assert candidate.asr_pattern_reasons(repeated,repeated,'id',4) == []
    assert candidate.asr_pattern_reasons('no no no','other words','id',4) == []
    assert candidate.asr_pattern_reasons(repeated,'different sentence','id',4) == ['asr_repetition_low_diversity']
    assert 'asr_character_rate_gt_40' in candidate.asr_pattern_reasons('x'*121,'short','th',3)


def test_existing_run_is_never_overwritten(candidate,tmp_path):
    with pytest.raises(FileExistsError):
        candidate.prepare(tmp_path)
