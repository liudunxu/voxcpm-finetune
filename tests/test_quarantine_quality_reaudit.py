import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('quarantine_quality_reaudit')


def result(module, score=.6, positive=True, role='current_mid_ast'):
    scores = [.1, .2, .1, .3, .1, .2] if positive else [.1, .2, .3, .1, .1, .2]
    return {'case':{'role':role, 'music_score':score},
            'windows':[{'start_sample':0, 'end_sample':48000, 'cosine_scores':scores,
                        **module.prompt_margins(scores)}],
            'crossmodel_review_priority':score>=.5 and positive}


def test_review_keeps_disagreement_uncertain_and_never_readds_old_controls(module):
    assert module.suspect_reason(result(module)) == 'cross_model_music_suspect'
    assert module.suspect_reason(result(module, positive=False)) == 'ast_music_unresolved'
    assert module.suspect_reason(result(module, score=.01)) == 'clap_music_unresolved'
    assert module.suspect_reason(result(module, score=.01, positive=False)) is None
    assert module.suspect_reason(result(module, role='previously_excluded_control')) is None


def test_verify_catches_forged_decision_missing_tail_and_changed_case(module):
    item = result(module)
    module.verify_result(item, item['case'], 48000, 16000)
    forged = copy.deepcopy(item)
    forged['windows'][0]['all_three_music_higher'] = False
    with pytest.raises(ValueError, match='aggregation'):
        module.verify_result(forged, item['case'], 48000, 16000)
    with pytest.raises(ValueError, match='windows'):
        module.verify_result(item, item['case'], 64000, 16000)
    with pytest.raises(ValueError, match='case'):
        module.verify_result(item, {**item['case'], 'music_score':.1}, 48000, 16000)


def test_subset_preserves_utf8_bytes_and_ref_pcm_quarantine_propagates(module, tmp_path):
    source, out = tmp_path/'source.jsonl', tmp_path/'out.jsonl'
    source.write_bytes('{"text":"ทดสอบ"}\n{"text":"kept"}\n'.encode())
    module.write_subset(source, out, [2])
    assert out.read_bytes() == b'{"text":"kept"}\n'
    with pytest.raises(FileExistsError):
        module.write_subset(source, out, [1])
    rows = [{'audio':'/a'}, {'audio':'/b', 'ref_audio':'/copy'}, {'audio':'/c', 'origin_audio':'/raw/a'}]
    acoustic = {p:{'pcm_sha256':p} for p in ('/a','/b','/c')}
    acoustic['/copy'] = acoustic['/a']
    flags = [{'audio':'/a', 'pcm_sha256':'/a', 'uses':[{'origin_audio':'/raw/a'}]}]
    assert module.exclude_flagged(rows, flags, acoustic)[0] == []


def test_output_accounting_handles_four_digit_ids_and_rejects_gaps(module, tmp_path):
    for i in range(998,1003):
        (tmp_path/f'{i:03d}.json').write_text('{}')
    assert [int(p.stem) for p in module.ordered_results(tmp_path,998,1003)] == list(range(998,1003))
    (tmp_path/'1000.json').unlink()
    with pytest.raises(ValueError, match='Missing'):
        module.ordered_results(tmp_path,998,1003)
