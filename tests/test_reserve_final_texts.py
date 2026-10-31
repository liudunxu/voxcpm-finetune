import importlib
import hashlib
import json
from pathlib import Path

import pytest


def test_reservation_excludes_known_or_suspect_text_without_repair(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module = importlib.import_module('reserve_final_texts')
    rows = [{'source_sentence_id':str(i), 'text':t} for i,t in enumerate([
        'Kalimat biasa untuk pemeriksaan.', 'Kalimat lama pernah dilatih.',
        'Kata bien Ã©levÃ© yang rusak.', 'Kalimat dengan (singkatan).',
        'Kalimat bersih lainnya.', 'Sebuah kalimat terakhir.'])]
    before = [dict(r) for r in rows]
    excluded = {module.prior.frozen.evaluation._norm(rows[1]['text'])}
    first, detail = module.select(rows, excluded, 'id', count=3)
    second, _ = module.select(list(reversed(rows)), excluded, 'id', count=3)
    assert first == second
    assert {r['source_sentence_id'] for r in first} == {'0','4','5'}
    assert detail['excluded'] == {'known_text':1, 'encoding_suspect':1, 'body_control_like_punctuation':1}
    assert rows == before
    with pytest.raises(ValueError, match='Insufficient'):
        module.select(rows, excluded, 'id', count=4)


def test_reserved_texts_are_protected_from_future_training(monkeypatch):
    directory = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(directory/'scripts'))
    module = importlib.import_module('reserve_final_texts')
    path = directory/'eval_cases/reserved_final_20261030.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    excluded, _ = importlib.import_module('incremental_mix').load_exclusions(directory/'eval_cases')
    assert len(rows) == 90
    for row in rows:
        assert module.input_issue(row) is None
        assert module.prior.frozen.evaluation._norm(row['text']) in excluded
        assert row['evaluation_use'] == 'once_after_candidate_selection'


def test_id_supplement_keeps_conditioning_and_does_not_rewrite_old_cases(monkeypatch):
    directory = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(directory/'scripts'))
    module = importlib.import_module('reserve_final_texts')
    old = [json.loads(line) for line in (directory/'eval_cases/fresh_text_holdout_20261028.jsonl')
           .read_text(encoding='utf-8').splitlines()]
    before = json.dumps(old, ensure_ascii=False, sort_keys=True)
    selected = [{'source_sentence_id':str(i), 'source_audio_filename':f'{i}.wav', 'text':text}
                for i,text in enumerate(('Kalimat bersih untuk pemeriksaan.',
                                          'Sebuah kalimat baru yang berbeda.'))]
    cases = module.supplement_cases(selected, old)
    assert len(cases) == 4
    assert {r['condition'] for r in cases} == {'plain', 'control'}
    assert len({r['sentence_id'] for r in cases}) == 2
    for row in cases:
        parent = next(r for r in old if r['case_id'] == row['conditioning_slot'])
        assert all(row[k] == parent[k] for k in ('ref_audio','ref_lang','control','condition','lang'))
        assert row['text'] == selected[int(row['source_sentence_id'])]['text']
        assert row['evaluation_use'] == 'development_only'
    assert json.dumps(old, ensure_ascii=False, sort_keys=True) == before
    with pytest.raises(ValueError, match='distinct clean'):
        module.supplement_cases([selected[0], selected[0]], old)
    with pytest.raises(ValueError, match='two encoding-risk'):
        module.supplement_cases(selected, [r for r in old if module.input_issue(r) != 'encoding_suspect'])


def test_development_selection_excludes_reserved_final_texts(monkeypatch):
    directory = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(directory/'scripts'))
    module = importlib.import_module('reserve_final_texts')
    final = [json.loads(line) for line in (directory/'eval_cases/reserved_final_20261030.jsonl')
             .read_text(encoding='utf-8').splitlines()]
    pool = [r for r in final if r['lang'] == 'id'] + [
        {'source_sentence_id':'test1', 'text':'Sebuah kalimat baru untuk pengujian.'},
        {'source_sentence_id':'test2', 'text':'Kalimat tambahan yang berbeda untuk pemeriksaan.'}]
    excluded = {module.prior.frozen.evaluation._norm(r['text']) for r in final}
    chosen, _ = module.select(pool, excluded, 'id', count=2)
    assert {r['source_sentence_id'] for r in chosen} == {'test1','test2'}


def test_frozen_id_supplement_is_protected_from_training(monkeypatch):
    directory = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(directory/'scripts'))
    module = importlib.import_module('reserve_final_texts')
    path = directory/'eval_cases/development_id_supplement_20261031.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'd1ea31c3974b21fc0674c5aafa00296794cba9cdb5f9ff129e5f97ba4809ae16'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    excluded, _ = importlib.import_module('incremental_mix').load_exclusions(directory/'eval_cases')
    assert len(rows) == 4 and len({r['sentence_id'] for r in rows}) == 2
    assert {r['lang'] for r in rows} == {'id'}
    assert all(module.prior.frozen.evaluation._norm(r['text']) in excluded for r in rows)
    assert all(r['evaluation_use'] == 'development_only' for r in rows)
