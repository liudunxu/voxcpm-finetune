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
