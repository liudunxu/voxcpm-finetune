from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('full_corpus_parallel')


def test_only_incomplete_last_write_can_be_reassigned(module, tmp_path):
    path = tmp_path / 'old.jsonl'
    original = b'{"id":1}\n{"id":2}\n{"id":'
    path.write_bytes(original)
    rows, fragment = module.complete_records(path)
    assert rows == [{'id': 1}, {'id': 2}] and fragment['bytes'] == 6
    assert path.read_bytes() == original
    path.write_bytes(b'{"id":1}\n{"id":2}')
    assert module.complete_records(path) == ([{'id': 1}, {'id': 2}], None)
    path.write_bytes(b'{"id":1}\nbroken\n{"id":2}\n')
    with pytest.raises(ValueError, match='Corrupt'):
        module.complete_records(path)


def test_old_results_require_matching_identity_hash_and_policy(module):
    task = {'id': 3, 'audio': '/frozen.wav', 'file_sha256': 'frozen-sha'}
    row = {**task, 'result': {'hyp': 'one sentence', 'comparisons': [{'cer': 0.}],
        'segments': [{'start': 0., 'end': 3., 'avg_logprob': -.1, 'no_speech_prob': .01}]}, 'reasons': []}
    assert module.validate_records([row], {3: task}) == {3}
    for changed in ({**row, 'file_sha256': 'changed'}, {**row, 'audio': '/other.wav'},
                    {**row, 'reasons': ['asr_text_mismatch']}):
        with pytest.raises(ValueError, match='Duplicate, changed'):
            module.validate_records([changed], {3: task})
    with pytest.raises(ValueError, match='Duplicate, changed'):
        module.validate_records([row, row], {3: task})


def test_handoff_preserves_completed_results_and_eight_shards_cover_only_pending(module, monkeypatch, tmp_path):
    previous, parent = tmp_path / 'previous', tmp_path / 'parent'
    previous.mkdir()
    parent.mkdir()
    (previous / 'manifests/current').mkdir(parents=True)
    (previous / 'manifests/current/train.jsonl').write_text('{}\n', encoding='utf-8')
    module.save_new(previous / 'plan.json', {'policy': module.clean.POLICY, 'inputs_sha256': {}})
    module.save_new(parent / 'plan.json', {})
    module.save_new(previous / 'metadata_issues.json', {})
    module._write_jsonl([], previous / 'protected_validation.jsonl')
    tasks = [{'id': i, 'audio': f'/{i}.wav', 'file_sha256': str(i)} for i in range(30)]
    module._write_jsonl(tasks, previous / 'tasks.jsonl')
    module._write_jsonl([{**r, 'reasons': ['pre_excluded'] if r['id'] == 29 else []} for r in tasks], previous / 'acoustic.jsonl')
    result = {'hyp': 'a line', 'comparisons': [{'cer': 0.}],
              'segments': [{'start': 0., 'end': 3., 'avg_logprob': -.1, 'no_speech_prob': .01}]}
    retained = [{**task, 'result': deepcopy(result), 'reasons': []} for task in tasks[:5]]
    module._write_jsonl(retained, previous / 'asr.jsonl')
    original = (previous / 'asr.jsonl').read_bytes()
    monkeypatch.setattr(module, 'PREVIOUS', previous)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.clean, 'PARENT', parent)
    monkeypatch.setattr(module.clean, 'historical_exclusions', lambda p: ([], [], {}, ['old-bad-pcm']))
    output = tmp_path / 'new'
    module.prepare(output)
    pending = module._read_manifest(output / 'pending.jsonl')
    assert {row['id'] for row in pending} == set(range(5, 29))
    shards = [pending[index::8] for index in range(8)]
    ids = [row['id'] for shard in shards for row in shard]
    assert len(ids) == len(set(ids)) == 24
    assert module._read_manifest(output / 'retained_asr.jsonl') == retained
    assert (previous / 'asr.jsonl').read_bytes() == original
    assert module.clean.read_json(output / 'plan.json')['inherited_pcm'] == ['old-bad-pcm']
    with pytest.raises(FileExistsError):
        module.prepare(output)
