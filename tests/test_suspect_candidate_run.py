import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('suspect_candidate_run')


def test_only_data_and_output_paths_change_from_r28(module, tmp_path):
    template = Path(__file__).resolve().parents[1] / 'configs/lora_joint3_r28_clean_1ep.yaml'
    original = yaml.safe_load(template.read_text(encoding='utf-8'))
    config = module.fixed_config(template, tmp_path)
    assert {key for key in config if config[key] != original[key]} == {
        'train_manifest', 'val_manifest', 'save_path', 'tensorboard'}
    assert config['num_iters'] == config['max_steps'] == 459
    assert config['warmup_steps'] == 45
    original['num_iters'] = 918
    changed = tmp_path / 'changed.yaml'
    changed.write_text(yaml.safe_dump(original), encoding='utf-8')
    with pytest.raises(ValueError, match='recipe changed'):
        module.fixed_config(changed, tmp_path)


def test_candidate_cannot_rewrite_retained_text_or_validation(module, tmp_path):
    parent, candidate = tmp_path / 'parent', tmp_path / 'candidate'
    parent.mkdir()
    candidate.mkdir()
    rows = [{'audio': '/one.wav', 'text': 'frozen one'}, {'audio': '/two.wav', 'text': 'frozen two'}]
    lines = [(json.dumps(row) + '\r\n').encode() for row in rows]
    (parent / 'train.jsonl').write_bytes(b''.join(lines))
    (candidate / 'train.jsonl').write_bytes(lines[1])
    for folder in (parent, candidate):
        (folder / 'val.jsonl').write_bytes(b'{"text":"protected"}\r\n')
    (candidate / 'excluded.jsonl').write_text(json.dumps({'row': 1, 'sample': rows[0]}) + '\n',
                                             encoding='utf-8')
    assert module.verify_subset(parent, candidate) == 1
    (candidate / 'train.jsonl').write_text(json.dumps(rows[1]) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='more than declared'):
        module.verify_subset(parent, candidate)
    (candidate / 'train.jsonl').write_bytes(lines[1])
    (candidate / 'val.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='more than declared'):
        module.verify_subset(parent, candidate)


def test_r29_tasks_do_not_collide_with_r28_or_change_evaluation_budget(module):
    cases = [{'case_id': f'{lang}_{i}', 'lang': lang} for lang in ('th', 'tl', 'id') for i in range(60)]
    old = module.runner.tasks(cases)
    new = module.runner.tasks(cases, 'r29')
    assert len(new) == 900
    assert {identifier for identifier, _, _ in old}.isdisjoint(identifier for identifier, _, _ in new)
    assert [(case, seed) for _, case, seed in old] == [(case, seed) for _, case, seed in new]
    assert all(len(new[index::4]) == 225 for index in range(4))


def test_prepare_never_overwrites_an_existing_run(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
