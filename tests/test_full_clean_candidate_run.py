import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('full_clean_candidate_run')


def test_final_manifest_bytes_are_pinned(module):
    root = Path(__file__).resolve().parents[1]
    # The local candidate is intentionally ignored; the test needs no dataset download.
    assert set(module.FROZEN) == {'summary.json', 'candidate/current/train.jsonl', 'candidate/current/val.jsonl'}
    original = yaml.safe_load((root / 'configs/lora_joint3_r29_suspect_excluded_459.yaml').read_text())
    config = module.fixed_config(root / 'configs/lora_joint3_r29_suspect_excluded_459.yaml', Path('/new/run'))
    assert {key for key in config if config[key] != original[key]} == {
        'train_manifest', 'val_manifest', 'save_path', 'tensorboard'}


def test_unfinished_or_modified_cleaning_cannot_start(module, monkeypatch, tmp_path):
    candidate, clean = tmp_path / 'candidate', tmp_path / 'clean'
    candidate.mkdir()
    clean.mkdir()
    monkeypatch.setattr(module, 'CANDIDATE', candidate)
    monkeypatch.setattr(module, 'CLEAN', clean)
    summary = {'status': 'complete', 'selected_files': 62, 'parent_inputs_sha256': {}, 'outputs_sha256': {}}
    for path, value in ((candidate / 'summary.json', summary), (candidate / 'status.json', {'stage': 'complete'}),
                        (clean / 'status.json', {'stage': 'running'}), (candidate / 'flagged.json', [])):
        path.write_text(json.dumps(value), encoding='utf-8')
    plan = {'inputs_sha256': {str(candidate / 'flagged.json'): module.runner.frozen.sha256(candidate / 'flagged.json')}}
    (candidate / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    monkeypatch.setattr(module, 'FROZEN', {'summary.json': module.runner.frozen.sha256(candidate / 'summary.json')})
    with pytest.raises(ValueError, match='Completed full cleaning'):
        module.candidate_inputs()
    (clean / 'status.json').write_text('{"stage":"complete"}', encoding='utf-8')
    _, inputs = module.candidate_inputs()
    assert str(candidate / 'flagged.json') in inputs
    (candidate / 'flagged.json').write_text('[{}]', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        module.candidate_inputs()


def test_existing_run_is_never_overwritten(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
