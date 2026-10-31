import hashlib
import importlib
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('r39_midpoint_probe')


def test_midpoint_jobs_front_complementary_risks_without_changing_coverage(module):
    cases = [case for case in module.frozen._read_manifest(
        Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
        if case['sentence_id'] in module.SENTENCES]
    jobs = module.midpoint_jobs(list(reversed(cases)))
    assert len(jobs) == len({job['id'] for job in jobs}) == 30
    assert {(job['case']['case_id'], job['seed']) for job in jobs} == {
        (case['case_id'], seed) for case in cases for seed in (42, 43, 44, 45, 46)}
    assert [(job['case']['sentence_id'], job['case']['condition']) for job in jobs[::5]] == [
        ('fresh_id_1973', 'plain'), ('fresh_th_1689', 'control'), ('fresh_tl_1855', 'plain'),
        ('fresh_id_1973', 'control'), ('fresh_th_1689', 'plain'), ('fresh_tl_1855', 'control')]
    assert [job['seed'] for job in jobs[:15]] == [
        45, 46, 42, 43, 44, 45, 42, 43, 44, 46, 43, 42, 44, 45, 46]
    assert all(job['strength'] == 1. and job['model'] == 'candidate' for job in jobs)


def test_midpoint_is_float32_mean_and_preserves_sources(module, tmp_path):
    left, right = tmp_path / 'u40', tmp_path / 'u80'
    left.mkdir(); right.mkdir()
    config = b'{"lora_config":{"rank":32}}\n'
    (left / 'lora_config.json').write_bytes(config)
    (right / 'lora_config.json').write_bytes(config)
    names = [f'layer{i:02d}' for i in range(96)]
    save_file({name: torch.ones(2, dtype=torch.float64) for name in names}, str(left / 'lora_weights.safetensors'))
    save_file({name: torch.full((2,), 3, dtype=torch.float64) for name in names}, str(right / 'lora_weights.safetensors'))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (left / 'lora_weights.safetensors', right / 'lora_weights.safetensors')}
    result = module.write_midpoint(left, right, tmp_path / 'mid')
    merged = load_file(str(tmp_path / 'mid/lora_weights.safetensors'))
    assert result['tensors'] == 96 and not result['actual_optimizer_checkpoint']
    assert (tmp_path / 'mid/lora_config.json').read_bytes() == config
    assert all(tensor.dtype == torch.float32 and torch.equal(tensor, torch.full((2,), 2.))
               for tensor in merged.values())
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    with pytest.raises(FileExistsError):
        module.write_midpoint(left, right, tmp_path / 'mid')


def test_midpoint_requires_identical_config_and_tensors(module, tmp_path):
    left, right = tmp_path / 'u40', tmp_path / 'u80'
    left.mkdir(); right.mkdir()
    (left / 'lora_config.json').write_bytes(b'left')
    (right / 'lora_config.json').write_bytes(b'right')
    with pytest.raises(ValueError, match='configuration'):
        module.write_midpoint(left, right, tmp_path / 'mid')
    assert not (tmp_path / 'mid').exists()


def test_midpoint_inherits_new_overread_rejection(module):
    old = {'case_id': 'fresh_tl_1855_plain', 'seed': 43, 'hyp': 'base', 'cer': .0695,
           'over_read': False, 'rms_dbfs': -30, 'speech_ratio': .86}
    row = {**old, 'hyp': 'repeated', 'cer': 1.0909, 'over_read': True}
    reason, _ = module.midpoint_rejection([row], {module.probe.key(old): old})
    assert reason == 'new_suspected_overread'
