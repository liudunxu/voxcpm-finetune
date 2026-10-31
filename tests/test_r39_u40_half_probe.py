import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('r39_u40_half_probe')


def test_half_probe_fronts_u40_failures_with_one_strength_and_no_grid(module):
    cases = [case for case in module.prior.frozen._read_manifest(
        Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
        if case['sentence_id'] in module.prior.SENTENCES]
    jobs = module.prior.midpoint_jobs(cases, risk_groups=module.RISK_GROUPS,
                                      prefix='half', strength=.5)
    assert len(jobs) == len({job['id'] for job in jobs}) == 30
    assert all(job['id'].startswith('half_') and job['strength'] == .5 for job in jobs)
    assert [(job['case']['sentence_id'], job['case']['condition']) for job in jobs[::5]] == [
        ('fresh_tl_1855', 'plain'), ('fresh_th_1689', 'control'), ('fresh_id_1973', 'plain'),
        ('fresh_id_1973', 'control'), ('fresh_th_1689', 'plain'), ('fresh_tl_1855', 'control')]
    assert [job['seed'] for job in jobs[:15]] == [
        43, 42, 44, 45, 46, 45, 42, 43, 44, 46, 45, 46, 42, 43, 44]
    assert {(job['case']['case_id'], job['seed']) for job in jobs} == {
        (case['case_id'], seed) for case in cases for seed in (42, 43, 44, 45, 46)}


def test_half_prepare_refuses_overwrite(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)


def test_shared_runner_requires_the_single_half_profile(module, tmp_path):
    (tmp_path / 'plan.json').write_text(
        '{"profile":"r39_u40_half","new_outputs":30,"reused_outputs":60,'
        '"training_steps":0,"max_workers":1,"lora_strength":0.75,"inputs_sha256":{}}')
    with pytest.raises(ValueError, match='single u40 half'):
        module.prior.run(tmp_path)
    assert not (tmp_path / 'run.started').exists()
