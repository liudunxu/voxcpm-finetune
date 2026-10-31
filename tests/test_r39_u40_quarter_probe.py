import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('r39_u40_quarter_probe')


def test_quarter_probe_fronts_new_half_failure_and_preserves_prior_risks(module):
    selected = {(sid, condition) for sid, condition, _seeds in module.RISK_GROUPS}
    cases = [case for case in module.prior.frozen._read_manifest(
        Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
        if (case['sentence_id'], case['condition']) in selected]
    jobs = module.quarter_jobs(list(reversed(cases)))
    assert len(jobs) == len({job['id'] for job in jobs}) == 20
    assert all(job['id'].startswith('quarter_') and job['strength'] == .25 for job in jobs)
    assert [(job['case']['sentence_id'], job['case']['condition']) for job in jobs[::5]] == [
        ('fresh_id_1816', 'control'), ('fresh_tl_1855', 'plain'),
        ('fresh_th_1689', 'control'), ('fresh_id_1973', 'plain')]
    assert [job['seed'] for job in jobs[:5]] == [45, 42, 43, 44, 46]
    assert {(job['case']['case_id'], job['seed']) for job in jobs} == {
        (case['case_id'], seed) for case in cases for seed in (42, 43, 44, 45, 46)}


def test_quarter_jobs_reject_extra_conditions(module):
    cases = [case for case in module.prior.frozen._read_manifest(
        Path('eval_cases/fresh_text_holdout_20261028.jsonl'))
        if case['sentence_id'] == 'fresh_id_1816']
    with pytest.raises(ValueError, match='exactly four'):
        module.quarter_jobs(cases)


def test_quarter_prepare_refuses_overwrite(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)


def test_shared_runner_requires_the_single_quarter_profile(module, tmp_path):
    (tmp_path / 'plan.json').write_text(
        '{"profile":"r39_u40_quarter","new_outputs":20,"reused_outputs":60,'
        '"training_steps":0,"max_workers":1,"lora_strength":0.5,"inputs_sha256":{}}')
    with pytest.raises(ValueError, match='single u40 quarter'):
        module.prior.run(tmp_path)
    assert not (tmp_path / 'run.started').exists()
