import importlib
import hashlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def module(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('quality_autopilot')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'WORK',tmp_path/'work/supervisor')
    return module


def test_worker_keeps_sandbox_and_automatic_review(module,tmp_path):
    args=module.command(tmp_path)
    assert '--approve-for-me' in args
    assert '--sandbox' not in args  # Mutually exclusive; approve-for-me selects workspace-write itself.
    assert '--ignore-rules' not in args and '--ignore-user-config' not in args
    assert not any('danger' in arg for arg in args)


def evidence(module):
    path=module.ROOT/'work/acceptance.json';path.parent.mkdir(parents=True)
    report=path.with_name('report.json');report.write_text('{"finished":true}')
    result={key:True for key in ('engineering_candidate_verified','independent_final_checked',
        'per_language_non_regression','stable_improvement','artifacts_verified')}
    result.update(production_changed=False,native_quality='unverified',
        by_lang={lang:{'independent_texts':30,'seeds':[42,43,44,45,46]} for lang in ('th','tl','id')},
        local_report_sha256={str(report):hashlib.sha256(report.read_bytes()).hexdigest()})
    path.write_text(json.dumps(result))
    return path,report,result


@pytest.mark.parametrize('failure',[None,'no_final','missing_lang','few_texts','repeated_seeds','tampered_report','no_report'])
def test_completion_requires_verified_artifacts_and_language_coverage(module,failure):
    path,report,result=evidence(module)
    if failure=='no_final':result['independent_final_checked']=False
    if failure=='missing_lang':del result['by_lang']['tl']
    if failure=='few_texts':result['by_lang']['th']['independent_texts']=29
    if failure=='repeated_seeds':result['by_lang']['id']['seeds']=[42]*5
    if failure=='tampered_report':report.write_text('changed')
    if failure=='no_report':result['local_report_sha256']={}
    path.write_text(json.dumps(result))
    if failure:
        with pytest.raises(ValueError):module.verify_completion(str(path))
    else:assert module.verify_completion(str(path))==result


@pytest.mark.parametrize('marker',['STOP','completed.json','approval_blocked.json'])
def test_terminal_or_stop_marker_prevents_new_worker(module,monkeypatch,marker):
    module.WORK.mkdir(parents=True);(module.WORK/marker).touch()
    monkeypatch.setattr(module,'invoke',lambda *args:pytest.fail('Unexpected worker launch'))
    module.run()


def test_supervisor_lock_prevents_duplicate_worker(module,monkeypatch):
    module.WORK.mkdir(parents=True)
    monkeypatch.setattr(module,'invoke',lambda *args:pytest.fail('Duplicate worker'))
    with (module.WORK/'supervisor.lock').open('a+') as lock:
        module.fcntl.flock(lock,module.fcntl.LOCK_EX|module.fcntl.LOCK_NB)
        module.run()


def test_probe_runs_a_real_child_and_preserves_structured_output(module,monkeypatch):
    result={'local_head':'abc','ssh_connected':True,'gpu_available':True,'no_mutations':True,'summary':'ok'}
    directory=module.WORK/'probe'
    code='import pathlib,sys;sys.stdin.read();pathlib.Path(sys.argv[1]).write_text(sys.argv[2])'
    monkeypatch.setattr(module,'command',lambda path,probe=False:
        [sys.executable,'-c',code,str(path/'result.json'),json.dumps(result)])
    real_sleep=module.time.sleep
    monkeypatch.setattr(module.time,'sleep',lambda _:real_sleep(.01))
    assert module.invoke(directory,'read only',probe=True)==result
    assert (directory/'events.jsonl').exists() and (directory/'prompt.md').read_text()=='read only'
