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


def test_worker_uses_kimi_stream_json_without_permission_bypass(module):
    args=module.command('read only')
    assert args[1:]==['-p','read only','--output-format','stream-json']
    for flag in ('--approve-for-me','--sandbox','--yolo','--auto','--ignore-rules','--ignore-user-config'):
        assert flag not in args
    assert not any('danger' in arg for arg in args)


def test_kimi_fallback_matches_launchagent_home(module,monkeypatch):
    monkeypatch.setattr(module.shutil,'which',lambda _:None)
    args=module.command('read only')
    assert args[0]==str(module.Path.home()/'.kimi-code/bin/kimi')
    assert module.Path(args[0]).is_absolute()


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
    code=('import json,sys;print(json.dumps({"role":"assistant","content":sys.argv[1]}));'
          'print(json.dumps({"role":"meta","type":"session.resume_hint"}))')
    monkeypatch.setattr(module,'command',lambda prompt:[sys.executable,'-c',code,json.dumps(result)])
    real_sleep=module.time.sleep
    monkeypatch.setattr(module.time,'sleep',lambda _:real_sleep(.01))
    assert module.invoke(directory,'read only',probe=True)==result
    assert json.loads((directory/'result.json').read_text())==result
    prompt=(directory/'prompt.md').read_text()
    assert prompt.startswith('read only') and 'JSON Schema' in prompt and 'ssh_connected' in prompt
    events=(directory/'events.jsonl').read_text().splitlines()
    assert json.loads(events[0])['role']=='assistant'


def test_parse_result_uses_last_assistant_json_and_rejects_prose(module,tmp_path):
    events=[{'role':'assistant','content':'{"value":1}'},
            {'role':'meta','content':'ignored'}, {'role':'assistant','content':'{"value":2}'}]
    (tmp_path/'events.jsonl').write_text('\n'.join(json.dumps(event) for event in events)+'\n')
    assert module.parse_result(tmp_path)=={'value':2}
    (tmp_path/'events.jsonl').write_text('{"role":"assistant","content":"not json"}\n')
    with pytest.raises(json.JSONDecodeError):
        module.parse_result(tmp_path)
