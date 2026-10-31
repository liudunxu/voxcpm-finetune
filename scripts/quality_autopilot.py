"""Persistent local supervisor for the user-authorized remote quality work."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work/quality_autopilot_20261031'
LABEL = 'com.voxft.quality-autopilot'
PROMPT = ROOT / 'docs/quality_autopilot_instructions.md'
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'status': {'type': 'string', 'enum': ['continue', 'waiting', 'completed', 'external_block', 'approval_blocked']},
        'summary': {'type': 'string'}, 'next_action': {'type': 'string'},
        'wait_seconds': {'type': 'integer'}, 'evidence_path': {'type': 'string'},
        'remote_jobs': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['status', 'summary', 'next_action', 'wait_seconds', 'evidence_path', 'remote_jobs'],
}
PROBE_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'local_head': {'type': 'string'}, 'ssh_connected': {'type': 'boolean'},
        'gpu_available': {'type': 'boolean'}, 'no_mutations': {'type': 'boolean'},
        'summary': {'type': 'string'}},
    'required': ['local_head', 'ssh_connected', 'gpu_available', 'no_mutations', 'summary'],
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f'.{os.getpid()}.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(pending, path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def command(prompt):
    executable = shutil.which('kimi') or str(Path.home() / '.kimi-code/bin/kimi')
    return [executable, '-p', prompt, '--output-format', 'stream-json']


def parse_result(directory):
    contents = []
    for line in (directory / 'events.jsonl').read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event.get('role') == 'assistant' and isinstance(event.get('content'), str) and event['content'].strip():
            contents.append(event['content'].strip())
    if not contents:
        raise ValueError('Kimi returned no assistant result')
    result = json.loads(contents[-1])
    if not isinstance(result, dict):
        raise ValueError('Kimi final result must be a JSON object')
    return result


def invoke(directory, prompt, probe=False):
    directory.mkdir(parents=True)
    schema = PROBE_SCHEMA if probe else SCHEMA
    write(directory / 'schema.json', schema)
    prompt += ('\n\n最终回复只输出一个符合以下 JSON Schema 的 JSON 对象；不要 Markdown、解释或其他文字：\n'
               + json.dumps(schema, ensure_ascii=False))
    (directory / 'prompt.md').write_text(prompt, encoding='utf-8')
    with (directory / 'events.jsonl').open('xb') as events, (directory / 'stderr.log').open('xb') as errors:
        child = subprocess.Popen(command(prompt), cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=events, stderr=errors, start_new_session=True)
        state = {'stage': 'probing' if probe else 'working', 'supervisor_pid': os.getpid(),
                 'worker_pid': child.pid, 'turn': str(directory), 'started_at': time.time()}
        write(WORK / 'status.json', state)
        try:
            while child.poll() is None:
                write(WORK / 'status.json', {**state, 'heartbeat': time.time()})
                if (WORK / 'STOP').exists():
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                    raise InterruptedError('User stop marker; existing remote nohup jobs preserved')
                time.sleep(5)
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            raise
    if child.returncode:
        raise RuntimeError(f'Worker exited {child.returncode}; retain events in {directory}')
    result = parse_result(directory)
    write(directory / 'result.json', result)
    if probe:
        if set(result) != set(PROBE_SCHEMA['required']) or not all(
                result.get(k) is True for k in ('ssh_connected', 'gpu_available', 'no_mutations')):
            raise RuntimeError('Read-only end-to-end probe failed')
    else:
        if set(result) != set(SCHEMA['required']) or result['status'] not in SCHEMA['properties']['status']['enum']:
            raise ValueError('Invalid worker result')
        if type(result['wait_seconds']) is not int or not isinstance(result['remote_jobs'], list):
            raise ValueError('Invalid worker wait/job fields')
    return result


def verify_completion(evidence_path):
    """A worker's closing sentence alone must never finish continuous iteration."""
    path = Path(evidence_path)
    if not path.is_absolute(): path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT / 'work'):
        raise ValueError('Completion evidence must be inside project work/')
    evidence = read(path)
    for key in ('engineering_candidate_verified', 'independent_final_checked',
                'per_language_non_regression', 'stable_improvement', 'artifacts_verified'):
        if evidence.get(key) is not True: raise ValueError('Incomplete acceptance: ' + key)
    if evidence.get('production_changed') is not False or evidence.get('native_quality') not in ('unverified', 'verified'):
        raise ValueError('Completion must preserve production and native-review boundaries')
    if set(evidence.get('by_lang', {})) != {'th', 'tl', 'id'}:
        raise ValueError('Missing target-language acceptance')
    for value in evidence['by_lang'].values():
        if value.get('independent_texts', 0) < 30 or len(set(value.get('seeds', []))) < 5:
            raise ValueError('Insufficient independent text/seed coverage')
    artifacts = evidence.get('local_report_sha256', {})
    if not artifacts: raise ValueError('Missing frozen acceptance reports')
    for filename, digest in artifacts.items():
        report = Path(filename)
        if not report.is_absolute(): report = ROOT / report
        if not report.resolve().is_relative_to(ROOT / 'work'):
            raise ValueError('Acceptance report outside work/')
        if hashlib.sha256(report.read_bytes()).hexdigest() != digest:
            raise ValueError('Acceptance report changed')
    return evidence


def run():
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'supervisor.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        if (WORK / 'STOP').exists() or (WORK / 'completed.json').exists() or (WORK / 'approval_blocked.json').exists():
            return
        failures = 0
        while not (WORK / 'STOP').exists():
            directory = WORK / 'turns' / f'{time.time_ns()}'
            prompt = PROMPT.read_text(encoding='utf-8')
            previous = WORK / 'last_result.json'
            if previous.exists():
                prompt += '\n\n上一轮后台交接（须核对实际文件与进程，不凭摘要重跑）：\n' + previous.read_text(encoding='utf-8')
            state_path = WORK / 'status.json'
            if state_path.exists() and read(state_path).get('stage') == 'retry_backoff':
                prompt += '\n\n监督器发现的上轮执行/验收异常，必须先处理：\n' + state_path.read_text(encoding='utf-8')
            try:
                result = invoke(directory, prompt)
                write(previous, result)
                failures = 0
                if result['status'] == 'completed':
                    evidence = verify_completion(result['evidence_path'])
                    write(WORK / 'completed.json', {'result': result, 'evidence': evidence, 'time': time.time()})
                    write(WORK / 'status.json', {'stage': 'completed', 'result': result, 'time': time.time()})
                    return
                if result['status'] == 'approval_blocked':
                    write(WORK / 'approval_blocked.json', result)
                    write(WORK / 'status.json', {'stage': 'approval_blocked', 'result': result, 'time': time.time()})
                    return
                minimum = 300 if result['status'] in ('waiting', 'external_block') else 30
                delay = min(1800, max(minimum, result['wait_seconds']))
                state = {'stage': result['status'], 'result': result, 'wake_at': time.time() + delay}
            except InterruptedError:
                break
            except Exception as error:
                failures += 1
                delay = min(1800, 60 * 2 ** min(failures, 5))
                state = {'stage': 'retry_backoff', 'error': str(error), 'consecutive_errors': failures,
                         'turn': str(directory), 'wake_at': time.time() + delay}
            write(WORK / 'status.json', {**state, 'supervisor_pid': os.getpid(), 'time': time.time()})
            while time.time() < state['wake_at'] and not (WORK / 'STOP').exists():
                time.sleep(min(5, max(.1, state['wake_at'] - time.time())))
        write(WORK / 'status.json', {'stage': 'stopped_by_user', 'time': time.time()})


def probe():
    prompt = ('用户已授权自动继续本项目远程微调。现在只验证后台执行链，不修改任何项目文件、远程数据或进程。'
              '只执行 git rev-parse HEAD，以及 ssh -p 51348 -o BatchMode=yes -o ConnectTimeout=8 '
              'root@connect.bjb1.seetacloud.com 读取 hostname 和 nvidia-smi --query-gpu=name,memory.total '
              '--format=csv,noheader。不要访问/显示任何密钥、.env或认证文件，不下载内容，不发消息。'
              '按schema返回真实结果。连接因沙箱受限时走自动审批；自动审批拒绝就如实失败，不能绕过。')
    try:
        result = invoke(WORK / 'probes' / str(time.time_ns()), prompt, probe=True)
    except Exception as error:
        write(WORK / 'status.json', {'stage': 'probe_failed', 'error': str(error), 'time': time.time()})
        raise
    write(WORK / 'probe_success.json', {'result': result, 'time': time.time()})
    write(WORK / 'status.json', {'stage': 'probe_passed', 'time': time.time()})
    print(json.dumps(result, ensure_ascii=False))


def install():
    if sys.platform != 'darwin': raise RuntimeError('This supervisor uses macOS launchd')
    if not (WORK / 'probe_success.json').exists(): raise RuntimeError('Run the read-only probe first')
    plist = Path.home() / 'Library/LaunchAgents' / f'{LABEL}.plist'
    plist.parent.mkdir(parents=True, exist_ok=True)
    if plist.exists(): raise FileExistsError('Do not replace an existing supervisor')
    value = {'Label': LABEL, 'ProgramArguments': [str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()), 'run'],
             'WorkingDirectory': str(ROOT), 'RunAtLoad': True, 'StartInterval': 60,
             'ProcessType': 'Background', 'StandardOutPath': str(WORK / 'supervisor.log'),
             'StandardErrorPath': str(WORK / 'supervisor.stderr.log'),
             'EnvironmentVariables': {'PATH': f"{Path.home()/'.kimi-code/bin'}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                                      'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'}}
    with plist.open('xb') as stream: plistlib.dump(value, stream)
    os.chmod(plist, 0o600)
    subprocess.run(['launchctl', 'bootstrap', f'gui/{os.getuid()}', str(plist)], check=True)
    write(WORK / 'installation.json', {'label': LABEL, 'plist': str(plist), 'time': time.time(),
          'policy': 'Local user LaunchAgent, single Kimi worker; non-interactive auto policy with static deny rules; no auth transfer.'})
    print('INSTALLED', LABEL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('probe', 'run', 'install', 'status'))
    args = parser.parse_args()
    if args.action == 'status':
        print(json.dumps(read(WORK / 'status.json'), ensure_ascii=False, indent=2))
    else: globals()[args.action]()
