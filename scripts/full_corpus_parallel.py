"""Continue a stopped full-corpus audit with eight processes, preserving completed ASR."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import full_corpus_clean as clean
from clean_candidate_run import stop
from fresh_baseline_eval import save_new
from ref_ab import runtime_environment, sha256
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.paths import ROOT

PREVIOUS = Path('/root/autodl-tmp/full_corpus_clean_20261029/run')
WORKERS = 8


def write_status(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def complete_records(path):
    lines = path.read_bytes().splitlines(keepends=True)
    records, fragment = [], None
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b'\n'):
                raise ValueError('Corrupt completed ASR record')
            fragment = {'bytes': len(line), 'note': 'Interrupted final write retained in original file; not a completed result.'}
    return records, fragment


def validate_records(records, eligible):
    seen = set()
    for row in records:
        task = eligible.get(row['id'])
        if (task is None or row['id'] in seen or row['audio'] != task['audio']
                or row['file_sha256'] != task['file_sha256']
                or row['reasons'] != clean.asr_reasons(row['result'])):
            raise ValueError('Duplicate, changed or out-of-scope ASR record')
        seen.add(row['id'])
    return seen


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    plan = clean.read_json(PREVIOUS / 'plan.json')
    clean.check_inputs(plan)
    tasks = _read_manifest(PREVIOUS / 'tasks.jsonl')
    acoustic = {row['audio']: row for row in _read_manifest(PREVIOUS / 'acoustic.jsonl')}
    if set(acoustic) != {task['audio'] for task in tasks}:
        raise ValueError('Acoustic stage incomplete')
    eligible = {task['id']: task for task in tasks if not acoustic[task['audio']]['reasons']}
    retained, fragment = complete_records(PREVIOUS / 'asr.jsonl')
    done = validate_records(retained, eligible)
    parent = clean.read_json(clean.PARENT / 'plan.json')
    _, _, inherited_inputs, inherited_pcm = clean.historical_exclusions(parent)
    work.mkdir(parents=True)
    shutil.copytree(PREVIOUS / 'manifests', work / 'manifests')
    for name in ('tasks.jsonl', 'protected_validation.jsonl', 'metadata_issues.json', 'acoustic.jsonl'):
        shutil.copyfile(PREVIOUS / name, work / name)
    _write_jsonl(retained, work / 'retained_asr.jsonl')
    _write_jsonl([task for task in tasks if task['id'] in eligible and task['id'] not in done], work / 'pending.jsonl')
    inputs = {**plan['inputs_sha256'], **inherited_inputs}
    for path in (PREVIOUS / 'plan.json', PREVIOUS / 'asr.jsonl', PREVIOUS / 'acoustic.jsonl',
                 *work.rglob('*.jsonl'), work / 'metadata_issues.json', *ROOT.joinpath('scripts').glob('*.py'),
                 *ROOT.joinpath('scripts').glob('*.sh'), *ROOT.joinpath('src').rglob('*.py')):
        inputs[str(path)] = sha256(path)
    save_new(work / 'plan.json', {**plan, 'inputs_sha256': inputs, 'inherited_pcm': inherited_pcm,
        'execution_workers': WORKERS, 'previous_work': str(PREVIOUS), 'retained_asr': len(retained),
        'eligible_asr': len(eligible), 'interrupted_fragment': fragment,
        'authorization': 'User: 要不要并行下gpu; eight independent GPU ASR processes',
        'handoff_note': 'Only execution concurrency changes; classifier and ASR parameters stay fixed. '
                        'Old completed ASR retained by ID/audio SHA, in-flight unrecorded work may be reassigned. '
                        'Also explicitly inherit historical exclusions embedded inside plans and their PCM aliases.'})
    clean.check_inputs(clean.read_json(work / 'plan.json'))


def worker(work, index):
    plan = clean.read_json(work / 'plan.json')
    if plan['policy'] != clean.POLICY or plan['execution_workers'] != WORKERS:
        raise ValueError('Frozen scope changed')
    directory = work / f'workers/{index}'
    directory.mkdir(parents=True, exist_ok=False)
    runtime_environment(directory / 'runtime')
    from faster_whisper import WhisperModel
    model = WhisperModel(plan['asr_path'], device='cuda', compute_type='int8_float16',
                         num_workers=1, cpu_threads=2, local_files_only=True)
    if model.model.device != 'cuda':
        raise RuntimeError('CUDA ASR required')
    tasks = _read_manifest(work / 'pending.jsonl')[index::WORKERS]
    save_new(directory / 'runtime.json', {'pid': os.getpid(), 'device': model.model.device,
        'compute_type': model.model.compute_type, 'cpu_threads': 2, 'files': len(tasks)})
    with (directory / 'items.jsonl').open('x', encoding='utf-8') as stream:
        for count, task in enumerate(tasks, 1):
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError('Disk below 4 GiB reserve')
            if sha256(task['audio']) != task['file_sha256']:
                raise ValueError('ASR input changed: ' + task['audio'])
            result = clean.transcribe(model, task['audio'], task['langs'][0], task['texts'])
            if sha256(task['audio']) != task['file_sha256']:
                raise ValueError('Audio changed during ASR: ' + task['audio'])
            record = {'id': task['id'], 'audio': task['audio'], 'file_sha256': task['file_sha256'],
                      'result': result, 'reasons': clean.asr_reasons(result)}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            write_status(directory / 'progress.json', {'completed': count, 'total': len(tasks),
                'last_id': task['id'], 'lang': task['langs'][0], 'time': time.time()})
            if count % 20 == 0:
                print(f'ASR worker{index} {count}/{len(tasks)}', flush=True)


def finish(work):
    plan = clean.read_json(work / 'plan.json')
    clean.check_inputs(plan)
    records = _read_manifest(work / 'retained_asr.jsonl')
    for index in range(WORKERS):
        records.extend(_read_manifest(work / f'workers/{index}/items.jsonl'))
    tasks = _read_manifest(work / 'tasks.jsonl')
    acoustic = {row['audio']: row for row in _read_manifest(work / 'acoustic.jsonl')}
    eligible = {task['id']: task for task in tasks if not acoustic[task['audio']]['reasons']}
    if validate_records(records, eligible) != eligible.keys():
        raise ValueError('Incomplete parallel ASR')
    if (work / 'asr.jsonl').exists():
        raise FileExistsError('Refusing merged ASR overwrite')
    _write_jsonl(sorted(records, key=lambda row: row['id']), work / 'asr.jsonl')
    clean.collect(work)


def run(work):
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Old GPU process must finish before handoff')
    prepare(work)
    plan = clean.read_json(work / 'plan.json')
    processes = []
    def status(stage, **extra):
        progress = [clean.read_json(path) for path in work.glob('workers/*/progress.json')]
        write_status(work / 'status.json', {'stage': stage, 'retained': plan['retained_asr'],
            'completed': plan['retained_asr'] + sum(row['completed'] for row in progress),
            'total': plan['eligible_asr'], 'time': time.time(), 'workers': WORKERS,
            'worker_pids': [p.pid for p in processes if p.poll() is None],
            'training_started': False, 'production_changed': False, **extra})
    try:
        for index in range(WORKERS):
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if free < 4 * 1024:
                raise RuntimeError('Insufficient GPU memory for another ASR process')
            with (work / f'worker{index}.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()),
                    'worker', '--work', str(work), '--index', str(index)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            while not (work / f'workers/{index}/runtime.json').exists() and process.poll() is None:
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('ASR worker failed during startup')
                status('starting_workers')
                time.sleep(2)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('ASR worker failed; preserve partial results')
            status('asr')
            time.sleep(10)
        if any(p.returncode for p in processes):
            raise RuntimeError('ASR worker failed')
        status('collecting')
        finish(work)
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise
    finally:
        for process in processes:
            stop(process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'worker'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=range(WORKERS))
    args = parser.parse_args()
    if args.action == 'worker':
        if args.index is None:
            parser.error('worker requires --index')
        worker(args.work.resolve(), args.index)
    else:
        run(args.work.resolve())
