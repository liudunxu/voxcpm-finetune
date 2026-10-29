"""Exclude the user's frozen 62 music-score suspects; preserve original audit artifacts."""
import argparse
from collections import Counter
from pathlib import Path
import time

from audit_training_quality import CLEAN, read
from audit_music import EVENTS
from control_ab_gate import pcm_conflicts
from fresh_baseline_eval import save_new
from full_corpus_parallel import write_status
from incremental_mix import audit_mix, describe_mix, target_text
from ref_ab import runtime_environment, sha256
from review_training_silence import exclude_flagged
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.eval import _norm
from voxft.paths import ROOT

MUSIC = Path('/root/autodl-tmp/music_audit_20261029')


def verify(inputs):
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError('Frozen quarantine input changed: ' + path)


def selected_flags(summary, tasks, acoustic):
    flags = []
    for item in summary['review_priority']:
        case = item['case']
        path = case['audio']
        task, measured = tasks[path], acoustic[path]
        if (max(item['scores'][label] for label in EVENTS) < .8
                or not measured['pcm_sha256'] or not task['origins']
                or any(case[key] != measured[key] for key in ('file_sha256', 'pcm_sha256'))
                or task['file_sha256'] != case['file_sha256']):
            raise ValueError('Music priority evidence differs from source audit')
        flags.append({**case, 'scores': item['scores'],
            'uses': [{'origin_audio': origin} for origin in task['origins']],
            'status': 'excluded_per_user_suspect_not_confirmed_music'})
    if len(flags) != 62 or len({r['audio'] for r in flags}) != 62:
        raise ValueError('User authorized the frozen set of 62 files')
    return flags


def filter_manifest(path, output, flagged, acoustic):
    rows = _read_manifest(path)
    positions, removed = exclude_flagged(rows, flagged, acoustic)
    kept = [rows[i-1] for i in positions]
    if exclude_flagged(kept, flagged, acoustic)[1]:
        raise ValueError('Music suspect leaked into retained rows')
    lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
    if len(lines) != len(rows):
        raise ValueError('Manifest line accounting differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(b''.join(lines[i-1] for i in positions))
    for row in removed:
        row['reasons'].append('music_score_suspect_user_excluded')
    _write_jsonl(removed, output.with_name(output.stem + '_excluded.jsonl'))
    summary, shares = describe_mix(kept)
    return {'before': len(rows), 'excluded': len(removed), 'summary': summary,
            'language_shares': shares,
            'reasons': dict(Counter(reason for row in removed for reason in row['reasons']))}


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    music_status = read(MUSIC/'run/status.json')
    main = read(CLEAN/'plan.json')
    postcheck = read(MUSIC/'postcheck.json')
    if (music_status['stage'] != 'complete' or music_status['completed'] != 8312
            or postcheck['priority_files'] != 62 or not postcheck['report_aggregation_verified']):
        raise ValueError('Completed music audit required')
    verify(postcheck['files_sha256'])
    inputs = {str(MUSIC/'postcheck.json'): sha256(MUSIC/'postcheck.json'),
              str(MUSIC/'run/plan.json'): sha256(MUSIC/'run/plan.json'),
              str(MUSIC/'run/summary.json'): sha256(MUSIC/'run/summary.json'),
              str(CLEAN/'plan.json'): sha256(CLEAN/'plan.json')}
    for name in ('acoustic.jsonl', 'tasks.jsonl', 'protected_validation.jsonl',
                 'manifests/current/train.jsonl', 'manifests/current/val.jsonl'):
        path = CLEAN/name
        inputs[str(path)] = main['inputs_sha256'][str(path)]
    for path, digest in main['inputs_sha256'].items():
        if Path(path).parent.name == 'eval_cases' and path.endswith('.jsonl'):
            inputs[path] = digest
    tasks = {r['audio']: r for r in _read_manifest(CLEAN/'tasks.jsonl')}
    acoustic = {r['audio']: r for r in _read_manifest(CLEAN/'acoustic.jsonl')}
    flags = selected_flags(read(MUSIC/'run/summary.json'), tasks, acoustic)
    for item in flags:
        inputs[item['audio']] = item['file_sha256']
    for directory in (ROOT/'scripts', ROOT/'src'):
        for pattern in ('*.py', '*.sh'):
            inputs.update({str(p): sha256(p) for p in directory.rglob(pattern)})
    verify(inputs)
    work.mkdir()
    save_new(work/'flagged.json', flags)
    inputs[str(work/'flagged.json')] = sha256(work/'flagged.json')
    save_new(work/'plan.json', {'inputs_sha256': inputs, 'selected_files': len(flags),
        'main_work': str(CLEAN), 'music_work': str(MUSIC),
        'authorization': 'User: 直接剔除就行了 (all 62 frozen music-score suspects, including 52 ASR-unflagged)',
        'policy': 'Exclude this frozen set and target/ref path, original-audio and exact-PCM reuse. '
                  'No listening prerequisite; uncertainty is retained. This does not calibrate AST scores '
                  'or expand into a new general music classifier gate.',
        'source_mutations': False, 'training_started': False, 'production_changed': False})
    outputs = {split: filter_manifest(CLEAN/f'manifests/current/{split}.jsonl',
               work/f'current_music_only/{split}.jsonl', flags, acoustic) for split in ('train', 'val')}
    verify(inputs)
    save_new(work/'current_music_only/summary.json', {
        'status': 'music_excluded_waiting_for_full_corpus_cleaning', 'outputs': outputs,
        'training_approved': False, 'training_started': False,
        'note': 'Music-only staging view. Main acoustic/ASR/metadata exclusions still need composition.',
        'outputs_sha256': {str(p.relative_to(work)): sha256(p)
                           for p in (work/'current_music_only').glob('*.jsonl')}})
    write_status(work/'status.json', {'stage': 'waiting_for_main', 'selected_files': 62,
        'current_music_only_removed': outputs['train']['excluded'], 'time': time.time(),
        'training_started': False, 'production_changed': False})
    print(f"EXCLUDED current music-only: {outputs['train']['excluded']} training rows", flush=True)


def finish(work):
    plan = read(work/'plan.json')
    verify(plan['inputs_sha256'])
    if read(CLEAN/'status.json')['stage'] != 'complete':
        raise RuntimeError('Main cleaning must complete before candidate composition')
    parent = read(CLEAN/'summary.json')
    if parent['status'] != 'complete' or parent['training_started']:
        raise ValueError('Unexpected parent candidate status')
    if (work/'candidate').exists() or (work/'summary.json').exists():
        raise FileExistsError('Refuse final candidate overwrite')
    parent_inputs = {str(CLEAN/'summary.json'): sha256(CLEAN/'summary.json')}
    parent_inputs.update({str(CLEAN/name): digest for name, digest in parent['outputs_sha256'].items()})
    verify(parent_inputs)
    flags = read(work/'flagged.json')
    acoustic = {r['audio']: r for r in _read_manifest(CLEAN/'acoustic.jsonl')}
    protected = _read_manifest(CLEAN/'protected_validation.jsonl')
    excluded = {_norm(target_text(r)) for p in plan['inputs_sha256']
                if Path(p).parent.name == 'eval_cases' and p.endswith('.jsonl')
                for r in _read_manifest(Path(p)) if r.get('text')}
    outputs, pool = {}, []
    for key in sorted(parent['outputs']):
        source, split = key.split('/')
        output = work/f'candidate/{source}/{split}.jsonl'
        outputs[key] = filter_manifest(CLEAN/f'candidate/{source}/{split}.jsonl', output, flags, acoustic)
        if source != 'current' and split == 'train':
            pool.extend(_read_manifest(output))
    checks = {}
    for name, rows in (('pool', pool), ('current', _read_manifest(work/'candidate/current/train.jsonl'))):
        checks[name] = {'metadata': audit_mix(rows, protected, excluded, max_dur=30),
                        'pcm': pcm_conflicts(rows, protected, acoustic)}
        if (not rows or checks[name]['metadata']['violations'] or checks[name]['metadata']['origins_above_3x']
                or any(checks[name]['pcm'].values()) or exclude_flagged(rows, flags, acoustic)[1]):
            raise ValueError('Composed candidate failed isolation: ' + name)
    verify({**plan['inputs_sha256'], **parent_inputs})
    save_new(work/'summary.json', {'status': 'complete', 'selected_files': 62, 'outputs': outputs,
        'checks': checks, 'parent_inputs_sha256': parent_inputs,
        'outputs_sha256': {str(p.relative_to(work)): sha256(p) for p in (work/'candidate').rglob('*.jsonl')},
        'source_mutations': False, 'training_started': False, 'training_approved': False,
        'production_changed': False, 'native_quality': 'unverified',
        'note': 'Use this candidate tree for future mixes; it subsets the full-cleaning candidate. '
                'Preserve all prior exclusions and inherit flagged.json paths/origins/PCM. '
                'Retained row bytes and original split unchanged; full old validation remains protected. '
                'All 62 excluded as suspects per user, not certified music contamination.'})
    write_status(work/'status.json', {'stage': 'complete', 'selected_files': 62, 'time': time.time(),
        'additional_current_rows_removed': outputs['current/train']['excluded'],
        'training_started': False, 'production_changed': False})
    print('COMPLETE music exclusions composed with main cleaning', flush=True)


def watch(work):
    plan = read(work/'plan.json')
    verify(plan['inputs_sha256'])
    deadline = time.monotonic() + 4*3600
    while True:
        status = read(CLEAN/'status.json')
        if status['stage'] == 'complete':
            finish(work)
            return
        if status['stage'] == 'failed' or time.monotonic() > deadline:
            raise RuntimeError('Main cleaning failed or exceeded the four-hour wait; preserve outputs')
        write_status(work/'status.json', {'stage': 'waiting_for_main', 'selected_files': 62,
            'main_completed': status['completed'], 'main_total': status['total'], 'time': time.time(),
            'training_started': False, 'production_changed': False})
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'watch', 'finish'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    runtime_environment(work.parent/'runtime')
    try:
        if args.action == 'prepare':
            prepare(work)
        elif args.action == 'watch':
            watch(work)
        else:
            finish(work)
    except BaseException as error:
        if work.exists():
            write_status(work/'status.json', {'stage': 'failed', 'error': str(error), 'time': time.time(),
                'training_started': False, 'production_changed': False})
        raise
