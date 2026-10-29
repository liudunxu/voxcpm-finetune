"""Conservatively clean every existing source used by r29; preserve frozen experiments."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from audit_training_silence import measure
from clean_candidate_run import ENCODING, read_json
from control_ab_gate import pcm_conflicts, verify_provenance
from fresh_baseline_eval import save_new
from incremental_mix import audit_mix, describe_mix, load_exclusions, row_issues, target_text
from ref_ab import memberships, runtime_environment, sha256
from review_training_silence import transcribe
from training_pool import inspect_audio
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.data.registry import get_source
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED, DATA_RAW, ROOT

PARENT = Path('/root/autodl-tmp/suspect_candidate_20261029/run')
POLICY = {'max_cer': .15, 'min_avg_logprob': -1., 'max_no_speech_prob': .6,
          'max_tail_seconds': .5, 'max_head_seconds': .5, 'clipping_run_ms': 1., 'workers': 4,
          'vad_filter': False, 'initial_prompt': None, 'beam_size': 5, 'temperature': 0,
          'condition_on_previous_text': False, 'asr_model': 'large-v3',
          'asr_tl_language_hint': None}
ROW_ONLY = {'evaluation_or_validation_text', 'validation_audio_speaker_or_session',
            'unverified_or_same_origin_reference', 'unverified_control_prefix'}


def check_inputs(plan, audio=False):
    if plan['policy'] != POLICY:
        raise ValueError('Cleaning policy changed')
    for path, digest in plan['inputs_sha256'].items():
        if audio or not path.endswith('.wav'):
            if sha256(path) != digest:
                raise ValueError(f'Frozen input changed: {path}')


def metadata_reasons(row, protected, excluded, raw):
    reasons = row_issues(row, protected, excluded, max_dur=30)
    text = target_text(row)
    if ENCODING.search(text):
        reasons.append('encoding_artifact')
    if any(ord(char) < 32 and char not in '\n\t\r' for char in text):
        reasons.append('control_character_in_text')
    if row.get('control') and (not row['text'].startswith(f"({row['control']})") or not (
            row.get('emotion_verified') is True or row.get('control_verified') is True)):
        reasons.append('unverified_control_prefix')
    try:
        verify_provenance(row, raw)
    except (ValueError, KeyError) as error:
        reasons.append('provenance:' + str(error))
    return sorted(set(reasons))


def historical_exclusions(parent):
    inherited, validation, inputs = [], [], {}
    for path, digest in parent['inputs_sha256'].items():
        name = Path(path).name
        if name not in {'val.jsonl', 'excluded.jsonl', 'flagged.json', 'plan.json'}:
            continue
        if sha256(path) != digest:
            raise ValueError(f'Inherited record changed: {path}')
        inputs[path] = digest
        if name == 'val.jsonl':
            validation.extend(_read_manifest(Path(path)))
        elif name == 'excluded.jsonl':
            inherited.extend(record['sample'] for record in _read_manifest(Path(path)))
        elif name == 'plan.json':
            inherited.extend(record['sample'] for record in read_json(path).get('additional_excluded', []))
        else:
            for item in read_json(path):
                inherited.append({'audio': item['audio']})
                inherited.extend({'audio': item['audio'], 'origin_audio': use.get('origin_audio')}
                                 for use in item.get('uses', []))
    pcm = set()
    if inherited:
        cache, = [path for path in parent['inputs_sha256'] if path.endswith('/cpu/audio_audit.json')]
        inputs[cache] = parent['inputs_sha256'][cache]
        if sha256(cache) != inputs[cache]:
            raise ValueError('Historical PCM audit changed')
        files = read_json(cache)['files']
        for row in inherited:
            if row['audio'] not in files:
                raise ValueError('Missing historical PCM: ' + row['audio'])
            pcm.add(files[row['audio']]['pcm_sha256'])
    return inherited, validation, inputs, sorted(pcm)


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent = read_json(PARENT / 'plan.json')
    excluded, inputs = load_exclusions(ROOT / 'eval_cases')
    inputs[str(PARENT / 'plan.json')] = sha256(PARENT / 'plan.json')
    sources, manifests, raw, protected = {}, {}, {}, []
    paths = []
    for source in sorted(parent['sources']):
        definition = get_source(source)
        if definition.license not in {'CC0', 'CC-BY-3.0', 'CC-BY-4.0', 'Apache-2.0'}:
            raise ValueError(f'Unapproved source: {source}')
        sources[source] = {'license': definition.license}
        origin = DATA_RAW / source / 'manifest.jsonl'
        inputs[str(origin)] = sha256(origin)
        raw.update({str(Path(row['audio']).resolve()): row for row in _read_manifest(origin)})
        paths.extend((f'{source}/{split}', DATA_PROCESSED / source / f'{split}.jsonl')
                     for split in ('train', 'val'))
    paths.extend((f'current/{split}', PARENT / f'{split}.jsonl') for split in ('train', 'val'))
    for key, path in paths:
        inputs[str(path)] = sha256(path)
        if str(path) in parent['inputs_sha256'] and inputs[str(path)] != parent['inputs_sha256'][str(path)]:
            raise ValueError(f'Parent manifest changed: {path}')
        manifests[key] = {'path': str(path), 'rows': _read_manifest(path)}
        if key.endswith('/val'):
            protected.extend(manifests[key]['rows'])
    inherited, old_validation, old_inputs, inherited_pcm = historical_exclusions(parent)
    inputs.update(old_inputs)
    protected.extend(row for row in old_validation if row.get('source_id') in sources)
    protected = list({(r['audio'], r.get('ref_audio'), r['text']): r for r in protected}.values())
    protected_keys = memberships(protected)
    protected_texts = {_norm(target_text(row)) for row in protected}
    tasks, issues = {}, {}
    for key, manifest in manifests.items():
        issues[key] = []
        for position, row in enumerate(manifest['rows'], 1):
            reasons = metadata_reasons(row, protected_keys if key.endswith('/train') else set(),
                                       excluded | protected_texts if key.endswith('/train') else excluded, raw)
            if reasons:
                issues[key].append({'row': position, 'reasons': reasons})
    index = {row['audio']: row for manifest in manifests.values() for row in manifest['rows']}
    for row in protected:
        index.setdefault(row['audio'], row)
    for key, manifest in {**manifests, 'protected/val': {'rows': protected}}.items():
        reason_map = {r['row']: r['reasons'] for r in issues.get(key, [])}
        for position, row in enumerate(manifest['rows'], 1):
            for field, prefix in (('audio', ''), ('ref_audio', 'ref_')):
                if not row.get(field):
                    continue
                path = row[field]
                task = tasks.setdefault(path, {'audio': path, 'durations': [], 'langs': [],
                    'texts': [], 'origins': [], 'uses': [], 'pre_reasons': []})
                task['durations'].append(row.get(prefix + 'duration'))
                task['langs'].append(row.get(prefix + 'lang'))
                if row.get(prefix + 'origin_audio'):
                    task['origins'].append(str(Path(row[prefix + 'origin_audio']).resolve()))
                text_row = row if not prefix else index.get(path)
                if text_row:
                    task['texts'].append(target_text(text_row))
                else:
                    task['pre_reasons'].append('missing_reference_transcript')
                task['uses'].append({'manifest': key, 'row': position, 'role': field})
                if not prefix:
                    task['pre_reasons'].extend(r for r in reason_map.get(position, []) if r not in ROW_ONLY)
    blocked = {str(Path(row[field]).resolve()) for row in inherited
               for field in ('audio', 'origin_audio') if row.get(field)}
    ordered = sorted(tasks.values(), key=lambda item: ('id' not in item['langs'],
        not any(use['manifest'].startswith('current/') for use in item['uses']), item['audio']))
    for number, task in enumerate(ordered):
        task['id'] = number
        for field in ('durations', 'langs', 'texts', 'origins', 'pre_reasons'):
            task[field] = list(dict.fromkeys(task[field]))
        if len(task['langs']) != 1 or task['langs'][0] not in {'id', 'th', 'tl', 'zh', 'en'}:
            task['pre_reasons'].append('conflicting_or_unsupported_language')
        if len({_norm(text) for text in task['texts']}) != 1:
            task['pre_reasons'].append('conflicting_transcripts')
        try:
            durations = [float(value) for value in task['durations']]
            valid_durations = bool(durations) and all(math.isfinite(value) for value in durations) and max(durations) - min(durations) <= .03
        except (TypeError, ValueError):
            valid_durations = False
        if not valid_durations:
            task['pre_reasons'].append('conflicting_durations')
        if {str(Path(task['audio']).resolve()), *task['origins']} & blocked:
            task['pre_reasons'].append('inherited_exclusion')
        path = Path(task['audio'])
        if not path.is_file():
            task['file_sha256'] = None
            task['pre_reasons'].append('missing_audio')
        else:
            task['file_sha256'] = sha256(path)
            inputs[str(path)] = task['file_sha256']
            expected = parent['inputs_sha256'].get(str(path))
            if expected and task['file_sha256'] != expected:
                raise ValueError(f'Parent audio changed: {path}')
        if number % 1000 == 0:
            print(f'Frozen audio {number}/{len(ordered)}', flush=True)
    from faster_whisper.utils import get_assets_path
    from faster_whisper.vad import get_speech_timestamps
    model = Path(parent['asr_path'])
    for path in list(model.iterdir()) + list(Path(get_assets_path()).glob('silero_vad*.onnx')):
        if path.is_file():
            inputs[str(path)] = sha256(path)
    inputs[inspect.getfile(get_speech_timestamps)] = sha256(inspect.getfile(get_speech_timestamps))
    for folder in (ROOT / 'src', ROOT / 'scripts'):
        for pattern in ('*.py', '*.sh'):
            inputs.update({str(path): sha256(path) for path in folder.rglob(pattern)})
    if shutil.disk_usage(work.parent).free < 5 * 1024**3:
        raise RuntimeError('Need 5 GiB for audit outputs and the existing 4 GiB reserve')
    work.mkdir(parents=True)
    for key, manifest in manifests.items():
        path = work / 'manifests' / (key + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest['path'], path)
        inputs[str(path)] = sha256(path)
    _write_jsonl(protected, work / 'protected_validation.jsonl')
    _write_jsonl(ordered, work / 'tasks.jsonl')
    save_new(work / 'metadata_issues.json', issues)
    for name in ('protected_validation.jsonl', 'tasks.jsonl', 'metadata_issues.json'):
        inputs[str(work / name)] = sha256(work / name)
    plan = {'inputs_sha256': inputs, 'sources': sources, 'asr_path': str(model), 'policy': POLICY,
            'manifests': {key: {'original': data['path'], 'rows': len(data['rows'])} for key, data in manifests.items()},
            'files': len(ordered), 'audio_hours': sum(float(t['durations'][0]) for t in ordered
                if 'conflicting_durations' not in t['pre_reasons']) / 3600,
            'authorization': 'User: 整体清洗一波微调数据呢; exclude every known/suspected issue',
            'inherited_pcm': inherited_pcm, 'training_started': False, 'production_changed': False,
            'note': 'Existing ten sources plus current targets/refs and protected validation. '
                    'Keep original audio, manifests and frozen evaluation. Exclude suspects conservatively; '
                    'ASR mismatch is not confirmed mispronunciation. No new training or native-quality certification.'}
    save_new(work / 'plan.json', plan)
    check_inputs(plan)
    print(json.dumps({k: plan[k] for k in ('files', 'audio_hours', 'policy')}, ensure_ascii=False), flush=True)


def asr_reasons(result):
    reasons = []
    if not result['hyp'].strip():
        reasons.append('empty_asr')
    if not result['comparisons'] or any(not math.isfinite(row['cer']) or row['cer'] > POLICY['max_cer']
                                         for row in result['comparisons']):
        reasons.append('asr_text_mismatch')
    weighted = [(max(0., segment['end'] - segment['start']), segment) for segment in result['segments']]
    total = sum(duration for duration, _ in weighted)
    if not total:
        reasons.append('no_asr_speech_duration')
    else:
        for field, threshold, below in (('avg_logprob', POLICY['min_avg_logprob'], True),
                                        ('no_speech_prob', POLICY['max_no_speech_prob'], False)):
            value = sum(duration * segment[field] for duration, segment in weighted) / total
            if not math.isfinite(value) or (value < threshold if below else value > threshold):
                reasons.append('asr_' + field)
    return reasons


def acoustic(task, speech_timestamps):
    result = {'id': task['id'], 'audio': task['audio'], 'file_sha256': task['file_sha256'],
              'reasons': list(task['pre_reasons']), 'pcm_sha256': None}
    if task['file_sha256'] is None:
        return result
    if sha256(task['audio']) != task['file_sha256']:
        raise ValueError('Audio changed after freezing: ' + task['audio'])
    try:
        row = {'audio': task['audio'], 'duration': task['durations'][0]}
        result.update(inspect_audio(row))
        measured = measure(task['audio'], float(row['duration']), task['file_sha256'], speech_timestamps)
        result.update({key: measured[key] for key in ('duration', 'rms', 'head_silence_sec', 'tail_silence_sec', 'vad')})
        result['reasons'].extend(measured['review_reasons'])
        if (measured['head_silence_sec'] is not None and measured['head_silence_sec'] > POLICY['max_head_seconds']
                or measured['vad']['head_seconds'] is not None and measured['vad']['head_seconds'] > POLICY['max_head_seconds']):
            result['reasons'].append('head_silence_over_0_5s')
        if result['max_full_scale_run_ms'] >= POLICY['clipping_run_ms']:
            result['reasons'].append('suspected_clipping')
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        result['reasons'].append('audio_inspection:' + str(error))
    if sha256(task['audio']) != task['file_sha256']:
        raise ValueError('Audio changed during inspection: ' + task['audio'])
    result['reasons'] = sorted(set(result['reasons']))
    return result


def filter_rows(rows, row_flags, measurements, blocked_paths, blocked_pcm, protected_pcm, seen):
    kept, removed = [], []
    for position, row in enumerate(rows, 1):
        reasons = list(row_flags.get(position, []))
        for field, origin in (('audio', 'origin_audio'), ('ref_audio', 'ref_origin_audio')):
            if not row.get(field):
                continue
            result = measurements[row[field]]
            if (result['reasons'] or result['pcm_sha256'] in blocked_pcm or any(
                    str(Path(row[key]).resolve()) in blocked_paths for key in (field, origin) if row.get(key))):
                reasons.append('suspect_' + field)
            if result['pcm_sha256'] in protected_pcm:
                reasons.append('validation_pcm_' + field)
        digest = measurements[row['audio']]['pcm_sha256']
        if row.get('ref_audio') and digest == measurements[row['ref_audio']]['pcm_sha256']:
            reasons.append('identical_target_reference_pcm')
        if ('pcm', digest) in seen:
            reasons.append('duplicate_target_pcm')
        origin = ('origin', str(Path(row.get('origin_audio') or row['audio']).resolve()))
        if seen[origin] >= 3:
            reasons.append('origin_exposure_above_3')
        if reasons:
            removed.append({'row': position, 'sample': row, 'reasons': sorted(set(reasons)),
                            'status': 'excluded_suspect_or_isolation_not_confirmed_bad_recording'})
        else:
            seen['pcm', digest] += 1
            seen[origin] += 1
            kept.append(position)
    return kept, removed


def collect(work):
    plan = read_json(work / 'plan.json')
    check_inputs(plan)
    tasks = _read_manifest(work / 'tasks.jsonl')
    measurements = {row['audio']: row for row in _read_manifest(work / 'acoustic.jsonl')}
    if set(measurements) != {task['audio'] for task in tasks}:
        raise ValueError('Incomplete acoustic audit')
    eligible = {path for path, row in measurements.items() if not row['reasons']}
    asr = {row['audio']: row for row in _read_manifest(work / 'asr.jsonl')}
    if set(asr) != eligible:
        raise ValueError('Incomplete ASR audit')
    for path, record in asr.items():
        if record['file_sha256'] != measurements[path]['file_sha256']:
            raise ValueError('ASR input differs: ' + path)
        measurements[path]['reasons'].extend(record['reasons'])
    task_index = {task['audio']: task for task in tasks}
    bad = [path for path, row in measurements.items() if row['reasons']]
    blocked_paths = {str(Path(path).resolve()) for audio in bad
                     for path in [audio, *task_index[audio]['origins']]}
    blocked_pcm = {measurements[path]['pcm_sha256'] for path in bad if measurements[path]['pcm_sha256']}
    blocked_pcm.update(plan.get('inherited_pcm', []))
    protected = _read_manifest(work / 'protected_validation.jsonl')
    protected_pcm = {measurements[row[field]]['pcm_sha256'] for row in protected
                     for field in ('audio', 'ref_audio') if row.get(field)} - {None}
    excluded, _ = load_exclusions(ROOT / 'eval_cases')
    flags = read_json(work / 'metadata_issues.json')
    outputs, pooled, seen_pool = {}, {'train': [], 'val': []}, {'train': Counter(), 'val': Counter()}
    for key in sorted(plan['manifests'], key=lambda key: (key.startswith('current/'), key)):
        source, split = key.split('/')
        path = work / 'manifests' / (key + '.jsonl')
        rows = _read_manifest(path)
        seen = Counter() if source == 'current' else seen_pool[split]
        positions, removed = filter_rows(rows, {r['row']: r['reasons'] for r in flags[key]},
            measurements, blocked_paths, blocked_pcm, protected_pcm if split == 'train' else set(), seen)
        kept = [rows[position - 1] for position in positions]
        if source != 'current':
            pooled[split].extend(kept)
        output = work / 'candidate' / source
        output.mkdir(parents=True, exist_ok=True)
        lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
        with (output / f'{split}.jsonl').open('xb') as stream:
            stream.write(b''.join(lines[position - 1] for position in positions))
        _write_jsonl(removed, output / f'{split}_excluded.jsonl')
        summary, shares = describe_mix(kept)
        outputs[key] = {'before': len(rows), 'excluded': len(removed), 'summary': summary,
                        'language_shares': shares,
                        'reasons': dict(Counter(reason for row in removed for reason in row['reasons']))}
    checks = {}
    for name, rows in (('pool', pooled['train']), ('current', _read_manifest(work / 'candidate/current/train.jsonl'))):
        checks[name] = {'metadata': audit_mix(rows, protected, excluded, max_dur=30),
                        'pcm': pcm_conflicts(rows, protected, measurements)}
        if not rows or checks[name]['metadata']['violations'] or checks[name]['metadata']['origins_above_3x'] or any(checks[name]['pcm'].values()):
            raise ValueError(f'Cleaned candidate failed final isolation: {name}')
    check_inputs(plan, audio=True)
    save_new(work / 'summary.json', {'status': 'complete', 'files': len(tasks), 'asr_files': len(asr),
        'suspect_files': len(bad), 'outputs': outputs, 'checks': checks,
        'files_with_reasons': {path: row['reasons'] for path, row in measurements.items() if row['reasons']},
        'training_started': False, 'production_changed': False, 'source_mutations': False,
        'native_quality': 'unverified', 'training_approved': False,
        'note': 'Candidates only. Preserve source split and retained row bytes. Full old validation remains protected. '
                'No ASR-driven text rewrite, trimming, ref replacement, new sampling or audio deletion. '
                'Near-duplicate, multi-speaker/music and native-language quality are not comprehensively certified.',
        'outputs_sha256': {str(path.relative_to(work)): sha256(path) for path in (work / 'candidate').rglob('*.jsonl')}})


def run(work):
    if work.exists():
        raise FileExistsError('Refusing implicit restart: ' + str(work))
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy; refusing overlapping audit')
    runtime_environment(work.parent / 'runtime')
    prepare(work)
    plan = read_json(work / 'plan.json')
    tasks = _read_manifest(work / 'tasks.jsonl')
    def status(stage, completed=0, total=0):
        data = {'stage': stage, 'completed': completed, 'total': total, 'time': time.time(),
                'training_started': False, 'production_changed': False}
        temporary = work / 'status.tmp'
        temporary.write_text(json.dumps(data), encoding='utf-8')
        temporary.replace(work / 'status.json')
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        options = VadOptions(min_silence_duration_ms=500, speech_pad_ms=0)
        speech = lambda audio, rate: get_speech_timestamps(audio, options, sampling_rate=rate)
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor, (work / 'acoustic.jsonl').open('x', encoding='utf-8') as stream:
            futures = [executor.submit(acoustic, task, speech) for task in tasks]
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    if count % 100 == 0:
                        status('acoustic', count, len(tasks))
                        print(f'Acoustic {count}/{len(tasks)}', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        allowed = {row['audio'] for row in results if not row['reasons']}
        pending = [task for task in tasks if task['audio'] in allowed]
        check_inputs(plan)
        from faster_whisper import WhisperModel
        model = WhisperModel(plan['asr_path'], device='cuda', compute_type='int8_float16',
                             num_workers=4, cpu_threads=4, local_files_only=True)
        if model.model.device != 'cuda':
            raise RuntimeError('CUDA ASR required; no CPU fallback')
        save_new(work / 'asr_runtime.json', {'device': model.model.device,
            'compute_type': model.model.compute_type, 'files': len(pending), 'policy': POLICY})
        def scan(task):
            if sha256(task['audio']) != task['file_sha256']:
                raise ValueError('ASR input changed: ' + task['audio'])
            result = transcribe(model, task['audio'], task['langs'][0], task['texts'])
            if sha256(task['audio']) != task['file_sha256']:
                raise ValueError('Audio changed during ASR: ' + task['audio'])
            return {'id': task['id'], 'audio': task['audio'], 'file_sha256': task['file_sha256'],
                    'result': result, 'reasons': asr_reasons(result)}
        status('asr', 0, len(pending))
        with ThreadPoolExecutor(max_workers=4) as executor, (work / 'asr.jsonl').open('x', encoding='utf-8') as stream:
            futures = [executor.submit(scan, task) for task in pending]
            try:
                for count, future in enumerate(as_completed(futures), 1):
                    if shutil.disk_usage(work).free < 4 * 1024**3:
                        raise RuntimeError('Disk below 4 GiB reserve')
                    result = future.result()
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    status('asr', count, len(pending))
                    if count % 20 == 0:
                        print(f'ASR {count}/{len(pending)}', flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        status('collecting')
        collect(work)
        status('complete', len(tasks), len(tasks))
    except BaseException:
        status('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    run(args.work.resolve())
