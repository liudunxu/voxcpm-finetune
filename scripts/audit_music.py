"""Bounded AudioSet event audit of current targets/refs; scores are review evidence only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import itertools
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from audit_training_quality import CLEAN, read, stats
from fresh_baseline_eval import save_new
from full_corpus_parallel import write_status
from ref_ab import runtime_environment, sha256
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.paths import ROOT

REPO = 'MIT/ast-finetuned-audioset-10-10-0.4593'
REVISION = 'f826b80d28226b62986cc218e5cec390b1096902'
ASSETS = {
    'README.md': (1165, 'fc4ece98b629cab4731db057bebdc4240fda9239'),
    'config.json': (26763, '9b822c630bc1a61312ac702cca4d8fb5a524e729'),
    'preprocessor_config.json': (297, 'd93e9e561c374604096fd5868c97e08130638f72'),
    'model.safetensors': (346404948, 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d'),
}
LABELS = ('Speech', 'Music', 'Singing', 'Song', 'Musical instrument')
EVENTS = LABELS[1:]
NOTE = ('Uncalibrated AudioSet sigmoid scores, not probabilities of training unsuitability. '
        'A score >=0.8 is a review priority, not an automatic exclusion or confirmed music. '
        'Low scores do not certify clean speech, single speaker, identity or native quality. '
        'No training, TTS/ASR, audio edits or changes to the running corpus-cleaning policy.')


def fetch(work):
    from huggingface_hub import hf_hub_download

    record = work.parent / 'model_assets.json'
    if record.exists():
        raise FileExistsError(record)
    if shutil.disk_usage(work.parent).free < 5 * 1024**3:
        raise RuntimeError('Need 5 GiB free before the 346 MB model download')
    paths = {}
    for name, (size, digest) in ASSETS.items():
        path = Path(hf_hub_download(REPO, name, revision=REVISION, token=False,
                    cache_dir='/root/autodl-tmp/hf_home/hub', endpoint='https://hf-mirror.com'))
        if path.stat().st_size != size:
            raise ValueError('Model asset size differs: ' + name)
        if name.endswith('.safetensors'):
            actual = sha256(path)
        else:
            data = path.read_bytes()
            actual = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        if actual != digest:
            raise ValueError('Model asset differs from pinned official revision: ' + name)
        paths[str(path)] = sha256(path)
        print('VERIFIED ' + name, flush=True)
    save_new(record, {'repo': REPO, 'revision': REVISION, 'license': 'BSD-3-Clause',
        'model_path': str(path.parent), 'files_sha256': paths,
        'provenance': f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'})


def windows(samples, window=160000, stride=80000):
    if samples <= 0 or window <= 0 or not 0 < stride <= window:
        raise ValueError('Invalid window parameters')
    last = max(0, samples - window)
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return [(start, min(samples, start + window)) for start in starts]


def select_files(rows, acoustic):
    files, skipped = {}, {}
    for number, row in enumerate(rows, 1):
        for field in ('audio', 'ref_audio'):
            if not row.get(field):
                continue
            path = row[field]
            measured = acoustic[path]
            if measured['reasons']:
                skipped[path] = measured['reasons']
                continue
            if path not in files:
                files[path] = {'audio': path, 'file_sha256': measured['file_sha256'],
                    'pcm_sha256': measured['pcm_sha256'], 'samples': round(measured['duration'] * 16000),
                    'lang': row['lang'], 'source_id': row['source_id'], 'uses': []}
            case = files[path]
            if (case['lang'], case['source_id']) != (row['lang'], row['source_id']):
                raise ValueError('Conflicting audio provenance: ' + path)
            case['uses'].append({'row': number, 'field': field})
    # Interleave sources so the first pilot covers all available sources, ID first.
    groups = defaultdict(list)
    for case in files.values():
        groups[case['source_id']].append(case)
    ordered = [sorted(groups[source], key=lambda r: (r['file_sha256'], r['audio']))
               for source in sorted(groups, key=lambda s: (groups[s][0]['lang'] != 'id', s))]
    return [case for group in itertools.zip_longest(*ordered) for case in group if case], skipped


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    assets_path = work.parent / 'model_assets.json'
    assets = read(assets_path)
    if assets['repo'] != REPO or assets['revision'] != REVISION:
        raise ValueError('Unexpected model revision')
    paths = [CLEAN / 'plan.json', CLEAN / 'acoustic.jsonl', CLEAN / 'manifests/current/train.jsonl', assets_path]
    parent = read(paths[0])
    for path in paths[1:3]:
        if sha256(path) != parent['inputs_sha256'][str(path)]:
            raise ValueError('Frozen parent changed: ' + str(path))
    rows = _read_manifest(paths[2])
    acoustic = {r['audio']: r for r in _read_manifest(paths[1])}
    cases, skipped = select_files(rows, acoustic)
    if not cases:
        raise ValueError('No acoustic-eligible current targets/refs')
    inputs = {str(p): sha256(p) for p in paths}
    inputs.update(assets['files_sha256'])
    for case in cases:
        if parent['inputs_sha256'][case['audio']] != case['file_sha256']:
            raise ValueError('Frozen acoustic file differs')
        inputs[case['audio']] = case['file_sha256']
    for directory in (ROOT / 'scripts', ROOT / 'src'):
        for path in itertools.chain(directory.rglob('*.py'), directory.rglob('*.sh')):
            inputs[str(path)] = sha256(path)
    work.mkdir()
    (work / 'items').mkdir()
    _write_jsonl(cases, work / 'cases.jsonl')
    save_new(work / 'skipped_acoustic.json', skipped)
    for name in ('cases.jsonl', 'skipped_acoustic.json'):
        inputs[str(work / name)] = sha256(work / name)
    save_new(work / 'plan.json', {'inputs_sha256': inputs, 'model_path': assets['model_path'],
        'repo': REPO, 'revision': REVISION, 'files': len(cases), 'current_training_rows': len(rows),
        'skipped_acoustic_files': len(skipped), 'labels': LABELS, 'review_threshold': .8,
        'window_seconds': 10, 'stride_seconds': 5, 'pooling': 'per-class max over all windows',
        'activation': 'sigmoid_multilabel', 'dtype': 'float32', 'batch_size': 1, 'sleep_seconds': .1,
        'by_lang': dict(Counter(r['lang'] for r in cases)), 'asr_snapshot': read(CLEAN / 'status.json'),
        'training_started': False, 'production_changed': False, 'note': NOTE})
    checked(work)
    print(f'PREPARED {len(cases)} audio files, {len(skipped)} already acoustic-flagged files skipped', flush=True)


def checked(work):
    plan = read(work / 'plan.json')
    for path, digest in plan['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen music diagnostic input changed: ' + path)
    return plan


def aggregate(window_scores):
    window_scores = list(window_scores)
    if not window_scores:
        raise ValueError('No event scores')
    for row in window_scores:
        if any(not np.isfinite(row[key]) or not 0 <= row[key] <= 1 for key in LABELS):
            raise ValueError('Invalid event score')
    return {label: max(row[label] for row in window_scores) for label in LABELS}


def summarize(items, threshold):
    by_lang = {}
    for lang in sorted({r['case']['lang'] for r in items}):
        rows = [r for r in items if r['case']['lang'] == lang]
        by_lang[lang] = {'files': len(rows), 'review_priority_files': sum(
            max(r['scores'][label] for label in EVENTS) >= threshold for r in rows),
            'scores': {label: stats(r['scores'][label] for r in rows) for label in LABELS}}
    priority = sorted((r for r in items if max(r['scores'][label] for label in EVENTS) >= threshold),
        key=lambda r: (-max(r['scores'][label] for label in EVENTS), r['case']['audio']))
    return {'files': len(items), 'by_lang': by_lang, 'review_priority_files': len(priority),
        'review_priority': [{'case': r['case'], 'scores': r['scores']} for r in priority],
        'automatic_exclusions': 0, 'training_started': False, 'production_changed': False, 'note': NOTE}


def run(work, limit):
    import torch
    import soundfile as sf
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    if (work / 'summary.json').exists():
        raise FileExistsError('Music audit already complete')
    plan = checked(work)
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())
    if not torch.cuda.is_available() or free < 6 * 1024:
        raise RuntimeError('Need CUDA and 6 GiB free')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.10)
    processor = ASTFeatureExtractor.from_pretrained(plan['model_path'], local_files_only=True)
    model, loading = ASTForAudioClassification.from_pretrained(plan['model_path'], local_files_only=True,
        use_safetensors=True, output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Model did not load exactly: ' + str(loading))
    if processor.sampling_rate != 16000 or processor.max_length != 1024 or model.config.num_labels != 527:
        raise ValueError('Unexpected AudioSet preprocessing or output')
    label_ids = {label: model.config.label2id[label] for label in LABELS}
    model.eval().to(device='cuda', dtype=torch.float32)
    started, before, new = time.time(), read(CLEAN / 'status.json'), 0
    items = []
    for number, case in enumerate(_read_manifest(work / 'cases.jsonl')):
        output = work / 'items' / f'{number:05d}.json'
        if output.exists():
            result = read(output)
            if result['case'] != case or result['scores'] != aggregate(w['scores'] for w in result['windows']):
                raise ValueError('Cached result differs')
            items.append(result)
            continue
        if limit and new >= limit:
            continue
        if shutil.disk_usage(work).free < 4 * 1024**3:
            raise RuntimeError('Disk below 4 GiB reserve')
        audio, rate = sf.read(case['audio'], dtype='float32', always_2d=True)
        if (rate != 16000 or audio.shape != (case['samples'], 1) or not np.isfinite(audio).all()
                or not 3 <= len(audio)/rate <= 30 or sha256(case['audio']) != case['file_sha256']):
            raise ValueError('Source audio changed or invalid')
        measurements = []
        for start, end in windows(len(audio)):
            inputs = processor(audio[start:end, 0], sampling_rate=rate, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                probabilities = model(**inputs).logits[0].sigmoid().cpu().numpy()
            if probabilities.shape != (527,) or not np.isfinite(probabilities).all():
                raise ValueError('Invalid AudioSet output')
            measurements.append({'start_sample': start, 'end_sample': end,
                'scores': {label: float(probabilities[index]) for label, index in label_ids.items()},
                'top5': [{'label': model.config.id2label[int(i)], 'score': float(probabilities[i])}
                         for i in np.argsort(probabilities)[-5:][::-1]]})
            time.sleep(plan['sleep_seconds'])
        if sha256(case['audio']) != case['file_sha256']:
            raise ValueError('Audio changed during classification')
        result = {'case': case, 'scores': aggregate([w['scores'] for w in measurements]), 'windows': measurements}
        save_new(output, result)
        items.append(result)
        new += 1
        if new % 25 == 0:
            write_status(work / 'status.json', {'stage': 'music_audit', 'completed': len(items),
                'total': plan['files'], 'time': time.time(), 'training_started': False, 'production_changed': False})
            print(f'MUSIC {len(items)}/{plan["files"]}', flush=True)
    checked(work)
    complete = len(items) == plan['files']
    save_new(work / f'runtime_{len(items)}.json', {'new_files': new, 'elapsed_seconds': time.time()-started,
        'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
        'device': str(next(model.parameters()).device), 'dtype': str(next(model.parameters()).dtype),
        'asr_before': before, 'asr_after': read(CLEAN / 'status.json'),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'torchaudio', 'transformers')}})
    save_new(work / ('summary.json' if complete else f'pilot_{len(items)}.json'),
             summarize(items, plan['review_threshold']))
    write_status(work / 'status.json', {'stage': 'complete' if complete else 'pilot_complete',
        'completed': len(items), 'total': plan['files'], 'time': time.time(),
        'training_started': False, 'production_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch', 'prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    work = args.work.resolve()
    runtime_environment(work.parent / 'runtime')
    if args.action == 'run':
        run(work, args.limit)
    elif args.action == 'fetch':
        fetch(work)
    else:
        prepare(work)
