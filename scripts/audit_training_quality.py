"""Bounded reference-identity diagnostics and CPU padding audit beside the frozen ASR run."""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from fresh_baseline_eval import BASE, SPEAKER, save_new
from full_corpus_parallel import write_status
from ref_ab import checked_pairs, runtime_environment, sha256
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.paths import ROOT, VOXCPM_REPO

CLEAN = Path('/root/autodl-tmp/full_corpus_clean_20261029/parallel8/run')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'count': 0}
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite diagnostic value')
    return {'count': len(values), 'mean': float(np.mean(values)),
            **{f'p{q}': float(np.percentile(values, q)) for q in (0, 1, 10, 50, 90, 99, 100)}}


def checked(work):
    plan = read(work / 'plan.json')
    for path, digest in plan['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Diagnostic input changed: ' + path)
    return plan


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent = read(CLEAN / 'plan.json')
    paths = [CLEAN / 'plan.json', CLEAN / 'acoustic.jsonl',
             CLEAN / 'manifests/current/train.jsonl', BASE / 'config.json']
    rows = _read_manifest(paths[2])
    measurements = {r['audio']: r for r in _read_manifest(paths[1])}
    available = {}
    for source in ('cv22_id', 'cv22_th'):
        path = CLEAN / f'manifests/{source}/train.jsonl'
        paths.append(path)
        available.update({r['audio']: r for r in checked_pairs(_read_manifest(path))})
    pairs, skipped, files = [], [], {}
    for position, row in enumerate(rows, 1):
        if not row.get('ref_audio'):
            continue
        original = available.get(row['audio'])
        keys = ('ref_audio', 'speaker', 'ref_speaker', 'origin_audio', 'ref_origin_audio', 'source_id')
        if original is None or any(row[k] != original[k] for k in keys) or row['speaker_verified'] is not True:
            raise ValueError('Reference provenance differs: ' + row['audio'])
        if any(measurements[row[k]]['reasons'] for k in ('audio', 'ref_audio')):
            skipped.append({'row': position, 'audio': row['audio'], 'ref_audio': row['ref_audio'],
                'reasons': {k: measurements[row[k]]['reasons'] for k in ('audio', 'ref_audio')}})
            continue
        pairs.append({'row': position, **{k: row[k] for k in ('audio', *keys)}, 'lang': row['lang']})
        for field, prefix in (('audio', ''), ('ref_audio', 'ref_')):
            path = row[field]
            data = {'audio': path, 'file_sha256': measurements[path]['file_sha256'],
                    'pcm_sha256': measurements[path]['pcm_sha256'], 'source_id': row['source_id'],
                    'speaker': row['speaker'], 'origin_audio': row[prefix + 'origin_audio']}
            if path in files and files[path] != data:
                raise ValueError('Conflicting reference metadata: ' + path)
            files[path] = data
    if not pairs:
        raise ValueError('No eligible reference pairs')
    for path in paths:
        expected = parent['inputs_sha256'].get(str(path))
        if expected and sha256(path) != expected:
            raise ValueError('Frozen parent changed: ' + str(path))
    paths.extend(p for p in SPEAKER.iterdir() if p.is_file())
    paths.extend([VOXCPM_REPO / 'src/voxcpm/training/packers.py',
                  VOXCPM_REPO / 'src/voxcpm/model/voxcpm2.py'])
    paths.extend(ROOT.joinpath('scripts').glob('*.py'))
    paths.extend(ROOT.joinpath('src').rglob('*.py'))
    if shutil.disk_usage(work.parent).free < 5 * 1024**3:
        raise RuntimeError('Need 5 GiB free, retaining existing 4 GiB reserve')
    work.mkdir()
    (work / 'embeddings').mkdir()
    _write_jsonl(pairs, work / 'pairs.jsonl')
    _write_jsonl(skipped, work / 'skipped_acoustic.jsonl')
    paths.extend([work / 'pairs.jsonl', work / 'skipped_acoustic.jsonl'])
    save_new(work / 'plan.json', {'inputs_sha256': {str(p): sha256(p) for p in paths},
        'files': files, 'pairs': len(pairs), 'skipped_pairs': len(skipped), 'speaker_model': str(SPEAKER),
        'training_started': False, 'production_changed': False, 'source_mutations': False,
        'authorization': 'User requested parallel work to improve fine-tuning quality; spare GPU memory.',
        'policy': 'WavLM ranking only, not a calibrated identity verdict or automatic deletion rule. '
                  'Same-source different declared IDs are nominal controls, not verified negatives. '
                  'No reference replacement, new identity labels, TTS/ASR, training or source edits.'})


def padding_seconds(samples, patch_samples, sample_rate):
    if samples <= 0 or patch_samples <= 0 or sample_rate <= 0:
        raise ValueError('Invalid sample counts/rate')
    return (-samples % patch_samples) / sample_rate


def packer_check(config):
    import torch

    path = VOXCPM_REPO / 'src/voxcpm/training/packers.py'
    spec = importlib.util.spec_from_file_location('audited_packer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class ShapeOnlyVAE:
        hop_length = math.prod(config['audio_vae_config']['encoder_rates'])
        sample_rate = config['audio_vae_config']['sample_rate']
        def encode(self, wav, rate):
            assert rate == self.sample_rate and wav.shape[-1] % self.hop_length == 0
            return torch.zeros(1, config['feat_dim'], wav.shape[-1] // self.hop_length)
    packer = module.AudioFeatureProcessingPacker(1, 8192, config['patch_size'],
                                               config['feat_dim'], ShapeOnlyVAE())
    text, target, ref = torch.tensor([11, 12, 13]), torch.zeros(51361), torch.zeros(64001)
    results = {}
    for name, output in [('plain', packer.process_tts_data(target, text)),
                         ('reference', packer.process_tts_data_with_ref(ref, target, text))]:
        tokens, _, _, _, mask, labels, _, _ = output
        last = int(torch.nonzero(mask).flatten()[-1])
        if not (labels.sum() == 1 and labels[last] == 1 and last == len(tokens) - 2 and mask[-1] == 0):
            raise ValueError('Unexpected final-patch stop/loss mask')
        results[name] = {'stop_on_last_target_patch': True, 'end_token_has_loss': False}
    return results


def cpu(work):
    checked(work)
    if (work / 'padding.jsonl').exists() or (work / 'padding_summary.json').exists():
        raise FileExistsError('Padding diagnostics already exist')
    rows = _read_manifest(CLEAN / 'manifests/current/train.jsonl')
    measured = {r['audio']: r for r in _read_manifest(CLEAN / 'acoustic.jsonl')}
    expected = read(CLEAN / 'plan.json')['inputs_sha256']
    config = read(BASE / 'config.json')
    rate = config['audio_vae_config']['sample_rate']
    patch = math.prod(config['audio_vae_config']['encoder_rates']) * config['patch_size']
    grouped, details = defaultdict(list), []
    for row in rows:
        record = measured[row['audio']]
        if record['file_sha256'] != expected[row['audio']]:
            raise ValueError('Acoustic input hash differs')
        seconds = padding_seconds(round(record['duration'] * rate), patch, rate)
        tail = record['tail_silence_sec']
        item = {'audio': row['audio'], 'lang': row['lang'], 'tail_seconds': tail,
                'added_padding_seconds': seconds,
                'tail_plus_padding_seconds': tail + seconds if tail is not None else None}
        grouped[row['lang']].append(item)
        details.append(item)
    _write_jsonl(details, work / 'padding.jsonl')
    save_new(work / 'padding_summary.json', {'rows': len(rows), 'patch_samples': patch,
        'sample_rate': rate, 'patch_seconds': patch / rate, 'packer_masks': packer_check(config),
        'by_lang': {lang: {key: stats(r[key] for r in group) for key in
            ('tail_seconds', 'added_padding_seconds', 'tail_plus_padding_seconds')}
            for lang, group in grouped.items()},
        'note': 'Arithmetic tail+padding is a diagnostic estimate, not a rescore of padded WAVs or '
                'proof of generated-tail causation. Shape-only VAE checks masks, not codec acoustics. '
                'Original PCM and official source untouched. Frozen evaluation gates unchanged.'})
    checked(work)


def summarize_pairs(pairs, files, vectors):
    rng = np.random.default_rng(42)
    scores, controls = [], defaultdict(list)
    pools = {source: [r for r in files.values() if r['source_id'] == source]
             for source in {r['source_id'] for r in pairs}}
    for row in pairs:
        target, ref = files[row['audio']], files[row['ref_audio']]
        score = float(vectors[row['audio']] @ vectors[row['ref_audio']])
        negatives, identities = [], set()
        pool = pools[row['source_id']]
        for index in rng.permutation(len(pool)):
            candidate = pool[index]
            if (candidate['speaker'] == row['speaker'] or candidate['speaker'] in identities
                    or candidate['origin_audio'] in (target['origin_audio'], ref['origin_audio'])
                    or candidate['pcm_sha256'] in (target['pcm_sha256'], ref['pcm_sha256'])):
                continue
            negatives.append(float(vectors[row['audio']] @ vectors[candidate['audio']]))
            identities.add(candidate['speaker'])
            if len(negatives) == 3:
                break
        controls[row['source_id']].extend(negatives)
        scores.append({**row, 'cosine': score, 'nominal_negative_cosines': negatives,
                       'identity_verdict': 'unverified_score_only'})
    return sorted(scores, key=lambda r: (r['cosine'], r['audio'])), {
        source: {'declared_same_speaker': stats(r['cosine'] for r in scores if r['source_id'] == source),
                 'nominal_different_speaker': stats(values)} for source, values in controls.items()}


def embed(work, limit):
    import torch
    import soundfile as sf
    from transformers import AutoFeatureExtractor, WavLMForXVector

    plan = checked(work)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this bounded parallel diagnostic')
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())
    if free < 6 * 1024:
        raise RuntimeError('Need 6 GiB free before starting the diagnostic')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.10)
    processor = AutoFeatureExtractor.from_pretrained(plan['speaker_model'], local_files_only=True)
    model, loading = WavLMForXVector.from_pretrained(plan['speaker_model'], local_files_only=True,
                                                    output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Speaker model did not load exactly: ' + str(loading))
    model.eval().to('cuda')
    started = time.time()
    before = read(CLEAN / 'status.json')
    new, vectors = 0, {}
    for path, record in sorted(plan['files'].items()):
        output = work / 'embeddings' / (record['file_sha256'] + '.json')
        if sha256(path) != record['file_sha256']:
            raise ValueError('Reference audio changed: ' + path)
        if output.exists():
            stored = read(output)
            if stored['file_sha256'] != record['file_sha256']:
                raise ValueError('Cached embedding input differs')
            vector = np.asarray(stored['embedding'], dtype=np.float64)
        else:
            if limit and new >= limit:
                continue
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError('Disk below 4 GiB reserve')
            audio, rate = sf.read(path, dtype='float32', always_2d=True)
            if rate != 16000 or audio.shape[1] != 1 or not 3 <= len(audio) / rate <= 30 or not np.isfinite(audio).all():
                raise ValueError('Invalid reference audio: ' + path)
            inputs = processor(audio[:, 0], sampling_rate=rate, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                vector = model(**inputs).embeddings[0].cpu().numpy().astype(np.float64)
            norm = np.linalg.norm(vector)
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError('Invalid speaker embedding')
            vector /= norm
            if sha256(path) != record['file_sha256']:
                raise ValueError('Audio changed during embedding')
            save_new(output, {'file_sha256': record['file_sha256'], 'embedding': vector.tolist()})
            new += 1
            time.sleep(.05)  # Keep this supplementary job from monopolizing ASR's device.
        if vector.shape != (model.config.xvector_output_dim,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1):
            raise ValueError('Invalid cached speaker embedding')
        vectors[path] = vector
        if len(vectors) % 50 == 0:
            write_status(work / 'status.json', {'stage': 'embedding', 'completed': len(vectors),
                'total': len(plan['files']), 'new_embeddings': new, 'time': time.time()})
            print(f'EMBED {len(vectors)}/{len(plan["files"])}', flush=True)
    after = read(CLEAN / 'status.json')
    save_new(work / f'gpu_{len(vectors)}.json', {'device': str(next(model.parameters()).device),
        'started': started, 'elapsed_seconds': time.time() - started, 'new_embeddings': new,
        'completed_audio': len(vectors), 'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'asr_before': before, 'asr_after': after})
    complete = len(vectors) == len(plan['files'])
    if complete:
        scores, groups = summarize_pairs(_read_manifest(work / 'pairs.jsonl'), plan['files'], vectors)
        _write_jsonl(scores, work / 'pair_scores.jsonl')
        save_new(work / 'reference_summary.json', {'pairs': len(scores), 'files': len(vectors),
            'by_source': groups, 'lowest_20_per_source': {source: [r for r in scores if r['source_id'] == source][:20]
                for source in groups}, 'automatic_exclusions': 0, 'identity_verified': False,
            'note': plan['policy'] + ' Lowest-ranked rows are review priorities, not established mismatches.'})
    checked(work)
    write_status(work / 'status.json', {'stage': 'complete' if complete else 'pilot_complete',
        'completed': len(vectors), 'total': len(plan['files']), 'time': time.time(),
        'training_started': False, 'production_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'cpu', 'embed'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    work = args.work.resolve()
    runtime_environment(work.parent / 'runtime')
    if args.action == 'prepare':
        prepare(work)
    elif args.action == 'cpu':
        cpu(work)
    else:
        embed(work, args.limit)
