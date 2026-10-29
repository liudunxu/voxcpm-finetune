"""Frozen 96-recording VAE roundtrip: native 40ms hop versus 160ms training patch."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from audit_training_quality import CLEAN, read, stats
from fresh_baseline_eval import BASE, WEIGHTS, save_new
from full_corpus_clean import asr_reasons
from full_corpus_parallel import complete_records, write_status
from ref_ab import runtime_environment, sha256
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.paths import ROOT, VOXCPM_REPO
from voxft.qc.audio import edge_silence, speech_ratio


def pad_samples(samples, multiple):
    if samples <= 0 or multiple <= 0:
        raise ValueError('Invalid sample count or alignment')
    return -samples % multiple


def select_cases(rows, acoustic, asr, hop, patch):
    groups = defaultdict(list)
    for position, row in enumerate(rows, 1):
        if row['lang'] not in ('id', 'th', 'tl'):
            continue
        measured, result = acoustic[row['audio']], asr.get(row['audio'])
        if measured['reasons'] or result is None or result['reasons'] or not 3 <= measured['duration'] <= 15:
            continue
        if result['file_sha256'] != measured['file_sha256'] or result['reasons'] != asr_reasons(result['result']):
            raise ValueError('ASR evidence differs from frozen acoustic input')
        samples = round(measured['duration'] * 16000)
        extra_hops = (pad_samples(samples, patch) - pad_samples(samples, hop)) // hop
        item = {'row': position, 'audio': row['audio'], 'source_id': row['source_id'],
                'lang': row['lang'], 'samples': samples, 'file_sha256': measured['file_sha256'],
                'extra_hops': extra_hops}
        groups[row['lang'], extra_hops].append(item)
    for lang in ('id', 'th', 'tl'):
        for bucket in range(4):
            pool = groups[lang, bucket]
            pool.sort(key=lambda r: (r['file_sha256'], r['audio']))
            if len(pool) < 8:
                raise ValueError(f'Need eight ASR-passed recordings in {lang}/padding bucket{bucket}')
    cases = [groups[lang, bucket][i] for i in range(8)
             for lang in ('id', 'th', 'tl') for bucket in range(4)]
    if len({r['audio'] for r in cases}) != 96:
        raise ValueError('Repeated source audio in codec probe')
    return cases


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    config = read(BASE / 'config.json')
    hop = math.prod(config['audio_vae_config']['encoder_rates'])
    patch = config['patch_size'] * hop
    if (hop, patch, config['audio_vae_config']['sample_rate'], config['audio_vae_config']['out_sample_rate']) != (640, 2560, 16000, 48000):
        raise ValueError('Unexpected codec time base')
    paths = [CLEAN / 'plan.json', CLEAN / 'acoustic.jsonl', CLEAN / 'manifests/current/train.jsonl',
             BASE / 'config.json', BASE / 'audiovae.pth']
    if sha256(paths[-1]) != WEIGHTS[paths[-1]]:
        raise ValueError('Base VAE weight changed')
    parent = read(CLEAN / 'plan.json')
    for path in paths:
        expected = parent['inputs_sha256'].get(str(path))
        if expected and sha256(path) != expected:
            raise ValueError('Frozen input changed: ' + str(path))
    acoustic = {r['audio']: r for r in _read_manifest(CLEAN / 'acoustic.jsonl')}
    asr = {}
    for path in [CLEAN / 'retained_asr.jsonl', *sorted(CLEAN.glob('workers/*/items.jsonl'))]:
        records, _ = complete_records(path)
        for row in records:
            if row['audio'] in asr:
                raise ValueError('Duplicate completed ASR record')
            asr[row['audio']] = row
    cases = select_cases(_read_manifest(paths[2]), acoustic, asr, hop, patch)
    if shutil.disk_usage(work.parent).free < 5 * 1024**3:
        raise RuntimeError('Need 5 GiB free, keeping 4 GiB reserve')
    work.mkdir()
    (work / 'items').mkdir()
    (work / 'wav').mkdir()
    _write_jsonl(cases, work / 'cases.jsonl')
    _write_jsonl([asr[r['audio']] for r in cases], work / 'selection_asr.jsonl')
    paths.extend([work / 'cases.jsonl', work / 'selection_asr.jsonl'])
    paths.extend(Path(r['audio']) for r in cases)
    for directory in (ROOT / 'scripts', ROOT / 'src', VOXCPM_REPO / 'src'):
        paths.extend(directory.rglob('*.py'))
        paths.extend(directory.rglob('*.sh'))
    save_new(work / 'plan.json', {'inputs_sha256': {str(p): sha256(p) for p in paths},
        'cases': 96, 'hop_samples': hop, 'patch_samples': patch, 'sample_rate': 16000, 'output_sample_rate': 48000,
        'selection': '32/lang, eight per incremental patch-padding bucket (0/40/80/120ms), '
                     '3-15s current training targets with completed acoustic+ASR pass at preparation.',
        'asr_snapshot': read(CLEAN / 'status.json'), 'dtype': 'float32', 'seed': 42,
        'training_started': False, 'production_changed': False,
        'authorization': 'Continue bounded parallel fine-tuning quality diagnostics.',
        'note': 'Codec reconstruction only, not TTS or a new model evaluation. '
                'Stratified mechanics probe, not population-estimated tail distribution or proof of LoRA causation. '
                'Original audio/labels, frozen scores and training source remain untouched.'})


def checked(work):
    plan = read(work / 'plan.json')
    for path, digest in plan['inputs_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen codec diagnostic input changed: ' + path)
    return plan


def metrics(audio, rate):
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid codec waveform')
    return {'duration': len(audio) / rate, 'sample_rate': rate, 'samples': len(audio),
            **edge_silence(audio, rate), 'speech_ratio': speech_ratio(audio, rate),
            'peak': float(np.abs(audio).max()), 'rms': float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))}


def summarize(items):
    groups = {}
    for lang in ('id', 'th', 'tl'):
        rows = [r for r in items if r['case']['lang'] == lang]
        groups[lang] = {'count': len(rows), 'tail_seconds': {
            name: stats(r['measurements'][name]['tail_silence_sec'] for r in rows)
            for name in ('original', 'hop_input', 'patch_input', 'hop_codec', 'patch_codec')}}
        groups[lang]['paired_tail_delta'] = {}
        for a, b in (('original', 'hop_input'), ('original', 'patch_input'),
                     ('hop_input', 'hop_codec'), ('patch_input', 'patch_codec'), ('hop_codec', 'patch_codec')):
            values = [r['measurements'][b]['tail_silence_sec'] - r['measurements'][a]['tail_silence_sec']
                      for r in rows if all(r['measurements'][k]['tail_silence_sec'] is not None for k in (a, b))]
            groups[lang]['paired_tail_delta'][b + '_minus_' + a] = stats(values)
    return groups


def run(work, limit):
    import torch
    import soundfile as sf
    from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2

    if (work / 'summary.json').exists():
        raise FileExistsError('Codec diagnostic already complete')
    plan = checked(work)
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())
    if not torch.cuda.is_available() or free < 8 * 1024:
        raise RuntimeError('Need CUDA and 8 GiB free for bounded codec diagnostic')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.manual_seed(plan['seed'])
    config = AudioVAEConfigV2(**read(BASE / 'config.json')['audio_vae_config'])
    vae = AudioVAEV2(config)
    checkpoint = torch.load(BASE / 'audiovae.pth', map_location='cpu', weights_only=True)
    vae.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    vae.eval().to(device='cuda', dtype=torch.float32)
    started, asr_before = time.time(), read(CLEAN / 'status.json')
    items, new = [], 0
    for number, case in enumerate(_read_manifest(work / 'cases.jsonl')):
        output = work / 'items' / f'{number:03d}.json'
        if output.exists():
            result = read(output)
            if result['case'] != case or any(sha256(p) != digest for p, digest in result['wav_sha256'].items()):
                raise ValueError('Cached codec output changed')
            items.append(result)
            continue
        if limit and new >= limit:
            continue
        if shutil.disk_usage(work).free < 4 * 1024**3:
            raise RuntimeError('Disk below 4 GiB reserve')
        audio, rate = sf.read(case['audio'], dtype='float32', always_2d=True)
        if rate != 16000 or audio.shape != (case['samples'], 1) or sha256(case['audio']) != case['file_sha256']:
            raise ValueError('Source recording changed')
        audio = audio[:, 0]
        measurements, digests = {'original': metrics(audio, rate)}, {}
        for arm, multiple in (('hop', plan['hop_samples']), ('patch', plan['patch_samples'])):
            padded = np.pad(audio, (0, pad_samples(len(audio), multiple)))
            measurements[arm + '_input'] = metrics(padded, rate)
            with torch.inference_mode():
                latent = vae.encode(torch.from_numpy(padded.copy()).reshape(1, 1, -1).to('cuda'), rate)
                decoded = vae.decode(latent).flatten().cpu().numpy()
            expected = len(padded) // plan['hop_samples'] * vae.decode_chunk_size
            if len(decoded) != expected:
                raise ValueError('Unexpected VAE reconstruction length')
            measurements[arm + '_codec'] = metrics(decoded, 48000)
            wav = work / 'wav' / f'{number:03d}_{arm}.wav'
            if wav.exists():
                raise FileExistsError('Orphan output; refuse silent overwrite: ' + str(wav))
            sf.write(wav, decoded, 48000, subtype='FLOAT')
            digests[str(wav)] = sha256(wav)
        if sha256(case['audio']) != case['file_sha256']:
            raise ValueError('Source changed during codec diagnostic')
        result = {'case': case, 'measurements': measurements, 'wav_sha256': digests}
        save_new(output, result)
        items.append(result)
        new += 1
        write_status(work / 'status.json', {'stage': 'roundtrip', 'completed': len(items), 'total': plan['cases'],
                                          'time': time.time(), 'training_started': False})
        print(f'CODEC {len(items)}/{plan["cases"]}', flush=True)
        time.sleep(.10)
    checked(work)
    runtime = {'device': str(next(vae.parameters()).device), 'dtype': str(next(vae.parameters()).dtype),
        'new_cases': new, 'elapsed_seconds': time.time() - started,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
        'asr_before': asr_before, 'asr_after': read(CLEAN / 'status.json')}
    save_new(work / f'runtime_{len(items)}.json', runtime)
    complete = len(items) == plan['cases']
    if complete:
        save_new(work / 'summary.json', {'by_lang': summarize(items), 'cases': len(items),
            'reconstruction_wavs': 2 * len(items), 'inputs_verified': True, 'training_started': False,
            'production_changed': False, 'native_quality': 'unverified', 'note': plan['note']})
    write_status(work / 'status.json', {'stage': 'complete' if complete else 'pilot_complete',
        'completed': len(items), 'total': plan['cases'], 'time': time.time(),
        'training_started': False, 'production_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    work = args.work.resolve()
    runtime_environment(work.parent / 'runtime')
    if args.action == 'prepare':
        prepare(work)
    else:
        run(work, args.limit)
