"""LM/DiT ablation with per-item rejection; no training or historical resampling."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch
import r31_failure_probe as prior

frozen, fast, review = prior.frozen, prior.fast, prior.review
read, save = prior.read, prior.save
ARMS = {'lm_only': {'lm': 1., 'dit': 0.}, 'dit_only': {'lm': 0., 'dit': 1.}}


@contextmanager
def component_profile(model, arm):
    expected = {id(m) for m in model.tts_model._iter_lora_modules()}
    modules = []
    for name, module in model.tts_model.named_modules():
        if id(module) not in expected:
            continue
        name = name.replace('_orig_mod.', '')
        group = ('lm' if name.startswith(('base_lm.', 'residual_lm.')) else
                 'dit' if name.startswith('feat_decoder.estimator.') else None)
        if group is None:
            raise ValueError(f'Unknown LoRA group: {name}')
        modules.append((name, module, group, module.scaling.detach().clone()))
    counts = Counter(group for _, _, group, _ in modules)
    if set(counts) != {'lm', 'dit'} or len(modules) != len(expected):
        raise ValueError('Incomplete LM/DiT module coverage')
    try:
        with torch.no_grad():
            for _, module, group, _ in modules:
                module.scaling.fill_(module._base_scaling * ARMS[arm][group])
        actual = {name: float(module.scaling.item()) for name, module, _, _ in modules}
        if any(actual[name] != module._base_scaling * ARMS[arm][group]
               for name, module, group, _ in modules):
            raise ValueError('Effective component scaling differs')
        yield {'arm': arm, 'group_counts': dict(counts), 'relative_scales': ARMS[arm],
               'effective_scaling': actual}
    finally:
        with torch.no_grad():
            for _, module, _, original in modules:
                module.scaling.copy_(original)
        if any(not torch.equal(module.scaling, original) for _, module, _, original in modules):
            raise ValueError('Component scales not restored')


def reject_reason(rows, baseline_rows, *, stage_end=False):
    """Conservative development rejection, not certification of actual language errors."""
    if len({prior.key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate probe result')
    latest = rows[-1]
    old = baseline_rows[prior.key(latest)]
    if latest['rms_dbfs'] < -60 and (latest.get('speech_ratio') or 0) < .05:
        return 'near_silence'
    if not frozen.evaluation._norm(latest['hyp']) and frozen.evaluation._norm(old['hyp']):
        return 'new_empty_asr'
    if latest['over_read'] and not old['over_read']:
        return 'new_suspected_overread'
    large = [r for r in rows if r['cer'] > .20 and r['cer'] - baseline_rows[prior.key(r)]['cer'] > .10]
    if len(large) >= 2:
        return 'two_large_paired_cer_regressions'
    if stage_end:
        for lang in sorted({r['lang'] for r in rows}):
            changes = [r['cer'] - baseline_rows[prior.key(r)]['cer'] for r in rows if r['lang'] == lang]
            if statistics.mean(changes) > .01:
                return 'stage_mean_cer_regression_' + lang
    return None


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent = read(prior.SOURCE / 'plan.json')
    review.checked(prior.SOURCE / 'plan.json', prior.SOURCE_PLAN_SHA)
    review.checked(prior.SOURCE / 'cases.jsonl', parent['inputs_sha256'][str(prior.SOURCE / 'cases.jsonl')])
    all_cases = frozen._read_manifest(prior.SOURCE / 'cases.jsonl')
    case_index = {c['case_id']: c for c in all_cases}
    baseline_path = prior.SOURCE / 'base_report.json'
    review.checked(baseline_path, parent['inputs_sha256'][str(baseline_path)])
    base = read(baseline_path)
    base_index = {prior.key(r): r for r in base['items']}
    risk_ids = ('fresh_th_1983_control', 'fresh_th_1994_control')
    first, second = [45, 43, 44, 42, 46], [43, 45, 44, 42, 46]
    jobs = [{'case': case_index[cid], 'seed': seed, 'stage': 'risk'}
            for pair in zip(first, second) for cid, seed in zip(risk_ids, pair)]
    guards = []
    for lang in ('id', 'th', 'tl'):
        pool = [c for c in all_cases if c['lang'] == lang and c['condition'] == 'plain'
                and c['sentence_id'] not in prior.SENTENCES
                and not any(marker in c['text'] for marker in ('(', ')', 'Ã', 'Â', 'Å', 'Ä', '\ufffd'))]
        selected = min(pool, key=lambda c: hashlib.sha256(c['sentence_id'].encode()).hexdigest())
        guards.extend(c for c in all_cases if c['sentence_id'] == selected['sentence_id'])
    jobs.extend({'case': c, 'seed': seed, 'stage': 'guard'} for c in guards for seed in frozen.SEEDS)
    if len(jobs) != 40 or len({(j['case']['case_id'], j['seed']) for j in jobs}) != 40:
        raise ValueError('Unexpected pilot budget')
    inputs = {p: h for p, h in parent['inputs_sha256'].items() if '/models--' in p}
    inputs.update(read(prior.SOURCE / 'trained.json')['inputs_sha256'])
    inputs[str(baseline_path)] = frozen.sha256(baseline_path)
    matched = []
    for job in jobs:
        case, seed = job['case'], job['seed']
        row = base_index[case['case_id'], seed]
        if any(row.get(k) != v for k, v in case.items()):
            raise ValueError('Baseline input differs')
        inputs[row['wav']] = base['wav_sha256'][row['wav']]
        inputs[case['ref_audio']] = parent['inputs_sha256'][case['ref_audio']]
        matched.append(row)
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src'):
        for path in (frozen.ROOT / folder).rglob('*.py'):
            inputs[str(path)] = frozen.sha256(path)
    for path in (Path(__file__).with_suffix('.sh'), prior.SOURCE / 'plan.json', prior.SOURCE / 'flagged.json',
                 prior.SOURCE / 'trained.json', prior.SOURCE / 'cases.jsonl'):
        inputs[str(path)] = frozen.sha256(path)
    for path, value in inputs.items():
        review.checked(path, value)
    if shutil.disk_usage(work.parent).free < 4.3 * 1024**3:
        raise RuntimeError('Need 4.3 GiB before pilot')
    work.mkdir()
    for name, obj in (('jobs.json', jobs), ('base_items.json', matched)):
        save(work / name, obj)
        inputs[str(work / name)] = frozen.sha256(work / name)
    save(work / 'plan.json', {**{k: parent[k] for k in ('base_path', 'asr_path', 'speaker_model_path')},
        'targets': {'base': 'base', 'r31': parent['targets']['r31']}, 'arms': ARMS,
        'inputs_sha256': inputs, 'max_outputs': 80, 'risk_per_arm': 10, 'guard_per_arm': 30,
        'training_steps': 0, 'production_changed': False, 'guard_sentences': sorted({c['sentence_id'] for c in guards}),
        'early_rejection': ['near-silence RMS<-60dBFS and speech_ratio<.05', 'new empty ASR',
                            'new suspected overread', 'two CER>.20 and paired delta>.10',
                            'stage-end any language mean CER delta>.01'],
        'note': 'Early rejection requested by user; preserve every failed WAV. Outcome-selected risk '
                'and metadata-selected guards, five seeds; no native certification or automatic promotion. '
                'Ablating a jointly trained adapter diagnoses sensitivity, not proof of a new training recipe.'})
    print('PREPARED two arms, at most 80 new outputs with per-item rejection', flush=True)


def worker(work, arm):
    plan = prior.strength.check_inputs(work)
    directory = work / arm
    frozen.runtime_environment(directory / 'runtime')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VOXFT_WHISPER_MODEL_LARGE=plan['asr_path'], VOXFT_SPK_EMB_MODEL=plan['speaker_model_path'],
        TORCHINDUCTOR_CACHE_DIR=str(directory / 'runtime/inductor'))
    frozen.evaluation.EVAL_DIR = directory / 'raw_reports'
    frozen.evaluation.infer.CHECKPOINT_DIR = directory
    jobs = read(work / 'jobs.json')
    base = {prior.key(r): r for r in read(work / 'base_items.json')}
    rows, reason = [], None
    with fast.cached_services([j['case'] for j in jobs]):
        save(directory / 'runtime.json', prior.baseline.gpu_services(plan, [j['case'] for j in jobs], 'r31'))
        model = frozen.evaluation.infer.get_model(plan['base_path'], plan['targets']['r31'])
        save(directory / 'full_scaling_check.json', prior.strength.check_scaling(model))
        with component_profile(model, arm) as effective:
            save(directory / 'effective_profile.json', effective)
            for position, job in enumerate(jobs):
                if shutil.disk_usage(work).free < 4 * 1024**3:
                    raise RuntimeError('Less than 4 GiB free')
                if not fast.claim(work / 'claims', f'{arm}_{position:02d}'):
                    raise FileExistsError('No retry of claimed output')
                case = {**job['case'], 'lora_component_profile': arm, 'component_scales': ARMS[arm]}
                report = frozen.evaluation.evaluate(plan['targets']['r31'], 'th', [case],
                    base=plan['base_path'], seeds=[job['seed']], cfg_value=1.8, inference_timesteps=20)
                row, = report['items']
                report['component_profile'] = effective
                report['wav_sha256'] = {row['wav']: frozen.sha256(row['wav'])}
                prior.verify(report, plan, case, 'r31', 1., job['seed'])
                levels = review.level_metrics(row['wav'])
                save(work / 'items' / f'{arm}_{position:02d}.json', report)
                rows.append({**row, **levels, 'stage': job['stage']})
                reason = reject_reason(rows, base, stage_end=position in (9, 39))
                frozen.write_json(directory / 'progress.json', {'completed': len(rows), 'reason': reason,
                                                                 'last_case': prior.key(row), 'time': time.time()})
                if reason:
                    break
        save(directory / 'restored.json', prior.strength.check_scaling(model))
    save(work / (arm + '_result.json'), {'arm': arm, 'decision': 'rejected' if reason else 'pilot_pass',
        'reason': reason, 'completed': len(rows), 'skipped_after_gate': len(jobs) - len(rows),
        'rows': rows, 'matched_base': [base[prior.key(r)] for r in rows],
        'native_quality': 'unverified', 'training_steps': 0})
    prior.strength.check_inputs(work)


def run(work):
    plan = prior.strength.check_inputs(work)
    if plan['arms'] != ARMS or plan['max_outputs'] != 80 or plan['training_steps'] != 0:
        raise ValueError('Pilot scope changed')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    with (work / 'run.started').open('x', encoding='utf-8') as stream:
        stream.write(frozen.sha256(work / 'plan.json'))
    for name in ('claims', 'items'):
        (work / name).mkdir()
    processes = []
    try:
        for arm in ARMS:
            with (work / f'{arm}.log').open('x', encoding='utf-8') as log:
                processes.append(subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()),
                    'worker', '--work', str(work), '--arm', arm], cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Pilot worker failed; no automatic retry')
            frozen.write_json(work / 'status.json', {'stage': 'evaluating', 'time': time.time(),
                'completed': len(list((work / 'items').glob('*.json'))), 'max_outputs': 80,
                'pids': [p.pid for p in processes if p.poll() is None]})
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Pilot worker failed')
        results = {arm: read(work / (arm + '_result.json')) for arm in ARMS}
        save(work / 'comparison.json', {'results': results, 'training_steps': 0, 'production_changed': False})
        frozen.write_json(work / 'status.json', {'stage': 'complete', 'time': time.time(),
            'completed': sum(r['completed'] for r in results.values()),
            'decisions': {arm: r['decision'] for arm, r in results.items()}, 'pids': []})
    except BaseException as error:
        frozen.write_json(work / 'status.json', {'stage': 'failed', 'time': time.time(), 'error': str(error)})
        raise
    finally:
        for process in processes:
            prior.stop(process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(ARMS))
    args = parser.parse_args()
    if args.action == 'worker':
        if args.arm is None:
            parser.error('worker needs --arm')
        worker(args.work, args.arm)
    else:
        {'prepare': prepare, 'run': run}[args.action](args.work)
