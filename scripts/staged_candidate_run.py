"""Frozen cold-start candidates gated inside the live training process."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

import r31_component_probe as probe

prior = probe.prior
frozen, read, save = prior.frozen, prior.read, prior.save
GATES = (150, 300, 459)
PILOT = Path('/root/autodl-tmp/r31_component_probe_20261030/run')
PILOT_SHA = '176812d3f4129274293183bacf5a7e07fcb847148d0cf6da3a7c5220e5c9dfb6'
R33_DEVELOPMENT = Path('/root/autodl-tmp/r33_development_20261030_v2/run')
R33_DEVELOPMENT_SHA = '12e21245f775913548f262519df6da818a8a4eb2ceb2ce79985c92b3e01619e5'
PAIRED = Path('/root/autodl-tmp/quality_reaudit_20261030/paired_candidate_20261031_v2')
PAIRED_SHA = '60789ed48d2c4c4c4cec3126c3f7c787b1fb75dc52d186fe7b82e434f2781a82'
PAIRED_COMPLETION_SHA = 'ed647b04198e11da6ee2a7ce1ea1aac8d3c49a7672dad58b2d1d3d6199f56b28'


def acoustic_gate(rows, baseline):
    """Reuse existing absolute-tail, duration and speech-ratio policy before resuming."""
    from voxft.train.runlog import _duration_gates
    matched = [baseline[prior.key(row)] for row in rows]
    reports = [{'label':label, **frozen.evaluation._report_metrics(items)}
               for label, items in (('base', matched), ('candidate', rows))]
    gates = _duration_gates(reports, tail_policy='absolute')['candidate']
    return {key: values for key, values in gates.items()
            if values and key != 'tail_silence_observations'}


def cer_observations(rows, base):
    from voxft.train.runlog import CER_NOISE, _regressed
    reports = [{'label':name, **frozen.evaluation._report_metrics(items)} for name, items in
               (('base', [base[prior.key(r)] for r in rows]), ('candidate', rows))]
    return {'noise_threshold':CER_NOISE, 'classification':_regressed(reports)['candidate'],
            'interpretation':'Review evidence only; below threshold does not establish non-regression.'}


VARIANTS = {
    'rank32': {'changes': {'r':32, 'alpha':32}, 'free_gib':5.5,
               'run_name':'lora_joint3_r32_rank32_staged',
               'hypothesis':'Reduce rank64→32 with alpha/r=1; hold other training settings fixed.'},
    'dit_only': {'changes': {'enable_lm':False}, 'free_gib':4.9,
                 'run_name':'lora_joint3_r33_dit_only_staged',
                 'hypothesis':'Freeze LM, train only DiT at original rank/alpha64; hold all other training settings fixed.'},
    'dit_lr5e5': {'changes': {'enable_lm':False}, 'learning_rate':5e-5, 'free_gib':5.,
                 'run_name':'lora_joint3_r34_dit_lr5e5_staged',
                 'hypothesis':'Relative to r33, only halve learning rate to 5e-5; same DiT-only rank64, '
                              'data, seed and 459-update schedule. Test smaller adaptation after a new '
                              'TH overread observation; this is not proof that learning rate caused it.'},
    'dit_paired': {'changes': {'enable_lm':False}, 'free_gib':5.5,
                   'run_name':'lora_joint3_r35_dit_paired_staged',
                   'hypothesis':'Test the fully screened original/derived duration-balanced candidate at '
                                'one nominal epoch with r33 DiT-only optimizer settings. Cleaning, mixing '
                                'and partial separation change together; no single-factor causal claim.'},
}


def candidate_config(parent, work, variant, *, train_rows=None):
    if variant not in VARIANTS or parent['lora']['r'] != 64 or parent['lora']['alpha'] != 64:
        raise ValueError('Expected a predeclared intervention against original rank64 r31')
    if parent['num_iters'] != 459 or parent['max_steps'] != 459:
        raise ValueError('Keep the original full learning-rate/progress schedule')
    result = {**parent, 'lora': {**parent['lora'], **VARIANTS[variant]['changes']},
            'learning_rate': VARIANTS[variant].get('learning_rate', parent['learning_rate']),
            'train_manifest': str(work / 'train.jsonl'), 'val_manifest': str(work / 'val.jsonl'),
            'save_path': str(work / 'checkpoint'), 'tensorboard': str(work / 'checkpoint/logs'),
            'save_interval': 150}
    if variant == 'dit_paired':
        if train_rows != 1881:
            raise ValueError('Expected the frozen 1881-row paired candidate')
        updates = math.ceil(train_rows / (parent['batch_size'] * parent['grad_accum_steps']))
        result.update(num_iters=updates, max_steps=updates, warmup_steps=math.ceil(updates / 10),
                      save_interval=1)  # The wrapper saves only declared gates, not every update.
    return result


def gate_steps(cfg, variant):
    if variant != 'dit_paired':
        return GATES
    block = math.ceil(cfg['num_iters'] / 3)
    return (block - 1, 2 * block - 1, cfg['num_iters'])


def paired_inputs():
    """Bind admission evidence and the exact chosen bytes; never repeat ASR or separation."""
    prior.review.checked(PAIRED / 'plan.json', PAIRED_SHA)
    prior.review.checked(PAIRED / 'completion.json', PAIRED_COMPLETION_SHA)
    completion = read(PAIRED / 'completion.json')
    if (completion['status'] != 'candidate_prepared' or completion['pairs_verified'] != 4985
            or completion['acoustic_checked'] != 4985):
        raise ValueError('Incomplete paired candidate')
    inputs = {**completion['outputs_sha256'], str(PAIRED / 'completion.json'):PAIRED_COMPLETION_SHA}
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    mix, mapping = read(PAIRED / 'mix.json'), read(PAIRED / 'audio_mapping.json')
    if (not mix['training_authorized'] or not mix['original_validation_unchanged']
            or mix['max_origin_exposure'] > 2 or mix['max_ref_exposure'] > 3
            or mix['tl_verified_refs'] != 0):
        raise ValueError('Candidate admission or exposure differs')
    for audit in read(PAIRED / 'isolation.json').values():
        if (audit['metadata']['violations'] or audit['metadata']['origins_above_3x']
                or audit['metadata']['validation_rows'] != 1634 or any(audit['pcm'].values())):
            raise ValueError('Candidate isolation failed')
    selected = {value['selected']:(raw, value) for raw, value in mapping.items()}
    rows = frozen._read_manifest(PAIRED / 'train.jsonl')
    if len(rows) != mix['train']['rows'] or len(rows) != 1881:
        raise ValueError('Candidate row count differs')
    for row in rows:
        for field, origin in (('audio', 'origin_audio'), ('ref_audio', 'ref_origin_audio')):
            if not row.get(field):
                continue
            raw, record = selected[row[field]]
            if record['decision'] not in ('original', 'vocals') or row[origin] != record['origin_audio']:
                raise ValueError('Unapproved or remapped candidate ancestry')
            inputs[raw] = record['original_sha256']
            inputs[row[field]] = record['file_sha256']
    if frozen.sha256(PAIRED / 'val.jsonl') != frozen.sha256(prior.SOURCE / 'val.jsonl'):
        raise ValueError('Original validation changed')
    return inputs, mix


def add_failure_canary(jobs, base_items, cases, baseline):
    """Put all five seeds of the newly observed r33 failure ahead of old pilot work."""
    case, = [c for c in cases if c['case_id'] == 'fresh_th_1844_control']
    new = [{'case':case, 'seed':seed, 'stage':'new_risk'} for seed in (44, 43, 45, 42, 46)]
    lookup = {prior.key(r):r for r in baseline}
    added = []
    for job in new:
        row = lookup[job['case']['case_id'], job['seed']]
        if any(row.get(k) != v for k, v in case.items()):
            raise ValueError('New canary does not match its frozen baseline')
        added.append(row)
    combined = new + jobs
    if len({(j['case']['case_id'], j['seed']) for j in combined}) != len(combined):
        raise ValueError('Duplicate canary output; no reruns')
    return combined, added + base_items


def prioritize_tail_risk(jobs):
    risk = [j for j in jobs if j['case']['case_id'] == 'fresh_tl_1913_control']
    if {j['seed'] for j in risk} != set(frozen.SEEDS) or len(risk) != 5:
        raise ValueError('Expected all five TL1913 tail-risk seeds')
    front = [j for j in jobs if j['stage'] == 'new_risk']
    rest = [j for j in jobs if j not in front and j not in risk]
    risk.sort(key=lambda j: (j['seed'] != 43, j['seed']))
    return front + [{**j, 'stage':'tail_risk'} for j in risk] + rest


def end_of_stage(jobs, position):
    if position == len(jobs) - 1:
        return True
    current, following = jobs[position], jobs[position + 1]
    return (current['stage'] != following['stage'] or
            current['stage'] == 'guard' and current['case']['lang'] != following['case']['lang'])


def prepare(work, variant='rank32'):
    if work.exists():
        raise FileExistsError(work)
    prior.review.checked(prior.SOURCE / 'plan.json', prior.SOURCE_PLAN_SHA)
    parent = read(prior.SOURCE / 'plan.json')
    prior.review.checked(PILOT / 'plan.json', PILOT_SHA)
    pilot = prior.strength.check_inputs(PILOT)
    inputs = {**parent['inputs_sha256'], **pilot['inputs_sha256']}
    jobs, base_items = read(PILOT / 'jobs.json'), read(PILOT / 'base_items.json')
    data_source, mix = prior.SOURCE, None
    if variant == 'dit_paired':
        candidate_inputs, mix = paired_inputs()
        if any(path in inputs and inputs[path] != digest for path, digest in candidate_inputs.items()):
            raise ValueError('Candidate provenance SHA conflict')
        inputs.update(candidate_inputs)
        data_source = PAIRED
    if variant in ('dit_lr5e5', 'dit_paired'):
        prior.review.checked(R33_DEVELOPMENT / 'plan.json', R33_DEVELOPMENT_SHA)
        prior.strength.check_inputs(R33_DEVELOPMENT)
        result = read(R33_DEVELOPMENT / 'result.json')
        if result['reason'] != 'new_suspected_overread' or result['completed'] != 290:
            raise ValueError('Expected frozen r33 expansion failure evidence')
        row, = [r for r in result['rows'] if prior.key(r) == ('fresh_th_1844_control', 44)]
        if not row['over_read'] or row['len_ratio'] <= 1.4:
            raise ValueError('Expected the new canary failure')
        inputs[row['wav']] = result['wav_sha256'][row['wav']]
        for name in ('plan.json', 'result.json'):
            path = R33_DEVELOPMENT / name
            inputs[str(path)] = frozen.sha256(path)
        baseline = read(prior.SOURCE / 'base_report.json')
        jobs, base_items = add_failure_canary(jobs, base_items,
            frozen._read_manifest(prior.SOURCE / 'cases.jsonl'), baseline['items'])
        for row in base_items:
            inputs[row['wav']] = baseline['wav_sha256'][row['wav']]
        if variant == 'dit_paired':
            jobs = prioritize_tail_risk(jobs)
    reserved = frozen.ROOT / 'eval_cases/reserved_final_20261030.jsonl'
    digest = 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c'
    prior.review.checked(reserved, digest)
    inputs[str(reserved)] = digest
    from incremental_mix import target_text
    heldout = {frozen.evaluation._norm(r['text']) for r in frozen._read_manifest(reserved)}
    if any(frozen.evaluation._norm(target_text(r)) in heldout
           for r in frozen._read_manifest(data_source / 'train.jsonl')):
        raise ValueError('Reserved final text in training')
    if any(frozen.evaluation._norm(j['case']['text']) in heldout for j in jobs):
        raise ValueError('Reserved final text in development pilot')
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    settings = VARIANTS[variant]
    if shutil.disk_usage(work.parent).free < settings['free_gib'] * 1024**3:
        raise RuntimeError(f"Need {settings['free_gib']} GiB for snapshots, bounded gates and 4 GiB reserve")
    work.mkdir()
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json'):
        shutil.copyfile(data_source / name, work / name)
    if mix is not None:
        shutil.copyfile(data_source / 'mix.json', work / 'mix.json')
    save(work / 'jobs.json', jobs)
    save(work / 'base_items.json', base_items)
    cfg = candidate_config(yaml.safe_load((prior.SOURCE / 'train.yaml').read_text(encoding='utf-8')), work,
                           variant, train_rows=mix['train']['rows'] if mix else None)
    gates = gate_steps(cfg, variant)
    updates = {str(step): min(step + 1, cfg['num_iters']) for step in gates}
    (work / 'train.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    from voxft.train.launcher import preflight
    frozen.require_fresh_checkpoint(cfg)
    issues = preflight(work / 'train.yaml', gpus=1)
    Path(cfg['save_path']).rmdir()
    if any(not i.startswith('警告') for i in issues):
        raise ValueError(issues)
    for path in work.iterdir():
        if path.is_file():
            inputs[str(path)] = frozen.sha256(path)
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src', 'third_party/VoxCPM/scripts'):
        for path in (frozen.ROOT / folder).rglob('*'):
            if path.suffix in ('.py', '.sh'):
                inputs[str(path)] = frozen.sha256(path)
    inputs[str(prior.SOURCE / 'plan.json')] = prior.SOURCE_PLAN_SHA
    inputs[str(PILOT / 'plan.json')] = PILOT_SHA
    early_rejection = list(pilot['early_rejection'])
    if variant == 'dit_paired':
        early_rejection.remove('stage-end any language mean CER delta>.01')
        early_rejection.append('small-pilot mean CER is observation only; severe per-output checks stay active')
    data = {'sources':mix['sources'], 'language_shares':mix['language_shares']} if mix else {}
    save(work / 'plan.json', {**{k: parent[k] for k in ('base_path', 'asr_path', 'speaker_model_path',
            'sources', 'language_shares')}, 'inputs_sha256': inputs, 'rank': cfg['lora']['r'],
        **data, 'data_source':str(data_source),
        'variant':variant, 'run_name':settings['run_name'], 'max_training_updates': cfg['num_iters'],
        'nominal_epochs':cfg['num_iters'] * cfg['batch_size'] * cfg['grad_accum_steps'] /
                         len(frozen._read_manifest(work / 'train.jsonl')),
        'training_authorized': True, 'authorization': 'User continuous iteration and hyperparameter optimization',
        'gates': list(gates), 'gate_optimizer_updates': updates,
        'pilot_outputs_per_gate_max': len(jobs), 'new_outputs_max': len(gates) * len(jobs),
        'pilot_mean_cer_observe_only':variant == 'dit_paired',
        'early_rejection': [*early_rejection, 'stage-end existing absolute-tail/duration/speech-ratio gates'],
        'tail_policy':'absolute', 'production_changed': False,
        'hypothesis': settings['hypothesis'],
        'limits': 'Same training seed, not a multi-seed training causal claim. Initialization shape changes RNG '
                  'consumption. Development pilot only; final independent regression and native quality pending.',
        'execution': 'Save complete checkpoint, wait inside the same live trainer; evaluate fixed five-seed '
                     'pilot. Reject exits without more updates; pass continues with optimizer/RNG/data iterator intact. '
                     'Intermediate step labels are zero-based; final label equals the completed update count. '
                     'No official source patches or resume.'})
    print(f'PREPARED {variant}; gates at optimizer updates {list(updates.values())}', flush=True)


class GateRejected(Exception):
    pass


def checkpoint_gate(work, step, optimizer, *, gates=GATES):
    if step not in gates:
        return
    snapshot = work / 'checkpoint' / f'step_{step:07d}'
    actual_updates = {int(state['step']) for state in optimizer.state.values() if 'step' in state}
    expected = min(step + 1, gates[-1])
    if actual_updates != {expected}:
        raise ValueError(f'Optimizer steps disagree: {actual_updates}')
    directory = work / f'gate_{step}'
    directory.mkdir()
    save(directory / 'checkpoint.json', {'step_label': step, 'optimizer_updates': expected,
        'target': str(snapshot),
        'inputs_sha256': {str(p): frozen.sha256(p) for p in snapshot.iterdir() if p.is_file()}})
    save(directory / 'ready.json', {'pid': os.getpid(), 'time': time.time()})
    decision = directory / 'decision.json'
    controller_pid = os.getppid()
    while not decision.exists():
        if controller_pid == 1 or os.getppid() != controller_pid:
            raise RuntimeError('Controller exited; refusing further training')
        time.sleep(1)
    result = read(decision)
    if result['decision'] != 'pilot_pass':
        raise GateRejected(result['reason'])


def train(work):
    cfg = yaml.safe_load((work / 'train.yaml').read_text(encoding='utf-8'))
    gates = gate_steps(cfg, read(work / 'plan.json')['variant'])
    path = frozen.VOXCPM_REPO / 'scripts/train_voxcpm_finetune.py'
    spec = importlib.util.spec_from_file_location('voxft_official_trainer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.save_checkpoint

    def save_and_wait(model, optimizer, scheduler, save_dir, step, *args, **kwargs):
        if step not in gates:
            return  # Skip redundant non-gate snapshots; training math is unchanged.
        original(model, optimizer, scheduler, save_dir, step, *args, **kwargs)
        checkpoint_gate(work, step, optimizer, gates=gates)

    module.save_checkpoint = save_and_wait
    try:
        module.train(**cfg)
    except GateRejected as error:
        save(work / 'training_stopped.json', {'reason': str(error), 'time': time.time(),
            'status': 'rejected_at_complete_checkpoint', 'step_label': read(work / 'checkpoint/latest/training_state.json')['step']})
    finally:
        module.save_checkpoint = original


def evaluate_gate(work, step):
    plan = read(work / 'plan.json')
    if step not in plan['gates']:
        raise ValueError('Undeclared evaluation gate')
    directory = work / f'gate_{step}'
    checkpoint = read(directory / 'checkpoint.json')
    for path, digest in checkpoint['inputs_sha256'].items():
        prior.review.checked(path, digest)
    frozen.runtime_environment(directory / 'runtime')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VOXFT_WHISPER_MODEL_LARGE=plan['asr_path'], VOXFT_SPK_EMB_MODEL=plan['speaker_model_path'],
        TORCHINDUCTOR_CACHE_DIR=str(directory / 'runtime/inductor'))
    frozen.evaluation.EVAL_DIR = directory / 'raw_reports'
    frozen.evaluation.infer.CHECKPOINT_DIR = directory
    jobs, base = read(work / 'jobs.json'), {prior.key(r): r for r in read(work / 'base_items.json')}
    target = checkpoint['target']
    runtime_plan = {**plan, 'targets': {'candidate': target}}
    rows, reason, acoustic_failures = [], None, {}
    (directory / 'items').mkdir()
    with prior.fast.cached_services([j['case'] for j in jobs]):
        save(directory / 'runtime.json', prior.baseline.gpu_services(runtime_plan, [j['case'] for j in jobs], 'candidate'))
        for position, job in enumerate(jobs):
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError('Disk reserve reached')
            claim = directory / 'claims'
            claim.mkdir(exist_ok=True)
            if not prior.fast.claim(claim, str(position)):
                raise FileExistsError('No implicit retry')
            report = frozen.evaluation.evaluate(target, 'th', [job['case']], base=plan['base_path'],
                seeds=[job['seed']], cfg_value=1.8, inference_timesteps=20)
            row, = report['items']
            report['wav_sha256'] = {row['wav']: frozen.sha256(row['wav'])}
            prior.verify(report, runtime_plan, job['case'], 'candidate', 1., job['seed'])
            save(directory / 'items' / f'{position:02d}.json', report)
            rows.append({**row, **prior.review.level_metrics(row['wav']), 'stage': job['stage']})
            # Check protection at each completed language, not only the entire budget.
            stage_end = end_of_stage(jobs, position)
            reason = probe.reject_reason(rows, base,
                stage_end=stage_end and not plan.get('pilot_mean_cer_observe_only', False))
            if stage_end:
                acoustic_failures = acoustic_gate(rows, base)
                if acoustic_failures and reason is None:
                    reason = 'stage_acoustic_regression'
            frozen.write_json(directory / 'progress.json', {'completed': len(rows), 'reason': reason,
                'last_case': prior.key(row), 'time': time.time()})
            if reason:
                break
    for path, digest in checkpoint['inputs_sha256'].items():
        prior.review.checked(path, digest)
    save(directory / 'result.json', {'decision': 'rejected' if reason else 'pilot_pass',
        'reason': reason, 'completed': len(rows), 'skipped': len(jobs)-len(rows),
        'acoustic_failures':acoustic_failures, 'tail_policy':'absolute',
        'cer_observations':cer_observations(rows, base),
        'rows': rows, 'matched_base': [base[prior.key(r)] for r in rows], 'native_quality': 'unverified'})


def run(work):
    plan = prior.strength.check_inputs(work)
    cfg = yaml.safe_load((work / 'train.yaml').read_text(encoding='utf-8'))
    gates = gate_steps(cfg, plan.get('variant'))
    if plan['gates'] != list(gates) or plan.get('variant') not in VARIANTS or plan['tail_policy'] != 'absolute':
        raise ValueError('Frozen experiment scope differs')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    frozen.require_fresh_checkpoint(cfg)
    with (work / 'run.started').open('x', encoding='utf-8') as f:
        f.write(frozen.sha256(work / 'plan.json'))
    frozen.runtime_environment(work / 'runtime')
    trainer, evaluator, bridge = None, None, None
    decisions = {}
    try:
        for line in Path('/root/voxcpm-finetune/.env').read_text(encoding='utf-8').splitlines():
            key, _, value = line.partition('=')
            if key.strip() in {'WANDB_API_KEY', 'WANDB_PROJECT'}:
                os.environ.setdefault(key.strip(), value.strip())
        with (work / 'train.log').open('x', encoding='utf-8') as log:
            trainer = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', __file__, 'train', '--work', str(work)],
                cwd=frozen.VOXCPM_REPO, stdout=log, stderr=subprocess.STDOUT)
        if os.environ.get('WANDB_API_KEY'):
            with (work / 'wandb.log').open('x', encoding='utf-8') as log:
                bridge = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', '-m', 'voxft.train.tb_wandb_bridge',
                    cfg['tensorboard'], plan['run_name']], cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
        while trainer.poll() is None:
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError('Disk reserve reached')
            for step in gates:
                directory = work / f'gate_{step}'
                if step in decisions or not (directory / 'ready.json').exists():
                    continue
                free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free < 12 * 1024:
                    raise RuntimeError('Less than 12 GiB free for gated evaluator beside waiting trainer')
                frozen.write_json(work / 'status.json', {'stage': 'gated_evaluation', 'step_label': step,
                    'training_pid': trainer.pid, 'time': time.time()})
                with (directory / 'eval.log').open('x', encoding='utf-8') as log:
                    evaluator = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', __file__, 'evaluate',
                        '--work', str(work), '--step', str(step)], cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT)
                while evaluator.poll() is None:
                    if trainer.poll() is not None:
                        raise RuntimeError('Waiting trainer exited during gate')
                    time.sleep(2)
                if evaluator.returncode:
                    raise RuntimeError('Gated evaluator failed; no retry')
                result = read(directory / 'result.json')
                decisions[step] = {k: result[k] for k in ('decision', 'reason', 'completed', 'skipped')}
                save(directory / 'decision.pending.json', decisions[step])
                (directory / 'decision.pending.json').rename(directory / 'decision.json')
            frozen.write_json(work / 'status.json', {'stage': 'training', 'time': time.time(),
                'decisions': decisions, 'training_pid': trainer.pid})
            time.sleep(2)
        if trainer.returncode:
            raise RuntimeError(f'Trainer failed: {trainer.returncode}')
        rejected = any(r['decision'] == 'rejected' for r in decisions.values())
        if not rejected and set(decisions) != set(gates):
            raise ValueError('Trainer ended without all declared gates')
        prior.strength.check_inputs(work)
        save(work / 'result.json', {'decision': 'rejected' if rejected else 'development_pilot_pass',
            'gates': decisions, 'production_changed': False, 'native_quality': 'unverified'})
        frozen.write_json(work / 'status.json', {'stage': 'rejected' if rejected else 'pilot_complete',
            'decisions': decisions, 'time': time.time(), 'training_pid': None})
    except BaseException as error:
        frozen.write_json(work / 'status.json', {'stage': 'failed', 'error': str(error), 'time': time.time()})
        raise
    finally:
        prior.stop(evaluator)
        # The official SIGTERM handler saves a checkpoint. On an exception at a
        # completed gate, kill the waiting trainer without rewriting frozen inputs.
        if trainer is not None and trainer.poll() is None:
            trainer.kill()
            trainer.wait()
        prior.stop(bridge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'train', 'evaluate'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='rank32')
    args = parser.parse_args()
    if args.action == 'evaluate':
        if args.step is None:
            parser.error('evaluate requires --step')
        evaluate_gate(args.work, args.step)
    elif args.action == 'prepare':
        prepare(args.work, args.variant)
    else:
        {'run': run, 'train': train}[args.action](args.work)
