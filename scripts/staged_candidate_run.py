"""Single-factor cold-start candidates gated inside the live training process."""
import argparse
import importlib.util
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


def acoustic_gate(rows, baseline):
    """Reuse existing absolute-tail, duration and speech-ratio policy before resuming."""
    from voxft.train.runlog import _duration_gates
    matched = [baseline[prior.key(row)] for row in rows]
    reports = [{'label':label, **frozen.evaluation._report_metrics(items)}
               for label, items in (('base', matched), ('candidate', rows))]
    gates = _duration_gates(reports, tail_policy='absolute')['candidate']
    return {key: values for key, values in gates.items()
            if values and key != 'tail_silence_observations'}


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
}


def candidate_config(parent, work, variant):
    if variant not in VARIANTS or parent['lora']['r'] != 64 or parent['lora']['alpha'] != 64:
        raise ValueError('Expected a predeclared intervention against original rank64 r31')
    if parent['num_iters'] != 459 or parent['max_steps'] != 459:
        raise ValueError('Keep the original full learning-rate/progress schedule')
    return {**parent, 'lora': {**parent['lora'], **VARIANTS[variant]['changes']},
            'learning_rate': VARIANTS[variant].get('learning_rate', parent['learning_rate']),
            'train_manifest': str(work / 'train.jsonl'), 'val_manifest': str(work / 'val.jsonl'),
            'save_path': str(work / 'checkpoint'), 'tensorboard': str(work / 'checkpoint/logs'),
            'save_interval': 150}


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
    if variant == 'dit_lr5e5':
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
    reserved = frozen.ROOT / 'eval_cases/reserved_final_20261030.jsonl'
    digest = 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c'
    prior.review.checked(reserved, digest)
    inputs[str(reserved)] = digest
    from incremental_mix import target_text
    heldout = {frozen.evaluation._norm(r['text']) for r in frozen._read_manifest(reserved)}
    if any(frozen.evaluation._norm(target_text(r)) in heldout
           for r in frozen._read_manifest(prior.SOURCE / 'train.jsonl')):
        raise ValueError('Reserved final text in training')
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    settings = VARIANTS[variant]
    if shutil.disk_usage(work.parent).free < settings['free_gib'] * 1024**3:
        raise RuntimeError(f"Need {settings['free_gib']} GiB for snapshots, bounded gates and 4 GiB reserve")
    work.mkdir()
    for name in ('train.jsonl', 'val.jsonl', 'flagged.json'):
        shutil.copyfile(prior.SOURCE / name, work / name)
    save(work / 'jobs.json', jobs)
    save(work / 'base_items.json', base_items)
    cfg = candidate_config(yaml.safe_load((prior.SOURCE / 'train.yaml').read_text(encoding='utf-8')), work, variant)
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
    save(work / 'plan.json', {**{k: parent[k] for k in ('base_path', 'asr_path', 'speaker_model_path',
            'sources', 'language_shares')}, 'inputs_sha256': inputs, 'rank': cfg['lora']['r'],
        'variant':variant, 'run_name':settings['run_name'], 'max_training_updates': 459,
        'training_authorized': True, 'authorization': 'User continuous iteration and hyperparameter optimization',
        'gates': list(GATES), 'gate_optimizer_updates': {'150': 151, '300': 301, '459': 459},
        'pilot_outputs_per_gate_max': len(jobs), 'new_outputs_max': len(GATES) * len(jobs),
        'early_rejection': [*pilot['early_rejection'], 'stage-end existing absolute-tail/duration/speech-ratio gates'],
        'tail_policy':'absolute', 'production_changed': False,
        'hypothesis': settings['hypothesis'],
        'limits': 'Same training seed, not a multi-seed training causal claim. Initialization shape changes RNG '
                  'consumption. Development pilot only; final independent regression and native quality pending.',
        'execution': 'Save complete checkpoint, wait inside the same live trainer; evaluate fixed five-seed '
                     'pilot. Reject exits without more updates; pass continues with optimizer/RNG/data iterator intact. '
                     'Official step_150/300 are zero-based save labels (151/301 updates). No source patches or resume.'})
    print(f'PREPARED {variant}; gates at optimizer updates 151,301,459', flush=True)


class GateRejected(Exception):
    pass


def checkpoint_gate(work, step, optimizer):
    if step not in GATES:
        return
    snapshot = work / 'checkpoint' / f'step_{step:07d}'
    actual_updates = {int(state['step']) for state in optimizer.state.values() if 'step' in state}
    expected = step + 1 if step < 459 else 459
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
    path = frozen.VOXCPM_REPO / 'scripts/train_voxcpm_finetune.py'
    spec = importlib.util.spec_from_file_location('voxft_official_trainer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.save_checkpoint

    def save_and_wait(model, optimizer, scheduler, save_dir, step, *args, **kwargs):
        if step not in GATES:
            return  # Skip redundant non-gate snapshots; training math is unchanged.
        original(model, optimizer, scheduler, save_dir, step, *args, **kwargs)
        checkpoint_gate(work, step, optimizer)

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
            reason = probe.reject_reason(rows, base, stage_end=stage_end)
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
        'rows': rows, 'matched_base': [base[prior.key(r)] for r in rows], 'native_quality': 'unverified'})


def run(work):
    plan = prior.strength.check_inputs(work)
    if plan['gates'] != list(GATES) or plan.get('variant') not in VARIANTS or plan['tail_policy'] != 'absolute':
        raise ValueError('Frozen experiment scope differs')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    cfg = yaml.safe_load((work / 'train.yaml').read_text(encoding='utf-8'))
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
            for step in GATES:
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
        if not rejected and set(decisions) != set(GATES):
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
    parser.add_argument('--step', type=int, choices=GATES)
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
