"""Bounded development evaluation; reuse outputs and keep diagnostic results separate."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import staged_candidate_run as staged

prior, probe = staged.prior, staged.probe
frozen, read, save = staged.frozen, staged.read, staged.save
CANDIDATE = Path('/root/autodl-tmp/r33_dit_only_20261030/run')
CANDIDATE_SHA = 'd1c19ebed1c74c165751bae49e3752894468698f3f0881ae8abb1f3bbf6efda1'
DIAGNOSTIC = Path('/root/autodl-tmp/r34_dit_lr5e5_20261030/run')
DIAGNOSTIC_SHA = '90574851e38af8582a2b9339137d3f887f957e5c942284a700c73cd908e79cb8'
R39 = Path('/root/autodl-tmp/r39_drop0_20261031_v2/run')
R39_SHA = '9b40d522071727eef4163ea70b5662098745c3c27617cd3103ee49074e631b5e'
R39_VERIFICATION_SHA = '3a9ab7f53efdad4206fed2646b347ccb73e96d6ee87a826220c8e2b26f53f8ea'


def publish(path, value):
    """Publish complete immutable JSON without overwriting another result."""
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    save(pending, value)
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def update(path, value):
    """Readers see either the old or new whole document, including stop decisions."""
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    save(pending, value)
    os.replace(pending, path)


def select_jobs(cases, pilot, *, canary_first=False):
    by_lang = {}
    for lang in ('id', 'th', 'tl'):
        pool = {c['sentence_id']: c for c in cases if c['lang'] == lang and c['condition'] == 'plain'
                and not any(mark in c['text'] for mark in ('Ã', 'Â', 'Å', 'Ä', '\ufffd'))}
        guards = {j['case']['sentence_id'] for j in pilot
                  if j['stage'] == 'guard' and j['case']['lang'] == lang}
        if len(guards) != 1:
            raise ValueError('Expected one existing guard per language')
        chosen = sorted(guards) + [sid for sid in prior.SENTENCES if sid in pool and sid not in guards]
        chosen += sorted(pool.keys() - set(chosen), key=lambda sid: hashlib.sha256(sid.encode()).hexdigest())
        by_lang[lang] = chosen[:10]
        if len(by_lang[lang]) != 10:
            raise ValueError('Insufficient distinct development texts')
    if canary_first:
        # Order changes only; include both conditions and all seeds, never just the failing take.
        by_lang['th'].remove('fresh_th_1844')
        by_lang['th'].insert(0, 'fresh_th_1844')
    jobs = []
    for stage in range(10):
        for lang in ('id', 'th', 'tl'):
            group = sorted((c for c in cases if c['sentence_id'] == by_lang[lang][stage]),
                           key=lambda c: c['condition'])
            if len(group) != 2 or {c['condition'] for c in group} != {'plain', 'control'}:
                raise ValueError('Expected paired plain/control cases')
            for case in group:
                for seed in frozen.SEEDS:
                    jobs.append({'id': f"{case['case_id']}_{seed}", 'case': case, 'seed': seed, 'stage': stage})
    if len({j['id'] for j in jobs}) != 300:
        raise ValueError('Unexpected or duplicate development coverage')
    return jobs


def parent_gate(profile, status, result):
    if profile == 'r39_early_diagnostic':
        gates = result.get('gates', {})
        if (status['stage'] != 'rejected' or result['decision'] != 'rejected'
                or set(gates) != {'39', '79', '118'}
                or any(gates[s] != {'decision':'pilot_pass', 'reason':None,
                                   'completed':45, 'skipped':0} for s in ('39', '79'))
                or gates['118'] != {'decision':'rejected', 'reason':'new_suspected_overread',
                                    'completed':3, 'skipped':42}):
            raise ValueError('Expected verified r39 early passes and final three-output rejection')
        return 79, 80, 45
    if profile == 'r34_diagnostic':
        gate = result.get('gates', {}).get('150', {})
        if (status['stage'] != 'rejected' or result['decision'] != 'rejected'
                or set(result.get('gates', {})) != {'150'}
                or gate.get('reason') != 'stage_mean_cer_regression_th' or gate.get('completed') != 5):
            raise ValueError('Expected the stopped five-output r34 mean-only gate')
        return 150, 151, 5
    if (profile not in ('r33', 'r33_early_diagnostic') or status['stage'] != 'pilot_complete'
            or result['decision'] != 'development_pilot_pass'):
        raise ValueError('All training gates must pass before expanding latest')
    if profile == 'r33_early_diagnostic':
        return 150, 151, 40
    return 459, 459, 40


def prepare(work, profile='r33'):
    if work.exists():
        raise FileExistsError(work)
    diagnostic = profile in ('r34_diagnostic', 'r33_early_diagnostic', 'r39_early_diagnostic')
    candidate, digest = (DIAGNOSTIC, DIAGNOSTIC_SHA) if profile == 'r34_diagnostic' else (CANDIDATE, CANDIDATE_SHA)
    if profile == 'r39_early_diagnostic':
        candidate, digest = R39, R39_SHA
        prior.review.checked(candidate / 'verification.json', R39_VERIFICATION_SHA)
        verification = read(candidate / 'verification.json')
        if (verification['status'] != 'rejected_verified' or verification['completed_outputs'] != 93
                or verification['optimizer_step_counts'] != {'118':96}
                or verification['unscored_wavs'] != 0 or verification['plan_sha256'] != digest):
            raise ValueError('Incomplete r39 terminal verification')
        for path, expected in verification['outputs_sha256'].items():
            prior.review.checked(path, expected)
    prior.review.checked(candidate / 'plan.json', digest)
    parent = prior.strength.check_inputs(candidate)
    step, updates, reused_count = parent_gate(profile, read(candidate / 'status.json'), read(candidate / 'result.json'))
    gate = candidate / f'gate_{step}'
    checkpoint = read(gate / 'checkpoint.json')
    if checkpoint['optimizer_updates'] != updates:
        raise ValueError('Expected completed latest training state')
    # Training inputs were verified above. Keep their parent plan as provenance;
    # inference workers only need to recheck model, source and evaluation inputs.
    inputs = {p:h for p,h in parent['inputs_sha256'].items() if '/models--' in p}
    inputs.update(checkpoint['inputs_sha256'])
    cases = frozen._read_manifest(prior.SOURCE / 'cases.jsonl')
    canary_first = profile in ('r33_early_diagnostic', 'r39_early_diagnostic')
    jobs = select_jobs(cases, read(candidate / 'jobs.json'), canary_first=canary_first)
    selected = {j['case']['case_id']: j['case'] for j in jobs}
    baseline_path = prior.SOURCE / 'base_report.json'
    baseline = read(baseline_path)
    prior.frozen.verify_report(baseline, {**parent, 'targets': {'base':'base'}}, cases, 'base')
    base = [r for r in baseline['items'] if r['case_id'] in selected]
    if len(base) != 300:
        raise ValueError('Incomplete matched base')
    inputs[str(baseline_path)] = frozen.sha256(baseline_path)
    for row in base:
        inputs[row['wav']] = baseline['wav_sha256'][row['wav']]
        inputs[row['ref_audio']] = parent['inputs_sha256'][row['ref_audio']]
    plan = {**{k: parent[k] for k in ('base_path', 'asr_path', 'speaker_model_path')},
            'targets': {'candidate':checkpoint['target'], 'base':'base'}}
    reused = []
    for path in sorted((gate / 'items').glob('*.json')):
        report = read(path)
        row, = report['items']
        prior.verify(report, plan, selected[row['case_id']], 'candidate', 1., row['seed'])
        inputs[str(path)] = frozen.sha256(path)
        inputs.update(report['wav_sha256'])
        reused.append({**row, **prior.review.level_metrics(row['wav'])})
    if len(reused) != reused_count or len({prior.key(r) for r in reused}) != reused_count:
        raise ValueError('Unexpected existing candidate outputs; no reruns')
    if profile == 'r39_early_diagnostic':
        lookup = {prior.key(r):r for r in base}
        if any(probe.reject_reason(reused[:i], lookup) for i in range(1, len(reused)+1)):
            raise ValueError('Existing r39 early outputs violate severe gate')
        if staged.acoustic_gate(reused, lookup, paired_extremes=True):
            raise ValueError('Existing r39 early outputs violate acoustic gate')
        reserved = frozen.ROOT / 'eval_cases/reserved_final_20261030.jsonl'
        prior.review.checked(reserved, 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c')
        heldout = {frozen.evaluation._norm(r['text']) for r in frozen._read_manifest(reserved)}
        if any(frozen.evaluation._norm(j['case']['text']) in heldout for j in jobs):
            raise ValueError('Reserved final text in diagnostic development')
        inputs[str(reserved)] = frozen.sha256(reserved)
    for path in (candidate / 'plan.json', candidate / 'result.json', candidate / 'verification.json',
                 gate / 'checkpoint.json', gate / 'result.json', prior.SOURCE / 'cases.jsonl'):
        inputs[str(path)] = frozen.sha256(path)
    comparison = []
    if profile == 'r34_diagnostic':
        # Compare learning rates at the same exposure; r33/459 is not this control.
        prior.review.checked(CANDIDATE / 'plan.json', CANDIDATE_SHA)
        equal_gate = CANDIDATE / 'gate_150'
        equal_checkpoint = read(equal_gate / 'checkpoint.json')
        if equal_checkpoint['optimizer_updates'] != updates:
            raise ValueError('Learning-rate comparison must use equal optimizer updates')
        inputs.update(equal_checkpoint['inputs_sha256'])
        for path in (CANDIDATE / 'plan.json', CANDIDATE / 'verification.json', equal_gate / 'checkpoint.json'):
            inputs[str(path)] = frozen.sha256(path)
        equal_plan = {**plan, 'targets':{'candidate':equal_checkpoint['target']}}
        for path in sorted((equal_gate / 'items').glob('*.json')):
            report = read(path)
            row, = report['items']
            prior.verify(report, equal_plan, selected[row['case_id']], 'candidate', 1., row['seed'])
            inputs[str(path)] = frozen.sha256(path)
            inputs.update(report['wav_sha256'])
            comparison.append({**row, **prior.review.level_metrics(row['wav'])})
        if len(comparison) != 40 or len({prior.key(r) for r in comparison}) != 40:
            raise ValueError('Expected 40 frozen r33/151 outputs')
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src'):
        for path in (frozen.ROOT / folder).rglob('*'):
            if path.suffix in ('.py', '.sh'):
                inputs[str(path)] = frozen.sha256(path)
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    if shutil.disk_usage(work.parent).free < 5.2 * 1024**3:
        raise RuntimeError('Need 5.2 GiB for bounded outputs, four runtimes and 4 GiB reserve')
    work.mkdir()
    for name, obj in (('jobs.json', jobs), ('base_items.json', base), ('reused.json', reused)):
        save(work / name, obj)
        inputs[str(work / name)] = frozen.sha256(work / name)
    if comparison:
        save(work / 'equal_exposure_r33.json', comparison)
        inputs[str(work / 'equal_exposure_r33.json')] = frozen.sha256(work / 'equal_exposure_r33.json')
    save(work / 'plan.json', {**plan, 'inputs_sha256':inputs, 'max_workers':4,
        'profile':profile, 'diagnostic_only':diagnostic, 'checkpoint_optimizer_updates':updates,
        'total_outputs':300, 'reused_outputs':reused_count, 'new_outputs_max':300-reused_count,
        'texts_per_language':10, 'seeds':frozen.SEEDS, 'tail_policy':'absolute',
        'paired_extreme_acoustics':profile == 'r39_early_diagnostic',
        'stages':10, 'early_rejection':([
            'existing per-output severe gates; existing absolute acoustic gates at stage boundaries',
            'mean CER uses existing CER_NOISE classification as observation, not automatic rejection',
            'diagnostic completion never promotes, resumes training or revises original rejection',
        ] if diagnostic else parent['early_rejection']), 'training_steps':0,
        'canary_first':canary_first,
        'note':'Development only. Same unmodified texts and paired five seeds; early-r33 profile puts '
               'the new TH1844 text first, otherwise existing guard first; '
               'known ID/TH stress texts next, remaining texts selected by ID hash without scores. '
               'r39 early profile tests the existing 80-update snapshot, not the rejected latest; '
               'same 300-job diagnostic coverage, 45 reused, at most 255 new. '
               'Four workers share stage barriers. On rejection finish only in-flight outputs and '
               'preserve them. No retries, no reserved final texts, no production change.',
        'production_changed':False, 'native_quality':'unverified'})
    print(f'PREPARED {profile}: 300 outputs, {reused_count} reused, at most {300-reused_count} new', flush=True)


def worker(work, index):
    plan = prior.strength.check_inputs(work)
    jobs = read(work / 'jobs.json')
    reused = {prior.key(r) for r in read(work / 'reused.json')}
    directory = work / f'worker{index}'
    frozen.runtime_environment(directory / 'runtime')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VOXFT_WHISPER_MODEL_LARGE=plan['asr_path'], VOXFT_SPK_EMB_MODEL=plan['speaker_model_path'],
        TORCHINDUCTOR_CACHE_DIR=str(directory / 'runtime/inductor'))
    frozen.evaluation.EVAL_DIR = directory / 'raw_reports'
    frozen.evaluation.infer.CHECKPOINT_DIR = directory
    with prior.fast.cached_services([j['case'] for j in jobs]):
        save(directory / 'runtime.json', prior.baseline.gpu_services(plan, [j['case'] for j in jobs], 'candidate'))
        while True:
            control = read(work / 'control.json')
            if control['stop']:
                return
            job = next((j for j in jobs if j['stage'] <= control['stage']
                        and (j['case']['case_id'], j['seed']) not in reused
                        and prior.fast.claim(work / 'claims', j['id'])), None)
            if job is None:
                time.sleep(.5)
                continue
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError('Disk reserve reached')
            report = frozen.evaluation.evaluate(plan['targets']['candidate'], 'th', [job['case']],
                base=plan['base_path'], seeds=[job['seed']], cfg_value=1.8, inference_timesteps=20)
            row, = report['items']
            report['wav_sha256'] = {row['wav']:frozen.sha256(row['wav'])}
            report['level_metrics'] = prior.review.level_metrics(row['wav'])
            prior.verify(report, plan, job['case'], 'candidate', 1., job['seed'])
            publish(work / 'items' / (job['id'] + '.json'), report)
            update(directory / 'progress.json', {'last_job':job['id'], 'time':time.time()})


def decision(rows, base, *, stage_end=False, diagnostic=False, paired_extremes=False):
    reason = probe.reject_reason(rows, base, stage_end=stage_end and not diagnostic)
    acoustics = staged.acoustic_gate(rows, base, paired_extremes=paired_extremes) if stage_end else {}
    return reason or ('stage_acoustic_regression' if acoustics else None), acoustics


def cer_observations(rows, base):
    return staged.cer_observations(rows, base)


def terminal_decision(reason, diagnostic):
    if reason:
        return 'diagnostic_stopped' if diagnostic else 'rejected'
    return 'diagnostic_complete' if diagnostic else 'development_pass'


def run(work):
    plan = prior.strength.check_inputs(work)
    if plan['max_workers'] != 4 or plan['total_outputs'] != 300 or plan['tail_policy'] != 'absolute':
        raise ValueError('Expanded development scope changed')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    with (work / 'run.started').open('x', encoding='utf-8') as stream:
        stream.write(frozen.sha256(work / 'plan.json'))
    for name in ('items', 'claims'):
        (work / name).mkdir()
    jobs = read(work / 'jobs.json')
    expected = {j['id']:j for j in jobs}
    base = {prior.key(r):r for r in read(work / 'base_items.json')}
    rows = read(work / 'reused.json')
    diagnostic = plan.get('diagnostic_only', False)
    paired_extremes = plan.get('paired_extreme_acoustics', False)
    present = {f"{r['case_id']}_{r['seed']}" for r in rows}
    processes, stages, seen = [], [], set()
    stage, reason, acoustics = 0, None, {}
    update(work / 'control.json', {'stage':0, 'stop':False})
    try:
        for index in range(4):
            (work / f'worker{index}').mkdir()
            with (work / f'worker{index}.log').open('x', encoding='utf-8') as log:
                processes.append(subprocess.Popen([sys.executable, '-X', 'utf8', '-u', __file__, 'worker',
                    '--work', str(work), '--index', str(index)], cwd=frozen.ROOT, stdout=log, stderr=subprocess.STDOUT))
        while True:
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Worker failed; preserve claimed outputs without retry')
            for path in sorted((work / 'items').glob('*.json')):
                if path.stem in seen:
                    continue
                if path.stem not in expected or path.stem in present:
                    raise ValueError('Duplicate or unexpected output')
                job, report = expected[path.stem], read(path)
                prior.verify(report, plan, job['case'], 'candidate', 1., job['seed'])
                row, = report['items']
                rows.append({**row, **report['level_metrics']})
                seen.add(path.stem)
                present.add(path.stem)
                if reason is None:
                    reason, acoustics = decision(rows, base, diagnostic=diagnostic,
                                                paired_extremes=paired_extremes)
            if reason is None and stage < plan['stages'] and all(
                    j['id'] in present for j in jobs if j['stage'] <= stage):
                reason, acoustics = decision(rows, base, stage_end=True, diagnostic=diagnostic,
                                            paired_extremes=paired_extremes)
                stages.append({'stage':stage, 'completed':len(rows), 'reason':reason,
                    'acoustic_failures':acoustics, 'cer_observations':cer_observations(rows, base)})
                stage += 1
            done = reason is not None or stage == plan['stages']
            update(work / 'control.json', {'stage':stage, 'stop':done, 'reason':reason})
            update(work / 'status.json', {'stage':'draining' if done else 'evaluating',
                'completed':len(rows), 'new_completed':len(seen), 'current_stage':stage,
                'reason':reason, 'pids':[p.pid for p in processes if p.poll() is None], 'time':time.time()})
            if all(p.poll() is not None for p in processes):
                if not done:
                    raise RuntimeError('Workers exited before a terminal decision')
                # Workers publish before exit; drain their final outputs on the next loop.
                if {p.stem for p in (work / 'items').glob('*.json')} <= seen:
                    break
            time.sleep(1)
        if len(rows) != 300 and reason is None:
            raise ValueError('Incomplete successful expansion')
        prior.strength.check_inputs(work)
        claims = {p.stem for p in (work / 'claims').glob('*.json')}
        if claims != seen:
            raise ValueError('Claimed output without a scored result')
        wavs = {str(p) for p in work.glob('worker*/auditions/*.wav')}
        new_rows = [r for r in rows if f"{r['case_id']}_{r['seed']}" in seen]
        if wavs != {r['wav'] for r in new_rows}:
            raise ValueError('Unscored or unmapped generated WAV')
        base_rows = [base[prior.key(r)] for r in rows]
        metrics = {name:frozen.evaluation._report_metrics(items) for name, items in
                   (('base', base_rows), ('candidate', rows))}
        comparison = None
        if (work / 'equal_exposure_r33.json').exists():
            old = {prior.key(r):r for r in read(work / 'equal_exposure_r33.json')}
            current = [r for r in rows if prior.key(r) in old]
            comparison = {'optimizer_updates':plan['checkpoint_optimizer_updates'], 'paired_outputs':len(current),
                'metrics':{name:frozen.evaluation._report_metrics(items) for name, items in
                    (('r33_lr1e4', [old[prior.key(r)] for r in current]), ('r34_lr5e5', current))},
                'note':'Same exposure and inference inputs; one training seed, development only.'}
        final_decision = terminal_decision(reason, diagnostic)
        save(work / 'result.json', {'decision':final_decision, 'diagnostic_only':diagnostic,
            'reason':reason, 'completed':len(rows), 'new_outputs':len(seen), 'skipped':300-len(rows),
            'stages':stages, 'metrics':metrics, 'rows':rows, 'matched_base':base_rows,
            'cer_observations':cer_observations(rows, base), 'equal_exposure_comparison':comparison,
            'native_quality':'unverified', 'production_changed':False,
            'wav_sha256':{r['wav']:frozen.sha256(r['wav']) for r in rows}})
        update(work / 'status.json', {'stage':final_decision if diagnostic else ('rejected' if reason else 'development_complete'),
            'completed':len(rows), 'new_completed':len(seen), 'reason':reason, 'pids':[], 'time':time.time()})
    except BaseException as error:
        update(work / 'control.json', {'stage':stage, 'stop':True, 'reason':str(error)})
        update(work / 'status.json', {'stage':'failed', 'error':str(error), 'time':time.time()})
        raise
    finally:
        for process in processes:
            prior.stop(process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=range(4))
    parser.add_argument('--profile', choices=('r33', 'r34_diagnostic', 'r33_early_diagnostic',
                                             'r39_early_diagnostic'), default='r33')
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.work, args.index)
    elif args.action == 'prepare':
        prepare(args.work, args.profile)
    else:
        globals()[args.action](args.work)
