"""Bounded 40/80 LoRA midpoint diagnosis with frozen endpoints and no training."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import r31_failure_probe as probe
import staged_candidate_run as staged

frozen, fast, review = probe.frozen, probe.fast, probe.review
read, save = probe.read, probe.save
U40 = Path('/root/autodl-tmp/r39_u40_20261031')
U80 = Path('/root/autodl-tmp/r39_early_20261031')
U40_PLAN_SHA = 'a392459546080476c7b4af734976e7f588c2927dddcdcec436c9911a8f2e6931'
U80_PLAN_SHA = 'f7e5ca1526e5e79e648f73292f0d61b33b8151e5a1c81a49ce6fb249675cdbef'
U40_VERIFICATION_SHA = 'ffa461073ce47bd5c9ce8c7c61553dad591b478da341ba779bec05f0b8bf56ca'
U80_VERIFICATION_SHA = '048735b2b6fdaeff2a6ea575ca05971cc4204bd20ba782a91fb5bec6699f7588'
SENTENCES = ('fresh_id_1973', 'fresh_th_1689', 'fresh_tl_1855')
RISK_GROUPS = (
    ('fresh_id_1973', 'plain', (45, 46, 42, 43, 44)),
    ('fresh_th_1689', 'control', (45, 42, 43, 44, 46)),
    ('fresh_tl_1855', 'plain', (43, 42, 44, 45, 46)),
)


def midpoint_jobs(cases):
    lookup = {(case['sentence_id'], case['condition']): case for case in cases}
    if len(lookup) != 6 or set(lookup) != {(sid, condition) for sid in SENTENCES for condition in ('plain', 'control')}:
        raise ValueError('Expected three paired development texts')
    groups = list(RISK_GROUPS)
    groups += [(sid, condition, frozen.SEEDS) for sid, condition in sorted(lookup)
               if (sid, condition) not in {(group[0], group[1]) for group in RISK_GROUPS}]
    jobs = []
    for sid, condition, seeds in groups:
        case = lookup[sid, condition]
        for seed in seeds:
            jobs.append({'id': f'mid_{case["case_id"]}_{seed}', 'case': case,
                         'model': 'candidate', 'strength': 1., 'seed': seed})
    if len(jobs) != 30 or len({job['id'] for job in jobs}) != 30:
        raise ValueError('Unexpected midpoint coverage')
    return jobs


def write_midpoint(u40_directory, u80_directory, destination):
    from safetensors.torch import load_file, save_file

    if destination.exists():
        raise FileExistsError(destination)
    source_config = (u40_directory / 'lora_config.json').read_bytes()
    if source_config != (u80_directory / 'lora_config.json').read_bytes():
        raise ValueError('Midpoint requires identical LoRA configuration bytes')
    left = load_file(str(u40_directory / 'lora_weights.safetensors'), device='cpu')
    right = load_file(str(u80_directory / 'lora_weights.safetensors'), device='cpu')
    if len(left) != 96 or left.keys() != right.keys():
        raise ValueError('Expected the same 96 LoRA tensors')
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            raise ValueError(f'Midpoint tensor mismatch: {name}')
    merged = {name: (left[name].float() + right[name].float()).div(2) for name in left}
    destination.mkdir()
    save_file(merged, str(destination / 'lora_weights.safetensors'), metadata={'format': 'pt'})
    (destination / 'lora_config.json').write_bytes(source_config)
    return {'formula': '0.5 * u40 + 0.5 * u80 in float32', 'tensors': len(merged),
            'u40_weights_sha256': hashlib.sha256((u40_directory / 'lora_weights.safetensors').read_bytes()).hexdigest(),
            'u80_weights_sha256': hashlib.sha256((u80_directory / 'lora_weights.safetensors').read_bytes()).hexdigest(),
            'weights_sha256': hashlib.sha256((destination / 'lora_weights.safetensors').read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256(source_config).hexdigest(),
            'actual_optimizer_checkpoint': False}


def checked_parent(root, plan_sha, verification_sha, *, reason, completed, updates):
    review.checked(root / 'run/plan.json', plan_sha)
    review.checked(root / 'verification_terminal.json', verification_sha)
    verified = read(root / 'verification_terminal.json')
    if (verified['status'] != 'diagnostic_stopped_verified' or verified['reason'] != reason
            or verified['completed_outputs'] != completed or verified['optimizer_updates'] != updates
            or verified['unscored_wavs'] != 0 or verified['plan_sha256'] != plan_sha):
        raise ValueError('Unexpected parent diagnostic state')
    for path, digest in verified['outputs_sha256'].items():
        review.checked(path, digest)
    return strength_plan(root), read(root / 'run/result.json'), verified


def strength_plan(root):
    return probe.strength.check_inputs(root / 'run')


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    u40_plan, u40_result, u40_verified = checked_parent(
        U40, U40_PLAN_SHA, U40_VERIFICATION_SHA,
        reason='new_suspected_overread', completed=150, updates=40)
    u80_plan, u80_result, u80_verified = checked_parent(
        U80, U80_PLAN_SHA, U80_VERIFICATION_SHA,
        reason='two_large_paired_cer_regressions', completed=197, updates=80)
    if any(u40_plan[key] != u80_plan[key] for key in ('base_path', 'asr_path', 'speaker_model_path')):
        raise ValueError('Midpoint parents use different model or ASR inputs')
    cases = list({job['case']['case_id']: job['case'] for job in read(U40 / 'run/jobs.json')
                  if job['case']['sentence_id'] in SENTENCES}.values())
    jobs = midpoint_jobs(cases)
    expected = {(job['case']['case_id'], job['seed']) for job in jobs}

    def endpoints(result, field):
        rows = [row for row in result[field] if probe.key(row) in expected]
        if len(rows) != 30 or {probe.key(row) for row in rows} != expected:
            raise ValueError('Incomplete five-seed endpoint coverage')
        return rows

    reused = {'base': endpoints(u40_result, 'matched_base'),
              'u40': endpoints(u40_result, 'rows'), 'u80': endpoints(u80_result, 'rows')}
    other_base = {probe.key(row): row for row in endpoints(u80_result, 'matched_base')}
    if {probe.key(row): row for row in reused['base']} != other_base:
        raise ValueError('Parent baselines differ')
    inputs = {}

    def include(mapping):
        for path, digest in mapping.items():
            if path in inputs and inputs[path] != digest:
                raise ValueError(f'Conflicting input hash: {path}')
            inputs[path] = digest

    for plan in (u40_plan, u80_plan):
        include(plan['inputs_sha256'])
    include(u40_verified['outputs_sha256'])
    include(u80_verified['outputs_sha256'])
    for path in (U40 / 'run/plan.json', U40 / 'verification_terminal.json',
                 U80 / 'run/plan.json', U80 / 'verification_terminal.json'):
        inputs[str(path)] = frozen.sha256(path)
    reserved = frozen.ROOT / 'eval_cases/reserved_final_20261030.jsonl'
    review.checked(reserved, 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c')
    heldout = {frozen.evaluation._norm(row['text']) for row in frozen._read_manifest(reserved)}
    if any(frozen.evaluation._norm(job['case']['text']) in heldout for job in jobs):
        raise ValueError('Reserved final text in midpoint development')
    inputs[str(reserved)] = frozen.sha256(reserved)
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src'):
        for path in (frozen.ROOT / folder).rglob('*'):
            if path.suffix in ('.py', '.sh'):
                inputs[str(path)] = frozen.sha256(path)
    for path, digest in inputs.items():
        review.checked(path, digest)
    if shutil.disk_usage(work.parent).free < 4.5 * 1024**3:
        raise RuntimeError('Need 4.5 GiB before midpoint probe')
    work.mkdir()
    for name, value in (('jobs.json', jobs), ('cases.json', cases), ('reused.json', reused)):
        save(work / name, value)
        inputs[str(work / name)] = frozen.sha256(work / name)
    interpolation = write_midpoint(Path(u40_plan['targets']['candidate']),
                                   Path(u80_plan['targets']['candidate']), work / 'model')
    interpolation.update(u40_optimizer_updates=40, u80_optimizer_updates=80)
    save(work / 'interpolation.json', interpolation)
    for path in (work / 'model/lora_weights.safetensors', work / 'model/lora_config.json',
                 work / 'interpolation.json'):
        inputs[str(path)] = frozen.sha256(path)
    save(work / 'plan.json', {
        **{key: u40_plan[key] for key in ('base_path', 'asr_path', 'speaker_model_path')},
        'targets': {'candidate': str(work / 'model'), 'base': 'base'}, 'inputs_sha256': inputs,
        'profile': 'r39_midpoint', 'new_outputs': 30, 'reused_outputs': 90,
        'training_steps': 0, 'max_workers': 1, 'seeds': frozen.SEEDS, 'tail_policy': 'absolute',
        'paired_extreme_acoustics': True, 'native_quality': 'unverified',
        'production_changed': False, 'final_holdout_used': False, 'interpolation': interpolation,
        'early_rejection': 'Per-output severe gates against matched base; complete five-seed condition '
                           'groups use absolute and paired-extreme acoustic gates. Mean CER observation only.',
        'note': 'Outcome-selected three-text midpoint diagnosis. All 90 base/u40/u80 endpoints are reused. '
                'The arithmetic weight midpoint is not an actual 60-update checkpoint. No training, retries, '
                'reserved final texts, production change or three-language acceptance claim.'})
    print('PREPARED r39 midpoint: 90 reused endpoints, at most 30 new', flush=True)


def midpoint_rejection(rows, base, *, stage_end=False):
    reason = staged.probe.reject_reason(rows, base)
    if reason:
        return reason, {}
    if stage_end:
        latest = rows[-1]
        group = [row for row in rows if row['case_id'] == latest['case_id']]
        if len(group) != 5 or {row['seed'] for row in group} != set(frozen.SEEDS):
            raise ValueError('Incomplete midpoint condition group')
        acoustic = staged.acoustic_gate(rows, base, paired_extremes=True)
        if acoustic:
            return 'stage_acoustic_regression', acoustic
    return None, {}


def run(work):
    plan = probe.strength.check_inputs(work)
    if (plan['profile'] != 'r39_midpoint' or plan['new_outputs'] != 30
            or plan['reused_outputs'] != 90 or plan['training_steps'] != 0 or plan['max_workers'] != 1):
        raise ValueError('Midpoint scope changed')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    with (work / 'run.started').open('x', encoding='utf-8') as stream:
        stream.write(frozen.sha256(work / 'plan.json'))
    (work / 'items').mkdir()
    frozen.runtime_environment(work / 'runtime')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VOXFT_WHISPER_MODEL_LARGE=plan['asr_path'], VOXFT_SPK_EMB_MODEL=plan['speaker_model_path'],
        TORCHINDUCTOR_CACHE_DIR=str(work / 'runtime/inductor'))
    frozen.evaluation.EVAL_DIR = work / 'raw_reports'
    frozen.evaluation.infer.CHECKPOINT_DIR = work
    jobs, reused = read(work / 'jobs.json'), read(work / 'reused.json')
    base = {probe.key(row): row for row in reused['base']}
    rows, stages, reason, acoustic = [], [], None, {}

    def status(stage, **extra):
        frozen.write_json(work / 'status.json', {'stage': stage, 'completed': len(rows), 'expected': 30,
            'pid': os.getpid(), 'time': time.time(), 'reason': reason, **extra})

    try:
        status('starting')
        with fast.cached_services(read(work / 'cases.json')):
            save(work / 'runtime.json', probe.baseline.gpu_services(
                plan, [job['case'] for job in jobs], 'candidate'))
            for index, job in enumerate(jobs):
                if shutil.disk_usage(work).free < 4 * 1024**3:
                    raise RuntimeError('Disk reserve reached')
                report = frozen.evaluation.evaluate(plan['targets']['candidate'], 'id', [job['case']],
                    base=plan['base_path'], seeds=[job['seed']], cfg_value=1.8, inference_timesteps=20,
                    lora_strength=job['strength'])
                row, = report['items']
                report['wav_sha256'] = {row['wav']: frozen.sha256(row['wav'])}
                report['level_metrics'] = review.level_metrics(row['wav'])
                probe.verify(report, plan, job['case'], 'candidate', job['strength'], job['seed'])
                save(work / 'items' / (job['id'] + '.json'), report)
                rows.append({**row, **report['level_metrics']})
                reason, acoustic = midpoint_rejection(rows, base, stage_end=(index + 1) % 5 == 0)
                if (index + 1) % 5 == 0:
                    stages.append({'stage': len(stages), 'completed': len(rows), 'reason': reason,
                                   'acoustic_failures': acoustic})
                status('diagnostic_stopped' if reason else 'evaluating', last_job=job['id'])
                print('DONE', job['id'], reason, flush=True)
                if reason:
                    break
        probe.strength.check_inputs(work)
        if {str(path) for path in work.glob('auditions/*.wav')} != {row['wav'] for row in rows}:
            raise ValueError('Unscored midpoint output')
        decision = 'diagnostic_stopped' if reason else 'diagnostic_complete'
        current = [base[probe.key(row)] for row in rows]
        metrics = {'base': frozen.evaluation._report_metrics(current),
                   'midpoint': frozen.evaluation._report_metrics(rows)}
        for name in ('u40', 'u80'):
            lookup = {probe.key(row): row for row in reused[name]}
            metrics[name] = frozen.evaluation._report_metrics([lookup[probe.key(row)] for row in rows])
        save(work / 'result.json', {'decision': decision, 'reason': reason, 'acoustic_failures': acoustic,
            'completed': len(rows), 'skipped': 30 - len(rows), 'rows': rows, 'matched_base': current,
            'reused': reused, 'stages': stages, 'metrics': metrics,
            'wav_sha256': {row['wav']: frozen.sha256(row['wav']) for row in rows},
            'interpolation': plan['interpolation'], 'training_steps': 0, 'production_changed': False,
            'native_quality': 'unverified', 'final_holdout_used': False})
        status(decision)
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.work)
