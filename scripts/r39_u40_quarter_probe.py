"""Bounded u40 quarter-strength diagnosis after the verified half-strength failure."""
import argparse
from pathlib import Path
import shutil

import r39_midpoint_probe as prior

frozen, read, save = prior.frozen, prior.read, prior.save
EXPANSION = Path('/root/autodl-tmp/r39_u40_half_expansion_20261031')
EXPANSION_PLAN_SHA = '25ac2ecc30c8e1fb1cdb3045cbe3c5c933ec2ea98bb025a767c73e7027ecbb99'
EXPANSION_VERIFICATION_SHA = '740743f1f7f30d0c46ca5a55b7d4cf9ddb80cca20463f9847767592691fb6a14'
RISK_GROUPS = (
    ('fresh_id_1816', 'control', (45, 42, 43, 44, 46)),
    ('fresh_tl_1855', 'plain', (43, 42, 44, 45, 46)),
    ('fresh_th_1689', 'control', (45, 42, 43, 44, 46)),
    ('fresh_id_1973', 'plain', (45, 46, 42, 43, 44)),
)


def quarter_jobs(cases):
    lookup = {(case['sentence_id'], case['condition']): case for case in cases}
    expected = {(sid, condition) for sid, condition, _seeds in RISK_GROUPS}
    if len(cases) != 4 or set(lookup) != expected:
        raise ValueError('Expected exactly four selected risk conditions')
    jobs = []
    for sid, condition, seeds in RISK_GROUPS:
        case = lookup[sid, condition]
        for seed in seeds:
            jobs.append({'id': f'quarter_{case["case_id"]}_{seed}', 'case': case,
                         'model': 'candidate', 'strength': .25, 'seed': seed})
    if len(jobs) != 20 or len({job['id'] for job in jobs}) != 20:
        raise ValueError('Unexpected quarter-strength coverage')
    return jobs


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    u40_plan, u40_result, u40_verified = prior.checked_parent(
        prior.U40, prior.U40_PLAN_SHA, prior.U40_VERIFICATION_SHA,
        reason='new_suspected_overread', completed=150, updates=40)
    prior.review.checked(EXPANSION / 'run/plan.json', EXPANSION_PLAN_SHA)
    prior.review.checked(EXPANSION / 'verification_terminal.json', EXPANSION_VERIFICATION_SHA)
    expansion_verified = read(EXPANSION / 'verification_terminal.json')
    if (expansion_verified['status'] != 'diagnostic_stopped_verified'
            or expansion_verified['reason'] != 'new_suspected_overread'
            or expansion_verified['completed_outputs'] != 77
            or expansion_verified['source_optimizer_updates'] != 40
            or expansion_verified['unscored_wavs'] != 0
            or expansion_verified['plan_sha256'] != EXPANSION_PLAN_SHA
            or expansion_verified['lora_strength'] != .5):
        raise ValueError('Unexpected half-strength expansion state')
    for path, digest in expansion_verified['outputs_sha256'].items():
        prior.review.checked(path, digest)
    expansion_plan = prior.strength_plan(EXPANSION)
    expansion_result = read(EXPANSION / 'run/result.json')
    if any(u40_plan[key] != expansion_plan[key]
           for key in ('base_path', 'asr_path', 'speaker_model_path', 'targets')):
        raise ValueError('Quarter-strength parents use different model inputs')
    selected = {(sid, condition) for sid, condition, _seeds in RISK_GROUPS}
    cases = list({job['case']['case_id']: job['case'] for job in read(prior.U40 / 'run/jobs.json')
                  if (job['case']['sentence_id'], job['case']['condition']) in selected}.values())
    jobs = quarter_jobs(cases)
    expected = {(job['case']['case_id'], job['seed']) for job in jobs}

    def endpoints(result, field):
        rows = [row for row in result[field] if prior.probe.key(row) in expected]
        if len(rows) != 20 or {prior.probe.key(row) for row in rows} != expected:
            raise ValueError('Incomplete five-seed endpoint coverage')
        return rows

    reused = {'base': endpoints(expansion_result, 'matched_base'),
              'u40': endpoints(u40_result, 'rows'),
              'half_u40': endpoints(expansion_result, 'rows')}
    other_base = {prior.probe.key(row): row for row in endpoints(u40_result, 'matched_base')}
    if {prior.probe.key(row): row for row in reused['base']} != other_base:
        raise ValueError('Parent baselines differ')
    inputs = {}

    def include(mapping):
        for path, digest in mapping.items():
            if path in inputs and inputs[path] != digest:
                raise ValueError(f'Conflicting input hash: {path}')
            inputs[path] = digest

    for plan in (u40_plan, expansion_plan):
        include(plan['inputs_sha256'])
    include(u40_verified['outputs_sha256'])
    include(expansion_verified['outputs_sha256'])
    for path in (prior.U40 / 'run/plan.json', prior.U40 / 'verification_terminal.json',
                 EXPANSION / 'run/plan.json', EXPANSION / 'verification_terminal.json'):
        inputs[str(path)] = frozen.sha256(path)
    reserved = frozen.ROOT / 'eval_cases/reserved_final_20261030.jsonl'
    prior.review.checked(reserved, 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c')
    heldout = {frozen.evaluation._norm(row['text']) for row in frozen._read_manifest(reserved)}
    if any(frozen.evaluation._norm(job['case']['text']) in heldout for job in jobs):
        raise ValueError('Reserved final text in quarter-strength development')
    inputs[str(reserved)] = frozen.sha256(reserved)
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src'):
        for path in (frozen.ROOT / folder).rglob('*'):
            if path.suffix in ('.py', '.sh'):
                inputs[str(path)] = frozen.sha256(path)
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    if shutil.disk_usage(work.parent).free < 4.5 * 1024**3:
        raise RuntimeError('Need 4.5 GiB before quarter-strength probe')
    work.mkdir()
    for name, value in (('jobs.json', jobs), ('cases.json', cases), ('reused.json', reused)):
        save(work / name, value)
        inputs[str(work / name)] = frozen.sha256(work / name)
    save(work / 'plan.json', {
        **{key: u40_plan[key] for key in ('base_path', 'asr_path', 'speaker_model_path')},
        'targets': {'candidate': u40_plan['targets']['candidate'], 'base': 'base'},
        'inputs_sha256': inputs, 'profile': 'r39_u40_quarter', 'output_view': 'quarter_u40',
        'lora_strength': .25, 'new_outputs': 20, 'reused_outputs': 60, 'training_steps': 0,
        'max_workers': 1, 'seeds': frozen.SEEDS, 'tail_policy': 'absolute',
        'paired_extreme_acoustics': True, 'checkpoint_optimizer_updates': 40,
        'parent_plan_sha256': EXPANSION_PLAN_SHA,
        'parent_verification_sha256': EXPANSION_VERIFICATION_SHA,
        'native_quality': 'unverified', 'production_changed': False, 'final_holdout_used': False,
        'early_rejection': 'Per-output severe gates against matched base; complete five-seed condition '
                           'groups use absolute and paired-extreme acoustic gates. Mean CER observation only.',
        'note': 'Single inference-only u40 strength 0.25 diagnostic after verified 0.5 failures. '
                'ID1816 control is first, followed by the three earlier complementary risks. '
                'All 60 base/u40/half endpoints are reused. No weight edit, training, retry, '
                'reserved final text, production change or acceptance claim.'})
    print('PREPARED u40 quarter strength: 60 reused endpoints, at most 20 new', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.work)
    else:
        prior.run(args.work)
