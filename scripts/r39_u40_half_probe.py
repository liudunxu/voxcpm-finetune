"""Single inference-only u40 half-strength diagnostic; no new weights or training."""
import argparse
from pathlib import Path
import shutil

import r39_midpoint_probe as prior

frozen, read, save = prior.frozen, prior.read, prior.save
RISK_GROUPS = (
    ('fresh_tl_1855', 'plain', (43, 42, 44, 45, 46)),
    ('fresh_th_1689', 'control', (45, 42, 43, 44, 46)),
    ('fresh_id_1973', 'plain', (45, 46, 42, 43, 44)),
)


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent, result, verified = prior.checked_parent(
        prior.U40, prior.U40_PLAN_SHA, prior.U40_VERIFICATION_SHA,
        reason='new_suspected_overread', completed=150, updates=40)
    cases = list({job['case']['case_id']: job['case'] for job in read(prior.U40 / 'run/jobs.json')
                  if job['case']['sentence_id'] in prior.SENTENCES}.values())
    jobs = prior.midpoint_jobs(cases, risk_groups=RISK_GROUPS, prefix='half', strength=.5)
    expected = {(job['case']['case_id'], job['seed']) for job in jobs}

    def endpoints(field):
        rows = [row for row in result[field] if prior.probe.key(row) in expected]
        if len(rows) != 30 or {prior.probe.key(row) for row in rows} != expected:
            raise ValueError('Incomplete five-seed endpoint coverage')
        return rows

    reused = {'base': endpoints('matched_base'), 'u40': endpoints('rows')}
    inputs = dict(parent['inputs_sha256'])
    inputs.update(verified['outputs_sha256'])
    for path in (prior.U40 / 'run/plan.json', prior.U40 / 'verification_terminal.json'):
        inputs[str(path)] = frozen.sha256(path)
    reserved = frozen.ROOT / 'eval_cases/reserved_final_20261030.jsonl'
    prior.review.checked(reserved, 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c')
    heldout = {frozen.evaluation._norm(row['text']) for row in frozen._read_manifest(reserved)}
    if any(frozen.evaluation._norm(job['case']['text']) in heldout for job in jobs):
        raise ValueError('Reserved final text in half-strength development')
    inputs[str(reserved)] = frozen.sha256(reserved)
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src'):
        for path in (frozen.ROOT / folder).rglob('*'):
            if path.suffix in ('.py', '.sh'):
                inputs[str(path)] = frozen.sha256(path)
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    if shutil.disk_usage(work.parent).free < 4.5 * 1024**3:
        raise RuntimeError('Need 4.5 GiB before half-strength probe')
    work.mkdir()
    for name, value in (('jobs.json', jobs), ('cases.json', cases), ('reused.json', reused)):
        save(work / name, value)
        inputs[str(work / name)] = frozen.sha256(work / name)
    save(work / 'plan.json', {
        **{key: parent[key] for key in ('base_path', 'asr_path', 'speaker_model_path')},
        'targets': {'candidate': parent['targets']['candidate'], 'base': 'base'}, 'inputs_sha256': inputs,
        'profile': 'r39_u40_half', 'output_view': 'half_u40', 'lora_strength': .5,
        'new_outputs': 30, 'reused_outputs': 60, 'training_steps': 0, 'max_workers': 1,
        'seeds': frozen.SEEDS, 'tail_policy': 'absolute', 'paired_extreme_acoustics': True,
        'checkpoint_optimizer_updates': 40, 'parent_plan_sha256': prior.U40_PLAN_SHA,
        'parent_verification_sha256': prior.U40_VERIFICATION_SHA,
        'native_quality': 'unverified', 'production_changed': False, 'final_holdout_used': False,
        'early_rejection': 'Per-output severe gates against matched base; complete five-seed condition '
                           'groups use absolute and paired-extreme acoustic gates. Mean CER observation only.',
        'note': 'Single inference-only u40 strength 0.5 diagnostic, not a grid or new checkpoint. '
                'TL1855 and TH1689 failures are tested before ID1973. All 60 base/u40 endpoints are reused. '
                'No weight edit, training, retry, reserved final text, production change or acceptance claim.'})
    print('PREPARED u40 half strength: 60 reused endpoints, at most 30 new', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.work)
    else:
        prior.run(args.work)
