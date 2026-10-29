"""r29: change only r28's data exclusions, then run the same 459 steps and evaluation."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

import clean_candidate_run as runner
from control_ab_gate import verify_rng
from fresh_baseline_eval import save_new
from voxft.data.pipeline import _read_manifest, dataset_summary
from voxft.data.registry import get_source
from voxft.train.launcher import preflight

DATA = Path('/root/autodl-tmp')
PARENT = DATA / 'clean_candidate_20261028/run'
CANDIDATE = DATA / 'training_silence_20261029/candidate_all_flags'
RUN_NAME = 'lora_joint3_r29_suspect_excluded_459'
TRAIN_SHA = '27ed1d5ff21d8c31067b09429d2c9ba78b7fa71556b2d23b5ec540c52196e0a5'
R28_REPORT_SHA = 'e1839c8910e4a4dfcec561cb205f00e0e7732d0e6b7ce527038f95a4b1763cbe'


def fixed_config(template, work):
    config = yaml.safe_load(Path(template).read_text(encoding='utf-8'))
    if (config['num_iters'] != 459 or config['max_steps'] != 459 or config['warmup_steps'] != 45
            or config['batch_size'] != 2 or config['grad_accum_steps'] != 8):
        raise ValueError('r28 fixed-step recipe changed')
    return {**config, 'train_manifest': str(work / 'train.jsonl'),
            'val_manifest': str(work / 'val.jsonl'), 'save_path': str(work / 'checkpoint'),
            'tensorboard': str(work / 'checkpoint/logs')}


def verify_subset(parent, candidate):
    lines = [line for line in (parent / 'train.jsonl').read_bytes().splitlines(keepends=True) if line.strip()]
    removed = _read_manifest(candidate / 'excluded.jsonl')
    positions = {row['row'] for row in removed}
    if (len(positions) != len(removed) or any(not 1 <= pos <= len(lines) for pos in positions)
            or any(json.loads(lines[row['row'] - 1]) != row['sample'] for row in removed)):
        raise ValueError('Exclusion record differs from parent')
    retained = b''.join(line for position, line in enumerate(lines, 1) if position not in positions)
    if ((candidate / 'train.jsonl').read_bytes() != retained
            or (candidate / 'val.jsonl').read_bytes() != (parent / 'val.jsonl').read_bytes()):
        raise ValueError('Candidate changes more than declared exclusions')
    return len(removed)


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent = runner.checked(PARENT, trained=True)
    mix = runner.read_json(CANDIDATE / 'mix.json')
    if (mix['status'] != 'candidate_quarantine_complete' or mix['excluded_rows'] != 81
            or mix['train']['rows'] != 7252 or mix['checks']['violations']
            or mix['checks']['origins_above_3x'] or any(mix['pcm'].values())):
        raise ValueError('Candidate quarantine gate changed')
    for name, digest in mix['outputs_sha256'].items():
        if runner.frozen.sha256(CANDIDATE / name) != digest:
            raise ValueError(f'Candidate changed: {name}')
    if (runner.frozen.sha256(CANDIDATE / 'train.jsonl') != TRAIN_SHA
            or runner.frozen.sha256(PARENT / 'r28_report.json') != R28_REPORT_SHA
            or verify_subset(PARENT, CANDIDATE) != 81):
        raise ValueError('Frozen data or r28 result changed')
    candidate_plan = runner.read_json(CANDIDATE / 'plan.json')
    inputs = {**parent['inputs_sha256'], **candidate_plan['inputs_sha256']}
    trained = runner.read_json(PARENT / 'trained.json')
    inputs.update(trained['inputs_sha256'])
    for path in (PARENT / 'plan.json', PARENT / 'trained.json', PARENT / 'r28_report.json',
                 CANDIDATE / 'mix.json', *[CANDIDATE / name for name in mix['outputs_sha256']]):
        inputs[str(path)] = runner.frozen.sha256(path)
    for root in (runner.frozen.ROOT / 'src', runner.frozen.ROOT / 'scripts',
                 runner.frozen.VOXCPM_REPO / 'src', runner.frozen.VOXCPM_REPO / 'scripts'):
        for pattern in ('*.py', '*.sh'):
            inputs.update({str(path): runner.frozen.sha256(path) for path in root.rglob(pattern)})
    for path, digest in inputs.items():
        if runner.frozen.sha256(path) != digest:
            raise ValueError(f'Frozen input changed: {path}')
    if shutil.disk_usage(work.parent).free < 8 * 1024**3:
        raise RuntimeError('Need 8 GiB for checkpoints, evaluation and the existing 4 GiB reserve')
    work.mkdir(parents=True, exist_ok=False)
    for name in ('train.jsonl', 'val.jsonl', 'excluded.jsonl'):
        shutil.copyfile(CANDIDATE / name, work / name)
    for name in ('cases.jsonl', 'base_report.json', 'r26_report.json', 'r28_report.json'):
        shutil.copyfile(PARENT / name, work / name)
    config = fixed_config(PARENT / 'train.yaml', work)
    config_path = work / 'train.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    runner.frozen.require_fresh_checkpoint(config)
    issues = preflight(config_path, gpus=1)
    Path(config['save_path']).rmdir()
    if any(not issue.startswith('警告') for issue in issues):
        raise ValueError(f'Training preflight failed: {issues}')
    save_new(work / 'mix.json', {**{key: value for key, value in mix.items() if key != 'outputs_sha256'},
        'source_candidate': str(CANDIDATE), 'source_outputs_sha256': mix['outputs_sha256'],
        'training_approved': True,
        'authorization': 'User: 接下来可以启动训练了吧', 'approval_scope': 'r29 fixed459 data-only experiment'})
    for path in work.iterdir():
        if path.is_file():
            inputs[str(path)] = runner.frozen.sha256(path)
    train = _read_manifest(work / 'train.jsonl')
    sources = {source_id: {'license': get_source(source_id).license,
                           **dataset_summary([row for row in train if row['source_id'] == source_id])}
               for source_id in sorted({row['source_id'] for row in train})}
    if any(source['license'] not in {'CC0', 'CC-BY-3.0', 'CC-BY-4.0', 'Apache-2.0'}
           for source in sources.values()):
        raise ValueError('Source outside existing license allowlist')
    plan = {
        **{key: parent[key] for key in ('evaluation_seeds', 'cfg_value', 'inference_timesteps',
           'retry_badcase', 'asr_model', 'base_path', 'asr_path', 'speaker_model_path')},
        'targets': {**parent['targets'], 'r29': str(work / 'checkpoint/latest')},
        'candidate_model': 'r29', 'run_name': RUN_NAME, 'config': str(config_path),
        'num_iters': 459, 'epochs': 459 * 16 / len(train), 'training_seed': 42,
        'rng_verification': verify_rng(), 'training_authorized': True,
        'authorization': 'User: 接下来可以启动训练了吧',
        'max_workers': 4, 'generation_budget': 900, 'production_changed': False,
        'additional_comparisons': [['r29', 'r28']], 'inputs_sha256': inputs,
        'train': dataset_summary(train), 'language_shares': mix['language_shares'],
        'validation': dataset_summary(_read_manifest(work / 'val.jsonl')), 'sources': sources,
        'audit': mix['checks'], 'pcm': mix['pcm'], 'preflight': issues,
        'excluded_rows_from_r28': 81, 'cumulative_excluded_from_original': 160,
        'note': 'Cold-start data-exclusion comparison versus r28: same initial base, seed, '
                '459 steps, batch/accumulation, learning rate/schedule, controls and projection. '
                'Effective epochs are approximately 1.013 on the reduced dataset. '
                'Reuse frozen base/r26/r28 reports; no regeneration, retry, promotion or downloads. '
                'Single training seed; native and acoustic acceptance remain unverified.',
    }
    save_new(work / 'plan.json', plan)
    runner.checked(work)
    print(json.dumps({'run_name': RUN_NAME, 'rows': len(train), 'steps': 459,
                      'effective_epochs': plan['epochs'], 'preflight': issues}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': runner.run}[args.action](args.work.resolve())
