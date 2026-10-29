"""r30: train the completed full-cleaning subset with the frozen r29 schedule."""
import argparse
import json
from pathlib import Path
import shutil

import yaml

import clean_candidate_run as runner
from control_ab_gate import pcm_conflicts, verify_rng
from fresh_baseline_eval import save_new
from incremental_mix import audit_mix, describe_mix, load_exclusions
from quarantine_music_candidates import verify
from review_training_silence import exclude_flagged
from suspect_candidate_run import fixed_config
from voxft.data.pipeline import _read_manifest, dataset_summary
from voxft.data.registry import get_source
from voxft.train.launcher import preflight

DATA = Path('/root/autodl-tmp')
PARENT = DATA / 'suspect_candidate_20261029/run'
CANDIDATE = DATA / 'music_excluded_20261029/run'
CLEAN = DATA / 'full_corpus_clean_20261029/parallel8/run'
RUN_NAME = 'lora_joint3_r30_full_clean_459'
FROZEN = {
    'summary.json': '19468c5170b8e9e3acae90edd7bc2d2990ff3c8bb929b91d197e405d0ef95465',
    'candidate/current/train.jsonl': 'fa069dc39abb1edd71c677511e2e3722a36e143b77e318f7fe93bc8abfb9aa08',
    'candidate/current/val.jsonl': 'a76c87eea41b4a4913a32c439386f84762563b4a3e533004da14dc342038eef3',
}
R29_REPORT_SHA = '90cbcfc4f571ea5d3379266e4b5a914860e54c93c211fd8e4a9a552e7d855af4'


def candidate_inputs():
    inputs = {str(CANDIDATE / name): digest for name, digest in FROZEN.items()}
    verify(inputs)
    summary = runner.read_json(CANDIDATE / 'summary.json')
    if (summary['status'] != 'complete' or summary['selected_files'] != 62
            or runner.read_json(CLEAN / 'status.json')['stage'] != 'complete'
            or runner.read_json(CANDIDATE / 'status.json')['stage'] != 'complete'):
        raise ValueError('Completed full cleaning and music exclusion required')
    for name in ('plan.json', 'flagged.json'):
        inputs[str(CANDIDATE / name)] = runner.frozen.sha256(CANDIDATE / name)
    inputs.update(runner.read_json(CANDIDATE / 'plan.json')['inputs_sha256'])
    inputs.update(summary['parent_inputs_sha256'])
    inputs.update({str(CANDIDATE / name): digest for name, digest in summary['outputs_sha256'].items()})
    verify(inputs)
    return summary, inputs


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent = runner.checked(PARENT, trained=True)
    summary, inputs = candidate_inputs()
    inputs.update(parent['inputs_sha256'])
    inputs.update(runner.read_json(PARENT / 'trained.json')['inputs_sha256'])
    if runner.frozen.sha256(PARENT / 'r29_report.json') != R29_REPORT_SHA:
        raise ValueError('Frozen r29 result changed')
    train, val = (_read_manifest(CANDIDATE / f'candidate/current/{split}.jsonl') for split in ('train', 'val'))
    protected = _read_manifest(CLEAN / 'protected_validation.jsonl')
    audio = {row['audio']: row for row in _read_manifest(CLEAN / 'acoustic.jsonl')}
    flags = runner.read_json(CANDIDATE / 'flagged.json')
    exclusions, eval_inputs = load_exclusions(runner.frozen.ROOT / 'eval_cases')
    inputs.update(eval_inputs)
    audit = audit_mix(train, protected, exclusions, max_dur=30)
    pcm = pcm_conflicts(train, protected, audio)
    if (len(train) != 5788 or len(val) != 895 or len(protected) != 1634
            or audit['violations'] or audit['origins_above_3x'] or any(pcm.values())
            or exclude_flagged(train + val, flags, audio)[1]):
        raise ValueError('Final candidate isolation or scope changed')
    for row in train + val + protected:
        for field in ('audio', 'ref_audio'):
            if row.get(field):
                inputs[row[field]] = audio[row[field]]['file_sha256']
    for path in (PARENT / 'plan.json', PARENT / 'trained.json', PARENT / 'r29_report.json',
                 runner.frozen.ROOT / 'docs/qc_gates.md'):
        inputs[str(path)] = runner.frozen.sha256(path)
    for root in (runner.frozen.ROOT / 'src', runner.frozen.ROOT / 'scripts',
                 runner.frozen.VOXCPM_REPO / 'src', runner.frozen.VOXCPM_REPO / 'scripts'):
        for pattern in ('*.py', '*.sh'):
            inputs.update({str(path): runner.frozen.sha256(path) for path in root.rglob(pattern)})
    verify(inputs)
    if shutil.disk_usage(work.parent).free < 8 * 1024**3:
        raise RuntimeError('Need 8 GiB for checkpoints, evaluation and the existing 4 GiB reserve')
    work.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val'):
        shutil.copyfile(CANDIDATE / f'candidate/current/{split}.jsonl', work / f'{split}.jsonl')
    for name in ('cases.jsonl', 'base_report.json', 'r26_report.json', 'r28_report.json', 'r29_report.json'):
        shutil.copyfile(PARENT / name, work / name)
    config = fixed_config(PARENT / 'train.yaml', work)
    config_path = work / 'train.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    runner.frozen.require_fresh_checkpoint(config)
    issues = preflight(config_path, gpus=1)
    Path(config['save_path']).rmdir()
    if any(not issue.startswith('警告') for issue in issues):
        raise ValueError(f'Training preflight failed: {issues}')
    train_summary, shares = describe_mix(train)
    old_shares = parent['language_shares']
    for lang in shares:
        shares[lang]['requested'] = old_shares[lang]['actual']
    save_new(work / 'mix.json', {
        'train': train_summary, 'validation': dataset_summary(val), 'language_shares': shares,
        'sampling': 'retained original rows once, no language rebalancing',
        'source_candidate': str(CANDIDATE), 'exclusion_flags': str(CANDIDATE / 'flagged.json'),
        'source_summary_sha256': FROZEN['summary.json'], 'checks': audit, 'pcm': pcm,
        'protected_validation_rows': len(protected), 'training_approved': True,
        'authorization': 'User: 继续微调', 'acoustic_and_native_quality': 'unverified',
    })
    for path in work.iterdir():
        if path.is_file():
            inputs[str(path)] = runner.frozen.sha256(path)
    sources = {source_id: {'license': get_source(source_id).license,
                          **dataset_summary([row for row in train if row['source_id'] == source_id])}
               for source_id in sorted({row['source_id'] for row in train})}
    if any(source['license'] not in {'CC0', 'CC-BY-3.0', 'CC-BY-4.0', 'Apache-2.0'} for source in sources.values()):
        raise ValueError('Source outside existing license allowlist')
    plan = {
        **{key: parent[key] for key in ('evaluation_seeds', 'cfg_value', 'inference_timesteps',
            'retry_badcase', 'asr_model', 'base_path', 'asr_path', 'speaker_model_path')},
        'targets': {**parent['targets'], 'r30': str(work / 'checkpoint/latest')},
        'candidate_model': 'r30', 'run_name': RUN_NAME, 'config': str(config_path),
        'num_iters': 459, 'epochs': 459 * 16 / len(train), 'training_seed': 42,
        'rng_verification': verify_rng(), 'training_authorized': True,
        'authorization': 'User: 继续微调', 'max_workers': 4, 'generation_budget': 900,
        'production_changed': False, 'additional_comparisons': [['r30', 'r29'], ['r30', 'r28']],
        'inputs_sha256': inputs, 'train': train_summary, 'validation': dataset_summary(val),
        'language_shares': shares, 'sources': sources, 'audit': audit, 'pcm': pcm, 'preflight': issues,
        'excluded_rows_from_r29': 7252 - len(train), 'protected_validation_rows': len(protected),
        'note': 'Cold start from the same base, seed and fixed 459-step optimizer schedule as r29. '
                'Only use the final full-cleaned and music-excluded current subset; no new pool rows. '
                'Effective exposure is 1.269 epochs, not added training steps. Validation also changes, '
                'so historical validation losses are not comparable. Fixed TTS cases/seeds/metrics remain '
                'the comparison basis; reuse old reports without regenerating WAVs. Cleaning also shifts '
                'language/ref shares; one training seed does not establish causality. '
                'TH share 17.7%, no trusted TL refs or control labels; native/acoustic acceptance unverified.',
    }
    save_new(work / 'plan.json', plan)
    runner.checked(work)
    print(json.dumps({'run_name': RUN_NAME, 'rows': len(train), 'val_rows': len(val),
                      'steps': 459, 'effective_epochs': plan['epochs'], 'preflight': issues},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': runner.run}[args.action](args.work.resolve())
