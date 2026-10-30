"""Read-only r31 optimizer, schedule, token-budget and adapter-capacity audit."""
import argparse
from collections import Counter
import math
from pathlib import Path
import re
import statistics

import torch
import yaml

import r31_failure_probe as prior


def spectrum(a, b):
    """Nonzero singular spectrum of B@A using only rank-sized QR cores."""
    _, ra = torch.linalg.qr(a.double().T, mode='reduced')
    _, rb = torch.linalg.qr(b.double(), mode='reduced')
    energy = torch.linalg.svdvals(rb @ ra.T).square()
    total = energy.sum()
    return {'update_frobenius': float(total.sqrt()),
            'top32_energy_share': float(energy[:32].sum() / total) if total > 0 else 0.,
            'rank95': int(torch.searchsorted(energy.cumsum(0), total * .95)) + 1 if total > 0 else 0}


def run(output):
    if output.exists():
        raise FileExistsError(output)
    from safetensors import safe_open
    from transformers import LlamaTokenizerFast
    from voxcpm.model.utils import mask_multichar_chinese_tokens
    from voxcpm.modules.audiovae import AudioVAEConfigV2
    from voxcpm.training.data import compute_sample_lengths
    from datasets import Dataset

    source, frozen = prior.SOURCE, prior.frozen
    prior.review.checked(source / 'plan.json', prior.SOURCE_PLAN_SHA)
    plan = prior.read(source / 'plan.json')
    cfg_path = source / 'train.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    trained = prior.read(source / 'trained.json')
    inputs = {**trained['inputs_sha256']}
    for name in ('train.yaml', 'train.jsonl', 'val.jsonl'):
        path = source / name
        inputs[str(path)] = plan['inputs_sha256'][str(path)]
    for path in (source / 'plan.json', source / 'trained.json', source / 'train.log', Path(__file__)):
        inputs[str(path)] = frozen.sha256(path)
    inputs.update({p: h for p, h in plan['inputs_sha256'].items() if p.startswith(plan['base_path'] + '/')})
    for path, digest in inputs.items():
        prior.review.checked(path, digest)

    rows = frozen._read_manifest(source / 'train.jsonl')
    model_cfg = prior.read(Path(plan['base_path']) / 'config.json')
    tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(plan['base_path'], local_files_only=True))
    token_rows = [{'text_ids': tokenizer(r['text']), 'duration': r['duration'],
                   'ref_audio': r.get('ref_audio'), 'ref_duration': r.get('ref_duration')} for r in rows]
    vae = AudioVAEConfigV2(**(model_cfg.get('audio_vae_config') or {}))
    fps = vae.sample_rate / math.prod(vae.encoder_rates)
    lengths = compute_sample_lengths(Dataset.from_list(token_rows), fps, model_cfg['patch_size'])
    budgets = {str(batch): {'cutoff_tokens': cfg['max_batch_tokens'] // batch,
                'dropped_rows': sum(n > cfg['max_batch_tokens'] // batch for n in lengths),
                'max_estimated_tokens': max(lengths)} for batch in (2, 4, 8)}

    latest = Path(plan['targets']['r31'])
    optimizer = torch.load(latest / 'optimizer.pth', map_location='cpu', weights_only=True)
    groups = [{k: v for k, v in g.items() if k != 'params'} for g in optimizer['param_groups']]
    steps = Counter(int(v['step']) for v in optimizer['state'].values() if 'step' in v)
    scheduler = torch.load(latest / 'scheduler.pth', map_location='cpu', weights_only=True)
    del optimizer
    params, profiles = Counter(), []
    with safe_open(latest / 'lora_weights.safetensors', framework='pt', device='cpu') as weights:
        for name in sorted(weights.keys()):
            if not name.endswith('.lora_A'):
                continue
            prefix = name.removesuffix('.lora_A')
            group = 'lm' if prefix.startswith(('base_lm.', 'residual_lm.')) else 'dit'
            if group == 'dit' and not prefix.startswith('feat_decoder.estimator.'):
                raise ValueError('Unexpected adapter group: ' + prefix)
            a, b = weights.get_tensor(name), weights.get_tensor(prefix + '.lora_B')
            params[group] += a.numel() + b.numel()
            profiles.append({'module': prefix, 'group': group, **spectrum(a, b)})
    text = (source / 'train.log').read_text(encoding='utf-8')
    logs = [{k: float(v) for k, v in re.findall(r'(loss/diff|loss/stop|lr|epoch|grad_norm): ([0-9.e+-]+)', line)}
            for line in text.splitlines() if line.startswith('[train]')]
    warm, total = cfg['warmup_steps'], cfg['max_steps']
    rates = [cfg['learning_rate'] * (s / warm if s < warm else
             .5 * (1 + math.cos(math.pi * (s - warm) / (total - warm)))) for s in range(cfg['num_iters'])]
    summary = {'config': cfg, 'effective_batch': cfg['batch_size'] * cfg['grad_accum_steps'],
        'nominal_sample_exposure': cfg['num_iters'] * cfg['batch_size'] * cfg['grad_accum_steps'] / len(rows),
        'optimizer_groups': groups, 'optimizer_tensor_step_counts': dict(steps), 'scheduler': scheduler,
        'schedule_applied_lr_at_update': {str(s): rates[s] for s in (0, 45, 150, 250, 350, 458)},
        'schedule_sum_lr': sum(rates), 'schedule_last_quarter_share': sum(rates[3*total//4:]) / sum(rates),
        'logged_grad_norm': {'count': len(logs), 'min': min(r['grad_norm'] for r in logs),
            'median': statistics.median(r['grad_norm'] for r in logs), 'max': max(r['grad_norm'] for r in logs),
            'above_clip': sum(r['grad_norm'] > cfg['max_grad_norm'] for r in logs), 'unlogged_steps': 'not_observed'},
        'token_budget_by_microbatch': budgets, 'adapter_parameters': dict(params),
        'adapter_spectrum_by_group': {g: {'modules': sum(p['group'] == g for p in profiles),
            'median_rank95': statistics.median(p['rank95'] for p in profiles if p['group'] == g),
            'median_top32_energy_share': statistics.median(p['top32_energy_share'] for p in profiles if p['group'] == g)}
            for g in params},
        'adapter_profiles': profiles,
        'validation_log': [line for line in text.splitlines() if line.startswith('[val]')],
        'notes': ['Source uses shuffled validation and at most 10 batches, not fixed per-language evaluation.',
                  'Train loss logs show last microbatch; logged LR is after scheduler.step for next update.',
                  'Small adapter spectrum does not prove that lower-rank retraining preserves quality.',
                  'Batch eligibility is a static estimate, not a peak-memory or throughput benchmark.',
                  'No training, inference, original checkpoint/source/report modification.'],
        'inputs_sha256': inputs}
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    prior.save(output, summary)
    print('AUDIT_COMPLETE', {k: summary[k] for k in ('effective_batch', 'adapter_parameters', 'adapter_spectrum_by_group', 'token_budget_by_microbatch')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
