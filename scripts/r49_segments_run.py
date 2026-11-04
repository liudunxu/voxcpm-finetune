"""r49: r30 recipe + MMS-FA pause segments (TL/ID 3–6s). Single variable = data.

Why (2026-11-03 review): production cues are short; TL training audio is FLEURS-only (p50 ~10s, 4 clips <=4s).
r28–r30 (459-step, 12h pool) beat base on ID in the frozen 900-output rescoring, while 10-update r46/u10 is ~base.
Industry practice: fix the data distribution before touching hyper-parameters; keep the recipe fixed.

Fixed: base snapshot, seed 42, LoRA r64/alpha64/dropout.05, LM+DiT, proj off, lr 1e-4, batch 2x8, wd .01,
max_grad_norm 1, val = r30 val (895 rows). Changed: train = r30 train (5788 rows) + QC-passed segments;
steps scaled to keep r30's ~1.269 epochs; warmup kept at ~10% of steps.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

R30 = Path('/root/autodl-tmp/full_clean_candidate_20261029/run')
SEG = Path('/root/autodl-tmp/seg_short_20261103/segments.jsonl')
OUT = Path('/root/autodl-tmp/r49_segments_20261103')
REPO = Path('/root/voxcpm-finetune')
PY = str(REPO / '.venv/bin/python')
RUN_NAME = 'lora_joint3_r49_r30_plus_segments'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def prepare():
    OUT.mkdir(exist_ok=False)
    base_rows = read_jsonl(R30 / 'train.jsonl')
    val_rows = read_jsonl(R30 / 'val.jsonl')
    segs = read_jsonl(SEG)
    train_audio = {r['audio'] for r in base_rows}
    val_audio = {r['audio'] for r in val_rows} | {r.get('origin_audio') for r in val_rows}
    bad = [s for s in segs if s['segment_of'] not in train_audio or s['audio'] in val_audio
           or s.get('origin_audio') in val_audio]
    if bad:
        raise ValueError(f'{len(bad)} segments not derived from r30 train rows or touching val')
    keep_keys = {'audio', 'text', 'duration', 'lang', 'source_id', 'speaker', 'speaker_verified',
                 'emotion_verified', 'origin_audio', 'session', 'segment_of', 'segment_start', 'segment_end',
                 'derivation'}
    seg_rows = [{k: v for k, v in s.items() if k in keep_keys} for s in segs]
    rows = base_rows + seg_rows
    (OUT / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (OUT / 'val.jsonl').write_bytes((R30 / 'val.jsonl').read_bytes())
    r30_cfg = (R30 / 'train.yaml').read_text()
    steps = math.ceil(459 * len(rows) / len(base_rows))
    warmup = round(steps * 45 / 459)
    cfg = (r30_cfg.replace(str(R30 / 'train.jsonl'), str(OUT / 'train.jsonl'))
           .replace(str(R30 / 'val.jsonl'), str(OUT / 'val.jsonl'))
           .replace(str(R30 / 'checkpoint'), str(OUT / 'checkpoint'))
           .replace('num_iters: 459', f'num_iters: {steps}').replace('max_steps: 459', f'max_steps: {steps}')
           .replace('save_interval: 459', f'save_interval: {steps}').replace('warmup_steps: 45', f'warmup_steps: {warmup}'))
    for k in (f'num_iters: {steps}', f'max_steps: {steps}', str(OUT / 'train.jsonl'), f'warmup_steps: {warmup}'):
        if k not in cfg:
            raise ValueError(f'yaml rewrite failed: {k}')
    (OUT / 'train.yaml').write_text(cfg)
    by_lang = {}
    for r in rows:
        b = by_lang.setdefault(r['lang'], {'rows': 0, 'hours': 0.0, 'le4s': 0})
        b['rows'] += 1
        b['hours'] += r['duration'] / 3600
        b['le4s'] += r['duration'] <= 4
    plan = {'run_name': RUN_NAME, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'steps': steps, 'warmup': warmup,
            'epochs': round(steps * 16 / len(rows), 3), 'rows': len(rows), 'base_rows': len(base_rows),
            'segment_rows': len(seg_rows), 'by_lang': {l: {**v, 'hours': round(v['hours'], 3)} for l, v in by_lang.items()},
            'inputs_sha256': {str(p): sha(p) for p in (R30 / 'train.jsonl', R30 / 'val.jsonl', R30 / 'train.yaml', SEG,
                                                      Path(__file__))},
            'single_variable': 'data: + MMS-FA 3–6s pause segments of existing TL/ID FLEURS train rows',
            'authorization': 'User 2026-11-03: 按照你建议来; 拿不准以业界最佳实践决策并执行',
            'production_changed': False}
    (OUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plan, ensure_ascii=False))


def train():
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    for p, d in plan['inputs_sha256'].items():
        if Path(p) != Path(__file__) and sha(p) != d:
            raise ValueError(f'input changed: {p}')
    env = {**os.environ, 'PATH': f'{REPO}/.venv/bin:/root/.local/bin:/usr/bin:/bin', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4'}
    (OUT / 'checkpoint/logs').mkdir(parents=True, exist_ok=True)
    bridge = subprocess.Popen([PY, '-X', 'utf8', '-u', '-m', 'voxft.train.tb_wandb_bridge',
                               str(OUT / 'checkpoint/logs'), RUN_NAME], cwd=REPO, env=env,
                              stdout=(OUT / 'bridge.log').open('w'), stderr=subprocess.STDOUT)
    code = subprocess.call([PY, '-X', 'utf8', '-u', str(REPO / 'third_party/VoxCPM/scripts/train_voxcpm_finetune.py'),
                            '--config_path', str(OUT / 'train.yaml')], cwd=REPO / 'third_party/VoxCPM', env=env,
                           stdout=(OUT / 'train.log').open('w'), stderr=subprocess.STDOUT)
    state = json.loads((OUT / 'checkpoint/latest/training_state.json').read_text()) if code == 0 else None
    if code != 0 or state['step'] < plan['steps'] - 1:
        raise RuntimeError(f'training failed: exit {code}, state {state}')
    time.sleep(90)
    bridge.terminate()
    (OUT / 'train.done.json').write_text(json.dumps({'state': state, 'lora_sha256':
                                                     sha(OUT / 'checkpoint/latest/lora_weights.safetensors')}))
    print('trained', state)


if __name__ == '__main__':
    {'prepare': prepare, 'train': train}[sys.argv[1]]()
