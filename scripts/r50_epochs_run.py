"""r50: r49 data (r30 pool + MMS-FA 3–6s segments) trained for 3 epochs — the budget of r26.

Why (2026-11-03): r26 (15h x 3 epochs, 1390 steps) was the strongest historical LoRA in the frozen 900-output
rescoring (ID severe 14->1, TH 14->6 vs base) but its weights were deleted. The 459-step series (~1.27 epochs)
is consistently but only mildly better than base. r17 (69h x 2ep) over-fitted and r27 (25h x 3ep) showed no gain,
so ~15h x 3 epochs is the best-supported budget. Single change vs r49: steps 626 -> 3 epochs (warmup ratio kept).
A 2-epoch snapshot is kept for diagnosis; only the final snapshot is evaluated unless it regresses.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

R49 = Path('/root/autodl-tmp/r49_segments_20261103')
OUT = Path('/root/autodl-tmp/r50_3ep_20261103')
REPO = Path('/root/voxcpm-finetune')
PY = str(REPO / '.venv/bin/python')
RUN_NAME = 'lora_joint3_r50_segments_3ep'
EPOCHS = 3


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def prepare():
    OUT.mkdir(exist_ok=False)
    rows = sum(1 for l in (R49 / 'train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip())
    steps = math.ceil(EPOCHS * rows / 16)
    warmup = round(steps * 45 / 459)
    two_ep = math.ceil(2 * rows / 16)
    cfg = (R49 / 'train.yaml').read_text().replace(str(R49 / 'checkpoint'), str(OUT / 'checkpoint'))
    cfg = (cfg.replace('num_iters: 626', f'num_iters: {steps}').replace('max_steps: 626', f'max_steps: {steps}')
           .replace('save_interval: 626', f'save_interval: {two_ep}').replace('warmup_steps: 61', f'warmup_steps: {warmup}'))
    for k in (f'num_iters: {steps}', f'max_steps: {steps}', f'save_interval: {two_ep}', f'warmup_steps: {warmup}',
              str(R49 / 'train.jsonl')):
        if k not in cfg:
            raise ValueError(f'yaml rewrite failed: {k}')
    (OUT / 'train.yaml').write_text(cfg)
    plan = {'run_name': RUN_NAME, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rows': rows, 'steps': steps,
            'warmup': warmup, 'two_epoch_step': two_ep, 'epochs': EPOCHS,
            'inputs_sha256': {str(p): sha(p) for p in (R49 / 'train.jsonl', R49 / 'val.jsonl', R49 / 'train.yaml')},
            'single_variable_vs_r49': 'training budget 1.27 -> 3 epochs (r26 budget)',
            'authorization': 'User 2026-11-03: 最好能整出来一个比历史的都好的微调模型; 拿不准以业界最佳实践决策并执行',
            'production_changed': False}
    (OUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plan, ensure_ascii=False))


def train():
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    for p, d in plan['inputs_sha256'].items():
        if sha(p) != d:
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
    ck = OUT / 'checkpoint'
    state = json.loads((ck / 'latest/training_state.json').read_text()) if code == 0 else None
    if code != 0 or state['step'] < plan['steps'] - 1:
        raise RuntimeError(f'training failed: exit {code}, state {state}')
    time.sleep(90)
    bridge.terminate()
    # disk: keep latest + the 2-epoch snapshot; drop step-0 / duplicate-final snapshots and optimizer states
    latest_sha = sha(ck / 'latest/lora_weights.safetensors')
    kept = {}
    for d in sorted(ck.glob('step_*')):
        step = int(d.name.split('_')[1])
        w = d / 'lora_weights.safetensors'
        if abs(step - plan['two_epoch_step']) <= 1 and w.exists():
            kept[d.name] = sha(w)
            (d / 'optimizer.pth').unlink(missing_ok=True)
        elif not w.exists() or sha(w) == latest_sha or step == 0:
            shutil.rmtree(d)
        else:
            kept[d.name] = sha(w)
    (ck / 'latest/optimizer.pth').unlink(missing_ok=True)
    (OUT / 'train.done.json').write_text(json.dumps({'state': state, 'lora_sha256': latest_sha, 'kept': kept}))
    print('trained', state, kept)


if __name__ == '__main__':
    {'prepare': prepare, 'train': train}[sys.argv[1]]()
