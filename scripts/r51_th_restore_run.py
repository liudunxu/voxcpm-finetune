"""r51: r50 recipe (r49 data, 3 epochs) + TH topped up to r26's TH volume. Single variable vs r50 = TH data volume.

Why: under the historical protocol r50 equals/edges r26 on ID/TL but is worse on TH (severe 6->14, p=.004).
r26 trained on 3.36h TH (gigaspeech2_th 1.78h); r50 has 2.13h (0.91h) after later cleaning cut TH to 14.7%.

Eligible TH rows (all must hold):
- in the full-corpus-cleaned per-source candidates (passed ASR/acoustic cleaning), train split only;
- not already in r50 train; no PCM duplicate of any r50 train/val/flagged audio;
- not on the cumulative flagged list (granary_pilot flagged.json: music 62 + granary 773) or any historical
  *excluded*.jsonl, by path / origin path / PCM (same rule as scripts/review_training_silence.exclude_flagged);
- session not shared with current val; text not a protected-validation or any eval text;
- AST AudioSet music bypass (pinned MIT/ast-finetuned-audioset-10-10-0.4593 @ f826b80d, 10s windows / 5s stride,
  per-class max) with Music/Singing/Song/Musical instrument all < 0.8 — the same threshold used to exclude the 62.
Additions per source mirror r26's TH composition (hours added): gigaspeech2_th .87, cv22_th .14, yodas_th .10,
fleurs_th .12. 1.3x is sampled (seed 42) before the music bypass, then trimmed to target.
"""
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
R49 = ROOT / 'r49_segments_20261103'
OUT = ROOT / 'r51_th_restore_20261104'
CLEAN = ROOT / 'full_corpus_clean_20261029/parallel8/run'
REPO = Path('/root/voxcpm-finetune')
PY = str(REPO / '.venv/bin/python')
RUN_NAME = 'lora_joint3_r51_th_restore_3ep'
ADD_HOURS = {'gigaspeech2_th': .87, 'cv22_th': .14, 'yodas_th': .10, 'fleurs_th': .12}
AST_REPO, AST_REV = 'MIT/ast-finetuned-audioset-10-10-0.4593', 'f826b80d28226b62986cc218e5cec390b1096902'
AST_SHA = 'ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d'
EVENTS = ('Music', 'Singing', 'Song', 'Musical instrument')
THRESHOLD = 0.8


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def rd(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def norm_text(t):
    t = t.split(')', 1)[-1] if t.startswith('(') else t
    return re.sub(r'\W+', '', t.lower())


def select():
    OUT.mkdir(exist_ok=False)
    r50_train, r50_val = rd(R49 / 'train.jsonl'), rd(R49 / 'val.jsonl')
    pcm = {}
    for line in (CLEAN / 'acoustic.jsonl').read_text(encoding='utf-8').splitlines():
        a = json.loads(line)
        pcm[a['audio']] = a.get('pcm_sha256')
    flagged = json.loads((ROOT / 'granary_pilot_20261030/run/flagged.json').read_text(encoding='utf-8'))
    blocked_path = {str(Path(p).resolve()) for it in flagged
                    for p in [it['audio'], *(u.get('origin_audio') for u in it['uses'])] if p}
    blocked_pcm = {it['pcm_sha256'] for it in flagged}
    for f in list(ROOT.glob('*/run/**/*excluded*.jsonl')) + list(CLEAN.glob('candidate/*/*excluded*.jsonl')):
        for r in rd(f):
            row = r.get('sample', r)
            for k in ('audio', 'origin_audio'):
                if row.get(k):
                    blocked_path.add(str(Path(row[k]).resolve()))
            if row.get('audio') in pcm and pcm[row['audio']]:
                blocked_pcm.add(pcm[row['audio']])
    used_audio = {r['audio'] for r in r50_train + r50_val}
    used_pcm = {pcm.get(a) for a in used_audio} - {None}
    val_sessions = {r.get('session') for r in r50_val if r.get('session')}
    protected_text = {norm_text(r.get('text', '')) for r in rd(CLEAN / 'protected_validation.jsonl')}
    eval_text = set()
    for f in list((REPO / 'eval_cases').glob('*.jsonl')) + list((ROOT / 'prod_review_20261103').glob('*_cases.jsonl')):
        for line in f.read_text(encoding='utf-8').splitlines():
            try:
                t = json.loads(line).get('text')
            except Exception:
                continue
            if t:
                eval_text.add(norm_text(t))
    rng = random.Random(42)
    picked, report = [], {}
    for src, hours in ADD_HOURS.items():
        rows = rd(CLEAN / f'candidate/{src}/train.jsonl')
        ok, why = [], {}
        for r in rows:
            reasons = []
            if r['audio'] in used_audio:
                reasons.append('in_r50')
            for f, o in (('audio', 'origin_audio'), ('ref_audio', 'ref_origin_audio')):
                if r.get(f) and (any(str(Path(r[k]).resolve()) in blocked_path for k in (f, o) if r.get(k))
                                 or pcm.get(r[f]) in blocked_pcm):
                    reasons.append(f'flagged_{f}')
            if pcm.get(r['audio']) in used_pcm and r['audio'] not in used_audio:
                reasons.append('pcm_dup')
            if r.get('ref_audio') and r['ref_audio'] in {v['audio'] for v in r50_val}:
                reasons.append('ref_is_val')
            if r.get('session') and r['session'] in val_sessions:
                reasons.append('val_session')
            t = norm_text(r.get('text', ''))
            if t in protected_text or t in eval_text:
                reasons.append('protected_or_eval_text')
            if reasons:
                for x in reasons:
                    why[x] = why.get(x, 0) + 1
            else:
                ok.append(r)
        rng.shuffle(ok)
        take, h = [], 0.0
        for r in ok:
            if h >= hours * 1.3 * 3600:
                break
            take.append(r)
            h += r['duration']
        picked += take
        report[src] = {'pool': len(rows), 'eligible': len(ok), 'sampled': len(take), 'sampled_hours': round(h / 3600, 3),
                       'rejected': why}
    (OUT / 'th_sampled.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in picked),
                                          encoding='utf-8')
    (OUT / 'select_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=1))


def fetch_ast():
    from huggingface_hub import hf_hub_download
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        p = Path(hf_hub_download(AST_REPO, name, revision=AST_REV, token=False,
                                 cache_dir=str(ROOT / 'hf_home/hub'), endpoint='https://hf-mirror.com'))
        if name == 'model.safetensors' and sha(p) != AST_SHA:
            raise ValueError('AST weights differ from pinned revision')
    print('AST ok', p.parent)
    return p.parent


def music():
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    mp = fetch_ast()
    proc = ASTFeatureExtractor.from_pretrained(mp, local_files_only=True)
    model = ASTForAudioClassification.from_pretrained(mp, local_files_only=True, use_safetensors=True)
    model.eval().to('cuda', torch.float32)
    ids = {l: model.config.label2id[l] for l in ('Speech',) + EVENTS}
    out = OUT / 'music_scores.jsonl'
    done = {json.loads(l)['audio'] for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open('a', encoding='utf-8') as f:
        for r in rd(OUT / 'th_sampled.jsonl'):
            if r['audio'] in done:
                continue
            wav, sr = sf.read(r['audio'], dtype='float32', always_2d=True)
            assert sr == 16000
            x = wav[:, 0]
            win, stride = 160000, 80000
            last = max(0, len(x) - win)
            starts = list(range(0, last + 1, stride))
            if starts[-1] != last:
                starts.append(last)
            best = {l: 0.0 for l in ids}
            for s in starts:
                inp = proc(x[s:s + win], sampling_rate=16000, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    p = model(**inp).logits[0].sigmoid().cpu().numpy()
                for l, i in ids.items():
                    best[l] = max(best[l], float(p[i]))
            f.write(json.dumps({'audio': r['audio'], 'scores': best,
                                'music_flag': any(best[e] >= THRESHOLD for e in EVENTS)}) + '\n')
            f.flush()
    print('music done')


def build():
    scores = {json.loads(l)['audio']: json.loads(l) for l in (OUT / 'music_scores.jsonl').read_text().splitlines()}
    sampled = rd(OUT / 'th_sampled.jsonl')
    assert all(r['audio'] in scores for r in sampled), 'music bypass incomplete'
    added, report = [], {}
    for src, hours in ADD_HOURS.items():
        rows = [r for r in sampled if r['source_id'] == src]
        clean = [r for r in rows if not scores[r['audio']]['music_flag']]
        take, h = [], 0.0
        for r in clean:
            if h >= hours * 3600:
                break
            take.append(r)
            h += r['duration']
        added += take
        report[src] = {'sampled': len(rows), 'music_flagged': len(rows) - len(clean), 'added': len(take),
                       'added_hours': round(h / 3600, 3)}
    base = rd(R49 / 'train.jsonl')
    rows = base + added
    (OUT / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    shutil.copy2(R49 / 'val.jsonl', OUT / 'val.jsonl')
    steps = math.ceil(3 * len(rows) / 16)
    warmup = round(steps * 45 / 459)
    cfg = (R49 / 'train.yaml').read_text()
    cfg = (cfg.replace(str(R49 / 'train.jsonl'), str(OUT / 'train.jsonl'))
           .replace(str(R49 / 'val.jsonl'), str(OUT / 'val.jsonl'))
           .replace(str(R49 / 'checkpoint'), str(OUT / 'checkpoint'))
           .replace('num_iters: 626', f'num_iters: {steps}').replace('max_steps: 626', f'max_steps: {steps}')
           .replace('save_interval: 626', f'save_interval: {steps}').replace('warmup_steps: 61', f'warmup_steps: {warmup}'))
    for k in (f'num_iters: {steps}', str(OUT / 'train.jsonl'), str(OUT / 'checkpoint'), f'warmup_steps: {warmup}'):
        assert k in cfg, k
    (OUT / 'train.yaml').write_text(cfg)
    by_lang = {}
    for r in rows:
        b = by_lang.setdefault(r['lang'], {'rows': 0, 'hours': 0.0})
        b['rows'] += 1
        b['hours'] += r['duration'] / 3600
    plan = {'run_name': RUN_NAME, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rows': len(rows),
            'added_th_rows': len(added), 'steps': steps, 'warmup': warmup, 'epochs': 3, 'added': report,
            'by_lang': {l: {'rows': v['rows'], 'hours': round(v['hours'], 3)} for l, v in by_lang.items()},
            'inputs_sha256': {str(p): sha(p) for p in (R49 / 'train.jsonl', R49 / 'val.jsonl', OUT / 'train.jsonl')},
            'single_variable_vs_r50': 'TH data volume (+r26-like composition), same recipe/3 epochs',
            'authorization': 'User 2026-11-04: 继续 (beat all historical models)', 'production_changed': False}
    (OUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plan, ensure_ascii=False, indent=1))


def train():
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    for p, d in plan['inputs_sha256'].items():
        assert sha(p) == d, p
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
    latest = sha(ck / 'latest/lora_weights.safetensors')
    for d in sorted(ck.glob('step_*')):          # keep only latest (disk); drop optimizer
        shutil.rmtree(d)
    (ck / 'latest/optimizer.pth').unlink(missing_ok=True)
    (OUT / 'train.done.json').write_text(json.dumps({'state': state, 'lora_sha256': latest}))
    print('trained', state, latest)


if __name__ == '__main__':
    {'select': select, 'music': music, 'build': build, 'train': train}[sys.argv[1]]()
