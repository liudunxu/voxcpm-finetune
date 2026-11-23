"""r54: r50 recipe + data unchanged except TH MMS-FA pause segments (3-6s, accurate FLEURS transcripts).

Why (docs/tts_finetune_deepdive_20261123.md §5): TH short cues are the largest remaining gap (short-set severe
base 38 -> r50 36), the p_stop probe showed short-cue severe outputs are content errors rather than stop timing,
and the same segmentation took TL short-cue severe 27 -> 14 in r49/r50. ID segments are already in r50.
Single variable = add TH segments. Fixed: r50 train/val, r50 YAML (r64/alpha64, dropout .05, LM+DiT, lr 1e-4,
batch 2x8, wd .01), 3 epochs (steps rescaled), seed as in the r50 YAML.

Inputs: segments from scripts/fleurs_segment_20261103.py --sources fleurs_th (QC CER<=.10, tail<=.20s, no eval text),
parents chosen by scripts/r54_th_segment_input.py (r53 exclusion rules vs r50). Music bypass: r53's pinned AST
over parent utterances; a segment is dropped if its parent is music-flagged.
Known bias (recorded, not fixed): Whisper writes Thai loanwords in Latin script, so code-switch segments are
over-rejected by the CER gate.

  python scripts/r54_th_segments_run.py music|build|train
"""
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp')
R50 = ROOT / 'r50_3ep_20261103'
BASE = ROOT / 'r49_segments_20261103'  # r50 train/val manifests
SEG = ROOT / 'seg_th_20261123/run/segments.jsonl'
OUT = ROOT / 'r54_th_segments_20261123'
RUN_NAME = 'lora_joint3_r54_th_segments_3ep'
KEEP = {'audio', 'text', 'duration', 'lang', 'source_id', 'speaker', 'speaker_verified', 'emotion_verified',
        'origin_audio', 'session', 'segment_of', 'segment_start', 'segment_end', 'derivation'}

# r53's frozen music()/train() are reused verbatim by pointing their module globals at this run.
_spec = importlib.util.spec_from_file_location('r53', HERE / 'r53_balanced_pool_run.py')
r53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r53)
r53.OUT, r53.RUN_NAME = OUT, RUN_NAME
rd, sha = r53.rd, r53.sha


def music():
    OUT.mkdir(exist_ok=True)
    parents = {}
    for s in rd(SEG):
        parents.setdefault(s['segment_of'], {'audio': s['segment_of']})
    # r53.music() scores every row of OUT/th_sampled.jsonl (here: the segments' parent utterances)
    (OUT / 'th_sampled.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in parents.values()), encoding='utf-8')
    r53.music()


def build():
    scores = {json.loads(line)['audio']: json.loads(line) for line in (OUT / 'music_scores.jsonl').read_text().splitlines()}
    segs = rd(SEG)
    assert all(s['segment_of'] in scores for s in segs), 'music bypass incomplete'
    base_train, val = rd(BASE / 'train.jsonl'), rd(BASE / 'val.jsonl')
    val_audio = {r['audio'] for r in val} | {r.get('origin_audio') for r in val}
    bad = [s for s in segs if s['segment_of'] in val_audio or s['audio'] in val_audio]
    if bad:
        raise ValueError(f'{len(bad)} segments touch r50 val')
    kept = [{k: v for k, v in s.items() if k in KEEP} for s in segs if not scores[s['segment_of']]['music_flag']]
    rows = base_train + kept
    (OUT / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    (OUT / 'val.jsonl').write_bytes((BASE / 'val.jsonl').read_bytes())
    steps = math.ceil(3 * len(rows) / 16)
    warmup = round(steps * 145 / 1480)  # r50 warmup ratio
    cfg = (R50 / 'train.yaml').read_text()
    cfg = (cfg.replace(str(BASE / 'train.jsonl'), str(OUT / 'train.jsonl'))
           .replace(str(BASE / 'val.jsonl'), str(OUT / 'val.jsonl'))
           .replace(str(R50 / 'checkpoint'), str(OUT / 'checkpoint')))
    for key, val_ in (('num_iters', steps), ('max_steps', steps), ('save_interval', steps), ('warmup_steps', warmup)):
        cfg = re.sub(rf'^{key}: \d+$', f'{key}: {val_}', cfg, flags=re.M)
    for k in (f'num_iters: {steps}', f'max_steps: {steps}', str(OUT / 'train.jsonl'), str(OUT / 'checkpoint'),
              f'warmup_steps: {warmup}'):
        assert k in cfg, k
    (OUT / 'train.yaml').write_text(cfg)
    by_lang = {}
    for r in rows:
        b = by_lang.setdefault(r['lang'], {'rows': 0, 'hours': 0.0, 'le4s': 0})
        b['rows'] += 1
        b['hours'] += r['duration'] / 3600
        b['le4s'] += r['duration'] <= 4
    plan = {'run_name': RUN_NAME, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rows': len(rows),
            'base_rows': len(base_train), 'th_segment_rows': len(kept),
            'th_segment_hours': round(sum(r['duration'] for r in kept) / 3600, 3),
            'music_flagged_parents': sum(v['music_flag'] for v in scores.values()),
            'steps': steps, 'warmup': warmup, 'epochs': 3,
            'by_lang': {l: {**v, 'hours': round(v['hours'], 3)} for l, v in by_lang.items()},
            'inputs_sha256': {str(p): sha(p) for p in (BASE / 'train.jsonl', BASE / 'val.jsonl', SEG, R50 / 'train.yaml',
                                                       OUT / 'train.jsonl', Path(__file__))},
            'change_vs_r50': 'add TH MMS-FA pause segments only; recipe and r50 data unchanged',
            'authorization': 'User 2026-10-30 continuous-iteration grant; 2026-11-23 "开机了，继续"',
            'production_changed': False}
    (OUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    {'music': music, 'build': build, 'train': r53.train}[sys.argv[1]]()
