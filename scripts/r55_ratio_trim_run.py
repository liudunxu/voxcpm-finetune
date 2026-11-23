"""r55 (candidate H, docs/tts_finetune_deepdive_20261123.md §4.5): r50 data minus length-ratio outliers.

Single variable vs r50: drop, per language, the rows with the smallest 1% and largest 5% of
duration / text-units (CosyVoice3 §3 rule; units = letters/marks for th/zh, words otherwise, control prefix
stripped). Everything else = r50: same val, r50 YAML (r64/alpha64, dropout .05, LM+DiT, lr 1e-4, batch 2x8,
wd .01), 3 epochs with steps rescaled. Why: in the r50 train set the high-ratio tail is pseudo-label word loss
("Tunggu ojak" 3.0s), sung lines and pause-heavy segments; audio that says more than its text plausibly teaches
over-reading, and the p_stop probe showed short-cue severe outputs are content errors, not stop timing.

  python scripts/r55_ratio_trim_run.py build|train
"""
import importlib.util
import json
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp')
R50 = ROOT / 'r50_3ep_20261103'
BASE = ROOT / 'r49_segments_20261103'
OUT = ROOT / 'r55_ratio_trim_20261123'
RUN_NAME = 'lora_joint3_r55_ratio_trim_3ep'
LOW, HIGH = 0.01, 0.05

_spec = importlib.util.spec_from_file_location('r53', HERE / 'r53_balanced_pool_run.py')
r53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r53)
r53.OUT, r53.RUN_NAME = OUT, RUN_NAME
rd, sha = r53.rd, r53.sha


def text_units(text, lang):
    text = re.sub(r'^\([^()]*\)', '', text or '')
    if lang in ('th', 'zh'):
        return sum(unicodedata.category(c)[0] in 'LNM' for c in text)
    return len(re.findall(r'\w+', text))


def trim(rows, low=LOW, high=HIGH):
    """Return (kept, dropped); per language, drop floor(n*low) lowest and floor(n*high) highest ratios."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        u = text_units(r['text'], r['lang'])
        by[r['lang']].append((r['duration'] / u if u else float('inf'), i))
    drop = set()
    for v in by.values():
        v.sort()
        n = len(v)
        drop |= {i for _, i in v[:int(n * low)]} | {i for _, i in v[n - int(n * high):] if int(n * high)}
    return [r for i, r in enumerate(rows) if i not in drop], [r for i, r in enumerate(rows) if i in drop]


def build():
    OUT.mkdir(exist_ok=False)
    base = rd(BASE / 'train.jsonl')
    kept, dropped = trim(base)
    (OUT / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept), encoding='utf-8')
    (OUT / 'dropped.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in dropped), encoding='utf-8')
    (OUT / 'val.jsonl').write_bytes((BASE / 'val.jsonl').read_bytes())
    steps = math.ceil(3 * len(kept) / 16)
    warmup = round(steps * 145 / 1480)
    cfg = (R50 / 'train.yaml').read_text()
    cfg = (cfg.replace(str(BASE / 'train.jsonl'), str(OUT / 'train.jsonl'))
           .replace(str(BASE / 'val.jsonl'), str(OUT / 'val.jsonl'))
           .replace(str(R50 / 'checkpoint'), str(OUT / 'checkpoint')))
    for key, val in (('num_iters', steps), ('max_steps', steps), ('save_interval', steps), ('warmup_steps', warmup)):
        cfg = re.sub(rf'^{key}: \d+$', f'{key}: {val}', cfg, flags=re.M)
    for k in (f'num_iters: {steps}', str(OUT / 'train.jsonl'), str(OUT / 'checkpoint'), f'warmup_steps: {warmup}'):
        assert k in cfg, k
    (OUT / 'train.yaml').write_text(cfg)
    by_lang = defaultdict(lambda: {'kept': 0, 'dropped': 0, 'dropped_hours': 0.0})
    for r in kept:
        by_lang[r['lang']]['kept'] += 1
    for r in dropped:
        by_lang[r['lang']]['dropped'] += 1
        by_lang[r['lang']]['dropped_hours'] += r['duration'] / 3600
    plan = {'run_name': RUN_NAME, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rows': len(kept),
            'dropped': len(dropped), 'low': LOW, 'high': HIGH, 'steps': steps, 'warmup': warmup, 'epochs': 3,
            'by_lang': {l: {**v, 'dropped_hours': round(v['dropped_hours'], 3)} for l, v in by_lang.items()},
            'dropped_by_source': {s: sum(r['source_id'] == s for r in dropped) for s in {r['source_id'] for r in dropped}},
            'inputs_sha256': {str(p): sha(p) for p in (BASE / 'train.jsonl', BASE / 'val.jsonl', R50 / 'train.yaml',
                                                       OUT / 'train.jsonl', Path(__file__))},
            'change_vs_r50': 'drop per-language length-ratio outliers (1%/5%) only',
            'authorization': 'User 2026-10-30 continuous-iteration grant; 2026-11-23 "能加速就加速下"',
            'production_changed': False}
    (OUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    {'build': build, 'train': r53.train}[sys.argv[1]]()
