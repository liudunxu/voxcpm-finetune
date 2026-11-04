"""Candidate follow-up for the 2026-11-03 production-condition review: evaluate one new LoRA (e.g. r49, r50)
with the same functions, ASR config and rules, and compare it with base / r46u10 / phase-1 winner on the
short-cue set and the full dev set. Writes <label>_analysis.json. Final pick + reserved: prod_review_final_20261103.py.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pr', HERE / 'prod_review_20261103.py')
pr = importlib.util.module_from_spec(spec)
sys.path.insert(0, '/root/voxcpm-finetune/src')
spec.loader.exec_module(pr)


def ctx(cases, labels):
    reports = {l: pr.load_items(json.loads((pr.ROOT / f'done_{cases}_{l}.json').read_text())['report']) for l in labels}
    return reports, pr.severe_flags(reports, 'short' if cases == 'short' else 'dev')


def evaluate(label):
    root = pr.ROOT
    winner = json.loads((root / 'phase1_analysis.json').read_text(encoding='utf-8'))['_winner']
    pr.log(f'{label} follow-up start (winner={winner})')
    pr.split_dev()
    for cases in ('short', 'dev_a', 'dev_b'):
        pr.gen_stage(cases, label)
        pr.asr_stage_async(cases, label)
    for cases in ('short', 'dev_a', 'dev_b'):
        pr.finalize(cases, label)
    for ref in ('base', 'r46u10', winner):
        for cases in ('short', 'dev_a', 'dev_b'):
            if not (root / f'done_{cases}_{ref}.json').exists():
                pr.gen_stage(cases, ref)
                pr.asr_stage_async(cases, ref)
                pr.finalize(cases, ref)
    labels = ['base', 'r46u10', winner, label]
    for l in labels:
        pr.combine_dev(l)
    s, d = pr.analyze_phase('short', labels), pr.analyze_phase('dev', labels)
    vs = lambda ref: pr.verdict([pr.compare(*ctx('short', labels), label, 'short', ref=ref),
                                 pr.compare(*ctx('dev', labels), label, 'dev', ref=ref)])
    out = {'label': label, 'winner': winner, 'short': s, 'dev': d,
           'vs_base': pr.verdict([s[label], d[label]]), 'vs_r46u10': vs('r46u10'), f'vs_{winner}': vs(winner),
           'totals': {l: {'severe': s['_totals'][l]['severe'] + d['_totals'][l]['severe'],
                          'short_cer': s['_totals'][l]['mean_cer'], 'dev_cer': d['_totals'][l]['mean_cer']}
                      for l in labels}}
    out['replace_production'] = out['vs_base']['pass'] and not out['vs_r46u10']['any_worse']
    (root / f'{label}_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log(f"{label} verdicts: base {out['vs_base']} r46u10 {out['vs_r46u10']} {winner} {out[f'vs_{winner}']}")


if __name__ == '__main__':
    evaluate(sys.argv[1])
