"""r49 follow-up for the 2026-11-03 production-condition review (same functions, ASR config and rules).

Compares r49 with base / r46u10 / phase-1 winner on the short-cue set and the full dev set (two halves,
same interim harm boundary). Reports of the other models are reused from the main review (same v2 ASR).
Then picks the single finalist among {phase-1 winner, r49} and runs the reserved set exactly once for it
(main review runs with DEFER_PHASE3=1).
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

R49 = '/root/autodl-tmp/r49_segments_20261103/checkpoint/latest'


def main():
    pr.MODELS['r49'] = R49
    root = pr.ROOT
    p1 = json.loads((root / 'phase1_analysis.json').read_text(encoding='utf-8'))
    winner = p1['_winner']
    pr.log(f'r49 follow-up start (winner={winner})')
    pr.split_dev()
    for cases in ('short', 'dev_a', 'dev_b'):
        pr.gen_stage(cases, 'r49')
        pr.asr_stage_async(cases, 'r49')
    for cases in ('short', 'dev_a', 'dev_b'):
        pr.finalize(cases, 'r49')
    for label in ('base', 'r46u10', winner):
        for cases in ('short', 'dev_a', 'dev_b'):
            if not (root / f'done_{cases}_{label}.json').exists():
                pr.gen_stage(cases, label)
                pr.asr_stage_async(cases, label)
                pr.finalize(cases, label)
    for label in ('base', 'r46u10', winner, 'r49'):
        pr.combine_dev(label)
    labels = ['base', 'r46u10', winner, 'r49']
    s = pr.analyze_phase('short', labels)
    d = pr.analyze_phase('dev', labels)
    sets_vs = lambda ref: [pr.compare(*_ctx('short', labels), 'r49', 'short', ref=ref),
                           pr.compare(*_ctx('dev', labels), 'r49', 'dev', ref=ref)]
    out = {'winner': winner, 'short': s, 'dev': d,
           'r49_vs_base': pr.verdict([s['r49'], d['r49']]),
           'r49_vs_r46u10': pr.verdict(sets_vs('r46u10')),
           f'r49_vs_{winner}': pr.verdict(sets_vs(winner))}
    out['r49_replace_production'] = out['r49_vs_base']['pass'] and not out['r49_vs_r46u10']['any_worse']
    (root / 'r49_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log(f"r49 verdicts: base {out['r49_vs_base']} r46u10 {out['r49_vs_r46u10']} {winner} {out[f'r49_vs_{winner}']}")
    # Final selection among the phase-1 winner and r49, then the reserved set exactly once.
    deferred = json.loads((root / 'phase3_deferred.json').read_text(encoding='utf-8'))
    eligible = []
    if deferred['verdicts'].get(winner, {}).get('replace_production'):
        eligible.append(winner)
    if out['r49_replace_production']:
        eligible.append('r49')
    if len(eligible) == 2:
        vw = out[f'r49_vs_{winner}']
        if vw['any_better'] and not vw['any_worse']:
            chosen = 'r49'
        elif vw['any_worse'] and not vw['any_better']:
            chosen = winner
        else:   # tie-break: fewer severe outputs over short + dev
            tot = lambda l: s['_totals'][l]['severe'] + d['_totals'][l]['severe']
            chosen = min(eligible, key=lambda l: (tot(l), l != 'r49'))
    else:
        chosen = eligible[0] if eligible else None
    final = {'eligible': eligible, 'chosen': chosen, 'winner': winner, 'phase2_verdicts': deferred['verdicts'],
             'r49': {k: out[k] for k in ('r49_vs_base', 'r49_vs_r46u10', f'r49_vs_{winner}', 'r49_replace_production')},
             'rule': 'eligible = pass vs base AND no worse language vs r46u10 on short+dev; if both eligible, '
                     'r49 vs winner verdict decides, tie -> fewer severe outputs (short+dev)'}
    if chosen is None:
        final['note'] = 'no candidate eligible to replace r46u10; reserved set NOT used'
        pr.log(final['note'])
    else:
        labels = ['base', 'r46u10', chosen]
        for label in labels:
            pr.gen_stage('reserved', label)
            pr.asr_stage_async('reserved', label)
        for label in labels:
            pr.finalize('reserved', label)
        p3 = pr.analyze_phase('reserved', labels)
        final['phase3'] = p3
        final['phase3_verdict'] = {'non_regression_vs_base': not any(v['worse'] for v in p3[chosen].values()),
                                   'any_better_vs_base': any(v['better'] for v in p3[chosen].values()),
                                   'no_worse_vs_r46u10': not any(v['worse'] for v in p3[f'{chosen}_vs_r46u10'].values())}
        final['note'] = 'reserved_next_20261102 used once for the single finalist; never use it for further tuning'
        pr.log(f"final {chosen}: {json.dumps(final['phase3_verdict'])}")
    (root / 'final.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log('r49 follow-up done')


def _ctx(cases, labels):
    reports = {l: pr.load_items(json.loads((pr.ROOT / f'done_{cases}_{l}.json').read_text())['report']) for l in labels}
    return reports, pr.severe_flags(reports, 'short' if cases == 'short' else 'dev')


if __name__ == '__main__':
    main()
