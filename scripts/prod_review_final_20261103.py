"""Final pick among the phase-1 winner and follow-up candidates, then the reserved set exactly once.

Eligible = pass vs base (no worse language on short+dev, >=1 better) AND no worse language vs production r46u10.
Among eligible candidates: a candidate that is significantly better than another (>=1 language better, none worse)
dominates it; remaining ties -> fewest severe outputs over short+dev, then lower dev mean CER.
reserved_next_20261102 is used once for the single finalist (base / r46u10 / finalist) and never for tuning again.
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
spec2 = importlib.util.spec_from_file_location('cand', HERE / 'prod_review_candidate_20261103.py')
cand = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(cand)


def main(followups):
    root = pr.ROOT
    deferred = json.loads((root / 'phase3_deferred.json').read_text(encoding='utf-8'))
    winner = deferred['winner']
    pool = [c for c in [winner] + followups if (root / f'done_dev_{c}.json').exists() or c == winner]
    labels = ['base', 'r46u10'] + pool
    for l in labels:
        if not (root / f'done_dev_{l}.json').exists():
            pr.combine_dev(l)
    sctx, dctx = cand.ctx('short', labels), cand.ctx('dev', labels)
    s, d = pr.analyze_phase('short', labels), pr.analyze_phase('dev', labels)

    def v(c, ref):
        return pr.verdict([pr.compare(*sctx, c, 'short', ref=ref), pr.compare(*dctx, c, 'dev', ref=ref)])
    summary = {}
    for c in pool:
        summary[c] = {'vs_base': v(c, 'base'), 'vs_r46u10': v(c, 'r46u10'),
                      'severe': s['_totals'][c]['severe'] + d['_totals'][c]['severe'],
                      'dev_cer': d['_totals'][c]['mean_cer'], 'short_cer': s['_totals'][c]['mean_cer']}
        summary[c]['eligible'] = summary[c]['vs_base']['pass'] and not summary[c]['vs_r46u10']['any_worse']
    eligible = [c for c in pool if summary[c]['eligible']]
    pairwise = {f'{a}_vs_{b}': v(a, b) for a in eligible for b in eligible if a != b}
    dominated = {b for a in eligible for b in eligible if a != b
                 and pairwise[f'{a}_vs_{b}']['any_better'] and not pairwise[f'{a}_vs_{b}']['any_worse']}
    finalists = [c for c in eligible if c not in dominated] or eligible
    chosen = min(finalists, key=lambda c: (summary[c]['severe'], summary[c]['dev_cer'])) if finalists else None
    final = {'pool': pool, 'summary': summary, 'pairwise': pairwise, 'eligible': eligible, 'chosen': chosen,
             'baseline_totals': {l: {'severe': s['_totals'][l]['severe'] + d['_totals'][l]['severe'],
                                     'dev_cer': d['_totals'][l]['mean_cer']} for l in ('base', 'r46u10')},
             'rule': __doc__}
    pr.log(f'final pick: eligible {eligible}, chosen {chosen}')
    if chosen:
        rl = ['base', 'r46u10', chosen]
        for l in rl:
            pr.gen_stage('reserved', l)
            pr.asr_stage_async('reserved', l)
        for l in rl:
            pr.finalize('reserved', l)
        p3 = pr.analyze_phase('reserved', rl)
        final['phase3'] = p3
        final['phase3_verdict'] = {'non_regression_vs_base': not any(x['worse'] for x in p3[chosen].values()),
                                   'any_better_vs_base': any(x['better'] for x in p3[chosen].values()),
                                   'no_worse_vs_r46u10': not any(x['worse'] for x in p3[f'{chosen}_vs_r46u10'].values())}
        final['note'] = 'reserved_next_20261102 used once for the single finalist; never use it for further tuning'
        pr.log(f"reserved {chosen}: {json.dumps(final['phase3_verdict'])}")
    else:
        final['note'] = 'no candidate eligible to replace r46u10; reserved set NOT used'
    (root / 'final.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log('final done')


if __name__ == '__main__':
    main(sys.argv[1:])
