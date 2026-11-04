"""Lean final step (user 2026-11-04: GPU is expensive, only beat r50, cut everything else).

A candidate replaces r50 only if, on short + dev (production protocol), it is significantly better than r50 in
>=1 language and worse in none (same sign test / CER margins). Only then is the unused holdout
reserved_r51_20261104 run, for r50 and that candidate only. Then a TH code-switch diagnostic on base + r50
(+ the winner, if any) to decide whether a code-switch data round is warranted.
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, '/root/voxcpm-finetune/src')
spec = importlib.util.spec_from_file_location('cand', HERE / 'prod_review_candidate_20261103.py')
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
pr = cand.pr


def main(candidates):
    root = pr.ROOT
    labels = ['base', 'r50'] + candidates
    for l in labels:
        pr.combine_dev(l)
    sctx, dctx = cand.ctx('short', labels), cand.ctx('dev', labels)
    out = {}
    for c in candidates:
        v = pr.verdict([pr.compare(*sctx, c, 'short', ref='r50'), pr.compare(*dctx, c, 'dev', ref='r50')])
        out[c] = {'vs_r50': v, 'beats_r50': v['any_better'] and not v['any_worse']}
    winners = [c for c in candidates if out[c]['beats_r50']]
    s, d = pr.analyze_phase('short', labels), pr.analyze_phase('dev', labels)
    tot = {l: s['_totals'][l]['severe'] + d['_totals'][l]['severe'] for l in labels}
    winner = min(winners, key=lambda c: tot[c]) if winners else None
    final = {'rule': __doc__, 'candidates': out, 'severe_short_plus_dev': tot, 'winner_over_r50': winner}
    pr.log(f'lean final: beats_r50 {winners}, winner {winner}')
    if winner:
        rl = ['r50', winner]
        for l in rl:
            pr.gen_stage('reserved2', l)
            pr.asr_stage_async('reserved2', l)
        for l in rl:
            pr.finalize('reserved2', l)
        rctx = cand.ctx('reserved2', rl)
        r = pr.compare(*rctx, winner, 'dev', ref='r50')
        final['reserved2'] = {'vs_r50': r, 'verdict': pr.verdict([r])}
        pr.log(f"reserved2 {winner} vs r50: {json.dumps(final['reserved2']['verdict'])}")
    diag = ['base', 'r50'] + ([winner] if winner else [])
    for l in diag:
        pr.gen_stage('th_codeswitch', l)
        pr.asr_stage_async('th_codeswitch', l)
    for l in diag:
        pr.finalize('th_codeswitch', l)
    final['th_codeswitch'] = pr.analyze_phase('th_codeswitch', diag)
    pr.log('th_codeswitch totals ' + json.dumps(final['th_codeswitch']['_totals']))
    (root / 'final_lean.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log('lean final done')


if __name__ == '__main__':
    main(sys.argv[1:])
