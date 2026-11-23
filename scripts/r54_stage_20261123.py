"""r54 staged evaluation (same flow as r53_stage_20261104.py, production conditions frozen in prod_review_20261103):

1. short-cue set (675 outputs) for r54; compare vs r50 by per-language new-vs-fixed severe sign test
2. if r54 is significantly worse than r50 in any language -> stop (no dev run)
   else run the dev set (two halves) for r54 and write r54_stage.json
No holdout run here: reserved_r51_20261104 is only for a candidate that beats r50 on both, decided separately.
Caveat (TODO A): the dev set has been reused for selection >=10 times; treat dev results as development evidence.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/root/voxcpm-finetune')
PY = str(REPO / '.venv/bin/python')
OUT = Path('/root/autodl-tmp/r54_th_segments_20261123')
sys.path.insert(0, str(REPO / 'src'))
spec = importlib.util.spec_from_file_location('cand', HERE / 'prod_review_candidate_20261103.py')
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
pr = cand.pr
pr.MODELS['r54'] = str(OUT / 'checkpoint/latest')


def main():
    if not (OUT / 'train.done.json').exists():
        raise SystemExit('r54 not trained')
    pr.gen_stage('short', 'r54')
    pr.asr_stage_async('short', 'r54')
    pr.finalize('short', 'r54')
    screen = pr.compare(*cand.ctx('short', ['base', 'r46u10', 'r50', 'r54']), 'r54', 'short', ref='r50')
    vs_base = pr.compare(*cand.ctx('short', ['base', 'r46u10', 'r50', 'r54']), 'r54', 'short', ref='base')
    worse = [l for l, v in screen.items() if v['worse']]
    pr.log('r54 short screen vs r50: worse %s %s' % (worse, json.dumps(
        {l: (v['base_severe'], v['cand_severe']) for l, v in screen.items()})))
    result = {'short_vs_r50': screen, 'short_vs_base': vs_base, 'screen_worse': worse}
    if not worse:
        for cases in ('dev_a', 'dev_b'):
            pr.gen_stage(cases, 'r54')
            pr.asr_stage_async(cases, 'r54')
        for cases in ('dev_a', 'dev_b'):
            pr.finalize(cases, 'r54')
        pr.combine_dev('r54')
        result['dev_vs_r50'] = pr.compare(*cand.ctx('dev', ['base', 'r46u10', 'r50', 'r54']), 'r54', 'dev', ref='r50')
        result['dev_vs_base'] = pr.compare(*cand.ctx('dev', ['base', 'r46u10', 'r50', 'r54']), 'r54', 'dev', ref='base')
    (OUT / 'r54_stage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log('r54 stage done')


if __name__ == '__main__':
    main()
