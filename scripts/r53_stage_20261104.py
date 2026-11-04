"""Staged r53 pipeline (derived from the r52 stage; r52 paused at 2 epochs by the user) (user 2026-11-04: speed up iteration, get the best version fast).

1. build + train r52 (after the soup evaluation releases the GPU)
2. screen: short cues (675) for r52 + TH code-switch diagnostic for base / r50 / r52
3. if r52 is significantly worse than r50 in any language on short cues -> stop r52 (no dev run)
   else run the dev set (two halves) for r52
4. lean final over {r51, soup, r52 if it has dev}: only a candidate that beats r50 gets the unused holdout.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/root/voxcpm-finetune')
PY = str(REPO / '.venv/bin/python')
sys.path.insert(0, str(REPO / 'src'))
spec = importlib.util.spec_from_file_location('cand', HERE / 'prod_review_candidate_20261103.py')
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
pr = cand.pr
pr.MODELS['r53'] = '/root/autodl-tmp/r53_balanced_pool_20261104/checkpoint/latest'


def run(*args):
    subprocess.run([PY, '-X', 'utf8', '-u', *args], cwd=REPO, check=True)


def main():
    root = pr.ROOT
    if not (Path('/root/autodl-tmp/r53_balanced_pool_20261104/train.done.json')).exists():
        if not (Path('/root/autodl-tmp/r53_balanced_pool_20261104/plan.json')).exists():
            run('scripts/r53_balanced_pool_run.py', 'build')
        run('scripts/r53_balanced_pool_run.py', 'train')
        pr.log('r53 trained')
    for l in ('r53', 'base', 'r50'):
        for cases in (('short', 'th_codeswitch') if l == 'r53' else ('th_codeswitch',)):
            pr.gen_stage(cases, l)
            pr.asr_stage_async(cases, l)
    pr.finalize('short', 'r53')
    labels = ['base', 'r50', 'r53']
    screen = pr.compare(*cand.ctx('short', ['base', 'r46u10', 'r50', 'r51', 'r53']), 'r53', 'short', ref='r50')
    worse = [l for l, v in screen.items() if v['worse']]
    pr.log(f'r53 short screen vs r50: worse {worse} ' + json.dumps({l: (v['base_severe'], v['cand_severe'])
                                                                   for l, v in screen.items()}))
    for l in labels:
        pr.finalize('th_codeswitch', l)
    cs = pr.analyze_phase('th_codeswitch', labels)
    (root / 'th_codeswitch_analysis.json').write_text(json.dumps(cs, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log('th_codeswitch totals ' + json.dumps(cs['_totals']))
    candidates = []
    if not worse:
        for cases in ('dev_a', 'dev_b'):
            pr.gen_stage(cases, 'r53')
            pr.asr_stage_async(cases, 'r53')
        for cases in ('dev_a', 'dev_b'):
            pr.finalize(cases, 'r53')
        candidates.append('r53')
    (root / 'r53_stage.json').write_text(json.dumps({'short_screen_vs_r50': screen, 'screen_worse': worse,
                                                     'candidates_for_final': candidates}, indent=2), encoding='utf-8')
    if candidates:
        run('scripts/prod_review_lean_final_20261104.py', *candidates)
    else:
        pr.log('r53 failed the short screen vs r50; no dev / holdout run')


if __name__ == '__main__':
    main()
