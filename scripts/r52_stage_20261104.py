"""Staged r52 pipeline (user 2026-11-04: speed up iteration, get the best version fast).

1. build + train r52 (after the soup evaluation releases the GPU)
2. screen: short cues (675) for r52 before any further diagnostics
3. stop if any language is worse than r50 or no language is significantly better;
   otherwise run TH code-switch diagnostics and the dev set (two halves)
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
pr.MODELS['r52'] = '/root/autodl-tmp/r52_th_codeswitch_20261104/checkpoint/latest'


def run(*args):
    subprocess.run([PY, '-X', 'utf8', '-u', *args], cwd=REPO, check=True)


def main():
    root = pr.ROOT
    if not (Path('/root/autodl-tmp/r52_th_codeswitch_20261104/train.done.json')).exists():
        if not (Path('/root/autodl-tmp/r52_th_codeswitch_20261104/plan.json')).exists():
            run('scripts/r52_th_codeswitch_run.py', 'build')
        run('scripts/r52_th_codeswitch_run.py', 'train')
        pr.log('r52 trained')
    # fastest decision first (user: tell me ASAP when the GPU can be shut down): short screen before anything else
    pr.gen_stage('short', 'r52')
    pr.asr_stage_async('short', 'r52')
    pr.finalize('short', 'r52')
    labels = ['base', 'r50', 'r52']
    screen = pr.compare(*cand.ctx('short', ['base', 'r46u10', 'r50', 'r51', 'r52']), 'r52', 'short', ref='r50')
    worse = [l for l, v in screen.items() if v['worse']]
    pr.log(f'r52 short screen vs r50: worse {worse} ' + json.dumps({l: (v['base_severe'], v['cand_severe'])
                                                                   for l, v in screen.items()}))
    if worse or not any(v['better'] for v in screen.values()):
        (root / 'r52_stage.json').write_text(json.dumps({'short_screen_vs_r50': screen, 'screen_worse': worse,
                                                         'stopped': 'not better than r50 on short cues'}, indent=2), encoding='utf-8')
        pr.log('r52 STOP: not better than r50 on short cues -> GPU can be shut down')
        return
    for l in labels:
        pr.gen_stage('th_codeswitch', l)
        pr.asr_stage_async('th_codeswitch', l)
    for l in labels:
        pr.finalize('th_codeswitch', l)
    cs = pr.analyze_phase('th_codeswitch', labels)
    (root / 'th_codeswitch_analysis.json').write_text(json.dumps(cs, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log('th_codeswitch totals ' + json.dumps(cs['_totals']))
    candidates = ['r51', 'soup']
    if not worse:
        for cases in ('dev_a', 'dev_b'):
            pr.gen_stage(cases, 'r52')
            pr.asr_stage_async(cases, 'r52')
        for cases in ('dev_a', 'dev_b'):
            pr.finalize(cases, 'r52')
        candidates.append('r52')
    (root / 'r52_stage.json').write_text(json.dumps({'short_screen_vs_r50': screen, 'screen_worse': worse,
                                                     'candidates_for_final': candidates}, indent=2), encoding='utf-8')
    run('scripts/prod_review_lean_final_20261104.py', *candidates)


if __name__ == '__main__':
    main()
