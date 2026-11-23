"""Candidate staged evaluation (r54, r55, ...) (same flow as r53_stage_20261104.py, production conditions frozen in prod_review_20261103):

  python scripts/candidate_stage_20261123.py <label> <run_dir> [gen_workers]

1. short-cue set (675 outputs) for <label>; compare vs r50 by per-language new-vs-fixed severe sign test
2. if the candidate is significantly worse than r50 in any language -> stop (no dev run)
   else run the dev set (two halves) and write <run_dir>/<label>_stage.json
No holdout run here: reserved_r51_20261104 is only for a candidate that beats r50 on both, decided separately.
Caveat (TODO A): the dev set has been reused for selection >=10 times; treat dev results as development evidence.
"""
import importlib.util
import json
import os
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


def main(label, out, gen_workers=None):
    if gen_workers:
        pr.GEN_WORKERS = gen_workers  # share the GPU with a concurrent training run
    # gen/asr run as fresh prod_review subprocesses that read MODELS from that file, so it must be registered there
    assert pr.MODELS.get(label) == str(out / 'checkpoint/latest'), f'add {label} to prod_review_20261103.MODELS'
    if not (out / 'train.done.json').exists():
        raise SystemExit(f'{label} not trained')
    pr.gen_stage('short', label)
    pr.asr_stage_async('short', label)
    pr.finalize('short', label)
    screen = pr.compare(*cand.ctx('short', ['base', 'r46u10', 'r50', label]), label, 'short', ref='r50')
    vs_base = pr.compare(*cand.ctx('short', ['base', 'r46u10', 'r50', label]), label, 'short', ref='base')
    worse = [l for l, v in screen.items() if v['worse']]
    pr.log(f'{label} short screen vs r50: worse %s %s' % (worse, json.dumps(
        {l: (v['base_severe'], v['cand_severe']) for l, v in screen.items()})))
    result = {'short_vs_r50': screen, 'short_vs_base': vs_base, 'screen_worse': worse}
    # STAGE_SHORT_ONLY=1 while a training run shares the GPU (dev gen uses 5 x ~7.8GB); rerun later for dev
    if not worse and os.environ.get('STAGE_SHORT_ONLY') != '1':
        for cases in ('dev_a', 'dev_b'):
            pr.gen_stage(cases, label)
            pr.asr_stage_async(cases, label)
        for cases in ('dev_a', 'dev_b'):
            pr.finalize(cases, label)
        pr.combine_dev(label)
        result['dev_vs_r50'] = pr.compare(*cand.ctx('dev', ['base', 'r46u10', 'r50', label]), label, 'dev', ref='r50')
        result['dev_vs_base'] = pr.compare(*cand.ctx('dev', ['base', 'r46u10', 'r50', label]), label, 'dev', ref='base')
    (out / f'{label}_stage.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pr.log(f'{label} stage done')


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
