"""TH code-switch development set (2026-11-04): FLEURS th_th test/dev sentences containing Latin-script words.

Why: r50's remaining TH gap vs r26 is concentrated on Thai text with embedded foreign words (th_1695 "Pa amb Oli"),
production also shows Latin tokens inside Thai lines (UAT "CEO"), and r51's training data has only ~19
code-switched TH rows. This is a development diagnostic, not a holdout: texts of the unused holdout
reserved_r51_20261104 and every training text are excluded; spent holdouts may appear.
Conditioning matches the dev set: zh reference 0001308 (4.85s -> 24 steps), plain, cfg 1.8, 5 seeds.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
REPO = Path('/root/voxcpm-finetune')
FLEURS = ROOT / 'hf_home/hub/datasets--google--fleurs/snapshots/70bb2e84b976b7e960aa89f1c648e09c59f894dd/data/th_th'
REF = ROOT / 'voxft_data/processed/fleurs_zh/audio/run_1owk1jss/0001308.wav'
OUT = ROOT / 'prod_review_20261103/th_codeswitch_cases.jsonl'
LATIN = re.compile(r'[A-Za-z]{2,}')
N = 40


def norm(t):
    return re.sub(r'\W+', '', t.lower())


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    blocked = set()
    for p in [ROOT / 'reserved_r51_20261104/cases.jsonl', ROOT / 'r51_th_restore_20261104/train.jsonl',
              *ROOT.glob('voxft_data/processed/*/train.jsonl')]:
        for line in p.read_text(encoding='utf-8').splitlines():
            t = json.loads(line).get('text', '')
            blocked.add(norm(t.split(')', 1)[-1] if t.startswith('(') else t))
    seen, rows = set(), []
    for split in ('test', 'dev'):
        for line in (FLEURS / f'{split}.tsv').read_text(encoding='utf-8').splitlines():
            cols = line.split('\t')
            if len(cols) < 3:
                continue
            sid, text = cols[1], cols[2].strip()
            if not LATIN.search(text) or re.search(r'\d', text) or not 10 <= len(text) <= 300:
                continue
            k = norm(text)
            if k in blocked or k in seen:
                continue
            seen.add(k)
            rows.append((hashlib.sha256(f'thcs:{split}:{sid}'.encode()).hexdigest(), split, sid, text))
    rows.sort()
    cases = [{'case_id': f'thcs_{split}_{sid}', 'sentence_id': f'thcs_{split}_{sid}', 'lang': 'th', 'text': text,
              'ref_audio': str(REF), 'ref_lang': 'zh', 'control': '', 'condition': 'plain', 'cfg_value': 1.8,
              'inference_timesteps': 24, 'numeric': False, 'source_dataset': 'google/fleurs', 'source_split': split,
              'source_sentence_id': sid, 'cohort': 'th_codeswitch_dev_20261104', 'evaluation_use': 'development_only',
              'latin_tokens': LATIN.findall(text)} for _, split, sid, text in rows[:N]]
    OUT.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    print(len(rows), 'eligible;', len(cases), 'written', OUT)


if __name__ == '__main__':
    main()
