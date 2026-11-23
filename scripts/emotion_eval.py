"""Emotion consistency, stage 0 (docs/tts_finetune_deepdive_20261123.md §7): measure before training.

Production takes emotion from the reference clip (control prefix unused), so we measure whether the output
follows the reference emotion and how stable it is per reference across sentences, base vs candidate.
Tool: emotion2vec+ large (FunASR MODEL_LICENSE: use/modify/share with attribution; evaluation only).

  calibrate  python scripts/emotion_eval.py calibrate <thai_ser manifest.jsonl> <out.json>
  build      python scripts/emotion_eval.py build <thai_ser manifest> <uat_ref_dir> <short_cases.jsonl> <out emo0_cases.jsonl>
  score      python scripts/emotion_eval.py score <out.json> <report_base.json> <report_cand.json> [...]

Metrics per output: cos(emb(ref), emb(out)) and top-1 label agreement; per reference: mean pairwise cos among its
outputs (stability). Model comparison is paired on (case_id, seed); 95% CI by bootstrap over references.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL = 'emotion2vec/emotion2vec_plus_large'
THAI_SER_MAP = {'angry': 'angry', 'happy': 'happy', 'sad': 'sad', 'neutral': 'neutral'}  # 'frustrated' has no class
TEXTS = {  # neutral production-style cues (eval_cases/omnivoice_prod_v2.jsonl; tl: 1 v2 + 2 tl_tail), no digits
    'th': ['อันนี้แหละที่เขากลัวกันแต่ต้องระวังการยักย้ายถ่ายเทด้วยนะคะ',
           'นะครับเราก็ได้ขอความร่วมมือในเรื่องนี้ไปด้วยนะครับ',
           'เพื่อรีเช็คดูว่ามีใครถูกแฮคถูกขโมยไหม'],
    'id': ['Dia tinggal sama neneknya tapi neneknya enggak pernah keluar rumah',
           'Iya iya sudah nanti saya kabarin oke',
           'Tukang soto bukan tukang gali asal ngomong nih'],
    'tl': ['Ang lahat ng mga mamamayan ng Lungsod ng Vatican ay mga Romano Katoliko.',
           'Magkita tayo bukas sa Makati.',
           'Busy ako ngayon, mamaya na lang.'],
}


class Emo:
    def __init__(self):
        from funasr import AutoModel
        self.m = AutoModel(model=MODEL, hub='hf', disable_update=True)
        self.cache = {}

    def __call__(self, path):
        if path not in self.cache:
            r = self.m.generate(str(path), granularity='utterance', extract_embedding=True)[0]
            labels = [l.split('/')[-1] for l in r['labels']]
            emb = np.asarray(r['feats'], dtype=np.float32).reshape(-1)
            self.cache[path] = (labels[int(np.argmax(r['scores']))], emb / (np.linalg.norm(emb) + 1e-9))
        return self.cache[path]


def rd(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]


def calibrate(manifest, out):
    emo = Emo()
    rows = [r for r in rd(manifest) if r['emotion'] in THAI_SER_MAP]
    conf, hit, hit_hi, n_hi = defaultdict(lambda: defaultdict(int)), 0, 0, 0
    for r in rows:
        pred, _ = emo(r['audio'])
        ok = pred == THAI_SER_MAP[r['emotion']]
        conf[r['emotion']][pred] += 1
        hit += ok
        if (r.get('agreement') or 0) >= 0.8:
            n_hi += 1
            hit_hi += ok
    res = {'n': len(rows), 'top1_acc': round(hit / len(rows), 4), 'n_agreement_ge_.8': n_hi,
           'top1_acc_agreement_ge_.8': round(hit_hi / max(n_hi, 1), 4), 'chance_4class': .25,
           'confusion': {k: dict(v) for k, v in conf.items()}}
    Path(out).write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=1))


def build(manifest, uat_dir, short_cases, out, seed=42):
    import soundfile as sf
    rng = random.Random(seed)
    refs = []
    rows = [r for r in rd(manifest) if r['emotion'] in THAI_SER_MAP and (r.get('agreement') or 0) >= 1.0
            and 3.0 <= sf.info(r['audio']).duration <= 8.0]
    for emo in THAI_SER_MAP:
        cand = [r for r in rows if r['emotion'] == emo]
        rng.shuffle(cand)
        picked, speakers = [], set()
        for r in cand:
            if r['speaker'] not in speakers:
                picked.append(r)
                speakers.add(r['speaker'])
            if len(picked) == 2:
                break
        refs += [{'ref': f"thaiser_{emo}_{r['speaker']}", 'ref_audio': r['audio'], 'ref_lang': 'th',
                  'ref_emotion_label': emo} for r in picked]
    refs += [{'ref': f'uat_{p.stem}', 'ref_audio': str(p.resolve()), 'ref_lang': 'zh', 'ref_emotion_label': None}
             for p in sorted(Path(uat_dir).glob('*.wav')) if 'crop' not in p.name]
    tmpl = rd(short_cases)[0]
    cases = []
    for ref in refs:
        dur = sf.info(ref['ref_audio']).duration
        steps = 28 if dur < 2 else 24 if dur < 5 else 22  # production adaptive steps (prod_review steps_for)
        for lang, texts in TEXTS.items():
            for i, text in enumerate(texts):
                cases.append({'case_id': f"emo0_{ref['ref']}_{lang}{i}", 'sentence_id': f'emo0_{lang}{i}',
                              'lang': lang, 'text': text, 'ref_audio': ref['ref_audio'], 'ref_arm': ref['ref'],
                              'ref_lang': ref['ref_lang'], 'ref_sec': round(dur, 3), 'ref_emotion_label': ref['ref_emotion_label'],
                              'control': '', 'cfg_value': tmpl['cfg_value'], 'inference_timesteps': steps,
                              'numeric': False, 'cohort': 'emotion_stage0_20261123', 'evaluation_use': 'development'})
    Path(out).write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    print(len(refs), 'refs', len(cases), 'cases')


def per_output(emo, items):
    out = {}
    for it in items:
        if not it.get('wav'):
            continue
        rl, re_ = emo(it['ref_audio'])
        ol, oe = emo(it['wav'])
        out[(it['case_id'], it['seed'])] = {'ref': it['ref_arm'], 'lang': it['lang'], 'cos': float(re_ @ oe),
                                            'agree': rl == ol, 'ref_label': rl, 'out_label': ol, 'emb': oe}
    return out


def stability(rows):
    by = defaultdict(list)
    for r in rows.values():
        by[r['ref']].append(r['emb'])
    vals = {}
    for ref, embs in by.items():
        e = np.stack(embs)
        s = e @ e.T
        n = len(e)
        vals[ref] = float((s.sum() - n) / (n * (n - 1))) if n > 1 else None
    return vals


def paired_ci(a, b, key, n_boot=2000, seed=0):
    keys = [k for k in a if k in b]
    by_ref = defaultdict(list)
    for k in keys:
        by_ref[a[k]['ref']].append(b[k][key] - a[k][key])
    refs = sorted(by_ref)
    means = [np.mean(by_ref[r]) for r in refs]
    rng = np.random.default_rng(seed)
    boots = [float(np.mean(rng.choice(means, size=len(means), replace=True))) for _ in range(n_boot)]
    return {'n': len(keys), 'refs': len(refs), 'mean_diff': round(float(np.mean(means)), 4),
            'ci95': [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}


def score(out, report_paths):
    emo = Emo()
    reps = {}
    for p in report_paths:
        rep = json.loads(Path(p).read_text(encoding='utf-8'))
        reps[rep['label']] = per_output(emo, rep['items'])
    labels = list(reps)
    res = {'model': MODEL, 'per_model': {}}
    for l, rows in reps.items():
        st = stability(rows)
        res['per_model'][l] = {
            'n': len(rows), 'mean_cos': round(float(np.mean([r['cos'] for r in rows.values()])), 4),
            'top1_agree': round(float(np.mean([r['agree'] for r in rows.values()])), 4),
            'by_lang_cos': {lg: round(float(np.mean([r['cos'] for r in rows.values() if r['lang'] == lg])), 4)
                            for lg in sorted({r['lang'] for r in rows.values()})},
            'mean_stability': round(float(np.mean([v for v in st.values() if v is not None])), 4),
            'stability_by_ref': {k: round(v, 4) for k, v in st.items() if v is not None},
        }
    base = labels[0]
    for c in labels[1:]:
        res[f'{c}_vs_{base}'] = {'cos': paired_ci(reps[base], reps[c], 'cos'),
                                 'agree': paired_ci(reps[base], reps[c], 'agree'),
                                 'by_lang_cos': {lg: paired_ci({k: v for k, v in reps[base].items() if v['lang'] == lg},
                                                               reps[c], 'cos')
                                                 for lg in sorted({r['lang'] for r in reps[base].values()})}}
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'per_model'} | {
        'per_model': {l: {k: v for k, v in m.items() if k != 'stability_by_ref'} for l, m in res['per_model'].items()}},
        ensure_ascii=False, indent=1))


if __name__ == '__main__':
    cmd, *args = sys.argv[1:]
    {'calibrate': lambda: calibrate(*args), 'build': lambda: build(*args),
     'score': lambda: score(args[0], args[1:])}[cmd]()
