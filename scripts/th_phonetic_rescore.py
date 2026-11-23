"""Re-score Thai short-cue reports phonetically and per sentence (no regeneration).

Why (2026-11-23): in the frozen short set, 'ไม่!' is ~10/15 severe for every model incl. base because Whisper
writes colloquial spellings (ม่าย/มัย); real model differences sit on 3-4 of the 15 TH sentences. Royin
romanization (pythainlp, tone-less) maps such spelling variants together while real misreads stay distinct
(อย่านะ yana vs อย่างนั้น yangnan). This is a secondary metric: the frozen prod_review rule is not replaced.

For TH items only: cer/over_read/suspected_truncation are recomputed on romanized text, then prod_review's
severe_flags/compare are applied unchanged. Also reports a sentence-level sign test (sentence = unit).

  python scripts/th_phonetic_rescore.py <reports_dir> base r50 r54 r55 --ref r50
"""
import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def phonetic(text, romanize=None):
    from voxft import eval as ev
    if romanize is None:
        from pythainlp.transliterate import romanize
    return ''.join(romanize(w) for w in ev._norm(text or '', keep_spaces=True).split())


def rescore_item(it, romanize=None):
    from voxft import eval as ev
    if it['lang'] != 'th':
        return it
    hyp, ref = phonetic(it.get('hyp'), romanize), phonetic(it['text'], romanize)
    return it | {'cer': ev._error_rate(hyp, ref), 'over_read': ev._over_read(hyp, ref),
                 'suspected_truncation': ev._is_truncated(hyp, ref)}


def sentence_sign(flags, items, cand, ref, lang):
    per = defaultdict(lambda: [0, 0])
    for k, f in flags[cand].items():
        if k in flags[ref] and items[cand][k]['lang'] == lang:
            sid = items[cand][k]['sentence_id']
            per[sid][0] += flags[ref][k]['severe']
            per[sid][1] += f['severe']
    worse = sum(c > r for r, c in per.values())
    better = sum(c < r for r, c in per.values())
    n = worse + better
    p_worse = sum(math.comb(n, j) for j in range(worse, n + 1)) / 2 ** n if n else 1.0
    return {'sentences': len(per), 'worse_sentences': worse, 'better_sentences': better, 'p_worse': round(p_worse, 4),
            'per_sentence': {s: v for s, v in sorted(per.items()) if v[0] != v[1]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reports_dir', type=Path)
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--ref', default='r50')
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('pr', HERE / 'prod_review_20261103.py')
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    raw = {l: pr.load_items(a.reports_dir / f'short_{l}.json') for l in a.labels}
    phon = {l: {k: rescore_item(it) for k, it in items.items()} for l, items in raw.items()}
    out = {}
    for name, reports in (('orthographic', raw), ('phonetic', phon)):
        flags = pr.severe_flags(reports, 'short')
        out[name] = {
            'th_severe': {l: sum(f['severe'] for k, f in flags[l].items() if reports[l][k]['lang'] == 'th') for l in a.labels},
            'vs_ref_th': {c: pr.compare(reports, flags, c, 'short', ref=a.ref)['th'] for c in a.labels if c != a.ref},
            'sentence_level_th': {c: sentence_sign(flags, reports, c, a.ref, 'th') for c in a.labels if c != a.ref},
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(HERE.parent / 'src'))
    main()
