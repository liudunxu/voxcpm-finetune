"""r54 input: eligible fleurs_th rows to cut into MMS-FA pause segments (docs/tts_finetune_deepdive_20261123.md §5).

Same eligibility as scripts/r53_balanced_pool_run.py select() (rules copied, not imported: that script is a frozen
experiment record and writes to its own run dir), relative to r50's data (r49_segments_20261103 train/val):
- full-corpus-cleaned per-source candidate, train split only;
- not on the cumulative flagged list or any historical *excluded*.jsonl, by path / origin path / PCM;
- not r50 val audio, no PCM duplicate of r50 val; session/speaker not shared with r50 val;
- text not protected-validation or any eval/prod_review case text.
Difference from r53: rows already in r50 train ARE allowed (segments derived from train rows, as r49 did from r30).
FLEURS has human transcripts (accurate-transcript rule, r53 lesson). Music bypass still required before training.

  python scripts/r54_th_segment_input.py --out /root/autodl-tmp/seg_th_20261123/input.jsonl
then: fleurs_segment_20261103.py align|qc|finalize --sources fleurs_th --src <input> --out <dir>
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
REPO = Path('/root/voxcpm-finetune')
CLEAN = ROOT / 'full_corpus_clean_20261029/parallel8/run'
BASE = ROOT / 'r49_segments_20261103'  # r50 = r49 data x 3 epochs
EXC = ROOT / 'exclusions_20261104'


def rd(p):
    return [json.loads(line) for line in Path(p).read_text(encoding='utf-8').splitlines() if line.strip()]


def norm_text(t):
    return re.sub(r'\W+', '', (t or '').lower())


def select(src='fleurs_th'):
    r50_val = rd(BASE / 'val.jsonl')
    pcm = {a['audio']: a.get('pcm_sha256') for a in rd(CLEAN / 'acoustic.jsonl')}
    flagged = json.loads((EXC / 'granary_pilot_20261030/run/flagged.json').read_text(encoding='utf-8'))
    blocked_path = {str(Path(p).resolve()) for it in flagged
                    for p in [it['audio'], *(u.get('origin_audio') for u in it['uses'])] if p}
    blocked_pcm = {it['pcm_sha256'] for it in flagged}
    for f in EXC.rglob('*excluded*.jsonl'):
        for r in rd(f):
            row = r.get('sample', r)
            for k in ('audio', 'origin_audio'):
                if row.get(k):
                    blocked_path.add(str(Path(row[k]).resolve()))
            if pcm.get(row.get('audio')):
                blocked_pcm.add(pcm[row['audio']])
    val_audio = {r['audio'] for r in r50_val}
    val_pcm = {pcm.get(a) for a in val_audio} - {None}
    val_sessions = {r.get('session') for r in r50_val if r.get('session')}
    val_speakers = {r.get('speaker') for r in r50_val
                    if r.get('speaker') and not str(r.get('speaker')).endswith(':unknown')}
    protected = {norm_text(r.get('text')) for r in rd(EXC / 'protected_validation.jsonl')}
    for f in list((REPO / 'eval_cases').glob('*.jsonl')) + list((ROOT / 'prod_review_20261103').glob('*_cases.jsonl')):
        for line in f.read_text(encoding='utf-8').splitlines():
            try:
                protected.add(norm_text(json.loads(line).get('text')))
            except (json.JSONDecodeError, AttributeError):
                continue
    protected.discard('')
    ok, why = [], {}
    for r in rd(CLEAN / f'candidate/{src}/train.jsonl'):
        reasons = []
        if any(str(Path(r[k]).resolve()) in blocked_path for k in ('audio', 'origin_audio') if r.get(k)) \
                or pcm.get(r['audio']) in blocked_pcm:
            reasons.append('flagged')
        if r['audio'] in val_audio or pcm.get(r['audio']) in val_pcm:
            reasons.append('r50_val_audio')
        if r.get('session') and r['session'] in val_sessions:
            reasons.append('val_session')
        if r.get('speaker') in val_speakers:
            reasons.append('val_speaker')
        if norm_text(r.get('text')) in protected:
            reasons.append('protected_or_eval_text')
        for x in reasons:
            why[x] = why.get(x, 0) + 1
        if not reasons:
            ok.append(r)
    return ok, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source', default='fleurs_th')
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    ok, why = select(a.source)
    a.out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ok), encoding='utf-8')
    print(json.dumps({'source': a.source, 'eligible': len(ok), 'hours': round(sum(r['duration'] for r in ok) / 3600, 3),
                      'rejected': why}, ensure_ascii=False))


if __name__ == '__main__':
    main()
