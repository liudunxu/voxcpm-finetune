"""2026-11-03: cut long FLEURS train utterances into 3–6s pause-bounded segments (TL/ID) with MMS forced alignment.

Why: TL training audio is FLEURS only (p50 ~10s, 4 clips <=4s, 61 independent 3–8s clips) while production cues
are short. Industry practice for long read-speech corpora is to segment at natural pauses using a CTC forced
aligner and keep the matching transcript span. The 3–30s training guard is respected (segments are 3–6s).

Rules (frozen here):
- Input = current safe candidate train.jsonl (inherits every historical exclusion); fleurs_tl / fleurs_id rows only.
- Rows whose text contains digits are skipped (verbalisation is ambiguous for alignment).
- Cut only where the aligned inter-word gap >= 0.15s; cut at the gap midpoint, keeping <=0.10s leading and
  <=0.15s trailing silence (project tail cap).
- Segment = consecutive words between cuts, 3.0–6.0s; text = the original whitespace tokens (punctuation kept).
- Reject if mean word alignment score < 0.6 or any word score < 0.2.
- QC: Whisper large-v3 (GPU float16, forced language) on each segment; keep CER <= 0.10 vs segment text.
- Segment text must not equal any eval/reserved case text. Speaker/session/origin copied from the source row; no ref.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

SRC = Path('/root/autodl-tmp/music_excluded_20261029/run/candidate/current/train.jsonl')
OUT = Path('/root/autodl-tmp/seg_short_20261103')
REPO = Path('/root/voxcpm-finetune')
LANG_SOURCES = {'fleurs_tl': 'tl', 'fleurs_id': 'id'}
GAP, LEAD, TAIL = 0.15, 0.10, 0.15
MIN_D, MAX_D = 3.0, 6.0


def norm_word(w):
    w = unicodedata.normalize('NFKD', w.lower())
    return ''.join(c for c in w if 'a' <= c <= 'z' or c == "'")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def align_part(k, n):
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle
    torch.set_num_threads(2)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = [r for r in read_jsonl(SRC) if r.get('source_id') in LANG_SOURCES]
    rows = [r for i, r in enumerate(rows) if i % n == k]
    model, tokenizer, aligner = bundle.get_model().to(dev).half().eval(), bundle.get_tokenizer(), bundle.get_aligner()
    out_rows = []
    (OUT / 'audio').mkdir(parents=True, exist_ok=True)
    for r in rows:
        if re.search(r'\d', r['text']):
            continue
        tokens = r['text'].split()
        words = [norm_word(t) for t in tokens]
        if not all(words):
            continue
        wav, sr = sf.read(r['audio'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, r['audio']
        with torch.inference_mode():
            emission, _ = model(torch.from_numpy(wav)[None].to(dev).half())
            emission = emission.float().cpu()
            spans = aligner(emission[0], tokenizer(words))
        ratio = len(wav) / emission.shape[1] / sr
        bounds, scores = [], []
        for word_spans in spans:
            bounds.append((word_spans[0].start * ratio, word_spans[-1].end * ratio))
            scores.append(sum(s.score * len(s) for s in word_spans) / sum(len(s) for s in word_spans))
        # cut candidates: gap midpoints
        cuts = [0]
        for i in range(len(bounds) - 1):
            if bounds[i + 1][0] - bounds[i][1] >= GAP:
                cuts.append(i + 1)
        cuts.append(len(bounds))
        # greedy: grow segments over cut points to land within [MIN_D, MAX_D]
        seg_id, a = 0, 0
        while a < len(cuts) - 1:
            best = None
            for b in range(a + 1, len(cuts)):
                w0, w1 = cuts[a], cuts[b] - 1
                start = max(0.0, bounds[w0][0] - LEAD)
                end = min(len(wav) / sr, bounds[w1][1] + TAIL)
                if w0 > 0:
                    start = max(start, (bounds[w0 - 1][1] + bounds[w0][0]) / 2)
                if w1 < len(bounds) - 1:
                    end = min(end, (bounds[w1][1] + bounds[w1 + 1][0]) / 2)
                d = end - start
                if d > MAX_D:
                    break
                if d >= MIN_D:
                    best = (b, start, end)
                    break
            if best is None:
                a += 1
                continue
            b, start, end = best
            w0, w1 = cuts[a], cuts[b] - 1
            seg_scores = scores[w0:w1 + 1]
            if sum(seg_scores) / len(seg_scores) >= 0.6 and min(seg_scores) >= 0.2 and (w0 > 0 or w1 < len(bounds) - 1):
                name = f"{Path(r['audio']).stem}_{r['source_id']}_s{seg_id}.wav"
                path = OUT / 'audio' / name
                sf.write(path, wav[int(start * sr):int(end * sr)], sr)
                out_rows.append({k2: v for k2, v in r.items() if k2 not in ('f0_std_st', 'energy_std_db', 'rate',
                                                                             'energy_range_db', 'duration')}
                                | {'source_id': r['source_id'] + '_seg', 'audio': str(path),
                                   'text': ' '.join(tokens[w0:w1 + 1]), 'duration': round(end - start, 3),
                                   'segment_of': r['audio'], 'segment_start': round(start, 3),
                                   'segment_end': round(end, 3), 'align_score_mean': round(sum(seg_scores) / len(seg_scores), 3),
                                   'align_score_min': round(min(seg_scores), 3), 'derivation': 'mms_fa_pause_segment_20261103'})
                seg_id += 1
            a = b
    (OUT / f'aligned_part{k}of{n}.jsonl').write_text(
        ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out_rows), encoding='utf-8')
    print(f'part {k}: {len(rows)} rows -> {len(out_rows)} segments', flush=True)


def qc_part(k, n):
    from difflib import SequenceMatcher  # noqa: F401
    sys.path.insert(0, str(REPO / 'src'))
    from voxft import eval as ev
    from faster_whisper import WhisperModel
    rows = [x for f in sorted(OUT.glob('aligned_part*.jsonl')) for x in read_jsonl(f)]
    rows = [x for i, x in enumerate(rows) if i % n == k]
    model = WhisperModel('large-v3', device='cuda', compute_type='float16', cpu_threads=2)
    out = []
    for x in rows:
        segs, _ = model.transcribe(x['audio'], language=x['lang'], vad_filter=True)
        hyp = ' '.join(s.text.strip() for s in segs)
        x['qc_hyp'] = hyp
        x['qc_cer'] = round(ev._error_rate(ev._norm(hyp), ev._norm(x['text'])), 4)
        out.append(x)
    (OUT / f'qc_part{k}of{n}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out),
                                                encoding='utf-8')
    print(f'qc part {k}: {len(out)}', flush=True)


def finalize():
    import soundfile as sf
    import numpy as np
    eval_texts = set()
    for f in list((REPO / 'eval_cases').glob('*.jsonl')) + [Path('/root/autodl-tmp/prod_review_20261103/short_cases.jsonl')]:
        for line in f.read_text(encoding='utf-8').splitlines():
            try:
                t = json.loads(line).get('text')
            except Exception:
                continue
            if t:
                eval_texts.add(re.sub(r'\W+', '', t.lower()))
    rows = [x for f in sorted(OUT.glob('qc_part*.jsonl')) for x in read_jsonl(f)]
    keep, reasons = [], {'qc_cer': 0, 'eval_text': 0, 'tail': 0}
    for x in rows:
        if x['qc_cer'] > 0.10:
            reasons['qc_cer'] += 1
            continue
        if re.sub(r'\W+', '', x['text'].lower()) in eval_texts:
            reasons['eval_text'] += 1
            continue
        wav, sr = sf.read(x['audio'], dtype='float32')
        frame = int(0.02 * sr)
        rms = np.array([np.sqrt(np.mean(wav[i:i + frame] ** 2)) for i in range(0, len(wav) - frame, frame)])
        thr = max(rms.max() * 10 ** (-40 / 20), 1e-4)
        voiced = np.where(rms > thr)[0]
        tail = (len(rms) - 1 - voiced[-1]) * 0.02 if len(voiced) else 99
        if tail > 0.20:
            reasons['tail'] += 1
            continue
        keep.append(x)
    (OUT / 'segments.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in keep), encoding='utf-8')
    by = {}
    for x in keep:
        b = by.setdefault(x['lang'], {'n': 0, 'hours': 0.0, 'sources': set()})
        b['n'] += 1
        b['hours'] += x['duration'] / 3600
        b['sources'].add(x['segment_of'])
    summary = {'input_segments': len(rows), 'kept': len(keep), 'rejected': reasons,
               'by_lang': {l: {'n': v['n'], 'hours': round(v['hours'], 3), 'source_utterances': len(v['sources'])}
                           for l, v in by.items()}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['align', 'qc', 'finalize'])
    ap.add_argument('shard', nargs='?', default='0/1')
    a = ap.parse_args()
    k, n = map(int, a.shard.split('/'))
    {'align': lambda: align_part(k, n), 'qc': lambda: qc_part(k, n), 'finalize': finalize}[a.cmd]()
