"""Read-only, bounded near-text leakage audit of current training versus frozen holdouts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

from audit_training_quality import CLEAN, read
from fresh_baseline_eval import save_new
from full_corpus_parallel import write_status
from incremental_mix import target_text
from ref_ab import runtime_environment, sha256
from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.eval import _norm
from voxft.paths import ROOT


def grams(text):
    return Counter(text[i:i+3] for i in range(len(text)-2))


def distance_at_most_two(a, b):
    if abs(len(a)-len(b)) > 2:
        return None
    previous = {j: j for j in range(min(2, len(b))+1)}
    for i, char in enumerate(a, 1):
        current = {0: i} if i <= 2 else {}
        for j in range(max(1, i-2), min(len(b), i+2)+1):
            current[j] = min(previous.get(j, 3)+1, current.get(j-1, 3)+1,
                             previous.get(j-1, 3)+(char != b[j-1]))
        if not current or min(current.values()) > 2:
            return None
        previous = current
    result = previous.get(len(b), 3)
    return result if result <= 2 else None


class HoldoutIndex:
    def __init__(self, records):
        self.records = defaultdict(list)
        self.postings = defaultdict(list)
        for row in records:
            text = _norm(target_text(row))
            if text:
                self.records[row['lang'], text].append(row['location'])
        for lang, text in self.records:
            if len(text) >= 40:
                for gram, count in grams(text).items():
                    self.postings[lang, gram].append((text, count))

    def matches(self, row):
        lang, text = row['lang'], _norm(target_text(row))
        result = []
        if (lang, text) in self.records:
            result.append({'distance': 0, 'normalized_holdout': text,
                           'holdout_locations': self.records[lang, text]})
        if len(text) < 40:
            return result
        shared = Counter()
        for gram, count in grams(text).items():
            for candidate, other_count in self.postings[lang, gram]:
                if candidate != text and abs(len(candidate)-len(text)) <= 2:
                    shared[candidate] += min(count, other_count)
        for candidate, common in sorted(shared.items()):
            # Two edits destroy at most six trigrams, including repeated trigrams.
            if common < max(len(text), len(candidate))-2-6:
                continue
            distance = distance_at_most_two(text, candidate)
            if distance is not None:
                result.append({'distance': distance, 'normalized_holdout': candidate,
                               'holdout_locations': self.records[lang, candidate]})
        return result


def run(work):
    if work.exists():
        raise FileExistsError(work)
    parent = read(CLEAN / 'plan.json')
    train = CLEAN / 'manifests/current/train.jsonl'
    validation = CLEAN / 'protected_validation.jsonl'
    eval_paths = sorted({Path(p) for p in parent['inputs_sha256']
                         if Path(p).parent.name == 'eval_cases' and p.endswith('.jsonl')})
    if not eval_paths:
        raise ValueError('Missing frozen evaluation exclusions')
    inputs = {str(CLEAN / 'plan.json'): sha256(CLEAN / 'plan.json')}
    for path in [train, validation, *eval_paths]:
        expected = parent['inputs_sha256'][str(path)]
        if sha256(path) != expected:
            raise ValueError('Frozen source changed: ' + str(path))
        inputs[str(path)] = expected
    for directory in (ROOT / 'src', ROOT / 'scripts'):
        for path in directory.rglob('*.py'):
            inputs[str(path)] = sha256(path)
    rows = _read_manifest(train)
    heldout = []
    for path in [validation, *eval_paths]:
        for number, row in enumerate(_read_manifest(path), 1):
            if row.get('lang') in ('id', 'th', 'tl', 'zh', 'en'):
                heldout.append({**row, 'location': {'file': str(path), 'row': number,
                               'text': target_text(row)}})
    index = HoldoutIndex(heldout)
    work.mkdir()
    save_new(work / 'plan.json', {'inputs_sha256': inputs, 'training_rows': len(rows),
        'holdout_unique_texts': len(index.records), 'minimum_near_length': 40, 'max_edits': 2,
        'normalization': 'existing voxft.eval._norm(target_text), per-language, NFC/lowercase/LNM',
        'policy': 'Exact matches at all lengths; nonidentical texts must both have >=40 normalized '
                  'characters and Levenshtein distance <=2. Flag text-leakage risk, not bad audio. '
                  'Exclusion overlay only; frozen source and running main-cleaning policy untouched.',
        'training_started': False, 'production_changed': False})
    started, flagged = time.time(), []
    for number, row in enumerate(rows, 1):
        matches = index.matches(row)
        if matches:
            flagged.append({'row': number, 'sample': row, 'matches': matches,
                'reasons': ['exact_holdout_text' if any(m['distance'] == 0 for m in matches)
                            else 'near_holdout_text_2edits_ge40']})
        if number % 100 == 0:
            write_status(work / 'status.json', {'stage': 'near_text', 'completed': number,
                'total': len(rows), 'flagged_rows': len(flagged), 'time': time.time()})
    for path, digest in inputs.items():
        if sha256(path) != digest:
            raise ValueError('Source changed during near-text audit: ' + path)
    _write_jsonl(flagged, work / 'exclusion_overlay.jsonl')
    save_new(work / 'summary.json', {'training_rows': len(rows), 'flagged_rows': len(flagged),
        'by_lang': dict(Counter(r['sample']['lang'] for r in flagged)),
        'by_reason': dict(Counter(reason for r in flagged for reason in r['reasons'])),
        'elapsed_seconds': time.time()-started, 'inputs_verified': True,
        'candidate_rebuild_pending': bool(flagged), 'training_started': False, 'production_changed': False,
        'note': 'Apply overlay and target/ref origin/PCM linkage after main cleaning, before any new mix. '
                'This is not semantic deduplication; short paraphrases and unknown-language rows are not covered. '
                'Historical scores are unchanged; no new generalization claim.'})
    write_status(work / 'status.json', {'stage': 'complete', 'completed': len(rows), 'total': len(rows),
        'flagged_rows': len(flagged), 'time': time.time(), 'training_started': False, 'production_changed': False})
    print(f'NEAR TEXT complete: {len(flagged)}/{len(rows)} training rows flagged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    runtime_environment(args.work.resolve().parent / 'runtime')
    run(args.work.resolve())
