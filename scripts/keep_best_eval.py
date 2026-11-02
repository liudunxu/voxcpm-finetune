"""Compare fixed-budget QC keep-best selection without generating new audio."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from voxft.eval import _norm, _report_metrics

SEEDS = (42, 43, 44, 45)
MATCH_FIELDS = ('text', 'lang', 'ref_audio', 'ref_lang', 'control')


def near_silence(row):
    return (row.get('rms_dbfs') is not None and row['rms_dbfs'] < -60
            and (row.get('speech_ratio') or 0) < .05)


def severe(row):
    return (row['cer'] > .20 or row.get('over_read', False)
            or not _norm(row.get('hyp', '')) or near_silence(row))


def rank(row):
    return (not _norm(row.get('hyp', '')), near_silence(row),
            row.get('over_read', False), row['cer'], row['seed'])


def metrics(rows):
    has_f0 = all(row.get('f0_std_st') is not None for row in rows)
    report = _report_metrics(rows if has_f0 else [{**row, 'f0_std_st': 0} for row in rows])
    if not has_f0:
        report.pop('mean_f0_std')
    return report


def selected(rows, seeds):
    groups = defaultdict(dict)
    for row in rows:
        if row['seed'] not in seeds:
            continue
        if row['seed'] in groups[row['case_id']]:
            raise ValueError(f"Duplicate {row['case_id']}/seed{row['seed']}")
        groups[row['case_id']][row['seed']] = row
    incomplete = [case for case, items in groups.items() if set(items) != set(seeds)]
    if incomplete:
        raise ValueError(f'Incomplete fixed seed budget: {incomplete[:3]}')
    return {case: min((items[seed] for seed in seeds), key=rank)
            for case, items in groups.items()}


def compare(base_rows, candidate_rows, seeds=SEEDS):
    seeds = tuple(seeds)
    base = selected(base_rows, seeds)
    candidate = selected(candidate_rows, seeds)
    if set(base) != set(candidate) or not base:
        raise ValueError('Base and candidate case sets differ')
    for case in base:
        for seed in seeds:
            old = next(row for row in base_rows if row['case_id'] == case and row['seed'] == seed)
            new = next(row for row in candidate_rows if row['case_id'] == case and row['seed'] == seed)
            if any(old.get(field) != new.get(field) for field in MATCH_FIELDS):
                raise ValueError(f'Input mismatch: {case}/seed{seed}')
    old_rows = [base[case] for case in sorted(base)]
    new_rows = [candidate[case] for case in sorted(candidate)]
    selections = []
    for old, new in zip(old_rows, new_rows):
        selections.append({'case_id': new['case_id'], 'lang': new['lang'],
            'base_seed': old['seed'], 'candidate_seed': new['seed'],
            'base_cer': old['cer'], 'candidate_cer': new['cer'],
            'base_severe': severe(old), 'candidate_severe': severe(new),
            'base_wav': old['wav'], 'candidate_wav': new['wav']})
    by_lang = {}
    for lang in ('th', 'tl', 'id'):
        rows = [row for row in selections if row['lang'] == lang]
        by_lang[lang] = {'cases': len(rows),
            'base_severe_cases': sum(row['base_severe'] for row in rows),
            'candidate_severe_cases': sum(row['candidate_severe'] for row in rows),
            'cer_regressions_over_0_05': sum(
                row['candidate_cer'] - row['base_cer'] > .05 for row in rows)}
    return {'selection_policy': {'seed_order': list(seeds),
            'rank': 'nonempty ASR, not near-silent, not over-read, then minimum CER; seed breaks ties',
            'severe': 'CER>0.20 or over-read or empty ASR or near-silence',
            'generation_retry_badcase': False, 'selection_is_post_hoc_qc': True},
        'cases': len(selections), 'metrics': {'base': metrics(old_rows),
            'candidate': metrics(new_rows)},
        'by_lang': by_lang, 'selections': selections,
        'native_quality': 'unverified', 'production_changed': False}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    base_doc = json.loads(args.base.read_text(encoding='utf-8'))
    candidate_doc = json.loads(args.candidate.read_text(encoding='utf-8'))
    result = compare(base_doc['items'], candidate_doc['items'], args.seeds)
    result.update(base_report=str(args.base), candidate_report=str(args.candidate),
        base_report_sha256=sha256(args.base), candidate_report_sha256=sha256(args.candidate))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'cases': result['cases'], 'by_lang': result['by_lang'],
        'metrics': {name: value['by_lang'] for name, value in result['metrics'].items()}},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
