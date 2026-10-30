"""Reserve clean unseen-local texts without running or inspecting model outcomes."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import unicodedata

from audit_near_text import HoldoutIndex
from control_ab_gate import build_cases
from holdout_texts import REVISION, parse_tsv
from incremental_mix import target_text
import r31_failure_probe as prior

DATA = Path('/root/autodl-tmp')
TSVS = {
    'id': ('id_id', '824cad6367283d95f545911eec051c488c182baa13a63227c6330af3c8f80741'),
    'th': ('th_th', '8db09d12ac384654f6677e141f6e55bf5bc42028160cbc0d2d9f7fac4fe25271'),
    'tl': ('fil_ph', '3e23f5be5ebf6ae133fb2f2c3a9f4ef21498fc2de0ce9161b02b94b3f2bb553b'),
}


def input_issue(row):
    text = row['text']
    if any(c in text for c in ('Ã', 'Â', 'Å', 'Ä', '\ufffd')):
        return 'encoding_suspect'
    if any(c in text for c in '()（）[]{}'):
        return 'body_control_like_punctuation'
    if any(unicodedata.category(c).startswith('C') for c in text):
        return 'control_or_unassigned_unicode'
    if prior.frozen.evaluation._is_numeric(row):
        return 'numeric'
    if not 10 <= len(text) <= 300:
        return 'length'
    return None


def select(rows, excluded, lang, count=30):
    eligible, reasons = {}, Counter()
    norm = prior.frozen.evaluation._norm
    for row in rows:
        key = norm(row['text'])
        reason = input_issue(row) or ('known_text' if key in excluded else None)
        if reason:
            reasons[reason] += 1
        else:
            eligible.setdefault(key, row)
    ordered = sorted(eligible.values(), key=lambda row: hashlib.sha256(
        f"final_20261030:{lang}:{row['source_sentence_id']}:{row['text']}".encode()).hexdigest())
    if len(ordered) < count:
        raise ValueError(f'Insufficient clean unseen texts for {lang}: {len(ordered)}')
    return ordered[:count], {'eligible':len(ordered), 'excluded':dict(reasons)}


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    frozen, norm = prior.frozen, prior.frozen.evaluation._norm
    paths = set((DATA/'voxft_data/processed').glob('*/train.jsonl'))
    paths |= set((DATA/'voxft_data/processed').glob('*/val.jsonl'))
    paths |= set((DATA/'voxft_data/raw').glob('*/manifest.jsonl'))
    paths |= set((Path('/root/voxcpm-finetune')/'eval_cases').glob('*.jsonl'))
    paths |= set((frozen.ROOT/'eval_cases').glob('*.jsonl'))
    for pattern in ('*/run/train.jsonl', '*/run/val.jsonl', '*/run/cases.jsonl', '*/cases.jsonl'):
        paths |= set(DATA.glob(pattern))
    if prior.SOURCE/'train.jsonl' not in paths or not paths:
        raise ValueError('Missing current corpus')
    inputs, known, excluded = {}, [], set()
    for path in sorted(paths):
        inputs[str(path)] = frozen.sha256(path)
        for number, row in enumerate(frozen._read_manifest(path), 1):
            if row.get('text'):
                excluded.add(norm(target_text(row)))
                if row.get('lang') in TSVS:
                    known.append({**row, 'location': {'file':str(path), 'row':number}})
    reports = sorted((DATA/'voxft_ckpt/eval').glob('*.json'))
    for path in reports:
        inputs[str(path)] = frozen.sha256(path)
        for number, row in enumerate(prior.read(path).get('items', []), 1):
            if row.get('text'):
                excluded.add(norm(target_text(row)))
                if row.get('lang') in TSVS:
                    known.append({**row, 'location': {'file':str(path), 'row':number}})
    prior.review.checked(prior.SOURCE/'plan.json', prior.SOURCE_PLAN_SHA)
    parent = prior.read(prior.SOURCE/'plan.json')
    inputs[str(prior.SOURCE/'plan.json')] = prior.SOURCE_PLAN_SHA
    old_cases = frozen._read_manifest(prior.SOURCE/'cases.jsonl')
    references = {lang: sorted({r['ref_audio'] for r in old_cases if r['ref_lang']==lang}) for lang in ('zh','en')}
    for paths2 in references.values():
        for path in paths2:
            inputs[path] = parent['inputs_sha256'][path]
            prior.review.checked(path, inputs[path])
    selected, counts = {}, {}
    for lang, (config, digest) in TSVS.items():
        path = DATA/'hf_home/hub/datasets--google--fleurs/snapshots'/REVISION/'data'/config/'test.tsv'
        prior.review.checked(path, digest)
        inputs[str(path)] = digest
        pool = parse_tsv(path.read_text(encoding='utf-8'))
        # Near-duplicate audit indexes only 30 candidates, avoiding a large corpus index.
        rejected = set()
        for _ in range(10):
            chosen, detail = select(pool, excluded | rejected, lang)
            index = HoldoutIndex([{**r, 'lang':lang, 'location':r['source_sentence_id']} for r in chosen])
            conflicts = {match['normalized_holdout'] for row in known if row['lang']==lang
                         for match in index.matches(row)}
            if not conflicts:
                break
            rejected |= conflicts
        else:
            raise ValueError('Repeated near-overlap; no automatic relaxation')
        selected[lang], counts[lang] = chosen, {**detail, 'near_rejected':len(rejected)}
    paired = build_cases(selected, references)
    positions = {(lang,r['source_sentence_id']):i for lang, rows in selected.items() for i,r in enumerate(rows)}
    cases = []
    for row in paired:
        condition = 'control' if positions[row['lang'],row['source_sentence_id']] % 2 else 'plain'
        if row['condition'] != condition:
            continue
        sentence = row['sentence_id'].replace('fresh_', 'reserved_20261030_', 1)
        cases.append({**row, 'sentence_id':sentence, 'case_id':sentence+'_'+condition,
            'cohort':'reserved_final_20261030', 'evaluation_use':'once_after_candidate_selection',
            'note':'Clean nonnumeric read-speech text reservation; no generated outcomes inspected. '
                   'No correction of source wording. Exact and >=40-char edit-distance<=2 local exclusion; '
                   'semantic/pretraining/speaker novelty and native quality remain unverified. '
                   'Existing stress/short/numeric development tests remain separate.'})
    if len(cases) != 90 or len({r['case_id'] for r in cases}) != 90:
        raise ValueError('Invalid reservation')
    for path in (Path(__file__),):
        inputs[str(path)] = frozen.sha256(path)
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    work.mkdir()
    from voxft.data.pipeline import _write_jsonl
    _write_jsonl(cases, work/'cases.jsonl')
    prior.save(work/'plan.json', {'inputs_sha256':inputs, 'cases_sha256':frozen.sha256(work/'cases.jsonl'),
        'by_lang':counts, 'texts_per_lang':30, 'conditions_per_lang':{'plain':15,'control':15},
        'seeds':frozen.SEEDS, 'outputs_per_model':450, 'base_required':True, 'evaluated':False,
        'known_manifest_count':len(paths), 'known_report_count':len(reports), 'known_normalized_texts':len(excluded),
        'policy':'Reserve against future training; do not tune on these scores. No source audio download, '
                 'model inference, expected-ASR prompt, production change or automatic pass claim.'})
    print('RESERVED 90 texts, 450 outputs/model only after candidate selection; no inference', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    prepare(parser.parse_args().work)
