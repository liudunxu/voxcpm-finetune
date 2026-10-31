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


def supplement_cases(selected, old_cases):
    """Keep the two encoding-risk slots' conditions, while retaining their old cases separately."""
    slots = sorted({r['sentence_id'] for r in old_cases if r['lang'] == 'id'
                    and input_issue(r) == 'encoding_suspect'})
    if len(slots) != len(selected) or len(slots) != 2:
        raise ValueError('Expected two encoding-risk ID texts and two clean supplements')
    if (any(input_issue(r) for r in selected)
            or len({prior.frozen.evaluation._norm(r['text']) for r in selected}) != 2):
        raise ValueError('Supplements must be two distinct clean texts')
    cases = []
    for slot, row in zip(slots, selected):
        pair = [r for r in old_cases if r['sentence_id'] == slot]
        if len(pair) != 2 or {r['condition'] for r in pair} != {'plain', 'control'}:
            raise ValueError('Expected both original conditions for each ID slot')
        sentence = f"supplement_20261031_id_{row['source_sentence_id']}"
        for old in pair:
            cases.append({**old, **row, 'sentence_id':sentence,
                'case_id':sentence+'_'+old['condition'], 'cohort':'development_id_supplement_20261031',
                'evaluation_use':'development_only', 'conditioning_slot':old['case_id'],
                'note':'Clean text supplement for development coverage, using the old slot conditioning. '
                       'Original encoding-risk case remains a separate stress case, unchanged. '
                       'No source wording correction or generated outcomes used for selection; '
                       'semantic/pretraining/speaker novelty and native quality unverified.'})
    return cases


def prepare(work, *, id_supplement=False):
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
    if id_supplement:
        reserved = frozen.ROOT/'eval_cases/reserved_final_20261030.jsonl'
        prior.review.checked(reserved, 'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c')
        paths.add(reserved)
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
    for lang, (config, digest) in ({'id':TSVS['id']} if id_supplement else TSVS).items():
        path = DATA/'hf_home/hub/datasets--google--fleurs/snapshots'/REVISION/'data'/config/'test.tsv'
        prior.review.checked(path, digest)
        inputs[str(path)] = digest
        pool = parse_tsv(path.read_text(encoding='utf-8'))
        # Near-duplicate audit indexes only 30 candidates, avoiding a large corpus index.
        rejected = set()
        for _ in range(10):
            chosen, detail = select(pool, excluded | rejected, lang, count=2 if id_supplement else 30)
            index = HoldoutIndex([{**r, 'lang':lang, 'location':r['source_sentence_id']} for r in chosen])
            conflicts = {match['normalized_holdout'] for row in known if row['lang']==lang
                         for match in index.matches(row)}
            if not conflicts:
                break
            rejected |= conflicts
        else:
            raise ValueError('Repeated near-overlap; no automatic relaxation')
        selected[lang], counts[lang] = chosen, {**detail, 'near_rejected':len(rejected)}
    paired = [] if id_supplement else build_cases(selected, references)
    positions = {(lang,r['source_sentence_id']):i for lang, rows in selected.items() for i,r in enumerate(rows)}
    cases = supplement_cases(selected['id'], old_cases) if id_supplement else []
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
    expected = 4 if id_supplement else 90
    if len(cases) != expected or len({r['case_id'] for r in cases}) != expected:
        raise ValueError('Invalid reservation')
    for path in (Path(__file__),):
        inputs[str(path)] = frozen.sha256(path)
    for path, digest in inputs.items():
        prior.review.checked(path, digest)
    work.mkdir()
    from voxft.data.pipeline import _write_jsonl
    _write_jsonl(cases, work/'cases.jsonl')
    prior.save(work/'plan.json', {'inputs_sha256':inputs, 'cases_sha256':frozen.sha256(work/'cases.jsonl'),
        'by_lang':counts, 'texts_per_lang':({'id':2} if id_supplement else 30),
        'conditions_per_lang':{'plain':2,'control':2} if id_supplement else {'plain':15,'control':15},
        'seeds':frozen.SEEDS, 'outputs_per_model':expected*len(frozen.SEEDS), 'base_required':True, 'evaluated':False,
        'known_manifest_count':len(paths), 'known_report_count':len(reports), 'known_normalized_texts':len(excluded),
        'policy':('Development supplement, protect from training; original cases and final90 unchanged. '
                  if id_supplement else 'Reserve against future training; do not tune on these scores. ') +
                 'No source audio download, '
                 'model inference, expected-ASR prompt, production change or automatic pass claim.'})
    print('RESERVED 2 ID development texts, 20 outputs/model; no inference' if id_supplement else
          'RESERVED 90 texts, 450 outputs/model only after candidate selection; no inference', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--id-supplement', action='store_true',
                        help='Freeze two new ID development texts; exclude the existing final90')
    args = parser.parse_args()
    prepare(args.work, id_supplement=args.id_supplement)
