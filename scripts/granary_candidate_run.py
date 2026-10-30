"""r31: bounded Granary selection on the existing safe pool, without rewriting speech.

Borrowed concepts: Curator's transcript views, hallucination observations and explicit
rejection provenance (Apache-2.0, revision in docs/granary_cleaning_20261030.md).
This does not reproduce NVIDIA's release high_quality classifier.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil

import yaml

import clean_candidate_run as runner
from control_ab_gate import pcm_conflicts
from fresh_baseline_eval import save_new
from full_clean_candidate_run import candidate_inputs, CANDIDATE, CLEAN
from incremental_mix import audit_mix, describe_mix, load_exclusions
from quarantine_music_candidates import verify
from review_training_silence import exclude_flagged
from suspect_candidate_run import fixed_config
from voxft.data.pipeline import _read_manifest, _write_jsonl, dataset_summary
from voxft.eval import _norm
from voxft.train.launcher import preflight
from voxft.train.runlog import _duration_gates, _regressed

DATA = Path('/root/autodl-tmp')
PARENT = DATA / 'full_clean_candidate_20261029/run'
METADATA = DATA / 'granary_pilot_20261030'
LABEL_ROOT = DATA / 'granary_v2_20261029'
REVISION = '81402b751b451b5fe4f4c29e8b40f92892f1e440'
LABEL_HASHES = {
    'id': '7ad795964b840347b022a062eb35d3b60f425a6b1f1ff60aff96b1d339815455',
    'tl': 'a749734d95ece34b3227bf79eaecc17b576fbe123c1aa54b82c595784982f82f',
    'th': 'b729925909837d9aae7cc08ffb97aad423b16d9c7812e177956bb7abfe9cc23e',
}


def source_index(raw, tsv):
    """Recover old sequential names only when the complete source order matches."""
    if any(len(meta) != 7 for meta in tsv):
        raise ValueError('Unexpected FLEURS metadata schema')
    ordered = sorted(tsv, key=lambda row: row[1])
    if (len(raw) != len(ordered) or len({row[1] for row in ordered}) != len(ordered)
            or any(row['text'] != meta[2]
                   or Path(row['audio']).name != f'{i:07d}.wav'
                   for i, (row, meta) in enumerate(zip(raw, ordered)))):
        raise ValueError('Incomplete or reordered original FLEURS metadata')
    return {row['audio']: meta[1] for row, meta in zip(raw, ordered)}


def label_reasons(row, label):
    if label is None:
        return ['missing_granary_label'], []
    # tn_raw is spoken form; itn_raw preserves wording in written form. itn_clean
    # deletes disfluencies, so agreement with it cannot authenticate a TTS target.
    views = [key for key in ('tn_raw', 'itn_raw')
             if isinstance(label.get(key), str) and _norm(label[key])
             and _norm(label[key]) == _norm(row['text'])]
    reasons = []
    if label.get('high_quality') is not True:
        reasons.append('granary_not_high_quality')
    if not views:
        reasons.append('granary_transcript_disagreement')
    return reasons, views


def asr_pattern_reasons(text, reference, lang, duration):
    """Conservative adaptations, not calibrated hallucination/voice-quality truth."""
    if _norm(text) == _norm(reference):
        return []  # A repeated word in both texts may be actual speech.
    words = text.split()
    reasons = []
    if duration > 0 and sum(map(len, words)) / duration > 40:
        reasons.append('asr_character_rate_gt_40')
    # Thai and Chinese whitespace chunks are not words. Short dialogue repetitions
    # also need acoustic evidence; do not flag "no no no" by lexical ratio alone.
    if lang in {'id', 'tl', 'en'} and len(words) >= 8:
        normalized = [_norm(word) for word in words if _norm(word)]
        if normalized and len(set(normalized)) / len(normalized) <= .4:
            reasons.append('asr_repetition_low_diversity')
    return reasons


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    parent = runner.checked(PARENT, trained=True)
    _, inputs = candidate_inputs()
    inputs.update(parent['inputs_sha256'])
    inputs.update(runner.read_json(PARENT / 'trained.json')['inputs_sha256'])
    train = _read_manifest(CANDIDATE / 'candidate/current/train.jsonl')
    validation = _read_manifest(CANDIDATE / 'candidate/current/val.jsonl')
    protected = _read_manifest(CLEAN / 'protected_validation.jsonl')
    acoustic = {row['audio']: row for row in _read_manifest(CLEAN / 'acoustic.jsonl')}
    asr = {row['audio']: row for row in _read_manifest(CLEAN / 'asr.jsonl')}
    new_flags, decisions = [], []
    metadata_paths = [CLEAN / 'asr.jsonl', METADATA / 'mapping_inspection.json']
    for lang, code in (('id', 'id'), ('tl', 'fil'), ('th', 'th')):
        raw_path = DATA / f'voxft_data/raw/fleurs_{lang}/manifest.jsonl'
        tsv_path = METADATA / f'{lang}_train.tsv'
        label_path = LABEL_ROOT / f'{code}/fleurs/{code}_asr.jsonl.gz'
        pcm_path = METADATA / f'{lang}_source_pcm.json'
        archive_path = METADATA / f'{lang}_archive_metadata.json'
        metadata_paths.extend([raw_path, tsv_path, label_path, pcm_path, archive_path])
        if runner.frozen.sha256(label_path) != LABEL_HASHES[lang]:
            raise ValueError('Pinned Granary manifest changed')
        tsv = list(csv.reader(tsv_path.read_text(encoding='utf-8').splitlines(), delimiter='\t', quoting=csv.QUOTE_NONE))
        names = source_index(_read_manifest(raw_path), tsv)
        with gzip.open(label_path, 'rt', encoding='utf-8') as file:
            labels = [json.loads(line) for line in file]
        label_index = {row['audio_filepath']: row for row in labels}
        if len(label_index) != len(labels) or any(row['source_lang'] != code or row['corpus'] != 'fleurs' for row in labels):
            raise ValueError('Granary language/corpus/identity mismatch')
        pcm = runner.read_json(pcm_path)
        archive = runner.read_json(archive_path)['entry']
        records = {row['audio']: row for row in pcm['records']}
        selected = [row for row in train if row['source_id'] == f'fleurs_{lang}']
        if (pcm['archive_sha_verified'] is not True or pcm['archive_sha256'] != archive['lfs']['oid']
                or pcm['archive_bytes'] != archive['size'] or len(records) != len(pcm['records'])
                or set(records) != {row['audio'] for row in selected}):
            raise ValueError('Incomplete source PCM verification')
        for row in selected:
            name = names[row['origin_audio']]
            record = records[row['audio']]
            if (record['original_filename'] != name or record['origin_audio'] != row['origin_audio']
                    or record['processed_file_sha256'] != acoustic[row['audio']]['file_sha256']):
                raise ValueError('Source mapping or audio changed')
            label = label_index.get(f'data_{code}_fleurs_train_{name.removesuffix(".wav")}.opus')
            reasons, views = label_reasons(row, label)
            if record['matched'] is not True:
                reasons.append('source_pcm_not_exact_crop')
            observed = asr[row['audio']]
            if observed['file_sha256'] != record['processed_file_sha256']:
                raise ValueError('ASR belongs to another audio')
            patterns = asr_pattern_reasons(observed['result']['hyp'], row['text'], lang, row['duration'])
            reasons += patterns
            decisions.append({'audio':row['audio'], 'source_id':row['source_id'], 'original_filename':name,
                'granary_identifier':label['audio_filepath'] if label else None,
                'high_quality':label.get('high_quality') if label else None,
                'matching_views':views, 'reasons':reasons, 'asr_pattern_reasons':patterns,
                'status':'excluded_uncertain' if reasons else 'retained_not_quality_certified'})
            if reasons:
                new_flags.append({'audio':row['audio'], 'pcm_sha256':acoustic[row['audio']]['pcm_sha256'],
                    'uses':[{'origin_audio':row['origin_audio']}], 'reasons':reasons, 'status':'uncertain'})
    positions, removed = exclude_flagged(train, new_flags, acoustic)
    retained = [train[position-1] for position in positions]
    excluded, eval_inputs = load_exclusions(runner.frozen.ROOT / 'eval_cases')
    inputs.update(eval_inputs)
    audit = audit_mix(retained, protected, excluded, max_dur=30)
    pcm = pcm_conflicts(retained, protected, acoustic)
    historical_flags = runner.read_json(CANDIDATE / 'flagged.json')
    if (not removed or audit['violations'] or audit['origins_above_3x'] or any(pcm.values())
            or exclude_flagged(retained, historical_flags + new_flags, acoustic)[1]):
        raise ValueError('No intervention or isolation failure')
    metadata_paths.extend([PARENT / 'plan.json', PARENT / 'trained.json',
                           runner.frozen.ROOT / 'docs/qc_gates.md',
                           runner.frozen.ROOT / 'docs/granary_cleaning_20261030.md'])
    for root in (runner.frozen.ROOT / 'src', runner.frozen.ROOT / 'scripts',
                 runner.frozen.VOXCPM_REPO / 'src', runner.frozen.VOXCPM_REPO / 'scripts'):
        metadata_paths.extend(path for path in root.rglob('*') if path.suffix in {'.py', '.sh'})
    inputs.update({str(path):runner.frozen.sha256(path) for path in metadata_paths})
    verify(inputs)
    # r30 consumed about 2.6 GiB for training + 900 evaluations; retain the 4 GiB reserve.
    if shutil.disk_usage(work.parent).free < 7 * 1024**3:
        raise RuntimeError('Need 7 GiB for the bounded run and existing 4 GiB reserve')
    work.mkdir(parents=True)
    lines = (CANDIDATE / 'candidate/current/train.jsonl').read_bytes().splitlines(keepends=True)
    (work / 'train.jsonl').write_bytes(b''.join(lines[position-1] for position in positions))
    shutil.copyfile(CANDIDATE / 'candidate/current/val.jsonl', work / 'val.jsonl')
    for name in ('cases.jsonl', *(f'{model}_report.json' for model in parent['targets'])):
        shutil.copyfile(PARENT / name, work / name)
    _write_jsonl(decisions, work / 'granary_decisions.jsonl')
    _write_jsonl(removed, work / 'excluded.jsonl')
    save_new(work / 'flagged.json', historical_flags + new_flags)
    config = fixed_config(PARENT / 'train.yaml', work)
    (work / 'train.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    issues = preflight(work / 'train.yaml',gpus=1)
    Path(config['save_path']).rmdir()
    if any(not issue.startswith('警告') for issue in issues):
        raise ValueError(f'Training preflight failed: {issues}')
    summary, shares = describe_mix(retained)
    for lang in shares:
        shares[lang]['requested'] = parent['language_shares'][lang]['actual']
    sources = {source: {'license':parent['sources'][source]['license'],
                       **dataset_summary([r for r in retained if r['source_id']==source])}
               for source in sorted({r['source_id'] for r in retained})}
    save_new(work / 'mix.json', {'train':summary,'language_shares':shares, 'sources':sources,
        'training_approved':True,'protected_validation_rows':len(protected),
        'granary_revision':REVISION,'granary_label_license':'OpenMDW-1.1',
        'retained_rows_byte_identical':True,'new_audio_rows':0,'relabelled_rows':0,
        'excluded_rows':len(removed),'validation':dataset_summary(validation)})
    inputs.update({str(path):runner.frozen.sha256(path) for path in work.iterdir() if path.is_file()})
    plan = {**parent, 'candidate_model':'r31','run_name':'lora_joint3_r31_granary_consensus_459',
        'targets':{**parent['targets'],'r31':str(work/'checkpoint/latest')},
        'config':str(work/'train.yaml'),'train':summary,'language_shares':shares,'sources':sources,
        'epochs':459*16/len(retained),'inputs_sha256':inputs,'audit':audit,'pcm':pcm,'preflight':issues,
        'tail_policy':'absolute','additional_comparisons':[['r31','r30'],['r31','r29']],
        'authorization':'User requested trying Granary-v2, borrowing cleaning logic, and relaxing tail silence.',
        'excluded_rows_from_r30':len(removed), 'excluded_rows_from_r29':7252-len(retained), 'granary_revision':REVISION,
        'note':'Same base/seed/459 optimizer steps; only data selection changes. Granary HQ plus '
               'agreement with tn_raw or wording-preserving itn_raw; no itn_clean, rewrite, recovery or new audio. '
               'Full original archive hashes and exact PCM crops checked. Flags are uncertain, not certified bad speech. '
               'Language shares/exposure change with filtering: not an isolated test of any one quality rule. '
               'Granary text postprocessing may use source reference: agreement is not independent ASR confirmation. '
               'Validation unchanged; full historical validation protected. No acoustic/native certification/promotion.'}
    save_new(work / 'plan.json',plan)
    runner.checked(work)
    print(json.dumps({'train':summary,'language_shares':shares,'excluded':len(removed),'preflight':issues},ensure_ascii=False),flush=True)


def run(work):
    runner.run(work)
    reports = [runner.read_json(work/f'{name}_report.json') for name in ('base','r30','r31')]
    save_new(work/'tail_policy_assessment.json', {
        'tail_policy':'absolute','duration':_duration_gates(reports,tail_policy='absolute'),
        'cer':_regressed(reports),'automatic_promotion':False,
        'inputs_sha256':{str(work/f'{name}_report.json'):runner.frozen.sha256(work/f'{name}_report.json')
                         for name in ('base','r30','r31')}})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--work',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.action](args.work.resolve())
