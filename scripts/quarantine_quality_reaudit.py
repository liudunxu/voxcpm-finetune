"""Verify the frozen music crosscheck and quarantine its unresolved training suspects."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import soundfile as sf

from control_ab_gate import pcm_conflicts
from incremental_mix import audit_mix, describe_mix, load_exclusions
from quality_music_crosscheck import ROOT, MUSIC, read, manifest, sha, save, prompt_margins, windows
from quarantine_music_candidates import verify
from review_training_silence import exclude_flagged
from voxft.paths import ROOT as SOURCE_ROOT

CURRENT = Path('/root/autodl-tmp/granary_pilot_20261030/run')
CLEAN = Path('/root/autodl-tmp/full_corpus_clean_20261029/parallel8/run')


def verify_result(result, case, frames, rate):
    if result['case'] != case or rate != 16000:
        raise ValueError('Crosscheck case or source rate changed')
    expected = windows(frames, rate)
    if [(w['start_sample'], w['end_sample']) for w in result['windows']] != expected:
        raise ValueError('Crosscheck does not cover the frozen source windows')
    for window in result['windows']:
        calculated = prompt_margins(window['cosine_scores'])
        if any(window[key] != value for key, value in calculated.items()):
            raise ValueError('Prompt contrast aggregation differs')
    positive = any(w['all_three_music_higher'] for w in result['windows'])
    if result['crossmodel_review_priority'] != (case['music_score'] >= .5 and positive):
        raise ValueError('Cross-model aggregation differs')


def suspect_reason(result):
    """Conservative quarantine of this finite reviewed set, not a calibrated general gate."""
    if result['case']['role'] == 'previously_excluded_control':
        return None
    positive = any(w['all_three_music_higher'] for w in result['windows'])
    if result['case']['music_score'] >= .5:
        return 'cross_model_music_suspect' if positive else 'ast_music_unresolved'
    return 'clap_music_unresolved' if positive else None


def write_subset(path, output, positions):
    lines = [line for line in path.read_bytes().splitlines(keepends=True) if line.strip()]
    if len(lines) != len(manifest(path)):
        raise ValueError('Manifest line accounting differs')
    with output.open('xb') as stream:
        stream.write(b''.join(lines[i-1] for i in positions))


def ordered_results(directory, start, stop):
    paths = sorted(directory.glob('*.json'), key=lambda p:int(p.stem))
    if [p.name for p in paths] != [f'{i:03d}.json' for i in range(start, stop)]:
        raise ValueError('Missing, duplicate or unmapped fresh result')
    return paths


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    cross = ROOT/'clap_all'
    if sha(cross/'plan.json') != '5abcfab35dff1d2ac15d0603f687f74684d7d5b15951ea8e91861c90de43ec02':
        raise ValueError('Wrong expanded crosscheck')
    inputs = read(ROOT/'plan.json')['inputs_sha256'].copy()
    inputs.update(read(cross/'plan.json')['inputs_sha256'])
    for p in (ROOT/'audit.py', ROOT/'plan.json', ROOT/'rows.jsonl', ROOT/'summary.json',
              cross/'plan.json', cross/'summary.json', CURRENT/'plan.json'):
        inputs[str(p)] = sha(p)
    parent = read(CURRENT/'plan.json')
    for name in ('acoustic.jsonl', 'tasks.jsonl', 'protected_validation.jsonl'):
        p = CLEAN/name
        inputs[str(p)] = parent['inputs_sha256'][str(p)]
    verify(inputs)
    cases = read(cross/'cases.json')
    reused = read(cross/'reused.json')
    paths = ordered_results(cross/'items', 176, 1087)
    if len(reused) != 176 or len(paths) != 911 or len(cases) != 1087:
        raise ValueError('Incomplete or unexpected crosscheck outputs')
    fresh = []
    for i, p in enumerate(paths, 176):
        if p.name != f'{i:03d}.json':
            raise ValueError('Missing, duplicate or unmapped fresh result')
        fresh.append(read(p))
        inputs[str(p)] = sha(p)
    results = reused + fresh
    groups = {}
    for result, case in zip(results, cases):
        info = sf.info(case['audio'])
        if info.channels != 1:
            raise ValueError('Unexpected channel count')
        verify_result(result, case, info.frames, info.samplerate)
        group = groups.setdefault(case['role'], {'n':0, 'all_three_music_higher':0,
                                                 'crossmodel_review_priority':0})
        group['n'] += 1
        group['all_three_music_higher'] += any(w['all_three_music_higher'] for w in result['windows'])
        group['crossmodel_review_priority'] += result['crossmodel_review_priority']
    summary = read(cross/'summary.json')
    if (summary['by_role'] != groups or summary['completed'] != 1087
            or summary['reused_outputs'] != 176 or summary['new_outputs'] != 911):
        raise ValueError('Crosscheck summary differs from raw scores')
    rows = manifest(CURRENT/'train.jsonl')
    current_by_audio = {row['audio']: row for row in rows}
    acoustic = {row['audio']:row for row in manifest(CLEAN/'acoustic.jsonl')}
    tasks = {row['audio']:row for row in manifest(CLEAN/'tasks.jsonl')}
    flags = []
    for result in results:
        reason = suspect_reason(result)
        if not reason:
            continue
        case = result['case']
        path = case['audio']
        measured, task = acoustic[path], tasks[path]
        if (path not in current_by_audio or not task['origins']
                or any(case[k] != measured[k] for k in ('file_sha256', 'pcm_sha256'))
                or task['file_sha256'] != case['file_sha256']):
            raise ValueError('Music evidence differs from current source')
        flags.append({'audio':path, 'file_sha256':case['file_sha256'],
            'pcm_sha256':case['pcm_sha256'], 'uses':[{'origin_audio':x} for x in task['origins']],
            'source_id':current_by_audio[path]['source_id'], 'reasons':[reason],
            'ast_music':case['music_score'], 'clap_windows':result['windows'],
            'status':'uncertain_conservatively_quarantined_not_confirmed_music'})
    counts = Counter(flag['reasons'][0] for flag in flags)
    if counts != {'cross_model_music_suspect':926, 'ast_music_unresolved':105, 'clap_music_unresolved':7}:
        raise ValueError('Reviewed finite suspect set changed')
    historical = read(CURRENT/'flagged.json')
    all_flags = historical + flags
    positions, removed = exclude_flagged(rows, all_flags, acoustic)
    retained = [rows[i-1] for i in positions]
    protected = manifest(CLEAN/'protected_validation.jsonl')
    validation = manifest(CURRENT/'val.jsonl')
    exclusions, eval_inputs = load_exclusions()
    inputs.update(eval_inputs)
    audit = audit_mix(retained, protected, exclusions, max_dur=30)
    pcm = pcm_conflicts(retained, protected, acoustic)
    if (audit['violations'] or audit['origins_above_3x'] or any(pcm.values())
            or exclude_flagged(retained + validation, all_flags, acoustic)[1]):
        raise ValueError('Historical exclusions or validation protection failed')
    # Explicitly account for every currently used ref, not just the target-only CLAP scope.
    refs = Counter(row['ref_audio'] for row in rows if row.get('ref_audio'))
    music_cases = manifest(MUSIC/'cases.jsonl')
    ref_scores = []
    for i, case in enumerate(music_cases):
        if case['audio'] not in refs:
            continue
        p = MUSIC/'items'/f'{i:05d}.json'
        if str(p) not in inputs or sha(p) != inputs[str(p)]:
            raise ValueError('Reference music evidence was not frozen')
        item = read(p)
        if item['case'] != case or case['file_sha256'] != inputs[case['audio']]:
            raise ValueError('Reference audio changed')
        ref_scores.append({'audio':case['audio'], 'uses':refs[case['audio']], 'scores':item['scores']})
    if {r['audio'] for r in ref_scores} != set(refs):
        raise ValueError('Missing current reference coverage')
    for root in (SOURCE_ROOT/'scripts', SOURCE_ROOT/'src'):
        inputs.update({str(p):sha(p) for p in root.rglob('*.py')})
    verify(inputs)
    if shutil.disk_usage(work.parent).free < 4.1*1024**3:
        raise RuntimeError('Retain 4 GiB disk reserve')
    work.mkdir()
    write_subset(CURRENT/'train.jsonl', work/'train.jsonl', positions)
    shutil.copyfile(CURRENT/'val.jsonl', work/'val.jsonl')
    save(work/'new_flags.json', flags)
    save(work/'flagged.json', all_flags)
    save(work/'excluded.json', removed)
    train, shares = describe_mix(retained)
    sources = {source:{'license':parent['sources'][source]['license'],
                      **describe_mix([r for r in retained if r['source_id']==source])[0]}
               for source in sorted({r['source_id'] for r in retained})}
    save(work/'mix.json', {'train':train, 'language_shares':shares, 'sources':sources,
        'before_rows':len(rows), 'excluded_rows':len(removed), 'direct_suspect_files':len(flags),
        'reason_counts':dict(counts), 'protected_validation_rows':len(protected),
        'validation_rows':len(validation), 'retained_rows_byte_identical':True,
        'new_audio_rows':0, 'relabelled_rows':0, 'training_started':False,
        'text_consistency_review_pending':True})
    save(work/'reference_coverage.json', {'uses':sum(refs.values()), 'unique_files':len(refs),
        'ast_music_ge_05':sum(r['scores']['Music']>=.5 for r in ref_scores),
        'max_ast_music':max(r['scores']['Music'] for r in ref_scores), 'items':ref_scores,
        'note':'Existing AST coverage only; target CLAP does not certify all refs.'})
    save(work/'verification.json', {'crosscheck_outputs':1087, 'reused':176, 'new':911,
        'raw_scores_aggregation_verified':True, 'audio_windows_verified':True,
        'audit':audit, 'pcm':pcm, 'original_validation_unchanged':
            sha(CURRENT/'val.jsonl')==sha(work/'val.jsonl'),
        'outputs_sha256':{str(p):sha(p) for p in work.iterdir() if p.is_file()}})
    inputs.update({str(p):sha(p) for p in work.iterdir() if p.is_file()})
    save(work/'plan.json', {'inputs_sha256':inputs, 'parent':str(CURRENT),
        'training_started':False, 'source_mutations':False, 'production_changed':False,
        'authorization':'User requires confirmed and suspected problem data excluded, and autonomous quality iteration.',
        'policy':'Finite conservative quarantine: 926 AST+CLAP suspects, 105 unresolved elevated AST cases, '
                 '7 CLAP-only control cases. Independent models are not independent truth and no music probability '
                 'or native quality is certified. Preserve original bytes and full historical exclusion/val protection. '
                 'No automatic training or admission of low-score data; transcript review remains pending.'})
    verify(inputs)
    print({'retained_rows':len(retained), 'hours':train['hours'], 'excluded':len(removed),
           'flags':dict(counts), 'refs':len(refs), 'plan_sha256':sha(work/'plan.json')}, flush=True)


def quarantine_reference(work):
    if work.exists():raise FileExistsError(work)
    parent=ROOT/'music_candidate'
    review=ROOT/'clap_reference'
    if (sha(parent/'plan.json')!='1128ca0d327abf365f2d829d72094463574187c90243bb7531c8a1d06e225f45'
            or sha(review/'plan.json')!='58eab68d9ce9b1301bcf80edc96dd719ba52eb5b42264c7f9bbd9a402c6957b3'):
        raise ValueError('Wrong frozen reference review')
    inputs=read(parent/'plan.json')['inputs_sha256'].copy()
    inputs.update(read(review/'plan.json')['inputs_sha256'])
    for p in (parent/'plan.json',review/'plan.json',review/'summary.json',review/'items/000.json',Path(__file__)):
        inputs[str(p)]=sha(p)
    verify(inputs)
    case=read(review/'cases.json')[0]
    item=read(review/'items/000.json')
    info=sf.info(case['audio'])
    verify_result(item,case,info.frames,info.samplerate)
    if not item['crossmodel_review_priority'] or read(review/'summary.json')['completed']!=1:
        raise ValueError('Frozen reference corroboration differs')
    acoustic={r['audio']:r for r in manifest(CLEAN/'acoustic.jsonl')}
    tasks={r['audio']:r for r in manifest(CLEAN/'tasks.jsonl')}
    if any(case[k]!=acoustic[case['audio']][k] for k in ('file_sha256','pcm_sha256')):
        raise ValueError('Reference evidence differs from acoustic audit')
    flag={**case,'uses':[{'origin_audio':origin} for origin in tasks[case['audio']]['origins']],
          'reasons':['cross_model_reference_music_suspect'],
          'status':'uncertain_conservatively_quarantined_not_confirmed_music'}
    flags=read(parent/'flagged.json')+[flag]
    rows=manifest(parent/'train.jsonl')
    positions,removed=exclude_flagged(rows,flags,acoustic)
    retained=[rows[i-1] for i in positions]
    protected=manifest(CLEAN/'protected_validation.jsonl')
    validation=manifest(parent/'val.jsonl')
    exclusions,eval_inputs=load_exclusions()
    inputs.update(eval_inputs)
    audit=audit_mix(retained,protected,exclusions,max_dur=30)
    pcm=pcm_conflicts(retained,protected,acoustic)
    if (len(removed)!=1 or audit['violations'] or audit['origins_above_3x'] or any(pcm.values())
            or exclude_flagged(retained+validation,flags,acoustic)[1]):
        raise ValueError('Unexpected reference propagation or isolation failure')
    work.mkdir()
    write_subset(parent/'train.jsonl',work/'train.jsonl',positions)
    shutil.copyfile(parent/'val.jsonl',work/'val.jsonl')
    save(work/'flagged.json',flags)
    save(work/'new_flags.json',[flag])
    save(work/'excluded.json',removed)
    train,shares=describe_mix(retained)
    previous=read(parent/'mix.json')
    sources={source:{'license':previous['sources'][source]['license'],
                     **describe_mix([r for r in retained if r['source_id']==source])[0]}
             for source in sorted({r['source_id'] for r in retained})}
    save(work/'mix.json',{**previous,'train':train,'language_shares':shares,'sources':sources,
        'before_rows':len(rows),'excluded_rows':len(removed),'cumulative_excluded_from_5015':5015-len(retained),
        'direct_suspect_files':1,'reason_counts':{'cross_model_reference_music_suspect':1},
        'text_consistency_review_pending':True,'training_started':False})
    save(work/'verification.json',{'audit':audit,'pcm':pcm,'reference_result_verified':True,
        'original_validation_unchanged':sha(CURRENT/'val.jsonl')==sha(work/'val.jsonl'),
        'outputs_sha256':{str(p):sha(p) for p in work.iterdir() if p.is_file()}})
    inputs.update({str(p):sha(p) for p in work.iterdir() if p.is_file()})
    save(work/'plan.json',{'inputs_sha256':inputs,'parent':str(parent),'training_started':False,
        'source_mutations':False,'production_changed':False,
        'policy':'Inherit all old flags; exclude the additional cross-model music-suspect reference '
                 'and its target/ref path/origin/PCM reuse. Keep original audio and uncertainty. Transcript review pending.'})
    verify(inputs)
    print({'retained_rows':len(retained),'hours':train['hours'],'plan_sha256':sha(work/'plan.json')},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT/'music_candidate')
    parser.add_argument('--reference', action='store_true')
    args=parser.parse_args()
    (quarantine_reference if args.reference else prepare)(args.work)
