"""Freeze an inherited, paired-QC candidate; no ASR reruns or training in this step."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import random
import re
import shutil
import unicodedata

import numpy as np
import soundfile as sf

from audit_training_silence import measure
from control_ab_gate import pcm_conflicts
from incremental_mix import audit_mix, describe_mix, load_exclusions
from quality_music_crosscheck import ROOT, read, manifest, sha, save
from quality_numeric import numeric_equivalence
from quality_separation_review import verify_expanded
from quarantine_music_candidates import verify
from review_training_silence import exclude_flagged
from training_pool import inspect_audio
from voxft.data.pipeline import _write_jsonl
from voxft.eval import _norm, _error_rate
from voxft.paths import ROOT as SOURCE

CURRENT = ROOT/'music_ref_candidate'
REVIEW = ROOT/'separation_review_all'
CLEAN = Path('/root/autodl-tmp/full_corpus_clean_20261029/parallel8/run')
SHARES = {'id':.85/3, 'th':.85/3, 'tl':.85/3, 'zh':.10, 'en':.05}
POLICY = {
    'dual_asr_cer_review': .05, 'shared_deleted_chars_review': 2,
    'derived_asr_increase': .02, 'derived_min_extra_errors': 2,
    'max_edge_seconds': .5, 'max_rms_loss_db': 6., 'max_speech_ratio_loss': .05,
    'max_clipping_run_ms': 1., 'max_target_repetitions': 2, 'seed': 42,
    'music': 'Any original or separated AST>=.5 / CLAP contrast is uncertain; quarantine origin/PCM/ref.',
    'content': 'Dual >5% residual disagreement, identical nonmatching ASR, shared deletion >=2 chars, '
               'or dual differing numeric values are uncertain quarantine reasons, never human error labels.',
    'format': 'NFKC is a diagnostic view only. Whole-text integer equivalence is exempt; '
              'Thai repetition, homophones and names remain uncertain, not automatically relabelled.',
    'derivative': 'Reject suspect derivative, retain eligible original. Never recover historical exclusions.',
    'scope': 'Only current 3975 targets and their 1010 reference-only files; original validation bytes unchanged.',
    'causality': 'Quality selection, duration balance and partial separation jointly change data; not single-factor evidence.',
}


def content_view(row, side):
    text = unicodedata.normalize('NFKC', row['text'])
    reference = _norm(text)
    hyps, errors, numeric = [], [], []
    for model in ('whisper', 'qwen'):
        hyp = unicodedata.normalize('NFKC', row[model+'_hyp'][side])
        equivalent = numeric_equivalence(text, hyp, row['lang'])
        normalized = reference if equivalent['numeric_only_difference'] else _norm(hyp)
        hyps.append(normalized)
        errors.append(_error_rate(normalized, reference))
        numeric.append(equivalent)
    deletions = []
    for hyp in hyps:
        deletions.append({i for tag,a,b,_,_ in SequenceMatcher(None,reference,hyp,autojunk=False).get_opcodes()
                          if tag=='delete' for i in range(a,b)})
    shared = len(deletions[0] & deletions[1])
    changed_numbers = []
    for item in numeric:
        a,b = [re.findall(r'⟦(\d+)⟧',item[key]) for key in ('reference_view','hypothesis_view')]
        changed_numbers.append(bool(a) and a!=b)
    reasons = []
    if all(e>POLICY['dual_asr_cer_review'] for e in errors): reasons.append('dual_asr_residual_disagreement')
    if hyps[0]==hyps[1]!=reference: reasons.append('same_asr_nonmatching_text')
    if shared>=POLICY['shared_deleted_chars_review']: reasons.append('shared_asr_suspected_deletion')
    if all(changed_numbers): reasons.append('dual_numeric_value_unresolved')
    return {'cer':errors, 'numeric_equivalence':[x['numeric_only_difference'] for x in numeric],
            'shared_deleted_chars':shared, 'reasons':reasons,
            'format_or_name_difference_possible':True, 'expected_characters':len(reference)}


def select_version(row, original, separated):
    before, after = content_view(row,0), content_view(row,1)
    source_reasons = list(before['reasons'])
    if not row['qwen_eos'][0]: source_reasons.append('original_asr_without_eos')
    if any(row['clap_all_three_music']) or max(row['ast_music'])>=.5:
        source_reasons.append('uncalibrated_music_suspect')
    source_reasons.extend(original['reasons'])
    reasons = list(separated['reasons'])
    if not row['qwen_eos'][1]: reasons.append('separated_asr_without_eos')
    a,b = row['original_metrics'],row['separated_metrics']
    if a['rms_dbfs']-b['rms_dbfs']>POLICY['max_rms_loss_db']: reasons.append('rms_loss_gt_6db')
    if a['speech_ratio']-b['speech_ratio']>POLICY['max_speech_ratio_loss']: reasons.append('speech_ratio_loss_gt_05')
    if any(y-x>POLICY['derived_asr_increase'] and
           (y-x)*before['expected_characters']>=POLICY['derived_min_extra_errors']-1e-9
           for x,y in zip(before['cer'],after['cer'])):
        reasons.append('asr_worsened_after_separation')
    if after['reasons'] and not before['reasons']: reasons.append('new_content_suspect_after_separation')
    return {'source_reasons':sorted(set(source_reasons)), 'derivative_reasons':sorted(set(reasons)),
            'selected':'excluded' if source_reasons else 'original' if reasons else 'vocals',
            'before':before, 'after':after, 'status':'uncertain_screening_not_native_verdict'}


def balanced_mix(rows):
    """Same duration sampling policy as mix_manifests, without rebuilding protected val."""
    groups = {lang:[r for r in rows if r['lang']==lang] for lang in SHARES}
    seconds = {lang:sum(r['duration'] for r in group) for lang,group in groups.items()}
    if not all(seconds.values()): raise ValueError('Empty language after quarantine')
    budget = min(sum(seconds.values()), *(2*seconds[l]/SHARES[l] for l in SHARES))
    rng=random.Random(42);exposures=Counter();mixed=[]
    for lang,group in groups.items():
        used=0.
        for _ in range(2):
            for row in rng.sample(group,len(group)):
                origin=row['origin_audio']
                if used>=budget*SHARES[lang] or exposures[origin]>=2: continue
                mixed.append(dict(row));used+=row['duration'];exposures[origin]+=1
    rng.shuffle(mixed)
    return mixed,budget


def run(work):
    if work.exists(): raise FileExistsError(work)
    pinned = {CURRENT/'plan.json':'e3e8fcff85ad7f12ca687495dfd48ae9aa7c26b6ff8e1a70365015335c452774',
              REVIEW/'plan.json':'c4a83e287b37fd7354bc1e735e644d192d8fe81dd8b66833ac5018132ca8e530',
              REVIEW/'comparison.json':'2cbed6e54e239e4543b9d716f80d17a8fd66499ca838501094ecf8cd23754f90',
              SOURCE/'eval_cases/reserved_final_20261030.jsonl':
                  'cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c'}
    verify({str(p):d for p,d in pinned.items()})
    verification=verify_expanded(REVIEW)
    if verification['pairs']!=4985: raise ValueError('Unexpected paired review scope')
    inputs=verification.pop('inputs_outputs_sha256')
    for p,d in read(CURRENT/'plan.json')['inputs_sha256'].items():
        if p in inputs and inputs[p]!=d: raise ValueError('Parent SHA conflict: '+p)
        inputs[p]=d
    exclusions,eval_inputs=load_exclusions();inputs.update(eval_inputs)
    inputs.update({str(p):sha(p) for folder in ('scripts','src') for p in (SOURCE/folder).rglob('*.py')})
    inputs[str(Path(__file__).with_suffix('.sh'))]=sha(Path(__file__).with_suffix('.sh'))
    for name in ('train.jsonl','val.jsonl','flagged.json','mix.json'):
        inputs[str(CURRENT/name)]=sha(CURRENT/name)
    for name in ('acoustic.jsonl','tasks.jsonl','protected_validation.jsonl'):
        inputs[str(CLEAN/name)]=sha(CLEAN/name)
    inputs.update({str(p):d for p,d in pinned.items()})
    verify(inputs)
    if shutil.disk_usage(work.parent).free<4.15*1024**3: raise RuntimeError('Retain 4 GiB reserve')
    work.mkdir();(work/'items').mkdir()
    save(work/'plan.json',{'inputs_sha256':inputs,'policy':POLICY,'authorization':
        '2026-10-31 autonomous derived-subset admission after full QC; preserve old frozen plans.',
        'training_started':False,'production_changed':False,'outputs':4985})
    save(work/'review_verification.json',verification)
    rows=manifest(CURRENT/'train.jsonl');val=manifest(CURRENT/'val.jsonl')
    protected=manifest(CLEAN/'protected_validation.jsonl')
    acoustic={r['audio']:r for r in manifest(CLEAN/'acoustic.jsonl')}
    tasks={r['audio']:r for r in manifest(CLEAN/'tasks.jsonl')}
    pairs=read(REVIEW/'pairs.json');reports=read(REVIEW/'comparison.json')['pairs']
    pair_by_audio={p['case']['audio']:p for p in pairs}
    used={r[field] for r in rows for field in ('audio','ref_audio') if r.get(field)}
    if len(rows)!=3975 or len(val)!=895 or len(protected)!=1634 or used!=set(pair_by_audio):
        raise ValueError('Current/paired/protected scope differs')
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    options=VadOptions(min_silence_duration_ms=500,speech_pad_ms=0)
    speech=lambda audio,rate:get_speech_timestamps(audio,options,sampling_rate=rate)
    def inspect(report):
        pair=pair_by_audio[report['original_audio']];case=pair['case'];raw=case['audio']
        original=inspect_audio({'audio':raw,'duration':case['duration']})
        if original['pcm_sha256']!=case['pcm_sha256'] or original['pcm_sha256']!=acoustic[raw]['pcm_sha256']:
            raise ValueError('Original PCM differs')
        original['reasons']=list(acoustic[raw]['reasons'])
        if original['max_full_scale_run_ms']>=1: original['reasons'].append('suspected_clipping')
        if not 0<pair['separated_metrics']['peak']<=1:
            wav,rate=sf.read(pair['vocals'],dtype='float32',always_2d=True)
            if rate!=16000 or wav.shape!=(pair['frames'],1) or not np.isfinite(wav).all():
                raise ValueError('Invalid separated format')
            separated={'pcm_sha256':hashlib.sha256(wav.astype('<f4').tobytes()).hexdigest(),
                       'reasons':['unusable_separated_amplitude']}
        else:
            measured=measure(pair['vocals'],case['duration'],pair['vocals_sha256'],speech)
            separated={**measured,**inspect_audio({'audio':pair['vocals'],'duration':case['duration']})}
            separated['reasons']=list(measured['review_reasons'])
            for edge in (measured['head_silence_sec'],measured['vad']['head_seconds']):
                if edge is None or edge>.5: separated['reasons'].append('head_boundary_unresolved_or_gt_05s')
            if separated['max_full_scale_run_ms']>=1: separated['reasons'].append('suspected_clipping')
        decision=select_version(report,original,separated)
        item={'review_id':report['review_id'],'original_audio':raw,'vocals':pair['vocals'],
              'original_sha256':case['file_sha256'],'vocals_sha256':pair['vocals_sha256'],
              'original':original,'separated':separated,**decision}
        save(work/'items'/f"{report['review_id']:04d}.json",item)
        return item
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(inspect,reports):
            results.append(item)
            if len(results)%100==0:
                if shutil.disk_usage(work).free<4*1024**3: raise RuntimeError('Disk reserve reached')
                print('PAIRED_ACOUSTIC',len(results),'/4985',flush=True)
    index={r['original_audio']:r for r in results};new_flags=[];mapping={}
    for item in results:
        raw=item['original_audio'];pair=pair_by_audio[raw];case=pair['case']
        if item['source_reasons']:
            new_flags.append({'audio':raw,'file_sha256':case['file_sha256'],
                'pcm_sha256':case['pcm_sha256'],'uses':[{'origin_audio':o} for o in tasks[raw]['origins']],
                'reasons':item['source_reasons'],'status':'uncertain_conservatively_quarantined',
                'paired_review_id':item['review_id']})
        selected=raw if item['selected']!='vocals' else pair['vocals']
        mapping[raw]={'selected':selected,'file_sha256':sha(selected),
            'pcm_sha256':item['original' if selected==raw else 'separated']['pcm_sha256'],
            'origin_audio':case['origin_audio'],'source_id':case['source_id'],
            'original_sha256':case['file_sha256'],'original_pcm_sha256':case['pcm_sha256'],
            'decision':item['selected'], 'derivative_allowed':item['selected']=='vocals'}
        acoustic[pair['vocals']]=item['separated']
    flags=read(CURRENT/'flagged.json')+new_flags
    positions,removed=exclude_flagged(rows,flags,acoustic)
    retained=[rows[i-1] for i in positions]
    # Reject on original ancestry before changing paths; every metadata field except audio/ref_audio stays intact.
    derived=[{**r,**{field:mapping[r[field]]['selected'] for field in ('audio','ref_audio') if r.get(field)}}
             for r in retained]
    mixed,budget=balanced_mix(derived)
    audits={}
    for name,group in (('original_pool',retained),('derived_pool',derived),('balanced_train',mixed)):
        audits[name]={'metadata':audit_mix(group,protected,exclusions,max_dur=30),
                      'pcm':pcm_conflicts(group,protected,acoustic)}
        if (audits[name]['metadata']['violations'] or audits[name]['metadata']['origins_above_3x']
                or any(audits[name]['pcm'].values()) or exclude_flagged(group,flags,acoustic)[1]):
            raise ValueError('Isolation failed: '+name)
    if exclude_flagged(val,flags,acoustic)[1]: raise ValueError('New suspicion overlaps unchanged val')
    _write_jsonl(retained,work/'original_pool.jsonl');_write_jsonl(derived,work/'eligible_pool.jsonl')
    _write_jsonl(mixed,work/'train.jsonl');shutil.copyfile(CURRENT/'val.jsonl',work/'val.jsonl')
    save(work/'flagged.json',flags);save(work/'new_flags.json',new_flags);save(work/'excluded.json',removed)
    save(work/'rejected_derivatives.json',[{'audio':r['vocals'],'file_sha256':r['vocals_sha256'],
         'pcm_sha256':r['separated']['pcm_sha256'],'original_audio':r['original_audio'],
         'reasons':r['derivative_reasons'],'status':'uncertain_derivative_only_original_separately_screened'}
         for r in results if r['derivative_reasons']])
    save(work/'audio_mapping.json',mapping);save(work/'isolation.json',audits)
    summary,shares=describe_mix(mixed)
    for lang in shares: shares[lang]['requested']=SHARES[lang]
    parent=read(CURRENT/'mix.json')
    origins=Counter(r['origin_audio'] for r in mixed)
    refs=Counter(r['ref_origin_audio'] for r in mixed if r.get('ref_audio'))
    if max(origins.values())>2: raise ValueError('Original target exposure exceeds 2x')
    save(work/'mix.json',{'train':summary,'eligible_pool':describe_mix(derived)[0],
        'language_shares':shares,'requested_seconds':budget,'source_rows':len(rows),'excluded_rows':len(removed),
        'unique_origins':len(origins),'max_origin_exposure':max(origins.values()),
        'unique_ref_origins':len(refs),'max_ref_exposure':max(refs.values(),default=0),
        'ref_uses':sum(refs.values()),'tl_verified_refs':sum(bool(r.get('ref_audio')) for r in mixed if r['lang']=='tl'),
        'sources':{s:{'license':parent['sources'][s]['license'],
            **describe_mix([r for r in mixed if r['source_id']==s])[0]} for s in sorted({r['source_id'] for r in mixed})},
        'selected_versions':dict(Counter(index[r['audio']]['selected'] for r in retained)),
        'new_flag_reasons':dict(Counter(reason for f in new_flags for reason in f['reasons'])),
        'derived_rejection_reasons':dict(Counter(reason for r in results for reason in r['derivative_reasons'])),
        'by_lang_review':{lang:{'files':sum(r['lang']==lang for r in reports),
            'original_music_hints':sum(r['clap_all_three_music'][0] for r in reports if r['lang']==lang),
            'separated_music_hints':sum(r['clap_all_three_music'][1] for r in reports if r['lang']==lang)} for lang in SHARES},
        'training_authorized':True,'training_started':False,'native_quality':'unverified',
        'original_validation_unchanged':sha(work/'val.jsonl')==sha(CURRENT/'val.jsonl'),
        'suggested_one_epoch_updates':(len(mixed)+15)//16,'suggested_update_policy':'Recompute from this manifest; freeze staged DiT-only plan before training.'})
    verify(inputs)
    save(work/'completion.json',{'status':'candidate_prepared','pairs_verified':4985,'acoustic_checked':len(results),
        'training_started':False,'production_changed':False,'native_quality':'unverified',
        'outputs_sha256':{str(p):sha(p) for p in work.rglob('*') if p.is_file()}})
    print('PAIRED_CANDIDATE_COMPLETE',len(retained),len(mixed),summary['hours'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args().work)
