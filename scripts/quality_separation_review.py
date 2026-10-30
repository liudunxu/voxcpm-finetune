"""Paired music and unprompted transcript checks; preserve all earlier outputs."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

from audit_music import LABELS, aggregate, windows
from quality_music_crosscheck import ROOT, MUSIC, PROMPTS, read, manifest, sha, save, run as clap_run
from quality_numeric import numeric_equivalence
from quality_transcript_crosscheck import run as qwen_run
from quarantine_music_candidates import verify
from review_training_silence import transcribe

CLEAN=Path('/root/autodl-tmp/full_corpus_clean_20261029/parallel8/run')
PILOT=ROOT/'separation_pilot'
REVIEW=ROOT/'separation_review'


def index_results(items, expected=None):
    indexed={}
    for item in items:
        path=item['case']['audio']
        if path in indexed:raise ValueError('Duplicate outcome: '+path)
        indexed[path]=item
    if expected is not None and set(indexed)!=set(expected):
        raise ValueError('Missing or extra paired outcomes')
    return indexed


def verify_pair_conditioning(pair, qwen):
    source=pair['case']
    for path,digest in ((source['audio'],source['file_sha256']),
                        (pair['vocals'],pair['vocals_sha256'])):
        case=qwen[path]['case']
        if (case['audio']!=path or case['file_sha256']!=digest
                or case['text']!=source['text'] or case['lang']!=source['lang']):
            raise ValueError('Paired ASR conditioning differs')


def prepare_expanded(work):
    if work.exists():raise FileExistsError(work)
    transcript=ROOT/'transcript_all';separation=ROOT/'separation_all'
    parents={transcript:'034d813a04e244324e8b0f4b22a5cb562cfa3913bc52853f77183ce3917887f1',
             separation:'4d93c149d0103076fc60f552c5a7ac29f426dee0fddc9f8a7ca1d5d1533845ed',
             REVIEW:'24bc3cd8d4d9cb71a33007fda003527d35a72c3e6c04d88c8ddf0a5150fc4e13'}
    inputs={}
    for directory,digest in parents.items():
        if sha(directory/'plan.json')!=digest:raise ValueError('Wrong frozen parent')
        inputs.update(read(directory/'plan.json')['inputs_sha256'])
        inputs[str(directory/'plan.json')]=digest
    review=read(REVIEW/'comparison.json')
    verify(review['outputs_sha256']);inputs.update(review['outputs_sha256'])
    inputs[str(REVIEW/'comparison.json')]=sha(REVIEW/'comparison.json')
    summary=read(transcript/'summary.json')
    if summary['completed']!=4985 or summary['eos_seen']!=4985:raise ValueError('Incomplete original ASR')
    inputs.update(summary['outputs_sha256'])
    inputs[str(transcript/'summary.json')]=sha(transcript/'summary.json')
    separated_summary=read(separation/'summary.json')
    if separated_summary['completed']!=6027:raise ValueError('Incomplete separation')
    inputs[str(separation/'results.json')]=separated_summary['results_sha256']
    inputs[str(separation/'summary.json')]=sha(separation/'summary.json')
    verify(inputs)
    separated={}
    for record in read(separation/'results.json').values():
        if sha(record['path'])!=record['sha256']:raise ValueError('Separated metadata changed')
        item=read(record['path']);path=item['case']['audio']
        if path in separated:raise ValueError('Duplicate separated source')
        separated[path]=item;inputs[record['path']]=record['sha256']
        inputs[item['vocals']]=item['vocals_sha256']
    cases=read(transcript/'cases.json');qwen={};pairs=[]
    outputs=read(transcript/'results.json')
    if len(cases)!=4985 or len(outputs)!=4985:raise ValueError('Unexpected current scope')
    for i,case in enumerate(cases):
        if case['review_id']!=i:raise ValueError('Original review ordering changed')
        record=outputs[str(i)];item=read(record['path'])
        if sha(record['path'])!=record['sha256']:raise ValueError('Original ASR output changed')
        if any(item['case'][key]!=case[key] for key in ('audio','file_sha256','lang','text')):
            raise ValueError('Original ASR conditioning changed')
        qwen[case['audio']]={'path':record['path'],'sha256':record['sha256']}
        inputs[record['path']]=record['sha256']
        pair=separated[case['audio']]
        if (pair['case']['file_sha256']!=case['file_sha256'] or pair['sample_rate']!=16000
                or pair['frames']!=round(case['duration']*16000)):
            raise ValueError('Separation source or length differs')
        pairs.append({**pair,'case':case})
    expected={p['case']['audio'] for p in pairs}|{p['vocals'] for p in pairs}
    for path in (REVIEW/'qwen/items').glob('*.json'):
        item=read(path);audio=item['case']['audio']
        if audio in expected and audio not in qwen:
            qwen[audio]={'path':str(path),'sha256':sha(path)}
    known=[]
    for directory in (ROOT/'clap_all',ROOT/'clap_reference',REVIEW/'clap'):
        known.extend(read(directory/'reused.json'))
        known.extend(read(p) for p in (directory/'items').glob('*.json'))
        for p in directory.rglob('*.json'):inputs[str(p)]=sha(p)
    clap={}
    for item in known:
        path=item['case']['audio']
        if path in clap and clap[path]!=item:raise ValueError('Conflicting reused CLAP outputs')
        clap[path]=item
    reuse={'clap':{p:r for p,r in clap.items() if p in expected},'qwen':qwen,
           'ast':read(REVIEW/'after_ast.json'),'whisper':read(REVIEW/'after_whisper.json')}
    selected={p['case']['audio']:p['case']['file_sha256'] for p in pairs};before={}
    for i,case in enumerate(manifest(MUSIC/'cases.jsonl')):
        if case['audio'] not in selected:continue
        path=MUSIC/'items'/f'{i:05d}.json';item=read(path)
        if item['case']!=case or case['file_sha256']!=selected[case['audio']]:
            raise ValueError('Original AST conditioning changed')
        before[case['audio']]=item;inputs[str(path)]=sha(path)
    if set(before)!=set(selected):raise ValueError('Incomplete original AST coverage')
    for directory in (Path(__file__).resolve().parent,Path(__file__).resolve().parents[1]/'src'):
        inputs.update({str(p):sha(p) for p in directory.rglob('*.py')})
    wrapper=Path(__file__).with_suffix('.sh')
    inputs[str(wrapper)]=sha(wrapper)
    verify(inputs)
    if shutil.disk_usage(ROOT).free<4.3*1024**3:raise RuntimeError('Retain 4 GiB disk reserve')
    work.mkdir();save(work/'pairs.json',pairs);save(work/'reuse.json',reuse)
    inputs[str(work/'pairs.json')]=sha(work/'pairs.json')
    inputs[str(work/'reuse.json')]=sha(work/'reuse.json')
    parent=read(REVIEW/'plan.json')
    policy=('Complete paired QC of all 4985 current target/ref files, four SHA-ordered shards. '
            'Reuse every available original and separated outcome, including failures. '
            'No relabelling, automatic training admission, old-exclusion recovery, final holdout or production changes. '
            'Both-model disagreement is uncertain evidence; numeric equivalence is not a rejection reason.')
    common={**parent,'inputs_sha256':inputs,'reuse':str(work/'reuse.json'),'policy':policy}
    for shard in range(4):
        directory=work/f'shard{shard}';directory.mkdir();subset=pairs[shard::4]
        save(directory/'pairs.json',subset)
        save(directory/'before_ast.json',{p['case']['audio']:before[p['case']['audio']] for p in subset})
        own={str(p):sha(p) for p in directory.iterdir()}
        save(directory/'plan.json',{**common,'inputs_sha256':{**inputs,**own},'outputs':len(subset)})
    save(work/'plan.json',{'inputs_sha256':{**inputs,**{str(work/f'shard{s}/plan.json'):
        sha(work/f'shard{s}/plan.json') for s in range(4)}},'outputs':len(pairs),'workers':4,
        'existing_original_asr':4985,'existing_paired_pilot':sum(p['vocals'] in reuse['whisper'] for p in pairs),
        'training_eligible':False,'policy':policy})
    print('PREPARED_EXPANDED_REVIEW',len(pairs),sha(work/'plan.json'),flush=True)


def prepare(work):
    if work.exists():raise FileExistsError(work)
    if sha(PILOT/'plan.json')!='d0c9f2915cfe065b9a3acbe87e5a722d16a74f937b19a0e17a9f65f6e36da83a':
        raise ValueError('Wrong separation pilot')
    inputs=read(PILOT/'plan.json')['inputs_sha256'].copy()
    summary=read(PILOT/'summary.json')
    if summary['completed']!=24:raise ValueError('Incomplete pilot')
    inputs.update(summary['outputs_sha256'])
    pairs=[read(p) for p in sorted((PILOT/'items').glob('*.json'))]
    selected={p['case']['audio'] for p in pairs}
    before={}
    for i,case in enumerate(manifest(MUSIC/'cases.jsonl')):
        if case['audio'] in selected:
            p=MUSIC/'items'/f'{i:05d}.json';item=read(p)
            if item['case']!=case:raise ValueError('Original AST input changed')
            before[case['audio']]=item;inputs[str(p)]=sha(p)
    if set(before)!=selected:raise ValueError('Original music coverage incomplete')
    ast=read(MUSIC.parent/'model_assets.json');clap=read(ROOT/'clap_assets.json')
    inputs.update(ast['files_sha256']);inputs.update(clap['files_sha256'])
    asr_path=read(CLEAN/'plan.json')['asr_path']
    for p in Path(asr_path).rglob('*'):
        if p.is_file():inputs[str(p)]=sha(p)
    for p in (PILOT/'plan.json',PILOT/'summary.json',MUSIC.parent/'model_assets.json',ROOT/'clap_assets.json',Path(__file__)):
        inputs[str(p)]=sha(p)
    for directory in (Path(__file__).resolve().parent,Path(__file__).resolve().parents[1]/'src'):
        inputs.update({str(p):sha(p) for p in directory.rglob('*.py')})
    verify(inputs)
    work.mkdir();save(work/'pairs.json',pairs);save(work/'before_ast.json',before)
    for p in work.iterdir():inputs[str(p)]=sha(p)
    save(work/'plan.json',{'inputs_sha256':inputs,'ast_path':ast['model_path'],'clap_path':clap['model_path'],
        'asr_path':asr_path,'outputs':24,'training_eligible':False,
        'policy':'Fixed source/separated pairs; reuse all available original outcomes. '
                 'No relabelling, rerolls, automatic training admission or production changes.'})
    print('PREPARED_REVIEW',sha(work/'plan.json'),flush=True)


def music(work):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    with (work/'music.started').open('x') as f:f.write(sha(work/'plan.json'))
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.10)
    processor=ASTFeatureExtractor.from_pretrained(plan['ast_path'],local_files_only=True)
    model,loading=ASTForAudioClassification.from_pretrained(plan['ast_path'],local_files_only=True,
                            use_safetensors=True,output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('AST did not load exactly')
    model.eval().to('cuda',dtype=torch.float32)
    ids={label:model.config.label2id[label] for label in LABELS};after={}
    reuse=read(plan['reuse']) if plan.get('reuse') else {}
    pairs=read(work/'pairs.json')
    (work/'ast_items').mkdir()
    for i,pair in enumerate(pairs):
        if pair['vocals'] in reuse.get('ast',{}):
            after[pair['vocals']]=reuse['ast'][pair['vocals']]
            continue
        if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('Disk reserve reached')
        wav,sr=sf.read(pair['vocals'],dtype='float32')
        if sr!=16000 or wav.ndim!=1 or sha(pair['vocals'])!=pair['vocals_sha256']:
            raise ValueError('Separated audio changed')
        measured=[]
        for start,end in windows(len(wav)):
            batch=processor(wav[start:end],sampling_rate=sr,return_tensors='pt').to('cuda')
            with torch.inference_mode():scores=model(**batch).logits[0].sigmoid().cpu().numpy()
            if not np.isfinite(scores).all():raise ValueError('Nonfinite AST scores')
            measured.append({'start_sample':start,'end_sample':end,'scores':{k:float(scores[v]) for k,v in ids.items()}})
        after[pair['vocals']]={'scores':aggregate(w['scores'] for w in measured),'windows':measured}
        save(work/'ast_items'/f'{i:04d}.json',{'audio':pair['vocals'],
             'file_sha256':pair['vocals_sha256'],'result':after[pair['vocals']]})
        if (i+1)%25==0:print('AST',i+1,'/'+str(len(pairs)),flush=True)
    save(work/'after_ast.json',after)
    del model,processor
    torch.cuda.empty_cache()
    if reuse:known=reuse['clap']
    else:
        known=read(ROOT/'clap_all/reused.json')
        known += [read(p) for p in (ROOT/'clap_all/items').glob('*.json')]
        known += [read(ROOT/'clap_reference/items/000.json')]
        known=index_results(known)
    cases=[];reused=[];before=read(work/'before_ast.json')
    for pair in pairs:
        source=pair['case'];raw=source['audio']
        for path,digest,score,role in (
                (raw,source['file_sha256'],before[raw]['scores']['Music'],'original_control'),
                (pair['vocals'],pair['vocals_sha256'],after[pair['vocals']]['scores']['Music'],'separated_vocals')):
            if path in known:
                item=known[path]
                if item['case']['file_sha256']!=digest or item['case']['music_score']!=score:
                    raise ValueError('Reused CLAP source or AST score changed')
                cases.append(item['case']);reused.append(item)
            else:
                cases.append({'audio':path,'file_sha256':digest,'source_id':source['source_id'],
                              'music_score':score,'role':role})
    directory=work/'clap';directory.mkdir()
    save(directory/'cases.json',cases);save(directory/'reused.json',reused)
    inputs={**plan['inputs_sha256'],str(work/'after_ast.json'):sha(work/'after_ast.json')}
    for p in directory.iterdir():inputs[str(p)]=sha(p)
    save(directory/'plan.json',{'inputs_sha256':inputs,'model_path':plan['clap_path'],'prompts':list(PROMPTS),
        'outputs':len(cases),'reused_outputs':len(reused),'new_outputs_max':len(cases)-len(reused),
        'roles':dict(Counter(c['role'] for c in cases)),'policy':plan['policy']})
    clap_run(directory)
    print('MUSIC_COMPLETE',flush=True)


def asr(work):
    import gc
    import torch
    from faster_whisper import WhisperModel
    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    with (work/'asr.started').open('x') as f:f.write(sha(work/'plan.json'))
    pairs=read(work/'pairs.json')
    reuse=read(plan['reuse']) if plan.get('reuse') else {}
    model=WhisperModel(plan['asr_path'],device='cuda',compute_type='int8_float16',cpu_threads=2,
                       num_workers=1,local_files_only=True)
    after={}
    (work/'whisper_items').mkdir()
    for i,pair in enumerate(pairs):
        if pair['vocals'] in reuse.get('whisper',{}):
            after[pair['vocals']]=reuse['whisper'][pair['vocals']]
            if [r['text'] for r in after[pair['vocals']]['comparisons']]!=[pair['case']['text']]:
                raise ValueError('Reused Whisper target text changed')
            continue
        if shutil.disk_usage(ROOT).free<4*1024**3 or sha(pair['vocals'])!=pair['vocals_sha256']:
            raise RuntimeError('Disk reserve reached or separated audio changed')
        source=pair['case']
        after[pair['vocals']]=transcribe(model,pair['vocals'],source['lang'],[source['text']])
        save(work/'whisper_items'/f'{i:04d}.json',{'audio':pair['vocals'],
             'file_sha256':pair['vocals_sha256'],'result':after[pair['vocals']]})
        if (i+1)%25==0:print('WHISPER',i+1,'/'+str(len(pairs)),flush=True)
    save(work/'after_whisper.json',after)
    del model;gc.collect();torch.cuda.empty_cache()
    baseline=reuse.get('qwen',{})
    if not reuse:
        for p in sorted((ROOT/'transcript_pilot_v2/items').glob('*.json')):
            item=read(p);baseline[item['case']['audio']]={'path':str(p),'sha256':sha(p)}
    cases=[];reused={}
    for pair in pairs:
        source=pair['case'];path=source['audio']
        if path in baseline:reused[path]=baseline[path]
        else:cases.append({**source,'role':'original_music_suspect'})
        if pair['vocals'] in baseline:
            reused[pair['vocals']]=baseline[pair['vocals']]
            continue
        result=after[pair['vocals']]
        cases.append({'audio':pair['vocals'],'file_sha256':pair['vocals_sha256'],'source_id':source['source_id'],
            'lang':source['lang'],'text':source['text'],'hyp':result['hyp'],'cer':result['comparisons'][0]['cer'],
            'origin_audio':source['origin_audio'],'duration':pair['frames']/16000,'role':'separated_vocals'})
    directory=work/'qwen';directory.mkdir()
    save(directory/'cases.json',cases);save(directory/'reused.json',reused)
    parent=read(ROOT/'transcript_pilot_v2/plan.json')
    inputs={**parent['inputs_sha256'],**plan['inputs_sha256'],str(work/'after_whisper.json'):sha(work/'after_whisper.json')}
    for p in directory.iterdir():inputs[str(p)]=sha(p)
    inputs.update({r['path']:r['sha256'] for r in reused.values()})
    save(directory/'plan.json',{**parent,'inputs_sha256':inputs,'new_outputs_max':len(cases),
        'selection':'Every fixed pair in this shard; reuse all available original and separated Qwen outcomes.',
        'policy':plan['policy']})
    qwen_run(directory)
    print('ASR_COMPLETE',flush=True)


def collect(work):
    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    verify(read(work/'qwen/summary.json')['outputs_sha256'])
    clap=read(work/'clap/reused.json')+[read(p) for p in (work/'clap/items').glob('*.json')]
    pairs=read(work/'pairs.json')
    if len(pairs)!=plan['outputs']:raise ValueError('Changed pair count')
    expected={p['case']['audio'] for p in pairs}|{p['vocals'] for p in pairs}
    if len(expected)!=2*len(pairs):raise ValueError('Duplicate source or separated paths')
    clap=index_results(clap,expected)
    qwen=[read(p) for p in (work/'qwen/items').glob('*.json')]
    for path,item in read(work/'qwen/reused.json').items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Original ASR changed')
        result=read(item['path'])
        if result['case']['audio']!=path:raise ValueError('Wrong reused ASR path')
        qwen.append(result)
    qwen=index_results(qwen,expected)
    before=read(work/'before_ast.json');after=read(work/'after_ast.json');whisper=read(work/'after_whisper.json')
    rows=[]
    for pair in pairs:
        source=pair['case'];raw=source['audio'];vocals=pair['vocals']
        verify_pair_conditioning(pair,qwen)
        words=whisper[vocals]
        rows.append({'original_audio':raw,'vocals':vocals,'lang':source['lang'],'role':source['role'],
            'review_id':source.get('review_id'),'source_id':source['source_id'],'text':source['text'],
            'whisper_hyp':[source['hyp'],words['hyp']], 'qwen_hyp':[qwen[p]['qwen_hyp'] for p in (raw,vocals)],
            'ast_music':[before[raw]['scores']['Music'],after[vocals]['scores']['Music']],
            'clap_all_three_music':[any(w['all_three_music_higher'] for w in clap[p]['windows']) for p in (raw,vocals)],
            'whisper_cer':[source['cer'],words['comparisons'][0]['cer']],
            'qwen_cer':[qwen[p]['qwen_raw_cer'] for p in (raw,vocals)],
            'qwen_eos':[qwen[p]['generation']['eos_seen'] for p in (raw,vocals)],
            'whisper_numeric':[numeric_equivalence(source['text'],hyp,source['lang']) for hyp in (source['hyp'],words['hyp'])],
            'qwen_numeric':[numeric_equivalence(source['text'],qwen[p]['qwen_hyp'],source['lang']) for p in (raw,vocals)],
            'original_metrics':pair['original_metrics'],'separated_metrics':pair['separated_metrics'],
            'training_eligible':False})
    save(work/'comparison.json',{'pairs':rows,'training_eligible':False,'listening_pending':True,
        'policy':plan['policy'],'outputs_sha256':{str(p):sha(p) for p in work.rglob('*.json')}})
    print('PAIRED_REVIEW_COMPLETE',len(rows),flush=True)


def collect_expanded(work):
    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    expected={p['case']['audio']:p for p in read(work/'pairs.json')};rows=[];outputs={}
    for shard in range(4):
        directory=work/f'shard{shard}';comparison=read(directory/'comparison.json')
        verify(comparison['outputs_sha256']);rows.extend(comparison['pairs'])
        outputs[str(directory/'comparison.json')]=sha(directory/'comparison.json')
    if len(rows)!=plan['outputs'] or len({r['original_audio'] for r in rows})!=len(rows):
        raise ValueError('Incomplete or duplicated full review')
    if {r['original_audio'] for r in rows}!=set(expected):raise ValueError('Review scope changed')
    for row in rows:
        pair=expected[row['original_audio']]
        if row['vocals']!=pair['vocals'] or row['text']!=pair['case']['text']:
            raise ValueError('Full paired conditioning changed')
    save(work/'comparison.json',{'pairs':sorted(rows,key=lambda r:r['review_id']),
         'training_eligible':False,'listening_pending':True,'outputs_sha256':outputs,'policy':plan['policy']})
    print('FULL_PAIRED_REVIEW_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','music','asr','collect','prepare-expanded','collect-expanded'))
    parser.add_argument('--work',type=Path,default=ROOT/'separation_review')
    args=parser.parse_args()
    {'prepare':prepare,'music':music,'asr':asr,'collect':collect,
     'prepare-expanded':prepare_expanded,'collect-expanded':collect_expanded}[args.action](args.work)
