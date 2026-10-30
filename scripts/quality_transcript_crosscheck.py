"""Fixed independent ASR pilot on current training data; no relabelling or old rescores."""
import argparse
from collections import Counter
import importlib.metadata
from pathlib import Path
import shutil
import time

from quality_music_crosscheck import ROOT, balanced, read, manifest, sha, save
from quality_numeric import numeric_equivalence
from quarantine_music_candidates import verify
from voxft.eval import _error_rate, _norm

LANGUAGES = {'id':'Indonesian', 'th':'Thai', 'tl':None, 'zh':'Chinese', 'en':'English'}


def select_cases(rows):
    enriched=[{**row,'numeric_review':numeric_equivalence(row['text'],row['hyp'],row['lang'])}
              for row in rows]
    groups={
        'whisper_disagreement':balanced([r for r in enriched if r['cer']>.05
                       and not r['numeric_review']['numeric_only_difference']],80),
        'low_disagreement_control':balanced([r for r in enriched if r['cer']<=.02],40),
    }
    cases=[{**row,'role':role} for role,group in groups.items() for row in group]
    if len({r['audio'] for r in cases})!=120:
        raise ValueError('Duplicate pilot targets')
    return cases,enriched


def prepare(work):
    if work.exists():raise FileExistsError(work)
    parent=ROOT/'music_candidate'
    if sha(parent/'plan.json')!='1128ca0d327abf365f2d829d72094463574187c90243bb7531c8a1d06e225f45':
        raise ValueError('Wrong current music candidate')
    inputs=read(parent/'plan.json')['inputs_sha256'].copy()
    inputs[str(parent/'plan.json')]=sha(parent/'plan.json')
    current={r['audio'] for r in manifest(parent/'train.jsonl')}
    original=manifest(ROOT/'rows.jsonl')
    all_numeric=[{'audio':r['audio'],'source_id':r['source_id'],'original_cer':r['cer'],
                 **numeric_equivalence(r['text'],r['hyp'],r['lang'])} for r in original]
    cases,enriched=select_cases([r for r in original if r['audio'] in current])
    if len(enriched)!=3976:raise ValueError('Current target accounting differs')
    assets=read(ROOT/'qwen_setup/assets.json')
    if assets['revision']!='5eb144179a02acc5e5ba31e748d22b0cf3e303b0':
        raise ValueError('Wrong independent ASR revision')
    inputs.update(assets['files_sha256'])
    for p in (ROOT/'qwen_setup/assets.json',Path(__file__),Path(__file__).with_name('quality_numeric.py')):
        inputs[str(p)]=sha(p)
    for directory in (Path(__file__).resolve().parents[1]/'src', Path(__file__).resolve().parent):
        inputs.update({str(p):sha(p) for p in directory.rglob('*.py')})
    overlay=ROOT/'qwen_setup/overlay'
    inputs.update({str(p):sha(p) for p in overlay.rglob('*') if p.is_file()
                   and p.suffix not in {'.pyc'} and '__pycache__' not in p.parts})
    verify(inputs)
    if shutil.disk_usage(ROOT).free<4.1*1024**3:raise RuntimeError('Retain disk reserve')
    work.mkdir()
    save(work/'numeric_review.json',{'rows':all_numeric,'numeric_only':sum(r['numeric_only_difference'] for r in all_numeric),
        'above_005_numeric_only':sum(r['original_cer']>.05 and r['numeric_only_difference'] for r in all_numeric),
        'user_policy':'Equivalent digit/word spellings are ignored as an exclusion reason. Other quality flags remain.',
        'old_scores_changed':False, 'original_labels_changed':False})
    save(work/'cases.json',cases)
    for p in work.iterdir():inputs[str(p)]=sha(p)
    save(work/'plan.json',{'inputs_sha256':inputs,'model_path':assets['model_path'],
        'new_outputs_max':120,'batch_size':4,'max_new_tokens':512,'dtype':'bfloat16','attention':'sdpa',
        'context':'','language_prompts':LANGUAGES,'expected_text_prompt':False,
        'selection':'80 source-balanced SHA-ordered Whisper CER>.05 disagreements excluding exact integer spelling '
                    'equivalence, plus 40 low-disagreement controls; no selection on independent ASR outcomes.',
        'training_started':False,'source_mutations':False,'production_changed':False,
        'policy':'Independent diagnostic only. Agreement is not transcription truth. Numeric-only differences are '
                 'not exclusion evidence; changed values, missing words and unresolved numeric formats need review. '
                 'No relabelling, old-score replacement, automatic admission/exclusion or final holdout use.'})
    print('PREPARED',len(cases),sha(work/'plan.json'),flush=True)


def run(work):
    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel

    plan=read(work/'plan.json')
    verify(plan['inputs_sha256'])
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<12*1024**3:
        raise RuntimeError('Need CUDA and 12 GiB free')
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    with (work/'run.started').open('x',encoding='utf-8') as f:f.write(sha(work/'plan.json'))
    (work/'items').mkdir()
    model=Qwen3ASRModel.from_pretrained(plan['model_path'],dtype=torch.bfloat16,device_map='cuda:0',
        attn_implementation=plan['attention'],max_inference_batch_size=plan['batch_size'],
        max_new_tokens=plan['max_new_tokens'],local_files_only=True)
    model.model.eval()
    model.model.generation_config.do_sample=False
    supported=model.get_supported_languages()
    if any(lang not in supported for lang in LANGUAGES.values() if lang):
        raise ValueError('Runtime language support differs')
    if 'Filipino' not in supported:raise ValueError('Filipino not supported by actual runtime')
    save(work/'runtime.json',{'versions':{p:importlib.metadata.version(p) for p in
        ('qwen-asr','transformers','torch','accelerate','tokenizers','huggingface-hub')},
        'supported_languages':supported,'actual_device':str(model.device),'actual_dtype':str(model.dtype),
        'attention':model.model.config._attn_implementation})
    original_generate=model.model.generate
    captured=[]

    def capture(*args,**kwargs):
        if kwargs.get('do_sample',model.model.generation_config.do_sample) is not False:
            raise ValueError('Expected deterministic decoding')
        output=original_generate(*args,**kwargs)
        sequences=output.sequences if hasattr(output,'sequences') else output
        new=sequences[:,kwargs['input_ids'].shape[1]:].detach().cpu()
        eos=model.model.generation_config.eos_token_id
        eos_ids={eos} if isinstance(eos,int) else set(eos)
        for row in new.tolist():
            captured.append({'generated_token_ids':row,'eos_seen':bool(eos_ids.intersection(row)),
                             'raw_decoded':model.processor.tokenizer.decode(row,skip_special_tokens=False)})
        return output

    model.model.generate=capture
    cases=read(work/'cases.json')
    results=[]
    for start in range(0,len(cases),plan['batch_size']):
        if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError('Disk reserve reached')
        batch=cases[start:start+plan['batch_size']]
        for row in batch:
            if sha(row['audio'])!=row['file_sha256']:raise ValueError('Audio changed')
        captured.clear()
        with torch.inference_mode():
            outputs=model.transcribe(audio=[r['audio'] for r in batch],context=['']*len(batch),
                language=[LANGUAGES[r['lang']] for r in batch],return_time_stamps=False)
        if len(outputs)!=len(batch) or len(captured)!=len(batch):
            raise ValueError('Unexpected ASR chunking or output accounting')
        for offset,(row,output,generation) in enumerate(zip(batch,outputs,captured)):
            result={'case':row,'qwen_hyp':output.text,'qwen_language':output.language,
                    'qwen_raw_cer':_error_rate(_norm(output.text),_norm(row['text'])),
                    'whisper_qwen_cer':_error_rate(_norm(output.text),_norm(row['hyp'])),
                    'numeric_review':numeric_equivalence(row['text'],output.text,row['lang']),
                    'generation':generation,'status':'unverified_review_only' if generation['eos_seen'] else 'token_limit_unresolved'}
            save(work/'items'/f'{start+offset:03d}.json',result)
            results.append(result)
        print('DONE',len(results),'/'+str(len(cases)),flush=True)
    verify(plan['inputs_sha256'])
    save(work/'summary.json',{'completed':len(results),'eos_seen':sum(r['generation']['eos_seen'] for r in results),
        'by_source':{source:{'n':sum(r['case']['source_id']==source for r in results),
            'both_raw_cer_gt005':sum(r['case']['cer']>.05 and r['qwen_raw_cer']>.05
                                   for r in results if r['case']['source_id']==source)}
            for source in sorted({r['case']['source_id'] for r in results})},
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'time':time.time(),
        'automatic_exclusions':0,'training_started':False,'source_mutations':False,
        'outputs_sha256':{str(p):sha(p) for p in sorted((work/'items').glob('*.json'))},'note':plan['policy']})


def prepare_expanded(work):
    if work.exists():raise FileExistsError(work)
    current=ROOT/'music_ref_candidate'
    pilot=ROOT/'transcript_pilot_v2'
    if sha(current/'plan.json')!='e3e8fcff85ad7f12ca687495dfd48ae9aa7c26b6ff8e1a70365015335c452774':
        raise ValueError('Wrong current data pool')
    parent=read(pilot/'plan.json')
    if sha(pilot/'plan.json')!='de667f522d4d051537b68b20b0838ea25db07e0588c141d111bcab0eff8c489a':
        raise ValueError('Wrong pilot')
    inputs=read(current/'plan.json')['inputs_sha256'].copy()
    inputs.update(parent['inputs_sha256'])
    for p in (current/'plan.json',pilot/'plan.json',pilot/'summary.json',Path(__file__)):
        inputs[str(p)]=sha(p)
    source=Path(__file__).resolve().parents[1]
    for directory in (source/'src',source/'scripts'):
        inputs.update({str(p):sha(p) for p in directory.rglob('*.py')})
    rows=manifest(current/'train.jsonl')
    original={r['audio']:r for r in manifest(ROOT/'rows.jsonl')}
    cases={r['audio']:{**original[r['audio']],'role':'current_target'} for r in rows}
    clean=Path('/root/autodl-tmp/full_corpus_clean_20261029/parallel8/run')
    acoustic={r['audio']:r for r in manifest(clean/'acoustic.jsonl')}
    asr={r['audio']:r for r in manifest(clean/'asr.jsonl')}
    tasks={r['audio']:r for r in manifest(clean/'tasks.jsonl')}
    for row in rows:
        path=row.get('ref_audio')
        if not path or path in cases:continue
        task=tasks[path];observed=asr[path]['result'];measured=acoustic[path]
        if len(task['langs'])!=1 or len(task['origins'])!=1 or len(observed['comparisons'])!=1:
            raise ValueError('Ambiguous reference-only metadata')
        source_id=Path(path).relative_to('/root/autodl-tmp/voxft_data/processed').parts[0]
        if source_id not in {'cv22_id','cv22_th'} or not row['ref_speaker'].startswith(source_id+':'):
            raise ValueError('Reference source/verified identity differs')
        cases[path]={'audio':path,'file_sha256':measured['file_sha256'],'pcm_sha256':measured['pcm_sha256'],
            'source_id':source_id,'lang':task['langs'][0],'origin_audio':task['origins'][0],
            'text':observed['comparisons'][0]['text'],'hyp':observed['hyp'],
            'cer':observed['comparisons'][0]['cer'],'duration':measured['duration'],'role':'current_reference_only'}
    ordered=sorted(cases.values(),key=lambda r:(r['file_sha256'],r['audio']))
    if len(rows)!=3975 or len(ordered)!=4985:raise ValueError('Current target/reference coverage differs')
    for i,case in enumerate(ordered):
        case['review_id']=i
        if inputs[case['audio']]!=case['file_sha256'] or not 0<case['duration']<=30:
            raise ValueError('Wrong audio or out-of-scope duration')
    reused={}
    pilot_summary=read(pilot/'summary.json')
    for p in sorted((pilot/'items').glob('*.json')):
        if sha(p)!=pilot_summary['outputs_sha256'][str(p)]:raise ValueError('Pilot output changed')
        item=read(p);path=item['case']['audio']
        inputs[str(p)]=sha(p)
        if path in cases:
            if any(item['case'][k]!=cases[path][k] for k in ('file_sha256','text','lang')):
                raise ValueError('Pilot conditioning differs')
            reused[path]={'path':str(p),'sha256':sha(p),'review_id':cases[path]['review_id']}
    fresh=[r for r in ordered if r['audio'] not in reused]
    verify(inputs)
    if shutil.disk_usage(ROOT).free<4.1*1024**3:raise RuntimeError('Retain disk reserve')
    work.mkdir()
    save(work/'cases.json',ordered);save(work/'reused.json',reused)
    for p in work.iterdir():inputs[str(p)]=sha(p)
    for shard in range(4):
        directory=work/f'shard{shard}';directory.mkdir()
        subset=fresh[shard::4]
        save(directory/'cases.json',subset)
        save(directory/'plan.json',{**parent,'inputs_sha256':{**inputs,str(directory/'cases.json'):sha(directory/'cases.json')},
            'new_outputs_max':len(subset),'selection':'Complete current target/reference-only scope, SHA ordered, four fixed shards.',
            'policy':parent['policy']+' All 4985 current target/ref files are included, so no CER threshold selects the expansion.'})
        inputs[str(directory/'plan.json')]=sha(directory/'plan.json')
    save(work/'plan.json',{'inputs_sha256':inputs,'total':len(ordered),'reused':len(reused),'new_outputs_max':len(fresh),
        'targets':3975,'reference_only':1010,'workers':4,'batch_size_per_worker':4,
        'training_started':False,'source_mutations':False,'automatic_exclusions':False,
        'policy':'Complete independent raw transcription coverage; retain old numeric views and emit new views separately. '
                 'No old rescoring, no automatic relabelling/admission/quarantine, no final holdout or production changes.'})
    print('PREPARED_EXPANSION',len(ordered),len(reused),len(fresh),sha(work/'plan.json'),flush=True)


def collect_expanded(work):
    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    cases=read(work/'cases.json')
    mapping={int(r['review_id']):r for r in read(work/'reused.json').values()}
    for shard in range(4):
        directory=work/f'shard{shard}'
        summary=read(directory/'summary.json')
        expected=read(directory/'cases.json')
        paths=sorted((directory/'items').glob('*.json'),key=lambda p:int(p.stem))
        if len(paths)!=len(expected) or summary['completed']!=len(expected):raise ValueError('Incomplete shard')
        for p,case in zip(paths,expected):
            item=read(p)
            if sha(p)!=summary['outputs_sha256'][str(p)] or item['case']!=case:
                raise ValueError('Changed output or conditioning')
            if case['review_id'] in mapping:raise ValueError('Duplicate outcome')
            mapping[case['review_id']]={'path':str(p),'sha256':sha(p),'review_id':case['review_id']}
    if set(mapping)!=set(range(len(cases))):raise ValueError('Missing outputs')
    numeric=[];eos=0
    for i,case in enumerate(cases):
        record=mapping[i];item=read(record['path'])
        if sha(record['path'])!=record['sha256']:raise ValueError('Changed reused outcome')
        eos+=item['generation']['eos_seen']
        numeric.append({'review_id':i,'audio':case['audio'],'source_id':case['source_id'],'role':case['role'],
                        'whisper':numeric_equivalence(case['text'],case['hyp'],case['lang']),
                        'qwen':numeric_equivalence(case['text'],item['qwen_hyp'],case['lang'])})
    save(work/'results.json',mapping)
    save(work/'numeric_review_current.json',numeric)
    save(work/'summary.json',{'completed':len(mapping),'reused':plan['reused'],'new':plan['new_outputs_max'],
        'eos_seen':eos,'training_started':False,'automatic_exclusions':0,'note':plan['policy'],
        'outputs_sha256':{str(work/name):sha(work/name) for name in ('results.json','numeric_review_current.json')}})
    print('COMPLETE_EXPANSION',len(mapping),'EOS',eos,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','prepare-expanded','collect-expanded'))
    parser.add_argument('--work',type=Path,default=ROOT/'transcript_pilot')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'prepare-expanded':prepare_expanded,'collect-expanded':collect_expanded}[args.action](args.work)
