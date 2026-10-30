"""Bounded source-separation experiment; derivatives never enter a training manifest."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from quality_music_crosscheck import ROOT, read, manifest, sha, save
from quarantine_music_candidates import verify
from voxft.qc.audio import edge_silence, speech_ratio, metallic_resonance, band_ratio_2_8k, spectral_rolloff_99

CLEAN=Path('/root/autodl-tmp/full_corpus_clean_20261029/parallel8/run')


def intervals(samples, chunk, overlap):
    if samples<=0 or not 0<overlap<chunk:raise ValueError('Invalid separation window')
    last=max(0,samples-chunk)
    starts=list(range(0,last+1,chunk-overlap))
    if starts[-1]!=last:starts.append(last)
    return [(start,min(start+chunk,samples)) for start in starts]


def blend_weights(start,end,total,overlap):
    weights=np.ones(end-start,dtype=np.float32)
    ramp=min(overlap,(end-start)//2)
    if ramp:
        fade=np.linspace(1/ramp,1,ramp,dtype=np.float32)
        if start:weights[:ramp]=fade
        if end<total:weights[-ramp:]=fade[::-1]
    return weights


def select_cases(current, suspects, controls, reference):
    selected=[]
    for lang,count in (('id',8),('th',3)):
        group=sorted((r for r in suspects if r['lang']==lang),key=lambda r:(r['file_sha256'],r['audio']))[:count]
        if len(group)!=count:raise ValueError('Insufficient corroborated suspects')
        selected.extend({**r,'role':'music_suspect'} for r in group)
    selected.append({**reference,'role':'music_suspect_reference'})
    for lang in ('id','th','tl'):
        group=sorted((r for r in controls if r['lang']==lang and r['audio'] in current),
                     key=lambda r:(r['file_sha256'],r['audio']))[:4]
        if len(group)!=4:raise ValueError('Insufficient current low-risk controls')
        selected.extend({**r,'role':'lower_risk_control_not_known_clean'} for r in group)
    if len(selected)!=24 or len({r['audio'] for r in selected})!=24:
        raise ValueError('Duplicate pilot sources')
    return selected


def prepare(work):
    if work.exists():raise FileExistsError(work)
    current=ROOT/'music_ref_candidate'
    if sha(current/'plan.json')!='e3e8fcff85ad7f12ca687495dfd48ae9aa7c26b6ff8e1a70365015335c452774':
        raise ValueError('Wrong current data pool')
    inputs=read(current/'plan.json')['inputs_sha256'].copy()
    inputs[str(current/'plan.json')]=sha(current/'plan.json')
    targets={r['audio'] for r in manifest(current/'train.jsonl')}
    original={r['audio']:r for r in manifest(ROOT/'rows.jsonl')}
    flags=read(ROOT/'music_candidate/new_flags.json')
    suspects=[original[r['audio']] for r in flags if r['reasons']==['cross_model_music_suspect']]
    pilot=ROOT/'transcript_pilot_v2'
    if read(pilot/'summary.json')['completed']!=120:raise ValueError('Transcript pilot incomplete')
    inputs.update(read(pilot/'plan.json')['inputs_sha256'])
    controls=[]
    for p in sorted((pilot/'items').glob('*.json')):
        result=read(p)
        if result['case']['role']=='low_disagreement_control':controls.append(result['case'])
        inputs[str(p)]=read(pilot/'summary.json')['outputs_sha256'][str(p)]
    reference=read(ROOT/'clap_reference/cases.json')[0]
    asr={r['audio']:r for r in manifest(CLEAN/'asr.jsonl')}
    task={r['audio']:r for r in manifest(CLEAN/'tasks.jsonl')}[reference['audio']]
    acoustic={r['audio']:r for r in manifest(CLEAN/'acoustic.jsonl')}[reference['audio']]
    comparison=asr[reference['audio']]['result']['comparisons']
    if len(comparison)!=1 or len(task['langs'])!=1 or len(task['origins'])!=1:
        raise ValueError('Ambiguous reference annotation')
    reference={**reference,'lang':task['langs'][0],'origin_audio':task['origins'][0],
               'duration':acoustic['duration'],'text':comparison[0]['text'],
               'hyp':asr[reference['audio']]['result']['hyp'],'cer':comparison[0]['cer']}
    cases=select_cases(targets,suspects,controls,reference)
    assets=read(ROOT/'separation_setup/assets.json')
    if assets['sha256']!='a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816':
        raise ValueError('Unexpected HDemucs asset')
    inputs[assets['model_path']]=assets['sha256']
    inputs[assets['architecture_source']]=assets['architecture_sha256']
    for p in (ROOT/'separation_setup/assets.json',pilot/'plan.json',pilot/'runtime.json',
              pilot/'summary.json',ROOT/'clap_reference/cases.json',Path(__file__)):
        inputs[str(p)]=sha(p)
    for directory in (Path(__file__).resolve().parent,Path(__file__).resolve().parents[1]/'src'):
        inputs.update({str(p):sha(p) for p in directory.rglob('*.py')})
    verify(inputs)
    if shutil.disk_usage(ROOT).free<4.15*1024**3:raise RuntimeError('Retain 4 GiB reserve')
    work.mkdir()
    save(work/'cases.json',cases)
    inputs[str(work/'cases.json')]=sha(work/'cases.json')
    save(work/'plan.json',{'inputs_sha256':inputs,'model_path':assets['model_path'],
        'model':'torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS','sources':assets['sources'],
        'source_rate':16000,'model_rate':44100,'chunk_seconds':10,'overlap_seconds':2,
        'dtype':'float32','resampler':{'lowpass_filter_width':64,'rolloff':.99,'resampling_method':'sinc_interp_hann'},
        'outputs':24,'seed':42,'normalization':'Per-clip mono mean/std before model; restore original scale afterward.',
        'training_eligible':False,'source_mutations':False,'production_changed':False,
        'policy':'User requested model removal of BGM/noise while retaining original voice. '
                 '12 corroborated suspects (8 ID +3 TH +1 ID ref) and 12 lower-risk controls (4/lang); '
                 'controls are not known-clean truth. Only vocals and residual written to an isolated experimental '
                 'directory, same duration, no trimming/compression/peak normalization. Keep all results and original '
                 'path/origin/PCM exclusions; no re-admission. Require independent content and voice-damage review.'})
    print('PREPARED',sha(work/'plan.json'),flush=True)


def metrics(wav,rate):
    return {'rms_dbfs':float(20*np.log10(max(1e-12,np.sqrt(np.mean(wav.astype(np.float64)**2))))),
            'peak':float(np.max(np.abs(wav))),'speech_ratio':speech_ratio(wav,rate),
            **edge_silence(wav,rate),**metallic_resonance(wav,rate),
            'band_ratio_2_8k':band_ratio_2_8k(wav,rate),'spectral_rolloff_99':spectral_rolloff_99(wav,rate)}


def run(work):
    import soundfile as sf
    import torch
    import torchaudio
    from torchaudio.functional import resample

    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<12*1024**3:
        raise RuntimeError('Need CUDA and 12 GiB free')
    with (work/'run.started').open('x',encoding='utf-8') as f:f.write(sha(work/'plan.json'))
    (work/'audio').mkdir();(work/'items').mkdir()
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=torchaudio.models.hdemucs_high(sources=plan['sources'])
    model.load_state_dict(torch.load(plan['model_path'],map_location='cpu',weights_only=True),strict=True)
    model.eval().to('cuda',dtype=torch.float32)
    vocal_index=plan['sources'].index('vocals')
    results=[]
    for i,case in enumerate(read(work/'cases.json')):
        if sha(case['audio'])!=case['file_sha256']:raise ValueError('Original audio changed')
        if shutil.disk_usage(ROOT).free<4.05*1024**3:raise RuntimeError('Disk reserve reached')
        raw,sr=sf.read(case['audio'],dtype='float32',always_2d=True)
        if sr!=16000 or raw.shape[1]!=1 or not np.isfinite(raw).all():raise ValueError('Unexpected source format')
        raw=raw[:,0]
        wav=resample(torch.from_numpy(raw.copy()),sr,plan['model_rate'],**plan['resampler'])
        center,scale=wav.mean(),wav.std()
        if scale.item()<1e-6:raise ValueError('Near-silent source')
        normalized=((wav-center)/scale).repeat(2,1).to('cuda')
        total=normalized.shape[-1]
        overlap=plan['model_rate']*plan['overlap_seconds']
        chunks=intervals(total,plan['model_rate']*plan['chunk_seconds'],overlap)
        sums=torch.zeros(total,device='cuda');weights=torch.zeros_like(sums)
        for start,end in chunks:
            with torch.inference_mode():estimate=model(normalized[None,:,start:end])[0,vocal_index].mean(0)
            if estimate.shape[-1]!=end-start:raise ValueError('Model changed window duration')
            weight=torch.from_numpy(blend_weights(start,end,total,overlap)).to('cuda')
            sums[start:end]+=estimate*weight;weights[start:end]+=weight
        if torch.any(weights<=0):raise ValueError('Uncovered samples')
        vocals=(sums/weights).cpu()*scale+center
        returned=resample(vocals,plan['model_rate'],sr,**plan['resampler']).numpy()
        if not 0<=len(returned)-len(raw)<=1:raise ValueError('Unexpected resampling length drift')
        output=returned[:len(raw)]
        if not np.isfinite(output).all():raise ValueError('Nonfinite separated signal')
        path=work/'audio'/f'{i:04d}_vocals.wav' if plan.get('full_scope') else work/'audio'/f'{i:02d}_vocals.wav'
        residual=work/'audio'/f'{i:02d}_removed.wav' if plan.get('store_residual',True) else None
        sf.write(path,output,sr,subtype='FLOAT')
        if residual is not None:sf.write(residual,raw-output,sr,subtype='FLOAT')
        roundtrip=resample(wav,plan['model_rate'],sr,**plan['resampler']).numpy()[:len(raw)]
        result={'case':case,'vocals':str(path),'vocals_sha256':sha(path),
            'removed':str(residual) if residual else None,'removed_sha256':sha(residual) if residual else None,
            'frames':len(raw),'sample_rate':sr,'resampling_extra_samples':len(returned)-len(raw),
            'chunks':chunks,'original_metrics':metrics(raw,sr),'separated_metrics':metrics(output,sr),
            'roundtrip_only_rms_error':float(np.sqrt(np.mean((raw-roundtrip)**2))),
            'removed_rms_ratio':float(np.sqrt(np.mean((raw-output)**2))/np.sqrt(np.mean(raw**2))),
            'training_eligible':False,'status':'experimental_unverified_voice_preservation'}
        save(work/'items'/f'{i:04d}.json',result);results.append(result)
        print('DONE',len(results),'/'+str(plan['outputs']),case['role'],flush=True)
    verify(plan['inputs_sha256'])
    save(work/'summary.json',{'completed':len(results),'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
        'training_eligible':False,'source_mutations':False,'content_review_pending':True,'listening_pending':True,
        'time':time.time(),'outputs_sha256':{str(p):sha(p) for directory in ('audio','items') for p in (work/directory).iterdir()},
        'note':plan['policy']})


def prepare_full(work):
    if work.exists():raise FileExistsError(work)
    pilot=ROOT/'separation_pilot'
    if sha(pilot/'plan.json')!='d0c9f2915cfe065b9a3acbe87e5a722d16a74f937b19a0e17a9f65f6e36da83a':
        raise ValueError('Wrong separation pilot')
    parent=read(pilot/'plan.json');inputs=parent['inputs_sha256'].copy()
    for p in (pilot/'plan.json',pilot/'summary.json',Path(__file__)):
        inputs[str(p)]=sha(p)
    for directory in (Path(__file__).resolve().parent,Path(__file__).resolve().parents[1]/'src'):
        inputs.update({str(p):sha(p) for p in directory.rglob('*.py')})
    original=Path('/root/autodl-tmp/granary_pilot_20261030/run/train.jsonl')
    rows=manifest(original)
    used={r[k] for r in rows for k in ('audio','ref_audio') if r.get(k)}
    audit=read(ROOT/'plan.json');inputs.update(audit['inputs_sha256'])
    current=manifest(ROOT/'music_ref_candidate/train.jsonl')
    allowed={r[k] for r in current for k in ('audio','ref_audio') if r.get(k)}
    source_rows={r['audio']:r for r in manifest(ROOT/'rows.jsonl')}
    acoustic={r['audio']:r for r in manifest(CLEAN/'acoustic.jsonl')}
    transcripts={r['audio']:r for r in manifest(CLEAN/'asr.jsonl')}
    tasks={r['audio']:r for r in manifest(CLEAN/'tasks.jsonl')}
    cases=[]
    for path in sorted(used,key=lambda p:(acoustic[p]['file_sha256'],p)):
        a=acoustic[path];task=tasks[path];asr=transcripts[path]['result']
        if len(task['langs'])!=1 or len(task['origins'])!=1 or len(asr['comparisons'])!=1:
            raise ValueError('Ambiguous current source metadata')
        case={'audio':path,'file_sha256':a['file_sha256'],'pcm_sha256':a['pcm_sha256'],
              'source_id':Path(path).relative_to('/root/autodl-tmp/voxft_data/processed').parts[0],
              'lang':task['langs'][0],'origin_audio':task['origins'][0],
              'text':asr['comparisons'][0]['text'],'hyp':asr['hyp'],'cer':asr['comparisons'][0]['cer'],
              'duration':a['duration'],'role':'training_target' if path in source_rows else 'reference_only',
              'excluded_from_current_training':path not in allowed,'review_id':len(cases)}
        if inputs[path]!=case['file_sha256']:raise ValueError('Input file not pinned')
        cases.append(case)
    if len(rows)!=5015 or len(cases)!=6027:raise ValueError('Wrong full separation scope')
    by_audio={r['audio']:r for r in cases}
    summary=read(pilot/'summary.json');reused={}
    if summary['completed']!=24:raise ValueError('Incomplete pilot')
    verify(summary['outputs_sha256'])
    for p in sorted((pilot/'items').glob('*.json')):
        item=read(p);case=by_audio[item['case']['audio']]
        if any(item['case'][k]!=case[k] for k in ('file_sha256','text','lang')):
            raise ValueError('Changed pilot input')
        reused[case['audio']]={'path':str(p),'sha256':sha(p),'review_id':case['review_id']}
    inputs.update(summary['outputs_sha256'])
    fresh=[r for r in cases if r['audio'] not in reused]
    verify(inputs)
    expected_bytes=sum(round(r['duration']*16000)*4+128 for r in fresh)
    if shutil.disk_usage(ROOT).free<4*1024**3+expected_bytes*1.02+200*1024**2:
        raise RuntimeError('Insufficient room for all vocals plus 4 GiB reserve')
    work.mkdir();save(work/'cases.json',cases);save(work/'reused.json',reused)
    for p in work.iterdir():inputs[str(p)]=sha(p)
    for shard in range(4):
        directory=work/f'shard{shard}';directory.mkdir();subset=fresh[shard::4]
        save(directory/'cases.json',subset)
        save(directory/'plan.json',{**parent,'inputs_sha256':{**inputs,str(directory/'cases.json'):sha(directory/'cases.json')},
            'outputs':len(subset),'full_scope':True,'store_residual':False,
            'policy':parent['policy']+' User subsequently requested full current target/reference separation. '
                     'All 6027 files used by the original 5015-row pool included; current exclusions remain. '
                     'Only vocals stored for fresh files; original-minus-vocals can be derived for review. No automatic admission.'})
        inputs[str(directory/'plan.json')]=sha(directory/'plan.json')
    save(work/'plan.json',{'inputs_sha256':inputs,'files':6027,'reused':len(reused),'new_outputs':len(fresh),
        'workers':4,'expected_new_wav_bytes':expected_bytes,'training_eligible':False,
        'source_mutations':False,'production_changed':False,
        'scope':'All target/ref files used by the pre-reaudit 5015-row pool, including quarantined suspects; '
                'not the entire unrelated raw download pool. Preserve source IDs and exclusions.'})
    print('PREPARED_FULL',6027,len(reused),len(fresh),expected_bytes,sha(work/'plan.json'),flush=True)


def collect_full(work):
    import soundfile as sf
    plan=read(work/'plan.json');verify(plan['inputs_sha256'])
    cases=read(work/'cases.json')
    mapping={r['review_id']:r for r in read(work/'reused.json').values()}
    for shard in range(4):
        directory=work/f'shard{shard}';summary=read(directory/'summary.json')
        expected=read(directory/'cases.json')
        if summary['completed']!=len(expected):raise ValueError('Incomplete shard')
        verify(summary['outputs_sha256'])
        paths=sorted((directory/'items').glob('*.json'),key=lambda p:int(p.stem))
        if len(paths)!=len(expected):raise ValueError('Missing or extra results')
        for p,case in zip(paths,expected):
            item=read(p)
            if item['case']!=case or case['review_id'] in mapping:raise ValueError('Duplicate or changed input')
            info=sf.info(item['vocals'])
            if info.samplerate!=16000 or info.channels!=1 or info.frames!=item['frames']:
                raise ValueError('Separated output format or duration changed')
            mapping[case['review_id']]={'path':str(p),'sha256':sha(p),'review_id':case['review_id']}
    if set(mapping)!=set(range(6027)):raise ValueError('Incomplete separation coverage')
    save(work/'results.json',mapping)
    save(work/'summary.json',{'completed':len(mapping),'reused':plan['reused'],'new':plan['new_outputs'],
        'training_eligible':False,'quality_review_pending':True,'source_mutations':False,
        'results_sha256':sha(work/'results.json'),'time':time.time()})
    print('COMPLETE_FULL',len(mapping),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','prepare-full','collect-full'))
    parser.add_argument('--work',type=Path,default=ROOT/'separation_pilot')
    args=parser.parse_args();{'prepare':prepare,'run':run,'prepare-full':prepare_full,'collect-full':collect_full}[args.action](args.work)
