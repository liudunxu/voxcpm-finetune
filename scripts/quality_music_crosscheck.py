"""Bounded independent CLAP review of remaining music risks; never certify or train."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path('/root/autodl-tmp/quality_reaudit_20261030')
MUSIC = Path('/root/autodl-tmp/music_audit_20261029/run')
PROMPTS = (
    'A clean, dry recording of a single person speaking.',
    'A recording of a person speaking with music playing in the background.',
    'The sound of a clear human voice in a quiet room.',
    'The sound of a human voice talking over background music.',
    'A solo spoken voice recorded in a quiet studio.',
    'A spoken voice accompanied by background instrumental music.',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def manifest(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def balanced(rows, count):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r:(r['file_sha256'], r['audio'])):
        groups[row['source_id']].append(row)
    selected = [row for group in itertools.zip_longest(*(groups[k] for k in sorted(groups)))
                for row in group if row is not None][:count]
    if len(selected) != count:
        raise ValueError('Insufficient frozen stratum')
    return selected


def select_cases(current, excluded):
    groups = {
        'current_high_ast': balanced([r for r in current if r['music_score'] >= .7], 80),
        'current_mid_ast': balanced([r for r in current if .5 <= r['music_score'] < .7], 40),
        'current_low_ast_control': balanced([r for r in current if r['music_score'] < .05], 40),
        'previously_excluded_control': balanced(excluded, 16),
    }
    selected = [{**row, 'role':role} for role, rows in groups.items() for row in rows]
    if len({r['audio'] for r in selected}) != 176 or len({r['pcm_sha256'] for r in selected}) != 176:
        raise ValueError('Duplicated target/control audio or PCM')
    return selected


def expand_cases(current, previous):
    cases = list(previous)
    seen = {r['audio'] for r in cases}
    for row in sorted(current,key=lambda r:(r['file_sha256'],r['audio'])):
        if row['music_score'] < .5 or row['audio'] in seen:
            continue
        cases.append({**row, 'role':'current_high_ast' if row['music_score'] >= .7 else 'current_mid_ast'})
        seen.add(row['audio'])
    if len(seen) != len(cases) or len({r['pcm_sha256'] for r in cases}) != len(cases):
        raise ValueError('Duplicate crosscheck cases')
    return cases


def prompt_margins(scores):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all() or np.any(np.abs(values) > 1.00001):
        raise ValueError('Expected six finite cosine scores')
    margins = values[1::2] - values[::2]
    return {'paired_music_minus_clean':margins.tolist(),
            'all_three_music_higher':bool(np.all(margins > 0))}


def windows(samples, rate):
    if samples <= 0 or rate <= 0:
        raise ValueError('Invalid audio')
    last = max(0, samples - rate*10)
    starts = list(range(0, last+1, rate*5))
    if starts[-1] != last:
        starts.append(last)
    return [(start,min(samples,start+rate*10)) for start in starts]


def checked(work):
    plan = read(work/'plan.json')
    for path,digest in plan['inputs_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen crosscheck input changed: '+path)
    return plan


def prepare(work, expand=False):
    if work.exists():
        raise FileExistsError(work)
    assets = read(ROOT/'clap_assets.json')
    if assets['revision'] != '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a':
        raise ValueError('Wrong independent model')
    old = read(MUSIC/'summary.json')['review_priority']
    excluded = [{**r['case'], 'music_score':r['scores']['Music']} for r in old]
    current = manifest(ROOT/'rows.jsonl')
    cases = select_cases(current, excluded)
    inputs = {str(p):sha(p) for p in (ROOT/'clap_assets.json', ROOT/'plan.json', ROOT/'summary.json',
              ROOT/'rows.jsonl', MUSIC/'summary.json', Path(__file__))}
    inputs.update(assets['files_sha256'])
    reused=[]
    if expand:
        pilot=ROOT/'clap_crosscheck'
        if sha(pilot/'plan.json') != '921597a908c4b9d2168120a6277eeb52e650d9d000baf004dce087b743f31b95':
            raise ValueError('Wrong frozen pilot')
        checked(pilot)
        if read(pilot/'summary.json')['completed'] != 176 or read(pilot/'cases.json') != cases:
            raise ValueError('Pilot coverage changed')
        for p in sorted((pilot/'items').glob('*.json')):
            result=read(p)
            if result['case'] != cases[len(reused)]:
                raise ValueError('Pilot output or order changed')
            reused.append(result)
            inputs[str(p)]=sha(p)
        if len(reused) != 176:
            raise ValueError('Incomplete pilot')
        for p in (pilot/'plan.json',pilot/'summary.json',pilot/'cases.json'):
            inputs[str(p)]=sha(p)
        cases=expand_cases(current,cases)
        if len(cases) != 1087:
            raise ValueError('Expected all 1031 current AST>=.5 plus 56 frozen controls')
    for case in cases:
        if sha(case['audio']) != case['file_sha256']:
            raise ValueError('Source audio changed')
        inputs[case['audio']] = case['file_sha256']
    if shutil.disk_usage(ROOT).free < 4.2*1024**3:
        raise RuntimeError('Retain 4 GiB disk reserve')
    work.mkdir()
    save(work/'cases.json', cases)
    inputs[str(work/'cases.json')] = sha(work/'cases.json')
    save(work/'reused.json',reused)
    inputs[str(work/'reused.json')]=sha(work/'reused.json')
    save(work/'plan.json', {'model_path':assets['model_path'], 'inputs_sha256':inputs,
        'prompts':list(PROMPTS), 'outputs':len(cases), 'reused_outputs':len(reused),
        'new_outputs_max':len(cases)-len(reused), 'roles':dict(Counter(r['role'] for r in cases)),
        'sample_rate':48000, 'window_seconds':10, 'stride_seconds':5, 'dtype':'float32',
        'training_started':False, 'source_mutations':False, 'production_changed':False,
        'policy':'Cosine prompt contrast is uncalibrated review evidence, not music probability or '
                 'quality certification. Previous exclusions are uncertain controls, not ground truth. '
                 'No source rewrite, re-admission, automatic training or final holdout use. '
                 'Selection uses fixed AST strata and file hashes before CLAP outputs; all results kept.'})
    checked(work)
    print(f'PREPARED {len(cases)} fixed music crosschecks, {len(reused)} reused', flush=True)


def run(work):
    import torch
    import soundfile as sf
    from torchaudio.functional import resample
    from transformers import ClapModel, ClapProcessor

    plan = checked(work)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    if torch.cuda.mem_get_info()[0] < 6*1024**3:
        raise RuntimeError('Need 6 GiB free for the independent audio model')
    torch.cuda.set_per_process_memory_fraction(.10)
    with (work/'run.started').open('x',encoding='utf-8') as stream:
        stream.write(sha(work/'plan.json'))
    (work/'items').mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    processor = ClapProcessor.from_pretrained(plan['model_path'], local_files_only=True)
    model, loading = ClapModel.from_pretrained(plan['model_path'], local_files_only=True,
        use_safetensors=False, weights_only=True, output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Model did not load exactly: '+str(loading))
    model.eval().to(device='cuda',dtype=torch.float32)

    def vectors(value):
        # Transformers 4 returns tensors; 5 returns projected pooler outputs.
        tensor = value if torch.is_tensor(value) else value.pooler_output
        return torch.nn.functional.normalize(tensor,dim=-1)

    with torch.inference_mode():
        text = vectors(model.get_text_features(**processor(text=plan['prompts'], return_tensors='pt',
                                                           padding=True).to('cuda')))
    if processor.feature_extractor.sampling_rate != 48000:
        raise ValueError('Unexpected sample rate')
    reused={r['case']['audio']:r for r in read(work/'reused.json')}
    results=[]
    for i,case in enumerate(read(work/'cases.json')):
        if case['audio'] in reused:
            result=reused[case['audio']]
            if result['case'] != case:
                raise ValueError('Reused input differs')
            results.append(result)
            continue
        if shutil.disk_usage(work).free < 4*1024**3 or sha(case['audio']) != case['file_sha256']:
            raise RuntimeError('Disk reserve reached or audio changed')
        wav,sr=sf.read(case['audio'],dtype='float32',always_2d=True)
        if sr != 16000 or wav.shape[1] != 1 or not np.isfinite(wav).all():
            raise ValueError('Unexpected source PCM')
        measured=[]
        for start,end in windows(len(wav),sr):
            audio=resample(torch.from_numpy(wav[start:end,0].copy()),sr,48000).numpy()
            batch=processor(audio=audio,sampling_rate=48000,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                embedding=vectors(model.get_audio_features(**batch))
                scores=(embedding @ text.T)[0].float().cpu().numpy()
            measured.append({'start_sample':start,'end_sample':end,'cosine_scores':scores.tolist(),
                             **prompt_margins(scores)})
        result={'case':case,'windows':measured,
                'crossmodel_review_priority':case['music_score'] >= .5 and any(
                    w['all_three_music_higher'] for w in measured),
                'status':'unverified_review_only'}
        save(work/'items'/f'{i:03d}.json',result)
        results.append(result)
        print('DONE',i+1,'/'+str(plan['outputs']),case['role'],flush=True)
    checked(work)
    save(work/'summary.json',{'completed':len(results),'reused_outputs':len(reused),
        'new_outputs':len(results)-len(reused),
        'by_role':{role:{'n':sum(r['case']['role']==role for r in results),
           'all_three_music_higher':sum(any(w['all_three_music_higher'] for w in r['windows'])
                for r in results if r['case']['role']==role),
           'crossmodel_review_priority':sum(r['crossmodel_review_priority'] for r in results if r['case']['role']==role)}
          for role in plan['roles']},
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'time':time.time(),
        'automatic_exclusions':0,'training_started':False,'source_mutations':False,
        'note':plan['policy']})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--work',type=Path,default=ROOT/'clap_crosscheck')
    parser.add_argument('--expand',action='store_true')
    args=parser.parse_args()
    if args.action == 'prepare':
        prepare(args.work,args.expand)
    else:
        run(args.work)
