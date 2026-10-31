"""Bounded r31 failure diagnosis: reuse endpoints, vary strength or punctuation only."""
import argparse
from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import control_ab_strength as strength
import fresh_baseline_eval as baseline
from clean_candidate_run import stop

frozen, fast, review = strength.frozen, strength.fast, strength.review
read, save = review.read_json, baseline.save_new
SOURCE = Path('/root/autodl-tmp/granary_pilot_20261030/run')
SOURCE_PLAN_SHA = '68cb179fadf983ff130763b07e0956aee0240876062f9cabbc72970111cc7e58'
SENTENCES = ('fresh_th_1983', 'fresh_th_1994', 'fresh_id_1816', 'fresh_id_1707')
R39_EARLY = Path('/root/autodl-tmp/r39_early_20261031')
R39_EARLY_PLAN_SHA = 'f7e5ca1526e5e79e648f73292f0d61b33b8151e5a1c81a49ce6fb249675cdbef'
R39_EARLY_VERIFICATION_SHA = '048735b2b6fdaeff2a6ea575ca05971cc4204bd20ba782a91fb5bec6699f7588'


def key(row):
    return row['case_id'], row['seed']


def punctuated(case):
    text = case['text']
    if '(' not in text or text.count('(') != text.count(')'):
        raise ValueError('Expected balanced body parentheses')
    edited = text.replace('(', ', ').replace(')', ',')
    if frozen.evaluation._norm(edited) != frozen.evaluation._norm(text):
        raise ValueError('Punctuation probe changed spoken characters')
    return {**case, 'case_id': case['case_id'] + '_commas', 'text': edited,
            'parent_case_id': case['case_id'], 'probe': 'body_parentheses_to_commas'}


def make_jobs(cases, completed):
    jobs = []
    for case in cases:
        for seed in frozen.SEEDS:
            if (case['case_id'], seed) not in completed:
                jobs.append(dict(view='original_r31', model='r31', strength=1., case=case, seed=seed))
            jobs.append(dict(view='half_r31', model='r31', strength=.5, case=case, seed=seed))
    for case in cases:
        if '(' in case['text']:
            for model in ('base', 'r31'):
                for seed in frozen.SEEDS:
                    jobs.append(dict(view='commas_' + model, model=model,
                                     strength=0. if model == 'base' else 1.,
                                     case=punctuated(case), seed=seed))
    return [{'id': f"{j['view']}_{j['case']['case_id']}_{j['seed']}", **j} for j in jobs]


def verify(report, plan, case, model, value, seed):
    strength.verify_report(report, {**plan, 'target': plan['targets'][model]}, [case], value, [seed])


def prepare(work):
    if work.exists():
        raise FileExistsError(work)
    review.checked(SOURCE / 'plan.json', SOURCE_PLAN_SHA)
    parent = read(SOURCE / 'plan.json')
    if read(SOURCE / 'status.json')['stage'] != 'stopped_by_user':
        raise ValueError('r31 must remain stopped')
    cases = [r for r in frozen._read_manifest(SOURCE / 'cases.jsonl') if r['sentence_id'] in SENTENCES]
    cases.sort(key=lambda c: (SENTENCES.index(c['sentence_id']), c['condition']))
    if len(cases) != 8:
        raise ValueError('Expected four texts, plain and control')
    for sid in SENTENCES:
        a, b = [c for c in cases if c['sentence_id'] == sid]
        if {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} != {'case_id', 'control', 'condition'}:
            raise ValueError('Control conditions differ in more than the prefix')
    refs = {c['ref_audio'] for c in cases}
    inputs = {p: h for p, h in parent['inputs_sha256'].items()
              if '/models--' in p or p in refs}
    trained = read(SOURCE / 'trained.json')
    inputs.update(trained['inputs_sha256'])
    for p in (SOURCE / 'plan.json', SOURCE / 'trained.json', SOURCE / 'cases.jsonl', SOURCE / 'flagged.json'):
        inputs[str(p)] = frozen.sha256(p)
    for folder in ('src', 'scripts', 'third_party/VoxCPM/src'):
        for p in (frozen.ROOT / folder).rglob('*.py'):
            inputs[str(p)] = frozen.sha256(p)
    inputs[str(Path(__file__).with_suffix('.sh'))] = frozen.sha256(Path(__file__).with_suffix('.sh'))
    plan = {k: parent[k] for k in ('base_path', 'asr_path', 'speaker_model_path',
                                  'cfg_value', 'inference_timesteps', 'evaluation_seeds')}
    plan.update(targets={k: parent['targets'][k] for k in ('base', 'r30', 'r31')},
                training_steps=0, production_changed=False, automatic_training=False, max_workers=4)
    selected = {c['case_id']: c for c in cases}
    reused = []
    sources = [(name, SOURCE / f'{name}_report.json') for name in ('base', 'r30')]
    snapshot = read(SOURCE.parent / 'stopped_20261030/stop_snapshot.json')
    sources += [('r31', Path(p)) for p in snapshot['item_sha256']]
    for model, path in sources:
        doc = read(path)
        if model == 'r31':
            review.checked(path, snapshot['item_sha256'][str(path)])
        else:
            review.checked(path, parent['inputs_sha256'][str(path)])
        for row in doc['items']:
            if row['case_id'] not in selected:
                continue
            single = {**doc, 'items': [row], 'wav_sha256': {row['wav']: doc['wav_sha256'][row['wav']]}}
            verify(single, plan, selected[row['case_id']], model, 0. if model == 'base' else 1., row['seed'])
            reused.append({'view': 'original_' + model, 'source_report': str(path),
                           'item': row, 'wav_sha256': single['wav_sha256']})
            inputs[str(path)] = frozen.sha256(path)
            inputs.update(single['wav_sha256'])
    completed = {key(r['item']) for r in reused if r['view'] == 'original_r31'}
    jobs = make_jobs(cases, completed)
    if Counter(r['view'] for r in reused) != Counter(original_base=40, original_r30=40, original_r31=38):
        raise ValueError('Unexpected frozen endpoint coverage')
    if Counter(j['view'] for j in jobs) != Counter(original_r31=2, half_r31=40, commas_base=30, commas_r31=30):
        raise ValueError('Unexpected generation budget')
    for path, digest in inputs.items():
        review.checked(path, digest)
    if shutil.disk_usage(work.parent).free < 4.5 * 1024**3:
        raise RuntimeError('Need 4.5 GiB before probe; preserve 4 GiB during execution')
    work.mkdir()
    for name, obj in (('cases.json', cases), ('reused.json', reused), ('jobs.json', jobs)):
        save(work / name, obj)
        inputs[str(work / name)] = frozen.sha256(work / name)
    plan.update(new_outputs=len(jobs), reused_outputs=len(reused), inputs_sha256=inputs,
                authorized='User continued the recorded bounded diagnosis on 2026-10-30.',
                note='Outcome-selected ID/TH development probe, not language acceptance. '
                     'Only two missing original endpoints generated in this separate run. '
                     'Other old WAVs/ASR/scores untouched. Half strength and punctuation are separate arms. '
                     'No training, TL quality conclusion, production changes or implicit retries.',
                diagnostic_near_silence_rms_dbfs=-60., diagnostic_cer_tolerance=.01)
    save(work / 'plan.json', plan)
    save(work / 'status.json', {'stage': 'prepared', 'new_outputs': len(jobs), 'time': time.time()})
    print('PREPARED 102 new outputs; 118 existing endpoints reused', flush=True)


def worker(work, index):
    plan = strength.check_inputs(work)
    directory = work / f'worker{index}'
    frozen.runtime_environment(directory / 'runtime')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      VOXFT_WHISPER_MODEL_LARGE=plan['asr_path'], VOXFT_SPK_EMB_MODEL=plan['speaker_model_path'],
                      TORCHINDUCTOR_CACHE_DIR=str(directory / 'runtime/inductor'))
    frozen.evaluation.EVAL_DIR = directory / 'raw_reports'
    frozen.evaluation.infer.CHECKPOINT_DIR = directory
    cases = read(work / 'cases.json')
    model_name = 'base' if index == 2 else 'r31'
    with fast.cached_services(cases):
        save(directory / 'runtime.json', baseline.gpu_services(plan, cases, model_name))
        if model_name == 'r31':
            model = frozen.evaluation.infer.get_model(plan['base_path'], plan['targets']['r31'])
            save(directory / 'scaling.json', strength.check_scaling(model))
        for job in read(work / 'jobs.json'):
            if job['model'] != model_name or not fast.claim(work / 'claims', job['id']):
                continue
            if shutil.disk_usage(work).free < 4 * 1024**3:
                raise RuntimeError('Less than 4 GiB free')
            report = frozen.evaluation.evaluate(
                plan['targets'][model_name], 'th', [job['case']], base=plan['base_path'],
                seeds=[job['seed']], cfg_value=1.8, inference_timesteps=20, lora_strength=job['strength'])
            report['wav_sha256'] = {r['wav']: frozen.sha256(r['wav']) for r in report['items']}
            verify(report, plan, job['case'], model_name, job['strength'], job['seed'])
            save(work / 'items' / (job['id'] + '.json'), report)
            frozen.write_json(directory / 'progress.json', {'last_job': job['id'], 'time': time.time()})
            print('DONE', job['id'], flush=True)
    strength.check_inputs(work)


def flags(row):
    return {'empty_asr': not frozen.evaluation._norm(row['hyp']),
            'near_silence': row['rms_dbfs'] < -60 and (row.get('speech_ratio') or 0) < .05,
            'suspected_overread': bool(row['over_read'])}


def collect(work):
    plan = strength.check_inputs(work)
    if (work / 'comparison.json').exists():
        raise FileExistsError(work / 'comparison.json')
    jobs = read(work / 'jobs.json')
    files = {p.stem: p for p in (work / 'items').glob('*.json')}
    if files.keys() != {j['id'] for j in jobs}:
        raise ValueError('Missing or extra probe outputs')
    views = {}
    for old in read(work / 'reused.json'):
        review.checked(old['item']['wav'], old['wav_sha256'][old['item']['wav']])
        views.setdefault(old['view'], []).append(deepcopy(old['item']))
    for job in jobs:
        doc = read(files[job['id']])
        verify(doc, plan, job['case'], job['model'], job['strength'], job['seed'])
        views.setdefault(job['view'], []).extend(doc['items'])
    levels = {r['wav']: review.level_metrics(r['wav']) for rows in views.values() for r in rows}
    summaries, indices = {}, {}
    for name, rows in views.items():
        for row in rows:
            row.update(levels[row['wav']])
            row['diagnostic_flags'] = flags(row)
        indices[name] = {(r.get('parent_case_id', r['case_id']), r['seed']): r for r in rows}
        if len(indices[name]) != len(rows):
            raise ValueError('Duplicate paired output')
        summaries[name] = {**frozen.evaluation._report_metrics(rows),
                           'diagnostic_counts': {f: sum(r['diagnostic_flags'][f] for r in rows)
                                                 for f in flags(rows[0])}}
    paired = []
    for k, row in sorted(indices['original_r31'].items()):
        paired.append({'case_id': k[0], 'seed': k[1], 'lang': row['lang'], 'condition': row['condition'],
                       'models': {name: {field: index[k].get(field) for field in
                                        ('cer', 'hyp', 'text', 'audio_sec', 'rms_dbfs', 'speaker_sim',
                                         'diagnostic_flags', 'wav')}
                                  for name, index in indices.items() if k in index}})
    save(work / 'comparison.json', {'summaries': summaries, 'paired': paired,
         'levels': levels, 'production_changed': False, 'training_steps': 0,
         'language_quality': 'unverified', 'note': plan['note'],
         'inputs_sha256': {str(work / 'plan.json'): frozen.sha256(work / 'plan.json'),
                           **{str(p): frozen.sha256(p) for p in files.values()}}})
    strength.check_inputs(work)


def run(work):
    plan = strength.check_inputs(work)
    if plan['new_outputs'] != 102 or plan['training_steps'] != 0 or plan['automatic_training']:
        raise ValueError('Probe scope changed')
    if (work / 'run.started').exists():
        raise FileExistsError('No implicit restart')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy')
    with (work / 'run.started').open('x', encoding='utf-8') as stream:
        stream.write(frozen.sha256(work / 'plan.json'))
    for name in ('claims', 'items'):
        (work / name).mkdir()
    processes = []
    def status(stage, **extra):
        frozen.write_json(work / 'status.json', {'stage': stage, 'time': time.time(),
            'completed': len(list((work / 'items').glob('*.json'))), 'expected': 102,
            'pids': [p.pid for p in processes if p.poll() is None], 'training_steps': 0, **extra})
    try:
        for index in range(4):
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            if free < 12 * 1024:
                raise RuntimeError('Insufficient GPU memory for next worker')
            with (work / f'worker{index}.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', str(Path(__file__).resolve()),
                    'worker', '--work', str(work), '--index', str(index)], cwd=frozen.ROOT,
                    stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            while not (work / f'worker{index}/progress.json').exists() and process.poll() is None:
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('Worker startup failed')
                status('starting')
                time.sleep(2)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Worker failed; no retries')
            status('evaluating')
            time.sleep(10)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('Worker failed')
        collect(work)
        status('complete')
    except BaseException as error:
        status('failed', error=str(error))
        raise
    finally:
        for process in processes:
            stop(process)


def punctuation_jobs(cases):
    if (len(cases) != 2 or {c['sentence_id'] for c in cases} != {'fresh_id_1973'}
            or {c['condition'] for c in cases} != {'plain', 'control'}):
        raise ValueError('Expected ID1973 plain/control pair')
    jobs = []
    for model in ('base', 'candidate'):
        for case in sorted(cases, key=lambda c: c['condition'] != 'plain'):
            for seed in frozen.SEEDS:
                changed = punctuated(case)
                jobs.append({'id':f"commas_{model}_{case['case_id']}_{seed}",
                             'model':model, 'case':changed, 'seed':seed,
                             'strength':0. if model == 'base' else 1.})
    return jobs


def prepare_r39_punctuation(work):
    if work.exists():
        raise FileExistsError(work)
    review.checked(R39_EARLY/'run/plan.json', R39_EARLY_PLAN_SHA)
    review.checked(R39_EARLY/'verification_terminal.json', R39_EARLY_VERIFICATION_SHA)
    verified = read(R39_EARLY/'verification_terminal.json')
    if (verified['status'] != 'diagnostic_stopped_verified' or verified['completed_outputs'] != 197
            or verified['reason'] != 'two_large_paired_cer_regressions' or verified['unscored_wavs'] != 0):
        raise ValueError('Expected verified r39 early failure')
    for path,digest in verified['outputs_sha256'].items():
        review.checked(path,digest)
    parent = strength.check_inputs(R39_EARLY/'run')
    result = read(R39_EARLY/'run/result.json')
    cases = list({j['case']['case_id']:j['case'] for j in read(R39_EARLY/'run/jobs.json')
                  if j['case']['sentence_id'] == 'fresh_id_1973'}.values())
    jobs = punctuation_jobs(cases)
    reused = {name:[r for r in result[field] if r['sentence_id'] == 'fresh_id_1973']
              for name,field in (('base','matched_base'), ('candidate','rows'))}
    expected = {(c['case_id'],s) for c in cases for s in frozen.SEEDS}
    for rows in reused.values():
        if len(rows) != 10 or {key(r) for r in rows} != expected:
            raise ValueError('Incomplete existing five-seed endpoints')
    inputs = dict(parent['inputs_sha256'])
    inputs.update(verified['outputs_sha256'])
    for path in (R39_EARLY/'run/plan.json', R39_EARLY/'verification_terminal.json'):
        inputs[str(path)] = frozen.sha256(path)
    for folder in ('src','scripts','third_party/VoxCPM/src'):
        for path in (frozen.ROOT/folder).rglob('*'):
            if path.suffix in ('.py','.sh'):
                inputs[str(path)] = frozen.sha256(path)
    for path,digest in inputs.items():
        review.checked(path,digest)
    if shutil.disk_usage(work.parent).free < 4.5*1024**3:
        raise RuntimeError('Need 4.5 GiB before punctuation probe')
    work.mkdir()
    for name,value in (('jobs.json',jobs), ('cases.json',cases), ('reused.json',reused)):
        save(work/name,value)
        inputs[str(work/name)] = frozen.sha256(work/name)
    save(work/'plan.json', {**{k:parent[k] for k in ('base_path','asr_path','speaker_model_path','targets')},
        'inputs_sha256':inputs, 'profile':'r39_early_punctuation', 'new_outputs':20, 'reused_outputs':20,
        'training_steps':0, 'max_workers':1, 'seeds':frozen.SEEDS, 'tail_policy':'absolute',
        'paired_extreme_acoustics':True, 'native_quality':'unverified', 'production_changed':False,
        'final_holdout_used':False, 'checkpoint_optimizer_updates':80,
        'early_rejection':'Per-output severe gates within each model against original input, and candidate '
                          'against same changed-text base; full five-seed condition groups use absolute '
                          'and paired-extreme acoustic gates. Mean CER observation only.',
        'note':'ID1973 outcome-selected diagnostic. Parentheses to commas only; preserve every word, '
               'quote, reference, control and seed. Base first, then candidate, no retries. '
               'Original 197-output rejection unchanged. No training or production normalizer change; '
               'success does not establish original-input candidate quality or TH/TL improvement.'})
    print('PREPARED r39 punctuation: 20 existing endpoints, at most20 new', flush=True)


def punctuation_rejection(rows, reused, *, stage_end=False):
    import staged_candidate_run as staged
    groups = {model:[r['item'] for r in rows if r['model']==model] for model in ('base','candidate')}
    last = rows[-1]
    model, latest = last['model'], last['item']
    original = {(r['case_id']+'_commas',r['seed']):r for r in reused[model]}
    comparisons = [(model+'_vs_original',groups[model],original)]
    if model == 'candidate':
        comparisons.append(('candidate_vs_commas_base',groups['candidate'],{key(r):r for r in groups['base']}))
    for label,current,baseline_rows in comparisons:
        reason = staged.probe.reject_reason(current,baseline_rows)
        if reason:
            return label+':'+reason, {}
        if stage_end:
            group = [r for r in current if r['case_id']==latest['case_id']]
            if {r['seed'] for r in group} != set(frozen.SEEDS) or len(group) != 5:
                raise ValueError('Incomplete condition group')
            acoustic = staged.acoustic_gate(group,baseline_rows,paired_extremes=True)
            if acoustic:
                return label+':stage_acoustic_regression', acoustic
    return None, {}


def run_r39_punctuation(work):
    plan = strength.check_inputs(work)
    if (plan['profile'] != 'r39_early_punctuation' or plan['new_outputs'] != 20
            or plan['training_steps'] != 0 or plan['max_workers'] != 1):
        raise ValueError('Punctuation scope changed')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU busy')
    with (work/'run.started').open('x',encoding='utf-8') as stream:
        stream.write(frozen.sha256(work/'plan.json'))
    (work/'items').mkdir()
    frozen.runtime_environment(work/'runtime')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        VOXFT_WHISPER_MODEL_LARGE=plan['asr_path'],VOXFT_SPK_EMB_MODEL=plan['speaker_model_path'],
        TORCHINDUCTOR_CACHE_DIR=str(work/'runtime/inductor'))
    frozen.evaluation.EVAL_DIR = work/'raw_reports'
    frozen.evaluation.infer.CHECKPOINT_DIR = work
    jobs, reused, rows, started_models = read(work/'jobs.json'), read(work/'reused.json'), [], set()
    reason, acoustic = None, {}
    def status(stage, **extra):
        frozen.write_json(work/'status.json', {'stage':stage,'completed':len(rows),'expected':20,
            'pid':os.getpid(),'time':time.time(),'reason':reason,**extra})
    try:
        status('starting')
        with fast.cached_services(read(work/'cases.json')):
            for i,job in enumerate(jobs):
                model = job['model']
                if model not in started_models:
                    save(work/f'runtime_{model}.json',baseline.gpu_services(plan,[j['case'] for j in jobs],model))
                    started_models.add(model)
                if shutil.disk_usage(work).free < 4*1024**3:
                    raise RuntimeError('Disk reserve reached')
                report = frozen.evaluation.evaluate(plan['targets'][model],'id',[job['case']],
                    base=plan['base_path'],seeds=[job['seed']],cfg_value=1.8,inference_timesteps=20,
                    lora_strength=job['strength'])
                row, = report['items']
                report['wav_sha256'] = {row['wav']:frozen.sha256(row['wav'])}
                report['level_metrics'] = review.level_metrics(row['wav'])
                verify(report,plan,job['case'],model,job['strength'],job['seed'])
                save(work/'items'/(job['id']+'.json'),report)
                rows.append({'model':model,'item':{**row,**report['level_metrics']}})
                reason,acoustic = punctuation_rejection(rows,reused,stage_end=(i+1)%5==0)
                status('diagnostic_stopped' if reason else 'evaluating',last_job=job['id'])
                print('DONE',job['id'],reason,flush=True)
                if reason:
                    break
        strength.check_inputs(work)
        if {str(p) for p in work.glob('auditions/*.wav')} != {r['item']['wav'] for r in rows}:
            raise ValueError('Unscored punctuation output')
        decision = 'diagnostic_stopped' if reason else 'diagnostic_complete'
        save(work/'result.json',{'decision':decision,'reason':reason,'acoustic_failures':acoustic,
            'rows':rows,'reused':reused,'completed':len(rows),'skipped':20-len(rows),
            'metrics':{name:frozen.evaluation._report_metrics(items) for name,items in
                       [(f'original_{m}',v) for m,v in reused.items()]+
                       [(f'commas_{m}',[r['item'] for r in rows if r['model']==m])
                        for m in ('base','candidate') if any(r['model']==m for r in rows)]},
            'wav_sha256':{r['item']['wav']:frozen.sha256(r['item']['wav']) for r in rows},
            'training_steps':0,'production_changed':False,'native_quality':'unverified','final_holdout_used':False})
        status(decision)
    except BaseException as error:
        status('failed',error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'collect'))
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=range(4))
    parser.add_argument('--profile', choices=('r31','r39_early_punctuation'), default='r31')
    args = parser.parse_args()
    if args.profile == 'r39_early_punctuation':
        if args.action not in ('prepare','run'):
            parser.error('Punctuation profile uses serial prepare/run only')
        {'prepare':prepare_r39_punctuation,'run':run_r39_punctuation}[args.action](args.work)
    elif args.action == 'worker':
        if args.index is None:
            parser.error('worker needs --index')
        worker(args.work, args.index)
    else:
        {'prepare': prepare, 'run': run, 'collect': collect}[args.action](args.work)
