"""2026-11-03 production-condition review: base vs r46/u10 vs surviving 459-step LoRAs.

Why: the r35–r48 "first new severe case vs base -> reject" gate rejects a quality-neutral
model 38–96% of the time (base itself is severe on ~2% of outputs and changing weights breaks
the same-seed pairing). Decide instead on net change (new − fixed) with a sign test, per language.

Phases (decision rules are frozen into plan.json at prepare time, before any output exists):
  1. short cues (45 1–4-word lines x zh-long / zh-1.8s / en refs x 5 seeds): base, r46u10, r28, r29, r30
     -> winner among r28/r29/r30 = fewest short-cue severe outputs (tie: lower mean CER)
  2. dev set fresh_text_holdout_20261028 (180 cases x 5 seeds) at production steps: base, r46u10, winner
  3. only if a candidate passes phases 1+2: one-time reserved_next_20261102 (90 x 5 seeds), base vs it.
Production approximation: cfg 1.8, best_of=1, no retry; steps by ref duration (<2s 28, 2–5s 24, >=5s 22).
Not covered (recorded as limitations): ZipEnhancer ref denoise, real DIS separated-vocal refs
(only voice_id cached server-side), server termination.reason (VoxCPM API here does not expose it).
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/prod_review_20261103')
REPO = Path('/root/voxcpm-finetune')
PY = str(REPO / '.venv/bin/python')
SEEDS = [42, 43, 44, 45, 46]
WORKERS = 4
LANGS = ('id', 'th', 'tl')
MODELS = {
    'base': 'base',
    'r46u10': '/root/autodl-tmp/r46_micro1_20261031/run/checkpoint/step_0000009',
    'r28': '/root/autodl-tmp/clean_candidate_20261028/run/checkpoint/latest',
    'r29': '/root/autodl-tmp/suspect_candidate_20261029/run/checkpoint/latest',
    'r30': '/root/autodl-tmp/full_clean_candidate_20261029/run/checkpoint/latest',
    # follow-up only (scripts/prod_review_r49_20261103.py); not part of the frozen phase lists
    'r49': '/root/autodl-tmp/r49_segments_20261103/checkpoint/latest',
    'r50': '/root/autodl-tmp/r50_3ep_20261103/checkpoint/latest',
    'r51': '/root/autodl-tmp/r51_th_restore_20261104/checkpoint/latest',
}
PHASE1 = ['base', 'r46u10', 'r28', 'r29', 'r30']
WINNER_POOL = ['r28', 'r29', 'r30']
ZH = '/root/autodl-tmp/voxft_data/processed/fleurs_zh/audio/run_1owk1jss/'
EN = '/root/autodl-tmp/voxft_data/processed/fleurs_en/audio/run_vha8op9_/'
DEV_CASES = REPO / 'eval_cases/fresh_text_holdout_20261028.jsonl'
RESERVED_CASES = REPO / 'eval_cases/reserved_next_20261102.jsonl'

# 1–4 word drama cues; includes the UAT failures (Sige!/Pa./Baik!/ได้เลย!/พ่อ).
SHORT_TEXTS = {
    'tl': ['Sige!', 'Pa.', 'Ano?', 'Oo naman.', 'Hindi!', 'Salamat po.', 'Tara na!', 'Bakit?',
           'Ayoko!', 'Teka lang.', 'Ikaw pala!', 'Totoo ba?', 'Halika rito.', 'Wala na.', 'Diyos ko!'],
    'id': ['Baik!', 'Apa?', 'Ya.', 'Tidak!', 'Terima kasih.', 'Ayo pergi!', 'Kenapa?',
           'Tunggu sebentar.', 'Siapa kamu?', 'Benarkah?', 'Sudahlah.', 'Ibu!', 'Maaf.', 'Jangan!',
           'Aku tahu.'],
    'th': ['ได้เลย!', 'พ่อ', 'อะไรนะ?', 'ไม่!', 'ขอบคุณค่ะ', 'ไปกันเถอะ', 'ทำไม?', 'รอเดี๋ยว',
           'คุณเป็นใคร?', 'จริงเหรอ?', 'พอแล้ว', 'แม่!', 'ขอโทษ', 'อย่านะ!', 'ฉันรู้แล้ว'],
}

RULES = {
    'severe_dev': 'empty hyp | over_read | suspected_truncation | cer>0.3 | audio_sec>1.8x pooled per-case median',
    'severe_short': 'empty hyp | over_read | suspected_truncation | cer>0.5 | audio_sec>max(2.0, 2.5x pooled per-case median)',
    'pooled_median': 'median audio_sec of all models in that phase for the same case_id (model-neutral)',
    'winner': 'among r28/r29/r30: fewest phase-1 short severe outputs summed over id/th/tl; tie -> lower short mean CER',
    'worse_lang': 'per set: one-sided sign test new>fixed p<0.10, or mean CER up >0.01 (dev) / >0.03 (short)',
    'better_lang': 'per set: one-sided sign test fixed>new p<0.05',
    'pass': 'no language worse on dev or short; >=1 language better on either set',
    'replace_production': 'production runs r46u10 (user, 2026-11-03): winner must pass vs base AND have no worse language vs r46u10 (same worse rule)',
    'phase3': 'only for the best passing candidate (winner first, else r46u10); reserved 90x5 must show no worse language; '
              'reserved set is used once and never for later tuning',
    'steps': '<2s ref 28, 2-5s 24, >=5s 22; cfg 1.8; retry disabled',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def steps_for(ref):
    import soundfile as sf
    dur = sf.info(ref).duration
    return (28 if dur < 2 else 24 if dur < 5 else 22), round(dur, 3)


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def prepare():
    import numpy as np
    import soundfile as sf
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'refs').mkdir()
    src = ZH + '0000763.wav'
    wav, sr = sf.read(src, dtype='float32')
    win, hop = int(1.8 * sr), int(0.05 * sr)
    energy = [float(np.sqrt(np.mean(wav[i:i + win] ** 2))) for i in range(0, len(wav) - win, hop)]
    start = int(np.argmax(energy)) * hop
    crop = wav[start:start + win].copy()
    fade = int(0.01 * sr)
    crop[:fade] *= np.linspace(0, 1, fade)
    crop[-fade:] *= np.linspace(1, 0, fade)
    short_ref = ROOT / 'refs/zh_short_1p8s.wav'
    sf.write(short_ref, crop, sr)
    refs = {'zh_long': ZH + '0001308.wav', 'zh_short': str(short_ref), 'en_long': EN + '0001225.wav'}
    short = []
    for lang, texts in SHORT_TEXTS.items():
        for i, text in enumerate(texts):
            for arm, ref in refs.items():
                steps, dur = steps_for(ref)
                short.append({'case_id': f'short_{lang}_{i:02d}_{arm}', 'sentence_id': f'short_{lang}_{i:02d}',
                              'lang': lang, 'text': text, 'ref_audio': ref, 'ref_arm': arm,
                              'ref_lang': arm[:2], 'ref_sec': dur, 'control': '', 'cfg_value': 1.8,
                              'inference_timesteps': steps, 'numeric': False, 'cohort': 'short_cue_20261103',
                              'evaluation_use': 'development'})
    write_jsonl(ROOT / 'short_cases.jsonl', short)
    for name, src_cases in (('dev', DEV_CASES), ('reserved', RESERVED_CASES)):
        rows = []
        for row in read_jsonl(src_cases):
            steps, dur = steps_for(row['ref_audio'])
            rows.append({**row, 'cfg_value': 1.8, 'inference_timesteps': steps, 'ref_sec': dur})
        write_jsonl(ROOT / f'{name}_cases.jsonl', rows)
    # Text exposure of the short cues in each candidate's training manifest (recorded, not a gate).
    exposure = {}
    for label, target in MODELS.items():
        if target == 'base':
            continue
        run = Path(target).parents[1] if 'checkpoint' in target else Path(target).parent
        texts = set()
        for manifest in run.glob('**/train*.jsonl'):
            for row in read_jsonl(manifest):
                t = row.get('text', '')
                texts.add(t.split(')', 1)[-1].strip() if t.startswith('(') else t.strip())
        exposure[label] = sorted({t for ts in SHORT_TEXTS.values() for t in ts if t in texts})
    weights = {label: sha(Path(t) / 'lora_weights.safetensors') for label, t in MODELS.items() if t != 'base'}
    plan = {'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'models': MODELS, 'weights_sha256': weights,
            'seeds': SEEDS, 'rules': RULES, 'refs': refs, 'short_ref_crop_start_sec': start / sr,
            'inputs_sha256': {str(p): sha(p) for p in [ROOT / 'short_cases.jsonl', ROOT / 'dev_cases.jsonl',
                                                       ROOT / 'reserved_cases.jsonl', short_ref, DEV_CASES,
                                                       RESERVED_CASES, REPO / 'src/voxft/eval.py',
                                                       REPO / 'src/voxft/infer.py', Path(__file__)]},
            'short_text_training_exposure': exposure,
            'limitations': ['no ZipEnhancer ref denoise', 'FLEURS refs, not DIS separated vocals',
                            'termination.reason not exposed; runaway measured by duration',
                            'r26 (best in CPU rescoring) weights were deleted 2026-11-02 and cannot be evaluated']}
    (ROOT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'short': len(short), 'exposure': exposure}, ensure_ascii=False))


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with (ROOT / 'driver.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


# container limits: 20 CPU cores (cfs quota), 90GiB RAM -> keep CPU threads low, ASR on GPU
GEN_WORKERS = 6      # generation-only GPU processes (~6GB each)
ASR_PROCS = 1        # shared GPU faster-whisper large-v3 float16 process per ASR job (GPU memory headroom)
ASR_THREADS = 2
MAX_ASR_JOBS = 1
INTERIM_HARM_P = 0.005   # group-sequential harm boundary at 50% of dev texts (O'Brien-Fleming-like)
_asr_jobs = []


def _env(threads):
    import os
    return {**os.environ, 'PATH': f'{REPO}/.venv/bin:/root/.local/bin:/usr/bin:/bin', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8', 'OMP_NUM_THREADS': str(threads),
            'MKL_NUM_THREADS': str(threads), 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def _jsonl_items(pattern):
    items = {}
    for f in sorted(ROOT.glob(pattern)):
        for row in read_jsonl(f):
            items[(row['case_id'], row['seed'])] = row
    return items


def _expected(cases_name):
    return len(read_jsonl(ROOT / f'{cases_name}_cases.jsonl')) * len(SEEDS)


def _adopt_gpu_reports(cases_name, label):
    """Reuse audio from the v1 run (GPU-ASR shards); text metrics are redone on CPU like every model."""
    paths = []
    done_v1 = ROOT / f'done_{cases_name}_{label}.json'
    if done_v1.exists() and 'v2' not in json.loads(done_v1.read_text()):
        paths = [json.loads(done_v1.read_text())['report']]
    else:
        logs = sorted(ROOT.glob(f'log_{cases_name}_{label}_shard*.log'))
        hits = [[l for l in lp.read_text(encoding='utf-8').splitlines() if l.startswith('REPORT_PATH ')] for lp in logs]
        if logs and all(hits):
            paths = [h[-1].split(' ', 1)[1] for h in hits]
    if not paths:
        return False
    rows = [i for p in paths for i in json.loads(Path(p).read_text(encoding='utf-8'))['items']]
    if len(rows) != _expected(cases_name):
        return False
    (ROOT / 'gen').mkdir(exist_ok=True)
    write_jsonl(ROOT / f'gen/{cases_name}_{label}_adopted.jsonl',
                [{**r, 'adopted_from': 'v1_gpu_run', 'hyp_gpu_asr': r.get('hyp')} for r in rows])
    log(f'adopted {len(rows)} v1 outputs for {cases_name}/{label}')
    return True


def gen_stage(cases_name, label, attempts=3):
    for attempt in range(attempts):
        try:
            return _gen_stage(cases_name, label)
        except RuntimeError as exc:          # e.g. transient CUDA OOM in one shard; outputs so far are kept
            log(f'gen retry {attempt + 1}/{attempts} for {cases_name}/{label}: {exc}')
            if attempt == attempts - 1:
                raise
            time.sleep(20)


def _gen_stage(cases_name, label):
    if len(_jsonl_items(f'gen/{cases_name}_{label}_*.jsonl')) >= _expected(cases_name):
        return
    if _adopt_gpu_reports(cases_name, label):
        return
    (ROOT / 'gen').mkdir(exist_ok=True)
    # long sentences grow a gen worker to ~7.8GB: 6 OOM next to the ASR proc, 5 fit (measured 2026-11-03)
    n = GEN_WORKERS if cases_name == 'short' else 5
    procs = [subprocess.Popen([PY, '-X', 'utf8', '-u', str(Path(__file__).resolve()), 'gen', label, cases_name,
                               f'{k}/{n}'], cwd=REPO, env=_env(2),
                              stdout=(ROOT / f'log_gen_{cases_name}_{label}_{k}.log').open('a'),
                              stderr=subprocess.STDOUT) for k in range(n)]
    log(f'gen {cases_name}/{label} ({n} GPU workers)')
    codes = [p.wait() for p in procs]
    got = len(_jsonl_items(f'gen/{cases_name}_{label}_*.jsonl'))
    if any(codes) or got < _expected(cases_name):
        raise RuntimeError(f'gen {cases_name}/{label} exit {codes}, {got}/{_expected(cases_name)}')
    log(f'gen done {cases_name}/{label}')


def asr_stage_async(cases_name, label):
    """Launch CPU ASR for a finished generation job; returns immediately (pipelined with the GPU)."""
    if (ROOT / f'done_{cases_name}_{label}.json').exists() and \
            'v2' in json.loads((ROOT / f'done_{cases_name}_{label}.json').read_text()):
        return
    while sum(1 for j in _asr_jobs if j['procs'] and any(p.poll() is None for p in j['procs'])) >= MAX_ASR_JOBS:
        time.sleep(5)
    procs = [subprocess.Popen([PY, '-X', 'utf8', '-u', str(Path(__file__).resolve()), 'asr', label, cases_name,
                               f'{k}/{ASR_PROCS}'], cwd=REPO, env=_env(ASR_THREADS),
                              stdout=(ROOT / f'log_asr_{cases_name}_{label}_{k}.log').open('a'),
                              stderr=subprocess.STDOUT) for k in range(ASR_PROCS)]
    _asr_jobs.append({'cases': cases_name, 'label': label, 'procs': procs})
    log(f'asr started {cases_name}/{label} ({ASR_PROCS} GPU procs)')


def finalize(cases_name, label):
    """Wait for that job's ASR, write the report and the done marker."""
    done = ROOT / f'done_{cases_name}_{label}.json'
    if done.exists() and 'v2' in json.loads(done.read_text()):
        return json.loads(done.read_text())['report']
    for j in _asr_jobs:
        if j['cases'] == cases_name and j['label'] == label:
            codes = [p.wait() for p in j['procs']]
            if any(codes):
                raise RuntimeError(f'asr {cases_name}/{label} exit {codes}')
    items = list(_jsonl_items(f'asr/{cases_name}_{label}_part*.jsonl').values())
    if len(items) != _expected(cases_name):
        raise RuntimeError(f'asr {cases_name}/{label}: {len(items)}/{_expected(cases_name)}')
    from voxft import eval as evaluation
    (ROOT / 'reports').mkdir(exist_ok=True)
    out = ROOT / f'reports/{cases_name}_{label}.json'
    items.sort(key=lambda i: (i['case_id'], i['seed']))
    out.write_text(json.dumps({'target': MODELS[label], 'label': label, 'cases': cases_name,
                               'asr': 'faster-whisper large-v3 float16 GPU (separate ASR procs); short-set tl forced language',
                               **evaluation._report_metrics(items), 'items': items},
                              ensure_ascii=False, indent=1), encoding='utf-8')
    done.write_text(json.dumps({'report': str(out), 'sha256': sha(out), 'v2': True}), encoding='utf-8')
    log(f'report {cases_name}/{label} -> {out}')
    return str(out)


def gen(label, cases_name, shard):
    from voxft import eval as evaluation, infer
    k, n = map(int, shard.split('/'))
    out = ROOT / f'gen/{cases_name}_{label}_{k}of{n}.jsonl'
    have = set(_jsonl_items(f'gen/{cases_name}_{label}_*.jsonl'))
    cases = [c for i, c in enumerate(read_jsonl(ROOT / f'{cases_name}_cases.jsonl')) if i % n == k]
    model = infer.get_model(None, None if label == 'base' else MODELS[label])
    ref_cache = {}
    with out.open('a', encoding='utf-8') as f:
        for case in cases:
            for seed in SEEDS:
                if (case['case_id'], seed) in have:
                    continue
                kw = infer._gen_kwargs(case['text'], case.get('ref_audio'), None, float(case['cfg_value']),
                                       int(case['inference_timesteps']), seed, case.get('control') or '')
                kw['retry_badcase'] = False
                wav, gen_sec = infer._run(model, kw)
                row = {**case, 'seed': seed, 'wav': wav, 'gen_sec': gen_sec,
                       'effective_cfg_value': float(case['cfg_value']),
                       'effective_inference_timesteps': int(case['inference_timesteps']),
                       **evaluation._acoustics(wav, case['text'], case.get('ref_audio'), ref_cache)}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                print(f"{case['case_id']} s{seed} gen={gen_sec}s", flush=True)


def asr(label, cases_name, shard):
    from difflib import SequenceMatcher
    from voxft import eval as ev          # loads .env (HF_HOME) before huggingface_hub is imported
    from faster_whisper import WhisperModel
    k, n = map(int, shard.split('/'))
    rows = sorted(_jsonl_items(f'gen/{cases_name}_{label}_*.jsonl').values(), key=lambda r: (r['case_id'], r['seed']))
    rows = [r for i, r in enumerate(rows) if i % n == k]
    out = ROOT / f'asr/{cases_name}_{label}_part{k}of{n}.jsonl'
    out.parent.mkdir(exist_ok=True)
    have = set(_jsonl_items(f'asr/{cases_name}_{label}_part*.jsonl'))
    model = WhisperModel('large-v3', device='cuda', compute_type='float16', cpu_threads=ASR_THREADS)
    with out.open('a', encoding='utf-8') as f:
        for r in rows:
            if (r['case_id'], r['seed']) in have:
                continue
            lang = r['lang']
            # short 1–4-word cues: auto-detect picks wrong languages on ~1s clips -> force tl (amendment 1)
            language = lang if (cases_name == 'short' or lang not in ev.AUTO_DETECT_LANGS) else None
            segs, _ = model.transcribe(r['wav'], language=language, vad_filter=True)
            hyp = ' '.join(s.text.strip() for s in segs)
            h, ref = ev._norm(hyp), ev._norm(r['text'])
            r = {**r, 'hyp': hyp, 'asr_language': language,
                 'similarity': round(SequenceMatcher(None, h, ref).ratio(), 4),
                 'cer': round(ev._error_rate(h, ref), 4),
                 'wer': round(ev._error_rate(ev._norm(hyp, True).split(), ev._norm(r['text'], True).split()), 4)
                 if lang in ev.WER_LANGS else None,
                 'numeric': ev._is_numeric(r), 'suspected_truncation': ev._is_truncated(hyp, r['text']),
                 'over_read': ev._over_read(hyp, r['text']), 'len_ratio': round(len(h) / len(ref), 3) if ref else None}
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()


def split_dev():
    rows = read_jsonl(ROOT / 'dev_cases.jsonl')
    sids = sorted({r['sentence_id'] for r in rows})
    half_a = set(sids[::2])
    if not (ROOT / 'dev_a_cases.jsonl').exists():
        write_jsonl(ROOT / 'dev_a_cases.jsonl', [r for r in rows if r['sentence_id'] in half_a])
        write_jsonl(ROOT / 'dev_b_cases.jsonl', [r for r in rows if r['sentence_id'] not in half_a])


def combine_dev(label):
    done = ROOT / f'done_dev_{label}.json'
    items = [i for h in ('dev_a', 'dev_b')
             for i in json.loads(Path(json.loads((ROOT / f'done_{h}_{label}.json').read_text())['report'])
                                 .read_text(encoding='utf-8'))['items']]
    out = ROOT / f'reports/dev_{label}.json'
    out.write_text(json.dumps({'label': label, 'items': items}, ensure_ascii=False), encoding='utf-8')
    done.write_text(json.dumps({'report': str(out), 'sha256': sha(out), 'v2': True}), encoding='utf-8')


def amend():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    plan.setdefault('amendments', []).append({
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(Path(__file__)),
        'seen_before_amending': 'only base short-cue results (no candidate output inspected)',
        'changes': [
            'ASR moved to CPU faster-whisper large-v3 int8 for ALL models (incl. adopted v1 audio) to free GPU memory '
            'for 7 generation workers; user asked to saturate the GPU and speed up evaluation',
            'short-cue set: Whisper language forced for tl (auto-detect produced Korean/Urdu/Indonesian text on ~1s '
            'clips in base); dev/reserved keep the historical tl auto-detect',
            f'dev set split into two halves by sentence_id; interim harm look after half A: a candidate stops if any '
            f'language has one-sided sign-test p_worse<{INTERIM_HARM_P} (group-sequential, O\'Brien-Fleming-like '
            'boundary); final rules unchanged on the full set',
            'v1 GPU outputs for base and r46u10 short are reused for audio only']})
    plan['amendments'].append({
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script_sha256': sha(Path(__file__)),
        'changes': ['container is capped at 20 CPU cores / 90GiB RAM (cgroup); CPU ASR overloaded it and gen '
                    'workers were SIGKILLed before producing output. ASR now runs in 2 dedicated GPU procs '
                    '(large-v3 float16, same as historical evals) for ALL models; 6 gen workers, 2 CPU threads each'],
        'seen_before_amending': 'still only base short-cue results'})
    (ROOT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print('amended')


# ---------------- analysis ----------------
def sign_p_one_sided(k, n):
    """P(X >= k) for X~Bin(n, .5)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def load_items(path):
    return {(i['case_id'], i['seed']): i for i in json.loads(Path(path).read_text(encoding='utf-8'))['items']}


def severe_flags(reports, kind):
    by_case = defaultdict(list)
    for items in reports.values():
        for (cid, _), it in items.items():
            by_case[cid].append(it.get('audio_sec') or 0)
    med = {cid: statistics.median(v) for cid, v in by_case.items()}
    flags = {}
    for label, items in reports.items():
        out = {}
        for key, it in items.items():
            hyp = (it.get('hyp') or '').strip()
            cer = it.get('cer') or 0
            a = it.get('audio_sec') or 0
            if kind == 'short':
                runaway = a > max(2.0, 2.5 * med[key[0]])
                sev = not hyp or it.get('over_read') or it.get('suspected_truncation') or cer > 0.5 or runaway
            else:
                runaway = a > 1.8 * med[key[0]]
                sev = not hyp or it.get('over_read') or it.get('suspected_truncation') or cer > 0.3 or runaway
            out[key] = {'severe': bool(sev), 'runaway': bool(runaway)}
        flags[label] = out
    return flags


def compare(reports, flags, cand, kind, ref='base'):
    res = {}
    cer_margin = 0.03 if kind == 'short' else 0.01
    for lang in LANGS:
        keys = [k for k, it in reports[ref].items() if it['lang'] == lang and k in reports[cand]]
        fb, fc = flags[ref], flags[cand]
        new = sum(fc[k]['severe'] and not fb[k]['severe'] for k in keys)
        fixed = sum(fb[k]['severe'] and not fc[k]['severe'] for k in keys)
        cb = statistics.mean(reports[ref][k]['cer'] or 0 for k in keys)
        cc = statistics.mean(reports[cand][k]['cer'] or 0 for k in keys)
        p_worse, p_better = sign_p_one_sided(new, new + fixed), sign_p_one_sided(fixed, new + fixed)
        res[lang] = {'n': len(keys), 'base_severe': sum(fb[k]['severe'] for k in keys),
                     'cand_severe': sum(fc[k]['severe'] for k in keys), 'new': new, 'fixed': fixed,
                     'base_runaway': sum(fb[k]['runaway'] for k in keys),
                     'cand_runaway': sum(fc[k]['runaway'] for k in keys),
                     'p_worse': round(p_worse, 4), 'p_better': round(p_better, 4),
                     'cer_base': round(cb, 4), 'cer_cand': round(cc, 4),
                     'worse': p_worse < 0.10 or cc - cb > cer_margin, 'better': p_better < 0.05}
    return res


def analyze_phase(cases_name, labels):
    reports = {l: load_items(json.loads((ROOT / f'done_{cases_name}_{l}.json').read_text())['report']) for l in labels}
    kind = 'short' if cases_name == 'short' else 'dev'
    flags = severe_flags(reports, kind)
    out = {c: compare(reports, flags, c, kind) for c in labels if c != 'base'}
    for c in labels:
        if c not in ('base', 'r46u10') and 'r46u10' in labels:
            out[f'{c}_vs_r46u10'] = compare(reports, flags, c, kind, ref='r46u10')
    out['_totals'] = {l: {'severe': sum(f['severe'] for f in flags[l].values()),
                          'mean_cer': round(statistics.mean(i['cer'] or 0 for i in reports[l].values()), 4)}
                      for l in labels}
    return out


def verdict(per_set):
    worse = any(v['worse'] for s in per_set for v in s.values())
    better = any(v['better'] for s in per_set for v in s.values())
    return {'pass': (not worse) and better, 'any_worse': worse, 'any_better': better}


def run():
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8'))
    for label, digest in plan['weights_sha256'].items():
        if sha(Path(MODELS[label]) / 'lora_weights.safetensors') != digest:
            raise ValueError(f'weights changed: {label}')
    for p, digest in plan['inputs_sha256'].items():
        if Path(p) != Path(__file__) and sha(p) != digest:
            raise ValueError(f'frozen input changed: {p}')
    split_dev()
    # Phase 1 short cues; GPU never waits on ASR (CPU) except at decision points.
    for label in PHASE1:
        gen_stage('short', label)
        asr_stage_async('short', label)
    # base/r46u10 dev half A do not depend on the winner: keep the GPU busy meanwhile.
    for label in ('base', 'r46u10'):
        gen_stage('dev_a', label)
        asr_stage_async('dev_a', label)
    for label in PHASE1:
        finalize('short', label)
    p1 = analyze_phase('short', PHASE1)
    totals = p1['_totals']
    winner = min(WINNER_POOL, key=lambda l: (totals[l]['severe'], totals[l]['mean_cer']))
    p1['_winner'] = winner
    (ROOT / 'phase1_analysis.json').write_text(json.dumps(p1, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'phase1 winner {winner}: {json.dumps(totals)}')
    gen_stage('dev_a', winner)
    asr_stage_async('dev_a', winner)
    gen_stage('dev_b', 'base')          # needed whenever any candidate survives the interim look
    asr_stage_async('dev_b', 'base')
    labels = ['base', 'r46u10', winner]
    for label in labels:
        finalize('dev_a', label)
    interim = analyze_phase('dev_a', labels)
    stopped = {c: [l for l, v in interim[c].items() if v['p_worse'] < INTERIM_HARM_P] for c in ('r46u10', winner)}
    stopped = {c: v for c, v in stopped.items() if v}
    (ROOT / 'interim_analysis.json').write_text(json.dumps({'interim': interim, 'stopped': stopped},
                                                           ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'interim stopped: {stopped}')
    active = [c for c in ('r46u10', winner) if c not in stopped]
    for label in active:
        gen_stage('dev_b', label)
        asr_stage_async('dev_b', label)
    for label in ['base'] + active:
        finalize('dev_b', label)
        combine_dev(label)
    verdicts = {c: {'pass': False, 'stopped_at_interim': stopped[c]} for c in stopped}
    if active:
        p2 = analyze_phase('dev', ['base'] + active)
        for c in active:
            verdicts[c] = verdict([p1[c], p2[c]])
        if winner in active and 'r46u10' in active:
            vs = verdict([p1[f'{winner}_vs_r46u10'], p2[f'{winner}_vs_r46u10']])
            verdicts[f'{winner}_vs_r46u10'] = vs
            verdicts[winner]['replace_production'] = verdicts[winner]['pass'] and not vs['any_worse']
        p2['_verdicts'] = verdicts
        (ROOT / 'phase2_analysis.json').write_text(json.dumps(p2, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'phase2 verdicts {json.dumps(verdicts)}')
    if os.environ.get('DEFER_PHASE3') == '1':
        # best practice: choose the single finalist (winner vs r49) on dev+short first, then touch reserved once
        (ROOT / 'phase3_deferred.json').write_text(json.dumps({'winner': winner, 'verdicts': verdicts},
                                                              ensure_ascii=False, indent=2), encoding='utf-8')
        log('phase3 deferred: reserved set will be used once for the final pick among winner/r49')
        return
    chosen = next((c for c in (winner, 'r46u10') if verdicts.get(c, {}).get('pass')), None)
    if chosen is None:
        log('no candidate passed phases 1+2; reserved set NOT used')
        (ROOT / 'final.json').write_text(json.dumps({'chosen': None, 'verdicts': verdicts}, indent=2))
        return
    p3_labels = ['base', 'r46u10'] + ([chosen] if chosen != 'r46u10' else [])
    for label in p3_labels:
        gen_stage('reserved', label)
        asr_stage_async('reserved', label)
    for label in p3_labels:
        finalize('reserved', label)
    p3 = analyze_phase('reserved', p3_labels)
    worse3 = any(v['worse'] for v in p3[chosen].values())
    v3 = {'non_regression': not worse3, 'any_better': any(v['better'] for v in p3[chosen].values())}
    final = {'chosen': chosen, 'phase3': p3, 'phase3_verdict': v3, 'verdicts': verdicts,
             'note': 'reserved_next_20261102 now used once; never use it for further tuning'}
    (ROOT / 'final.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'phase3 {chosen}: {json.dumps(v3)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'run', 'gen', 'asr', 'amend'])
    ap.add_argument('rest', nargs='*')
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / 'src'))
    if args.cmd in ('gen', 'asr'):
        {'gen': gen, 'asr': asr}[args.cmd](*args.rest)
    else:
        {'prepare': prepare, 'run': run, 'amend': amend}[args.cmd]()
