"""Release r50 to FrankLiuDundun/voxcpm-finetune-lora (user request 2026-11-04: replace that repo with r50).

steps: merge (assert every LoRA layer merged; non-weight files symlinked to save disk) -> lora/ + card +
training config + evaluation -> smoke synth -> upload (huggingface.co via network_turbo, xet on) -> verify remote
LFS sha256 equals local -> keep manifest; caller deletes the merged dir afterwards.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

BASE = Path('/root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915')
LORA = Path('/root/autodl-tmp/r50_3ep_20261103/checkpoint/latest')
OUT = Path('/root/autodl-tmp/release_r50_20261104')
REPO_ID = 'FrankLiuDundun/voxcpm-finetune-lora'
LORA_SHA = 'f08bcb27b294488a9a4082f1ef1db10b7e72f9a44e60d2d3f39d3f9bf7276d41'
sys.path.insert(0, '/root/voxcpm-finetune/src')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def merge():
    from safetensors.torch import load_file, save_file
    from voxft.lora.merge import read_lora_config
    assert sha(LORA / 'lora_weights.safetensors') == LORA_SHA
    OUT.mkdir(exist_ok=False)
    cfg = read_lora_config(LORA)
    scale = float(cfg.get('alpha', cfg['r'])) / float(cfg['r'])
    lw = load_file(str(LORA / 'lora_weights.safetensors'))
    pairs = {k[:-7] + '.weight': (k, k[:-7] + '.lora_B') for k in lw if k.endswith('.lora_A')}
    merged = 0
    for f in sorted(BASE.glob('*.safetensors')):
        t = load_file(str(f))
        for bk, (a, b) in pairs.items():
            if bk in t:
                t[bk] = t[bk].float().add_(lw[b].float() @ lw[a].float() * scale).to(t[bk].dtype)
                merged += 1
        save_file(t, str(OUT / f.name), metadata={'format': 'pt'})
    assert merged == len(pairs), f'merged {merged}/{len(pairs)} LoRA layers'
    for f in BASE.iterdir():
        if f.suffix != '.safetensors' and f.is_file() and f.name != 'README.md':
            os.symlink(f.resolve(), OUT / f.name)
    (OUT / 'lora').mkdir()
    for n in ('lora_weights.safetensors', 'lora_config.json'):
        shutil.copy2(LORA / n, OUT / 'lora' / n)
    shutil.copy2('/root/autodl-tmp/r50_3ep_20261103/train.yaml', OUT / 'training_config.yaml')
    shutil.copy2('/root/autodl-tmp/release_r50_README.md', OUT / 'README.md')
    fin = json.loads(Path('/root/autodl-tmp/prod_review_20261103/final.json').read_text(encoding='utf-8'))
    r50 = json.loads(Path('/root/autodl-tmp/prod_review_20261103/r50_analysis.json').read_text(encoding='utf-8'))
    ev = {'model': 'r50 (r30 pool + MMS-FA TL/ID 3-6s segments, 3 epochs, 1480 steps)',
          'lora_sha256': LORA_SHA, 'protocol': 'cfg1.8; steps by ref duration 28/24/22; 5 seeds; no retry; '
          'Whisper large-v3; per-language sign test on new-vs-fixed severe outputs',
          'short_and_dev_vs_base_r46u10_r29': {k: r50[k] for k in ('vs_base', 'vs_r46u10', 'totals')},
          'final_selection': {k: fin[k] for k in ('eligible', 'chosen', 'summary')},
          'reserved_once': {'verdict': fin['phase3_verdict'], 'totals': fin['phase3']['_totals'],
                            'r50_vs_base': fin['phase3']['r50'], 'r50_vs_r46u10': fin['phase3']['r50_vs_r46u10']},
          'native_quality': 'unverified (naturalness/accent/emotion not reviewed by native speakers)'}
    (OUT / 'evaluation.json').write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MERGED', merged, 'layers; scale', scale, flush=True)


def smoke():
    import soundfile as sf
    from voxft import infer
    model = infer._import_voxcpm().from_pretrained(str(OUT), load_denoiser=False)
    for text in ('Sige, tara na!', 'Terima kasih banyak.', 'ขอบคุณมากนะ'):
        wav = model.generate(text=text, cfg_value=1.8, inference_timesteps=22, seed=42,
                             reference_wav_path='/root/autodl-tmp/voxft_data/processed/fleurs_zh/audio/run_1owk1jss/0001308.wav')
        dur = len(wav) / model.tts_model.sample_rate
        assert 0.3 < dur < 6, (text, dur)
        print('SMOKE', text, round(dur, 2), 's', flush=True)


def upload():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    info = api.upload_folder(folder_path=str(OUT), repo_id=REPO_ID, repo_type='model',
                             commit_message='r50: th/tl/id joint LoRA (r30 clean pool + MMS-FA TL/ID 3-6s segments, '
                                            '14.5h x 3ep = 1480 steps); production-condition review: severe '
                                            'short 80->56 / dev 33->11 / reserved 11->5 vs base; vs r46-u10 no worse language')
    print('UPLOADED', info, flush=True)


def verify():
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    info = api.model_info(REPO_ID, files_metadata=True)
    remote = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    checks = {}
    for name in ('model.safetensors', 'lora/lora_weights.safetensors', 'audiovae.pth'):
        local = sha(OUT / name)
        checks[name] = {'local': local, 'remote': remote.get(name), 'ok': local == remote.get(name)}
    manifest = {'repo': REPO_ID, 'revision': info.sha, 'checks': checks,
                'files': sorted(remote), 'production_changed_by_assistant': False}
    Path('/root/autodl-tmp/release_r50_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('VERIFY', json.dumps(manifest), flush=True)
    assert all(c['ok'] for c in checks.values())


if __name__ == '__main__':
    {'merge': merge, 'smoke': smoke, 'upload': upload, 'verify': verify}[sys.argv[1]]()
