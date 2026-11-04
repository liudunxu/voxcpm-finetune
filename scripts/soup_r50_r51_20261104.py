"""Model soup of r50 + r51 LoRAs (Wortsman et al. 2022), no training.

Both share base, LoRA init seed, recipe and 3-epoch budget; they differ only in TH data volume and are
complementary in the 2026-11-04 review (r50 better on long TH, r51 better on short TH/TL cues).
Exact uniform average of the deltas as one rank-128 LoRA: A = [A50; A51], B = [B50/2, B51/2], alpha = r = 128
(scale 1, same as the parents' 64/64), so B @ A == (B50 A50 + B51 A51) / 2 for every adapted layer.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

R50 = Path('/root/autodl-tmp/r50_3ep_20261103/checkpoint/latest')
R51 = Path('/root/autodl-tmp/r51_th_restore_20261104/checkpoint/latest')
OUT = Path('/root/autodl-tmp/soup_r50_r51_20261104')


def main():
    OUT.mkdir(exist_ok=False)
    a, b = load_file(str(R50 / 'lora_weights.safetensors')), load_file(str(R51 / 'lora_weights.safetensors'))
    assert a.keys() == b.keys()
    ca, cb = json.loads((R50 / 'lora_config.json').read_text()), json.loads((R51 / 'lora_config.json').read_text())
    assert ca['lora_config'] == cb['lora_config'] and ca['lora_config']['r'] == ca['lora_config']['alpha'] == 64
    out, max_err = {}, 0.0
    for k in a:
        if k.endswith('.lora_A'):
            out[k] = torch.cat([a[k], b[k]], 0).contiguous()
        elif k.endswith('.lora_B'):
            out[k] = torch.cat([a[k] * 0.5, b[k] * 0.5], 1).contiguous()
        else:
            raise ValueError(f'unexpected key {k}')
    for k in a:
        if k.endswith('.lora_A'):
            kb = k[:-7] + '.lora_B'
            want = (a[kb].float() @ a[k].float() + b[kb].float() @ b[k].float()) / 2
            got = out[kb].float() @ out[k].float()
            max_err = max(max_err, (want - got).abs().max().item())
    save_file(out, str(OUT / 'lora_weights.safetensors'))
    cfg = json.loads(json.dumps(ca))
    cfg['lora_config']['r'] = cfg['lora_config']['alpha'] = 128
    cfg['soup_of'] = {'r50': str(R50), 'r51': str(R51), 'weights': [0.5, 0.5]}
    (OUT / 'lora_config.json').write_text(json.dumps(cfg, indent=2))
    print('SOUP', len(out), 'tensors; max |delta error|', max_err)
    assert max_err < 1e-4


if __name__ == '__main__':
    main()
