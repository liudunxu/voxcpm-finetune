#!/bin/bash
set -euo pipefail
run_root=/root/autodl-tmp/music_excluded_20261029
export PYTHONPATH="$run_root/source/src:$run_root/source/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 PYTHONUTF8=1
export HF_HOME=/root/autodl-tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt
export XDG_CACHE_HOME=/root/autodl-tmp/cache TORCH_HOME=/root/autodl-tmp/torch_home
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$run_root/source"
exec /root/voxcpm-finetune/.venv/bin/python -X utf8 -u scripts/quarantine_music_candidates.py "$@" --work "$run_root/run"
