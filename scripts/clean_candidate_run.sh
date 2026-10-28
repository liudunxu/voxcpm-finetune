#!/bin/bash
set -euo pipefail
root=/root/autodl-tmp/clean_candidate_20261028
venv=/root/voxcpm-finetune/.venv
export PATH="$venv/bin:/root/.local/bin:$PATH"
export PYTHONPATH="$root/source/src:$root/source/scripts:$root/source/third_party/VoxCPM/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 PYTHONUTF8=1
export HF_HOME=/root/autodl-tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt
export XDG_CACHE_HOME=/root/autodl-tmp/cache TORCH_HOME=/root/autodl-tmp/torch_home
export TMPDIR="$root/runtime/tmp" TORCHINDUCTOR_CACHE_DIR="$root/runtime/inductor"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$venv/lib/python3.11/site-packages/nvidia/cublas/lib:$venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$root/source"
exec "$venv/bin/python" -X utf8 -u scripts/clean_candidate_run.py "$@" --work "$root/run"
