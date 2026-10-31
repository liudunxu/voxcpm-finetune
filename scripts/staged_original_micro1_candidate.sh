#!/bin/bash
set -euo pipefail
run_root=/root/autodl-tmp/r46_micro1_20261031
run_venv=/root/voxcpm-finetune/.venv
export PATH="$run_venv/bin:/root/.local/bin:$PATH"
export PYTHONPATH="$run_root/source/src:$run_root/source/scripts:$run_root/source/third_party/VoxCPM/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 PYTHONUNBUFFERED=1 PYTHONUTF8=1
export HF_HOME=/root/autodl-tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt
export XDG_CACHE_HOME=/root/autodl-tmp/cache TORCH_HOME=/root/autodl-tmp/torch_home
export TMPDIR="$run_root/runtime/tmp" TORCHINDUCTOR_CACHE_DIR="$run_root/runtime/inductor"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$run_venv/lib/python3.11/site-packages/nvidia/cublas/lib:$run_venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$run_root/source"
exec "$run_venv/bin/python" -X utf8 -u scripts/staged_candidate_run.py "$@" --variant dit_original_lr5e5_rank32_drop0_wd01_micro1 --work "$run_root/run"
