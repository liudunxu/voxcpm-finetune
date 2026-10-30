#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/quality_reaudit_20261030
source_dir="$root/paired_candidate_20261031_v2_source"
export PYTHONPATH="$source_dir/scripts:$source_dir/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/root/autodl-tmp/hf_cache
export VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data
export TMPDIR=/root/autodl-tmp/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec /root/voxcpm-finetune/.venv/bin/python -X utf8 -u \
    "$source_dir/scripts/paired_quality_candidate.py" --work "$root/paired_candidate_20261031_v2"
