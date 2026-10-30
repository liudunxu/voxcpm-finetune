#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/quality_reaudit_20261030
source_dir="$root/separation_review_all_source"
work="$root/separation_review_all"
python=/root/voxcpm-finetune/.venv/bin/python
overlay="$root/qwen_setup/overlay"
export PYTHONPATH="$source_dir/scripts:$source_dir/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME=/root/autodl-tmp/hf_cache
export XDG_CACHE_HOME=/root/autodl-tmp/xdg_cache
export TMPDIR=/root/autodl-tmp/tmp
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/torchinductor_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"

"$python" -X utf8 -u "$source_dir/scripts/quality_separation_review.py" prepare-expanded --work "$work"
run_shard() {
    directory="$work/shard$1"
    "$python" -X utf8 -u "$source_dir/scripts/quality_separation_review.py" music --work "$directory" > "$directory/music.log" 2>&1
    PYTHONPATH="$overlay:$source_dir/scripts:$source_dir/src" \
        "$python" -X utf8 -u "$source_dir/scripts/quality_separation_review.py" asr --work "$directory" > "$directory/asr.log" 2>&1
    "$python" -X utf8 -u "$source_dir/scripts/quality_separation_review.py" collect --work "$directory" > "$directory/collect.log" 2>&1
}
pids=()
for shard in 0 1 2 3; do
    run_shard "$shard" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo 'REVIEW_INCOMPLETE: preserve partial outputs; no automatic rerun' >&2
    exit 1
fi
"$python" -X utf8 -u "$source_dir/scripts/quality_separation_review.py" collect-expanded --work "$work"
