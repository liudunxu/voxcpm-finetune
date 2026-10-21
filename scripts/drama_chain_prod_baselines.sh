#!/bin/bash
# drama_chain_prod 同口径对照补跑：等 r19 主验收链结束后，
# 依次跑 base 与 r13x（各 3 分片 + 合并），口径与主链完全一致
# （cfg 1.8 / 20 步 / 禁重试 / seeds 42 43 44）。
# 用法：nohup bash scripts/drama_chain_prod_baselines.sh > /root/autodl-tmp/logs/dramaprod_baselines.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
LOGD=/root/autodl-tmp/logs
MAP=$LOGD/r19_report_map.jsonl
CASES=eval_cases/drama_chain_prod_20261021.jsonl
MAIN_LOG=$LOGD/r19_pipeline.log

echo "=== $(date -Is) 等待 r19 主链结束 ==="
while true; do
  if grep -q "PIPELINE_DONE" "$MAIN_LOG" 2>/dev/null; then break; fi
  if grep -q "FATAL" "$MAIN_LOG" 2>/dev/null; then echo "FATAL: 主链失败，不再补跑对照"; exit 1; fi
  sleep 60
done
echo "=== $(date -Is) 主链完成，开始对照补跑 ==="

run_shards() {  # $1=tag $2=target $3=label_prefix
  local tag=$1 target=$2 prefix=$3
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag 启动 3 分片 ==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$target" \
      --lang th --texts-file "$CASES" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$LOGD/dramaprod_${tag}_shard${k}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do
    wait "$p" || { echo "FATAL: $tag 分片进程 $p 非零退出"; rc=1; }
  done
  [ "$rc" -ne 0 ] && exit 1
  ls "$EVAL_DIR" | sort > "$after"
  local new=$(comm -13 "$before" "$after")
  rm -f "$before" "$after"
  local shards=$(echo "$new" | grep "^${prefix}_shard[0-9]of3_.*\.json$" | sort)
  local n=$(echo "$shards" | grep -c . || true)
  if [ "$n" -ne 3 ]; then
    echo "FATAL: $tag 新分片 $n ≠ 3"; echo "$new"; exit 1
  fi
  local shard_paths=$(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
  uv run python -X utf8 -u -m voxft.eval --merge $shard_paths \
    > "$LOGD/dramaprod_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/dramaprod_merge_${tag}.log" | tail -1)
  local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  if [ "$count" != "336" ]; then
    echo "FATAL: $tag 合并 $count 条 ≠ 336"; exit 1
  fi
  echo "{\"tag\": \"dramaprod_$tag\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $tag 合并完成 ($count 条) → $merged"
}

run_shards "base" "base" "base"
run_shards "r13x" "/root/autodl-tmp/voxft_ckpt/lora_omni5_r13x_idemph_3ep/latest" "lora_omni5_r13x_idemph_3ep_latest"
echo "=== $(date -Is) 对照补跑完成 BASELINES_DONE ==="
