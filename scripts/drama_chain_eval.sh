#!/bin/bash
# drama_chain 生产链路对齐验收（2026-10-21）：zh ref + 情绪前缀 + 短 cue + 短 ref 2×2
# 集合 eval_cases/drama_chain_20261021.jsonl（112 行 = 28 文本 × ref长短 × 情绪有无）
# base 与 r18 同口径各跑 3 seed × 3 分片，cfg 1.8 / 20 步 / 禁重试（eval 内置）。
# 用法：nohup bash scripts/drama_chain_eval.sh > /root/autodl-tmp/logs/drama_chain_pipeline.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
LOGD=/root/autodl-tmp/logs
CASES=eval_cases/drama_chain_20261021.jsonl
EXPECTED=336   # 112 rows × 3 seeds
MAP=$LOGD/drama_chain_report_map.jsonl

run_shards() {  # $1=target(base|ckpt路径) $2=报告前缀 $3=tag
  local target=$1 prefix=$2 tag=$3
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag ($target) 启动 3 分片 ==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$target" \
      --lang th --texts-file "$CASES" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$LOGD/drama_${tag}_shard${k}.log" 2>&1 &
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
    > "$LOGD/drama_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/drama_merge_${tag}.log" | tail -1)
  local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  if [ "$count" != "$EXPECTED" ]; then
    echo "FATAL: $tag 合并 $count 条 ≠ $EXPECTED"; exit 1
  fi
  echo "{\"tag\": \"$tag\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $tag 合并完成 ($count 条) → $merged"
}

run_shards "base" "base" "base"
run_shards "/root/autodl-tmp/voxft_ckpt/lora_omni5_r18_idaccent_3ep/latest" \
  "lora_omni5_r18_idaccent_3ep_latest" "r18"
echo "=== $(date -Is) drama_chain 验收全部完成 PIPELINE_DONE ==="
