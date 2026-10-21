#!/bin/bash
# r19_emotion 训练→验收自动链：
#   1) 等训练进程退出并核验 training_state 步数 == 1202
#   2) drama_chain_prod（生产形态指令，112 case × 3 seed，3 分片 + 合并）
#   3) v2 全量 118 case × 3 seed（跨轮可比口径）
#   4) id 韵律探针 18 case × 3 seed
# 口径与历轮一致：cfg 1.8 / 20 步 / 禁重试 / seeds 42 43 44。
# 用法：nohup bash scripts/r19_eval_pipeline.sh > /root/autodl-tmp/logs/r19_pipeline.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
RUN=lora_omni5_r19_emotion_3ep
CKPT=/root/autodl-tmp/voxft_ckpt/$RUN/latest
LOGD=/root/autodl-tmp/logs
MAP=$LOGD/r19_report_map.jsonl

echo "=== $(date -Is) 等待训练结束 ==="
while pgrep -f "train_voxcpm""_finetune" > /dev/null; do sleep 60; done
STEP=$(python3 -c "import json;print(json.load(open('$CKPT/training_state.json'))['step'])" 2>/dev/null || echo 0)
echo "=== $(date -Is) 训练进程已退出，latest step=$STEP ==="
if [ "$STEP" -ne 1202 ]; then echo "FATAL: 步数 $STEP ≠ 1202，训练未正常完成"; exit 1; fi

run_shards() {  # $1=tag $2=cases_file $3=expected_items
  local tag=$1 cases=$2 expected=$3
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag 启动 3 分片 ==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$CKPT" \
      --lang th --texts-file "$cases" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$LOGD/r19_${tag}_shard${k}.log" 2>&1 &
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
  local prefix=${RUN}_latest
  local shards=$(echo "$new" | grep "^${prefix}_shard[0-9]of3_.*\.json$" | sort)
  local n=$(echo "$shards" | grep -c . || true)
  if [ "$n" -ne 3 ]; then
    echo "FATAL: $tag 新分片 $n ≠ 3"; echo "$new"; exit 1
  fi
  local shard_paths=$(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
  uv run python -X utf8 -u -m voxft.eval --merge $shard_paths \
    > "$LOGD/r19_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/r19_merge_${tag}.log" | tail -1)
  local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  if [ "$count" != "$expected" ]; then
    echo "FATAL: $tag 合并 $count 条 ≠ $expected"; exit 1
  fi
  echo "{\"tag\": \"$tag\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $tag 合并完成 ($count 条) → $merged"
}

run_shards "dramaprod" "eval_cases/drama_chain_prod_20261021.jsonl" 336
run_shards "v2" "eval_cases/omnivoice_prod_v2.jsonl" 354
run_shards "idprobe" "eval_cases/id_prosody_probe_20261020.jsonl" 54
echo "=== $(date -Is) r19 验收全部完成 PIPELINE_DONE ==="
