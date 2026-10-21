#!/bin/bash
# r19 验收恢复链（v2 shard2 挂起后的接管脚本）：
#   0) 等主链 FATAL 退出（v2 shard0/1 已完成，分片报告已在 EVAL_DIR）
#   1) 重跑 v2 shard 2/3，与已完成的 shard0/1 合并（354 条）
#   2) id 探针 3 分片 + 合并（54 条）
#   3) base 对照 drama_chain_prod 3 分片 + 合并（336 条）
#   4) r13x 对照 drama_chain_prod 3 分片 + 合并（336 条）
# 用法：nohup bash scripts/r19_recovery_chain.sh > /root/autodl-tmp/logs/r19_recovery.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
LOGD=/root/autodl-tmp/logs
MAP=$LOGD/r19_report_map.jsonl
R19=/root/autodl-tmp/voxft_ckpt/lora_omni5_r19_emotion_3ep/latest
R13X=/root/autodl-tmp/voxft_ckpt/lora_omni5_r13x_idemph_3ep/latest

echo "=== $(date -Is) 等主链 FATAL/退出 ==="
while pgrep -f "r19_eval_pipeline[.]sh" > /dev/null; do sleep 30; done
grep -q "FATAL" $LOGD/r19_pipeline.log && echo "主链已 FATAL，接管" || echo "主链已退出（未见 FATAL，仍按恢复流程走）"

merge_check() {  # $1=tag $2=expected $3..=shard paths
  local tag=$1 expected=$2; shift 2
  uv run python -X utf8 -u -m voxft.eval --merge "$@" \
    > "$LOGD/recovery_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/recovery_merge_${tag}.log" | tail -1)
  local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  if [ "$count" != "$expected" ]; then
    echo "FATAL: $tag 合并 $count 条 ≠ $expected"; exit 1
  fi
  echo "{\"tag\": \"$tag\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $tag 合并完成 ($count 条) → $merged"
}

# 1) v2：补 shard2，与已完成的 shard0/1 合并（这两份已逐条核验是 v2：120/117 条）
V2_S0=$EVAL_DIR/lora_omni5_r19_emotion_3ep_latest_shard0of3_279d54741d7441d9b4793e0dfa783ac1.json
V2_S1=$EVAL_DIR/lora_omni5_r19_emotion_3ep_latest_shard1of3_9741624778b5418a93376456663d3fbb.json
echo "=== $(date -Is) v2 补跑 shard 2/3 ==="
before=$(mktemp); after=$(mktemp)
ls "$EVAL_DIR" | sort > "$before"
uv run python -X utf8 -u -m voxft.eval "$R19" \
  --lang th --texts-file eval_cases/omnivoice_prod_v2.jsonl --seeds 42 43 44 \
  --cfg-value 1.8 --inference-timesteps 20 --shard 2/3 \
  > "$LOGD/r19_v2_shard2_recovery.log" 2>&1 || { echo "FATAL: v2 shard2 重跑非零退出"; exit 1; }
ls "$EVAL_DIR" | sort > "$after"
SHARD2=$(comm -13 "$before" "$after" | grep "^lora_omni5_r19_emotion_3ep_latest_shard2of3_.*\.json$" | tail -1)
rm -f "$before" "$after"
[ -z "$SHARD2" ] && { echo "FATAL: 找不到 v2 shard2 新报告"; exit 1; }
merge_check "v2" 354 "$V2_S0" "$V2_S1" "$EVAL_DIR/$SHARD2"

run3() {  # $1=tag $2=target $3=prefix $4=cases $5=expected
  local tag=$1 target=$2 prefix=$3 cases=$4 expected=$5
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag 启动 3 分片 ==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$target" \
      --lang th --texts-file "$cases" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$LOGD/recovery_${tag}_shard${k}.log" 2>&1 &
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
  [ "$(echo "$shards" | grep -c .)" -ne 3 ] && { echo "FATAL: $tag 新分片数不对"; echo "$new"; exit 1; }
  merge_check "$tag" "$expected" $(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
}

run3 "idprobe" "$R19" "lora_omni5_r19_emotion_3ep_latest" "eval_cases/id_prosody_probe_20261020.jsonl" 54
run3 "dramaprod_base" "base" "base" "eval_cases/drama_chain_prod_20261021.jsonl" 336
run3 "dramaprod_r13x" "$R13X" "lora_omni5_r13x_idemph_3ep_latest" "eval_cases/drama_chain_prod_20261021.jsonl" 336
echo "=== $(date -Is) 恢复链全部完成 RECOVERY_DONE ==="
