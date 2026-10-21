#!/bin/bash
# r19 恢复链 v2（接管 r19_recovery_chain.sh，区别：id 探针与 v2 shard2 并行跑）
#   0) id 探针单进程全量 54 条（立即启动，与 v2 shard2 并行；报告无 shard 后缀，不撞检测）
#   1) 等 v2 shard2（原恢复链启动的孤儿进程）退出 → 合并 v2（354 条）
#   2) base 对照 drama_chain_prod 3 分片 + 合并（336 条）
#   3) r13x 对照 drama_chain_prod 3 分片 + 合并（336 条）
#   4) 等 id 探针收尾（正常早已完成）→ 核验 54 条并记 MAP
# 用法：nohup bash scripts/r19_recovery2.sh > /root/autodl-tmp/logs/r19_recovery2.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
LOGD=/root/autodl-tmp/logs
MAP=$LOGD/r19_report_map.jsonl
R19=/root/autodl-tmp/voxft_ckpt/lora_omni5_r19_emotion_3ep/latest
R13X=/root/autodl-tmp/voxft_ckpt/lora_omni5_r13x_idemph_3ep/latest
V2_S0=$EVAL_DIR/lora_omni5_r19_emotion_3ep_latest_shard0of3_279d54741d7441d9b4793e0dfa783ac1.json
V2_S1=$EVAL_DIR/lora_omni5_r19_emotion_3ep_latest_shard1of3_9741624778b5418a93376456663d3fbb.json

# --- 0) id 探针并行启动（单进程不分片） ---
ls "$EVAL_DIR" | sort > /tmp/r19_idp_before.txt
stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$R19" \
  --lang id --texts-file eval_cases/id_prosody_probe_20261020.jsonl --seeds 42 43 44 \
  --cfg-value 1.8 --inference-timesteps 20 \
  > "$LOGD/recovery2_idprobe.log" 2>&1 &
IDP_PID=$!
echo "=== $(date -Is) id 探针已并行启动 pid=$IDP_PID ==="

merge_check() {  # $1=tag $2=expected $3..=shard paths
  local tag=$1 expected=$2; shift 2
  uv run python -X utf8 -u -m voxft.eval --merge "$@" \
    > "$LOGD/recovery2_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/recovery2_merge_${tag}.log" | tail -1)
  local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  if [ "$count" != "$expected" ]; then
    echo "FATAL: $tag 合并 $count 条 ≠ $expected"; exit 1
  fi
  echo "{\"tag\": \"$tag\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $tag 合并完成 ($count 条) → $merged"
}

# --- 1) 等 v2 shard2 孤儿进程退出，合并 v2 ---
echo "=== $(date -Is) 等 v2 shard2 退出 ==="
while pgrep -f "omnivoice_prod_v2[.]jsonl.*shard 2/3" > /dev/null; do sleep 30; done
SHARD2=$(ls "$EVAL_DIR" | grep "^lora_omni5_r19_emotion_3ep_latest_shard2of3_.*\.json$" \
  | grep -v f4557d711aaf4cb9a0abd6c30caf42e6 | tail -1)
[ -z "$SHARD2" ] && { echo "FATAL: 找不到 v2 shard2 新报告"; exit 1; }
CNT=$(python3 -c "import json;print(len(json.load(open('$EVAL_DIR/$SHARD2'))['items']))")
[ "$CNT" -ne 117 ] && { echo "FATAL: v2 shard2 报告 $CNT 条 ≠ 117（拿错文件？）"; exit 1; }
merge_check "v2" 354 "$V2_S0" "$V2_S1" "$EVAL_DIR/$SHARD2"

run3() {  # $1=tag $2=target $3=prefix $4=expected
  local tag=$1 target=$2 prefix=$3 expected=$4
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag 启动 3 分片 ==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$target" \
      --lang th --texts-file eval_cases/drama_chain_prod_20261021.jsonl --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$LOGD/recovery2_${tag}_shard${k}.log" 2>&1 &
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

run3 "dramaprod_base" "base" "base" 336
run3 "dramaprod_r13x" "$R13X" "lora_omni5_r13x_idemph_3ep_latest" 336

# --- 4) id 探针收尾核验 ---
echo "=== $(date -Is) 等 id 探针收尾 ==="
wait "$IDP_PID" || { echo "FATAL: id 探针非零退出"; exit 1; }
IDP_R=$(comm -13 /tmp/r19_idp_before.txt <(ls "$EVAL_DIR" | sort) \
  | grep "^lora_omni5_r19_emotion_3ep_latest_[0-9a-f]\{32\}\.json$" | tail -1)
[ -z "$IDP_R" ] && { echo "FATAL: 找不到 id 探针报告"; exit 1; }
CNT=$(python3 -c "import json;print(len(json.load(open('$EVAL_DIR/$IDP_R'))['items']))")
[ "$CNT" -ne 54 ] && { echo "FATAL: id 探针 $CNT 条 ≠ 54"; exit 1; }
echo "{\"tag\": \"idprobe\", \"merged\": \"$EVAL_DIR/$IDP_R\"}" >> "$MAP"
echo "--- $(date -Is) idprobe 完成 ($CNT 条) → $EVAL_DIR/$IDP_R"
echo "=== $(date -Is) 恢复链 v2 全部完成 RECOVERY_DONE ==="
