#!/bin/bash
# r20（lora_omni5_r20_emofix_3ep）训练→wandb 桥接→验收自动链：
#   1) 杀掉 r19 残留的 wandb 桥接；启动 r20 桥接 + 训练（1162 步）
#   2) 核验 latest step == 1162
#   3) drama_chain_prod 3 分片 + 合并（336 条；base/r13x 对照报告已有，不重跑）
#   4) v2 3 分片 + 合并（354 条）
#   5) id 探针单进程（54 条）
# 口径与历轮一致：cfg 1.8 / 20 步 / 禁重试 / seeds 42 43 44。
# 用法：nohup bash scripts/r20_driver.sh > /root/autodl-tmp/logs/r20_driver.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

PY=/root/voxcpm-finetune/.venv/bin/python
TRAIN=/root/voxcpm-finetune/third_party/VoxCPM/scripts/train_voxcpm_finetune.py
EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
LOGD=/root/autodl-tmp/logs
MAP=$LOGD/r20_report_map.jsonl
RUN=lora_omni5_r20_emofix_3ep
CKPT=/root/autodl-tmp/voxft_ckpt/$RUN
STEPS=1162

# 1) 训练 + 桥接
pkill -f "tb_wandb_bridge.*r19_emotion" 2>/dev/null; sleep 1
nohup stdbuf -oL -eL "$PY" -X utf8 -u -m voxft.train.tb_wandb_bridge \
  "$CKPT/logs" "$RUN" > "$LOGD/wandb_bridge_r20.log" 2>&1 &
BRIDGE=$!
echo "=== $(date -Is) r20 训练启动（桥接 pid=$BRIDGE）==="
cd /root/voxcpm-finetune/third_party/VoxCPM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stdbuf -oL -eL "$PY" -X utf8 -u \
  "$TRAIN" --config_path /root/voxcpm-finetune/configs/$RUN.yaml \
  > "$LOGD/train_r20.log" 2>&1
RC=$?
cd /root/voxcpm-finetune
kill "$BRIDGE" 2>/dev/null
[ "$RC" -ne 0 ] && { echo "FATAL: 训练退出码 $RC"; exit "$RC"; }
STEP=$(python3 -c "import json;print(json.load(open('$CKPT/latest/training_state.json'))['step'])" 2>/dev/null || echo 0)
echo "=== $(date -Is) 训练退出，latest step=$STEP ==="
[ "$STEP" -ne "$STEPS" ] && { echo "FATAL: 步数 $STEP ≠ $STEPS"; exit 1; }

merge_check() {  # $1=tag $2=expected $3..=shard paths
  local tag=$1 expected=$2; shift 2
  uv run python -X utf8 -u -m voxft.eval --merge "$@" \
    > "$LOGD/r20_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/r20_merge_${tag}.log" | tail -1)
  local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  [ "$count" != "$expected" ] && { echo "FATAL: $tag 合并 $count 条 ≠ $expected"; exit 1; }
  echo "{\"tag\": \"$tag\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $tag 合并完成 ($count 条) → $merged"
}

run3() {  # $1=tag $2=cases $3=expected
  local tag=$1 cases=$2 expected=$3
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag 启动 3 分片 ==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$CKPT/latest" \
      --lang th --texts-file "$cases" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$LOGD/r20_${tag}_shard${k}.log" 2>&1 &
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
  local shards=$(echo "$new" | grep "^${RUN}_latest_shard[0-9]of3_.*\.json$" | sort)
  [ "$(echo "$shards" | grep -c .)" -ne 3 ] && { echo "FATAL: $tag 新分片数不对"; echo "$new"; exit 1; }
  merge_check "$tag" "$expected" $(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
}

run3 "dramaprod" "eval_cases/drama_chain_prod_20261021.jsonl" 336
run3 "v2" "eval_cases/omnivoice_prod_v2.jsonl" 354

# id 探针单进程
ls "$EVAL_DIR" | sort > /tmp/r20_idp_before.txt
uv run python -X utf8 -u -m voxft.eval "$CKPT/latest" \
  --lang id --texts-file eval_cases/id_prosody_probe_20261020.jsonl --seeds 42 43 44 \
  --cfg-value 1.8 --inference-timesteps 20 \
  > "$LOGD/r20_idprobe.log" 2>&1 || { echo "FATAL: id 探针非零退出"; exit 1; }
IDP_R=$(comm -13 /tmp/r20_idp_before.txt <(ls "$EVAL_DIR" | sort) \
  | grep "^${RUN}_latest_[0-9a-f]\{32\}\.json$" | tail -1)
CNT=$(python3 -c "import json;print(len(json.load(open('$EVAL_DIR/$IDP_R'))['items']))" 2>/dev/null || echo 0)
[ "$CNT" -ne 54 ] && { echo "FATAL: id 探针 $CNT 条 ≠ 54"; exit 1; }
echo "{\"tag\": \"idprobe\", \"merged\": \"$EVAL_DIR/$IDP_R\"}" >> "$MAP"
echo "--- $(date -Is) idprobe 完成 ($CNT 条) → $EVAL_DIR/$IDP_R"
echo "=== $(date -Is) r20 全部完成 PIPELINE_DONE ==="
