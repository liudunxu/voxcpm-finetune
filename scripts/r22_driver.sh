#!/bin/bash
# r22（lora_omni5_r22_idboost_cont）：从 r21 checkpoint 续训 628 步（+1.5ep，LR 接续余弦尾部）
# 数据：r21_nothaiser + gigaspeech2_id 800 行（音色地道性优先，跑飞容忍——用户 2026-10-21 口径）
#   0) 等 GPU 空闲
#   1) 拷 r21 latest → r22 目录当续训起点；启动 wandb 桥接 + 训练
#   2) 核验日志含 "Loaded LoRA weights"（真续训而非重训）且最终 step==2159
#   3) dramaprod → v2 → id 探针（口径与历轮一致）
# 用法：nohup bash scripts/r22_driver.sh > /root/autodl-tmp/logs/r22_driver.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

PY=/root/voxcpm-finetune/.venv/bin/python
TRAIN=/root/voxcpm-finetune/third_party/VoxCPM/scripts/train_voxcpm_finetune.py
EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
LOGD=/root/autodl-tmp/logs
MAP=$LOGD/r22_report_map.jsonl
RUN=lora_omni5_r22_idboost_cont
CKPT=/root/autodl-tmp/voxft_ckpt/$RUN
R21CKPT=/root/autodl-tmp/voxft_ckpt/lora_omni5_r21_nothaiser_cont

echo "=== $(date -Is) r22 启动（等 GPU 空闲）==="
while pgrep -f "train_voxcpm""_finetune" > /dev/null || pgrep -f "voxft[.]eval" > /dev/null; do sleep 30; done

mkdir -p "$CKPT"
cp -r "$R21CKPT/latest" "$CKPT/latest" || { echo "FATAL: 拷贝 r21 latest 失败"; exit 1; }
echo "=== $(date -Is) r21 latest 已拷为 r22 起点 ==="

nohup stdbuf -oL -eL "$PY" -X utf8 -u -m voxft.train.tb_wandb_bridge \
  "$CKPT/logs" "$RUN" > "$LOGD/wandb_bridge_r22.log" 2>&1 &
BRIDGE=$!
cd /root/voxcpm-finetune/third_party/VoxCPM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stdbuf -oL -eL "$PY" -X utf8 -u \
  "$TRAIN" --config_path /root/voxcpm-finetune/configs/$RUN.yaml \
  > "$LOGD/train_r22.log" 2>&1
RC=$?
cd /root/voxcpm-finetune
kill "$BRIDGE" 2>/dev/null
[ "$RC" -ne 0 ] && { echo "FATAL: 训练退出码 $RC"; exit "$RC"; }
grep -q "Loaded LoRA weights" "$LOGD/train_r22.log" || { echo "FATAL: 未见 LoRA 续训加载日志（变成从头训了）"; exit 1; }
STEP=$(python3 -c "import json;print(json.load(open('$CKPT/latest/training_state.json'))['step'])" 2>/dev/null || echo 0)
echo "=== $(date -Is) 训练退出，latest step=$STEP ==="
[ "$STEP" -ne 2159 ] && { echo "FATAL: 步数 $STEP ≠ 2159"; exit 1; }

merge_check() {  # $1=tag $2=expected $3..=shard paths
  local tag=$1 expected=$2; shift 2
  uv run python -X utf8 -u -m voxft.eval --merge "$@" \
    > "$LOGD/r22_merge_${tag}.log" 2>&1 || { echo "FATAL: $tag 合并失败"; exit 1; }
  local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$LOGD/r22_merge_${tag}.log" | tail -1)
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
      > "$LOGD/r22_${tag}_shard${k}.log" 2>&1 &
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

ls "$EVAL_DIR" | sort > /tmp/r22_idp_before.txt
uv run python -X utf8 -u -m voxft.eval "$CKPT/latest" \
  --lang id --texts-file eval_cases/id_prosody_probe_20261020.jsonl --seeds 42 43 44 \
  --cfg-value 1.8 --inference-timesteps 20 \
  > "$LOGD/r22_idprobe.log" 2>&1 || { echo "FATAL: id 探针非零退出"; exit 1; }
IDP_R=$(comm -13 /tmp/r22_idp_before.txt <(ls "$EVAL_DIR" | sort) \
  | grep "^${RUN}_latest_[0-9a-f]\{32\}\.json$" | tail -1)
CNT=$(python3 -c "import json;print(len(json.load(open('$EVAL_DIR/$IDP_R'))['items']))" 2>/dev/null || echo 0)
[ "$CNT" -ne 54 ] && { echo "FATAL: id 探针 $CNT 条 ≠ 54"; exit 1; }
echo "{\"tag\": \"idprobe\", \"merged\": \"$EVAL_DIR/$IDP_R\"}" >> "$MAP"
echo "--- $(date -Is) idprobe 完成 ($CNT 条) → $EVAL_DIR/$IDP_R"
echo "=== $(date -Is) r22 全部完成 PIPELINE_DONE ==="
