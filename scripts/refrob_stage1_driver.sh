#!/bin/bash
# refrob 阶段 1 驱动：6 条件 × (base, r8) × 3 分片 = 648 次生成。
# 每个条件内部 3 分片并行、wait 收口、按 target 合并；条件间串行（控显存）。
# 分片归属用「启动前后 EVAL_DIR 快照差集」判定（同目录有历史 shard 报告，
# 不能按文件名模式捞）；任一分片失败或合并条数不对即中止，不出残缺报告。
# 用法：nohup bash scripts/refrob_stage1_driver.sh > /root/autodl-tmp/refrob_20261020/logs/driver.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

WORK=/root/autodl-tmp/refrob_20261020
EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
R8=/root/autodl-tmp/voxft_ckpt/lora_omni5_r8/latest
CONDS="clean babble_sir15 babble_sir8 lowpass4k lowpass3k combo_babble8_lp3k"
MAP=$WORK/report_map.jsonl
EXPECTED_PER_TARGET=54   # 18 case × 3 seed

for cond in $CONDS; do
  CASES="eval_cases/refrob_20261019_${cond}.jsonl"
  [ -f "$CASES" ] || { echo "FATAL: 缺 case 文件 $CASES"; exit 1; }
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) condition=$cond 启动 3 分片 ==="
  pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval base "$R8" \
      --lang th --texts-file "$CASES" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$WORK/logs/eval_${cond}_shard${k}.log" 2>&1 &
    pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do
    wait "$p" || { echo "FATAL: condition=$cond 分片进程 $p 非零退出"; rc=1; }
  done
  [ "$rc" -ne 0 ] && exit 1
  ls "$EVAL_DIR" | sort > "$after"
  new=$(comm -13 "$before" "$after")
  rm -f "$before" "$after"

  for prefix in base lora_omni5_r8_latest; do
    shards=$(echo "$new" | grep "^${prefix}_shard[0-9]of3_.*\.json$" | sort)
    n=$(echo "$shards" | grep -c . || true)
    if [ "$n" -ne 3 ]; then
      echo "FATAL: condition=$cond target=$prefix 新分片报告 $n 份 ≠ 3，中止"
      echo "$new"
      exit 1
    fi
    shard_paths=$(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
    uv run python -X utf8 -u -m voxft.eval --merge $shard_paths \
      > "$WORK/logs/merge_${cond}_${prefix}.log" 2>&1 || {
        echo "FATAL: condition=$cond target=$prefix 合并失败"; exit 1; }
    merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
      "$WORK/logs/merge_${cond}_${prefix}.log" | tail -1)
    COUNT=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
    if [ "$COUNT" != "$EXPECTED_PER_TARGET" ]; then
      echo "FATAL: condition=$cond target=$prefix 合并报告 $COUNT 条 ≠ $EXPECTED_PER_TARGET"
      exit 1
    fi
    echo "{\"condition\": \"$cond\", \"target\": \"$prefix\", \"merged\": \"$merged\", \"shards\": \"$shard_paths\"}" >> "$MAP"
    echo "--- $(date -Is) condition=$cond target=$prefix 合并完成: $merged ($COUNT 条)"
  done
done
echo "=== $(date -Is) 全部 6 条件完成 ==="
