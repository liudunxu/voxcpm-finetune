#!/bin/bash
# refrob 阶段 2/3 验收驱动：
#   阶段A：v2 全量 118 case × 3 seed × {base, r8, r12a, r12b}（干净红线回归）
#   阶段B：refrob 6 条件 × 3 seed × {r12a, r12b}（降质套件，base/r8 用阶段 1 报告配对）
# 口径同阶段 1：cfg 1.8 / 20 步 / 禁重试 / seeds 42 43 44 / 3 分片并行 / 快照差集收报告。
# 任一分片失败或合并条数不符即中止。
# 用法：nohup bash scripts/refrob_stage2_eval.sh > /root/autodl-tmp/refrob_20261020/logs/eval2_driver.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

WORK=/root/autodl-tmp/refrob_20261020
EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
CKPT=/root/autodl-tmp/voxft_ckpt
R8=$CKPT/lora_omni5_r8/latest
RA=$CKPT/lora_omni5_r12a_refrob_clean/latest
RB=$CKPT/lora_omni5_r12b_refrob_deg50/latest
MAP=$WORK/report_map_stage2.jsonl

run_shards() {  # $1=tag $2=cases_file $3=expected_per_target $4...=targets
  local tag=$1 cases=$2 expected=$3; shift 3
  local before after
  before=$(mktemp); after=$(mktemp)
  ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
  echo "=== $(date -Is) $tag 启动 3 分片（targets: $*）==="
  local pids=()
  for k in 0 1 2; do
    stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval "$@" \
      --lang th --texts-file "$cases" --seeds 42 43 44 \
      --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
      > "$WORK/logs/eval2_${tag}_shard${k}.log" 2>&1 &
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
  for t in "$@"; do
    local prefix
    if [ "$t" = "base" ]; then prefix=base; else prefix=$(basename "$(dirname "$t")")_latest; fi
    local shards=$(echo "$new" | grep "^${prefix}_shard[0-9]of3_.*\.json$" | sort)
    local n=$(echo "$shards" | grep -c . || true)
    if [ "$n" -ne 3 ]; then
      echo "FATAL: $tag target=$prefix 新分片 $n ≠ 3"; echo "$new"; exit 1
    fi
    local shard_paths=$(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
    uv run python -X utf8 -u -m voxft.eval --merge $shard_paths \
      > "$WORK/logs/merge2_${tag}_${prefix}.log" 2>&1 || {
        echo "FATAL: $tag target=$prefix 合并失败"; exit 1; }
    local merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
      "$WORK/logs/merge2_${tag}_${prefix}.log" | tail -1)
    local count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
    if [ "$count" != "$expected" ]; then
      echo "FATAL: $tag target=$prefix 合并 $count 条 ≠ $expected"; exit 1
    fi
    echo "{\"tag\": \"$tag\", \"target\": \"$prefix\", \"merged\": \"$merged\", \"shards\": \"$shard_paths\"}" >> "$MAP"
    echo "--- $(date -Is) $tag target=$prefix 合并完成 ($count 条)"
  done
}

# 阶段 A：干净红线回归（v2 全量）
run_shards "v2clean" "eval_cases/omnivoice_prod_v2.jsonl" 354 base "$R8" "$RA" "$RB"

# 阶段 B：降质套件（只跑 r12a/r12b；base/r8 复用阶段 1 报告）
for cond in clean babble_sir15 babble_sir8 lowpass4k lowpass3k combo_babble8_lp3k; do
  run_shards "refrob_${cond}" "eval_cases/refrob_20261019_${cond}.jsonl" 54 "$RA" "$RB"
done
echo "=== $(date -Is) 验收全部完成 ==="
