#!/bin/bash
# 短 cue 门（阶段 1 式量化）：base/r8/r12a × short_cue_probe_20261020（117 条）× 3 seed。
# 回答「r12a 在 1-8s 级短文本上是否还有可测退化/改善」，再决定要不要开短 cue 训练轮。
# 用法：nohup bash scripts/short_cue_gate_eval.sh > /root/autodl-tmp/refrob_20261020/logs/short_gate.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
cd /root/voxcpm-finetune

WORK=/root/autodl-tmp/refrob_20261020
EVAL_DIR=/root/autodl-tmp/voxft_ckpt/eval
CKPT=/root/autodl-tmp/voxft_ckpt
CASES=eval_cases/short_cue_probe_20261020.jsonl
EXPECTED=351   # 117 × 3 seed
MAP=$WORK/report_map_short_gate.jsonl

before=$(mktemp); after=$(mktemp)
ls "$EVAL_DIR" 2>/dev/null | sort > "$before"
echo "=== $(date -Is) 短 cue 门启动 3 分片 ==="
pids=()
for k in 0 1 2; do
  stdbuf -oL -eL uv run python -X utf8 -u -m voxft.eval \
    base "$CKPT/lora_omni5_r8/latest" "$CKPT/lora_omni5_r12a_refrob_clean/latest" \
    --lang th --texts-file "$CASES" --seeds 42 43 44 \
    --cfg-value 1.8 --inference-timesteps 20 --shard $k/3 \
    > "$WORK/logs/short_gate_shard${k}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "FATAL: 分片 $p 非零退出"; rc=1; }
done
[ "$rc" -ne 0 ] && exit 1
ls "$EVAL_DIR" | sort > "$after"
new=$(comm -13 "$before" "$after"); rm -f "$before" "$after"

for prefix in base lora_omni5_r8_latest lora_omni5_r12a_refrob_clean_latest; do
  shards=$(echo "$new" | grep "^${prefix}_shard[0-9]of3_.*\.json$" | sort)
  n=$(echo "$shards" | grep -c . || true)
  [ "$n" -ne 3 ] && { echo "FATAL: $prefix 新分片 $n ≠ 3"; echo "$new"; exit 1; }
  shard_paths=$(echo "$shards" | sed "s|^|$EVAL_DIR/|" | tr '\n' ' ')
  uv run python -X utf8 -u -m voxft.eval --merge $shard_paths \
    > "$WORK/logs/merge_short_${prefix}.log" 2>&1 || { echo "FATAL: $prefix 合并失败"; exit 1; }
  merged=$(grep -o '/root/autodl-tmp/voxft_ckpt/eval/[^ ]*\.json' \
    "$WORK/logs/merge_short_${prefix}.log" | tail -1)
  count=$(python3 -c "import json;print(len(json.load(open('$merged'))['items']))")
  [ "$count" != "$EXPECTED" ] && { echo "FATAL: $prefix 合并 $count ≠ $EXPECTED"; exit 1; }
  echo "{\"tag\": \"short_gate\", \"target\": \"$prefix\", \"merged\": \"$merged\"}" >> "$MAP"
  echo "--- $(date -Is) $prefix 合并完成 ($count 条)"
done
echo "=== $(date -Is) 短 cue 门全部完成 ==="
