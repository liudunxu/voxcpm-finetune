#!/bin/bash
# refrob 阶段 2/3：串行训练 A（干净 ref）→ B（50% ref 降质），各带 wandb 桥接伴随进程。
# 任一失败即中止；完成后校验 latest 权重与配置存在。
# 用法：nohup bash scripts/refrob_stage2_train.sh > /root/autodl-tmp/refrob_20261020/logs/train_driver.log 2>&1 &
set -uo pipefail
export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:$PATH
WORK=/root/autodl-tmp/refrob_20261020
PY=/root/voxcpm-finetune/.venv/bin/python
TRAIN=/root/voxcpm-finetune/third_party/VoxCPM/scripts/train_voxcpm_finetune.py

train_one() {
  local name=$1
  local cfg=/root/voxcpm-finetune/configs/${name}.yaml
  local ckpt=/root/autodl-tmp/voxft_ckpt/${name}
  echo "=== $(date -Is) 训练启动: $name ==="
  nohup stdbuf -oL -eL "$PY" -X utf8 -u -m voxft.train.tb_wandb_bridge \
    "${ckpt}/logs" "${name}" > "$WORK/logs/wandb_${name}.log" 2>&1 &
  local bridge=$!
  cd /root/voxcpm-finetune/third_party/VoxCPM
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True stdbuf -oL -eL "$PY" -X utf8 -u \
    "$TRAIN" --config_path "$cfg" > "$WORK/logs/train_${name}.log" 2>&1
  local rc=$?
  cd /root/voxcpm-finetune
  kill "$bridge" 2>/dev/null
  if [ "$rc" -ne 0 ]; then echo "FATAL: $name 训练退出码 $rc"; exit "$rc"; fi
  for f in lora_weights.safetensors lora_config.json; do
    [ -f "${ckpt}/latest/$f" ] || { echo "FATAL: $name 缺 latest/$f"; exit 1; }
  done
  echo "=== $(date -Is) 训练完成: $name ==="
}

train_one lora_omni5_r12a_refrob_clean
train_one lora_omni5_r12b_refrob_deg50
echo "=== $(date -Is) 两组训练全部完成 ==="
