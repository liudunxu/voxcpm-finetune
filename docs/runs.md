# 微调运行记录

每轮一行摘要 + 一段明细，**按时间倒序追加**。字段由 `python -m voxft.train.runlog`
从 configs/、mix.json、train.log、eval 报告自动抽取，不要手改数字。

用途：下一轮微调的起点参照、周报汇总、以及「这个结论是哪一轮、用什么数据得出的」回溯。

## lora_omni_joint5

- 生成时间：2026-09-14 13:08:40
- 结论：**不通过：vi 在生产短 cue 上 CER 0.0686→0.5278（3 seed × 18 样本），疑似漏尾 0→0.2778；step_500 同样退化(0.4299)，非单纯过训**
- 下一步：补 cue 长度数据：gigaspeech2 dev 分片（p50 4.2-6.2s 自然口语，Apache-2.0）；FLEURS p50 10-14s 与线上 1-3s cue 严重错配

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 1274 |
| warmup / weight_decay / max_grad_norm | 127 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni/train.jsonl |
| 样本数 / 语种条数 | 20370 / {"en": 1126, "id": 3079, "ms": 3013, "th": 4774, "tl": 2620, "vi": 3582, "zh": 2176} |
| val loss 首→末 | 1.0074 (step 0) → 0.8165 (step 1273)，最优 0.8165 |

### 数据配比（按时长）

- 总时长 **53.3127 h** / 20370 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：2

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 2.6658 |
| id | 0.17 | 0.17 | 9.0616 |
| ms | 0.17 | 0.17 | 9.062 |
| th | 0.17 | 0.17 | 9.0634 |
| tl | 0.17 | 0.17 | 9.0617 |
| vi | 0.17 | 0.1701 | 9.0661 |
| zh | 0.1 | 0.1 | 5.3321 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni_joint5_latest CER | lora_omni_joint5_step_0000500 CER |
|---|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.126 (WER 0.2278) | 0.1653 (WER 0.2944) |
| ms | 0.0852 (WER 0.1762) | 0.0856 (WER 0.1429) | 0.119 (WER 0.1706) |
| th | 0.0282 | 0.0287 | 0.0465 |
| tl | 0.1166 (WER 0.0944) | 0.098 (WER 0.0833) | 0.0998 (WER 0.0903) |
| vi | 0.0686 (WER 0.1) | 0.5278 (WER 0.5944) | 0.4299 (WER 0.4611) |
| zh | 0.0 | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.176 | 0.1709 |
| 疑似漏尾 | 0.0952 | 0.1548 | 0.1786 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni_joint5_latest` 红线：id 0.1036→0.1260；vi 0.0686→0.5278
  - 噪声级（未触发红线，照实记录）：ms 0.0852→0.0856；th 0.0282→0.0287
- `lora_omni_joint5_step_0000500` 红线：id 0.1036→0.1653；ms 0.0852→0.1190；th 0.0282→0.0465；vi 0.0686→0.4299

报告文件：`base_c04dd9d80a164c91a0731f9d29fbf510.json`、`lora_omni_joint5_latest_f4c83666f59149d79dc52b0db84bbb39.json`、`lora_omni_joint5_step_0000500_3eed4673dfb34abcbd92004a5c96f6c6.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

非数字类 case 15/15 完全无变化；退化全部集中在数字/货币短句。FLEURS 长度(5.7-21.7s)的 vi 探针上 LoRA 反而更好：CER 0.0218→0.0175、漏尾 0.0833→0.0，证明问题是短 cue 出分布而非越南语能力下降。

---
