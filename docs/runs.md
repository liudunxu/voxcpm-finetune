# 微调运行记录

每轮一行摘要 + 一段明细，**按时间倒序追加**。字段由 `python -m voxft.train.runlog`
从 configs/、mix.json、train.log、eval 报告自动抽取，不要手改数字。

用途：下一轮微调的起点参照、周报汇总、以及「这个结论是哪一轮、用什么数据得出的」回溯。

## lora_omni5_r3

- 生成时间：2026-09-14 16:25:48
- 结论：**不通过，且整体劣于 round 2：vi 0.0686→0.2839、ms 0.0852→0.1053、总体 0.0774→0.1259；只有 th 0.0141、tl 0.0998 与 r2 持平且优于基座**
- 下一步：停止调配比。已证明离线指标分辨率不足：vi_digit_3 在 base/r1/r2/r3 上是 0.000/1.4375/0.000/0.958（非单调、整条翻转），ms 在配比完全相同的 r2/r3 之间差 0.042。交付 round 2，数字类残余退化写进模型卡并交盲听裁定

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 1984 |
| warmup / weight_decay / max_grad_norm | 198 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni3/train.jsonl |
| 样本数 / 语种条数 | 31739 / {"en": 1632, "id": 5445, "ms": 4404, "th": 7013, "tl": 3826, "vi": 6216, "zh": 3203} |
| val loss 首→末 | 1.0212 (step 0) → 0.8823 (step 1983)，最优 0.8823 |

### 数据配比（按时长）

- 总时长 **77.6356 h** / 31739 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 3.8813 |
| id | 0.17 | 0.17 | 13.1992 |
| ms | 0.17 | 0.17 | 13.1966 |
| th | 0.17 | 0.17 | 13.198 |
| tl | 0.17 | 0.17 | 13.1986 |
| vi | 0.17 | 0.17 | 13.199 |
| zh | 0.1 | 0.1 | 7.7629 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni5_r3_latest CER |
|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.128 (WER 0.2278) |
| ms | 0.0852 (WER 0.1762) | 0.1053 (WER 0.1746) |
| th | 0.0282 | 0.0141 |
| tl | 0.1166 (WER 0.0944) | 0.0998 (WER 0.0903) |
| vi | 0.0686 (WER 0.1) | 0.2839 (WER 0.3556) |
| zh | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.1259 |
| 疑似漏尾 | 0.0952 | 0.1429 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r3_latest` 红线：id 0.1036→0.1280；ms 0.0852→0.1053；vi 0.0686→0.2839

报告文件：`base_c04dd9d80a164c91a0731f9d29fbf510.json`、`lora_omni5_r3_latest_7f53befe6a7d4649b14393820a8128ed.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

本轮假设（提高 FLEURS 份额以恢复带数字训练时长）被实测否掉：vi 带数字时长从 r2 的 1.44h 恢复到 2.30h（接近 r1 的 2.41h），但 vi CER 反而从 0.1176 恶化到 0.2839。所以 r1→r2 的改善不是数字覆盖带来的，把 vi 从 100% FLEURS 降到 41% 才是关键（短 cue 长度分布）。r3 唯一确定收益是 val loss 更低（0.8823 vs r2 的 0.9176），再次说明 val loss 与验收指标不同向，不能用来选 checkpoint

---

## lora_omni5_r2

- 生成时间：2026-09-14 15:01:15
- 结论：**不通过（id 0.1036→0.1280、vi 0.0686→0.1176 过红线），但相比 round 1 大幅好转：th 0.0282→0.0141、ms 0.0852→0.0630、tl 0.1166→0.0998 三语种优于基座，vi 疑似漏尾 0.2778→0.0，总体 0.0774→0.0827 基本持平**
- 下一步：已定位并启动 round 3：gigaspeech2/yodas 的带数字样本实测为 0.0%，FLEURS 是 20.6-24.1%，round 2 把 vi/id 的 FLEURS 份额压到 7/17 导致带数字训练时长各降约 40%（vi 2.41h→1.44h、id 2.17h→1.30h）。round 3 只改 vi/id 为 fleurs 11 + gs2 6，th/tl/ms/zh/en 配比不动

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 2068 |
| warmup / weight_decay / max_grad_norm | 206 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni2/train.jsonl |
| 样本数 / 语种条数 | 33082 / {"en": 1632, "id": 6121, "ms": 4404, "th": 7013, "tl": 3826, "vi": 6883, "zh": 3203} |
| val loss 首→末 | 1.0775 (step 0) → 0.9767 (step 2067)，最优 0.9176 |

### 数据配比（按时长）

- 总时长 **77.6308 h** / 33082 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 3.8813 |
| id | 0.17 | 0.17 | 13.1965 |
| ms | 0.17 | 0.17 | 13.1966 |
| th | 0.17 | 0.17 | 13.198 |
| tl | 0.17 | 0.17 | 13.1986 |
| vi | 0.17 | 0.17 | 13.197 |
| zh | 0.1 | 0.1 | 7.7629 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni5_r2_latest CER | lora_omni5_r2_step_0000750 CER |
|---|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.128 (WER 0.2389) | 0.1633 (WER 0.2833) |
| ms | 0.0852 (WER 0.1762) | 0.063 (WER 0.1151) | 0.1158 (WER 0.1873) |
| th | 0.0282 | 0.0141 | 0.0435 |
| tl | 0.1166 (WER 0.0944) | 0.0998 (WER 0.0903) | 0.1277 (WER 0.1194) |
| vi | 0.0686 (WER 0.1) | 0.1176 (WER 0.1389) | 0.1405 (WER 0.1944) |
| zh | 0.0 | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.0827 | 0.1135 |
| 疑似漏尾 | 0.0952 | 0.0952 | 0.1429 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r2_latest` 红线：id 0.1036→0.1280；vi 0.0686→0.1176
- `lora_omni5_r2_step_0000750` 红线：id 0.1036→0.1633；ms 0.0852→0.1158；th 0.0282→0.0435；tl 0.1166→0.1277；vi 0.0686→0.1405

报告文件：`base_c04dd9d80a164c91a0731f9d29fbf510.json`、`lora_omni5_r2_latest_aab2ce1c1ace4d768b8d3ae31b451265.json`、`lora_omni5_r2_step_0000750_b11ca19f51604fdea7ce66e22243b767.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

残余退化全部集中在 3 条数字 case（vi_digit +0.098、vi_digit_2 +0.216、id_digit_2 +0.190），非数字 case 无退化。同时修好一批：ms_digit_3（RM45,000 基座念成印尼盾 Rp，r2 全对）、id_digit_3（GA 876 + pukul 06.15）、tl_digit、ms_digit_2。⚠️ id_digit_2 的 r2 输出被 Whisper 转写成 Harganya Rp250.000.（基座是 250 ribu rupiah），高度疑似 ASR 把口播金额归一成符号形式而非 TTS 念错——CER 分不开这两者，需盲听裁定。step_750（val loss 最优 0.9176）在所有语种上都比 latest 差，再次说明 val loss 不是可用的 checkpoint 选择依据

---

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
