---
license: apache-2.0
base_model: openbmb/VoxCPM2
base_model_relation: finetune
tags: [voxcpm, voxcpm2, tts, lora, fine-tune, multilingual, dubbing, thai, indonesian, tagalog]
language: [th, id, tl, zh, en]
---

# VoxCPM 2 三语种配音微调（th / id / tl）

在 [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)（Apache-2.0）上做的 **LoRA 联合微调**，
目标是提升泰语 / 印尼语 / Tagalog 的**短剧配音**合成质量，并对齐主调方 OmniVoice 的真实调用口径
（单句 cue、跨语言 reference-only、cfg 1.8 / 20 步）。

一个 LoRA 同时覆盖三个语种（id 占比特意加重到 47.5%，针对印尼语口音校准），
另含少量中文与英文回放做防遗忘。**验收是分语种做的**，见下面「验收结果」。

当前版本为 **r18**（2026-10-21）：15 小时混合数据 × 3 epoch（1390 步），是 v2 验收集上
总体 CER 最优的一轮（base 0.0490 → **0.0393**），泰语显著优于基座（ΔCER −0.0321，
95% CI [−0.050, −0.017]），id/tl 同步改善，音色相似度无漂移；
id 韵律探针 CER 0.059 → **0.0013**（历轮最佳）。

> 注：2026-10 起目标语种收缩为 th/tl/id（vi/ms 退出），验收集为 v2 过滤后的 118 case，
> 与更早期版本的 182 case 口径**不可直接横比**，请以本卡内的 base 对照为准。

## 仓库内容

| 路径 | 内容 |
|---|---|
| 根目录 | LoRA 已 merge 进基座的**完整模型**，可直接被 `from_pretrained` 加载 |
| `lora/` | LoRA 增量本体：`lora_weights.safetensors` + `lora_config.json` |
| `training_config.yaml` | 训练配置，复现用 |

## 加载方式

**A. 用 merge 后的完整模型（零改动，推荐）**

```python
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("FrankLiuDundun/voxcpm-finetune-lora",
                               load_denoiser=False, optimize=True)
```

OmniVoice 侧只需把环境变量指过来：`VOXCPM_MODEL_ID=FrankLiuDundun/voxcpm-finetune-lora`。

**B. 用 LoRA 增量（省 4.3GB 下载）**

```python
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=True,
                               lora_weights_path="/path/to/lora")
loaded, skipped = model.load_lora("/path/to/lora")
assert loaded and not skipped, (len(loaded), skipped)   # 必须校验，失败即停
```

> ⚠️ **`lora/` 里的两个文件必须一起用**。`from_pretrained` 会自动读同目录的
> `lora_config.json` 来对齐 `r` / `alpha`；只拿 `lora_weights.safetensors` 会回落到默认
> `r=8`，形状不匹配、权重被**静默跳过**，加载看起来"成功"但等于没加载。
> 本 LoRA 是 `r=64, alpha=64, dropout=0.05, enable_lm=true, enable_dit=true, enable_proj=false`。

## 验收结果

口径：`eval_cases/omnivoice_prod_v2.jsonl`（th/tl/id + zh/en 回放，共 **118 case**）
× seed 42/43/44 = **354 条**，`cfg_value=1.8`、`inference_timesteps=20`、`retry_badcase=False`
（禁用坏例重试以保证 A/B 同条件），跨语言 reference-only，参考音频为线上在用的 4.47s 菲律宾语 ref。
ASR 用 `large-v3`。CER/WER 越低越好；th 词间无空格只有 CER。

| 语种 | base CER | 本模型 CER | 变化 | 疑似漏尾 base→本模型 |
|---|---|---|---|---|
| th | 0.1155 | **0.0834** | **↓28%（CI 显著）** | 0.0521 → 0.0625 |
| id | 0.0230 | **0.0157** | ↓32% | 0.000 → 0.0104 |
| tl | 0.0184 | **0.0128** | ↓30% | 0.000 → 0.000 |
| zh（回放） | 0.0515 | 0.0758 | ↑0.024 ⚠️ | 0.000 → 0.000 |
| en（回放） | 0.0153 | 0.0138 | ≈持平 | 0.000 → 0.000 |
| **总体** | **0.0490** | **0.0393** | **↓20%** | 0.0141 → 0.0198 |

说话人相似度（对参考音频，WavLM X-vector）：0.9121 → **0.9171**（无漂移，网内 ±0.02）。

**id 韵律探针**（18 条跨语言 ref id case × 3 seed）：CER **0.0013**，
为历轮最佳（base 0.059 / 上一轮 0.018）。

> ⚠️ zh 回放 CER +0.024，**未触发 0.05 退化红线**；zh 是防遗忘回放语种不是目标语种。
> 目标语种的最终语言质量（自然度、口音地道性）以母语者盲听为准，离线 CER 只做诊断；
> **印尼语口音的母语评审尚未完成**，本版本按离线指标先行发布。

## 已知问题（重要，请先读）

1. **数字与货币类 case 方差极大，别用单 seed 结论。** 数字/金额短句在 base 与各轮微调上
   都可能整条翻转（同一条 case 在不同 seed 下 CER 可从 0.000 跳到 1.4+）。
   本版验收用 3 个 seed 配对缓解，但数字类仍是最不稳定的部分。
2. **规避建议**：VoxCPM2 的文本归一化只对 zh/en 生效，`th/id/tl` 是**原样透传**，
   所以阿拉伯数字会直接进模型（`fil` 在主调方已有 verbalize 数字的 TN）。
   **把数字 verbalize 扩到 th/id 会比换模型更有效**。
3. **没有改善、也不要期待改善的部分**：
   - **情绪表现力 / 去念稿感**：本轮训练数据里没有已核实的真人表演或情感语料
     （见下面数据来源），所以**不声称情绪表现力提升**。
   - **塑料音 / 金属音 / 娃娃音**：这类音色伪影主要是 AudioVAE、CFG 与采样步数的产物，
     而本 LoRA 只作用于 LM 与 DiT 的 q/k/v/o（`enable_proj=false`，`stop_proj` / `stop_head` 全冻结），
     结构上修不到它们。线上对应的处理是降 CFG 到 1.2–1.6 并把步数提到 24–30 重试。
   - **参考音频克隆能力**：训练数据的 `ref_audio` 覆盖率约 20%（仅 Common Voice 配对行），
     实测说话人相似度 0.9121 → 0.9171（**未下降**）。
4. **印尼语口音说明**：本版 id 数据全部来自印尼本土来源（FLEURS id / Gigaspeech2 id /
   Common Voice id），占比加重到 47.5%，训练集中**不含任何马来语（ms）数据**；
   基座预训练自带的口音偏向无法靠微调保证根除，口音地道性以印尼语母语者盲听为准
   （本版本发布时该评审尚未完成）。

## 训练数据与署名

全部为可商用、无 SA/NC/ND 限制的来源。**含 CC-BY-SA 的数据一律未使用**（例如 THAI-SER）。

| 来源 | 语种 | 许可 | 署名 / 链接 |
|---|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | th, tl(`fil_ph`), id, zh(`cmn_hans_cn`), en(`en_us`) | **CC-BY-4.0** | Google FLEURS，Arora et al. 2022；文本取 `raw_transcription`（保留大小写与标点） |
| [Gigaspeech 2](https://huggingface.co/datasets/speechcolab/gigaspeech2) | th, id（`dev` 分片） | **Apache-2.0** | SpeechColab，Du et al. 2025 |
| [Common Voice 22（社区镜像 fsicoli/common_voice_22_0）](https://huggingface.co/datasets/fsicoli/common_voice_22_0) | th, id | **CC0** | Mozilla Common Voice 22.0 |
| [YODAS2-Sidon th](https://huggingface.co/datasets/Chalermdej/yodas2_sidon_th_tts) | th | **CC-BY-3.0** | 上游为 YouTube 语料的 YODAS2 + Sidon 降噪 |

CC-BY 要求署名，上表即为署名。音频经二次加工（16kHz 重采样、裁首尾、3–30s 过滤、
尾静音硬上限 0.15s）后用于训练，加工不改变上游许可义务。

配比（按**有效音频时长**，不是条数），id 占比特意加重：

| 语种 | 时长 | 占比 | 构成 |
|---|---|---|---|
| id | 7.13 h | 47.5% | FLEURS + Gigaspeech2 + Common Voice 22 |
| th | 3.36 h | 22.4% | FLEURS + Gigaspeech2 + Common Voice 22 + YODAS2 |
| tl | 3.36 h | 22.4% | FLEURS |
| zh | 0.77 h | 5.1% | FLEURS（防遗忘回放） |
| en | 0.38 h | 2.6% | FLEURS（防遗忘 + Taglish 英文词） |
| **合计** | **15.01 h** | 100% | 7412 条训练 / 1149 条验证 |

## 训练配置

LoRA `r=64, alpha=64, dropout=0.05`，`enable_lm=true, enable_dit=true, enable_proj=false`；
`learning_rate=1e-4`，`batch_size=2 × grad_accum=8 × 1 GPU = 等效 16`，
**3 epoch = 1390 步**，`warmup=139`，`weight_decay=0.01`，`max_grad_norm=1.0`，
`max_batch_tokens=8192`，`sample_rate=16000` / `out_sample_rate=48000`。
单卡 RTX 4090D 48GB。交付的是 `latest`（step 1390）——历史实验一致表明
**val loss 与验收指标不相关**，不按 val loss 选 checkpoint。

> 训练强度的经验：同一份数据从 1 epoch 加到 3 epoch 总体 CER 明显改善（0.066 → 0.041），
> 但继续堆数据量/步数（77h、4177 步的两轮更大尝试）反而过拟合退化——**12–15h × 3ep 是当前的甜点位**。

## 推理建议

与线上保持一致：`cfg_value=1.8`、`inference_timesteps=20`、`normalize=False`
（基座文本归一化只对 zh/en 生效，非中文一律走英语规则，会破坏 th/id/tl 的正字法）、
reference-only 模式、参考音频 4–5s 最佳。中文参考音频的音色贴合度高于英文
（基座特性），跨语言克隆场景优先选中文 ref。
