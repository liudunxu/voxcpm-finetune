---
license: cc-by-sa-4.0
base_model: openbmb/VoxCPM2
base_model_relation: finetune
tags: [voxcpm, voxcpm2, tts, lora, fine-tune, multilingual, dubbing, thai, indonesian, tagalog]
language: [th, id, tl, zh, en]
---

# VoxCPM 2 三语种配音微调（th / id / tl）

在 [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)（Apache-2.0）上做的 **LoRA 联合微调**，
目标是提升泰语 / 印尼语 / Tagalog 的**短剧配音**合成质量，并对齐主调方 OmniVoice 的真实调用口径
（单句 cue、跨语言 reference-only、cfg 1.8 / 20 步）。

一个 LoRA 同时覆盖三个语种，另含少量中文与英文回放做防遗忘。**验收是分语种做的**，见下面「验收结果」。

当前版本为 **r23**（2026-10-21）：16.6 小时混合数据，从 r20 系checkpoint续训而来
（r20 基于 r13x 配比 + 情绪前缀训练，r21→r23 逐轮加训 id/th/tl 增量数据，累计 2909 步）。
历史评测记录了**情绪指令下的音色一致性**（情绪前缀导致的
说话人相似度漂移 Δsim 从上一稳定版的 −0.024 收窄到 **−0.013**）与**三语种 CER 全优基座**。

> **2026-10-22 复核警告**：r23 训练见过 v2 的 **21/30 条 TL 评测文本**，
> 以及 drama 的 **7/8 条 TL 独立文本**。下表分数保留作历史记录，
> **不能把 TL 收益或总体均值当作独立泛化/完整质量验收证明**。
> 本次未核线上加载权重、未更换部署；声学、口音、自然度与情绪质量未完成验收。
> 审计与下一单变量实验见 `docs/finetune_playbook.md` 的 2026-10-22 节。

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
| th | 0.1155 | **0.0877** | **↓24%** | 0.0521 → 0.0417 |
| id | 0.0230 | **0.0177** | ↓23% | 0.000 → 0.0104 |
| tl | 0.0184 | **0.0135** | ↓27% | 0.000 → 0.000 |
| zh（回放） | 0.0515 | 0.0880 | ↑0.037 ⚠️ | 0.000 → 0.0278 |
| en（回放） | 0.0153 | **0.0129** | ↓16% | 0.000 → 0.000 |
| **总体** | **0.0490** | **0.0423** | **↓14%** | 0.0141 → 0.0169 |

说话人相似度（对参考音频，WavLM X-vector）：0.9121 → **0.9149**（无漂移，网内 ±0.02）。

**生产链路验收**（`eval_cases/drama_chain_prod_20261021.jsonl`：112 case = 28 句短剧台词
× 中文正常/短参考音 × 有无生产形态英文情绪前缀，× 3 seed = 336 条）：
th 0.1125→**0.0773**、tl 0.0106→0.0139、id 0.0688→0.0752；
**情绪前缀导致的音色漂移 Δsim(emo−plain)：base −0.0140、上一稳定版 −0.0236、本版 −0.0127**。

**id 韵律探针**（18 条跨语言 ref id case × 3 seed）：CER **中位 0.0000**
（54 条中 **46 条 CER=0**，不是原记载的 52 条；均值 **0.0672** 受少数异常拉高。
最差项为英文 ref 的 `id_ms_colloquial_en` / seed43，CER **3.1622**、时长 **20.8s**，
ASR 显示疑似额外续写/重复；尚未人工听评，不能说仅有轻度错误）。

> ⚠️ zh 回放 CER **+0.0365**，**未触发 0.05 退化红线**；zh 是防遗忘回放语种不是目标语种。
> 目标语种的最终语言质量（自然度、口音地道性）以母语者盲听为准，离线 CER 只做诊断；
> **印尼语口音的母语评审尚未完成**，本版本按离线指标先行发布。

## 已知问题（重要，请先读）

1. **数字与货币类 case 方差极大，别用单 seed 结论。** 数字/金额短句在 base 与各轮微调上
   都可能整条翻转（同一条 case 在不同 seed 下 CER 可从 0.000 跳到 1.4+）。
   本版验收用 3 个 seed 配对缓解，但数字类仍是最不稳定的部分。
2. **规避建议**：VoxCPM2 的文本归一化只对 zh/en 生效，`th/id/tl` 是**原样透传**，
   所以阿拉伯数字会直接进模型（`fil` 在主调方已有 verbalize 数字的 TN）。
   **把数字 verbalize 扩到 th/id 会比换模型更有效**。
3. **id 口语台词在低 seed 下仍有小概率重复/跑飞**：中文参考音 + 印尼语口语台词链路
   4/96 条 CER>0.3，最高 **0.9677**，ASR 提示疑似重复/额外输出，尚待逐条听评；集中在戏剧语域
   （训练数据以朗读/自然口语为主，无印尼语戏剧语料）。生产侧建议沿用内容 QC + 换 seed 重试。
4. **情绪指令**：本版起训练包含生产形态英文情绪前缀（Thai 情感语料 + id/tl 中性前缀接种），
   情绪指令下的音色一致性显著改善（Δsim −0.024→−0.013）；但**情绪"表现力"本身未经母语者评审**，
   只声称"加了情绪指令不再明显跑偏音色"，不声称情绪演技提升。
5. **塑料音 / 金属音 / 娃娃音**：尚未通过同输入诊断区分参考、生成、VAE 和后处理的贡献，
   不能宣称 LoRA 结构上修不到。本 LoRA 更新 LM/DiT；即使 `stop_proj` / `stop_head` 冻结，
   它们接收的 LM hidden states 仍可变化。先做固定输入多 seed 模型/参数单变量与必要的 VAE
   重建诊断，再决定定向训练；降 CFG/加步数也不是已证实的通用修复。
6. **印尼语口音说明**：本版 id 数据全部来自印尼本土来源（FLEURS id / Gigaspeech2 id /
   Common Voice id，id 占比 35.9% 为三语种最大），训练集中**不含任何马来语（ms）数据**；
   基座预训练自带的口音偏向无法靠微调保证根除，口音地道性以印尼语母语者盲听为准
   （本版本发布时该评审尚未完成）。

## 训练数据与署名

⚠️ **本版本（r23）的训练血统包含 THAI-SER（CC-BY-SA-4.0）泰语情感语料**（r20 阶段引入，
后续 r21–r23 在其权重基础上续训）。因此本模型整体按 **CC-BY-SA-4.0** 发布——再分发
或二次微调后的分发须以相同许可共享并署名。其余来源均为可商用许可。

| 来源 | 语种 | 许可 | 署名 / 链接 |
|---|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | th, tl(`fil_ph`), id, zh(`cmn_hans_cn`), en(`en_us`) | **CC-BY-4.0** | Google FLEURS，Arora et al. 2022；文本取 `raw_transcription`（保留大小写与标点） |
| [Gigaspeech 2](https://huggingface.co/datasets/speechcolab/gigaspeech2) | th, id（`dev` 分片） | **Apache-2.0** | SpeechColab，Du et al. 2025 |
| [Common Voice 22（社区镜像 fsicoli/common_voice_22_0）](https://huggingface.co/datasets/fsicoli/common_voice_22_0) | th, id | **CC0** | Mozilla Common Voice 22.0 |
| [YODAS2-Sidon th](https://huggingface.co/datasets/Chalermdej/yodas2_sidon_th_tts) | th | **CC-BY-3.0** | 上游为 YouTube 语料的 YODAS2 + Sidon 降噪 |
| [THAI-SER](https://github.com/vistec-AI/THAI-SER) | th | **CC-BY-SA-4.0** | VISTEC & AIResearch.in.th，Suwanmanee et al.；情感标注语音，用于情绪指令跟随训练 |

CC-BY / CC-BY-SA 要求署名，上表即为署名。音频经二次加工（16kHz 重采样、裁首尾、3–30s 过滤、
尾静音硬上限 0.15s）后用于训练，加工不改变上游许可义务。

配比（按**有效音频时长**，不是条数）：

| 语种 | 时长 | 占比 | 构成 |
|---|---|---|---|
| id | 5.95 h | 35.9% | FLEURS + Gigaspeech2 + Common Voice 22 |
| tl | 5.15 h | 31.1% | FLEURS |
| th | 4.58 h | 27.7% | FLEURS + Gigaspeech2 + Common Voice 22 + YODAS2 |
| zh | 0.59 h | 3.5% | FLEURS（防遗忘回放） |
| en | 0.29 h | 1.8% | FLEURS（防遗忘 + Taglish 英文词） |
| **合计** | **16.55 h** | 100% | 7996 条训练 / 1367 条验证 |

## 训练配置

LoRA `r=64, alpha=64, dropout=0.05`，`enable_lm=true, enable_dit=true, enable_proj=false`；
`learning_rate=1e-4`，`batch_size=2 × grad_accum=8 × 1 GPU = 等效 16`，
`weight_decay=0.01`，`max_grad_norm=1.0`，`max_batch_tokens=8192`，
`sample_rate=16000` / `out_sample_rate=48000`。单卡 RTX 4090D 48GB。
本版为续训链：r20（**1162 步**，3ep，含情绪前缀训练）→ r21（+369 步）→ r22（+628 步）→
r23（+750 步，含本轮 th/tl 增量数据），**累计 2909 步**。
延长余弦总步数并恢复 scheduler 可使 LR 回升，不能描述为连续接余弦尾部；
代码/本地复现与未核远端轨迹的边界见 `docs/finetune_playbook.md`。
交付的是 `latest`；val loss 不能代替分语种生成验收，不据此单独选 checkpoint。

> 训练强度的经验：同一份数据从 1 epoch 加到 3 epoch 总体 CER 明显改善（0.066 → 0.041），
> 但更大数据/步数的组合尝试出现退化；数据、配比与预算同时改变，
> **不能据此推导通用最佳时长/步数或“超过某步数必过拟合”**。

## 推理建议

与线上保持一致：`cfg_value=1.8`、`inference_timesteps=20`、`normalize=False`
（基座文本归一化只对 zh/en 生效，非中文一律走英语规则，会破坏 th/id/tl 的正字法）、
reference-only 模式、参考音频 4–5s 最佳。中文参考音频的音色贴合度高于英文
（基座特性），跨语言克隆场景优先选中文 ref。
