---
license: apache-2.0
base_model: openbmb/VoxCPM2
base_model_relation: finetune
tags: [voxcpm, voxcpm2, tts, lora, fine-tune, multilingual, dubbing, thai, vietnamese, indonesian, malay, tagalog]
language: [th, vi, id, ms, tl, zh, en]
---

# VoxCPM 2 五语种配音微调（th / vi / id / ms / tl）

在 [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)（Apache-2.0）上做的 **LoRA 联合微调**，
目标是提升泰语 / 越南语 / 印尼语 / 马来语 / Tagalog 的**短剧配音**合成质量，
并对齐主调方 OmniVoice 的真实调用口径（单句 cue、跨语言 reference-only、cfg 1.8 / 20 步）。

一个 LoRA 同时覆盖五个语种，另含中文与英文回放做防遗忘。**验收是分语种做的**，见下面「验收结果」。

当前版本为 **r8**（2026-09），在 r2 配方（joint_omni2）基础上用 v2 验收集重新验证：
漏尾率降到历次最低（3.85% → 1.32%），ms 的 CER 降到历次最好（0.068 → 0.012）。

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

**B. 用 LoRA 增量（省 4.6GB 下载）**

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
> 本 LoRA 是 `r=64, alpha=64, dropout=0.05, enable_lm=true, enable_dit=true, enable_proj=false`，
> 实测 `loaded=384, skipped=0`。

## 验收结果

口径：`eval_cases/omnivoice_prod_v2.jsonl`，**182 个生产形态 case × seed 42–46 = 910 条**，
`cfg_value=1.8`、`inference_timesteps=20`、`retry_badcase=False`（禁用坏例重试以保证 A/B 同条件），
跨语言 reference-only，参考音频覆盖 **13 条、4 种语言**（zh ×4 / en ×4 / tl ×4 / 线上那条 4.47s fil），
避免单条 ref 带来的说话人混淆。ASR 用 `large-v3`。
CER/WER 越低越好；th 词间无空格只有 CER，vi 的 WER 是音节级口径，不与词级横向比。

| 语种 | base CER | 本模型 CER | 变化 | 疑似漏尾 base→本模型 |
|---|---|---|---|---|
| th | 0.1142 | **0.0957** | ↓16% | 0.069 → 0.044 |
| ms | 0.0681 | **0.0116** | **↓83%** | 0.094 → **0.000** |
| vi | 0.0730 | **0.0647** | ↓11% | 0.056 → 0.025 |
| id | 0.0230 | **0.0194** | ↓16% | 0.000 → 0.006 |
| tl | 0.0184 | **0.0146** | ↓21% | 0.000 → 0.000 |
| zh | 0.0580 | 0.0775 | ↑0.020 ⚠️ | 0.000 → 0.000 |
| en | 0.0150 | 0.0173 | ↑0.002 | 0.000 → 0.000 |
| **总体** | **0.0568** | **0.0423** | **↓26%** | **0.0385 → 0.0132** |

文本相似度 0.9597 → **0.9702**。剔除数字类 case 后总体 CER 0.0539 → **0.0401**。

> ⚠️ zh 回放 CER +0.0195，**未触发 0.05 退化红线**，漂移集中在 `zh_nat_01` / `zh_nat_10`
> 两条已知难 case（长句自然口语）；zh 是回放语种不是目标语种。en +0.002 属噪声级。

### 质检指标（阈值移植自 OmniVoice 生产口径）

| 指标 | base | 本模型 | 说明 |
|---|---|---|---|
| 少读率（漏尾） | 0.0385 | **0.0132** | **历次最低**；ms 从 9.4% 清零 |
| 多读率（>1.4× 参考长度） | 0.0011 | 0.0011 | 与基座持平，无跑飞 |
| 说话人相似度（对参考音频，WavLM X-vector） | 0.9091 | **0.9121** | 略升；同人上限约 0.997、跨说话人约 0.514 |
| 平均时长 | 5.82 s | 6.04 s | +3.8%，主要来自尾部静音 |
| 尾部静音 | 0.102 s | 0.185 s | p90 0.28s、最长 0.46s，**0/910 越过 0.5s 上限** |
| 有声占比 | 0.900 | 0.868 | 与尾静音变长同源 |
| 语速（字/秒） | 12.91 | 12.22 | 主要被尾静音摊薄，非发音变慢 |
| 金属音检出率 | 0.0011 | 0.0055 | **该指标不可用**，见下方说明 |

`speaker_sim` 用 **WavLM X-vector**（`microsoft/wavlm-base-plus-sv`，VoxCeleb 上训的说话人验证
模型，512 维）的余弦。尺子自检：跨说话人 0.32-0.62（均值 0.514）、同人 0.995-0.998，
**间隔 0.48**，有足够区分力。（与 OmniVoice 生产用的 modelscope ERes2NetV2 **刻度不可互换**。）

> ⚠️ **金属音检出率这一行请忽略，该指标在本仓库的用法下不可用。** 它的阈值移植自
> OmniVoice 生产代码，但那是在 OmniVoice **后处理过**的音频上标定的（上线前有 peak
> ceiling 0.94、level match、可选 noise gate），而这里分析的是裸模型输出。
> 此前用 84 条对照人工盲听的结果是 **4 误报 / 1 漏报 / 0 命中**。
> **不要用它判断本模型是否引入金属音。**

### 人工盲听

**本版本（r8）未做母语盲听**，验收结论全部来自上面的离线轨（182 case × 5 seed 的
配对 A/B）。上一版（r2，同配方）在 84 对全评盲听下无语种退化、截断 2→0；
本版离线指标全面不劣于 r2（漏尾率更低、ms 更好），但**自然度与情绪表现仍建议
使用方自行 A/B 盲听确认**后再上生产。

## 已知问题（重要，请先读）

1. **数字与货币类 case 方差极大，别用单 seed 结论。** 数字/金额短句在 base 与各轮微调上
   都可能整条翻转（同一条 case 在不同 seed 下 CER 可从 0.000 跳到 1.4+）。
   本版验收用 5 个 seed 配对缓解，但数字类仍是最不稳定的部分。
2. **规避建议**：VoxCPM2 的文本归一化只对 zh/en 生效，`th/vi/id/ms` 是**原样透传**，
   所以阿拉伯数字会直接进模型。主调方对 `fil` 已有 verbalize 数字的 TN；
   **把同样的处理扩到 th/vi/id/ms 会比换模型更有效**。
3. **没有改善、也不要期待改善的部分**：
   - **情绪表现力 / 去念稿感**：本轮训练数据里没有任何已核实的真人表演或情感语料
     （见下面数据来源），所以**不声称情绪表现力提升**。
   - **塑料音 / 金属音 / 娃娃音**：这类音色伪影主要是 AudioVAE、CFG 与采样步数的产物，
     而本 LoRA 只作用于 LM 与 DiT 的 q/k/v/o（`enable_proj=false`，`stop_proj` / `stop_head` 全冻结），
     结构上修不到它们。线上对应的处理是降 CFG 到 1.2–1.6 并把步数提到 24–30 重试。
   - **参考音频克隆能力**：训练数据没有可靠的同说话人身份，`ref_audio` 覆盖率为 **0%**
     （低于官方建议的 30–50%）。按"缺少可靠身份不强凑"的原则没有伪造配对。
     实测说话人相似度 0.9091 → 0.9121（**未下降**），但对音色细节有要求的场景
     仍建议自行 A/B 盲听确认。
4. **尾部静音变长**：0.102s → 0.185s（p90 0.28s、最长 0.46s，**0/910 越过 0.5s 上限**），
   总时长 +3.8%。成因是训练数据里约三分之一是 YouTube 自发口语（比朗读多犹豫与停顿）。
   OmniVoice 生产默认 `trim_silence_vad=True` 会裁掉首尾静音，所以线上基本无感；
   但若下游不做裁切，同一段台词会略长，**需要卡 cue 时长的场景请注意**。

## 训练数据与署名

全部为可商用、无 SA/NC/ND 限制的来源，因此本权重可对外分发。
**含 CC-BY-SA 的数据一律未使用**（例如 THAI-SER、Porjai），因为 SA 对模型权重是否构成
"改编物"没有判例，本项目按保守红线处理。

| 来源 | 语种 | 许可 | 署名 / 链接 |
|---|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | th, tl(`fil_ph`), vi, id, ms(`ms_my`), zh(`cmn_hans_cn`), en(`en_us`) | **CC-BY-4.0** | Google FLEURS，Arora et al. 2022；文本取 `raw_transcription`（保留大小写与标点） |
| [Gigaspeech 2](https://huggingface.co/datasets/speechcolab/gigaspeech2) | th, vi, id（`dev` 分片） | **Apache-2.0** | SpeechColab，Du et al. 2025 |
| [YODAS2-Sidon th](https://huggingface.co/datasets/Chalermdej/yodas2_sidon_th_tts) | th | **CC-BY-3.0** | 上游为 YouTube 语料的 YODAS2 + Sidon 降噪 |

CC-BY 要求署名，上表即为署名。音频经二次加工（16kHz 重采样、Silero VAD 裁首尾、
3–30s 过滤）后用于训练，加工不改变上游许可义务。

配比（按**有效音频时长**，不是条数）：

| 语种 | 时长 | 构成（权重份） |
|---|---|---|
| th | 12.79 h | FLEURS 5 + YODAS2 3 + Gigaspeech2 9 |
| vi | 12.79 h | FLEURS 7 + Gigaspeech2 10 |
| id | 12.78 h | FLEURS 7 + Gigaspeech2 10 |
| tl | 12.78 h | FLEURS 17 |
| ms | 12.78 h | FLEURS 17 |
| zh | 7.52 h | FLEURS 10（防遗忘回放） |
| en | 3.76 h | FLEURS 5（防遗忘 + Taglish/Manglish 英文词） |
| **合计** | **75.20 h** | 32307 条训练 / 971 条验证，单条原始音频最多曝光 3 次 |

各语种请求占比与实际占比一致（th/tl/vi/id/ms 各 17%，zh 10%，en 5%）。

## 训练配置

LoRA `r=64, alpha=64, dropout=0.05`，`enable_lm=true, enable_dit=true, enable_proj=false`；
`learning_rate=1e-4`，`batch_size=2 × grad_accum=8 × 1 GPU = 等效 16`，
**1 epoch = 2020 步**，`warmup=202`，`weight_decay=0.01`，`max_grad_norm=1.0`，
`max_batch_tokens=8192`，`sample_rate=16000` / `out_sample_rate=48000`。
单卡 RTX 4090D 48GB。
验证集 `loss/total` 0.8580（step 1750）→ 0.8436（step 2000）→ 0.9456（step 2019，末值）；
交付的是 `latest`（step 2020）——历史三轮一致表明 **val loss 与验收指标不相关**，
不按 val loss 选 checkpoint。

完整逐轮记录（含四次未通过的尝试与失败归因）见训练仓库的 `docs/runs.md`。

## 推理建议

与线上保持一致：`cfg_value=1.8`、`inference_timesteps=20`、`normalize=False`
（基座文本归一化只对 zh/en 生效，非中文一律走英语规则，会破坏 th/vi/id/ms 的正字法）、
reference-only 模式、参考音频 4–5s 最佳。中文参考音频的音色贴合度高于英文
（基座特性），跨语言克隆场景优先选中文 ref。
