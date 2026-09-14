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

口径：`eval_cases/omnivoice_prod.jsonl`，**28 个生产形态 case × seed 42/43/44 = 84 条**，
`cfg_value=1.8`、`inference_timesteps=20`、`retry_badcase=False`（禁用坏例重试以保证 A/B 同条件），
统一使用主调方线上那条 **4.47s 菲律宾语参考音频做跨语言 reference-only**，
ASR 用 `large-v3`。CER/WER 越低越好；th 词间无空格只有 CER，vi 的 WER 是音节级口径，不与词级横向比。

| 语种 | base CER | 本模型 CER | base WER | 本模型 WER | 疑似漏尾 base→本模型 |
|---|---|---|---|---|---|
| th | 0.0282 | **0.0141** ↓50% | — | — | 0.0 → 0.0 |
| ms | 0.0852 | **0.0630** ↓26% | 0.1762 | **0.1151** ↓35% | 0.133 → 0.133 |
| tl | 0.1166 | **0.0998** ↓14% | 0.0944 | **0.0903** | 0.167 → 0.167 |
| id | 0.1036 | 0.1280 ↑24% | 0.2111 | 0.2389 | 0.2 → 0.2 |
| vi | 0.0686 | 0.1176 ↑71% | 0.1000 | 0.1389 | 0.0 → 0.0 |
| zh | 0.0 | 0.0 | — | — | 0.0 → 0.0 |
| en | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 → 0.0 |
| **总体** | **0.0774** | 0.0827 | — | — | 0.0952 → 0.0952 |

文本相似度 0.9477 → 0.9442（基本不变）。

### 质检指标（阈值移植自 OmniVoice 生产口径）

| 指标 | base | 本模型 | 说明 |
|---|---|---|---|
| CER（**剔除数字类** case） | 0.0022 | 0.0030 | 两边都已饱和：非数字台词基座本来就几乎全对 |
| 少读率（漏尾） | 0.0952 | 0.0952 | 相同 |
| 多读率（>1.4× 参考长度） | 0.0119 | 0.0119 | 相同，无跑飞 |
| 金属音检出率 | 0.0119（1/84） | 0.0357（3/84） | **该指标不可用**，见下方说明 |
| 说话人相似度（对参考音频，WavLM X-vector） | 0.9421 | 0.9411 | 基本持平；同人上限约 0.997、跨说话人约 0.514 |
| 同 ref 跨 seed 音色一致性 | 0.9603 | **0.9622** | 略稳 |
| 语速（字/秒） | 10.78 | 9.41 | 见下条：不是变慢，是尾部静音变长 |
| 有声段语速（字/秒） | 11.52 | 11.32 | **−1.7%，基本没变** |
| 尾部静音 | 0.084 s | **0.292 s** | 最长 0.46s、p90 0.42s，**0/84 越过 0.5s 上限** |

`speaker_sim` 用 **WavLM X-vector**（`microsoft/wavlm-base-plus-sv`，VoxCeleb 上训的说话人验证
模型，512 维）的余弦。尺子自检：跨说话人 0.32-0.62（均值 0.514）、同人 0.995-0.998，
**间隔 0.48**，有足够区分力。（早先用 librosa MFCC 余弦，实测全部饱和在 0.985-0.996、
分辨不出差异，已废弃。与 OmniVoice 生产用的 modelscope ERes2NetV2 **刻度不可互换**。）

**音色贴合度随参考音频的语言显著变化**，这一点对使用者最重要：

| 参考音频语言 | base 贴合度 | 本模型贴合度 | 跨 seed 一致性 base→本模型 |
|---|---|---|---|
| Tagalog（4.47s） | 0.9421 | 0.9411 | 0.9603 → **0.9622** |
| 中文（4.77s） | 0.9250 | 0.9264 | 0.9577 → **0.9635** |
| 英文（4.32s） | **0.8587** | **0.8673** | 0.9461 → **0.9613** |

⇒ **英文参考音频的音色贴合度明显最低（0.86 vs 0.94）**，这是基座特性、不是本模型引入的
（base 同样低）。在跨语言克隆场景下，**能选中文参考音频就别选英文的**，贴合度高约 0.07。
⚠️ 每种 ref 语言目前只有 1 条样本，ref 语言与 ref 说话人/录音质量是混淆的，
上面是方向性结论而非精确估计。

本模型对贴合度几乎没有改变（−0.0010 / +0.0013 / +0.0086），但**跨 seed 一致性三种 ref 下都略升**
（+0.0018 / +0.0058 / +0.0152）。贴合度改不动是架构性的：本 LoRA `enable_proj=false`，
而 `enc_to_lm_proj` 与 `fusion_concat_proj` 正是参考音频条件进入 LM/DiT 的通路，
它们被冻结在基座权重上。**要提升音色贴合度必须开 `enable_proj` 并且训练数据里带同说话人
`ref_audio`**（本次训练 ref 覆盖率为 0%）。

> ⚠️ **金属音检出率这一行请忽略，该指标在本仓库的用法下不可用。** 它的阈值移植自
> OmniVoice 生产代码，但那是在 OmniVoice **后处理过**的音频上标定的（上线前有 peak
> ceiling 0.94、level match、可选 noise gate），而这里分析的是裸模型输出。
> 拿 84 条对照人工盲听的结果是 **4 误报 / 1 漏报 / 0 命中**：自动检出的 4 条
> （base 1 条、本模型 3 条）人工全部判 `noise=False`、自然度 5/5；
> 而人工唯一标了 `noise=True` 的那条，自动 score 只有 0.0694，远低于门限。
> 4 条误报的 score 全挤在 0.29-0.33 刚好压线。**不要用它判断本模型是否引入金属音。**

### 人工盲听（84 对全评，甲乙随机盲化）

同一批 84 条音频，A=基座、B=本模型，顺序随机且不暴露来源，逐条打自然度与可懂度（1-5）
并标记截断/噪音。

| 语种 | 自然度 A→B | 可懂度 A→B | B 胜 / 平 / 负 | 截断 A→B | 噪音 A→B |
|---|---|---|---|---|---|
| tl | 4.61 → **5.00** | 4.83 → **5.00** | **5 / 13 / 0** | 0→0 | 0→0 |
| vi | 4.83 → 4.83 | 4.78 → **4.94** | 2 / 14 / 2 | 0→0 | 0→0 |
| ms | 4.87 → 4.80 | 4.80 → **5.00** | 2 / 11 / 2 | **1→0** | 0→0 |
| th | 4.75 → 4.75 | 5.00 → 4.92 | 2 / 8 / 2 | 0→0 | **1→0** |
| id | 5.00 → 4.93 | 5.00 → 4.93 | 0 / 14 / 1 | **1→0** | 0→0 |
| zh | 5.00 → 4.67 | 4.67 → 4.67 | 0 / 2 / 1 | 0→0 | 0→0 |
| en | 4.67 → 5.00 | 5.00 → 5.00 | 1 / 2 / 0 | 0→0 | 0→0 |
| **总体** | **4.81 → 4.87** | | **12 胜 / 64 平 / 8 负** | **2→0** | **1→0** |

按最小差值门槛（`负 − 胜 ≥ 2` 且 `负 ≥ 3`）**没有语种触发退化**。
`id`（0胜/14平/1负）与 `zh`（0胜/2平/1负）若按朴素的「负 > 胜」会误判成退化，
那只是 12-18 条样本里的单条听感波动。

**诚实的结论**：64/84 是平局，总体自然度 4.81→4.87 在这个样本量下**不构成"更自然"的结论**。
可以声称的是：`tl` 自然度与可懂度双升且 5 胜 0 负、`vi` 可懂度 4.78→4.94、
**截断从 2 例降到 0 例**。不能声称的是：整体自然度提升、情绪表现力提升、去念稿感改善。

盲听还交叉验证了一条自动指标：8 条「人工判 B 更差」里有 6 条是 **B 的音频明显变长而 CER 完全没变**，
最极端的 `ms_manglish`（seed 43）从 1.76s 变成 3.52s、文本反而更准（CER 0.094→0.000）
但自然度被从 5 打到 3 —— 与上面「尾部静音 0.084s→0.292s」的自动测量指向同一个缺陷。

**上面这张表用的是 Tagalog（菲律宾语）参考音频。换 ref 语言结论会变**，因为线上真实用法
是「中文或英文参考音频 → 目标语种」，所以三种都测了（同一批 28 case × seed 42/43/44）：

| 参考音频 | 总体 CER base→本模型 | **非数字 CER** base→本模型 | 触发退化的语种 |
|---|---|---|---|
| Tagalog | 0.0774 → 0.0827 | 0.0022 → 0.0030 | id、vi |
| 中文 | 0.1260 → **0.1128** | 0.0277 → 0.0268 | 仅 tl |
| 英文 | 0.1537 → **0.1179** | **0.0614 → 0.0045（−93%）** | 仅 id |

⇒ **用中文或英文参考音频时本模型整体优于基座**，退化面比 Tagalog ref 小得多。
另外基座自己在中/英 ref 下就差得多（0.0774 → 0.126 / 0.1537），说明跨语言 ref 距离越远越难，
这是基座特性。⚠️ 每种 ref 语言只有 1 条样本，ref 语言与 ref 说话人是混淆的，
上面是方向性结论。

**读法**：th / ms / tl 三个语种优于基座，zh / en 无退化，**所有非数字台词零退化**
（非数字 CER 两边都是 0.002-0.003，已饱和）；id 与 vi 的退化完全集中在数字与货币短句上。

## 已知问题（重要，请先读）

1. **vi / id 的数字与货币短句可能退化。** 逐 case 看：`vi_digit`（"Chuyến bay 926..."）
   CER 0.020→0.118、`vi_digit_2`（"Giá vé là 250.000 đồng."）0.373→0.588、
   `id_digit_2`（"Harganya 250 ribu rupiah."）0.381→0.571。
   同时**修好**了另一批：`ms_digit_3`（"RM45,000"，基座会念成印尼盾 `Rp`）、
   `id_digit_3`（"GA 876 ... pukul 06.15"）、`tl_digit`、`th_digit`、`th_digit_2`、`vi_time`。
   ⇒ 数字类整体是**有得有失**，不是单向变好。
   其中 `id_digit_2` 高度疑似评测假象：模型输出被 Whisper 转写成 `Harganya Rp250.000.`，
   而参考文本是 `250 ribu rupiah` —— 可能是 ASR 把口播金额归一成了符号形式，不一定是念错，
   CER 分不开这两者，需母语盲听裁定。
2. **数字类指标方差极大，别用单 seed 结论。** 同一条 `vi_digit_3`（"Anh ấy sinh năm 1995."）
   在 base / 三次微调上分别是 `0.000 / 1.4375 / 0.000 / 0.958`，整条翻转、非单调；
   ms 在**配比完全相同**的两轮之间也差出 0.042。28 case × 3 seed 的分辨率已接近 run 间方差。
3. **规避建议**：VoxCPM2 的文本归一化只对 zh/en 生效，`th/vi/id/ms` 是**原样透传**，
   所以阿拉伯数字会直接进模型。主调方对 `fil` 已有 verbalize 数字的 TN；
   **把同样的处理扩到 th/vi/id/ms 会比换模型更有效**（`fil` 的数字 case 在本次验收里表现最好）。
4. **没有改善、也不要期待改善的部分**：
   - **情绪表现力 / 去念稿感**：本轮训练数据里没有任何已核实的真人表演或情感语料
     （见下面数据来源），所以**不声称情绪表现力提升**。
   - **塑料音 / 金属音 / 娃娃音**：这类音色伪影主要是 AudioVAE、CFG 与采样步数的产物，
     而本 LoRA 只作用于 LM 与 DiT 的 q/k/v/o（`enable_proj=false`，`stop_proj` / `stop_head` 全冻结），
     结构上修不到它们。线上对应的处理是降 CFG 到 1.2–1.6 并把步数提到 24–30 重试。
   - **参考音频克隆能力**：训练数据没有可靠的同说话人身份，因此 `ref_audio` 覆盖率为 **0%**
     （低于官方建议的 30–50%）。按"缺少可靠身份不强凑"的原则没有伪造配对，
     ref 条件路径未被本 LoRA 专门训练。实测说话人相似度 0.9935 → 0.9942（**未下降**），
     但该指标是 MFCC 余弦的低可信档、动态范围很小，只能排除"明显损坏"；
     对音色细节有要求的场景仍建议自行 A/B 盲听确认。
5. **尾部静音变长**：0.084s → 0.292s（最长 0.46s、p90 0.42s，**0/84 越过 0.5s 上限**），
   导致总时长 +12.3% 而**有声段语速几乎不变**（11.52 → 11.32 字/秒）。
   成因大概是训练数据里 32% 是 YouTube 自发口语（比朗读多犹豫与停顿）。
   OmniVoice 生产默认 `trim_silence_vad=True` 会裁掉首尾静音，所以线上基本无感；
   但若下游不做裁切，同一段台词会长约 12%，**需要卡 cue 时长的场景请注意**。

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

| 语种 | 时长 | 构成 |
|---|---|---|
| th | 13.20 h | FLEURS 5 + YODAS2 3 + Gigaspeech2 9（权重份） |
| vi | 13.20 h | FLEURS 7 + Gigaspeech2 10 |
| id | 13.20 h | FLEURS 7 + Gigaspeech2 10 |
| tl | 13.20 h | FLEURS 17 |
| ms | 13.20 h | FLEURS 17 |
| zh | 7.76 h | FLEURS 10（防遗忘回放） |
| en | 3.88 h | FLEURS 5（防遗忘 + Taglish/Manglish 英文词） |
| **合计** | **77.63 h** | 33082 条训练 / 996 条验证，单条原始音频最多曝光 3 次 |

各语种请求占比与实际占比均为 100%（th/tl/vi/id/ms 各 17%，zh 10%，en 5%）。

## 训练配置

LoRA `r=64, alpha=64, dropout=0.05`，`enable_lm=true, enable_dit=true, enable_proj=false`；
`learning_rate=1e-4`，`batch_size=2 × grad_accum=8 × 1 GPU = 等效 16`，
**1 epoch = 2068 步**，`warmup=206`，`weight_decay=0.01`，`max_grad_norm=1.0`，
`max_batch_tokens=8192`，`sample_rate=16000` / `out_sample_rate=48000`。
单卡 RTX 4090D 48GB，显存占用约 24GB。
验证集 `loss/total` 1.0775（step 0）→ 0.9176（step 750，最优）→ 0.9767（step 2067）；
交付的是 `latest`（step 2068），因为**验证 loss 与验收指标在本轮不一致**
（step 750 在所有语种上都不如 latest），所以不按 val loss 选 checkpoint。

完整逐轮记录（含两次未通过的尝试与失败原因）见训练仓库的 `docs/runs.md`。

## 推理建议

与线上保持一致：`cfg_value=1.8`、`inference_timesteps=20`、`normalize=False`
（基座文本归一化只对 zh/en 生效，非中文一律走英语规则，会破坏 th/vi/id/ms 的正字法）、
reference-only 模式、参考音频 4–5s 最佳。
