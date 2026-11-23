# 开源 TTS 微调方法深挖：东南亚语种社区微调、停止/幻觉对齐、官方微调配方（2026-11-23）

## 结论与边界

接续 [voicestudio_training_transfer_20261120.md](voicestudio_training_transfer_20261120.md) 与
[dots_tts_training_review_20261120.md](dots_tts_training_review_20261120.md)，不重复已覆盖的 MOSS / IndexTTS /
CosyVoice / Qwen3 / Ming / Confucius4 / k2-fsa OmniVoice / dots / LongCat / Chatterbox / Dia / CSM / OuteTTS。
本次为联网静态调研 + 本地源码核对，**未训练、未生成音频、未改生产**。

新增最值得做的三件事（按成本排序，均为待验证假设）：

1. **解码期停止诊断（零训练）**：VoxCPM2 停止判定是 `stop_head(...).argmax()`（=固定 p_stop>0.5，未校准），
   上限 `max_len = min(text_len × retry_badcase_ratio_threshold + 10, max_len)`
   （`third_party/VoxCPM/src/voxcpm/model/voxcpm2.py:654`、`:1113`）。LoRA 改了 LM 隐状态而 stop 头冻结，
   **「stop 头在 LoRA 后失准」是推断未验证**。先在已知坏例（短 cue 跑飞、句末专名重复）上逐 patch 记录 p_stop 轨迹，
   区分「p_stop 始终起不来」与「有峰值但 <0.5 被漏掉」，再决定是否做阈值/长度先验实验。
2. **在 VoxCPM2 本体上已有 RL 先例：LoCA**（arXiv [2610.28035](https://arxiv.org/abs/2610.28035)，2026-10-20，已核摘要）。
   以 ASR 对齐定位内容错误区间、把句级 advantage 分配到声学帧，加正/负阈值防漂移；摘要报告多语 CV3 子集
   内容错误相对预训练 VoxCPM2 **−20.5%**、相对 GROW −14.3%。其前身 GROW（[2608.03215](https://arxiv.org/abs/2608.03215)，
   DiTAR 上 WER 2.016→1.558、SIM .676→.715，已核摘要）直接在 flow-matching 回归上做组相对 advantage 加权 + W2 约束，
   **不需要离散似然、不需要 ODE→SDE**——这是此前文档判定「RL 当前不适用连续 latent」的反例，结论需修正为
   「有可行路线，但仍排在数据主线之后」。
3. **短句失败是跨架构普遍问题，没有现成模型解决**：viXTTS（<10 词差、句尾怪音）、F5-TTS
   [#460](https://github.com/SWivid/F5-TTS/issues/460)（≤5 词丢首词/噪声）均有记录 ⇒ 继续「准确转写短片段 + 短句回归集」主线，
   不期待换模型或换配方自动解决。

## 1. 东南亚语种社区/厂商微调（逐条见原文链接）

| 模型 | 数据 / 做法 | 已知问题 | 许可 |
|---|---|---|---|
| [F5-TTS-THAI](https://huggingface.co/VIZINTZOR/F5-TTS-THAI)（[repo](https://github.com/VYNCX/F5-TTS-THAI)） | CV th ~160h + Porjai ~300h + CV-EN ~40h；v2 改 IPA 前端 | v1 漏词/重复，v2 改善但仍有错读；非泰语参考需泰文音译 ref_text | CC-BY-4.0 / MIT |
| [F5-TTS-Vietnamese-ViVoice](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice) | viVoice+VLSP 1000h；ASR 过滤坏转写、demucs 去 BGM、剔 <1s/>30s | 无指标 | NC-SA |
| [viVoice 数据集](https://huggingface.co/datasets/capleaf/viVoice) | 句边界切分、去多余静音；人工抽 1000 条估转写错误率 1.8% | — | NC-SA |
| [viXTTS](https://huggingface.co/capleaf/viXTTS) | XTTS-v2 续训 + 扩 tokenizer | **<10 词不稳定、句尾怪音** | CPML |
| [XTTSv2 新语种微调](https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages) | 扩词表 2000；GPT 全量 lr 5e-6 | **原文：微调 HiFiGAN 解码器反而变差** | 未写 |
| [zalopay/vietnamese-tts](https://huggingface.co/zalopay/vietnamese-tts) | F5 微调，200+h | 人名/生僻词差 | CC-BY-4.0 |
| [F5 印尼语 v1](https://huggingface.co/Eempostor/F5-TTS-INDO-FINETUNE)/[v2](https://huggingface.co/Eempostor/F5-TTS-INDO-FINETUNE-V2) | 新闻 43h / YouTube 132h | 情绪/音色匹配差、长文幻觉 | NC |
| [Vesper 印尼语 F5](https://github.com/happyvictor-Vesper/F5_TTS-finetuned-for-Indonesian) | id-tts-v2 ~67h（+INESCO 3h） | ref_text 不符或语种不符即出错 | NC |
| Tagalog | 未找到克隆型社区微调；只有单说话人 [mms-tts-tgl](https://huggingface.co/facebook/mms-tts-tgl) | — | — |

可借鉴：
- **转写准确度优先**（hynt ASR 过滤、viVoice 人工抽检估错误率）与 r53 教训一致。补一条可操作做法：
  双 ASR 一致过滤后**人工抽检约 1000 条估转写错误率**再入库；th 第二 ASR 可选 typhoon-whisper，vi 可选 PhoWhisper（均待核许可与接入成本）。
- 句边界切分 + 去多余静音 = 现行 MMS-FA 停顿切分 + 尾静音上限，vi 备料直接复用；**尾静音裁切不得裁到最后一个词**（F5 [#1207](https://github.com/SWivid/F5-TTS/issues/1207) 的 ref/text 失配教训）。
- 冻结声学解码器（XTTS 实测微调 HiFiGAN 变差）与现行冻结 AudioVAE 一致，维持。
- 「短句加句末标点更稳」（#460 的 "No..."）仅网友个案，可作推理侧 ≥3 seed 小诊断，不改生产文本。

不可借鉴：扩词表/IPA 前端（VoxCPM2 对 th/tl/id 0 UNK，改前端即改预训练分布；F5-THAI v1→v2 是用发音换重复）；
F5 的 speed/总时长参数（NAR 显式时长，VoxCPM2 无对应）；XTTS repetition_penalty（离散 token）；
全量微调超参；NC 数据源；合成语音阶段（[2609.25951](https://arxiv.org/abs/2609.25951)、[Wayu-Paxa 2609.03502](https://arxiv.org/abs/2609.03502)，仅核摘要）。

## 2. 停止 / 幻觉 / 重复的对齐方法

| 方法 | 需要 | 报告效果（来源） | 对 VoxCPM2+LoRA |
|---|---|---|---|
| **LoCA** | ASR 奖励 + 对齐 | CV3 内容错误 −20.5%（[2610.28035](https://arxiv.org/abs/2610.28035)，全文已核，见下） | **已在 VoxCPM2 上做过**；冻结 stop 头，改善内容不直接修停止 |
| **GROW** | ASR/SIM 奖励 | WER 2.016→1.558（[2608.03215](https://arxiv.org/abs/2608.03215)） | 适配连续 latent，可 LoRA 小实验 |
| FlowTTS-GRPO | ODE→SDE | F5 hard CER 9.00→7.86，LoRA r32（[2606.23190](https://arxiv.org/html/2606.23190v1)，未核） | 成本高；指出只 RL 声学侧可懂度不升 ⇒ 须覆盖 LM |
| Koel-TTS 偏好对 | 多候选 + CER/SSIM Pareto 选对 | DPO unseen CER 2.56→0.62（[2502.05236](https://arxiv.org/html/2502.05236v1)，未核） | 选对规则可借（**chosen 用真实录音会退化**）；DPO 本身需离散似然 |
| Llasa verifier 重排 | 多候选 + ASR | test-hard WER 11.09→4.38（[2502.04128](https://arxiv.org/html/2502.04128)，未核） | 零训练；与 DIS 现有共享预算重试同类，不新增 |
| VALL-E 2 RAS | 离散分布 | 无单独消融（[2406.05370](https://arxiv.org/html/2406.05370v1)） | 仅借「重复检测」做触发器 |
| MELLE stop 正样本加权 | — | （[2407.08551](https://arxiv.org/html/2407.08551v2)） | 已试 stop 权重 1→2 无效（r41）；不再试 |
| Koel 注意力先验 / ELLA-V / RALL-E / F5R / DMOSpeech2 / SALAD | cross-attn、音素交错、时长预测器或语义 token | — | 需改架构，不做 |

**LoCA 全文核对**（Table 6 等）：冻结 AudioVAE、patch encoder 与 **stop head**；未说明 LoRA 还是全量、训练哪些模块；
数据为 Emilia 5 万条（zh/en/de/fr/ja/ko），组大小 8，lr 2e-6、warmup 50、wd 0、梯度裁剪 0.5，曲线至 2000 步；**单训练 seed**；
奖励 = exp(−2.5·Qwen3-ASR 错误率) + 0.5×声纹（组内标准化）；严重失败 SFR = 错误率>0.5 / 全静音 / 撞长度上限 / 生成错误 /
尾静音≥1.2s / 内部静音≥1.5s。分语种错误率 base→LoCA：zh 3.996→3.238、en 5.193→4.507、de 5.551→3.878、es 4.571→3.608、
fr 10.657→9.027、it 5.023→3.731、ru 6.219→4.723、hard-zh 9.952→7.186、hard-en 7.665→7.143。**无 th/id/tl/vi 结果，不报告时长/多读效应，
未发布代码**。⇒ 复现需自写 rollout + ASR 对齐 + 帧级 advantage 回归，工程量为「一个小项目」，不是改配置。

## 3. 上游 VoxCPM 进展（本地 `git fetch` 核对）

submodule 钉 `f772e49`，上游 main `300e426` 领先 13 提交，**全部是工程项，无新训练方法**：数据预处理提速
（`preprocessing_num_workers`、rank0 先建缓存）、相对音频路径按清单目录解析、HF Hub 自动下载、WebUI i18n。
bump 风险一条：新版 `_manifest_features` **按首行推断清单 schema**，若字段首行为 null 或跨行类型不一致可能加载失败；
本项目清单为绝对路径（`pipeline.py:904`），相对路径改动无影响。bump 前用现行 r50 清单跑一次加载冒烟。

## 4. 官方/其他模型微调配方

### 4.1 VoxCPM2 官方与第三方（最相关）

- 官方 [finetune 文档](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html) / [FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html) /
  [openbmb.cn](https://openbmb.cn/docs/en/guides/voxcpm/voxcpm-2/finetune)：LoRA lr 1e-4、r32/64、alpha r 或 2r；单说话人 1–3 epoch、
  更多常变差；**新语言**才需 500h+ 全量并混 10–20% 中英。th/tl/id/vi/en 均在官方 30 语内 ⇒ 本项目属「风格/领域」档，与现行 LoRA 路线一致。
  FAQ 自认 val loss 不能代表听感，应以推理挑 checkpoint（与本项目结论一致）。
- 上游 issue 与本项目风险同形：[#357](https://github.com/OpenBMB/VoxCPM/issues/357) 基座单词句（"Tak."）幻读多余音节——
  **短 cue 多读是基座固有**，不是 LoRA 引入；[#352](https://github.com/OpenBMB/VoxCPM/issues/352) **越南语偶发读完停不下来、尾部胡言**——
  vi 回归目标语种后须进短句/句末回归集；[#169](https://github.com/OpenBMB/VoxCPM/issues/169) 阿拉伯语全量微调后「给 prompt 时无视文本」；
  [#306](https://github.com/OpenBMB/VoxCPM/issues/306) 加泰罗尼亚语 ref 100% 仍被西语口音覆盖，无结论。
- **[arXiv 2606.26618](https://arxiv.org/abs/2606.26618)（v2 2026-09-24，已核摘要）**：VoxCPM2 单 LoRA 联合训 Khmer+韩语 25.5h，
  Khmer（基座不支持）MOS 3.85→4.23（r64，p<.001）；**韩语（基座已支持）无显著收益，且摘要称「高 rank 甚至伤害质量」**。
  **全文已核（Table III/IV）**：韩语 MOS base 3.65 / r8 3.73 / r16 3.71 / r32 3.76（p=.49）/ **r64 3.48（p=.02 显著变差）** / r128 3.54；
  Khmer r32 4.03 / r64 4.23 / r128 4.08；韩语发音分 r32 最高。**但迁移要打折**：其配置为 α=2r（LoRA 缩放是本项目 α=r 的 2 倍）、
  10k 步 × batch 16 ≈ 8 epoch（本项目 3 epoch）、无 replay、评审一致性低（Krippendorff α .26/.31）——韩语 r64 退化与过训/缩放混杂，
  且 r32 对韩语也无显著收益。⇒ r32 vs r64 仍值得在新口径下测（旧 r32 那轮被已证伪的首例即停门判退，不构成证据），但先验强度只算中等。
  另：其 val loss 最低 rank 与 MOS 最佳不一致，再次说明不以 val loss 选。

### 4.2 其他模型

| 模型 | 官方微调 | 可借之处 |
|---|---|---|
| Fish Speech（[S2 beta 文档快照](https://github.com/fishaudio/fish-speech/blob/3578e4e7099ee85464756dab27a3af86b5a21331/docs/en/finetune.md)） | LoRA r8/α16、lr 1e-4、只训 LLAMA；main 已删文档 | **不推荐在 RL 后训练模型上再微调**（分布偏移）——若将来做 LoCA 式 RL，顺序应是 SFT→RL，RL 后不再 SFT |
| Orpheus（[config](https://github.com/canopyai/Orpheus-TTS/blob/main/finetune/config.yaml)） | 全参 lr 5e-5、1 epoch；每说话人约 300 条 | 遗忘多发生在早期 —— 与 replay 纪律一致，无新动作 |
| ZipVoice（[run_finetune.sh](https://github.com/k2-fsa/ZipVoice/blob/master/egs/zipvoice/run_finetune.sh)） | 全量 lr 1e-4、10k 步 | **训练样本时长上限取数据 99 分位** —— 见下「时长上限」臂 |
| Kani-TTS（[pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)） | LoRA r16、rsLoRA、`max_duration_sec` 12 | 同上时长上限思路 |
| UtterTune（[2508.09767](https://arxiv.org/html/2508.09767v1)） | **LoRA 只挂 LM 注意力**，日语重音 .472→.975，CER 略升 | 「只开 LM」臂（本项目试过 DiT-only 与 LM+DiT，未试 LM-only） |
| MiniMax-Speech（[2505.07916](https://arxiv.org/abs/2505.07916)，仅摘要） | 冻结基座，speaker embedding / 每情绪独立 LoRA | 「按语种挂载 LoRA」的零训练思路 |
| Step-Audio-EditX（[scripts](https://github.com/stepfun-ai/Step-Audio-EditX/tree/main/scripts)） | SFT/DPO/GRPO（TRL） | 离散 token，不直接适用 |
| NeuTTS（[TRAINING.md](https://github.com/neuphonic/neutts/blob/main/TRAINING.md)） | 全量 10h 约 1000–2000 步 | 无新信息 |
| Higgs v2/TTS3、VibeVoice（仅 ASR 微调）、Spark-TTS、Kokoro | 无官方 TTS 微调入口 | 无 |

### 4.3 由此得到的、未试过的单变量候选（按信息量排序）

本地核对后剔除：**cosine 调度**（官方脚本已是 `get_cosine_schedule_with_warmup`，`yaml_builder` warmup=10% 步数）、
α=2r/rsLoRA（等价于已扫过的 lr/强度）、最早 checkpoint（r44/u10 已测）、checkpoint 平均（soup 已测）、逐条响度归一（与身份纪律冲突）。

| 候选 | 依据 | 为什么值得测 / 风险 |
|---|---|---|
| A. **r32 vs r64**（新口径重测） | 2606.26618：基座已支持语种高 rank 伤质量 | 旧 r32 结论来自失效判退门；成本 = 一轮训练 |
| B. **训练时长上限**（如 ≤12s 或按线上 cue 99 分位） | ZipVoice/Kani；短 cue 出分布是首要失败 | 直接对准多读；须按语种配额补总时长，否则混入「少数据」变量 |
| C. **LM-only LoRA** | UtterTune；停止/发音在 LM | 可能减少 r50 的音色代价（ERes2NetV2 −0.047）；官方称 DiT 对音质关键，须带声学门 |
| D. **带 ref 比例剂量**（30%→上限） | 生产 100% reference-only；#306 用 100% | 只能在 th/id/vi 可信身份内做，不伪造；r10 旧结论方向不利 |
| E. `training_cfg_rate` .1→.2 | #169「给 prompt 时无视文本」；线上 cfg 1.8 | 探索性；该参数在基座 `dit_config.cfm_config`，需复制基座 config 改值，`yaml_builder` 有守卫不允许当顶层参数 |
| F. 按语种挂载/强度（零训练） | MiniMax/UtterTune | r50 目前无语种退化 ⇒ 暂不需要；某语种将来退化时作为兜底，需 DIS 传语种 |

以上每项都是独立单变量，**排在 r54（TH/ID 准确短片段）之后**，A 与 B 优先（成本低、与主失败模式直接相关）。

### 4.4 k2-fsa/OmniVoice 微调方式复查（2026-11-23，用户点名；补 11-20 文档 §3 未覆盖的部分）

> 仍是 k2-fsa 的 600+ 语模型，不是生产同名服务；权重 CC-BY-NC，**只借方法不借权重/输出**。

已核原文：[lora_finetuning.md](https://github.com/k2-fsa/OmniVoice/blob/main/docs/lora_finetuning.md)、
[training.md](https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md)、
[train_config_finetune.json](https://raw.githubusercontent.com/k2-fsa/OmniVoice/master/examples/config/train_config_finetune.json)、
[data_preparation.md](https://github.com/k2-fsa/OmniVoice/blob/master/docs/data_preparation.md)、
[data_preparation_advanced.md](https://github.com/k2-fsa/OmniVoice/blob/master/docs/data_preparation_advanced.md)。

| 项 | k2-fsa 做法（原文） | 对本项目 |
|---|---|---|
| LoRA 目标层 | `q/k/v/o` **+ `gate/up/down_proj`（MLP）**；r16/α32/dropout .05；lr 1e-4；约 3% 参数可训 | **可借，新单变量 G**：本项目各轮只挂 attention q/k/v/o。VoxCPM2 `LoRAConfig.target_modules_lm/_dit` 可直接配置（`voxcpm2.py:148-149`；MiniCPM4 MLP 名为 `gate_proj/up_proj/down_proj`，`modules/minicpm4/model.py:229-231`），**无需改 submodule**。QLoRA 等「全线性层」实践亦支持；须核对 merge 与 OmniVoice 推理读取 `lora_config.json` 时目标层一致 |
| 全量训练的 I/O 层 | `modules_to_save: audio_embeddings, audio_heads`（音频 token 进出层全量训） | 对应 VoxCPM2 的投影层/停止头；`enable_proj`（r40）曾被已失效判退门否定。若 p_stop 诊断证实停止失准，可与「只解冻 stop_proj/stop_head」合并设计，不单独重开 |
| 全量微调配置 | 实际 json：lr **1e-5**、5000 步、warmup 1%、wd .01、grad clip 1.0（README 表写 5e-5，与 json 不一致，以 json 为准） | 与官方 VoxCPM 全量 lr 1e-5 一致，本项目不做全量 |
| 条件丢弃 | `drop_cond_ratio 0.1`、`language_ratio 0.8`（20% 样本不给语种标签） | VoxCPM2 无语种 token，语种丢弃不适用；cfg 丢弃已是 `training_cfg_rate 0.1` |
| 训练 prompt | `prompt_ratio_range [0, 0.3]`：prompt 取自**同一条**音频前缀（NAR 掩码预测） | **不借**：同源前缀 = continuation 形态，生产是 reference-only；本项目禁止同 `origin_audio` 配 ref |
| 参考降质 | Sidon 降噪原音 + 对**少量** prompt 加噪声/RIR（比例、SNR 未公开） | 与 11-20 结论相同，沿用 `ref_robustness` 阶段门，不扩 profile |
| 语种平衡 | data config 每个数据集 `repeat` 次数 | 本项目按有效时长配额、3× 上限，更严格，不改 |
| 错误处理 | `--skip_errors` 默认 False，失败写 `errors.jsonl` | 与本项目「失败显式中止/隔离」一致 |
| 数字 TN | `normalize_text=True`：zh/en 用 WeTextProcessing，其他语种整数用 `num2words` | **陷阱实测**：`num2words(1250, lang='fil')` 输出**芬兰语**「tuhat kaksisataa…」，`lang='tl'` NotImplementedError；th/id/vi 输出正确。生产 fil 走自有 filipino normalizer 不受影响，但**任何 Tagalog 数字处理不得用 num2words**。th/id/vi 的 num2words 可作为数据侧数字 verbalize 候选（基座 TN 对这三语透传数字） |
| 跨语言克隆 | README：参考音口音会带入输出 | 与本项目 zh/en ref→目标语的口音风险一致，母语评审仍是唯一判据 |

**新增单变量候选 G：LoRA 扩到 MLP 层**（LM 与 DiT 的 `gate/up/down_proj`，其余 r50 配方不变）。
依据是 k2-fsa 默认与「全线性层 LoRA」通行做法；可训参数约增数倍，需同步看音色代价。排在 A/B 之后、C 之前。

### 4.5 CosyVoice3 深挖（2026-11-23，用户点名；不重复 11-20 已写的 DiffRO / 奖励投机 / CosyVoice2 GRPO）

来源：[论文全文](https://arxiv.org/html/2505.17589)、[官方 llm.py](https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/llm/llm.py)、
[SFT 配置](https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml)、
[CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval)、[0.5B-2512 模型卡](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)。
仓库已迁至 QwenAudio 组织（旧链接重定向）。下列加 ✔ 的数字本人已复核原文/源码，其余为子代理读取。

**数据管线（§3）**
- ✔ 多 ASR 过滤：Faster-Whisper large-v3 / Canary-1B / SeamlessM4T-v2 三家转写，**两两平均 WER <15%** 才收。
- ✔ **按时长/文本比剔除两端**：语音 token 数 / 文本 token 数，**丢最小 1% 与最大 5%**——专门去掉「短音长文」与「短文长音」错配。
- ✔ **按强制对齐停顿重写标点**（MFA）：停顿 ≥300ms 补逗号、≤50ms 删表示停顿的标点。
- ✔ 音量逐条 `raw/max×0.6`（**与本项目禁逐条峰值归一冲突，不借**）。
- 说话人分离 + VAD + 事件检测切 <30s 单人片段，MossFormer2 降噪，按首尾帧能量剔除首尾词不完整片段。
- TN 自训练与「rare case」稳定性数据均用 CosyVoice 合成音频（**违反合成语料红线，不借**）。

**停止与长度（✔ 源码 `llm.py:173-216`）**：`min_len = 文本 token × 2`、`max_len = 文本 token × 20`，未到 min_len 前屏蔽 EOS；
离散 token 另有 RAS 防重复采样。对照 VoxCPM2：上限 `文本 token × 6 + 10` patch **已比它紧**；但**下限固定 2 patch、与文本长度无关**
（`voxcpm2.py:474`）——若 p_stop 诊断发现「少读/提前停」也存在 p_stop 早升，可加「按文本比例的最小长度」作为推理侧单变量，生产默认不变。

**发音控制（§2.3，✔ 表 13）**：扩词表混入拼音/CMU 音素；只替换**单音字/词**并混排（RepMono+MixPhn）在测试集纠错 zh 15/15、en 9/9，
优于全替换（69.2%/72.7%）。VoxCPM2 需扩 embedding/lm_head，LoRA 训不到、submodule 只读 ⇒ **不借**；泰语音译人名、Tagalog「CEO」
仍走 DIS 读音表（推理侧改写），不在训练侧教模型读非规范拼写。

**SFT 配方**：官方 recipe 是**全量**、lr 1e-5、llm/flow/hifigan 分开训，维护者称暂无 LoRA 计划（[#1158](https://github.com/QwenAudio/CosyVoice/issues/1158)）；
单说话人至少 1 小时、「repeat data is not ok」（[#321](https://github.com/QwenAudio/CosyVoice/issues/321)）。社区报告加说话人名 prompt 后 WER 上升
（[#1514](https://github.com/QwenAudio/CosyVoice/issues/1514)，非官方）。不借配方；「不靠重复凑量」与本项目 3× 上限一致。

**规模**：0.5B 上数据 3k→170k 小时 test-hard CER 27.60→6.96，继续到百万小时收益趋平；模型 0.5B→1.5B 收益小且 en 反而变差。
这是预训练量级规律，不推翻本项目 r17/r27/r53「LoRA 加量无收益或变差」的实测。CV3-0.5B 在 to-ko（zh prompt）WER 41.0 差于 CosyVoice2 24.8——
换代也会在单个跨语言方向退化，佐证「任一语种退化即不通过」。

**CV3-Eval**：9 语（zh/en/ja/ko/de/fr/ru/it/es），**无 th/tl/id/vi**；跨语言子集按「每目标语 × 3 种 prompt 语种」各约 600 条，hard 子集单列。
数据不可用，但**矩阵设计可借**：新开发集按「zh/en/fil ref × th/tl/id/vi 目标」成矩阵，hard（句末专名、外文借词、缩写）单列。
许可：代码与 0.5B-2512 权重标 Apache-2.0，但 GitHub 页另有「academic purposes only」字样，商用前须再核（本项目不用其权重，仅记录）。

**新增单变量候选**
- **H. 训练集按「有声时长 / 文本长度」分语种剔除两端**（最小 1%、最大 5%，其余同 r50）：自动化版本的 `text_duration_review`，
  直接对准短文长音 ⇒ 多读/跑飞。th 按字符、tl/id/vi/en 按词（`rate` 量纲规则）。成本低，**与 B（时长上限）同属长度分布类，二者先做 H**
  （H 不减少短句、只去错配；B 会砍长句总时长）。
- **I. 按 MMS-FA 停顿重写训练文本标点**（音频不变）：可能改善句内停顿与句末收尾；须先确认泰文对齐（见 r54 泰语切分）可用，排在 H 之后。
- 新开发集按 CV3-Eval 矩阵设计（流程改动，非训练臂）。

### 4.6 Confucius4-TTS 复查（2026-11-23，用户点名；补 11-20 文档 §2.1）

来源：[repo](https://github.com/netease-youdao/Confucius4-TTS)、[train_t2s.yaml](https://raw.githubusercontent.com/netease-youdao/Confucius4-TTS/main/config/train_t2s.yaml)、
[论文](https://arxiv.org/html/2608.11650)、issues（`gh api` 全量列出 22 条）。

**微调方式**：两阶段**全量**——T2S（离散语义 token AR，AdamW lr 2e-4、wd .01、betas (.9,.95)、cosine、warmup 2000/100k 步、batch 2）
与 S2A（flow-matching，训练时冻结 T2S、speaker encoder w2v-bert、style encoder CAMPPlus）。无 LoRA 入口。数据 TSV：
`lang / wav / norm_text / semantic_ids / ref_audio_paths（可多条）`。**配方不借**（全量、离散 token、超参量级不同）。

**最有用的是第三方基线（论文 Table 5，MiniMax-MLS-Test，已核原文）**——首次拿到 VoxCPM2 在目标语种上的独立评测：

| 语种 | VoxCPM2 WER | VoxCPM2 SIM | Confucius4 WER | Confucius4 SIM |
|---|---|---|---|---|
| id | 1.17 | 0.795 | 1.12 | 0.765 |
| th | 2.19 | 0.841 | 1.56 | 0.773 |
| vi | **4.19** | 0.793 | 1.61 | 0.751 |

⇒ **vi 是 VoxCPM2 基座最弱的目标语种**（WER 约为 id 的 3.6 倍），与上游 [#352](https://github.com/OpenBMB/VoxCPM/issues/352)（越南语停不下来）一致，
说明 vi 数据轮的提升空间最大、也最该先建 base 基线。VoxCPM2 的 SIM 三语均更高——与本项目「内容改善伴随音色代价」需兼顾一致。
（跨语言 th→zh / vi→zh 方向与生产 zh/en→目标相反，不引用。）

**数据**：多 ASR 交叉错误率 **<2.5%** 才收（比 CosyVoice3 的两两 WER<15% 严得多，比本项目伪标注计划 CER≤0.05 更严）；
每说话人至少 2 条、ref/target 取同人不同录音（本项目已是此规则）；按语种重采样（无公式）；每语种约 1000h **合成语音**补极短句（不借）。
⇒ 双 ASR 一致性门槛的业界取值从 2.5% 到 15% 不等，本项目 0.05 处于中间偏严，**不改**；首跑记录各门槛下的产出率再定。

**issues 的两条教训**（非模型、属文本前端）：
- [#1](https://github.com/netease-youdao/Confucius4-TTS/issues/1)「固定位置吞字、换参考仍在同处」——维护者确认是**分句**导致，修复后字数正确。
  与本项目「先排上下游再怪模型」一致：同一文本、跨参考/seed 稳定复现的漏字，先查 DIS 分句/TN。
- [#19](https://github.com/netease-youdao/Confucius4-TTS/issues/19)：`wetext` 英文 TN 把 "Sun" 扩成 "Sunday"。**本机复现（wetext 0.2.1）**：
  `Sun Li is here.`→`Sunday Li is here.`、`Wed the bride now.`→`Wednesday the bride now.`、`Sat Nam`→`Saturday Nam`。
  **VoxCPM 的英文 TN 用的正是 wetext**（`text_normalize.py:167`，生产 OmniVoice vendored 副本 `:369`、`:420`）。
  读代码推断的生产链路：DIS `VOXCPM_NORMALIZE` 默认 true（`backends/voxcpm.py:106`），仅 tl/fil 设 `spoken_text_normalized=True` 跳过服务端 TN
  （`:1110`）⇒ **en 目标会经 wetext，中文姓「孙」(Sun) 等专名会被念成 Sunday**。en 已于 11-17 升为目标语种，属生产风险；
  修复在 DIS/OmniVoice（如 en 也走客户端 TN 或关闭 wetext 星期缩写展开），本仓库不改生产，仅登记。未在线上实际调用验证。

## 5. 建议顺序（不打断现行主线）

1. 现行不变：r54 → vi/en 数据轮；新开发集、按句聚类、2 训练 seed。**更正**：ID 的 MMS-FA 片段已在 r49/r50 中（629 行），
   r54 的真实增量只有 **TH**。原切分脚本只认 a–z（`norm_word`），泰文全被跳过；已扩展 `scripts/fleurs_segment_20261103.py`
   （`--sources fleurs_th --out <新目录>`，pythainlp newmm 分词 + uroman 罗马化喂 MMS_FA，段文本取原泰文子串；TL/ID 默认行为不变，
   测试 `tests/test_fleurs_segment.py`）。泰语 MMS_FA 对齐质量未实测，首跑先小样本看对齐分与 Whisper CER 产出率。
   之后依次测 H（时长/文本比剔两端，4.5）、A（r32 vs r64）、B（时长上限）、G（MLP LoRA），各自单变量。
2. **零训练 p_stop 诊断——已完成（2026-11-23），否定「停止失准」假设**：`scripts/stop_prob_probe.py`（forward hook，不改 submodule）
   重放 r50 冻结短句全部 56 条严重输出（56/56 时长复现）+ 44 条同句干净对照：严重 55 条 clean_stop、1 条 never_rises；
   严重与对照的停止行为几乎一致（中位 6 步；停前最大 p_stop p90 .427 vs .438），仅 1 条撞长度上限。⇒ 按冻结规则判 `not_stop_timing`，
   **阈值/长度先验/最小长度 A/B 均不做**。短句严重坏例主要是内容错（错读/错词），停止头本身工作正常，
   这也解释了 r41（stop 权重 ×2）为何无效。局限：短句约 6 步，模式标签分辨率低，但严重/对照对比不依赖标签。
3. **LoCA/GROW 式 RL 预研**：全文已核（无代码、单 seed、无目标语种）；仅在 r54 后、有新开发集和明确奖励
   （内容 + 长度比 + 声纹 + 尾部多读惩罚）时立项，按小项目估算工程量。
4. 数据入库加「人工抽检估转写错误率」一步。

所有方向仍遵守：分语种、≥5 seed、按句聚类、一次性留出、母语未评如实标注、不自动切生产。
