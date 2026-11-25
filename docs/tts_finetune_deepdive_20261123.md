# 开源 TTS 微调方法深挖：东南亚语种社区微调、停止/幻觉对齐、官方微调配方（2026-11-23，2026-11-25 补充）

## 结论与边界

接续 [voicestudio_training_transfer_20261120.md](voicestudio_training_transfer_20261120.md) 与
[dots_tts_training_review_20261120.md](dots_tts_training_review_20261120.md)，不重复已覆盖的 MOSS / IndexTTS /
CosyVoice / Qwen3 / Ming / Confucius4 / k2-fsa OmniVoice / dots / LongCat / Chatterbox / Dia / CSM / OuteTTS。
本次为联网静态调研 + 本地源码核对，**未训练、未生成音频、未改生产**。
2026-11-25 增补 FireRedTTS3（§8）与 Microsoft VibeVoice（§9）：重点是语义表征、身份/语言条件、
结构化声学计划及长上下文的迁移边界，不改变前述实验排序，也不新增训练轮。

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

## 6. VoxCPM2 预训练语种分布 vs 本项目微调配比（2026-11-23，用户点名）

**预训练（[技术报告](https://arxiv.org/html/2606.06928v1) §3.5–3.6、[模型卡](https://huggingface.co/openbmb/VoxCPM2)，全文已读）**：
总量 2M+ 小时、中英为主；其余 28 语「roughly 1K to 50K hours each, depending on data availability and annotation quality」；
**未公开 id/vi/th/tl 各自小时数**。Stage 3 退火 SFT 采用「balanced language-level sampling ratio」（比例未公开）。

基座分语种质量（官方）：

| 语种 | MiniMax-MLS WER（VoxCPM2 / MiniMax） | MLS SIM | 内部 30 语基准 |
|---|---|---|---|
| id | 1.08 / 1.24 | 80.0 | WER 1.36 |
| vi | **3.31 / 0.88** | 80.6 | WER 1.56 |
| th | 2.96 / 2.70 | 84.0 | CER 0.94 |
| tl | 不覆盖 | — | WER 2.63（四语最高） |

第三方 Confucius4 Table 5 同测试集另得 id 1.17 / th 2.19 / **vi 4.19**（§4.6）。

**r50 微调数据实测（14.47h）**：id 6.62h **45.7%** / tl 4.76h 32.9% / th 2.13h **14.7%** / zh 0.59h 4.1% / en 0.37h 2.5%
（等权设计各 28.3%、回放 15%）；≤4s 条数 id 788 / tl 708 / th 378。

**结论：调配比（再平衡），不加总量。**
1. 即便按每语种最少 1K 小时算，r50 的 TH 2.13h 也只占基座 TH 数据的 ≤0.2%——LoRA 不是在教语言，而是把分布推向短句/配音域；
   加总量已被 r17（69h）、r53（25h）否定。
2. **TH 欠配（14.7% vs 28.3%）且短句最差**；r51（同类 TH 加量 2.13→3.37h）平手，说明「同类加量」不是杠杆。
   **在跑的 r54 正是「准确短段提 TH 份额」的检验**（TH 约 2.1→4.4h，≈26%，接近等权）。
3. **ID 超配（45.7%）**，其中 `gigaspeech2_id` 伪标注 1779 行；ID 已是最佳语种（开发集严重 13→0），暂不动；
   TH/vi 补齐后可做「ID 降到等权」单变量，检验是否释放容量。
4. **vi 是基座最大缺口**（MLS 3.31 vs MiniMax 0.88），vi/en 数据轮给 vi 完整目标份额（准确转写源优先）。
5. **回放 6.6% 低于设计 15%**；en 已升为目标语种却只有 2.5%，vi/en 轮需给 en 独立目标份额，不与回放混算。

## 7. 下一阶段：情绪一致性（2026-11-23，用户要求纳入）

**口径**：生产不用控制前缀（60/60 归档 `control_instruction` 为空），情绪来自**参考音**（源演员同句切片）。
因此「情绪一致性」= ① 输出情绪跟随参考（参考愤怒→输出愤怒）；② 同角色跨 cue 情绪稳定、不忽强忽弱。
与 timbre_v1 的音色一致性是两件事：timbre_v1 测的是说话人相似度，其「情绪 high/medium」臂用的是控制前缀，不是生产路径。

**风险假设（未测）**：r50 训练数据几乎全是朗读（FLEURS/CV/GigaSpeech2，表演档 0），LoRA 可能把输出**压平**，
即内容改善的同时削弱对参考情绪的跟随。本项目从未自动测过情绪，现有只有盲听表单里的 `emotion_fit_1_5` 人工字段。

**阶段 0：先测量（不训练）**
- 工具：emotion2vec+ large（[HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large)，约 300M，9 类：angry/disgusted/fearful/happy/
  neutral/other/sad/surprised/unknown，输出情绪表征可算相似度）；CV3-Eval 的情绪子集即用它算情绪准确率。
  许可为自定义「model-license」，仅作评测工具、不分发，用前核原文。官方未报告跨语种精度——
  th/tl/id/vi 上的分类可信度要先用 thai_ser（带人工情绪标签）抽样校准。
- 参考：UAT 生产参考（timbre_v1 的 8 条 + DIS `uat_replay` 其余可追溯参考，按 emotion2vec + 人工粗标情绪分层），
  补 thai_ser 各情绪的 th 参考（同语种对照）。不使用来源不明的爬取情绪集作参考。
- 目标文本：每语种中性文本（情绪只能来自参考）+ 情绪一致文本两组，短/中长度，生产口径（cfg 1.8、自适应步数、单次生成、≥5 seed）。
- 指标：参考↔输出的 emotion2vec 表征余弦与 top-1 一致率（主）；同参考跨句情绪表征离散度（一致性）；
  F0 范围/能量动态作辅助（不作门限）。统计按参考聚类 bootstrap。
- 闸门：r50 相对 base 的情绪跟随显著下降（参考簇 95% CI 全在 0 以下）⇒ 情绪一致性列入下一轮**验收门**（任一语种显著退化即不通过），
  并启动阶段 1；不显著 ⇒ 只把指标加进常规验收，不专门训练。
- 人工：情绪对错跨语种大体可感知，但按现行纪律「情绪」仍记未验证，需母语或至少用户声学盲听补充（是否加入用户盲听范围待用户确认）。

**阶段 1：训练杠杆（仅闸门触发时，单变量逐个）**
1. **同说话人 + 同情绪的 ref→target 配对**：教模型「照参考情绪说」，正对应生产形态。thai_ser（TH，41h、200 名演员、5 情绪、
   有说话人与情绪标签）可直接构造；不同录音、同情绪、同人，不伪造身份。注意与 IndexTTS2 的「中性 ref→情绪 target」相反——
   那是为解耦音色与情绪，我们要的是情绪跟随参考。
2. 表演语料小比例回注：thai_ser impro（TH）；Basis global-emotions（id/th/fil，gated，待申请）是 id/tl 唯一现货。
3. 降低朗读占比（与配比再平衡 §6 一起设计，避免同时改两件事）。
4. 推理侧不改：不启用控制前缀（生产不用）。

## 8. FireRedTTS3：语义表征和条件路径值得研究，不能直接变成下一轮 LoRA（2026-11-25）

来源：[技术报告](https://arxiv.org/html/2608.17492v2)、[官方仓库](https://github.com/FireRedTeam/FireRedTTS3)、
[Base 推理源码](https://github.com/FireRedTeam/FireRedTTS3/blob/main/fireredtts3/llm/fireredtts3_base.py)。
公开树是推理实现；[微调入口询问 #13](https://github.com/FireRedTeam/FireRedTTS3/issues/13)仍开放且零答复，
**没有可直接复用的 FireRedTTS3 官方微调脚本或完整数据配方**。以下为方法迁移分析，未运行其权重或开训练。

### 8.1 先还原其训练结构，避免把“1.7B backbone”误当整个方法

- **RedAE**：24kHz 波形先形成 50Hz 表征，再降到 25Hz continuous latent；训练时以冻结的
  FireRedAudio Audio Encoder 为语义教师，该教师经过 ASR、说话人验证等理解任务训练，
  用 MSE 让 RedAE latent 对齐教师特征，同时保留 GAN、Mel 重建与 feature-matching 损失。
  教师在 tokenizer 训练后移除，不参与 TTS 推理。RedAE 使用 **50 万小时音频、32×H800、55 万步**。
- **Base**：Qwen3-1.7B-Base 只是 backbone，另有 Aggregator、DiT、RedAE 和 CAM++。
  CAM++ 说话人向量同时注入 LLM 与 DiT，目标文本前加语言 tag，参考音仍经 RedAE 作为 prompt；
  训练是 flow loss＋stop loss。第一阶段用 **260 万小时中英、17 万步**，第二阶段再用
  **56 万小时、24 语＋21 方言**扩展。论文未给 Base 第二阶段步数和分语种数据量。
- **Instruct**：没有 Base 的显式 speaker embedding 与语言 tag，不能混写两种克隆路径。
  音色设计/编辑先把自由指令展开成文本计划；音色设计计划含 12 个声学属性，
  再用 33 万小时音色设计/编辑数据训练 4 万步。论文未公开完整 12 字段清单和可复现数据构造比例。
- CFG 训练时 backbone 条件丢弃率 0.1；使用 speaker 条件时其丢弃率也是 0.1，clean historical latents 始终保留。
  这与本项目的 `training_cfg_rate=0.1`、plain/带 ref 混合有理念重合，但条件张量和损失实现不同，
  **不能拿相同数值证明等价，也不因此把候选 E 提前**。
- 论文的 MiniMax/Seed 分数是短句自动指标，没有按生成长度报告音色/韵律漂移曲线，
  也没有单独消融语义蒸馏。[issue #10](https://github.com/FireRedTeam/FireRedTTS3/issues/10)正询问这两点，
  截至本次核对无官方答复，不能把平均分提升直接归因为“长程误差累积已解决”。

### 8.2 对当前项目真正可借的内容

| 方法 | 可借鉴部分 | 对 voxcpm_finetune 的处置 |
|---|---|---|
| RedAE 的冻结语义教师 | 让连续声学表征同时受语音内容与说话人信息约束，而非只优化波形重建 | **仅作独立 tokenizer 长期研究**。若未来自研/重训表征，可在同数据、同下游生成器下做有无语义损失消融；当前 LoRA 不动 AudioVAE，不能靠新增一条 loss 复刻 |
| CAM++ 同时注入 LLM/DiT | 给身份一条显式、可审计的条件路径，而非只依赖整段参考 latent | **仅作结构假设**，可解释其相似度为何值得研究；VoxCPM2 需新投影层、训练与推理链，非 YAML/LoRA 开关。不得以 `enable_proj=true` 冒充等价实现，r40 旧失败也不能被当成新结构证据 |
| 目标语言 tag＋多语续训 | 语言元数据须在训练、缓存和推理一致保留；多语能力后接专门阶段 | 本项目继续分语种数据/验收与回放，**不给无槽位的 VoxCPM2 临时加 token/自然语言前缀**。两阶段大规模训练不重演，r51/r54 已否定 TH 同类加量/短段翻倍是杠杆 |
| CFG/speaker dropout 与 clean history | 保留无条件能力、同时保护连续历史，是条件训练的合理约束 | 现行 cfg 丢弃、可信 plain/ref 混合与 continuation 纪律已经覆盖；不复制数值，不随机丢生产参考，不新增一轮相同 dropout 搜索 |
| flow＋stop 联合目标 | 内容与停止是不同学习目标，停止头要独立诊断 | 只支持现有“分开测量”的原则。r41 stop 权重 ×2 已失败，2026-11-23 p_stop 探针又否定停止失准，**不借 stop 权重、min 6 patches 或 max 400**；其时间轴、标签和 VoxCPM2 不同 |
| Instruct 先形成结构化声学计划 | 把自由风格描述变为可审计字段，比从 F0/能量猜表演更可靠 | 仅作情绪阶段 0/1 的标注 schema 参考；不虚构未公开 12 字段，不用自动声学统计伪造标签。生产情绪来自参考且不用控制前缀，**不上线额外 planner/LLM** |
| Base 的参考音＋参考文本、语言提示 | 应分别测参考文本有无、同语/跨语参考，而不是把 MLS 同语分数外推到生产 | 若未来做引擎 A/B，纳入现有 zh/en/fil ref × th/tl/id/vi target 矩阵，固定 ref、正文、seed、重试预算，并与当前 r50 比较；这是**评测设计，不是给 VoxCPM 新增 ASR 或训练数据** |

**落地结论：近期没有因 FireRedTTS3 新增的 LoRA 训练臂。**最有价值的新假设是底层语义表征和显式身份条件，
都属于自研底座/结构研究，不能插入 r50 的普通数据轮。当前先完成已中断的 r55 开发集、vi/en 备料、
情绪阶段 0 与母语评审；FireRedTTS3 只作为 th/vi/id 的局部外部对照，不改变现有计划。

### 8.3 明确不借的做法

1. **不换用其 RedAE 或超参。**RedAE 为 24kHz、25Hz latent；VoxCPM2 是 16kHz 编码器输入、FSQ/AudioVAE 与另一套 patch 条件，
   权重和表示不能直接互换。其 CFG 2.0、10 steps、停止阈值也不是当前生产 cfg 1.8/自适应步数的替代配方。
2. **不把 speaker embedding 当身份真值或情绪解耦证明。**CAM++ 仍从同一参考音提取，RedAE prompt 也保留参考风格；
   它没有自动解决短/噪参考、跨语言口音或身份/情绪解耦。身份配对继续遵守已验证同人、不同 origin_audio 与 split 隔离。
3. **不照搬预训练规模和课程。**50 万小时 tokenizer＋260 万/56 万小时 TTS 是另一量级的基础训练，
   与当前 14.47h 的领域 LoRA 不构成“加数据就会同样提升”的证据，也不能覆盖本项目多次加量无收益的实测。
4. **不使用其输出补 VoxCPM 语料。**项目合成语料红线不变；且其代码/HF 虽标 Apache-2.0，
   克隆 Usage Disclaimer 又限学术研究用途，[许可询问 #12](https://github.com/FireRedTeam/FireRedTTS3/issues/12)无官方答复，
   书面澄清前不用于生产或训练。
5. **不把 Instruct 编辑当作现有配音修复捷径。**语义编辑可以局部保留未编辑区，声学编辑则改变整段属性，
   两者都是另一模型和独立输入输出契约，不能自动保证字幕窗、角色身份或正文不变；
   若将来单独研究，编辑后仍须重做文本、身份和声学 QC，不能混入本轮 LoRA。
6. **不把公开自动分数当微调收益。**MiniMax-MLS 是同语参考且有参考转写，FireRed 表中对手也是 VoxCPM2 基座而非 r50；
   本项目继续同参考、跨语言正文、≥5 seed、按句聚类、母语盲听与任一语种退化即整轮不通过。

## 9. VibeVoice：冻结表征是有效佐证，长播客设计不是短 cue 配方（2026-11-25）

来源：[技术报告](https://arxiv.org/html/2508.19205)、[长文 1.5B 模型卡](https://huggingface.co/microsoft/VibeVoice-1.5B)、
[Realtime 模型卡与文档](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)、[ASR 微调入口](https://github.com/microsoft/VibeVoice/blob/main/finetuning-asr/README.md)。
早期长文 TTS 代码已移除，当前公开的官方微调脚本属于 **ASR**，不是 TTS；ASR 的 LoRA rank、学习率和长音频标签格式
不能直接搬到 VoxCPM2 TTS。其 toy dataset 也明确是 TTS 合成音频，仅作演示，不构成可训练数据或配方证据。

### 9.1 公开训练结构与版本边界

- 长文版有独立的 acoustic/semantic 两类连续 tokenizer，均工作在 **7.5Hz**，把 24kHz 音频压缩 3200 倍。
  Semantic tokenizer 以 ASR proxy task 训练，文本 decoder 在 tokenizer 预训练后移除；生成目标仍是 acoustic latent。
- 两类 tokenizer 预训练后都冻结，**只训练 LLM 与 diffusion head**。长文上下文课程为 4k→16k→32k→64k，
  目标是 90 分钟多人对话；Realtime 则移除 semantic tokenizer、课程只到 8k，并以预嵌入音色换取低延迟和滥用防护。
  不能把长文版的任意声音 prompt、多人对话和 Realtime 的预设声音合并成同一能力。
- 其 diffusion head 是 DDPM 噪声预测＋DPM-Solver，论文给出 guidance 1.3、10 steps；
  VoxCPM2 是另一套 flow-matching/CFM 与 patch 生成路径，不能直接复制这些推理数值。
- 官方明确长文版跨语言迁移是未专门优化的 emergent ability，可能不稳定；Realtime 对三个词以内也提示稳定性下降。
  因而没有证据把 7.5Hz 或长上下文课程解释成短 cue 质量/停止稳定性的改进。

### 9.2 对当前项目的可迁移结论

| 方法/现象 | 可借鉴或应吸取的教训 | 对 voxcpm_finetune 的处置 |
|---|---|---|
| 冻结 acoustic/semantic tokenizer，只训 LLM＋diffusion | 预训练声学表征不一定要随领域数据微调，避免破坏通用重建 | **支持现行冻结 AudioVAE、只适配 LM/DiT 的路线**，是结构佐证而非新增实验；也不能据此证明 LoRA 一定优于所有全量方案 |
| Semantic tokenizer 的 ASR proxy task | 独立语义表征可帮助长序列内容建模，和 FireRed 的语义教师目标相呼应 | 仅作底层表征长期研究；VibeVoice 是额外 semantic 输入分支，FireRed 是把语义注入同一 RedAE，二者不等价，不能直接加到当前 LoRA |
| 7.5Hz＋逐步扩到 64K | 低帧率降低长上下文序列成本，课程解决的是长播客可扩展性 | **不降帧、不做 90 分钟训练、不改上下文长度**；当前主要失败是短 cue 出分布，Realtime 超短输入不稳反而要求保留短句回归 |
| 角色 ID＋声音特征＋分角色文本的混合输入 | 多角色任务需要显式角色—声音—正文绑定 | 若未来做长对话，可借鉴其数据组织原则；当前单 cue/reference-only 不添加 `Speaker 1` 等伪前缀，四说话人上限也不替代 DIS 全局角色映射 |
| 官方刻意不对训练数据降噪，接受随机 BGM | 不干净目标/参考会把配乐、环境或事件发声带给输出，且生成时不可控 | **作为反面教训**：继续音乐排除、参考/目标分别检查和派生音损伤审计，不为了涌现唱歌/BGM放松干净语料门 |
| 不做 TN，依赖 LLM 自行处理复杂文本 | 官方同时承认中文发音和特殊符号 corner cases | 不取消 `language_config`/分语种 TN 与数字、专名、Taglish 回归；训练/推理文本形态继续一致，不借“大模型自己会读”的假设 |
| 跨语言靠重复采样碰好结果 | 采样可能偶尔成功，但不证明跨语言条件训练有效 | 不增加默认生成预算；任何 A/B 固定相同预算并报告逐 seed，不能把多次挑选的最好一次当模型收益 |
| ASR LoRA 脚本与合成 toy 数据 | ASR 训练可关注长音频角色段、hotwords 和时间戳 | 只属独立 ASR 课题；**不复制其 ASR LoRA 超参，不使用合成 toy/输出补 VoxCPM 语料**，现行合成语料红线不变 |

### 9.3 最终行动

- **没有因 VibeVoice 新增的 LoRA 训练臂。**冻结表征继续；语义分支、7.5Hz、角色混合输入都不是当前短 cue 的直接修复。
- 若未来另立长播客/长对话项目，先证明确有长上下文需求，再研究其课程与多角色打包；
  不能用八组长对话的主观成绩或论文 tokenizer 重建分数替代 th/tl/id/vi/en 的生产验收。
- 当前仍执行既有数据主线、情绪阶段 0、分语种多 seed 与母语盲听；不切换引擎、不新增 ASR/TTS 调用、不启动 GPU。
