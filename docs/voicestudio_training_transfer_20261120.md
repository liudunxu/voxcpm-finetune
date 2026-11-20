# VoiceStudio 相关 TTS 训练方法对 VoxCPM2 的可迁移性（2026-11-20）

## 结论与本次边界

**有值得借鉴的内容，但主要是数据构造、条件训练和验收方法，不是直接搬训练脚本或超参。**
当前最佳仍是 r50；生产权重、五目标语种备料主线和普通 LoRA 配方均不变。本文只完成官方论文、公开源码与本项目的静态对照，**未训练、未生成音频、未启动 GPU，也未修改现行实验计划**。

dots.tts 的 SOAR、MeanFlow 和 EOS 详细分析已在
[dots_tts_training_review_20261120.md](dots_tts_training_review_20261120.md)，本文不重复推导，重点补齐
MOSS、IndexTTS、CosyVoice、Qwen3-TTS、Ming、Confucius4 与 k2-fsa/OmniVoice 等 VoiceStudio 相关引擎。

对当前项目最值得保留的新增结论：

1. **MOSS 的同说话人多时间窗参考构造与真人短词/短句补分布**，可直接映射到当前短 cue 和 ref 缺口；
   但声纹相似度只能在已验证身份内部做排序，不能拿来证明同人。
2. **MOSS 的纯文本扰动增强**是一个低成本、可与真人音频分开的候选臂；它不是合成语音补量，
   也不能通过删句末标点来规避 EOS 风险。
3. **k2-fsa/OmniVoice 的“降质 prompt → 干净 target”与本项目已有 ref 鲁棒性方向一致**，
   属于独立论文佐证，不是另开一套退化类型或立即开训的理由。
4. **Ming 的独立 `stop_head` 训练入口**说明停止头可以作为单独训练面管理，为 dots EOS 解耦假设补充了工程参照；
   其停止权重、标签屏蔽位置和模型模板不能直接复制。
5. **IndexTTS / CosyVoice / Qwen 的 RL、DPO、GSPO 只适合作为后期研究储备**。
   CosyVoice 已公开报告声纹、情绪和内容指标之间的奖励投机，不能用单一 ASR 奖励替代当前分语种验收。

## 一览：哪些能借，哪些不能

| 来源与方法 | 对 VoxCPM2 的可迁移性 | 当前处置 |
|---|---|---|
| MOSS：真人短词/短句补充 | 与已证实的短 cue 出分布问题直接对应 | 继续现行准确转写、MMS-FA 短片段主线 |
| MOSS：同说话人随机时间窗参考对 | 可嵌入现有 `pair_references`，改善参考窗多样性 | 作为将来的单变量数据臂，先核身份与 split |
| MOSS：标点、空白等纯文本扰动 | 不改音频、不使用合成语料，可测试输入鲁棒性 | 仅在有真实失败形态后做小比例单变量实验 |
| k2-fsa/OmniVoice：降质 prompt、干净 target | 与现有 `ref_robustness` 的结构相同 | 复用已冻结阶段 0；不擅加噪声/混响档 |
| IndexTTS2：同人参考/目标、情绪阶段训练 | 可借“中性参考→情绪目标”的数据关系 | 与 timbre_v1 既有候选一致，须等可信表演语料 |
| Ming：LoRA 之外独立训练 stop/head 模块 | 给 EOS 解耦实验提供工程参照 | 与 dots EOS 假设合并设计，不另起重复训练轮 |
| IndexTTS2.5 / CosyVoice：GRPO / DiffRO | 有后训练思路，但模型表示、奖励和成本均不同 | 暂缓；先解决数据与可校准验收 |
| Qwen3-TTS：高质量 CPT、人类偏好 DPO、GSPO | 强化“干净核心优先”和偏好数据价值 | 不在当前连续 latent 架构上直接套用 |
| MOSS/Index：token 数时长控制 | VoxCPM2 没有对应条件通道和推理契约 | 不属于当前 LoRA；不能靠改 YAML 获得 |
| Index：语种边界/token embedding | 需要新语种条件或逐 token 标音前端 | 不改当前无 language token 的 VoxCPM2 架构 |
| MOSS：RVQ channel loss；CosyVoice/Qwen：codec 训练 | VoxCPM2 使用连续 AudioVAE latent，监督形式不同 | 不移植，不借其数值调当前 loss |
| dots：SOAR / MeanFlow / EOS detach | 原理可研究，已有专篇 | 按专篇顺序，不因本文提前或扩预算 |

## 1. MOSS：最值得借的是数据构造，不是 8B 训练参数

### 1.1 真人短词和短句覆盖

[MOSS-TTS 技术报告](https://arxiv.org/abs/2603.18090) §5.1.3 明确把 dictionary-style
单字、单词录音作为独立补充，指出自然网络语料缺少这些输入会导致超短输入不可靠。
这与本项目「长朗读训练在短 cue 上出分布」的历史实测方向一致。

可迁移边界：

- 当前仍优先使用**准确转写的真人 3–6s 片段**及 MMS-FA 停顿切分，这已在 r49/r50 路径上产生有效结果。
- MOSS 的“单字/单词”不等于可以直接给现有 VoxCPM2 塞入亚秒音频；本项目加工入口仍有 3s 下限。
  真要覆盖更短真人样本，必须另行验证 packer、patch、stop 监督和长度门，不能删除守卫后静默训练。
- Confucius4 也使用额外数据覆盖极短输入，但其中包含合成语音；本项目只接受其“覆盖缺口”的问题定义，
  **不接受用合成音频补量的做法**。

### 1.2 同说话人、多时间窗参考构造

MOSS §5.1.3 对同一说话人的其他片段随机裁出多个参考窗，再用 WavLM 相似度选择候选。
它解决的是“参考长度与位置变化下仍能提取稳定音色”，不是身份发现。

本项目可借用的最小版本：

- 先完成现有身份/session/原音频隔离，只在 `speaker_verified=true` 的同一说话人内部选 ref；
- 继续使用不同 `origin_audio`，不把同条录音裁成 target/ref 伪装成独立条件；
- 在当前允许的 3–10s 范围内改变参考起点、长度和录音，不把 MOSS 的 30s 上限搬过来；
- 若要借鉴声纹排序，只能将它作为**已验证同人的候选质量排序**，并保留随机窗臂作对照。
  不得像通用聚类那样把高分当身份证据；本项目已有“名义异人也会高分”的反例。
- 还需防止“只挑与 target 声学条件最像的 ref”降低跨录音、跨语种鲁棒性；生产是中文/英文参考到目标语，
  不能只用同语种、同环境近邻训练。

这与当前 `src/voxft/data/pipeline.py:796-822` 的随机参考选择相比，新增价值是**参考窗覆盖策略**，
不是更换配对身份规则。只有形成明确假设后才做单变量数据实验；不同时调整 ref 比例、语种配比和普通超参。

### 1.3 纯文本扰动：可以借，但不能伤正常标点和 EOS

MOSS 对已验证音频—文本对只做文本侧扰动：连续/混合标点、异常空白、标点缺失和少量脏字符，音频不变。
这属于输入鲁棒性增强，不属于合成语音。

对 VoxCPM2 可形成的假设是：少量这类样本可提高对真实异常输入的容忍度，同时正常文本仍由原干净核心保护。
但必须遵守：

- 先存在真实失败 case 再立项，不为“增强鲁棒性”泛泛扩数据；
- 保留原始准确文本、变换类型、随机种子和来源，clean/augmented 可分别审计；
- 不把句末标点随机删除当作修复句末专名重复；这会同时扰动韵律和停止条件；
- 不在推理侧改写用户文本，也不把训练增强解释成生产 normalizer 修复；
- 验收必须另含正常标点、短 cue、句末专名和多读/少读，而不只看平均 CER。

### 1.4 原音质量评估与模板一致性比新增阈值更重要

MOSS 在**降噪前原音**上计算 DNSMOS / AudioBox PQ，以免增强器改变质量估计。
这与本项目“分离派生音可能受损、保留独立合格原音”的实测一致。可保留的原则是：

- 原音与派生音分别留存检查；不能因增强后指标更高就覆盖原音结果；
- 不直接复制 DNSMOS>2.8、PQ>6.5 或 Confucius4 的跨 ASR<2.5% 为本项目硬门，
  这些模型、语言和音频分布不同；现行双 ASR CER≤0.05 计划仍须先测产出率与误拒；
- 对 code-switch 和近亲语种，不把“ASR 语种=文本语种”当独立口音证明。

MOSS 的[公开 SFT 文档](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md)
还给出两个有用警示：

- 模板/processor/训练打包与推理不一致时，**训练 loss 可以正常而推理仍崩成乱码**。
  本项目应继续把 reference-only 的 `[103 ref 104][text][101 target 102]`、loss mask、stop 标签和实际推理路径一起验收。
- 它支持预编码 target/ref audio codes、checkpoint 携带推理配置和代码。
  本项目已有严格 LoRA 配置与 merge 检查；编码缓存只属于提速工程，当前训练远短于评测，不应优先改写数据链。

MOSS 的 RVQ channel loss、8B 优化器设置、四阶段大规模预训练和显式 `tokens` 时长通道都不直接适用于当前 LoRA。
尤其其 P2 强化克隆、P3 恢复平衡、P4 扩长上下文是**百万小时级预训练课程**，不是对 r50 重排 epoch 的依据。

## 2. Confucius4 / IndexTTS：跨语言与情绪数据关系可参考

### 2.1 Confucius4：同人不同录音与分语种平衡，现有纪律已覆盖

[Confucius4-TTS 报告](https://arxiv.org/html/2608.11650)采用同一说话人的不同录音作为 target/ref，
联合训练说话人编码器，再冻结 T2S 训练声学模块；同时进行多系统 ASR 过滤和按语种重采样。

对本项目有效的部分：

- 同人不同录音、参考条件训练、分语种采样都与现有路线一致，支持继续补可信 ref，而非伪造身份；
- 其参考克隆与 continuation 是两种不同质量路径，不能把 continuation 分数当本服务 reference-only 证据；
- 公开源码中的长参考随机裁窗可作为 MOSS 参考窗策略的另一处实现参照。

不能移植的部分：

- 可学习 speaker encoder、语义 codec、S2A prompt 及语言指令不是现有 VoxCPM2 的 LoRA 面；
- 论文使用每语种约 1000h 合成语音补短尾，本项目合成语料红线不变；
- 语种均衡不意味着复制新的重采样公式，现行五语种配比仍须单独冻结并记录实际时长份额；
- [T2S 数据集实现](https://github.com/netease-youdao/Confucius4-TTS/blob/main/confuciustts/dataset/t2s_dataset.py)
  对任意单条异常会随机换样继续，这不是本项目应借鉴的容错方式；现有原则是保留失败并显式中止/隔离。

### 2.2 IndexTTS2：情绪/身份解耦需要数据关系，不只是标签

[IndexTTS2 报告](https://arxiv.org/html/2506.21619)的做法包括：

- 同一说话人至少两条录音，分作 prompt 与 target；
- 先用全量数据建立基础能力，再用高质量情绪数据训练情绪条件，最后冻结条件器做全量稳健化；
- 用 GRL 和说话人分类器让情绪 embedding 去掉说话人信息；
- 用独立情绪参考和自然语言情绪控制降低对目标语情绪数据的需求。

VoxCPM2 没有 IndexTTS2 的独立情绪 audio channel、speaker perceiver 或 GRL，因此不能直接复制其损失。
可借鉴的是更朴素的数据关系：**同一已验证说话人的中性参考 → 情绪目标**，并用 plain/control 与音色一致性共同验收。
这正是 timbre_v1 已列的候选之一，不是新增即时任务。Basis 等来源未获访问、文本与 steer 未核实前，不能伪造这种配对。

IndexTTS2 的随机变速增强只可作为低优先级研究项：它可能扩充语速覆盖，但不等于可强制满足任意字幕窗，
还可能改变自然度和音色。译文物理不可行仍由 DIS retranslation 解决，不开“变速数据→精确时长”的隐含承诺。

### 2.3 IndexTTS2.5：多语对照方法值得学，语种槽位不值得硬搬

[IndexTTS2.5 报告](https://arxiv.org/html/2601.03888v5)在相同数据、优化器和 100K-step 预算下比较了
语种边界、逐 token 语种 embedding、自然语言语种指令三种策略。值得借鉴的是**同条件拆解语言建假设**，而不是某个赢家数值。

- 当前 VoxCPM2 已支持目标文本 tokenizer，且无现成 language token；不因这篇论文临时增加语种 embedding。
- TH 句内外文和 TL Taglish 应先由真实文本、TN/读音表和母语评审定位；逐 token 语种标注需要额外前端，
  不是普通 LoRA 的免费能力。
- 数据管线中的“只有检测到伴奏才做 Demucs 分离”与本项目派生损伤经验一致；不要恢复全量无差别分离。
  其“短段合并到最长 25s”服务长文目标，与当前短 cue 缺口相反，不能照搬。

## 3. k2-fsa/OmniVoice：参考降质和任务专属模块可借，许可仍不放开

> 这里的 OmniVoice 是 k2-fsa 的 600+ 语模型，不是生产服务同名仓库。

[论文](https://arxiv.org/html/2604.00688)和[训练/LoRA 文档](https://github.com/k2-fsa/OmniVoice/blob/main/docs/lora_finetuning.md)
提供两类参考：

1. **prompt denoising**：给参考注入噪声/混响，target 仍保持干净，让模型学习从差参考保留身份而不复制环境。
   这与本项目 `docs/ref_robustness_plan_20261012.md` 的“降质 ref → 干净 target”完全同构。
   当前已冻结 profile 只有 babble/lowpass/combo，且明确排除背景噪声和混响；外部论文支持该方向，
   **不自动改写冻结矩阵**。先按现有阶段门判断基线是否有可测退化，再决定是否另立扩展计划。
2. **LoRA + 任务专属模块全量训练**：其 PEFT LoRA 除 attention/MLP 外还保留 audio embeddings/heads 可训练。
   对 VoxCPM2 的启发是“适配器与任务头可分开管理”，可辅助设计 stop head 实验；
   但它是离散多 codebook 的非自回归模型，audio head 与 VoxCPM 的 stop head 并不等价。

其 full-codebook random masking 依赖离散 NAR 结构，不能移植到 VoxCPM2。
语种 repetition 的 `beta=0.8` 也是其 581k 小时、600+ 语种的选择；当前项目已有按有效时长的显式配额，
不复制该数值，不靠超额重复解决少数语种真实音频不足。模型权重 CC-BY-NC 和合成/数据纪律不因方法可借鉴而改变。

## 4. RL / 偏好优化：保留为后训练方向，当前不开

### 4.1 IndexTTS2.5 GRPO

IndexTTS2.5 §3.3 对每个输入采样 4 个候选，用冻结 ASR 的 WER 和说话人验证相似度构成奖励，
对 text-to-semantic 模块做 GRPO。其公开结果证明的是其五语种模型上的收益，不含本项目东南亚语种，也不是 r50 对照。

若将来研究 VoxCPM 后训练，至少要增加当前真正关心的**多读、少读、时长失控和身份回退**惩罚，
并按语种校准 ASR；不能只优化 WER+SIM。训练时的多候选成本也必须单独计量，不能把训练 rollout 与生产重试混为一谈。

### 4.2 CosyVoice：奖励投机是公开实证，不是假设

[CosyVoice3 报告](https://arxiv.org/html/2505.17589)的 DiffRO 用 Token2Text、Gumbel-Softmax 和 token-level KL
直接优化离散语音 token。VoxCPM2 是连续 AudioVAE latent + DiT，不存在可直接替换的 Token2Text 离散目标。

更重要的是其 §5.4 明确报告：

- RL 改善内容指标时，部分声纹指标略降；
- 加入说话人奖励可能反过来提高 WER；
- 加入情绪奖励可能损害发音；
- 难例、 rare words、绕口令和重复词上的改善更弱。

这与本项目 r50 已有“内容改善伴随 ERes2NetV2 音色代价”的现实风险一致。
所以多奖励并非天然解决问题，任何后训练都必须保留固定 r50 基线、分语种无语种退化和母语盲听。

[CosyVoice2 公开 GRPO 配方](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/grpo/cosyvoice2)
是另一条实现：语音 token 经声学模型解码后用 SenseVoice 计算拼音级奖励。
它不是 CosyVoice3 DiffRO 的同一算法，也只提供中文结果，不能外推到 th/tl/id/vi/en。

### 4.3 Qwen3-TTS：高质量 CPT 与人类偏好值得参考，DPO 不是现成开关

[Qwen3-TTS 报告](https://arxiv.org/html/2601.15621v1)采用通用预训练→高质量数据 CPT→长上下文扩展，
后训练再做人类偏好 DPO、规则奖励 GSPO 和轻量说话人微调。

对当前项目有支持作用的是：

- 先保住准确、干净核心，再谈偏好或规则优化；高质量 CPT 的结论与 r53 的“脏增量有害”方向一致；
- 若长期积累可靠的同输入人工胜负对，人类偏好数据比单一声纹/ASR 自动奖励更有价值；
- 但 VoxCPM2 连续 latent + 扩散生成的序列概率不等同 Qwen 离散 token 概率，DPO/GSPO 不能直接对现有 LoRA 开关启用。

当前不建设 RL/DPO 训练链，不用合成偏好伪造人工反馈，也不把历史“A/B 差不多”改写成有方向的偏好对。

## 5. Ming + dots：把停止头实验设计得更具体

[Ming 官方 SFT](https://github.com/inclusionAI/Ming-3.0-tts/tree/main/sft)同时支持 full/LoRA，
默认允许单独训练 `linear_proj_audio`、`flowloss`、`stop_head`、`spk_head`，并冻结音频 codec。
其[训练实现](https://github.com/inclusionAI/Ming-3.0-tts/blob/main/sft/model.py)还分别记录 flow/stop loss。

对本项目的价值是证明“backbone 适配器之外，停止头可作为独立训练面”是工程上可行的。
VoxCPM2 当前实际结构仍是：

- `stop_proj + stop_head` 从未 detach 的 `lm_hidden` 计算停止 logits；
- 普通 r50 LoRA 不训练这些停止模块；
- reference-only packer 只在最后目标位置标停止正例，target 之外不算损失。

因此可将 dots 的 EOS detach 假设和 Ming 的模块选择合并为一个**后续独立小实验**：

- 冻结 AudioVAE 与声学 DiT，比较只解冻 `stop_proj/stop_head`、以及 detach 后训练停止头的差异；
- 使用准确文本、完整边界、短句和句末专名案例，保留正常长句与 plain/reference-only 回归；
- 首轮不改变停止标签定义；Ming 的末端三个位置 ignore、0.01 stop 权重等是其自身设计，
  若未来要改标签或权重，必须另设单变量，不能混进首个实验；
- 直接验收尾部多读、提前停止、正文完整性、多 seed 稳定性及音色回退，不以 stop loss 下降晋级。

Ming 的 `speaker_dropout=0.5` 是其全局说话人向量条件 dropout；VoxCPM2 使用音频前缀且已有 plain/带 ref 混合，
不能复制数值或声称同义。其 `min_duration=1.0`、LoRA rank、学习率等同样不是 VoxCPM 的推荐值。

SOAR、MeanFlow 和 dots 的 detach 证据边界继续以专篇为准：SOAR 先测 off-trajectory 误差，
MeanFlow 只做质量稳定后的独立加速项目；三者不能合并成一轮“全面修复”。

## 6. 没有新增可复现训练方法的条目

| 模型/引擎 | 本次核到的边界 | 对 voxcpm_finetune 的处置 |
|---|---|---|
| Chatterbox V3 / Turbo / Nano | 官方公开树以推理代码为主，未找到官方训练入口；单语种包和 10→1 步蒸馏只有结果描述，且项目已停止维护 | 不从第三方补丁反推官方配方；只保留“单语种精调/低步数解码”研究方向 |
| Dia-1.6B | 官方公开树未找到训练入口，模型卡只有推理与克隆示例；英文对话和非语言标签不构成多语训练配方 | 不借鉴超参；非语言标签须另有真实、准确标注才可研究 |
| CSM-1B | HF 卡有通用 Trainer 示例，但明确非英语能力有限，没有目标语种的完整训练数据配方 | 不新增路线 |
| OuteTTS | 模型卡列出 Llama 训练参数与 10K 微调样例，但新增组件为 CC-BY-NC-SA，且无目标东南亚语种覆盖 | 不搬训练参数，不使用其输出补量 |
| MOSS-TTS-Nano | 有完整 SFT 入口，但基本沿用 MOSS 的编码、teacher forcing 与字段打包，没有新的目标语种证据 | 只作同族工程参考，不重复建文档或实验 |
| Breeze / 其他预设音或运行时封装 | 没有可改变 VoxCPM 训练的新公开方法；原许可和语种限制不变 | 不进入训练计划 |

## 7. 建议顺序：新增方法不得打断已确认主线

1. **当前不变**：vi/en 备料、五语种新配比、准确转写、MMS-FA 短片段、Basis 申请与母语评审优先。
2. **数据候选仍须逐个单变量**：
   - 已有主线的准确短句继续；
   - 有可信同人覆盖后，再评估多时间窗/跨录音参考；
   - 有真实异常输入证据后，再评估纯文本扰动；
   - 有可信中性/情绪同人录音后，再评估中性 ref→情绪 target。
   每项都不得同时改变 r50 普通超参或其他数据维度。
3. **ref 鲁棒性沿用既有阶段门**：先证明降质参考造成可测退化，再决定是否训练；不因 k2-fsa 论文直接扩展 profile。
4. **结构研究保持独立**：EOS 停止头优先围绕已冻结的句末专名/短 cue 风险；SOAR 先做无训练诊断；
   MeanFlow 仅用于质量稳定后的降步数。
5. **RL/DPO/GSPO 暂缓**：只有可靠偏好/奖励、足够独立开发集和完整成本预算均具备时再立项。
6. 所有训练臂均沿用现行生产口径、分语种、≥5 seed、按句聚类和一次性留出要求；
   已反复使用的开发集不能继续冒充最终独立验证，母语未评项仍保持未知。

**本文的最终建议是“补充数据构造与实验假设”，不是重启超参搜索，更不是马上开一轮 RL 或全量训练。**

## 主要官方来源

- [MOSS-TTS 技术报告](https://arxiv.org/abs/2603.18090)；[MossTTSDelay SFT](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/README.md)；[训练打包实现](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/finetuning/dataset.py)
- [IndexTTS2 报告](https://arxiv.org/html/2506.21619)；[IndexTTS2.5 报告](https://arxiv.org/html/2601.03888v5)
- [CosyVoice3 报告](https://arxiv.org/html/2505.17589)；[CosyVoice2 GRPO 配方](https://github.com/FunAudioLLM/CosyVoice/tree/main/examples/grpo/cosyvoice2)
- [Qwen3-TTS 报告](https://arxiv.org/html/2601.15621v1)
- [Ming-3.0-TTS SFT](https://github.com/inclusionAI/Ming-3.0-tts/tree/main/sft)；[LoRA 配置](https://github.com/inclusionAI/Ming-3.0-tts/blob/main/sft/configs/dense_lora.yaml)
- [Confucius4-TTS 报告](https://arxiv.org/html/2608.11650)；[官方训练/数据代码](https://github.com/netease-youdao/Confucius4-TTS)
- [k2-fsa/OmniVoice 报告](https://arxiv.org/html/2604.00688)；[训练与微调入口](https://github.com/k2-fsa/OmniVoice/tree/main/examples)；[LoRA 文档](https://github.com/k2-fsa/OmniVoice/blob/main/docs/lora_finetuning.md)
- [dots.tts 技术报告](https://arxiv.org/html/2606.07080)；本项目[专项分析](dots_tts_training_review_20261120.md)
- [Chatterbox 官方仓库](https://github.com/resemble-ai/chatterbox)；[Dia 官方仓库](https://github.com/nari-labs/dia)；[CSM 模型卡](https://huggingface.co/sesame/csm-1b)；[OuteTTS 模型卡](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B)
