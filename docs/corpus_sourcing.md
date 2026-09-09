# TH / TL 表演语料获取：核实结论与可执行路径

回答「Tagalog 表演语料只能自建，那我去哪找」。**2026-09 逐项核实**，区分「我拉到了页面/API 确认」和「只是搜索提到」。

**一句话结论：**

- **TL：确实没有任何可商用的现成表演/情绪语料。** 但有一条我核实到的、之前没记录的现成路 —— MagicHub 的 **1285 小时 / 514 人菲律宾语自发对话**（专有授权，询价）。它是自然对话不是短剧表演、没有情绪标签，但**有真实说话人身份**，能同时解决「自然口语锚点」和「ref 配对身份」两个缺口。情绪表演仍然只能定制采集或自建。
- **TH：有现成的，但我们已经在用的两个最好的都带 CC-BY-SA。** 新核实到一个许可干净、**带 speaker ID** 的：Nexdata **1004 小时泰语**（商业买断，低噪，WAR 98%）。
- **最省力的动作不是继续找数据，是一次录制解决三件事** —— 见 [§5 自建方案](#5-自建方案一次录制解决三件事)。

---

## 1. 我核实过程中发现的三个错误信息

这三条如果不纠正，会直接导致错误决策。

### 1.1 LAION 的 DramaBox 短剧配音数据是 **TTS 合成的，不是真人**

搜索时最容易兴奋的就是这批：`laion/dramabox-voice-acting-data-annotated`，CC-BY-4.0，10 万–100 万条，标签写着 `voice-acting`、`text-to-speech`，数据卡还专门说「每个样本含同一说话人的两个情绪场景，用 `CUT TO:` 分隔，切开后得到同音色不同情绪的配对片段」—— 听起来完全是我们要的东西。LAION 有 10+ 个同族数据集。

**但数据卡的 Models Used 一栏写着源头是 `ResembleAI/Dramabox`（TTS 模型）和 `gemini-2.5-pro-tts`，文件名是 `{prompt_id}_seed{NN}_part1.mp3`（seed = 生成采样）。** 这是「用 TTS 生成 → 增强 → 超分到 48k → Whisper 标注」的**合成数据**。

AGENTS.md 对 `drama_tl` / `drama_th` 的要求写得很明确：**「不用模型合成语音补量」**。拿另一个 TTS 的输出去教我们的 TTS 表演，会把对方的声学伪影一起学进来，而且情绪标签是生成时的 prompt、不是真实表演的标注。**不要用。**

唯一可借用的：它的**标注 schema**（同音色跨情绪配对、场景切分、word-level timestamps）可以作为我们自建标注格式的参考。音频不能用。

### 1.2 AGENTS.md 里「MagicHub ASR-SFDuSC 免费注册」这条是错的

AGENTS.md 把它列为「有规模的真人语料」的免费选项之一。核实原文：

> Open Source ASR Corpus **4.58 hours** ... 4,073 utterances contributed by **ten speakers** ... Speech Style: **scripted monologue** ... This work is licensed under a **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**

三重不可用：**NC**（禁止商用）+ **ND**（禁止演绎，微调就是演绎）+ 只有 4.58 小时 10 个人的**朗读**语料。对我们的场景可用性为 0。已从 AGENTS.md 删除该条。

MagicHub 真正有价值的是另一个库，见 §2.1。

### 1.3 一篇被引用的「JaiTTS 泰语 VoxCPM 模型」论文不存在

调研过程中出现了一条看起来很关键的竞品情报：JaiTTS-v1.0，arXiv 2604.27607，声称基于 VoxCPM 改造、用了 1 万小时自建泰语语料、CER 1.94%。

**arXiv API 查这个 ID 返回 0 条，全文搜 "JaiTTS" 也返回 0 条。这篇论文是编造的**，连带那些 CER 数字和人评结果。不要基于它做任何决策。

（顺带：`biodatlab/ThonburianTTS` 是真实存在的泰语 TTS，3148 次下载，许可标 `cc`。`typhoon-ai/typhoon-whisper-large-v3` 也真实存在，见 §4。）

**教训**：这个领域的调研结论必须核实到页面/API 原文。低资源语种的数据来源本来就少，一条编造的「有 1 万小时现成语料」足以让人跳过真正该做的自建工作。

---

## 2. TL（Tagalog / Taglish）

### 2.1 现成能买的：MagicHub ASR-BigFTagaCSC ← 本次最有价值的发现

直接拉页面核实到的原文：

| 字段 | 值 |
|---|---|
| 名称 | ASR-BigFTagaCSC: A Filipino (Tagalog) Conversational Speech Corpus（MDT-ASR-E076） |
| 时长 | **1285 小时** |
| 说话人 | **514 人** |
| 内容 | Spontaneous Conversation，按话题（金融/汽车/SNS/房产/智能终端等） |
| 音频 | 16 kHz / 16 bit / WAV + TXT 转写 |
| 录音 | 手机，室内外 |
| 许可 | **Proprietary**（专有），联系 `business@magicdatatech.com` |
| 情绪标签 | 无 |

**为什么它值得优先询价：**

- **514 个真实说话人身份** —— 这是 `yodas_th` 之类 YouTube 抓取源给不了的（那些 speaker_id 是视频级近似身份，不能做 ref）。有身份才能做 ref 配对，才能按说话人统一响度
- 16 kHz / WAV + 现成转写，直接对齐我们的 `manifest.jsonl` 格式，加工链路零改造
- 自发对话（不是念稿），是「自然口语锚点」这一档最需要的东西
- 话题制录音意味着有一定情绪起伏，虽然不是表演

**它不能解决的：** 没有情绪标签，不是短剧表演，短剧需要的哭腔/讽刺/克制愤怒这些拿不到。所以它是**配角不是主力** —— 对应我们配比表里的「自然口语 30%」那一档，不是「真人短剧/表演 45%」那一档。

### 2.2 已核实的 TL 死路（不要再花时间查）

| 来源 | 核实结果 |
|---|---|
| HF 全量搜 `tagalog` / `filipino` | 音频类只有 `Speech-data/Filipino-Tagalog-Speech-Dataset`（**CC-BY-NC-ND**）、`welyjesch/Enc_Tagalog_SparkTTS_dataset`（**无许可**）、`DataoceanAI/Dolphin_*`（无许可 tag，7 次下载的 sample）。其余全是文本数据集 |
| SEACrowd 生态 | 23 个 th/tl/fil 数据集，**全部是 `modality:text` 或图像/翻译/评测**，没有一个音频 |
| `liva-ai/yapdo-convo` | 含 tl，但 **`cardData.license` 为 None（完全没有许可声明）**，且 `n<1K` 是样本量。调研里说它 CC-BY-4.0 是错的。无许可 = 商用不可用 |
| `liva-ai/code-switching-asr` | CC-BY-4.0 且有 tl，但 **`n<1K`**，是 demo 样本不是语料 |
| Common Voice tl | mozilla-foundation 在 HF 上现在只有 17_0 / 13_0；`fsicoli/common_voice_22_0` 的 tl config 查不到。与 AGENTS.md 记录的 `recordedHours=0` 一致 |
| 菲律宾高校（UP PLD 48.6h、FSC 75h、Silencio 2h） | 均为 **CC-BY-NC 或 NC**，学术用途。商用禁止 |
| YouTube / teleserye 抓取 | ABS-CBN、GMA、Viva 全是标准许可，不存在 CC 授权的菲短剧频道。抓取违反 ToS 且触及 RA8293 表演者邻接权。**商用产品不可用** |

### 2.3 付费定制采集

能同时解决「表演 + 情绪标签 + 说话人身份 + 干净商用许可」四件事的**唯一**路径。核实到的在售/可定制厂商：

| 厂商 | 核实状态 | 备注 |
|---|---|---|
| **Magic Data / MagicHub** | ✅ 页面已核实 | 已有 1285h 现货对话；同厂可做定制采集，`business@magicdatatech.com` |
| **Nexdata（= 数据堂国际品牌）** | ✅ 泰语 SKU 已核实 | 菲律宾语有 306h 话题对话、522h 朗读、1100h 全双工多通道。走 sales 询价 |
| **Appen** | ~ 官网有 Expressive TTS 定制采集条目 | 明确宣传「expressive TTS data」，最贴合需求 |
| **Datatang / 海天瑞声** | ~ 有菲律宾语女声合成库 | 海天那个是朗读，不适用 |
| **Defined.ai** | ~ 市场有 TL 自发对话（49+78+27+70h） | 多为电话窄带，采样率不适合 TTS |

**所有厂商都不公开报价，一律走 sales。** 询价模板见 §6。

---

## 3. TH（泰语）

### 3.1 CC-BY-SA 传染性 —— 先说清这条红线

我们在用的两个最好的泰语源都是 SA：

- `airesearch/thai-ser` → **CC-BY-SA-4.0**
- `CMKL/Porjai-Thai-voice-dataset-central` → **CC-BY-SA-4.0**（另注意 Porjai 的 `pattani` / `khummuang` 子集是 **CC-BY-NC-SA**，比 SA 更差，直接排除）

核实到的合规判断（**这是保守合规立场，不是法律意见，重大决策请咨询法务**）：

- SA 的触发条件是**「向公众分享改编物」**。CC 官方明确说过「模型若基于 SA 内容训练且公开发布，建议以同许可发布」—— 但那是**保守建议，不是法律要求**，且模型权重是否构成「改编物」在法律上**无定论、无判例**
- **结论 A（推荐）**：**LoRA 权重和 merge 后的完整模型都不外发**，只通过 API 交付合成音频 → SA 不触发。SA 不追及模型输出（除非输出实质复现了原音频，TTS 不会）
- **结论 B（红线）**：一旦把含 THAI-SER / Porjai 训练数据的 LoRA 或 merge 权重**公开发布**（传 HF、随客户交付、开源），保守读法要求同以 CC-BY-SA 发布 → 对闭源商用产品是致命的
- **结论 C**：EU DSM 指令第 4 条 TDM 例外、日本著作权法 30-4 条、美国 fair use 任一成立时，CC 条件可被整体架空。**单纯挂 CC 许可本身不构成 TDM 保留**（权利人需机器可读地明示保留）
- **可行的规避**：① 权重永不外发（最稳）；② 直接向版权方谈商业授权 —— **THAI-SER 的出资方就是 AIS + DEPA，有明确的谈判主体**（VISTEC / airesearch）；③ 任何需要对外发布的实验只用 Apache-2.0 / CC-BY 源

**这一条要写进产品合规红线**：`docs/finetune_playbook.md` 的 Phase 8 交付清单里，「上传 HF」这个动作在用了 SA 数据的情况下是不允许的。

### 3.2 新核实到的泰语源

| 来源 | 许可 | 规模 | 类型 | 有 speaker ID | 可用性 |
|---|---|---|---|---|---|
| **Nexdata 1004 Hours Thai**（SKU 1687，2025-07 上架） | **商业买断** | 1004h | 真人自然语音，低背景噪声，16kHz/16bit/mono WAV | ✅ **speaker ID + gender + noise**，WAR 98% | **8/10** 许可干净 + 有身份，是 `yodas_th` 的直接升级替代 |
| **`speechcolab/gigaspeech2` th** | **Apache-2.0** | 极大（10M–100M 段） | YouTube/播客口语，th/id/vi | ❌ | 6/10 **许可最干净**（无 SA/NC），但短句为主、无身份。`gated: auto` 需 HF 登录同意条款 |
| `nectec/LOTUSDIS` | CC-BY-SA-4.0 | 16.1 万条 / 11GB | 远场会议自发对话 | ✅ speaker_id + mic | 4/10 SA + 远场混响，不能当 TTS 目标 |
| Nexdata 211h 泰语全双工自发对话 | 商业买断 | 211h | 手机自发对话 | ✅ | 7/10 |
| `doyze/thai-tts-dataset` | `license: other`（research-only） | 14.92h | 单人播客 | 单人 | 0/10 |
| `dubbing-ai/vaja-thai` | other | 不明 | 不明 | 不明 | 未核实，规模太小 |
| OpenSpeechHub 的三个泰语集（TTS-665k-Th / TTS-R24000-Th / STT-v2-Th） | **无 license tag、无数据卡** | 66.5 万 / 133 万条 | 不明 | ❌ | **0/10 禁用** —— 同组织还挂着原神/星穹铁道/动漫语音的 rip，来源不可追溯 |

**已核实「不存在」**：THAI-SER 之后没有任何新的泰语情感/表演语料公开。THAI-SER 自己的 LRE 2026 正式版就写明，此前的泰语语料（ORCHID-SPEECH、NECTEC-ATR 2004）都只服务 ASR。Nexdata / Datatang 的泰语 SKU 里**没有现成情感库** —— 泰语的情绪表演语料同样只能定制采集。

**泰剧抓取同样不可行**：已核实 GMMTV 在 YouTube 上主动发起版权打击与下架（含二创和泰语配音视频），ONE31 / Ch3 同样。不存在 CC 授权的泰剧渠道。

### 3.3 泰语转写可以换更好的模型

`typhoon-ai/typhoon-whisper-large-v3`（SCB-10X，**MIT**，arXiv 2601.13044）在 Whisper Large v3 架构上用约 **11,000 小时泰语**微调，训练数据含 Gigaspeech2 + CommonVoice + 内部策展公共媒体，自带泰语数字/重复标记/歧义的归一化处理。在 Gigaspeech2、TVSpeech、FLEURS 泰语测试集上是 SOTA。

**但不是即插即用**，两个障碍：

1. `library_name: transformers` —— 走 HF transformers + torch，**不是 faster-whisper 的 CTranslate2 格式**。我们的 `pipeline._whisper_model` 用的是 `faster_whisper.WhisperModel`，要么用 `ct2-transformers-converter` 转格式，要么单开一条 transformers 转写路径
2. 模型卡写着「By using this model, you agree to the OpenTyphoon Terms and Conditions」—— MIT 之外还有一层 T&C，商用前要读一遍
3. **只有泰语**，对 TL 无帮助

优先级：**先把数据搞到手，再优化转写模型**。转写质量影响的是 `thai_ser` impro 那批没有原文的条目的文本准确性，是二阶问题。

---

## 4. 三条路的取舍

| 路线 | 时间 | 钱 | 能解决 | 不能解决 |
|---|---|---|---|---|
| **A. 买现货**（MagicHub TL 1285h / Nexdata TH 1004h） | 周级 | 中 | 自然口语锚点、真实说话人身份、干净许可 | ❌ 情绪标签 ❌ 短剧表演风格 |
| **B. 定制采集**（Magic/Nexdata/Appen/Datatang） | 月级（周期长） | 高 | ✅ 表演 ✅ 指定情绪 ✅ 身份合同锁定 ✅ 商用买断无 SA/NC | 起订量大（建议 100–300h）、需要自己写脚本 |
| **C. 自建**（自己找演员录 + 走「素材导入」链路） | 周级 | 低–中 | ✅ 全部，且脚本完全贴合产品台词风格 | 需要自己找人、自己标注 |

**推荐组合：A + C 并行，B 作为下一步扩量的选项。**

理由：A 立刻补上「自然口语」和「说话人身份」两个缺口（这是现成的、买得到就能用）；C 用最小成本拿到真正决定情绪表现力上限的表演语料，而且**顺带解决跨语言 ref**（见下节）；B 周期和起订量都不适合首轮实验。

---

## 5. 自建方案：一次录制解决三件事

这是最省力、性价比最高的路径 —— **不是「去录一批语料」，而是设计一场同时产出三样东西的录制**。

三个缺口一次补齐：

1. **真人表演/对白语料**（配比表的 45% 主力档）
2. **跨语言同人 ref** —— `docs/finetune_playbook.md` 的决策点 2 里那个「拿不到就无法验证」的硬约束。让同一个演员在同一个场次里既录目标语言台词、又录中文和英文短句，`speaker_verified=true` 就有了真人依据，不需要 MFCC 猜测
3. **可信情绪标签** —— 录制时按情绪脚本走，`emotion_verified` / `control_verified` 当场就是真的，不是事后猜

### 5.1 规模建议

首轮实验够用即可，不要一次录太多：

| 项 | TH | TL |
|---|---|---|
| 演员数 | 6–8 人（女 4 / 男 3，覆盖 20s–50s 年龄段） | 同 |
| 每人目标语料 | 5–8 小时 | 5–8 小时 |
| 每人中/英 ref | **各 20–30 条 3–10s 短句**（关键，不能省） | 同 |
| 合计 | 30–60h | 30–60h |

30–60h ≈ 1–2 万条 3–30s 片段 ≈ 单卡 1 epoch 600–1300 步，几小时跑完一轮，迭代速度合适（对照 playbook Phase 2.1 的规模表）。

**中/英 ref 那 20–30 条是整场录制里最便宜也最关键的部分** —— 每人多花 10 分钟，换来的是唯一能真正建立「中/英 ref → 目标语言」通路的训练信号。不录这批，playbook 的 R3 实验永远只能标「未验证」。

### 5.2 情绪 × 场景矩阵

不要让演员「随便演」。按矩阵走，保证每种情绪**跨多名演员**出现（否则模型会学成「这个情绪 = 这个人的音色」）：

| 情绪类别 | 自动短语池能覆盖？ | 录制时要什么 |
|---|---|---|
| 中性 / 平静 | ✅ `neutral` | 日常对白基准，占最大比例 |
| 愤怒 / 克制愤怒 | ⚠️ 只有 `angry`（愤怒地） | **克制愤怒要手写 control**：压低音量、咬牙 |
| 惊讶 / 难以置信 | ✅ `surprised` | 疑问句、反问 |
| 伤心 / 哭腔 | ⚠️ 只有 `sad` | **哭腔要手写 control**：声音发抖、带哽咽 |
| 开心 / 带笑说话 | ⚠️ 只有 `happy` | **带笑说话要手写 control**：笑意在声音里 |
| 害怕 / 担心 | ✅ `fearful` | 急促、气声 |
| 讽刺 / 挖苦 | ❌ 池里没有 | **必须手写 control** |
| 烦躁 / 厌恶 | ✅ `frustrated` / `disgust` | 叹气、拖音 |
| 语速变化 | ✅ `rate_label: slow/fast` | 各录一批，快语速要**真说得快**，不要后期加速 |
| 音量变化 | ✅ `volume_label: quiet/loud` | 耳语和喊叫各录一批 |

标 ✅ 的可以直接填 `emotion` 字段走自动短语池；标 ⚠️/❌ 的必须手写 `control_zh` + `control_en` 并置 `control_verified=true`。这两条路的区别见 playbook Phase 2.3.3。

### 5.3 交付格式：直接对齐 manifest.jsonl

让录制方按这个命名和结构交付，加工链路零改造：

```
delivery/
  actor01/
    th/         # 目标语言台词
      actor01_th_0001.wav
      actor01_th_0002.wav
    zh_ref/     # 同一人的中文 ref
      actor01_zh_ref_0001.wav
    en_ref/     # 同一人的英文 ref
      actor01_en_ref_0001.wav
  actor02/
  ...
  manifest.jsonl
  session_log.csv   # 每条对应的录制场景/情绪，人工核实用
```

录音规格：**48kHz/24bit WAV 单声道，安静室内，单人，不要加任何后期处理**（不降噪、不压缩、不归一化 —— 我们的加工链路自己做响度对齐，上游归一化会抹掉「音量=情绪强度」这条线索，而且抹掉了恢复不回来）。

`manifest.jsonl` 每行（字段含义见 playbook Phase 2.3.1）：

```jsonl
{"audio":"actor01/th/actor01_th_0001.wav","text":"裸台词，不要自己加括号","lang":"th","speaker":"actor01","speaker_namespace":"cast_2026q4","speaker_verified":true,"session":"rec01","emotion":"angry","emotion_verified":true,"control_zh":"愤怒地，音量压低","control_en":"angry, keeping the volume down","control_verified":true}
{"audio":"actor01/zh_ref/actor01_zh_ref_0001.wav","lang":"zh","speaker":"actor01","speaker_namespace":"cast_2026q4","speaker_verified":true,"session":"rec01_ref","reference_only":true}
{"audio":"actor01/en_ref/actor01_en_ref_0001.wav","lang":"en","speaker":"actor01","speaker_namespace":"cast_2026q4","speaker_verified":true,"session":"rec01_ref","reference_only":true}
```

**`speaker_namespace` 全场统一一个值**（例如 `cast_2026q4`），这样跨 TH/TL 两个语言时同一演员的 ID 是一致的。挑一两个演员的场次写进 `holdout.json` 钉成永久验证集（playbook Phase 2.4）。

### 5.4 台词从哪来

不要现编。直接从**产品实际要配的短剧台词**里选，按情绪分类抽 3–30s 的完整单人句。这样录出来的语料分布和线上推理分布一致，比通用情绪脚本有效得多。TL 记得包含 Taglish 句（句内英文词、品牌名、数字），对应线上「RAW 被念成英文」那类反馈。

---

## 6. 定制采集询价模板（可直接发）

发给 Magic Data / Nexdata / Appen / Datatang 的 sales。中英双语，对方多半是中国厂商的国际线。

```
Subject: RFQ — Custom Expressive Speech Corpus Collection (Thai + Tagalog, commercial buyout)

背景 / Background
我们在做一个商用短剧配音 TTS 产品（中/英文参考音频 → 泰语、Tagalog 输出），
需要定制采集一批真人表演语料用于模型微调。
We build a commercial short-drama dubbing TTS product and need a custom-collected
acted speech corpus for model fine-tuning.

需求 / Requirements
1. 语言 Languages: 泰语 (th-TH) 与 菲律宾语/Tagalog (tl-PH)，可分两批
2. 类型 Style: 短剧对白表演（acted dialogue），不是朗读、不是呼叫中心
3. 规模 Volume: 首轮 100 小时 / 语言；请同时报 50h / 100h / 300h 三档单价
4. 演员 Speakers: 每语言 ≥ 10 人，男女各半，年龄 20–50 岁分布；
   需有配音或表演经验，演员身份可追溯
5. 情绪 Emotions: 每条带情绪标注，需覆盖以下 10 类，且**每类由 ≥3 名不同演员**录制：
   中性 / 愤怒 / 克制愤怒 / 惊讶 / 伤心 / 哭腔 / 开心 / 带笑说话 / 讽刺 / 害怕
   neutral, angry, restrained anger, surprised, sad, tearful, happy,
   speaking-while-laughing, sarcastic, fearful
6. 附加维度 Extra dimensions: 语速（慢/正常/快）、音量（耳语/正常/喊叫）分别标注
7. 台词 Script: 我方提供台词脚本（取自实际短剧剧本），贵方按脚本组织录制；
   Tagalog 需包含 Taglish 句（句内英文词、品牌名、数字）
8. 音频 Audio: 48kHz / 24bit / WAV / 单声道，安静录音棚，
   **不做降噪、不做压缩、不做响度归一化**（我方自行处理）
9. 单条时长 Utterance length: 3–30 秒完整单人语流，不要孤立词
10. 转写 Transcript: 逐条精确转写，含标点；泰语保留声调符号
11. 额外交付 Extra: 每位演员另录 **20–30 条中文短句和 20–30 条英文短句**
    （3–10 秒，内容不限），用于同一说话人的跨语言参考音频。
    这是必需项，请单独报价。

合同条款 / Contract terms（关键，请在报价时明确确认）
- **商用买断，许可中不得含 ShareAlike (SA) 或 NonCommercial (NC) 条款**
  Full commercial buyout; license must NOT contain SA or NC clauses
- 演员声音与表演的完整授权，可追溯至每位演员的书面同意
  Full voice/performance rights from every speaker, traceable to written consent
- 我方有权将数据用于训练闭源模型并商用其输出，无需公开模型权重或数据来源
- 数据独家性 / 是否转售给第三方，请说明

请回复 / Please reply with
1. 现货 vs 定制：是否已有可直接授权的泰语/菲律宾语**表演或情感**语料？
   （注意：朗读语料、ASR 对话语料我们另有渠道，此处只要表演/情感）
2. 三档规模（50/100/300h）的单价与总价
3. 交付周期（从签约到全量交付）
4. 是否接受我方提供台词脚本
5. 5–10 条**免费试听样本**（需覆盖至少 3 种情绪），用于评估录音质量与表演水平
6. 标注交付格式（是否支持自定义 JSONL 字段）
7. 合同许可条款原文样本
```

**收到报价后的判断标准**：先听样本，样本里情绪表演假、或者录音有棚噪/混响，直接换厂商 —— 这类数据买回来也是废的，因为**我们没有降噪链路**（无 demucs，官方 zipenhancer 依赖 modelscope 且不在 lock 里）。

---

## 7. 建议动作（按 ROI 排序）

1. **本周**：向 `business@magicdatatech.com` 发 §6 的询价，同时问 ASR-BigFTagaCSC（1285h 现货）的价格和试用样本。向 Nexdata 问 1004h 泰语的价格和样本
2. **本周**：把「**LoRA 权重与 merge 后模型一律不对外分发**」写进产品合规红线（THAI-SER / Porjai 都是 CC-BY-SA）
3. **本月**：启动 §5 的自建录制 —— 6–8 名演员 × 5–8h，**务必包含每人 20–30 条中/英 ref**，台词从产品实际短剧剧本里选
4. **可选**：如果 THAI-SER 的数据要用于任何会对外发布权重的实验，向 VISTEC / airesearch（出资方 AIS + DEPA）谈商业授权
5. **不要做**：抓 YouTube 泰剧/菲剧；用 OpenSpeechHub 那批无许可泰语集；用 LAION DramaBox 合成音频补量；再花时间找「现成的 TL 表演语料」（已核实不存在）

---

## 附：核实状态标记

- ✅ = 我拉到了页面原文或 HF/arXiv API 响应确认
- ~ = 只有搜索结果或第三方提及，未二次核实，**不要写进采购单**
- ❌ = 已核实为错误或不存在

| 条目 | 状态 |
|---|---|
| MagicHub ASR-BigFTagaCSC 1285h / 514 人 / 专有授权 | ✅ |
| MagicHub ASR-SFDuSC 4.58h / 10 人 / CC-BY-NC-ND | ✅（AGENTS.md 原记录有误，已更正） |
| Nexdata 1004 Hours Thai（SKU 1687）字段与 WAR 98% | ✅ |
| `speechcolab/gigaspeech2` th = Apache-2.0 | ✅ |
| `nectec/LOTUSDIS` = CC-BY-SA-4.0 | ✅ |
| `liva-ai/yapdo-convo` 无许可声明 | ✅ |
| LAION DramaBox 族 = TTS 合成音频 | ✅（数据卡 Models Used 明确） |
| `typhoon-ai/typhoon-whisper-large-v3` = MIT / 11000h 泰语 | ✅ |
| JaiTTS arXiv 2604.27607 | ❌ **不存在，arXiv API 返回 0 条** |
| CC-BY-SA 对模型权重的传染性分析 | ✅ 引用来源已核实，但**法律结论无判例，属保守合规立场** |
| Appen Expressive TTS 定制采集 | ~ 官网条目存在，未核实泰语/菲语库存 |
| Defined.ai TL 自发对话 SKU 与时长 | ~ |
| Nexdata 菲律宾语 306h / 522h / 1100h | ~ |
| 菲律宾高校 UP PLD 48.6h / FSC 75h / Silencio 2h | ~ 许可为 NC |
| Chulalongkorn CCOST（Zenodo 17366698） | ~ Zenodo 超时，未核实 |
| Common Voice Scripted Speech 26.0 泰语小时数 | ~ |
