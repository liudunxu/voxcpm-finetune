# VoxCPM 2 短剧配音 LoRA 微调流程（泰语 TH / Tagalog TL）

面向的生产形态：**参考音频是中文或英文发音，输出是泰语或 Tagalog/Taglish 台词**，要求情绪可被中英文控制前缀驱动、发音清晰不漏词、音色跟着 ref 走。

本文是可以照着一步步执行的流程。开发约定与踩坑记录见 [AGENTS.md](../AGENTS.md)，数据源取舍背景见 [README.md](../README.md)。

---

## 目录

- [1. 预期边界：微调能修什么，不能修什么](#1-预期边界)
- [2. 三个决策点（先读，决定后面所有步骤）](#2-三个决策点)
- [Phase 0 · 环境准备](#phase-0--环境准备一次性)
- [Phase 1 · 先建评测集](#phase-1--先建评测集在训练之前)
- [Phase 2 · 数据准备](#phase-2--数据准备)
- [Phase 3 · 逐源加工与检查](#phase-3--逐源加工与检查)
- [Phase 4 · 混合与检查](#phase-4--混合与检查)
- [Phase 5 · 训练](#phase-5--训练)
- [Phase 6 · 验收](#phase-6--验收)
- [Phase 7 · 迭代调参：症状 → 动作](#phase-7--迭代调参症状--动作)
- [清晰度专项 · 全链路控制点](#清晰度专项--全链路控制点)
- [Phase 8 · 导出与上线](#phase-8--导出与上线)
- [附录 A · 坑速查](#附录-a--坑速查)
- [附录 B · 全流程 checklist](#附录-b--全流程-checklist)

---

## 1. 预期边界

先对齐预期，否则后面所有的调参都是在追一个拿不到的目标。

**微调能改善的：**

| 维度 | 能改善到什么程度 | 靠什么 |
|---|---|---|
| 目标语言发音 / 重音 / 声调 | 明显改善，尤其是泰语声调、Taglish 句内英文词的读法 | 目标语料占比 + 发音锚点源（`fleurs` / `porjai_th` / `filswitch`） |
| **清晰度 / 可懂度** | 明显改善，但**上限由数据入口决定** —— 没有降噪链路，糊的录音进去就是糊的模型出来 | 三条战线：① 只收干净录音 + 训练文本必须与音频一致 ② 首尾裁切不啃掉词首清辅音 ③ 推理 `inference_timesteps=20`。详见[清晰度专项](#清晰度专项--全链路控制点) |
| 自然对白韵律、去念稿感 | 改善，上限由**真人表演语料的数量和质量**决定 | `drama_*` / `thai_ser` impro 占比；表演源不能裁掉换气 |
| 情绪指令响应 | 改善，但只对**训练里出现过的可信标签**生效 | `(控制指令)正文` 前缀样本占比 25–50%，且标签 `*_verified=true` |
| 漏词 / 吞字 | 改善 | 训练文本准确 + 尾静音 <0.5s + 不拼接孤立词 |
| 音色漂移 | 改善（保住基座克隆能力） | LoRA 而非全量；ref 覆盖 30–50%；中英回放 15% |
| 生成停不下来 / 尾巴拖长 | 改善 | 尾静音裁切生效（朗读源必须走 VAD，见附录 A） |

**微调不能修的（不要指望，也不要为它调参）：**

- 翻译错误、输入台词本身缺词 —— 那是上游文本链路的问题
- 剪辑截断、时间轴对不上
- 角色选角不合适（ref 选错人）
- BGM / 音效混入 —— 本项目**不做声源分离和降噪**（无 demucs 依赖，官方 zipenhancer 需 modelscope 且不在 lock 里）。有对白轨就喂对白轨，没有就在素材导入的试听环节把 BGM 重的条目判「丢弃」
- 上游已经逐条峰值归一化抹掉的动态范围 —— 恢复不了

**指标的正确用法：** CER/WER、疑似漏尾、F0 全部只是**诊断信号**。通过与否由至少两名母语评审盲听决定。`f0_std_st` 含泰语词汇声调误差，不能当自然度用；`energy_range_db` 是能量分位差，**不是 SNR**。禁止用 `min_snr_db` / `min_f0_std` 硬筛（代码里启用会直接报错）。

---

## 2. 三个决策点

### 决策点 1 · 表演语料从哪来：现成的补不上情绪，只能自建那一档

**逐项核实结论见 [docs/corpus_sourcing.md](corpus_sourcing.md)**（含厂商询价模板和自建录制脚本模板）。这里只给结论。

已核实、不要重复调研的死路：Common Voice tl 官方 `recordedHours=0`，YODAS/YODAS2 Sidon 的 224 语种里没有 tl/fil，OpenSLR 无菲律宾语资源，SEACrowd 的 23 个 th/tl/fil 数据集全是文本无音频，HF 上 `modality:audio` 匹配 filipino/tagalog 的只有厂商 sample（`n<1K`，多为 CC-BY-NC-ND、gated 或**完全没有许可声明**）。`filipino_emotion` 连数据卡都没有，实测中位时长约 1.6s，绝大多数低于 3s 下限。**泰语同样没有公开的情感/表演语料** —— THAI-SER 之后没有新的，Nexdata/Datatang 的泰语 SKU 里也没有现成情感库。

**两个陷阱**（都很容易踩，因为看起来完全对口）：

- `laion/dramabox-voice-acting-data-annotated` —— CC-BY-4.0、10 万–100 万条、标签写着 `voice-acting`、数据卡还讲「同说话人跨情绪配对片段」。**但它是 TTS 合成的**（源头 `ResembleAI/Dramabox` + `gemini-2.5-pro-tts`，文件名带 `_seed{NN}`）。违反「不用模型合成语音补量」，情绪标签也是生成 prompt 不是真实表演标注。只有标注 schema 可参考
- MagicHub **`ASR-SFDuSC`** —— 4.58h / 10 人**朗读**，**CC-BY-NC-ND**（NC 禁商用、ND 禁演绎，微调就是演绎）。可用性 0。别和下面那个库搞混

**现成能买的（补的是「自然口语 + 说话人身份」，不是情绪）**：

- **MagicHub `ASR-BigFTagaCSC`** —— **1285h / 514 人**菲律宾语自发对话，16kHz WAV + TXT 转写，专有授权需询价。514 个**真实说话人身份**是 YouTube 抓取源给不了的（那些 speaker_id 是视频级近似身份，不能做 ref），能同时补自然口语锚点和 ref 配对身份。**但无情绪标签、不是表演**，对应配比表的「自然口语 30%」档，不是「真人短剧/表演 45%」档
- **Nexdata 1004 Hours Thai**（SKU 1687）—— 商业买断、低背景噪声、16kHz mono WAV、**带 speaker ID + gender**、WAR 98%。许可干净的泰语自然语音，是 `yodas_th` 的升级替代
- `speechcolab/gigaspeech2` th —— **Apache-2.0**（无 SA/NC 污染），但短句为主、无说话人身份，`gated: auto`

**结论：表演/情绪这一档只能定制采集或自建。** 最省力的做法不是继续找数据，而是**一次录制同时解决三件事**（详见 sourcing 文档 §5）：

1. 真人表演对白语料（配比表 45% 主力档）
2. **跨语言同人 ref** —— 让同一演员在同一场次里既录目标语言台词、又录 20–30 条中文和英文短句。每人多花 10 分钟，就拿到了决策点 2 里那个「拿不到就无法验证」的硬约束的唯一解
3. **可信情绪标签** —— 按情绪脚本录，`emotion_verified` / `control_verified` 当场就是真的

规模：每语言 6–8 名演员 × 5–8h ≈ 30–60h ≈ 1–2 万条 ≈ 单卡 1 epoch 600–1300 步，几小时一轮。**台词直接从产品实际要配的短剧剧本里选**，分布和线上推理一致，比通用情绪脚本有效得多。

首轮 TL 表演语料目标：**5–10h 干净真人对白**，覆盖多名女声、男声和不同年龄段，重点补质疑、克制愤怒、担心、讽刺、哭腔、带笑说话、自然停顿。**每种情绪必须由 ≥3 名不同演员录** —— 否则模型学成「这个情绪 = 这个人的音色」。

TH 好一些：`thai_ser` 的 impro（即兴对话）子集有 `actor_id` 和情绪标签，是开源里唯一带可靠身份的泰语表演语料。**但它是 CC-BY-SA-4.0**，触发交付红线，见 Phase 8。

### 决策点 2 · 跨语言同人 ref 是硬约束，不是可以凑的字段

生产场景的 ref 是中文/英文，目标是泰语/Tagalog。要让 LoRA 学到「按 ref 的音色说目标语言」，训练集里必须有**同一个人**的 ref 音频和目标音频配对。

**红线（`preflight` 会拦，也不许绕）：**

- 不许拿另一个人的英文声音配这个人的泰语目标 —— 教出来的是错误的音色映射
- 不许用 MFCC 聚类相似度当身份证据。聚类只能辅助审计
- 不许因为 ref 覆盖率不够就调低阈值强凑。身份未知 → 默认不配 ref
- 中英文回放（`aishell3` / `replay_en`）**不是**跨语言同人 ref 的替代品，它只防中英能力遗忘

**`speaker_verified=true` 的含义是「人工或可靠原始元数据确认了这是同一个人」**，不是「有一列叫 speaker」。同视频、相同角色名、声音听起来像，都不够。同一演员跨数据源时使用统一的 `speaker_namespace` 和 ID。

**性价比最高的一步（如果预算允许）：** 找 5–10 名目标语言演员，每人录 3–10s 的中文或英文短句若干条（不需要标准，需要是同一个人），作为 `reference_only=true` 行导入。这一小批双语录音是唯一能真正建立「中/英 ref → 目标语言」通路的训练信号，几十分钟录音的收益远大于再加 50 小时无 ref 语料。

拿不到时怎么办：**不要声称已完成跨语言 ref 验证。** 走 R1/R2 把语言和风格调好，ref 通路靠「同语言同人 ref」保住机制不被 LoRA 破坏，然后在 Phase 6 用 A/B 实测中英 ref 的音色保持有没有退化 —— 基座本身有跨语言零样本克隆能力，LoRA 的任务是别把它弄坏。

### 决策点 3 · 先分别训 TH 和 TL，不要一上来做联合模型

先各自跑通、各自验收，确认有效后再考虑联合多语种。联合模型会把两个语言的语料量、情绪覆盖、ref 覆盖问题混在一起，出问题时无法归因。

**执行顺序：TH 先行**（有 `thai_ser` impro 这个开源表演源，数据准备成本低于 TL），跑通完整链路、把评测集和盲听流程建立起来，再把同一套流程复制到 TL。

---

## Phase 0 · 环境准备（一次性）

### 0.1 拉代码与依赖

```bash
git clone --recurse-submodules <repo>
cd voxcpm_finetune
uv sync --group qc          # qc 组 = faster-whisper + PyAV，转写和素材导入都需要
# 已克隆但 submodule 为空：
git submodule update --init --recursive
uv run pytest               # 应该全绿
```

Python 3.11（`.python-version` 已固定）。torch 平台自动分流：macOS 走 PyPI 轮子（CPU/MPS），Linux 走 pytorch-cu124 index。**训练只在 Linux GPU 机执行，数据集下载也只在远程做**，本地不下载语料。

### 0.2 远程 `.env`

复制 `.env.example` 为 `.env`。远程 GPU 机的关键四项：

```dotenv
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx
WANDB_PROJECT=voxcpm-finetune
VOXCPM_BASE_PATH=/root/autodl-tmp/models/VoxCPM2

VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data
VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt
HF_HOME=/root/autodl-tmp/hf_home
HF_ENDPOINT=https://hf-mirror.com
```

**系统盘小，`VOXFT_DATA_ROOT` / `VOXFT_CKPT_ROOT` / `HF_HOME` 三个都必须指到大盘**（`/root/autodl-tmp/*`），否则几十 GB 语料会把系统盘写满。

镜像相关不用手动管：导入 `voxft` 时 `paths.load_dotenv()` 先加载 `.env`，然后 `_disable_xet_on_mirror()` 在 `HF_ENDPOINT` 非 huggingface.co 时自动 `setdefault("HF_HUB_DISABLE_XET","1")`。**但独立 `hf` / `huggingface-cli` 命令不读项目 `.env`**，要用就得自己 export 三个变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/root/autodl-tmp/hf_home
```

优先用 `python -m voxft.data.prefetch`，它把这些都处理好了。

### 0.3 预取权重

```bash
uv run python -m voxft.data.prefetch --whisper large-v3      # ~3GB，转写标签用
uv run python -m voxft.data.prefetch --repo openbmb/VoxCPM2  # 基座
```

large-v3 国内直连 huggingface.co 常在 SSL 握手就超时。加载已有 3 次重试并打印 endpoint。下载完可以把路径固定进 `.env`，避免以后走网络：

```bash
echo 'VOXFT_WHISPER_MODEL_LARGE=/root/autodl-tmp/hf_home/...' >> .env
```

### 0.4 启动工作台

```bash
nohup uv run voxft-ui > ui.out 2>&1 &      # 端口 6006
```

**必须用 `nohup` 或 tmux，别把 stdout 挂在会断的终端上。** 库函数一律走 `progress` 回调不 print，但如果 UI 进程的 stdout 是已断开的 pty，仍然可能出问题。

页面 6 个 Tab：数据集 / 素材导入 / 训练 / 试听 / 模型管理 / 日志。下面所有 CLI 命令都有对应的页面操作，用哪个都行 —— **但页面下拉框的 `choices` 只在 `build_ui` 时算一次**，任何运行后变化的列表（已加工数据集 / 配置 / LoRA / 上传目录）要靠事件输出或重新点 Tab 刷新。

---

## Phase 1 · 先建评测集（在训练之前）

顺序反了会导致「训完了才发现没法判断好坏」，然后只能凭感觉调参。

### 1.1 评测集要求

- **未见演员、未见会话**。绝对不能从训练 ref 里取
- 每个目标语言 **80–100 条**
- 逐条固定 `text` / `lang` / `ref_audio` / `ref_lang` / `control` / `seed`
- 存成 JSONL，进 git，多轮实验共用同一份 —— 换了评测集，前后轮的结论就不可比了

### 1.2 覆盖矩阵

按这几个维度交叉，别只测一种组合：

| 维度 | 取值 |
|---|---|
| ref 语言 | 中文 / 英文 / 目标语言本身 |
| 控制前缀 | 无 / 基本情绪 / 复合指令（哭腔、带笑、克制愤怒） |
| 角色 | 女主 / 其他女声 / 男声 |
| 语体 | 普通口语 / 强情绪 / Taglish（仅 TL）/ 长句 / 短句 |
| **清晰度压力** | 长句（>20s）/ 多音节难词 / 数字与英文混排 / 快语速（`语速快`）/ 耳语（`轻声`）/ 句尾辅音收尾 |
| 回归 | 中文回放 / 英文回放（检查有没有灾难性遗忘） |

清晰度压力用例要占评测集的 **20–30%**，并且 `kind` 标成 `clarity`，报告里单独分组看。这类 case 平时不会暴露问题，但一旦模型吐字变糊，它们最先崩 —— 等用户反馈才发现就晚了。泰语重点测声调对立的最小对（同音节不同声调）和句尾塞音；Taglish 重点测句内英文词、品牌名、数字读法（对应线上「RAW 被念成英文」类反馈）。

保留用户反馈里的难词和漏尾句，**但先让母语者确认台词与预期读法** —— 翻译改写不属于训练能自动修复的范围，拿错台词当 ground truth 会把结论带偏。

### 1.3 JSONL 格式

相对 `ref_audio` 路径按 JSONL 所在目录解析。

```jsonl
{"case_id":"th_female_surprise_enref","text":"เธอทำแบบนี้ได้ยังไง","lang":"th","ref_audio":"refs/female_en.wav","ref_lang":"en","control":"surprised, in disbelief","speaker":"heldout_f01","kind":"emotion","seed":42}
{"case_id":"th_female_plain_zhref","text":"ฉันไม่คิดว่ามันจะจบแบบนี้","lang":"th","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"","speaker":"heldout_f01","kind":"plain","seed":42}
{"case_id":"th_male_angry_zhref","text":"หยุดพูดซะที ฉันฟังมามากพอแล้ว","lang":"th","ref_audio":"refs/male_zh.wav","ref_lang":"zh","control":"愤怒地，音量压低","speaker":"heldout_m01","kind":"emotion","seed":42}
{"case_id":"tl_female_taglish_enref","text":"Hindi mo alam na buntis ka?","lang":"tl","ref_audio":"refs/female_en.wav","ref_lang":"en","control":"surprised, restrained","speaker":"heldout_f02","kind":"emotion","seed":42}
{"case_id":"th_clarity_final_stop","text":"อยากบอกให้รู้ว่าไม่อยากไปแล้ว","lang":"th","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"","speaker":"heldout_f01","kind":"clarity","seed":42}
{"case_id":"tl_clarity_numbers_taglish","text":"May 3 appointments ako bukas ng 10:30 ng umaga sa BGC.","lang":"tl","ref_audio":"refs/male_en.wav","ref_lang":"en","control":"语速快","speaker":"heldout_m01","kind":"clarity","seed":42}
{"case_id":"zh_regression","text":"你到底想怎么样，我已经解释过很多遍了。","lang":"zh","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"烦躁地","speaker":"heldout_f03","kind":"antiforget","seed":42}
{"case_id":"en_regression","text":"Are you okay? I was worried about you.","lang":"en","ref_audio":"refs/male_en.wav","ref_lang":"en","control":"","speaker":"heldout_m02","kind":"antiforget","seed":42}
```

`kind` 是自定义标签，报告里原样保留，方便分组看结果。`lang` 只支持 `th` / `tl` / `zh` / `en`。

放这里：

```bash
mkdir -p eval/refs            # 参考音频
$EDITOR eval/th_holdout.jsonl eval/tl_holdout.jsonl
```

### 1.4 先跑一次基线（R0）

**基座必须永久保留为固定评测基线**，后面每一轮都跟它比。

```bash
uv run python -m voxft.eval base \
  --texts-file eval/th_holdout.jsonl --seeds 42 43 44
```

这会为每条 case × 每个 seed 生成音频，跑 large-v3 转写，算 CER/WER/疑似漏尾，写报告到 `$VOXFT_CKPT_ROOT/eval/`（文件名带 uuid，多次运行不覆盖）。

三个 seed 是为了区分「稳定的问题」和「采样噪声」。只跑一个 seed 就把某个 case 判成坏例，很可能是运气。

---

## Phase 2 · 数据准备

### 2.1 首轮配比（实验起点，不是已验证最优值）

按**过滤后的训练音频时长**计算，不是按条数。

| 数据角色 | TH | TL |
|---|---:|---:|
| 真人短剧 / 即兴表演 | 45%（`thai_ser` impro + `drama_th`） | 45%（`drama_tl`） |
| 自然口语 | 35%（审核后的 `yodas_th`） | 30%（自有自然 TL/Taglish） |
| 发音补充 | 5%（`fleurs_th` / `porjai_th`） | 10%（`filswitch` / `fleurs_tl`） |
| 中文回放 | 10%（`aishell3`） | 10%（`aishell3`） |
| 英文回放 | 5%（`replay_en`） | 5%（`replay_en`） |

**小源不足时不强凑配比。** 混合器会记录请求时长和实际时长占比，差异看 `mix.json`。宁可某个角色少一点，也不要靠 3× 重复把小源撑到目标占比 —— 重复曝光会过拟合。

首轮规模建议：

| 阶段 | 训练总时长 | 条数量级（3–30s） | 1 epoch 步数（单卡 batch2×accum8） |
|---|---|---|---|
| 试跑（验证链路） | 1–2h | 500–1500 | 30–100 |
| 首轮实验 | 20–40h | 1–2 万 | 600–1300 |
| 扩量 | 80–150h | 4–8 万 | 2500–5000 |

步数公式：`ceil(条数 × epoch / (batch_size × grad_accum × GPU数))`。这是按清单条数的**近似**，官方 loader 的丢尾和长度过滤会有偏差，以日志里的实际样本数为准。单步耗时需要实测（音频序列长，`batch_size=2` 是官方示例值，激活显存大，**不要调大**）。

先从首轮实验规模开始，几小时能跑完一轮，才有迭代速度。

### 2.2 下载公开源

```bash
# 只在远程执行。先小样本试跑确认列映射和过滤条件生效，再全量。
uv run python -m voxft.data.download --source thai_ser --max-samples 200
uv run python -m voxft.data.download --source thai_ser        # 整包 12.7GB（含 4 路麦克风）

uv run python -m voxft.data.download --source yodas_th --max-samples 200
uv run python -m voxft.data.download --source yodas_th        # 整包 26.6GB

uv run python -m voxft.data.download --source fleurs_th
uv run python -m voxft.data.download --source filswitch
uv run python -m voxft.data.download --source aishell3        # ~20GB，Apache-2.0
```

产物落在 `$VOXFT_DATA_ROOT/raw/<source>/manifest.jsonl`。

下载器已经按 registry 做了行级过滤，**过滤条件缺列或值无效时不放行**（宁可少数据也不让未知数据进训练）：

- `thai_ser`：`turn_type=impro` 且 `agreement ≥ 0.7`；音频列按 `mic_con → mic_clip → mic_middle` 顺序取，**不用 `mic_zoom`**（网络录音，质量差）。这个源**没有名为 `audio` 的列**，全靠 registry 的 `audio_cols` 映射 —— 映射错了整个源会在下载阶段被静默跳过
- `yodas_th`：`grade_avg ∈ {S+, S}` 且 `dnsmos_overall ≥ 3.2`。`utt_id.rsplit("-", 3)[0]` 保留可能含 `-` 的完整视频 ID 作为 session。`speaker_id` 是视频级近似身份，**不作 ref 依据**
- `filswitch`：转换 parquet 可能只有元数据，音频在原仓库的独立 FLAC 文件里。`bytes=None` 不等于无音频，下载器会继续按其中的 HF 地址下载（保留 revision、镜像、认证、缓存）。**不要把音频地址套到 `refs/convert/parquet` 分支**。如果日志出现「写入 0 条」，先看具体的缺文本/音频读取失败统计，读取失败会记录原因、不静默跳过整包
- `aishell3`：content.txt 的同一正文列交错汉字与拼音，下载器已剔除拼音。**旧 processed 清单必须重新加工**，不能直接混入
- `filipino_speech`：过滤 `speech_type=machine` 与 `num_words < 4`，只保留完整句。**不拼接孤立词、不用随机抖动停顿伪造对白**（中位 0.63s / num_words 中位 1，拼接会训出报菜名式念稿感）

进度监控：`snapshot_download` 的进度条不传给单文件，用缓存目录大小轮询。xet 下载分两阶段（downloading → reconstructing），进度「回退」属正常。

### 2.3 自建短剧语料（主力，也是唯一可信的情绪/ref 来源）

#### 2.3.1 准备原始清单

已经切好、有文本的真人对白，直接写 `data/raw/<source>/manifest.jsonl`。音频路径相对 JSONL 所在目录。

```jsonl
{"audio":"audio/actor01_th_001.wav","text":"เธอทำแบบนี้ได้ยังไง","lang":"th","speaker":"actor01","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording01","emotion":"surprised","emotion_verified":true,"control_zh":"惊讶地，语气克制","control_en":"surprised, restrained","control_verified":true}
{"audio":"audio/actor01_zh_ref.wav","lang":"zh","speaker":"actor01","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording02","reference_only":true}
{"audio":"audio/actor02_th_014.wav","text":"ฉันไม่รู้จะพูดอะไรแล้ว","lang":"th","speaker":"actor02","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording01"}
```

字段规则：

- `audio` / `text` 必填。`text` 是**裸台词**，不要自己加括号前缀 —— 控制指令放 `control_zh` / `control_en`，加工阶段按 `control_ratio` 自动拼
- `speaker_verified=true` 见 [决策点 2](#决策点-2--跨语言同人-ref-是硬约束不是可以凑的字段)
- `session` 是同一次录音/同一集的分组键，用于 train/val 隔离。**同一集的切片必须有相同 session**，否则会被拆到 train 和 val 两边造成泄漏
- `reference_only=true` 的行只作为该 split 内的 ref 候选，**不作为训练目标**，时长约束放宽到 3–10s，文本可空
- 没有可信标签就**不写** `emotion_verified` / `control_verified` —— 缺失会让该条保持裸文本，这是正确的降级；写个 `false` 也一样，但不要凭猜测写 `true`

#### 2.3.2 成片素材导入

素材是成片视频或音轨时，用导入链路代替手工切片：

```
PyAV 解码 → Whisper VAD 定边界(medium) → 隔 ≤0.7s 的相邻区间合并成 3–30s 候选
（超长的在最安静的一帧切开，不切在词中间）→ large-v3 逐条转写 + 语种过滤
→ 页面逐条试听标注 → 追加进 manifest.jsonl → 可选自动重新加工
```

```bash
# 需要 uv sync --group qc（faster-whisper + PyAV）；只在远程执行
uv run python -m voxft.data.ingest --input /root/autodl-tmp/drama/ep01.mp4 --max-items 20   # 试跑，不追加
uv run python -m voxft.data.ingest --input /root/autodl-tmp/drama/ep01.mp4                  # 切分+转写
uv run python -m voxft.data.ingest --input ep02.mp4 ep03.mp4                                 # 批量，素材 ID 取文件名
uv run python -m voxft.data.ingest --input ep01.mp4 --append --holdout ep01 --process --out drama_th_v2
```

产物结构：

```
data/raw/drama_th/manifest.jsonl        # 追加目标，加工读它
data/raw/drama_th/holdout.json          # {"sessions": ["ep01"]}：钉住的素材只进验证集
data/raw/drama_th/ingest/<素材ID>/
    source.wav                          # 解码后的 16k 单声道全轨（重切不必重解码）
    clips/0001_s0012.34_e0018.90.wav
    candidates.jsonl                    # 全量候选，含坏例与「丢弃/待定」；人工标注写回这里
```

`--source` 默认 `drama_tl`，做泰语要显式 `--source drama_th`。

要点：

- **Tagalog 系不能只认 tl 语种。** 短剧台词是 Taglish，句内英文词多的样本 Whisper 会判成 en。`Source.languages()` 对 tl 默认放行 `("tl","en")`，只认 tl 会把最该保留的 code-switch 样本全部误杀
- **转写可断点续跑**：每 100 条落盘，重跑跳过已转写行，坏例只排除不删除。`--max-items` 试跑不回写原清单。推理异常必须中止，不能当语料坏例吞掉
- **同一素材重切会替换它上次追加的行**（按 `ingest_video`），不会叠加成近似重复样本；再按音频绝对路径去重
- `session` 自动设为素材 ID
- **不做声源分离/降噪**，BGM 重的条目在试听环节判「丢弃」

#### 2.3.3 人工标注（这一步的质量直接决定情绪表现力上限）

在页面「素材导入」Tab 逐条试听并标注。标注入口的校验规则和加工阶段一致，别等加工时才炸：

- `verdict` ∈ 保留 / 丢弃 / 待定。**标「保留」必须有台词文本**
- 原始 `text` 必须是裸台词，以 `(` 或 `（` 开头会直接报错
- 勾了「已核实是本人」才写 `speaker_verified=true`，且必须提供真实 speaker ID（不能是空或 `default`）。**切分不出说话人，身份必须人工核实。** 只写 ID 不勾选 → 落成未验证身份 → 不配 ref、不调响度
- `control_zh` / `control_en` 只能中英文，含泰文字符会报错（`_assert_control_lang`）。**不要写目标语言的控制前缀** —— 线上 prompt 就是中英文

**情绪标签的两条路：**

1. `emotion` 填下表里的键 + `emotion_verified=true` → 自动从短语池随机取一条中/英文指令

   | emotion 键 | 中文 | 英文 |
   |---|---|---|
   | `neutral` | 平静地 / 语气自然 / 正常语调 | neutral tone / calm and natural / plain delivery |
   | `angry` `anger` | 愤怒地 | angry |
   | `happy` `happiness` | 开心地 | happy |
   | `sad` `sadness` | 伤心地 | sad |
   | `fearful` `fear` | 害怕地 | fearful |
   | `surprised` | 惊讶地 | surprised |
   | `frustration` `frustrated` | 烦躁地 | frustrated |
   | `disgust` | 厌恶地 | disgusted |

   另有 `rate_label ∈ {slow, fast}` 和 `volume_label ∈ {quiet, loud}`，需要 `control_verified=true` 才生效。

2. **短剧最需要的那些表演方式不在上面的池子里** —— 哭腔、带笑说话、讽刺、克制愤怒、咬牙切齿、气声。这些必须手写 `control_zh` + `control_en` 并置 `control_verified=true`，此时优先用你写的自由文本。

   ```jsonl
   {"control_zh":"带着哭腔，声音发抖","control_en":"tearful, voice trembling","control_verified":true}
   {"control_zh":"讽刺地，语速慢","control_en":"sarcastic, slow paced","control_verified":true}
   ```

   每条控制描述都必须有对应的真实录音支持，并且**经过试听核验**。「生气」不会被自动扩写成「大声喊叫」—— 那是伪造标签。

**不要凭能量/F0/空格数猜测表演。** 未知标签保持裸文本，比例不足就不足，不伪造。

### 2.4 钉住验证集

`split_records` 的随机分组结果依赖清单长度。**追加新素材后重新加工，上一轮的验证组会被整体重排进训练集** —— 已训 run 的评测结论随之作废。

挑一集写进 `data/raw/<source>/holdout.json`：

```bash
uv run python -m voxft.data.ingest --input ep05.mp4 --append --holdout ep05 --process --out drama_th_v2
```

或手写 / 页面填「钉进验证集的素材 ID」：

```json
{"sessions": ["ep05"]}
```

钉住的分组不参与 shuffle，永远只进验证集。`stats.json` 的 `holdout_pinned_records` 可核对。矛盾组合（钉住了却 `val_ratio=0`、或全部素材都被钉住）会直接报错，不会静默把钉住的数据喂进训练。

自有 acted / natural 两份清单如果共享演员或会话，**需在分源前统一安排 holdout** —— 混合器会拒绝跨源 train/val 泄漏。也可以先合并原始清单统一加工，接受两类语料内部的自然时长配比。

---

## Phase 3 · 逐源加工与检查

### 3.1 加工

先试跑，看丢弃统计是否符合预期，再全量：

```bash
uv run python -m voxft.data.pipeline --source thai_ser --max-items 200 --out thai_ser_probe
```

正式加工。**每次写新的音频子目录（`_v2` 后缀），不覆盖旧清单引用的音频** —— 旧产物不自动清理，确认无训练/混合清单引用后再人工清：

```bash
uv run python -m voxft.data.pipeline --source thai_ser      --out thai_ser_v2
uv run python -m voxft.data.pipeline --source yodas_th      --out yodas_th_v2
uv run python -m voxft.data.pipeline --source fleurs_th     --out fleurs_th_v2
uv run python -m voxft.data.pipeline --source aishell3      --out aishell3_v2

uv run python -m voxft.data.pipeline --source drama_th \
  --manifest /root/autodl-tmp/voxft_data/raw/drama_th/manifest.jsonl --out drama_th_v2
uv run python -m voxft.data.pipeline --source replay_en \
  --manifest /path/to/replay_en.jsonl --out replay_en_v2
```

`--source` 用 local 源（`drama_*` / `replay_en`）时必须配 `--manifest`。

加工链路：16k 重采样 → 裁首尾 → 时长过滤（3–30s）→ 质检 → 声学描述 → 身份/会话隔离切分 → 按说话人响度对齐 → 可信控制前缀 → split 内 ref 配对。

### 3.2 自动生效的策略（按 registry 分流，CLI 不暴露）

`options_for(source_id)` 按 registry 的 `expressive` 字段自动决定，**这是有意的，不要手工统一**：

| | 表演语料（`drama_*` / `thai_ser`） | 朗读语料（其余） |
|---|---|---|
| 首尾裁切 | 帧 RMS 门限 `0.02`（回落 `peak×0.01`） | **Silero VAD** 定边界（`edge_vad=True`） |
| 换气声 | **保留** —— 那是表演的一部分，裁掉模型就学不会换气 | 裁掉 |
| `control_ratio` | 0.5 | 0.25 |

其他默认：`ref_audio_ratio=0.5`、`ref_control_ratio=0.3`、`ref_min/max_dur=3/10s`、`val_ratio=0.02`、`val_max=200`、`target_dbfs=-24`、`control_zh_ratio=0.5`、`seed=42`。

CLI 可调：`--utmos-min` `--whisper-lang` `--control-ratio` `--asr-min-logprob` `--asr-max-no-speech` `--ref-audio-ratio` `--ref-control-ratio` `--val-ratio`。
`--min-snr-db` 只兼容旧参数，**启用会报错**。

### 3.3 检查 stats.json

产物：`$VOXFT_DATA_ROOT/processed/<out>/{train,val}.jsonl`、`stats.json`。

```bash
cat $VOXFT_DATA_ROOT/processed/drama_th_v2/stats.json | python -m json.tool
```

**丢弃计数**（`drop_*`）—— 数字异常大说明门限或数据有问题，不要直接往下走：

| 键 | 含义 | 异常时怎么办 |
|---|---|---|
| `drop_transcribe` | ASR 置信度不合格 | 只对 `needs_transcribe` 源生效。门限沿用 faster-whisper 解码器默认（时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`）。砍太多就 `--asr-min-logprob -1.5`。日志会打印原因分类 |
| `drop_decode` | 音频解码失败 | 检查容器/采样率；mp4/mkv 需要 PyAV |
| `drop_duration` | 不在 3–30s | 素材本身太碎，考虑调 ingest 的 `--min-dur` |
| `drop_lang` | 检测语种不符 | TL 源确认放行的是 `("tl","en")` |
| `drop_whisper` | 转写相似度 < `0.55` | 训练文本与音频对不上，该丢 |
| `drop_utmos` | UTMOS 音质分低 | UTMOS 权重源已失效，默认不启用 |

**覆盖统计**（`train` / `val` 各一份，来自 `dataset_summary`）—— 这几个数决定训练效果：

| 键 | 目标 | 不达标的后果 |
|---|---|---|
| `with_control / rows` | 25–50% | <25% 预检会警告，情绪指令跟随弱 |
| `with_ref_audio / rows` | 30–50% | <30% 预检会警告，克隆能力易被 LoRA 损害 |
| `with_ref_control / rows` | ~30% | ref + control 联合覆盖，最贴近生产形态的组合 |
| `cross_language_refs` | > 0 | 为 0 说明没有真正的跨语言同人 ref，见 [决策点 2](#决策点-2--跨语言同人-ref-是硬约束不是可以凑的字段) |
| `speakers` | 多名 | 只有 1–2 人会让模型把情绪绑到音色上 |
| `emotions` | 分布均衡 | 某情绪只来自某一名演员 = 学到的是音色不是情绪 |
| `max_exposure` | ≤ 3 | 超过说明混合重复过头 |
| `hours` | 对照 Phase 2.1 | 实际可用时长 |

联合覆盖目标（有足够已审核标签与身份时）：**ref+control 30% / ref+裸文本 20% / 无 ref+control 20% / 无 ref+裸文本 30%**。逐源加工后再混合，**最终覆盖不保证自动达到** —— 缺口用真实标注补，不要调低阈值凑。

`stats.json` 还记录本次用的 `edge_vad` / `edge_trim_ratio` 和完整 `options`，旧产物能反推当时门限。

### 3.4 响度对齐的边界

`target_dbfs=-24`：**仅对 `speaker_verified=true` 的说话人做整体增益**。防削波时整个说话人共同回退，保留条内与条间的相对动态。

- 禁止单条峰值归一化（会抹掉表演的强弱对比）
- 未知身份不统一调响度
- 上游已经逐条归一的（如 `yodas_th`）动态恢复不了，也不据此标音量指令

---

## Phase 4 · 混合与检查

### 4.1 混合

前提：所有参与混合的数据集名称都已完成 Phase 3 加工。

```bash
# TH
uv run python -m voxft.data.pipeline --out th_drama_v2 --mix \
  thai_ser_v2=25 drama_th_v2=20 yodas_th_v2=35 fleurs_th_v2=5 aishell3_v2=10 replay_en_v2=5

# TL
uv run python -m voxft.data.pipeline --out tl_drama_v2 --mix \
  drama_tl_v2=45 natural_tl_v2=30 filswitch_v2=10 aishell3_v2=10 replay_en_v2=5
```

权重是**目标时长占比**（相对值，不必凑满 100，但凑满便于对照）。`--mix` 必须配 `--out`，输出不能覆盖输入，输入不能重复。

混合规则：

- 按**有效音频时长**采样，不按条数
- 每条原始目标音频全局最多 **3×** 曝光（嵌套混合也检查），`max_repeat` 只能是 1/2/3
- ref 复用另行统计（`max_ref_exposure`），3× 上限只针对训练目标
- 验证集不重复采样
- 混合器会拒绝跨源 train/val 泄漏

**关键：目标重复 3× 再训练 3 epoch = 最多约 9 次曝光。所以先跑 1 epoch。**

### 4.2 检查 mix.json

```bash
cat $VOXFT_DATA_ROOT/processed/th_drama_v2/mix.json | python -m json.tool
```

- `parts` — 请求的权重
- `datasets[].actual_duration_share` — **实际时长占比**。和请求值差得多说明某个源不够量，被强行拉高了曝光
- `datasets[].max_exposure` — 该源的最大重复次数。接近 3 就是量不够
- `counts` / `train` / `val` — 同 Phase 3.3 的覆盖统计，这次是混合后的最终值

**这里的 `train.with_control` / `train.with_ref_audio` 就是预检要看的数**。不达标先回去补数据，不要直接开训。

---

## Phase 5 · 训练

### 5.1 生成配置

```bash
uv run python -m voxft.train.yaml_builder \
  --train /root/autodl-tmp/voxft_data/processed/th_drama_v2/train.jsonl \
  --val   /root/autodl-tmp/voxft_data/processed/th_drama_v2/val.jsonl \
  --base  /root/autodl-tmp/models/VoxCPM2 \
  --run   th_r1_e1 \
  --epochs 1 --gpus 1
```

输出 `configs/th_r1_e1.yaml` 和 `configs/th_r1_e1.plan.json`（记录卡数、epoch、等效 batch、训练条数，供预检比对）。

**`--base` 必须指向已下载的本地模型目录**，不能是 HF 仓库 ID —— 官方训练脚本要求本地目录。省略时用 `.env` 的 `VOXCPM_BASE_PATH`。

生成的 LoRA 配置（官方推荐 + 本项目默认，**不要随手改**）：

```yaml
learning_rate: 1.0e-4        # LoRA 用 1e-4；全量微调用 1e-5（1/10）
lora:
  enable_lm: true
  enable_dit: true           # 对音质至关重要，必开
  enable_proj: false
  r: 64                      # 语言+风格适配用 64；纯说话人适配 32 就够
  alpha: 64
  dropout: 0.05              # 情感语料体量小，0 容易几百步就过拟合到固定腔调
sample_rate: 16000           # AudioVAE 编码器输入，不是输出采样率
out_sample_rate: 48000       # 仅推理
batch_size: 2
grad_accum_steps: 8          # 等效 batch = 2 × 8 × GPU数
num_workers: 8
max_grad_norm: 1.0
weight_decay: 0.01
warmup_steps: <num_iters × 0.1>
save_interval: 250
valid_interval: 250
```

`epochs` 允许 1–3，`warmup_steps` 自动设为 `num_iters × 0.1`。**`epochs` 与显式 `num_iters`/`max_steps` 只能选一种**，同时给会报错。手动步数仅供受控实验。

**`training_cfg_rate=0.1` 不在训练 YAML 里。** 它属于基座 `config.json` 的 `dit_config.cfm_config`。往 overrides 里塞会直接报错，放在 YAML 顶层预检会拦。这是防过拟合忽略文本的关键项，确认基座里是 0.1。

**更改 GPU 数或训练清单后必须重新生成配置** —— `.plan.json` 会和实际比对，不一致预检报错。

### 5.2 预检

```bash
# 页面：训练 Tab → 预检
# 或直接在 Python 里调 launcher.preflight("configs/th_r1_e1.yaml", gpus=1)
```

预检会检查：官方训练脚本存在、清单每行合法 JSON、每条音频文件真实存在、每条有训练文本、ref 有已验证同人身份（`speaker_verified=true` 且 `ref_speaker == speaker`）、ref 不等于目标音频、**train/val 不共享音频/ref/说话人/会话**、基座 `training_cfg_rate=0.1`、`.plan.json` 与实际卡数条数一致、`save_path` 可写。

两条警告线：带前缀目标 <25%、同人 ref 覆盖 <30%。还有 `num_iters × effective_batch > 3 × 训练条数` 时警告超 3 epoch。

**警告不是可以忽略的形式主义。** 出现就回 Phase 2/3 补数据。

### 5.3 启动

预检通过后，在 GPU 机器上：

```bash
uv run python -m voxft.train.launcher configs/th_r1_e1.yaml 1
```

这条命令**打印**训练命令，复制到远程执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /path/to/third_party/VoxCPM/scripts/train_voxcpm_finetune.py \
  --config_path /path/to/configs/th_r1_e1.yaml
```

多卡：

```bash
cd third_party/VoxCPM && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=4 scripts/train_voxcpm_finetune.py --config_path <yaml>
```

也可以从页面「训练」Tab 直接启动（`launcher.start_local` 起后台子进程，日志写 run 目录下 `train.log`）。已有任务在跑时会拒绝再启动。

wandb 桥接自动转发 TensorBoard 指标，沿用原配置。

### 5.4 监控

- **不同实验用不同 run 名，不覆盖旧 run。** 命名带上轮次和变量：`th_r1_e1`、`th_r2_ctrl50_e1`
- 同配置恢复会从 `latest/` 继续
- 每 250 步保存 + 验证，训练结束官方脚本也保存一次
- **不以 loss 最低作为选 checkpoint 的依据。** 情感语料的 loss 和听感经常不同向
- **训练结束会自动只保留最新 5 次 LoRA 运行**（`cleanup_lora_runs(keep=5)`，按 `latest/` 修改时间排序，全量微调目录不受影响）。**好的 run 及时备份出去**，否则会被后续实验挤掉

---

## Phase 6 · 验收

### 6.1 自动指标（诊断，不是结论）

```bash
uv run python -m voxft.eval \
  base \
  $VOXFT_CKPT_ROOT/th_r1_e1/step_750 \
  $VOXFT_CKPT_ROOT/th_r1_e1/latest \
  --texts-file eval/th_holdout.jsonl --seeds 42 43 44
```

第一个位置参数是 `base`（纯基座）或若干 LoRA checkpoint 目录。多个 target 会依次跑并在最后打印对比表。

报告写到 `$VOXFT_CKPT_ROOT/eval/<label>_<uuid>.json`，含逐条：`hyp`（ASR 转写）、`similarity`、`cer`、`wer`（仅 tl/en）、`suspected_truncation`、韵律描述、`wav` 路径、`gen_sec`，以及待填的 `human_review`。

汇总看 `mean_cer`、`suspected_truncation_rate`、`mean_similarity`。

**读指标的正确姿势：**

- **ASR 误差 ≠ 发音错误。** Whisper 对泰语声调和 Taglish 本身就有误差。CER 只能看趋势和相对差异，不能当绝对门限
- **疑似漏尾 ≠ 真实截断。** 判据是 `len(hyp) < 0.6 × len(ref)` 或尾部 8 字符相似度 <0.5，ASR 同义转写也会触发。**必须听音确认**
- 泰语归一化保留声调/元音组合符（Unicode M 类），Taglish 不强制单一 ASR 语言（`language=None`）
- CER/WER 可以 >1（插入错误），不截断
- **F0/能量仅描述，不是越高越好**

评测时 `retry_badcase=False` —— 禁用自动换种子重试，避免掩盖差异。普通试听保持 `retry_badcase=True`。

### 6.2 A/B 对比

页面「试听」Tab。`synthesize_ab` 用**同一个模型**禁用/启用适配器，固定 seed，严格读 checkpoint 的 `lora_config`（官方 JSON 的嵌套层），检查 `load_lora` 返回的 `(loaded, skipped)` 和 missing keys：**loaded 非空、skipped 为空、missing 为空**，任一不满足直接抛错停止，绝不伪装成 LoRA 输出。

A/B 是验收「微调是否更听指令」的核心手段：同一 control 前缀，基座 vs LoRA。

### 6.3 母语盲听（唯一的通过判据）

至少两名母语评审，随机盲听同条件 A/B，四项分别 1–5 分：

| 字段 | 评什么 |
|---|---|
| `naturalness_1_5` | 自然度、去念稿感、对白韵律 |
| `emotion_fit_1_5` | 情绪匹配度、是否真的跟随了控制前缀 |
| `intelligibility_1_5` | 清晰度、发音准确、有没有吞字漏字 |
| `speaker_similarity_1_5` | 克隆音色与 ref 的相似度 |
| `cutoff` / `noise` | 真实截断 / 噪声，布尔标记 |

**分组看结果**：按 ref 语言（中/英/目标语）、角色（女主/其他女声/男声）、情绪类型、以及 `kind=clarity` 分别统计。整体均值会掩盖「男声崩了女声好了」「普通句没问题但长句和快语速糊了」这两种情况 —— 后者正是清晰度回归的典型形态，见[清晰度回归的判定](#清晰度回归的判定)。

**通过标准：自然度/情绪改善，且清晰度与音色不退化。** 两个条件都要满足。情绪涨了但音色漂移，不算通过。

自动报告**不生成「通过」结论**，结论由人写。

### 6.4 实验顺序

| 轮次 | 变量 | 目的 | 判据 |
|---|---|---|---|
| **R0** | 无 | 基座固定评测基线；旧 LoRA 若可用也测一次 | — |
| **R1** | 修正后的高质量目标语料 + 中英回放 | 1 epoch，确认发音/漏词**没有退化** | CER 不升、intelligibility 不降 |
| **R2** | 加入真人表演 + 可信中英控制标签 | 相同训练预算，重点比自然度和情绪 | naturalness / emotion_fit 上升 |
| **R3** | 补跨语言同人 ref 配对 | 比较中/英文 ref → TH/TL 的音色保持 | speaker_similarity 不低于基座 |

**每轮从同一基座开始，尽量固定总训练时长与采样曝光，只改变要检验的那一个数据因素。** 同时改两个变量就没法归因了。

R3 没有真实同人跨语言素材时**不要声称完成该验证** —— 记为「未验证」，而不是「通过」。

### 6.5 选 checkpoint

达标后才试第二个 epoch。出现内容服从、情绪或音色退化就**选择更早的 checkpoint**（`step_250` / `step_500` / …），不以 loss 最低为依据。

用同一份评测集把中间 checkpoint 都跑一遍再决定，不要只看 `latest`。

---

## Phase 7 · 迭代调参：症状 → 动作

**一次只改一个因素**，改完重跑 Phase 4→6。表里「先看什么」是归因步骤，跳过它很容易改错地方。

| 症状 | 先确认 | 动作 |
|---|---|---|
| 发音不准、重音/声调怪 | eval CER 趋势 + 盲听 `intelligibility` | 加发音锚点源占比（`fleurs_*` / `porjai_th` / `filswitch`）；确认**训练文本准确** —— ASR 转写错误会直接教坏发音，高价值子集要人工校对 |
| 念稿感、平读、不像对白 | 表演语料 `actual_duration_share` 是否到了 45% | 加 impro / 短剧占比；确认表演源 `edge_vad=False`（换气被裁掉就学不会呼吸）；朗读源占比过高会拉回念稿感 |
| 情绪前缀不跟随 | A/B 同 control 前缀 base vs LoRA；`with_control / rows` | <25% → 补可信标签（不是伪造）；检查 `with_ref_control` 是否接近 30%；确认线上推理走的是 reference-only 模式 |
| 带前缀变好但无前缀变差 | eval 里 `kind=plain` 的 case | `control_ratio` 太高，降到 0.25–0.35，保住裸文本推理路径 |
| 哭腔/带笑/讽刺学不出来 | 这些不在自动短语池里 | 必须手写 `control_zh`+`control_en` 并置 `control_verified=true`，且要有对应真实录音 |
| 某种情绪 = 某个人的音色 | `stats.json` 的 `emotions` × `speakers` 交叉 | 补其他演员的同类情绪；情绪覆盖必须跨演员 |
| 音色漂移、克隆变差 | 盲听 `speaker_similarity` + A/B | ref 覆盖是否 <30%；`enable_dit` 是否 true；epoch 是否过多；先回退到更早 checkpoint；仍不行则 `r` 从 64 降到 32 |
| 中文/英文能力退化 | 评测集里的 `antiforget` case | 回放占比提到 10% zh + 5% en；确认 `aishell3` 正文没混入拼音（旧清单必须重加工） |
| 生成停不下来、尾巴拖长 | `suspected_truncation_rate` + 听音确认 | 尾静音必须 <0.5s；朗读源确认 `edge_vad=True`（RMS 门限单独用有悬崖，见附录 A） |
| 漏词、吞字 | `cer`/`wer` + 盲听 | 训练文本质量；提高 ASR 置信度严格度（`--asr-min-logprob` 调到 -0.9）；针对具体难词补真实语料 |
| **吐字软、含糊、发闷**（不是漏词，是音不清） | 线上 `inference_timesteps` 是 20 还是 10；`cfg_value` 是否偏离 2.0 | 先修推理参数（最便宜）；再看是否过训练 → 回退更早 checkpoint；确认 `enable_dit=true` |
| **句尾辅音 / 尾音被截断** | 训练音频的 `tail_keep`；泰语句尾塞音、Tagalog 收尾辅音与喉塞音 | 确认 `tail_keep=0.3` 没被改；`suspected_truncation` 高的条目听音确认是数据问题还是生成问题；补句尾辅音清晰的锚点源（`porjai_th` / `fleurs_*`） |
| **长句后半段变糊** | eval 里 `kind=clarity` 长句 case 的 CER 分布 | `max_batch_tokens=8192` 的长度过滤让长样本在训练里欠曝光 → 提高长句（20–30s）在语料里的占比；推理加 `inference_timesteps` |
| **快语速时清晰度崩** | `语速快` 前缀的 case | 训练里缺快语速的清晰样本 → 补 `control_verified=true` + `rate_label=fast` 的真实条目，不要用加速音频伪造 |
| **某个 step 之后音质突然劣化** | train.log 的 loss 曲线有没有尖刺 | 梯度尖峰。回退到尖刺前一个 checkpoint；确认 `max_grad_norm=1.0` 没被关掉 |
| 有杂音、BGM | 盲听 `noise` | 素材导入阶段判「丢弃」。没有降噪链路，不要试图在训练侧修 |
| loss 降但听感变差 | — | 不以 loss 为准，选更早 checkpoint |
| 几百步就固定成一个腔调 | `dropout` 是否为 0 | 保持 0.05；减少 epoch；检查 `max_exposure` 是否接近 3 |
| 语料加了但效果没变 | `mix.json` 的 `actual_duration_share` | 新源量太小被稀释了；或者被 3× 重复撑起来的，看 `max_exposure` |

### 超参数调整顺序

按这个优先级动，前面的没排除不要动后面的：

1. **数据**（占比、标签质量、ref 覆盖、语料量）—— 90% 的问题在这里
2. **epoch / checkpoint 选择** —— 1 → 更早的 checkpoint → 2 → 3
3. **`control_ratio`**（0.25 ↔ 0.5）
4. **LoRA `r`**（64 ↔ 32）—— 语言+风格用 64，纯说话人适配 32 够；降 r 可以减轻对克隆能力的损害
5. **`dropout`**（0.05 ↔ 0.1）
6. **`learning_rate`**（1e-4，最多在 5e-5 ↔ 2e-4 之间试）
7. **`batch_size` / `grad_accum_steps`** —— 改了要重新生成配置，等效 batch 变化会连带影响 lr

`enable_dit: true` **不要关**，那是音质的关键。`training_cfg_rate=0.1` 不要动。

### 推理侧参数

生产/试听默认（OmniVoice 生产实践对齐）：

| 参数 | 值 | 说明 |
|---|---|---|
| `cfg_value` | 2.0 | 音频坏例重试时降到 1.2–1.6 |
| `inference_timesteps` | 20 | 生产基线是 10，本项目默认 20 换更高音质；坏例时加步数 |
| `retry_badcase` | True | `max_times=3`、`ratio_threshold=6.0`。**A/B 和批量评测必须关掉** |
| `load_denoiser` | False | 去噪器依赖 modelscope，试听不需要 |

**两个必须一致的地方：**

1. **推理侧 LoRA 配置与训练完全一致**（官方 JSON 的 `lora_config` 嵌套层）。`infer.get_model` 会读 checkpoint 里的配置，加载不完整直接抛错
2. **带控制前缀时必须走 reference-only 模式**：只传 `reference_wav_path`，**不要传 `prompt_text`**。combined 模式会拼成 `prompt_text + "(控制)正文"`，前缀跑到句中就失效了。`infer._gen_kwargs` 已按此处理（有 control 时不传 `prompt_text`），生产链路要照抄这个逻辑

控制前缀格式就是文本前缀，没有独立条件通道：`(控制指令)正文`。`clean_control` 会清掉指令里的括号，避免破坏格式。

---

## 清晰度专项 · 全链路控制点

清晰度不是一个开关，它在五个阶段各有独立的损耗点，任何一处漏了都会在最终音频上表现为「糊」。而且它和另外两个目标**直接冲突**：

- 与「裁静音」冲突 —— 门限调高会啃掉词首清辅音
- 与「保留换气」冲突 —— 表演源不裁，底噪和口水音就留在训练数据里
- 与「情绪强度」冲突 —— 喊叫/耳语动态范围大，要么削波要么听不清

所以要按阶段逐个控制点检查，**不要指望找到某一个参数把它调好**。

### 战线 1 · 数据入口（决定上限，权重最大）

没有降噪链路，糊的录音进去就是糊的模型出来。

| 控制点 | 位置 | 怎么做 |
|---|---|---|
| 只收干净录音 | 素材导入试听 | 本项目**不做声源分离/降噪**（无 demucs 依赖，官方 zipenhancer 需 modelscope 且不在 lock 里）。BGM、混响、口水音重的条目直接判「丢弃」，不要试图用训练修 |
| 上游音质分级 | registry `row_filters` | `yodas_th` 已有 `dnsmos_overall ≥ 3.2` + `grade_avg ∈ {S+, S}`。这是现成的清晰度过滤，**不要放宽** |
| UTMOS 音质过滤 | `--utmos-min 3.5` | 实现在 `qc/utmos.py`（移植自 OmniVoice，Apache-2.0）。**当前权重源已失效，默认不启用**（所有源 `qc` 都不是 `full`）。能拿到权重就给朗读源加一道；表演源慎用 —— UTMOS 偏好录音棚音色，会把有呼吸感的表演判低 |
| 训练文本与音频一致 | `drop_whisper` | 相似度门限 0.55。**文本错了就是教模型错误发音**，危害比音频糊一点更大 |
| 含糊音过滤 | `drop_transcribe` | `needs_transcribe` 源：时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`（faster-whisper 解码器自己的默认）。清晰度不达标就 `--asr-min-logprob -0.9` 调严 |
| 语种过滤 | `drop_lang` | 混入其他语种会教错音系。TL 放行 `(tl, en)` 是为保 Taglish，不是放松标准 |
| 不拼接孤立词 | `filipino_speech` 过滤 | 中位 0.63s / `num_words` 中位 1。拼接出的伪句词间韵律错乱，直接损伤可懂度 |
| 发音锚点源 | 混合占比 5–10% | TH 首选 `porjai_th`（录音棚级标准泰语，700h，TTS 专用）；`fleurs_*` 干净朗读；TL 用 `filswitch` 教句内英文词与数字读法。**锚点源是朗读体，占比过高带来念稿感** —— 5–10% 是上限不是起点 |
| 人工校对高价值子集 | 手工 | 表演语料的转写错误率高于朗读。至少把要进评测集的难词、以及情绪最强那批条目人工过一遍 |

### 战线 2 · 首尾裁切（清晰度最容易在这里被误伤）

`trim_silence` 的门限是三重保护，每一重都对应一个清晰度事故：

```python
thr = max(peak * floor,                    # floor=1e-3，约 -60dB 绝对下限
          min(noise * 3.0,                 # 上限：底噪(帧RMS p10) × 3
              max(voiced * edge_ratio,     # 相对有声电平(帧RMS p99)
                  peak * 0.01)))           # voiced 退化时的回落
```

- **`noise * 3.0` 这个上限是清晰度的保险丝**：几乎没有静音的条目上，p10 就是轻声语音本身。不设上限会把弱辅音整段裁掉
- **`min_run=0.25` 是第二个保险丝**：首尾低电平段必须**连续** ≥0.25s 才裁。词首清辅音 /s/ 只有 80–120ms、/h/ 只有 40–80ms，都比 0.25s 短，所以不会被误啃。**调低 `min_run` 或调高门限会直接损伤清晰度 —— 这两个值不是可以随手优化的旋钮**
- 边界按**帧 RMS** 判定而非逐采样点：一声口水音的单个尖峰不该把整段空白留下
- 众包语料底噪高，只用相对峰值的固定门限（−60dB）经常整条裁不动 —— 这就是要三重取大的原因

**`tail_keep=0.3`**：尾部最多留 0.3s（官方要求 <0.5s，防「生成停不下来」）。裁太狠会截断尾音 —— 泰语的句尾塞音、Tagalog 的收尾辅音与喉塞音（正字法里常不写出来，最容易被忽略）最先丢；留太多会训出拖沓。0.3 是这个权衡的落点，不要改。

**表演源 / 朗读源的分流本身就是清晰度决策**（由 registry 的 `expressive` 字段自动决定）：

| | 表演源（`drama_*` / `thai_ser`） | 朗读源 |
|---|---|---|
| `edge_vad` | False | True |
| `edge_trim_ratio` | 0.02（回落 `peak×0.01`） | 0.06 |
| 换气 / 呼吸声 | **保留** —— 裁掉模型就学不会换气 | 裁掉 |
| 代价 | 底噪和口水音也留着 | RMS 门限有悬崖（附录 A），必须走 VAD |

某个表演源底噪实在重时，正确做法是**在素材导入阶段丢弃那些条目**，不是提高全局裁切门限 —— 后者会啃掉所有条目的词首辅音。

### 战线 3 · 响度与削波

`apply_speaker_gain(target_dbfs=-24)`：

- 对同一说话人施加**同一个**增益，使其响度中位数落在 −24 dBFS，**条与条之间的相对强弱完整保留**
- 防削波：`gain = min(gain, 0.97 / peak)`，peak 取该说话人**所有条目**的最大值 —— 所以是整个说话人共同回退。0.97 的余量就是防削波用的
- **削波是不可逆的清晰度损失**，写进训练音频就修不回来
- 逐条峰值归一会把喊叫和耳语拉到同一响度，抹掉「音量 = 情绪强度」这条线索 —— 这正是情感语料训完仍然平淡的原因之一；而且耳语被放大时底噪一起被放大
- **只对 `speaker_verified=true` 的说话人生效**（未验证组可能混人）。未知身份不调响度
- 上游已经逐条归一的（如 `yodas_th`）动态恢复不了，也不据此标音量指令

### 战线 4 · 训练配置

| 项 | 值 | 与清晰度的关系 |
|---|---|---|
| `enable_dit` | **true** | 官方明确「对音质至关重要」。不要为省显存关掉 |
| `sample_rate` | 16000 | **AudioVAE 编码器输入**，不是输出采样率。与 `out_sample_rate=48000` 搞混会导致训练/推理不匹配，输出发闷发糊 |
| `max_grad_norm` | 1.0 | 情感语料动态大，不裁剪更容易出梯度尖峰。尖峰会让音质在某个 step 之后突然劣化 —— **看到 loss 尖刺就往前一个 checkpoint 回退** |
| `batch_size` | 2 | 官方示例值。音频序列长、激活显存大，**不要调大**（OOM 会逼你降 `max_batch_tokens`，那才是真正的损失） |
| `max_batch_tokens` | 8192 | 长度过滤。超长样本被官方 loader 丢弃/截断，**实际训练分布和你以为的不一样** —— 以日志里的实际样本数为准 |
| 曝光次数 | 先 1 epoch | 混合 3× 重复 × 3 epoch = 最多 9 次曝光。过训练会过拟合到固定腔调，同时伴随高频细节丢失（听感就是「糊」） |

### 战线 5 · 推理侧（不用重训就能拿到的清晰度）

最容易被忽略，但**收益最快**。

| 参数 | 默认 | 清晰度影响 |
|---|---|---|
| `inference_timesteps` | **20** | 生产基线是 10，本项目默认 20 换更高音质。**这是最便宜的清晰度提升**，代价是推理时间翻倍。先确认线上到底用的是 20 还是 10 |
| `cfg_value` | **2.0** | 过高 → 发紧、失真、金属感；过低 → 含糊、吐字软。坏例重试时降到 1.2–1.6 并加步数 |
| `out_sample_rate` | 48000 | 仅推理输出，与训练的 16000 不冲突 |
| `load_denoiser` | False | 去噪器依赖 modelscope，不在 lock 里。**别指望它修训练数据的问题** |
| `retry_badcase` | True（3 次 / ratio 6.0） | 生产兜底。但**评测时必须关掉**，否则换种子会掩盖清晰度问题 |
| ref 模式 | reference-only | 带 control 前缀时**不要传 `prompt_text`**。combined 模式前缀跑到句中失效，情绪不对会被感知成「不清楚」 |

**如果线上清晰度不如试听页**，先逐项比对这六个参数，再怀疑模型 —— 十有八九是 `inference_timesteps` 或 `cfg_value` 不一致。

### 清晰度回归的判定

Phase 6 的 `intelligibility_1_5` 必须**单独打分**，不能和 naturalness 混在一起。三种常见误判：

- **自然度涨了、清晰度掉了 → 不算通过。** 念稿感消失但吐字变软，是很典型的过拟合表现
- **只看 `mean_cer` 会漏。** Whisper 对糊音常常「脑补」出正确文本，CER 看不出来，人耳一听就知道
- **只看总均值会漏。** 按 `kind=clarity` 分组、按 ref 语言分组看 —— 中英 ref 通常比目标语 ref 更容易吐字不清

---

## Phase 8 · 导出与上线

### 8.1 merge LoRA

```bash
uv run python -m voxft.lora.merge \
  --lora-dir $VOXFT_CKPT_ROOT/th_r2_e1/step_1000 \
  --out /root/autodl-tmp/merged/th_r2_e1_s1000
# --base 省略时用 .env 的 VOXCPM_BASE_PATH
```

支持官方嵌套 `lora_config`。页面「模型管理」Tab 同样能做。

### 8.2 许可红线：含 CC-BY-SA 数据的权重不得对外分发

**这条在上传/交付任何权重之前先看。**

我们在用的两个最好的泰语源都是 **CC-BY-SA-4.0**：`thai_ser`（THAI-SER）和 `Porjai-central`。Porjai 的 `pattani` / `khummuang` 子集更差，是 **CC-BY-NC-SA**，直接排除。

- SA 的触发条件是**「向公众分享改编物」**。模型权重算不算「改编物」**在法律上无定论、无判例**；CC 官方说过「模型若基于 SA 内容训练且公开发布，建议以同许可发布」，但那是**保守合规建议，不是法律要求**
- **红线：含 SA 数据训练的 LoRA 与 merge 后的完整模型一律不对外分发** —— 不传 HF、不随客户交付、不开源。只通过 API 交付合成音频。SA 不追及模型输出（除非输出实质复现了原音频，TTS 不会）
- 确实需要对外发布权重时，两条路：① 向版权方谈商业授权 —— **THAI-SER 的出资方是 AIS + DEPA，有明确的谈判主体**（VISTEC / airesearch）；② 该实验只用 Apache-2.0 / CC-BY 源（如 `speechcolab/gigaspeech2` th、`aishell3`）
- EU DSM 指令第 4 条 TDM 例外、日本著作权法 30-4 条、美国 fair use 任一成立时 CC 条件可被架空，且**单纯挂 CC 许可本身不构成 TDM 保留**。但这属于法务判断，不要自己下结论

**以上是保守合规立场，不是法律意见。重大决策请咨询法务。**

### 8.3 上传 HF

```python
from voxft.hub import sync
sync.upload_folder("/root/autodl-tmp/merged/th_r2_e1_s1000", "<org>/<repo>", kind="model")
```

或页面上传功能。**只有训练与盲听实际通过后才发布新权重**，且必须先过 [8.2](#82-许可红线含-cc-by-sa-数据的权重不得对外分发) 的许可检查。

### 8.4 交付前最后一遍

- [ ] **训练数据里没有 CC-BY-SA / NC 源；如果有，权重不对外分发，只走 API**（见 8.2）
- [ ] 用**生产链路的实际调用方式**（不是本项目试听页）跑一遍评测集，确认参数一致 —— 特别是 reference-only 模式
- [ ] **`kind=clarity` 分组用生产参数复跑过，`intelligibility` 没有相对基座退化**（试听页达标不代表线上达标，多半是 `inference_timesteps` 不同）
- [ ] 中英 ref → 目标语言的音色保持已 A/B 实测，没有相对基座退化
- [ ] 中英回放 case 没有灾难性遗忘
- [ ] 选定的 checkpoint 已备份（`cleanup_lora_runs` 只保留最新 5 次）
- [ ] 评测报告、盲听评分、`stats.json`、`mix.json`、`.plan.json` 一起归档，下一轮要对照

---

## 附录 A · 坑速查

**首尾裁切的 RMS 悬崖**：RMS 门限是相对该条「有声电平」（p99 帧 RMS）的。留白电平只要在有声电平 −24dB 以内（朗读档 `0.06` 的门限），`rms > thr` 覆盖整条、**一帧都裁不动，输出长度与输入完全一致** —— 看起来就是「裁静音没生效」，其实是生效了但门限没跨过。FLEURS 那个问题就是这个。所以朗读源改用 Silero VAD（faster-whisper 自带，无需额外权重），不受电平影响，实测首尾误差 ±0.04s。

**`min_silence_duration_ms` 用 500，不是 faster-whisper 默认的 2000**。默认值是给长音频分段设计的，会把不足 2s 的尾部底噪并进语音块，实测多留 1.0s。

**喂 VAD 的数组必须 float32**。float64 抛 ONNX `Unexpected input data type`，中断整轮加工。

**首尾低电平段必须连续 ≥0.25s 才裁**（`trim_silence(min_run=)`）。否则提高门限会啃掉词首清辅音（/s/ 只有 80–120ms）。

**Gradio 流式**：按钮必须直接绑定生成器函数。用 `lambda` 包一层会把生成器对象本身渲染进文本框。

**Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。运行后变化的列表必须通过事件输出或 `Tab.select` 刷新。

**库函数不许 print**：UI 进程的 stdout 可能是已断开的 pty，`print` 抛 `[Errno 5] Input/output error`，会把一次**已经成功**的操作报成失败（filswitch 写完 2709 条清单后显示「失败」就是这个）。一律走 `progress` 回调，CLI 侧传 `progress=print`。

**`hf` 走镜像必须关 xet**：hf-mirror 只代理 HF API，不代理 xet 的 CAS 服务器（`cas-server.xethub.hf.co`），reconstruction 阶段会直连并报 401。项目入口自动处理，命令行要自己 export `HF_HUB_DISABLE_XET=1`。

**`datasets` 锁定 `<4`**：5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突。

**Common Voice 22 官方已从 HF 撤架**：数据源已切到社区镜像 `fsicoli/common_voice_22_0`（非 gated，脚本式数据集，流式下载带 `trust_remote_code`）。

**`thai_ser` 没有名为 `audio` 的列**（四路麦 `mic_clip`/`mic_con`/`mic_middle`/`mic_zoom`），必须靠 registry 的 `audio_cols` 映射，否则整个源在下载阶段被静默跳过。`mic_zoom` 是网络录音，不用。

**`yodas_th` 的 session**：`utt_id.rsplit("-", 3)[0]` 保留可能含 `-` 的完整视频 ID。`speaker_id` 是视频级近似身份，不作 ref 依据。无原始连续时间关系就不拼接短句。

**视频容器解码走 PyAV**：soundfile 读不了 mp4/mkv，qc 组已显式声明 `av>=12`（本来就是 faster-whisper 的传递依赖）。不引入系统 ffmpeg 依赖，本地 macOS 与远程行为一致。

**`third_party/VoxCPM/` 是官方 submodule，只读，禁止直接修改。** 训练入口是其内 `scripts/train_voxcpm_finetune.py`。升级：

```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```

**旧数据迁移**：所有旧 processed/mixed 清单重加工为新名称（`_v2`），不要混用旧伪说话人 refs、拼接音频或拼音污染正文。旧 THAI-SER manifest 缺 `turn_type` 会明确报错，需在远程重下。原能量差字段改名 `energy_range_db`，启用旧 `min_snr_db`/`min_f0_std` 硬筛会报错。

---

## 附录 B · 全流程 checklist

### Phase 0 环境
- [ ] `uv sync --group qc` 成功，`uv run pytest` 全绿
- [ ] submodule 已初始化（`third_party/VoxCPM/scripts/train_voxcpm_finetune.py` 存在）
- [ ] `.env` 的 `VOXFT_DATA_ROOT` / `VOXFT_CKPT_ROOT` / `HF_HOME` 都指向大盘
- [ ] large-v3 与基座已预取到本地
- [ ] UI 用 `nohup`/tmux 启动，stdout 没挂在会断的终端上

### Phase 1 评测集
- [ ] 80–100 条 / 语言，未见演员未见会话
- [ ] 覆盖矩阵齐全（ref 语言 × 前缀 × 角色 × 语体 + 中英回归）
- [ ] **`kind=clarity` 的压力用例占 20–30%**（长句 / 难词 / 数字英文混排 / 快语速 / 耳语 / 句尾辅音）
- [ ] 母语者已确认台词与预期读法
- [ ] R0 基线报告已生成并归档

### Phase 2 数据
- [ ] 每个源先 `--max-samples` / `--max-items` 试跑
- [ ] `thai_ser` 只留 impro + agreement ≥0.7，不用 `mic_zoom`
- [ ] `filswitch` 音频真的下载到了（不是「分片下载成功」就算）
- [ ] 自建清单的 `speaker_verified` 是人工核实过的，不是「有一列 speaker」
- [ ] `session` 填了，同一集的切片同 session
- [ ] 中英同人 ref 以 `reference_only=true` 独立行导入
- [ ] 哭腔/带笑/讽刺等复合指令手写了 `control_zh`+`control_en` 且 `control_verified=true`
- [ ] 控制前缀只有中英文，没有目标语言
- [ ] holdout 已钉住（`holdout.json`）

### Phase 3 加工
- [ ] 输出用新名称（`_v2`），没覆盖旧音频
- [ ] `drop_*` 计数已逐项看过，没有异常大的
- [ ] 表演源 `edge_vad=false`、朗读源 `edge_vad=true`
- [ ] **抽听 10 条产物，词首清辅音（/s/ 80–120ms、/h/ 40–80ms）没被裁掉，句尾辅音完整**
- [ ] **响度对齐后抽听，喊叫条目没有削波（`gain ≤ 0.97/peak` 生效）**
- [ ] `with_control / rows` ≥ 25%
- [ ] `with_ref_audio / rows` ≥ 30%
- [ ] `cross_language_refs` 的实际值已记录（为 0 就明确标注「跨语言 ref 未验证」）
- [ ] `emotions` × `speakers` 没有「某情绪只来自某一人」
- [ ] `holdout_pinned_records` 与预期一致

### Phase 4 混合
- [ ] `actual_duration_share` 与请求占比的差异可解释
- [ ] `max_exposure` ≤ 3
- [ ] train/val 无跨源泄漏（混合器已拒绝，确认没报错）
- [ ] 混合后的 `with_control` / `with_ref_audio` 仍达标
- [ ] 发音锚点源在 5–10% 区间（够修发音，又不至于带回念稿感）

### Phase 5 训练
- [ ] 基座 `dit_config.cfm_config.training_cfg_rate == 0.1`
- [ ] `enable_dit: true`
- [ ] `sample_rate: 16000` / `out_sample_rate: 48000` 没被搞混
- [ ] 预检无 error，两条警告已处理或已记录原因
- [ ] run 名带轮次和变量，不覆盖旧 run
- [ ] `.plan.json` 与实际 GPU 数、清单条数一致
- [ ] 1 epoch 起步
- [ ] 好的 checkpoint 已备份（自动清理只留最新 5 次）

### Phase 6 验收
- [ ] 同一份评测集、同样的 seeds、`retry_badcase=False`
- [ ] LoRA 加载校验通过（loaded 非空、skipped/missing 为空）
- [ ] A/B 是同模型禁用/启用适配器，不是两次独立加载
- [ ] ≥2 名母语评审盲听，四项分数都填了
- [ ] 按 ref 语言 / 角色 / 情绪分组看过，不只看总均值
- [ ] **`intelligibility_1_5` 按 `kind=clarity` 单独分组看过，没有和 naturalness 混判**
- [ ] 中间 checkpoint 也评过，不只是 `latest`
- [ ] 结论是「自然度/情绪改善 **且** 清晰度/音色不退化」才判通过

### 清晰度专项（逐条对照正文五条战线）
- [ ] **入口**：BGM / 混响 / 口水音重的条目在素材导入阶段已丢弃，没有指望训练修
- [ ] **入口**：`yodas_th` 的 `dnsmos_overall ≥ 3.2` + `grade_avg ∈ {S+, S}` 没被放宽
- [ ] **入口**：高价值子集（评测集难词、强情绪条目）的训练文本已人工校对
- [ ] **裁切**：`min_run=0.25` 和 `tail_keep=0.3` 没被改动
- [ ] **裁切**：`thr` 的 `noise × 3.0` 上限还在（防止把轻声语音当底噪裁掉）
- [ ] **响度**：`target_dbfs=-24` 只对 `speaker_verified=true` 生效，未知身份没被统一调响度
- [ ] **训练**：`max_grad_norm=1.0` 没关，loss 曲线无未处理的尖刺
- [ ] **训练**：日志实际样本数与清单条数的差距可解释（`max_batch_tokens=8192` 长度过滤）
- [ ] **推理**：线上 `inference_timesteps=20`、`cfg_value=2.0`，与试听页一致
- [ ] **推理**：线上带 control 前缀时走 reference-only，没传 `prompt_text`

### Phase 7/8 迭代与交付
- [ ] 本轮只改了一个因素
- [ ] 生产链路用 reference-only 模式（带 control 时不传 `prompt_text`）已确认
- [ ] 生产链路六个推理参数逐项与试听页比对过
- [ ] merge 后的模型在生产调用方式下复测过
- [ ] 报告 + 评分 + stats + mix + plan 一起归档
