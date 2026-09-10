# 越南语(VI) / 印尼语(ID) 接入：设计依据、语料核实状态与流程差异

**2026-09 接入。** 核实标记沿用 [corpus_sourcing.md](corpus_sourcing.md)：`✅` = 拉到了本地文件/页面/API 原文；`~` = 只有间接线索，未二次核实，**不要写进采购单或对外承诺**；`❌` = 核实后确认不可用。

## 0. 一句话结论

- **基座本来就支持 vi/id，这不是「教模型一门新语言」。** VoxCPM2 官方 30 语种含 Vietnamese 与 Indonesian，且在其自有基准上 vi/id 的分数**优于已经在跑的 tl** ✅。所以 vi/id 的微调目标是**配音域风格 + 说话人 + 情绪指令跟随**，不是抢救发音。
- **submodule 零改动**，全部工作在外层工作台 ✅。基座代码里没有语言列表、language token 或 lang_id，tokenizer `byte_fallback=True`，vi/id 实测 0 UNK。
- **表现力语料是唯一的真缺口**，vi 和 id 与 Tagalog 处境相同：本轮**没有核实到任何**可商用的开源真人情感/表演语料 ~。首轮微调的验收目标必须降级（见 §5）。
- **许可比泰语干净**：本轮注册的 vi/id 源全是 Apache-2.0 / CC-BY-4.0 / CC0，**没有 CC-BY-SA 红线** ✅ → 含这些源训练的 vi/id 权重原则上可以对外分发，这是与 TH 的实质差别（对比 [corpus_sourcing.md §3.1](corpus_sourcing.md)）。

---

## 1. 基座能力依据（全部可在本地核对）

引用路径相对 `third_party/VoxCPM/`。

| 事实 | 出处 | 状态 |
|---|---|---|
| 30 语种列表含 Indonesian、Vietnamese | `README.md:56-57` | ✅ |
| 「无需语言标签，直接输入任一支持语言的文本」 | `README.md:47` | ✅ |
| 内部 30 语种 ASR 基准：**id WER 1.36%**、**vi WER 1.56%**、tl WER 2.63%、th CER 0.94% | `README.md:540,558,556,555` | ✅ |
| MiniMax-MLS-test WER：id 1.084、vi 3.307、th 2.961 | `README.md:471,483,480` | ✅ |
| MiniMax-MLS-test SIM（说话人相似度）：id 80.0、vi 80.6、th 84.0 | `README.md:504,516,513` | ✅ |
| HF 模型卡 frontmatter 显式列 `id` / `vi`（与 `th` / `tl` 并列） | 本地快照 `models--openbmb--VoxCPM2/.../README.md` | ✅ |
| 代码里不存在语言列表 / language token / lang_id；唯一语种分支是 CJK 判定 | `src/voxcpm/model/utils.py:40-138` | ✅ |
| manifest 与训练 YAML 字段里没有语种轴（`audio`/`text`/`ref_audio`/`dataset_id`/`duration`/`ref_duration`/`is_prompt`） | `src/voxcpm/training/data.py:13-16` | ✅ |
| 控制前缀 `(指令)正文` 是纯字符串拼接，与语种无关 | `src/voxcpm/cli.py:68-70` | ✅ |

**tokenizer 实测**（用 `LlamaTokenizerFast.from_pretrained` + `mask_multichar_chinese_tokens` 复刻 `src/voxcpm/model/voxcpm2.py:1144,186` 的真实调用路径）：

| 语言 | token/字符 | UNK | byte-fallback 占比 |
|---|---|---|---|
| vi | 0.786 | **0** | 32.6% |
| id | 0.366 | **0** | 0.0% |
| th（已跑通） | 1.989 | **0** | 73.8% |
| tl（已跑通） | 0.366 | **0** | 0.0% |

词表 73440 条、`byte_fallback=True`。越南语 74 个声调相关字符里 33 个是词表单字符条目，其余 41 个（`ế ộ ở ằ ắ ữ ự …`）走字节回退——**无损**，且泰语 73.8% 走字节回退都已经在跑，vi 严格更轻松。`max_length=8192`、`max_batch_tokens=8192` 下配音单句尺度无压力，不需要调参。

---

## 2. 本轮注册进来的源

`src/voxft/data/registry.py`。加进 registry 即自动生效：下载分派、`options_for()` 的加工分流（首尾裁切方式、控制前缀比例、ASR 置信度门限）、ingest 的 `--source` choices、UI 下拉与【首选】标记。

| 源 id | 语种 | 角色 | 许可 | 规模 | speaker ID | 状态 | 可用性 |
|---|---|---|---|---|---|---|---|
| `drama_vi` | vi | expressive | 按自有授权 | 自建 | ✅ 人工核实 | ✅ 通路已验证（与 `drama_tl` 同构） | **10/10** 唯一可信表现力来源 |
| `drama_id` | id | expressive | 按自有授权 | 自建 | ✅ 人工核实 | ✅ 同上 | **10/10** 同上 |
| `gigaspeech2_vi` | vi | anchor | Apache-2.0 | **未核实** | ❌ | ~ 覆盖 vi 已由本项目 `corpus_sourcing.md:127` 记录；**字段形态未核实** | 6/10 许可最干净，但要先试跑 |
| `gigaspeech2_id` | id | anchor | Apache-2.0 | **未核实** | ❌ | ~ 同上 | 6/10 同上 |
| `fleurs_vi` | vi | anchor | CC-BY-4.0 | **未核实**（FLEURS 各语种通常十余小时，但本轮未取原文） | ❌ | ~ config `vi_vn` 按 `{lang}_{country}` 模式推得 | 5/10 干净朗读，只能当发音锚点 |
| `fleurs_id` | id | anchor | CC-BY-4.0 | **未核实** | ❌ | ~ config `id_id` 同上 | 5/10 同上 |
| `cv22_vi` | vi | anchor | CC0 | **未核实**（该 locale 的 validated 小时数没取到原文） | 众包 `client_id` | ~ locale 码按 CV 惯例 | 3/10 众包噪音大，走 Whisper 校验 |
| `cv22_id` | id | anchor | CC0 | **未核实** | 众包 `client_id` | ~ 同上 | 3/10 同上 |

**为什么只注册这 8 个**：本轮网络受限，无法拉 HF API/数据卡原文。AGENTS.md 的红线是「低资源语种的调研结论必须核实到页面/API 原文」，所以凡是规模、许可、字段形态没取到原文的候选一律不进代码，只进 §3 的待核实清单。已注册的 8 个里，`drama_*` 是本仓库已跑通的 `local` 通路，另外 6 个都沿用本仓库 th/tl 已经在用的仓库与 config 命名模式——**注册的是可复现的路径，不是编造的数字**，所以每个源的 `note` 字段都写着「先 `--max-samples` 试跑」。

### 2.1 三个语种的 code-switch 策略（`Source.languages()`）

| 语种 | 转写/质检放行 | 理由 |
|---|---|---|
| tl | `(tl, en)` | Taglish，句内英文可过半，只认 tl 会误杀最该保留的样本 |
| **id** | `(id, en)` | 印尼语日常口语与短剧台词混英文程度接近 Taglish |
| **vi** | `(vi,)` 从严 | 混英以词内借词为主，Whisper 判 vi 较稳；**实测 `drop_lang` 误杀再放开**，改法见下 |

`fleurs_id` / `cv22_id` 用 `accept_langs=("id",)` **覆盖掉** id 的英文放行：它们是有权威文本的朗读语料，语种检测不符意味着错行，不是 code-switch。

放开 vi 的英文放行（确认误杀后再做，别提前）：
```python
# registry.py，给具体某个源加，不要改 CODE_SWITCH_ACCEPT 的默认值
Source("drama_vi", "vi", ..., accept_langs=("vi", "en")),
```

---

## 3. 待核实清单（远程有网时逐条做，做完把状态从 ~ 改成 ✅/❌）

这些是**候选**，不是结论。逐条给出核实命令；核到原文再决定是否进 registry。

```bash
# 通用：拉数据集元信息（看 cardData.license / gated / tags）
curl -s "https://huggingface.co/api/datasets/<repo>?full=true" | python -m json.tool | head -60
# 通用：拉数据卡原文（YAML frontmatter + 正文表格，最有用）
curl -sL "https://huggingface.co/datasets/<repo>/raw/main/README.md" | head -120
# 通用：确认 config/split 是否存在
curl -s "https://huggingface.co/api/datasets/<repo>/parquet" | python -c "import sys,json;print(list(json.load(sys.stdin)))"
```

| # | 待核实项 | 要确认什么 | 命令/入口 |
|---|---|---|---|
| 1 | `speechcolab/gigaspeech2` 的 vi/id | **字段形态**：是内嵌 `audio` 数组，还是 `path`+`start`+`end` 的长音频切片？后者本项目的 `_load_audio` 不支持（会整段读入，时长过滤会大面积丢弃） | `download --source gigaspeech2_vi --max-samples 20`，看日志的时长分布与 `audio` 列名；再 `curl .../parquet` |
| 2 | gigaspeech2 的 gated 类型与条款 | `gated: auto` 还是 `manual`；条款有没有限制商用 | `curl ".../api/datasets/speechcolab/gigaspeech2?full=true"` 看 `gated` |
| 3 | `google/fleurs` 的 `vi_vn` / `id_id` | config 名是否确切存在、各自小时数 | `curl ".../api/datasets/google/fleurs/parquet"` |
| 4 | `fsicoli/common_voice_22_0` 的 `vi` / `id` | locale 是否存在、validated 小时数、众包噪音水平 | 同上 + https://commonvoice.mozilla.org/datasets 的语言统计表 |
| 5 | **VIVOS**（越南语朗读，规模/说话人数/许可全未核实） | 现在托管在哪（HF / OpenSLR / 原发布方）、小时数、说话人数、**能否商用、能否用于训练模型** | HF 搜 `vivos`；OpenSLR 资源列表搜 Vietnamese |
| 6 | **YODAS2 / Sidon 整理版有没有 vi/id** | 泰语用的 `Chalermdej/yodas2_sidon_th_tts` 有没有 vi/id 对应版本（带 DNSMOS + 三路 ASR 分级，对本项目最合用） | `curl ".../api/datasets?author=Chalermdej&full=true"`；`curl ".../api/datasets?search=yodas2"` |
| 7 | **MagicHub 印尼语/越南语库** | 区分 scripted monologue 朗读（价值低）与自发对话（价值高）；免费档常是 **CC-BY-NC-ND**（NC 禁商用、ND 禁演绎 = 微调，**可用性 0**），有价值的那档通常要询价 | https://magichub.com/datasets/ 搜 Indonesian / Vietnamese |
| 8 | **Nexdata 越南语/印尼语现货** | SKU、小时数、**是否带 speaker ID + gender**、许可形态（商业买断？） | nexdata.ai 数据集列表页 |
| 9 | **HF 系统扫描**：有没有我漏掉的真人情感/对话语料 | 真人录音（排除 TTS 合成）、规模 ≥5h、许可可商用、最好带 speaker ID | `curl ".../api/datasets?language=language:vi&other=other:modality:audio&full=true&limit=100&sort=downloads"`，`vi` 换 `id` 再跑一遍 |
| 10 | 越南语/印尼语**情感语音**语料是否存在 | 这是决定「要不要自建」的关键结论，必须核到原文才能写死 | 同 #9，关键词加 emotion/expressive |

**核实纪律**（来自 `corpus_sourcing.md §1` 的教训）：这个领域出现过完全编造的论文和不存在的许可声明。**任何「有现成大规模语料」的说法，在拉到页面/API 原文之前都当作不存在**——一条编造的结论足以让人跳过真正该做的自建工作。

**已确认可用的排除项**：`laion/dramabox-voice-acting-data-annotated` 这类看着对口的短剧配音数据集，源头是 TTS 合成（`ResembleAI/Dramabox`、`gemini-2.5-pro-tts`），违反本项目「不用模型合成语音补量」约定，**vi/id 同样禁用** ❌。

---

## 4. vi/id 与 th/tl 的流程差异

这些是接入时必须知道的口径差异，不是可选优化。

### 4.1 越南语是声调语言（6 声调）

- `f0_std_st` **同样不能当自然度指标**——它包含词汇声调，和泰语一个道理（AGENTS.md 已有此条）。
- **句尾嘎裂声有被裁掉的风险（未实测）**：`ngã` / `nặng` 调带 creaky voice，能量低，RMS 首尾裁切可能把它当静音啃掉。现有防线是 `trim_silence(min_run=0.25)`（首尾低电平段必须连续 ≥0.25s 才裁）+ 朗读源走 Silero VAD。**验收时重点听句尾**，尤其是以 `ngã`/`nặng` 调音节收尾的句子；真出现就调低该源的 `edge_trim_ratio`，不要动 `min_run`（调它会啃词首清辅音）。
- `rate` 指标是**音节/秒**（越南语正字法按音节空格分隔），不与 tl/en/id 的词/秒同量纲，别横向比。

### 4.2 基座不做 vi/id 的文本归一化 ⚠️

`src/voxcpm/utils/text_normalize.py:172` 是 **zh/en 二分**：`lang = "zh" if contains_chinese(text) else "en"`，非中文一律走**英语** TN 规则和 `inflect` 的英语数字拼读。也就是说开了归一化，越南语/印尼语的数字会被念成英文单词。

- **推理侧必须保持 `normalize=False`**（`src/voxcpm/core.py:193` 的默认值，本项目现状就是不开）。这不是 vi/id 引入的新问题，th/tl 同样如此。
- **台词里的数字/货币/日期要在数据导入时人工写成念法**（越南语/印尼语拼写），因为基座不代劳，而本项目也不打算改只读的 submodule。评测集同理：`Rp 5 juta` / `5 triệu đồng` 要先决定念法再写进 case 文本。

### 4.3 控制前缀的守卫边界

`pipeline._assert_control_lang` 现在挡泰文区 + 越南语专属字符（`ơ ư đ` 与带声调元音）。**印尼语是纯 ASCII 拉丁字母，与英文无法区分，守卫挡不住**——只能靠标注规范：素材导入页的「控制描述（中文/英文）」两个框，不要往里填印尼语。

### 4.4 验收口径

`voxft.eval` 的 `--lang` 现在支持 `th/tl/vi/id/zh/en`。

- vi/id 都是 Whisper 标准语种，**强制指定解码语种**（`AUTO_DETECT_LANGS` 只有 `tl`），让 CER 在不同 checkpoint 之间可比。
- **vi 的 WER 是音节级错误率**（`WER_LANGS` 含 vi/id/tl/en）。它和 id/tl/en 的词级 WER 不同量纲，`WER_LANGS` 的注释里写了，报告里别横向比。
- 泰语保留声调组合符的 `_norm` 逻辑（保留 Unicode M 类）对越南语同样必要：NFC 归一后越南语声调字符是 L 类预组合字符，不会被误删。

### 4.5 许可：vi/id 没有 SA 红线

本轮注册的 vi/id 源：Apache-2.0（gigaspeech2）+ CC-BY-4.0（FLEURS）+ CC0（CV22 镜像）+ 自有授权（drama_*）。**没有 CC-BY-SA，没有 NC/ND** → 不触发 [corpus_sourcing.md §3.1](corpus_sourcing.md) 那条「含 SA 数据的权重不得对外分发」红线。

但两条仍然成立：① 后续若核实进任何 SA/NC/ND 源（MagicHub 免费档就是 NC-ND），红线立刻恢复；② `gigaspeech2` 是 `gated`，条款要自己读一遍再商用，Apache-2.0 说的是**数据**许可，不代替仓库条款。

---

## 5. 首轮配比与降级后的验收目标

**执行顺序建议：TH → TL → ID → VI。** 理由：id 的基座基线最好（内部基准 WER 1.36%、MLS 1.084）、正字法纯 ASCII 无 byte-fallback、许可最干净；vi 的 MLS WER 3.307 偏高且有声调裁切风险，放在最后一个啃。

首轮在没有 `drama_*` 自建素材时的配比（按**过滤后训练音频时长**）：

| 数据角色 | VI | ID | 说明 |
|---|---:|---:|---|
| 真人短剧/即兴表演 | **0%** | **0%** | 无已核实来源，**不伪造、不用 TTS 合成补量** |
| 自然口语 | 75% | 75% | `gigaspeech2_*`（试跑通过为前提）。这是本轮唯一能撑起「非念稿」的档位 |
| 发音补充 | 10% | 10% | `fleurs_*` + `cv22_*`。**不要超 10%**：playbook 的既有规则是发音锚点超量会把念稿感带回来 |
| 中文回放 | 10% | 10% | `aishell3` 防遗忘 |
| 英文回放 | 5% | 5% | `replay_en` |

**这一轮的验收目标必须降级，写进结论里**：

- ✅ 可以声称：发音准确度、口语韵律自然度、参考音频克隆能力不退化、中英文指令跟随不劣化
- ❌ **不可以声称**：情绪表现力有改善、去念稿感达成

表演档为 0 时，模型学到的「表现力上限」就是 gigaspeech2/FLEURS/CV22 的表现力上限——网页口语和朗读。**情绪表现力必须等 `drama_vi` / `drama_id` 自建素材到位后开第二轮**，录制方案直接复用 [corpus_sourcing.md §5](corpus_sourcing.md)（情绪 × 场景矩阵、交付格式对齐 manifest.jsonl、一次录制同时产出目标语料 + 跨语言同人 ref + 固定评测集）。

自建素材到位后回到 TH/TL 的配比形态（表演 45% / 口语 30% / 发音 10% / 中英回放 15%），混合命令：

```bash
# 只在远程执行；名称均需先完成加工
# 首轮（无自建素材，表演档 0）：
uv run python -m voxft.data.pipeline --out id_anchor_v1 --mix \
  gigaspeech2_id_v1=75 fleurs_id_v1=5 cv22_id_v1=5 aishell3_v2=10 replay_en_v2=5
# 自建素材到位后回到 TH/TL 形态：
uv run python -m voxft.data.pipeline --out id_drama_v1 --mix \
  drama_id_v1=45 gigaspeech2_id_v1=30 fleurs_id_v1=5 cv22_id_v1=5 aishell3_v2=10 replay_en_v2=5
uv run python -m voxft.data.pipeline --out vi_drama_v1 --mix \
  drama_vi_v1=45 gigaspeech2_vi_v1=30 fleurs_vi_v1=5 cv22_vi_v1=5 aishell3_v2=10 replay_en_v2=5
```

LoRA 预设不变（`r=64 / alpha=64 / dropout=0.05`、lr `1e-4`、`enable_dit=true`、1 epoch 起步）。官方把 language 与 speaker、domain 并列为 5–10 分钟音频即可适配的目标（`README.md:591`），vi/id 属于已支持语言的适配，不需要为它们改超参。

---

## 6. 落地检查清单

```bash
# 本地（不触网、不下载语料）
uv run pytest                                      # 66 passed
uv run python -m voxft.data.download --help         # choices 含 8 个新源
uv run python -m voxft.data.ingest --help           # --source 含 drama_vi / drama_id
uv run python -m voxft.eval --help                  # --lang {th,tl,vi,id,zh,en}

# 远程（逐步放量，先确认 config 名与字段形态再全量）
uv run python -m voxft.data.download --source fleurs_vi --max-samples 50
uv run python -m voxft.data.download --source gigaspeech2_id --max-samples 20
uv run python -m voxft.data.pipeline --source fleurs_vi --out fleurs_vi_v1 --max-items 50
uv run python -m voxft.eval base --lang vi --texts-file eval/vi_holdout.jsonl --seeds 42 43 44
```

- [ ] gigaspeech2 试跑：`audio` 列存在、时长落 3–30s。**若是 `path`+`start`+`end` 长音频形态，直接停用本源**，不要改 `_load_audio` 去猜时间戳语义（`corpus_sourcing.md` 的 FilSwitch 教训：外链音频要保留 revision/镜像/认证，猜形态会静默丢数据）
- [ ] fleurs / cv22 的 config 名与小时数核实，把本文 §2 表格的 `~` 改成 `✅`/`❌`
- [ ] §3 十条待核实项逐条跑完，结论回写本文与 AGENTS.md
- [ ] 自建 `drama_vi` / `drama_id` 素材到位前，任何验收报告不得声称情绪表现力改善
- [ ] 评测集 80–100 条/语种：中/英文 ref × 无前缀/有前缀 × 女声/男声 × 普通口语/强情绪/长短句；vi 额外加**以 `ngã`/`nặng` 调音节收尾**的句子（对应 §4.1 的裁切风险）
