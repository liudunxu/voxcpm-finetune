# 马来语(MS) 接入：设计依据、语料核实状态与流程差异

**2026-09 接入。** 核实标记沿用 [corpus_sourcing.md](corpus_sourcing.md)：`✅` = 拉到了本地文件/页面/API 原文；`~` = 只有间接线索，未二次核实，**不要写进采购单或对外承诺**；`❌` = 核实后确认不可用。

> 命名提醒：FLEURS 的马来语 config 叫 **`ms_my`**，结尾的 `my` 是**国家码马来西亚**，不是缅甸语（Burmese 的 ISO 码才是 `my`）。本轮就因为这个栽过一次，写代码/查数据时都注意。

## 0. 一句话结论

- **基座本来就支持 ms，这不是「教模型一门新语言」。** VoxCPM2 官方 30 语种含 Malay（`README.md:57`），内部基准 **ms WER 1.75%** ✅。
- **但 ms 是五个目标语种里唯一一个基座输给竞品的**：同一张表里 Fish S2-Pro 是 **1.41%**，比 VoxCPM2 好（对比 id 1.36 vs 1.68、vi 1.56 vs 5.56、tl 2.63 vs 4.00，其余三个都是基座赢）✅。**这正好意味着 ms 的提升空间最大**，是联合微调里最值得投入的语种。
- **submodule 零改动**，全部工作在外层工作台 ✅。ms 是纯 ASCII 拉丁正字法，tokenizer 实测 **0.394 tok/char、0 UNK、0% byte-fallback**，与印尼语同档，是五个语种里文本侧最轻的。
- **表现力语料仍然是唯一真缺口**，ms 与 tl/vi/id 处境相同：本轮**没有核实到任何**可商用的开源真人情感/表演语料 ❌。首轮表演档为 0%，验收目标必须降级（见 §5）。
- **许可干净但有坑**：本轮注册的 ms 源是 CC-BY-3.0 + CC-BY-4.0 + 自有授权，**没有 CC-BY-SA 红线** → 权重原则上可对外分发。但马来西亚最大的语料方 `mesolitica`（Malaysia-AI）**全线没有许可声明**，且其 `Malaysian-TTS` 是 **F5-TTS 合成的**，两者都不能用（见 §3.1）。

---

## 1. 基座能力依据（全部可在本地核对）

引用路径相对 `third_party/VoxCPM/`（submodule commit `f772e49`，`2.0.3-33-gf772e49`）。

| 事实 | 出处 | 状态 |
|---|---|---|
| 30 语种列表含 Malay | `README.md:57` | ✅ |
| 「无需语言标签，直接输入任一支持语言的文本」 | `README.md:47` | ✅ |
| 内部 30 语种 ASR 基准：**ms WER 1.75%**，Fish S2-Pro **1.41%** | `README.md:546` | ✅ |
| 同表对照：id WER 1.36%（:540）、vi WER 1.56%（:558）、tl WER 2.63% | `README.md:528-560` | ✅ |
| HF 模型卡 frontmatter 显式列 `ms` | 本地快照 `models--openbmb--VoxCPM2/.../README.md` | ✅ |
| **MiniMax-MLS-test 的 WER 与 SIM 两张表都没有 ms** | `README.md:455-516` | ✅ |
| 代码里不存在语言列表 / language token / lang_id；唯一语种分支是 CJK 判定 | `src/voxcpm/model/utils.py:40-138` | ✅ |
| manifest 与训练 YAML 字段里没有语种轴 | `src/voxcpm/training/data.py:13-16` | ✅ |

**⚠️ ms 没有 SIM（说话人相似度）交叉验证**：MLS 那两张 24 语种表只有 Indonesian / Thai / Vietnamese / Cantonese，没有 Malay。所以「ms 的克隆能力基线是多少」这个问题**没有官方数字可引用**，只能自己用 `voxft.eval` 的 A/B 实测。

**tokenizer 实测**（`LlamaTokenizerFast.from_pretrained` 复刻 `src/voxcpm/model/voxcpm2.py:1144` 的真实调用路径，词表 73440 条、`byte_fallback=True`）：

| 语言 | token/字符 | UNK | byte-fallback 占比 |
|---|---|---|---|
| **ms** | **0.394** | **0** | **0.0%** |
| id | 0.373 | 0 | 0.0% |
| vi | 0.850 | 0 | 35.3% |
| th（已跑通） | 2.205 | 0 | 80.4% |

马来语走拉丁字母，词表天然覆盖，**一个字节回退都不需要**，序列预算上完全无压力（`max_batch_tokens: 8192`）。这一点 ms 比 th/vi 都轻松，与 id 同级。

---

## 2. 本轮注册进来的源

`src/voxft/data/registry.py`。加进 registry 即自动生效：下载分派、`options_for()` 的加工分流（首尾裁切方式、控制前缀比例）、ingest 的 `--source` choices、UI 下拉与【首选】标记。

| 源 id | 角色 | 许可 | 规模 | speaker ID | 状态 | 可用性 |
|---|---|---|---|---|---|---|
| `drama_ms` | expressive | 按自有授权 | 自建 | ✅ 人工核实 | ✅ 通路已验证（与 `drama_tl`/`drama_vi` 同构） | **10/10** 唯一可信表现力来源 |
| `yodas2_ms` | anchor | CC-BY-3.0 | **未核实** | ❌（视频级近似身份） | ~ config `ms000` 已由 parquet API 核实存在；**字段形态未核实** | 7/10 ms 自然口语首选 |
| `fleurs_ms` | anchor | CC-BY-4.0 | **未核实** | ❌ | ✅ config `ms_my` 已由 parquet API 核实存在（103 个 config 之一） | 5/10 干净朗读，只当发音锚点 |

**为什么只注册这 3 个**：AGENTS.md 的红线是「低资源语种的调研结论必须核实到页面/API 原文」，所以凡是规模、许可、字段形态没取到原文的候选一律不进代码，只进 §3 的待核实清单。`yodas2_ms` 与已在用的 `yodas_th` 是同一上游家族（YODAS2 + Sidon 降噪），许可同为 CC-BY-3.0；注册的是**可复现的路径，不是编造的数字**，所以 note 里写着「先 `--max-samples` 试跑」。

### 2.1 code-switch 策略（`Source.languages()`）

| 语种 | 转写/质检放行 | 理由 |
|---|---|---|
| tl | `(tl, en)` | Taglish |
| id | `(id, en)` | 印尼语日常口语混英文程度接近 Taglish |
| **ms** | **`(ms, en)`** | **Manglish**：马来西亚口语与短剧台词句内混英文极普遍，只认 ms 会误杀最该保留的样本 |
| vi | `(vi,)` 从严 | 混英以词内借词为主 |

`fleurs_ms` 用 `accept_langs=("ms",)` **覆盖掉**英文放行：它是有权威文本的朗读语料，语种检测不符意味着错行，不是 code-switch。`yodas2_ms` 不覆盖（YouTube 自发口语，Manglish 是常态）。

---

## 3. 待核实清单（远程有网时逐条做，做完把状态从 ~ 改成 ✅/❌）

这些是**候选**，不是结论。逐条给出核实命令；核到原文再决定是否进 registry。

```bash
# 通用：拉数据集元信息（看 cardData.license / gated / tags / usedStorage）
curl -s "https://huggingface.co/api/datasets/<repo>?full=true" | python -m json.tool | head -60
# 通用：拉数据卡原文（YAML frontmatter + 正文，最有用）
curl -sL "https://huggingface.co/datasets/<repo>/raw/main/README.md" | head -120
# 通用：确认 config/split 是否存在
curl -s "https://huggingface.co/api/datasets/<repo>/parquet" | python -c "import sys,json;print(sorted(json.load(sys.stdin)))"
```

| # | 待核实项 | 要确认什么 | 命令/入口 |
|---|---|---|---|
| 1 | **`yodas2_ms` 字段形态** | 自动转换的 parquet 里音频是内嵌 bytes 还是只有元数据（音频留在 `.tar.gz` 分片里）；时长是否落 3-30s；`metadata.json` 的 YouTube video ID 字段名叫什么（要映射到 `session_col`，否则同一视频的切片会跨 train/val 泄漏） | `download --source yodas2_ms --max-samples 20`，看日志时长分布与列名 |
| 2 | `fleurs_ms` 小时数 | config `ms_my` 已确认存在，但各语种通常十余小时，ms 具体多少没取到原文 | `download --source fleurs_ms --max-samples 50` |
| 3 | **`mesolitica` 全线授权** | Malaysia-AI 是马来语最大的语料方，`Malaysian-STT-Whisper`(10M+ 条)、`dedup-Malaysian-Emilia`(70GB)、`pseudolabel-malaysian-youtube-*`(1M+) 规模都很大，但**没有一个标了 license**。要问清：能否商用、能否用于训练对外发布的模型、要不要署名 | 联系 Malaysia-AI（对应泰语 MagicHub 的询价路径），或查 https://github.com/malaysia-ai/dataset 的授权说明 |
| 4 | `deepdml/common_voice_26_0` 的 `ms` | Common Voice **22** 镜像确认没有 ms，但 **26** 这个镜像的 cardData 里**有 ms**（50 语种，东南亚含 id/ms/th/vi）。问题是 `gated: manual` 且**无许可声明**——CV 官方数据本身是 CC0，镜像为什么不标？validated 小时数多少？ | `curl -s ".../api/datasets/deepdml/common_voice_26_0?full=true"`；申请访问后看 config 列表 |
| 5 | `malaysia-ai/malay-conversational-speech-corpus` | **自发对话**语料（最缺的那一档），但只有 0.2GB 且无许可声明。是真人吗？多少人？能不能商用？ | 数据卡 + 联系 malaysia-ai |
| 6 | `SaLTUNIMAS/sarawak-malay-asr` | CC-BY-4.0 但规模 1K<n<10K，且是**砂拉越马来语**（方言）。与标准马来语差异多大、值不值得低比例混入 | 数据卡 + 抽样听 |
| 7 | `freococo/myanmar_cele_voices` 那类 license=other 的娱乐语音在 ms 侧有没有对应物 | 短剧/综艺/配音类语料是表现力档的唯一希望 | `curl -s ".../api/datasets?filter=language:ms&filter=modality:audio&limit=100&full=true&sort=downloads"` 逐个看 license |
| 8 | ms 有没有**情感语音**语料 | 这是决定「要不要自建 `drama_ms`」的关键结论，必须核到原文才能写死 | 同 #7，关键词加 emotion/expressive/drama |

**核实纪律**（来自 `corpus_sourcing.md §1` 的教训）：这个领域出现过完全编造的论文和不存在的许可声明。**任何「有现成大规模语料」的说法，在拉到页面/API 原文之前都当作不存在。**

### 3.1 已排除项（核实过，别再重复调研）

| 源 | 结论 | 依据 |
|---|---|---|
| **`mesolitica/Malaysian-TTS`** | ❌ **禁用：是合成语音** | 数据卡原文：「Malaysian **Synthetic** TTS dataset. Generate using Malaysian-F5-TTS-v2」。182GB / 含 Husein 300h、Anwar Ibrahim 269h 等名人音色。违反本项目「不用模型合成语音补量」约定，与 `laion/dramabox-voice-acting-data-annotated` 同一个坑；名人音色还有额外的肖像/声音权风险 |
| **`espnet/floras`** | ❌ 不可用：长音频基准，不是可切分语料 | 数据卡原文：「benchmark For **LO**ng-form **R**ecognition **A**nd **S**ummarization」「raw long-form conversational audio」。47 语种含 ms/id，CC-BY-3.0 许可很干净，但整包 **3TB** 且是未切分长音频——本项目管线要 3-30s + text，形态不对（与 `freococo/9000hours_voa_burmese_audio` 同一类问题） |
| `fsicoli/common_voice_22_0` 的 ms | ❌ **不存在** | cardData 103 语种，东南亚只有 `id / lo / th / vi`，**没有 ms**（也没有 tl）。所以不会有 `cv22_ms`，别再照 th/vi/id 的模式去找 |
| `speechcolab/gigaspeech2` 的 ms | ❌ **不存在** | cardData `configs: ['th','id','vi']`，只有三个。vi/id 的自然口语首选在 ms 侧没有对应物，这是 `yodas2_ms` 必须顶上的原因 |
| `disco-eth/WorldSpeech` / `Centi234/WorldSpeech` | ❌ NC | `cc-by-nc-4.0`，禁商用 |
| `MERaLiON/sea_audiobench_datasets_*` | ❌ NC-ND | `cc-by-nc-4.0` / `cc-by-nc-nd-4.0`，NC 禁商用且 ND 禁演绎（微调就是演绎） |
| `LULab/myMediEval_speech` 等 | ❌ NC-SA + gated manual | 医疗领域，许可与领域都不对 |
| `DatarrX/burmese-synthetic-speech-corpus` 之类合成集 | ❌ 合成语音 | 同 `Malaysian-TTS` 的理由 |

---

## 4. ms 与 th/tl/vi/id 的流程差异

### 4.1 ms 与 id 高度互通，这是收益也是风险 ⚠️

马来语与印尼语是同一语言的两种标准化变体，词汇与语法大量重叠（`Saya tidak tahu` 两边都一样）。对联合微调：

- **收益**：id 的语料对 ms 有正迁移，联合训练时两者互相增强，这是把 id 和 ms 放进同一个 LoRA 的主要理由。
- **风险**：口音、拼写与词汇差异会被模型抹平（马来西亚 `saya`/印尼 `aku` 的语用差异、`-a` 尾音的读法、英语借词的发音习惯）。**盲听必须由马来西亚母语者做，不能拿印尼语听感代替**——这是 ms 验收里最容易偷懒也最容易出错的一环。
- **数据侧后果**：`yodas2_ms` 是 YouTube 抓取，上游语种标签可能把印尼语内容误标成 ms（反之亦然）。Whisper 的语种检测同样会混淆两者，所以 `drop_lang` 过滤**挡不住 id/ms 互串**。试跑时要抽样听，确认拿到的确实是马来西亚口音。

### 4.2 控制前缀守卫拦不住 ms

`pipeline._assert_control_lang` 挡泰文区 + 越南语专属字符。**马来语与印尼语一样是纯 ASCII 拉丁字母，与英文无法区分，守卫挡不住**（`tests/test_pipeline.py` 里有一条断言把这个洞固定下来）。只能靠标注纪律：素材导入页的「控制描述（中文/英文）」两个框不要往里填马来语。

### 4.3 基座不做 ms 的文本归一化 ⚠️

`src/voxcpm/utils/text_normalize.py:172` 是 **zh/en 二分**：`lang = "zh" if contains_chinese(text) else "en"`，非中文一律走**英语** TN 规则和 `inflect` 的英语数字拼读。开了归一化，马来语的 `RM 50` / `5 juta` 会被念成英文。

- **推理侧必须保持 `normalize=False`**（`src/voxcpm/core.py:193` 的默认值，本项目现状就是不开）。这不是 ms 引入的新问题，th/tl/vi/id 同样如此。
- **台词里的数字/货币/日期要在数据导入时人工写成马来语念法**，因为基座不代劳，本项目也不改只读的 submodule。评测集同理：`RM 50` 要先决定念法（`lima puluh ringgit`）再写进 case 文本，否则 CER/WER 失真、只能当盲听材料。

### 4.4 Manglish 是 ms 的常态，不是噪声

马来西亚口语大量句内混英文（`I tak tahu lah`、`you nak makan tak`）。这与 Taglish 同构，所以：

- `CODE_SWITCH_ACCEPT` 给 ms 放行 `en`（否则 Whisper 判成 en 的样本会被 `drop_lang` 全杀掉，而那些恰恰是最该保留的自然口语）。
- **不要把 Manglish 当脏数据清洗掉**。线上短剧台词本来就是这个形态，洗干净了反而不像。
- 但 `yodas2_ms` 的 YouTube 来源里可能混入纯英文内容被误标成 ms，这类要靠试听淘汰，不能靠语种过滤（它本来就被放行了）。

### 4.5 验收口径

`voxft.eval` 的 `--lang` 现在支持 `th/tl/vi/id/ms/zh/en`。

- ms 是 Whisper large-v3 标准语种，**强制指定解码语种**（`AUTO_DETECT_LANGS` 只有 `tl`），让 CER 在不同 checkpoint 之间可比。
- **ms 词间用空格分隔，词级 WER 有意义**，所以 `WER_LANGS` 含 ms（与 id/tl/en 同量纲，可以横向比）。这与 th（词间无空格，只有 CER）和 vi（音节级 WER）不同。
- 联合 run 必须看报告的 **`by_lang`** 段：逐语种与 `eval base` 的同一份 case 对比，**任一语种退化即算失败**。
- ms 不是声调语言，但 `f0_std_st` 依然**不作通过门限**（AGENTS.md 铁律），自然度只能盲听。

### 4.6 许可：ms 没有 SA 红线

本轮注册的 ms 源：CC-BY-3.0（`yodas2_sidon`）+ CC-BY-4.0（FLEURS）+ 自有授权（`drama_ms`）。**没有 CC-BY-SA，没有 NC/ND** → 不触发 [corpus_sourcing.md §3.1](corpus_sourcing.md) 那条「含 SA 数据的权重不得对外分发」红线。

两条仍然成立：① 后续若核实进任何 SA/NC/ND 源，红线立刻恢复；② **`mesolitica` 那批「无许可声明」的源在拿到书面授权前一律不得使用**——无声明不等于开放，默认是全权保留。CC-BY-3.0 还要求署名，交付文档里要写清上游来源。

---

## 5. 首轮配比与降级后的验收目标

ms 现在纳入**五语种联合微调**（`th / tl / vi / id / ms`），配比见 [finetune_playbook.md 的「五语种联合首轮配比」](finetune_playbook.md)：每个目标语种各 17%，中文回放 10%、英文回放 5%。

ms 那 17% 内部的拆分（首轮，无自建素材时）：

| 数据角色 | MS 占比 | 说明 |
|---|---:|---|
| 真人短剧 / 即兴表演 | **0%** | 无已核实来源，**不伪造、不用 TTS 合成补量**（`Malaysian-TTS` 就是合成的，禁用） |
| 自然口语 | 13%（17 中的） | `yodas2_ms`（试跑通过为前提）。**这是 ms 唯一能撑起「非念稿」的档位**，因为 gigaspeech2 没有 ms |
| 发音补充 | 4%（17 中的） | `fleurs_ms`。**不要超**：playbook 的既有规则是发音锚点超量会把念稿感带回来 |

**这一轮的验收目标必须降级，写进结论里**：

- ✅ 可以声称：发音准确度、口语韵律自然度（前提是 `yodas2_ms` 试跑通过）、参考音频克隆能力不退化、中英文指令跟随不劣化、**WER 相对基座 1.75% 的改善**（这是 ms 最有说服力的一项，因为基座在 ms 上本来就输给竞品）
- ❌ **不可以声称**：情绪表现力有改善、去念稿感达成

表演档为 0 时，模型学到的「表现力上限」就是 yodas2/FLEURS 的上限——网页口语和朗读。**情绪表现力必须等 `drama_ms` 自建素材到位后开第二轮**，录制方案直接复用 [corpus_sourcing.md §5](corpus_sourcing.md)（情绪 × 场景矩阵、交付格式对齐 manifest.jsonl、一次录制同时产出目标语料 + 跨语言同人 ref + 固定评测集）。录制时**同一演员顺手录 20-30 条中文/英文短句**，那批 `reference_only=true` 行是跨语言 ref 通路的唯一解（playbook 决策点 2）。

自建素材到位后回到 TH/TL 的配比形态（表演 45% / 口语 30% / 发音 10% / 中英回放 15%）。

LoRA 预设不变（`r=64 / alpha=64 / dropout=0.05`、lr `1e-4`、`enable_dit=true`、1 epoch 起步）。官方把 language 与 speaker、domain 并列为 5–10 分钟音频即可适配的目标（`README.md:591`），ms 属于已支持语言的适配，不需要为它改超参。**注意 r=64 是本项目自己的工程决策，官方 yaml 的默认值是 r=32/alpha=32/dropout=0.0，写文档时别把 64 说成官方推荐。**

---

## 6. 落地检查清单

```bash
# 本地（不触网、不下载语料）
uv run pytest                                      # 72 passed
uv run python -m voxft.data.download --help         # choices 含 drama_ms/yodas2_ms/fleurs_ms
uv run python -m voxft.data.ingest --help           # --source 含 drama_ms
uv run python -m voxft.eval --help                  # --lang {th,tl,vi,id,ms,zh,en}

# 远程（逐步放量，先确认 config 名与字段形态再全量）
uv run python -m voxft.data.download --source fleurs_ms --max-samples 50
uv run python -m voxft.data.download --source yodas2_ms --max-samples 20
uv run python -m voxft.data.pipeline --source fleurs_ms --out fleurs_ms_v1 --max-items 50
uv run python -m voxft.eval base --lang ms --texts-file eval/ms_holdout.jsonl --seeds 42 43 44
```

- [ ] `yodas2_ms` 试跑：`audio` 列存在、时长落 3–30s。**若转换 parquet 只有元数据、音频留在 `.tar.gz` 分片里，直接停用本源**，不要给 `_load_audio` 加 WebDataset 解包去猜形态（FilSwitch 的教训：猜形态会静默丢数据）
- [ ] `yodas2_ms` 试跑后把 YouTube video ID 映射到 `session_col`，否则同一视频的切片会跨 train/val 泄漏
- [ ] **抽样听 `yodas2_ms`，确认拿到的确实是马来西亚口音而不是印尼语**（Whisper 与上游标签都分不清两者，`drop_lang` 挡不住 id/ms 互串）
- [ ] `fleurs_ms` 小时数核实，把本文 §2 表格的 `~` 改成 `✅`/`❌`
- [ ] §3 八条待核实项逐条跑完，结论回写本文与 AGENTS.md
- [ ] 自建 `drama_ms` 素材到位前，任何验收报告不得声称情绪表现力改善
- [ ] 评测集 80–100 条：中/英文 ref × 无前缀/有前缀 × 女声/男声 × 普通口语/强情绪/长短句；**ms 额外加 Manglish 句内混英的 case**（对应 §4.4，这是线上真实形态）与**含 `RM`/数字的 case**（对应 §4.3，验证念法是数据侧写好的）
- [ ] 盲听必须由**马来西亚母语者**做，不能用印尼语听感代替（§4.1）
