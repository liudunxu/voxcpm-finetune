# voxft — VoxCPM 2 微调工作台

基于 [VoxCPM 2](https://github.com/OpenBMB/VoxCPM) 的短剧配音微调工作台：中/英文参考音频克隆 → 泰语、Tagalog/Taglish，结合中英文情绪/语气前缀。涵盖数据加工、混合、LoRA 训练、离线验收、merge 和 HF 同步。

开发约定见 [AGENTS.md](AGENTS.md)。本轮只调整微调数据、配置及离线评测，不调整翻译或生产配音链路。

## 环境与启动

```bash
git clone --recurse-submodules <repo>
uv sync                       # Python 3.11；macOS CPU/MPS，Linux CUDA 12.4
uv run voxft-ui               # Gradio 6006
# 已克隆但 submodule 为空时：
git submodule update --init --recursive
```

复制 `.env.example` 为 `.env`，配置 `HF_TOKEN`、`WANDB_API_KEY`、`VOXCPM_BASE_PATH`。数据下载、真实音频加工和训练在远程 GPU 机器执行；本地只做开发与合成小样本测试。

远程 `.env` 的大盘设置示例（按实际目录调整）：

```dotenv
VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data
VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt
HF_HOME=/root/autodl-tmp/hf_home
HF_ENDPOINT=https://hf-mirror.com
```

```bash
uv sync --group qc
uv run python -m voxft.data.prefetch --whisper large-v3
uv run python -m voxft.data.prefetch --repo openbmb/VoxCPM2
```

项目入口自动读取 `.env`，镜像模式默认禁用 Xet。独立 `hf` 命令不读 `.env`，需自行设置 `HF_ENDPOINT`、`HF_HOME`、`HF_HUB_DISABLE_XET=1`。

FilSwitch 的转换 parquet 可能只是小体积元数据：下载器会继续按其中的 HF 地址下载原仓库 FLAC（保留 revision 与缓存）。如果日志出现“写入 0 条”，先看具体的缺文本/音频读取失败统计；不能仅凭分片下载成功判定音频已下载。更新下载器后需重启工作台进程。

## 数据策略：先修监督，再扩规模

微调可改善目标语言发音/重音、自然对白韵律、情绪指令响应，以及模型造成的漏词和音色漂移；不能保证解决所有同类反馈。错误翻译、输入台词缺词、剪辑截断和角色选角不在本轮范围。笑声等非语言发声可以训练，但必须有对应真人音频和明确、经过试听核验的控制描述。

### 数据源取舍

| 数据源 | 在微调中的用途 | 限制与当前处理 |
|---|---|---|
| `thai_ser` | 泰语表演/即兴情绪对白 | 仅保留 `turn_type=impro`、agreement ≥ 0.7；保留 actor、强度、轮次元数据；无正文时 large-v3 转写后人工校对 |
| `yodas_th` | 经人工抽检的泰语自然语流 | 视频 ID 不等于真实说话人；默认不配 ref、不合成音量标签、不拼接短句 |
| `drama_tl` / `drama_th` | 自有授权、真人短剧对白主力 | JSONL 导入；要求核实身份、转写、标签和来源，不用模型合成语音补量 |
| `filswitch` | 低比例 Taglish 发音/切换补充 | **新闻朗读**，不是自然对话或情绪主力；许可另行核实 |
| `filipino_emotion` | 待审候选 | 缺文本、可靠身份及完整来源信息；不再列为表现力首选，默认不生成可信控制/refs |
| FLEURS / CV22 / Porjai | 小比例发音补充 | 朗读风格，不宜充当去念稿感主力；身份未知者不配 ref |
| `aishell3` | 中文多说话人回放 | 约 85h；清除原始正文中交错的拼音，仅保留汉字 |
| `replay_en` | 英文多说话人回放 | 自备审核 JSONL，例如经授权核验的 VCTK；不是跨语言同人 ref 的替代品 |

来源核对：[THAI-SER 数据卡](https://huggingface.co/datasets/airesearch/thai-ser)、[YODAS 数据卡](https://huggingface.co/datasets/Chalermdej/yodas2_sidon_th_tts)、[FilSwitch 数据卡](https://huggingface.co/datasets/qwerttyuiiop/FilSwitch)、[AISHELL-3](https://www.openslr.org/93/)、[VCTK](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full)。数据卡的许可标记不代替对原录音来源、演员授权及商用范围的核验。

### 首轮配比（实验起点，不是已验证最优值）

按**过滤后训练音频时长**计算，先分别训练 TH / TL LoRA，确认有效后再考虑联合多语种。

| 数据角色 | TH | TL |
|---|---:|---:|
| 真人短剧/即兴表演 | 45%（THAI-SER impro + 自有对白） | 45%（自有真人对白） |
| 自然口语 | 35%（审核后的 YODAS） | 30%（自有自然 TL/Taglish） |
| 发音补充 | 5% | 10%（FilSwitch 等） |
| 中文回放 | 10% | 10% |
| 英文回放 | 5% | 5% |

TL 先补 5–10h 干净真人对白做试验，覆盖多名女声、男声和年龄段；重点补质疑、克制愤怒、担心、讽刺、哭腔、带笑说话与自然停顿，避免某种情绪只来自某一名演员。每条必须是完整、单人、可听清的 3–30s 语流，不把孤立词或无真实连续时间关系的句子拼成长音频。

混合器按时长采样，每条原始目标音频最多 3 次曝光（嵌套混合也检查），验证集不重复。小源不足时不强凑配比，`mix.json` 记录请求时长、实际时长占比、唯一目标/ref 数、各自最大曝光、control/ref 联合覆盖。3× 上限针对训练目标，ref 复用另行统计。**目标重复 3× 再训练 3 epoch，相当于最多约 9 次曝光**，因此先跑 1 epoch。

### 真人对白 / 跨语言同人 ref 导入

原始清单示例，音频路径相对 JSONL 所在目录；以下是格式模板，不包含实际录音：

```jsonl
{"audio":"audio/actor01_tl_001.wav","text":"Hindi mo alam na buntis ka?","lang":"tl","speaker":"actor01","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording01","emotion":"surprised","emotion_verified":true,"control_zh":"惊讶地，语气克制","control_en":"surprised, restrained","control_verified":true}
{"audio":"audio/actor01_en_ref.wav","lang":"en","speaker":"actor01","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording02","reference_only":true}
```

- `speaker_verified=true` 是人工/可靠原始身份确认，不是“有一列 speaker”。仅 MFCC 相似、同视频、相同角色名均不够；同一演员跨数据源时使用统一 `speaker_namespace` 和 ID。
- 同人中文/英文 ref 作为 `reference_only=true` 独立行导入，3–10s，文本可空；只参与该 split 内配对，不作为训练目标。**不要拿另一个人的英文声音配目标人声**。同语言 ref 是有效基础，但不等同于跨语言训练。
- 按身份、会话、原始音频的连通组先隔离 train/val，再配 ref；同人中/英文候选优先。仅一个独立组时不会伪造验证集，需要另备未见演员评测集。
- 目标联合覆盖：ref+control 30%、ref+裸文本 20%、无 ref+control 20%、无 ref+裸文本 30%。这是有足够已审核标签与身份时的目标；逐源加工后再混合，**最终覆盖不保证自动达到**，检查 `stats.json` / `mix.json`，缺口用真实标注补齐。
- 控制前缀只能是中/英文。有可信 `emotion_verified` 才映射基本情绪；“生气”不会自动扩写为“大声喊叫”。哭腔、笑声、停顿、速度/音量等细节需 `control_verified` 和真实录音支持。未知标签保持裸文本，不能凭能量/F0 猜标签。
- 仅对已验证身份使用同人统一增益，防削波时整个说话人共同回退，保留相对动态；未知身份不调统一响度。已经被上游逐条归一的动态无法凭此恢复。
- `f0_std_st`、`energy_std_db`、`energy_range_db`、`rate` 仅供诊断。能量分位差不是 SNR；泰语 F0 含词汇声调、`rate` 也不是可靠词速，不据此硬筛平读或生成指令。
- 首尾裁切按源分流：**朗读语料**（FLEURS/Porjai/FilSwitch）用 **Silero VAD** 定语音边界（`Options.edge_vad`，faster-whisper 自带，无需额外权重）；**表演语料**（短剧、THAI-SER impro）走**帧 RMS** 门限 0.02，回落到 `peak×0.01`，**保留抽气声**——那是表演的一部分，裁掉模型就学不会换气。RMS 门限相对该条"有声电平"（p99 帧 RMS），且首尾低电平段必须连续 ≥0.25s 才裁（`trim_silence(min_run=)`），否则提高门限会啃掉词首清辅音（/s/ 80–120ms）。**RMS 单独用有个悬崖**：留白电平只要在有声电平 −24dB 以内（朗读档 0.06 的门限），`nz` 覆盖整条、一帧都裁不动，输出长度与输入完全一致——看起来就是"裁静音没生效"。VAD 按语音/非语音分类不受电平影响，实测首尾误差 ±0.04s。两个必须记住的常数：`min_silence_duration_ms` 要用 **500** 而非 faster-whisper 默认的 2000（默认值是给长音频分段用的，会把不足 2s 的尾部底噪并进语音块，实测多留 1.0s）；喂 VAD 的数组必须是 **float32**，float64 会抛 ONNX `Unexpected input data type` 中断整轮加工。`stats.json` 记录本次用的 `edge_vad`/`edge_trim_ratio`，产物可反推门限。

```bash
# 以下命令只在远程执行；本地不下载语料。
uv run python -m voxft.data.download --source thai_ser
uv run python -m voxft.data.pipeline --source thai_ser --out thai_ser_v2
uv run python -m voxft.data.pipeline --source drama_tl --manifest /path/to/acted_tl.jsonl --out drama_tl_v2
uv run python -m voxft.data.pipeline --source drama_tl --manifest /path/to/natural_tl.jsonl --out natural_tl_v2
uv run python -m voxft.data.pipeline --source replay_en --manifest /path/to/replay_en.jsonl --out replay_en_v2
# 下列名称均需先完成加工；页面也支持逐行填写相同多源配比。
uv run python -m voxft.data.pipeline --mix drama_tl_v2=45 natural_tl_v2=30 filswitch_v2=10 aishell3_v2=10 replay_en_v2=5 --out tl_drama_v2
```

自有 acted/natural 两份清单如果共享演员/会话，需在分源前统一安排 holdout；混合器会拒绝跨源 train/val 泄漏。也可先合并原始清单统一加工，接受两类语料内部的自然时长配比。

### 成片素材导入（切分 → 转写 → 试听标注 → 追加）

素材是成片视频或音轨时，用「素材导入」页（或 CLI）代替手工切片：PyAV 解码 → Whisper VAD 定边界（medium）→ 隔 ≤0.7s 的相邻区间合并成 3–30s 候选（超长的在最安静的一帧切开，不切在词中间）→ large-v3 逐条转写并按语种过滤（tl 源放行 `tl`/`en`，Taglish 不会被误杀）→ 页面逐条试听、标说话人/情绪、判定保留或丢弃 → 追加进 `data/raw/<source>/manifest.jsonl`，可选自动重新加工。

```
data/raw/drama_tl/manifest.jsonl        # 追加目标，加工读它
data/raw/drama_tl/holdout.json          # {"sessions": ["ep01"]}：钉住的素材只进验证集
data/raw/drama_tl/ingest/<素材ID>/
    source.wav                          # 解码后的 16k 单声道全轨（重切不必重解码）
    clips/0001_s0012.34_e0018.90.wav
    candidates.jsonl                    # 全量候选，含坏例与「丢弃/待定」；人工标注写回这里
```

```bash
# 只在远程执行；需要 uv sync --group qc（faster-whisper + PyAV）
uv run python -m voxft.data.ingest --input /root/autodl-tmp/drama/ep01.mp4
uv run python -m voxft.data.ingest --input ep02.mp4 ep03.mp4          # 批量，素材 ID 取文件名
uv run python -m voxft.data.ingest --input ep01.mp4 --max-items 20    # 试跑，不追加
uv run python -m voxft.data.ingest --input ep01.mp4 --append --holdout ep01 --process --out drama_tl_v2
```

- **不做声源分离/降噪**：没有 demucs 依赖，官方 zipenhancer 需要 modelscope（不在 lock 里，试听也统一 `load_denoiser=False`）。有对白轨（dialogue stem）就喂对白轨；只有成片混音轨时，BGM/音效重的条目在试听环节判「丢弃」。
- **身份必须人工核实**：切分不出说话人。只有勾了「已核实是本人」才写 `speaker_verified=true`，那是 ref 配对与同人响度对齐的前提；只写 ID 不勾选则落成未验证身份，不配 ref、不调响度。
- `session` 自动设为素材 ID，保证同一集的切片不会被拆到 train/val 两边（`origin_audio` 是每条切片自己，光靠它每条都会独立成组）。
- **追加会让旧验证集泄漏**：`split_records` 的随机分组结果依赖清单长度，追加新素材后重新加工，上一轮的验证组会被整体重排进训练集，已训 run 的评测结论随之作废。挑一集写进 `holdout.json` 钉住（页面填「钉进验证集的素材 ID」）：钉住的分组不参与 shuffle，永远只进验证集，`stats.json` 的 `holdout_pinned_records` 可核对。矛盾组合（钉住了却 `val_ratio=0`、或全部素材都被钉住）直接报错，不会静默把钉住的数据喂进训练。
- 同一素材重切会**替换**它上次追加的行（按 `ingest_video`），不会叠加成近似重复样本；再按音频绝对路径去重。
- 转写复用 `pipeline._transcribe_manifest`：每 100 条落盘、重跑跳过已转写行、坏例只排除不删除。文本由 Whisper 生成的源（`filipino_emotion`、`tagalog_tts`、`thai_ser` impro）另按解码置信度丢掉听不清的条目——时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`，取 faster-whisper 解码器自己的默认门限；`--asr-min-logprob` / `--asr-max-no-speech` 可调，丢弃数记在 `stats.json` 的 `drop_transcribe`，日志会打印原因分类以便判断门限是否过紧。

## 训练方案

默认 LoRA：`r=64 / alpha=64 / dropout=0.05`，`enable_lm=true / enable_dit=true / enable_proj=false`，lr `1e-4`，weight decay `0.01`，grad norm `1.0`；采样率 `16000`（VAE 输入），`48000` 仅推理输出。暂不默认全量微调。

页面默认按 1 epoch 计算步数：`ceil(训练条数 × epoch / (batch_size × 梯度累积 × GPU数))`，默认 batch=2、累积=8。这是按清单条数的近似，官方 loader 的丢尾、长度过滤会有偏差；观察日志实际样本数。允许 1–3 epoch，手动步数仅供受控实验。更改 GPU 数或训练清单后必须重新生成配置。

`training_cfg_rate=0.1` 来自基座 `config.json` 的 `dit_config.cfm_config`（省略时使用模型默认），**不是训练 YAML 顶层参数**。预检会检查它、完整训练/验证清单、ref 身份与集合隔离。生成器另写 `.plan.json` 记录卡数、epoch、等效 batch，不向官方 YAML 塞额外字段。

```bash
# 路径按远程实际目录填写；--base 必须指向已下载的本地模型目录。
uv run python -m voxft.train.yaml_builder --train /root/autodl-tmp/voxft_data/processed/tl_drama_v2/train.jsonl --val /root/autodl-tmp/voxft_data/processed/tl_drama_v2/val.jsonl --base /path/to/VoxCPM2 --run tl_drama_e1 --epochs 1 --gpus 1
uv run python -m voxft.train.launcher configs/tl_drama_e1.yaml 1
```

最后一条打印训练命令；在 GPU 机器上先通过页面预检再启动。wandb 桥接沿用原配置，保存/验证默认每 250 步，结束时官方脚本也保存。不同实验用不同 run 名，不覆盖旧 run；同配置恢复会从 `latest/` 继续。

实验顺序：

1. R0：基座保留为固定评测基线；旧 LoRA 若可用也测一次。
2. R1：仅用修正后的高质量目标语言 + 中英回放，1 epoch，确定发音/漏词没有退化。
3. R2：加入真人表演及可信中英控制标签，训练相同预算，重点比较自然度和情绪。
4. R3：有真实同人跨语言素材后补 ref 配对，比较中/英文 ref → TH/TL 的音色保持；无素材时不要声称完成该验证。

每轮从同一基座开始，尽量固定总训练时长与采样曝光，仅改变要检验的数据因素。达标后才试第二个 epoch；出现内容服从、情绪或音色退化就选择更早 checkpoint，不以 loss 最低作为唯一依据。

## 离线验收

试听页 A/B 严格读取 checkpoint 的 `lora_config`，检查 loaded/skipped/missing keys，用同一模型禁用/启用适配器；加载失败直接停止。A/B 和批量评测关闭自动坏例重试，避免换种子掩盖差异。普通试听参数保持不变。

准备未见演员、未见会话的固定评测 JSONL（不要取训练 ref），逐条保存条件，例如：

```jsonl
{"case_id":"tl_female_surprise_enref","text":"Hindi mo alam na buntis ka?","lang":"tl","ref_audio":"refs/female_en.wav","ref_lang":"en","control":"surprised, in disbelief","speaker":"heldout_f01","kind":"emotion","seed":42}
{"case_id":"tl_female_plain_zhref","text":"Hindi talaga kami bagay sa isa't isa.","lang":"tl","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"","speaker":"heldout_f01","kind":"plain","seed":42}
```

```bash
uv run python -m voxft.eval base /path/to/run/latest --texts-file /path/to/eval.jsonl --seeds 42 43 44
uv run pytest
```

建议每个目标语言先固定 80–100 条：覆盖中/英文 ref、无前缀/有前缀、女主/其他女声/男声、普通口语/强情绪/Taglish/长短句，并加中英文回放回归。保留用户反馈里的难词及漏尾句，但先让母语者确认台词与预期读法；翻译改写不是本轮训练标签自动修复项。

报告保存逐条条件、CER、适用语言的 WER、疑似漏尾、音频路径与待填 `human_review`；多次运行不覆盖。ASR 无法代替母语发音判定，泰语保留声调组合符，Taglish 不强制单一 ASR 语言。F0/能量仅描述，不是越高越好。

验收由至少两名母语评审随机盲听同条件 A/B：自然度、情绪匹配、清晰度、克隆音色分别评分，标记真实截断/噪声/发音错。分 ref 语言、角色及情绪查看结果；目标是自然度/情绪改善且清晰度与音色不退化。自动报告不生成“通过”结论。

## 旧数据迁移

- 所有旧 processed/mixed 清单重加工为新名称（如 `_v2`），不要混用旧伪说话人 refs、拼接音频或拼音污染正文。
- 旧 THAI-SER manifest 缺 `turn_type` 会明确报错，需在远程重下；过滤条件缺列/无效值不再静默放行。
- Whisper 调用不再传不支持的 `batched`，ndarray 始终先转 16k。每 300 条及退出时原子保存完整原清单；推理异常中止，保留已完成转写与被排除的原始记录。`--max-items` 试跑不回写原清单。
- 每次加工写新的音频子目录，避免失败/重跑覆盖旧清单引用的音频；旧音频不自动删除，确认无训练/混合清单引用后再人工清理。
- 原能量差字段改名 `energy_range_db`；启用旧 `min_snr_db`/`min_f0_std` 硬筛会报错。

## 模型管理

```bash
uv run python -m voxft.lora.merge --lora-dir /path/to/run/latest --out /path/to/merged
```

合并支持官方嵌套 `lora_config`；页面上传 HF 功能沿用原流程。只有训练与盲听实际通过后才发布新权重。
