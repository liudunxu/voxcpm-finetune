# AGENTS.md

## 项目简介
VoxCPM 2（OpenBMB TTS）微调工作台：Tagalog/泰语/越南语/印尼语高质量语料的下载与加工、跨语言（中文→目标语言）混合微调、LoRA/全量训练管理、wandb 监控、LoRA merge、HuggingFace 同步。Gradio 页面端口 **6006**。

## 环境
- Python 3.11（.python-version 已固定），依赖由 **uv** 管理：`uv sync`（本地开发）、`uv sync --group qc`（启用 whisper 质检 + PyAV 视频解码）。
- torch 平台分流（见 pyproject `[tool.uv.sources]`）：macOS → PyPI 轮子（CPU/MPS）；Linux → pytorch-cu124 index（CUDA 12.4）。训练只在 Linux GPU 机执行。
- **数据集下载只在远程 GPU 机器进行**：本地开发环境不下载数据集，也无需本地验证下载流程。
- 密钥走 `.env`（复制 `.env.example`）：`HF_TOKEN`、`WANDB_API_KEY`、`WANDB_PROJECT`、`VOXCPM_BASE_PATH`、`HF_ENDPOINT`（国内默认 `https://hf-mirror.com`，导入 voxft 即自动加载）。

## 目录结构
- `src/voxft/data/` — 数据源清单（registry）、下载（download）、加工管线（pipeline）、成片导入（ingest）
- `src/voxft/train/` — yaml_builder（生成官方训练配置）、launcher（子进程启动训练）、tb_wandb_bridge（TensorBoard→wandb 桥接）
- `src/voxft/lora/merge.py` — LoRA 合并导出完整模型
- `src/voxft/hub/sync.py` — HuggingFace 上传
- `src/voxft/qc/utmos.py` — UTMOS 音质打分（移植自 OmniVoice，注明来源）
- `src/voxft/ui/` — Gradio 6 Tab 页面（数据集/素材导入/训练/试听/模型管理/日志）
- `third_party/VoxCPM/` — **官方仓库 submodule，只读，禁止直接修改**；训练入口为其内 `scripts/train_voxcpm_finetune.py`
- `configs/` — 生成的训练 YAML；`data/`、`checkpoints/` 为大文件产物（已 gitignore）

## 常用命令
```bash
uv sync                              # 安装依赖
uv run voxft-ui                      # 启动微调工作台（端口 6006）
uv run python -m voxft.data.prefetch --whisper large-v3   # 预取权重（自动用 .env 镜像设置）
uv run python -m voxft.data.download --source fleurs_th --max-samples 100
# 成片导入（需 --group qc）：PyAV 解码 → VAD 切 3-30s → large-v3 逐条转写；标注在页面「素材导入」做
uv run python -m voxft.data.ingest --input /root/autodl-tmp/drama/ep01.mp4
uv run python -m voxft.data.ingest --input ep01.mp4 --append --holdout ep01 --process
# 离线诊断（需 --group qc）：ASR 内容误差/疑似漏尾；自然度与情绪另做母语盲听
uv run python -m voxft.eval base checkpoints/<run>/latest --lang th --ref-audio ref.wav --control "愤怒地，语速快"
uv run pytest                        # 测试（testpaths=tests，不会去收 third_party 的官方脚本）
# 训练（在 GPU 机器上，由页面生成的命令）：
cd third_party/VoxCPM && torchrun --nproc_per_node=N scripts/train_voxcpm_finetune.py --config_path <生成的yaml>
```

## 标准工作流
1. 数据集 Tab：下载或导入已审核 JSONL → 16k / 裁静音 / 3–30s / 质检 / 声学描述 → 按身份、会话、原音频隔离 train/val → 已验证说话人整体增益 → 可信控制前缀 → split 内 ref 配对
2. 素材导入 Tab（自备成片）：喂视频/音轨 → 自动切分转写 → 逐条试听淘汰 BGM 重的、标说话人与情绪 → 追加进原始清单 → 可选自动加工；钉一集进 `holdout.json` 当固定验证集
3. 混合：首轮目标语言 85% + 中文 10% + 英文 5%（实验起点），**按有效音频时长**采样；原始目标音频最多 3×（含嵌套混合），ref 复用另统计，验证集不重复，实际占比/曝光/联合覆盖写进 mix.json
4. 训练 Tab：填表单 → 生成 YAML → 启动；wandb 桥接自动转发指标
5. 试听 Tab：加载 checkpoint 试听；模型管理 Tab：merge LoRA / 上传 HF

## 微调铁律（来自官方文档/FAQ，不要违反）
- VoxCPM 2：`sample_rate=16000`（AudioVAE 编码器输入）、`out_sample_rate=48000`（仅推理）
- 数据 JSONL 字段：`audio`、`text` 必填；`ref_audio`（同说话人）30–50% 样本；时长 3–30s；尾静音 <0.5s（否则"生成停不下来"）
- LoRA：lr=1e-4、`enable_dit: true`（音质关键）、r=32（说话人）/**64（语言+风格，本项目默认）**；全量：lr=1e-5（LoRA 的 1/10）
- **控制前缀进入训练文本**：VoxCPM2 的情绪/语气控制是 `(控制指令)正文`（官方 `cli.build_final_text`）。训练全裸文本有损害指令跟随能力的风险。目标 25–50% 带前缀，但只使用 `emotion_verified=true` 或 `control_verified=true` 的可信标签，不从能量/F0/空格数猜测表演。其余保持裸文本；不足比例不伪造标签
- 控制前缀只写中英文（线上 prompt 就是中英文），不要写目标语言
- 防过拟合忽略文本：`training_cfg_rate=0.1`（基座 `dit_config.cfm_config`；不是训练 YAML 顶层参数）、`weight_decay=0.01`；先 1 epoch，按验收最多 3 epoch。混合重复次数还需乘 epoch 计实际曝光
- 推理侧 LoRA 配置与训练完全一致（官方 JSON 的 `lora_config` 嵌套层）；`load_lora` 返回 `(loaded_keys, skipped_keys)`，loaded 非空、skipped/missing 为空，失败即停止
- **训练的 ref 模式与推理的 reference-only 模式结构完全一致**（`packers.process_tts_data_with_ref` = `[103 ref 104][text][101 target 102]`）。带控制前缀时必须走 reference-only：combined 模式会拼成 `prompt_text + "(控制)正文"`，前缀跑到句中就失效（`infer._gen_kwargs` 已按此处理）
- Common Voice 22 官方已从 HF 撤架：数据源已切换到社区镜像 `fsicoli/common_voice_22_0`（非 gated，脚本式数据集，流式下载带 `trust_remote_code`）

## 来自 OmniVoice 生产实践的结论（/Users/dunxu.liu/workspace/others/OmniVoice）
其配音生产链路：基座模型 + 参考音频零样本克隆（不用 LoRA），强依赖 `reference_wav_path` 克隆与逐语言文本归一化。据此：
- **微调方式首选 LoRA**：全量微调更易损害参考音频克隆泛化（生产核心能力）；r=32 说话人适配 / r=64 语言风格适配（本项目默认 64/64/dropout 0.05），`enable_dit: true` 必开
- **数据必须保留克隆能力**：30–50% 样本带已验证同说话人 `ref_audio`（默认目标 0.5，其中 ref+control 目标 0.3）；缺少可靠身份不强凑。跨语言同人 ref 以 `reference_only=true` 行导入，与目标共用真实身份；中英文回放不是跨语言同人 ref 的替代品
- **试听/推理默认参数**：`cfg_value=2.0`、`inference_timesteps=20`（生产基线 10，本项目默认 20 换更高音质）、`retry_badcase=True`（max_times=3、ratio_threshold=6.0）；音频坏例重试时降 CFG 至 1.2–1.6 并加步数
- 跨语言（中文/英文→Tagalog/泰语/越南语/印尼语）：目标语言为主 + 中英文回放；先分别验证各语种 LoRA（顺序 TH → TL → ID → VI），再考虑联合模型。vi/id 首轮表演档为 0，只验发音与口语韵律，不声称情绪改善

## 踩坑与约定（已修复问题的沉淀，勿回退）
- **Gradio 流式**：按钮必须直接绑定生成器函数；用 `lambda` 包一层会把生成器对象本身渲染进文本框
- **Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。任何运行后变化的列表（已加工数据集/配置/LoRA/上传目录）必须通过事件输出或 `Tab.select` 刷新
- **`.env` 加载顺序**：`paths.py` 必须**先** `load_dotenv()` **再**计算路径常量（`VOXFT_CKPT_ROOT`/`VOXFT_DATA_ROOT` 依赖此顺序）
- **大盘约定（远程）**：`VOXFT_DATA_ROOT`、`VOXFT_CKPT_ROOT`、`HF_HOME` 都指到 `/root/autodl-tmp/*`；系统盘小，下载/缓存勿落 `~`
- **HF 生态**：`datasets` 锁定 `<4`（5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突）；`hf`/`huggingface-cli` 不读项目 `.env`，命令行需手动 `export`（三个都要：`HF_ENDPOINT`/`HF_HUB_DISABLE_XET`/`HF_HOME`）——**优先用 `python -m voxft.data.prefetch`**，它导入 voxft 时就把这些处理好了；`snapshot_download` 的进度条不传给单文件，进度监控用缓存目录大小轮询；xet 下载分两阶段（downloading→reconstructing），进度"回退"属正常。**但走镜像必须关 xet**：hf-mirror 只代理 HF API，不代理 xet 的 CAS 服务器（cas-server.xethub.hf.co），reconstruction 阶段会直连并报 401。`paths._disable_xet_on_mirror()` 在 HF_ENDPOINT 非 huggingface.co 时自动 `setdefault("HF_HUB_DISABLE_XET","1")`；命令行用 `hf` 时要自己 export
- **训练默认**：`batch_size=2 + 梯度累积=8`（等效 batch 还需乘 GPU 数）；页面按 1 epoch 自动算步数，换清单/卡数要重建配置，`.plan.json` 留审计；启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`save/valid_interval=250`；训练结束自动只保留最新 5 次 LoRA 运行
- **`max_grad_norm=1.0`、`num_workers=8`**：官方 v2 配置就是这两个值（早先本文写的"官方默认 0 = 不裁剪"是错的，已改）。情感语料动态大，不裁剪更容易出梯度尖峰
- **推理**：`load_denoiser=False`（去噪器依赖 modelscope，试听不需要）
- **Whisper 权重**：large-v3 约 3GB，国内直连 huggingface.co 常在 SSL 握手就超时。加载带 3 次重试并打印 endpoint；失败时报错里给了预下载命令。可用 `VOXFT_WHISPER_MODEL` / `VOXFT_WHISPER_MODEL_LARGE` 指向本地目录
- **万级转写必须能断点续跑**：每 300 条及退出时原子保存完整原清单（包括坏例、尚未处理的行），重跑跳过已转写行。WhisperModel.transcribe 不接受 `batched`；ndarray 输入先转 16k。推理异常必须中止，不可当语料坏例吞掉。`--max-items` 试跑不回写原清单
- **数据源首选**：泰语 `thai_ser` 仅 impro / 审核后 `yodas_th`；Tagalog 自有真人 `drama_tl`；`filipino_emotion` 仅待审候选。`filswitch` 是新闻朗读，仅低比例补 Taglish 发音。越南语/印尼语表现力只有自建 `drama_vi` / `drama_id`，公开锚点首选 `gigaspeech2_vi/id`（Apache-2.0，但**字段形态未核实，先 `--max-samples 20` 试跑**），`fleurs_*` / `cv22_*` 只当发音补充。中英文回放 `aishell3` / 自备 `replay_en`。不能把朗读数据当去念稿感主力
- **FilSwitch 下载**：转换 parquet 可以只有元数据，音频在原仓库的独立 FLAC 文件。`bytes=None` 不等于无音频；共享 `_load_audio` 支持内嵌 bytes、HF URL/路径和已解码数组，外链通过 hf_hub_download 保留原 revision、镜像、认证和缓存。不要把音频地址套到 `refs/convert/parquet` 分支；读取失败必须记录原因，不能静默跳过整包
- **code-switch 语种不能只认目标语种**：Tagalog 短剧台词是 Taglish、印尼语日常口语混英文，句内英文词多的样本 Whisper 会判成 en，只认目标语种会把最该保留的 code-switch 样本全部误杀。`Source.languages()` 查 `CODE_SWITCH_ACCEPT`，对 tl / id 默认放行 `en`；**vi 默认从严**（混英以词内借词为主），实测 `drop_lang` 误杀再给该源加 `accept_langs=("vi","en")`，别提前放开。有权威文本的朗读源（`fleurs_id` / `cv22_id`）用 `accept_langs` **覆盖掉**默认放行——语种不符意味着错行，不是 code-switch
- **`yodas_th` 会话**：`utt_id.rsplit("-", 3)[0]` 保留可能含 `-` 的完整视频 ID。speaker_id 为视频级近似身份，不作 ref 依据；上游逐条峰值归一，不据此标音量。无原始连续时间关系就不拼接
- **数据身份**：MFCC 聚类仅供审计，不能证明同人，更不能调低阈值强凑 ref。身份未知默认不配 ref。同一演员跨源使用统一 speaker_namespace/ID；先隔离 train/val，再在集合内配 ref，混合与训练前再次检查泄漏
- **`filipino_speech`**：过滤 `machine` 与 `num_words<4`，只保留完整句；不再拼接孤立词或随机抖动停顿来伪造对白。行过滤缺列或无效数值时不放行
- **`thai_ser` 没有名为 `audio` 的列**（四路麦 `mic_clip/mic_con/mic_middle/mic_zoom`），必须靠 registry 的 `audio_cols` 映射，否则整个源在下载阶段被静默跳过；`mic_zoom` 是网络录音，不用
- **响度**：仅已验证说话人统一增益到 −24 dBFS；防削波时整个说话人共同回退，禁止单条峰值归一。未知身份不统一调响度；上游已抹掉的动态不能恢复
- **声学描述**：`f0_std_st` 含泰语声调和清浊音误差，不能代表自然度；`energy_range_db` 为能量分位差，不是 SNR。拒绝 min_snr_db/min_f0_std 硬筛，不把这些指标自动变成情绪指令
- **AISHELL-3**：约 85h；content.txt 的同一正文列交错汉字与拼音，必须剔除拼音。旧 processed 清单重新加工，不能直接混入
- **离线验收**：逐 case 固定 text/lang/ref_audio/ref_lang/control/seed；A/B 禁用自动坏例重试，普通试听保持原设置。CER/WER/疑似漏尾仅诊断，自然度/情绪/音色/真实截断需母语盲听，F0 不作通过门限
- **重加工**：每次写新音频子目录，不覆盖旧清单引用的音频；旧产物不自动清理。原始 reference-only 与人工审核标记的 JSONL 格式、远程执行命令见 README
- **追加素材会让旧验证集泄漏**：`split_records` 的随机分组结果依赖清单长度，追加新素材后重新加工，上一轮的验证组会被整体重排进训练集，已训 run 的评测结论随之作废。挑一集写进 `data/raw/<source>/holdout.json`（`{"sessions": ["素材ID"]}`）钉住；钉住的分组不参与 shuffle，永远只进验证集，`stats.json` 的 `holdout_pinned_records` 可核对。矛盾组合（钉住了却 `val_ratio=0`、或全部素材都被钉住）直接报错，不静默把钉住的数据喂进训练。`ingest` 的 `session` 自动设为素材 ID，否则同一集的切片会各自成组跨 train/val
- **库函数不许 print**：UI 进程的 stdout 可能是已断开的 pty（启动 voxft-ui 的 SSH/tmux/JupyterLab 终端关掉后进程还在跑），`print` 抛 `[Errno 5] Input/output error`，会把一次**已经成功**的操作报成失败——filswitch 写完 2709 条清单后显示"失败"就是 `download_source` 结尾那句与 progress 重复的 print。一律走 `progress` 回调，CLI 侧传 `progress=print`（download/merge 已改，utmos 的 print 已删）。启动 UI 用 `nohup ... > ui.out 2>&1 &` 或 tmux，别把 stdout 挂在会断的终端上
- **视频容器解码走 PyAV**：soundfile 读不了 mp4/mkv，qc 组已显式声明 `av>=12`（本来就是 faster-whisper 的传递依赖）。不引入系统 ffmpeg 依赖，本地 macOS 与远程行为一致
- **首尾裁切按源分流**：`Options.edge_trim_ratio` 朗读 0.06（约 −24dB 相对有声电平）/ 表演 0.02。**RMS 门限单独用有个悬崖**：留白电平只要在有声电平 −24dB 以内，`rms > thr` 覆盖整条、一帧都裁不动，输出长度与输入**完全一致**——FLEURS「裁静音没生效」就是这个，不是没跑。所以朗读源改用 `Options.edge_vad`（Silero VAD，faster-whisper 自带、无需额外权重）定边界，不受电平影响，实测首尾误差 ±0.04s；表演源仍走 RMS 0.02（回落到 `peak×0.01`），抽气声是表演的一部分，裁掉模型就学不会换气。首尾低电平段还必须连续 ≥0.25s 才裁（`trim_silence(min_run=)`），否则提高门限会啃掉词首清辅音（/s/ 80–120ms）。两个必记常数：`min_silence_duration_ms` 用 **500** 而非 faster-whisper 默认 2000（默认给长音频分段用，会把不足 2s 的尾部底噪并进语音块，实测多留 1.0s）；喂 VAD 的数组必须 **float32**，float64 抛 ONNX `Unexpected input data type` 中断整轮加工。`stats.json` 记录 `edge_vad`/`edge_trim_ratio`，旧产物能反推当时门限
- **听不清的按 ASR 置信度丢**：只对 `needs_transcribe` 的源启用（`filipino_emotion`/`tagalog_tts`/`thai_ser` impro）——转写结果就是训练文本，没有原文可比相似度。门限沿用 faster-whisper 解码器自己的默认（时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`），`--asr-min-logprob`/`--asr-max-no-speech` 可调，数量记在 `stats.json` 的 `drop_transcribe`，日志打印原因分类。这不违反"拒绝 min_snr_db/min_f0_std 硬筛"——被禁的是能量分位差与 F0 这类伪指标，不是 ASR 自己的置信度
- **Tagalog 无可商用的开源真人表演语料**（已核实，别重复调研）：Common Voice tl 官方 `recordedHours=0`（社区镜像也无 tl 音频）；YODAS/YODAS2 Sidon 的 224 个语种子集里没有 tl/fil；OpenSLR 无菲律宾语资源；HF 上 `modality:audio` 匹配 filipino/tagalog 的只有厂商 sample（`n<1K`，且多为 CC-BY-NC-ND 或 gated）；SEACrowd 的 23 个 th/tl/fil 数据集全是 text/图像，无音频；`liva-ai/yapdo-convo` 含 tl 但**完全没有许可声明**。`filipino_emotion` 连数据卡都没有。有规模的真人语料只能付费或用「素材导入」自建 `drama_tl`，逐项核实结论与询价模板见 `docs/corpus_sourcing.md`
- **MagicHub 别记错两个库**：`ASR-SFDuSC` 是 4.58h / 4073 条 / **10 人**的 **scripted monologue 朗读**，许可 **CC-BY-NC-ND 4.0**——NC 禁商用、ND 禁演绎（微调就是演绎），对本项目可用性为 0（早先本文写的"免费注册可用"是错的）。真正有价值的是 `ASR-BigFTagaCSC`（MDT-ASR-E076）：**1285h / 514 人自发对话**，16kHz WAV + TXT 转写，手机录、室内外，**专有授权需询价** `business@magicdatatech.com`。514 个真实说话人身份是 YouTube 抓取源给不了的，能同时补「自然口语锚点」和「ref 配对身份」，但**无情绪标签、不是表演**，只对应配比表的自然口语档
- **泰语源有 CC-BY-SA 红线**：`thai_ser` 与 `Porjai-central` 都是 **CC-BY-SA-4.0**（Porjai 的 `pattani`/`khummuang` 更是 **NC-SA**，直接排除）。SA 的触发条件是"向公众分享改编物"，模型权重算不算改编物**无判例**，CC 官方那句"应同许可发布"是保守建议不是法律要求。据此定的红线：**含 SA 数据训练的 LoRA 与 merge 后完整模型一律不对外分发**（不传 HF、不随客户交付、不开源），只通过 API 交付合成音频；要对外发布就向版权方谈商业授权（THAI-SER 出资方是 AIS + DEPA，有明确谈判主体）或只用 Apache-2.0/CC-BY 源。许可干净的泰语现货：Nexdata **1004h**（SKU 1687，商业买断，低噪，**带 speaker ID + gender**，WAR 98%）、`speechcolab/gigaspeech2` th（**Apache-2.0**，`gated:auto`，但短句为主无身份）
- **LAION DramaBox 那批"短剧配音数据"是 TTS 合成的，禁止用于补量**：`laion/dramabox-voice-acting-data-annotated` 看着完全对口（CC-BY-4.0、10万-100万条、标签带 `voice-acting`、数据卡还写了"同说话人跨情绪配对片段"），但数据卡 Models Used 明确源头是 `ResembleAI/Dramabox` 与 `gemini-2.5-pro-tts`，文件名 `{prompt_id}_seed{NN}_part1.mp3` 的 seed 就是生成采样。违反本项目"不用模型合成语音补量"的约定，且情绪标签是生成 prompt 不是真实表演标注。只有**标注 schema** 可参考。同理禁用 OpenSpeechHub 三个泰语集（无 license tag、无数据卡，同组织还挂动漫语音 rip）
- **低资源语种的调研结论必须核实到页面/API 原文**：本轮就出现过一篇编造的竞品论文（"JaiTTS arXiv 2604.27607，1万小时泰语，CER 1.94%"——arXiv API 查该 ID 与全文搜索均返回 0 条），以及把无许可的 `yapdo-convo` 说成 CC-BY-4.0。一条编造的"有现成大规模语料"足以让人跳过真正该做的自建工作
- **泰语转写可换 `typhoon-ai/typhoon-whisper-large-v3`**（SCB-10X，MIT，arXiv 2601.13044，约 11000h 泰语微调，自带泰语数字/重复标记归一化，Gigaspeech2/TVSpeech/FLEURS 泰语 SOTA）。**但不是即插即用**：`library_name: transformers`，不是 faster-whisper 的 CTranslate2 格式，要么 `ct2-transformers-converter` 转格式要么单开转写路径；模型卡在 MIT 之外另有一层 OpenTyphoon T&C 需商用前阅读；且**只有泰语**，对 TL 无帮助。优先级低于把数据搞到手
- **越南语/印尼语接入结论**（详见 `docs/vi_id_support.md`，每条都带 submodule 内的 `文件:行号`）：基座官方 30 语种**已含 vi/id** 且有实测分数（内部 30 语种基准 id WER 1.36% / vi 1.56%，**优于 tl 的 2.63%**），代码里无语言列表/language token/lang_id，tokenizer `byte_fallback=True` 对 vi/id 实测 **0 UNK** → **submodule 零改动**，别试图加语种信号，基座没有对应槽位。两者都**没有已核实的真人表演/情感语料**，处境同 Tagalog，首轮表演档为 **0%** 且验收只能声称发音/口语韵律/克隆不退化，**不能声称情绪表现力改善**。基座文本归一化是 zh/en 二分（`text_normalize.py:172`，非中文一律走英语规则），所以推理侧必须保持 `normalize=False`，**台词里的数字/货币念法是数据侧责任**，评测 case 保留阿拉伯数字时 CER/WER 会失真、只作盲听。vi 是 6 声调语言：`f0_std_st` 同样不代表自然度，**以 `ngã`/`nặng` 调收尾的句子句尾嘎裂声能量低、有被 RMS 裁掉的风险（未实测，验收重点听）**，`rate` 是音节/秒，**WER 是音节级**不与词级横向比。控制前缀守卫拦得住越南语专属字符，**拦不住印尼语**（纯 ASCII 与英文无法区分），只能靠标注纪律。许可上 vi/id 本轮全是 Apache-2.0 / CC-BY / CC0，**无 SA 红线，权重可对外分发**（后续核实进 SA/NC/ND 源则红线恢复）
- **网络受限时不得凭记忆写语料结论**：vi/id 接入那轮 HF API 全部被限流，处置是**只注册可由本仓库既有事实复现的源**（th/tl 已在用的仓库与 config 命名模式、`corpus_sourcing.md` 已核实过的 gigaspeech2 覆盖 th/id/vi），把 VIVOS / MagicHub / Nexdata / YODAS2-vi-id 等候选全部写进 `docs/vi_id_support.md §3` 的待核实清单并附远程核实命令，规模与许可一律标「未核实」不写数字。宁可留空也不要填一个看起来像结论的编造值

## Submodule 升级
```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```
