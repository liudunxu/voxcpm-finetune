# AGENTS.md

## 项目简介
VoxCPM 2（OpenBMB TTS）微调工作台：Tagalog/泰语高质量语料的下载与加工、跨语言（中文→目标语言）混合微调、LoRA/全量训练管理、wandb 监控、LoRA merge、HuggingFace 同步。Gradio 页面端口 **6006**。

## 环境
- Python 3.11（.python-version 已固定），依赖由 **uv** 管理：`uv sync`（本地开发）、`uv sync --group qc`（启用 whisper 质检）。
- torch 平台分流（见 pyproject `[tool.uv.sources]`）：macOS → PyPI 轮子（CPU/MPS）；Linux → pytorch-cu124 index（CUDA 12.4）。训练只在 Linux GPU 机执行。
- **数据集下载只在远程 GPU 机器进行**：本地开发环境不下载数据集，也无需本地验证下载流程。
- 密钥走 `.env`（复制 `.env.example`）：`HF_TOKEN`、`WANDB_API_KEY`、`WANDB_PROJECT`、`VOXCPM_BASE_PATH`、`HF_ENDPOINT`（国内默认 `https://hf-mirror.com`，导入 voxft 即自动加载）。

## 目录结构
- `src/voxft/data/` — 数据源清单（registry）、下载（download）、加工管线（pipeline）
- `src/voxft/train/` — yaml_builder（生成官方训练配置）、launcher（子进程启动训练）、tb_wandb_bridge（TensorBoard→wandb 桥接）
- `src/voxft/lora/merge.py` — LoRA 合并导出完整模型
- `src/voxft/hub/sync.py` — HuggingFace 上传
- `src/voxft/qc/utmos.py` — UTMOS 音质打分（移植自 OmniVoice，注明来源）
- `src/voxft/ui/` — Gradio 4 Tab 页面（数据集/训练/试听/模型管理）
- `third_party/VoxCPM/` — **官方仓库 submodule，只读，禁止直接修改**；训练入口为其内 `scripts/train_voxcpm_finetune.py`
- `configs/` — 生成的训练 YAML；`data/`、`checkpoints/` 为大文件产物（已 gitignore）

## 常用命令
```bash
uv sync                              # 安装依赖
uv run voxft-ui                      # 启动微调工作台（端口 6006）
uv run python -m voxft.data.download --source fleurs_th --max-samples 100
# 效果客观对比（需 --group qc）：贴合度↑ / 截断率↓ / 语调起伏↑
uv run python -m voxft.eval base checkpoints/<run>/latest --lang th --ref-audio ref.wav --control "愤怒地，语速快"
uv run pytest                        # 测试（testpaths=tests，不会去收 third_party 的官方脚本）
# 训练（在 GPU 机器上，由页面生成的命令）：
cd third_party/VoxCPM && torchrun --nproc_per_node=N scripts/train_voxcpm_finetune.py --config_path <生成的yaml>
```

## 标准工作流
1. 数据集 Tab：选数据源 → 下载（行过滤 + 情绪/会话列一并落盘）→ 加工（16k / 裁静音 / 3–30s / 质检 / 表现力指标 / 伪说话人聚类 / 按说话人响度对齐 / 控制前缀 / ref_audio 配对）→ 输出 JSONL
2. 混合：目标语言 80–90% + 中文 10–20%（防灾难性遗忘），按行重复拼接，**小语料重复上限 3×**（超出会缩水并记进 mix.json）
3. 训练 Tab：填表单 → 生成 YAML → 启动；wandb 桥接自动转发指标
4. 试听 Tab：加载 checkpoint 试听；模型管理 Tab：merge LoRA / 上传 HF

## 微调铁律（来自官方文档/FAQ，不要违反）
- VoxCPM 2：`sample_rate=16000`（AudioVAE 编码器输入）、`out_sample_rate=48000`（仅推理）
- 数据 JSONL 字段：`audio`、`text` 必填；`ref_audio`（同说话人）30–50% 样本；时长 3–30s；尾静音 <0.5s（否则"生成停不下来"）
- LoRA：lr=1e-4、`enable_dit: true`（音质关键）、r=32（说话人）/**64（语言+风格，本项目默认）**；全量：lr=1e-5（LoRA 的 1/10）
- **控制前缀必须进训练文本**：VoxCPM2 的情绪/语气控制没有独立条件通道，就是文本前缀 `(控制指令)正文`（官方 `cli.build_final_text`）。训练文本全是裸文本时，LoRA 会把基座的指令跟随能力冲掉，情绪 prompt 越调越不灵。加工默认给 25–50% 样本自动加前缀，**其余保持裸文本**以保住无前缀推理路径
- 控制前缀只写中英文（线上 prompt 就是中英文），不要写目标语言
- 防过拟合忽略文本：`training_cfg_rate=0.1`（勿设 0）、`weight_decay=0.01`、1–3 epoch 即停
- 推理侧 LoRA 配置必须与训练完全一致；`load_lora` 返回的 `skipped_keys` 应为空
- **训练的 ref 模式与推理的 reference-only 模式结构完全一致**（`packers.process_tts_data_with_ref` = `[103 ref 104][text][101 target 102]`）。带控制前缀时必须走 reference-only：combined 模式会拼成 `prompt_text + "(控制)正文"`，前缀跑到句中就失效（`infer._gen_kwargs` 已按此处理）
- Common Voice 22 官方已从 HF 撤架：数据源已切换到社区镜像 `fsicoli/common_voice_22_0`（非 gated，脚本式数据集，流式下载带 `trust_remote_code`）

## 来自 OmniVoice 生产实践的结论（/Users/dunxu.liu/workspace/others/OmniVoice）
其配音生产链路：基座模型 + 参考音频零样本克隆（不用 LoRA），强依赖 `reference_wav_path` 克隆与逐语言文本归一化。据此：
- **微调方式首选 LoRA**：全量微调更易损害参考音频克隆泛化（生产核心能力）；r=32 说话人适配 / r=64 语言风格适配（本项目默认 64/64/dropout 0.05），`enable_dit: true` 必开
- **数据必须保留克隆能力**：30–50% 样本带同说话人 `ref_audio`（本项目默认 0.4）；干净单人录音，杜绝多人/背景噪声样本
- **试听/推理默认参数**：`cfg_value=2.0`、`inference_timesteps=20`（生产基线 10，本项目默认 20 换更高音质）、`retry_badcase=True`（max_times=3、ratio_threshold=6.0）；音频坏例重试时降 CFG 至 1.2–1.6 并加步数
- 跨语言（中文→Tagalog/泰语）：目标语言数据为主 + 中文 10–20% 混入，防止中文音色能力退化

## 踩坑与约定（已修复问题的沉淀，勿回退）
- **Gradio 流式**：按钮必须直接绑定生成器函数；用 `lambda` 包一层会把生成器对象本身渲染进文本框
- **Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。任何运行后变化的列表（已加工数据集/配置/LoRA/上传目录）必须通过事件输出或 `Tab.select` 刷新
- **`.env` 加载顺序**：`paths.py` 必须**先** `load_dotenv()` **再**计算路径常量（`VOXFT_CKPT_ROOT`/`VOXFT_DATA_ROOT` 依赖此顺序）
- **大盘约定（远程）**：`VOXFT_DATA_ROOT`、`VOXFT_CKPT_ROOT`、`HF_HOME` 都指到 `/root/autodl-tmp/*`；系统盘小，下载/缓存勿落 `~`
- **HF 生态**：`datasets` 锁定 `<4`（5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突）；`hf`/`huggingface-cli` 不读项目 `.env`，命令行需手动 `export`；`snapshot_download` 的进度条不传给单文件，进度监控用缓存目录大小轮询；xet 下载分两阶段（downloading→reconstructing），进度"回退"属正常。**但走镜像必须关 xet**：hf-mirror 只代理 HF API，不代理 xet 的 CAS 服务器（cas-server.xethub.hf.co），reconstruction 阶段会直连并报 401。`paths._disable_xet_on_mirror()` 在 HF_ENDPOINT 非 huggingface.co 时自动 `setdefault("HF_HUB_DISABLE_XET","1")`；命令行用 `hf` 时要自己 export
- **训练默认**：`batch_size=2 + 梯度累积=8`（等效 16，官方示例）防 OOM；启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`save/valid_interval=250`；训练结束自动只保留最新 5 次 LoRA 运行
- **`max_grad_norm=1.0`、`num_workers=8`**：官方 v2 配置就是这两个值（早先本文写的"官方默认 0 = 不裁剪"是错的，已改）。情感语料动态大，不裁剪更容易出梯度尖峰
- **推理**：`load_denoiser=False`（去噪器依赖 modelscope，试听不需要）
- **Whisper 权重**：large-v3 约 3GB，国内直连 huggingface.co 常在 SSL 握手就超时。加载带 3 次重试并打印 endpoint；失败时报错里给了预下载命令。可用 `VOXFT_WHISPER_MODEL` / `VOXFT_WHISPER_MODEL_LARGE` 指向本地目录
- **万级转写必须能断点续跑**：`_transcribe_manifest` 每 300 条把已转好的文本写回原始 `manifest.jsonl`，重跑自动跳过已有文本的行。改这段时别退回"跑完才写一次"
- **数据源首选**（`registry.preferred_sources()`，页面下拉带【首选】）：泰语表现力 `thai_ser` / 泰语口语锚点 `yodas_th`；Tagalog 表现力 `filipino_emotion` / Taglish 锚点 `filswitch`；中文防遗忘 `aishell3`。**锚点不要用朗读语料**（FLEURS/CV22/Porjai）当主力，朗读腔本身就在加重念稿感
- **Tagalog 系不能只认 tl 语种**：短剧台词是 Taglish，句内英文多的样本 Whisper 会判成 en，只认 tl 会把最该保留的 code-switch 样本全部误杀。`Source.languages()` 对 tl 默认放行 `("tl","en")`
- **`yodas_th` 的会话是 utt_id 里的 YouTube video id**（`session_prefix_sep="-"` 取前缀），拼接只在同一视频内按顺序做，等于还原原始连续语流；它中位 3.5s 偏短，靠 `concat_target=8.0` 补上
- **数据**：跨说话人乱配 ref 会损害克隆。无说话人列的源（FLEURS 全系 / 情感语料 / thai20k）现在走 **MFCC 伪说话人聚类**（`pipeline.cluster_pseudo_speakers`，丢 c0 + 语料级均值归一）再配对；聚类日志里簇数接近样本数就说明阈值偏高，调低 `pseudo_speaker_threshold`。Porjai 仍为 0。场景化配比见 README「数据策略」
- **`filipino_speech` 是坑**：22 万条里中位时长 0.63s、`num_words` 中位数 1，且 `speech_type=machine` 占 12.6 万。把孤立单词等间隔粘成长样本会直接训出"报菜名"式念稿声。registry 已自动过滤 `machine` 与 `num_words<4`，拼接改为**同一次录音内 + 句末补标点 + 停顿 0.15–0.5s 抖动**，目标长度 6s
- **`thai_ser` 没有名为 `audio` 的列**（四路麦 `mic_clip/mic_con/mic_middle/mic_zoom`），必须靠 registry 的 `audio_cols` 映射，否则整个源在下载阶段被静默跳过；`mic_zoom` 是网络录音，不用
- **响度**：按说话人整体增益对齐到 −24 dBFS，**不做逐条峰值归一**——逐条归一会把喊叫和耳语拉到同一响度，抹掉"音量=情绪强度"，这是情感语料训完仍然平淡的原因之一
- **加工产物多了几列**（`emotion`/`control`/`f0_std_st`/`energy_std_db`/`rate`/`snr_db`），训练脚本只读 `audio`/`text`/`ref_audio`/`duration`，多余列无害；`f0_std_st` 是"robotic"的量化抓手

## Submodule 升级
```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```
