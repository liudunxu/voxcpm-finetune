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
uv run python -m voxft.eval base checkpoints/<run>/latest --lang th   # 效果客观对比（需 --group qc）
uv run pytest                        # 测试
# 训练（在 GPU 机器上，由页面生成的命令）：
cd third_party/VoxCPM && torchrun --nproc_per_node=N scripts/train_voxcpm_finetune.py --config_path <生成的yaml>
```

## 标准工作流
1. 数据集 Tab：选数据源 → 下载 → 加工（16k 重采样 / 裁尾静音 / 响度归一化 / 3–30s 过滤 / 可选 UTMOS·Whisper 质检 / ref_audio 配对）→ 输出 JSONL
2. 混合：目标语言 80–90% + 中文 10–20%（防灾难性遗忘），按行重复拼接
3. 训练 Tab：填表单 → 生成 YAML → 启动；wandb 桥接自动转发指标
4. 试听 Tab：加载 checkpoint 试听；模型管理 Tab：merge LoRA / 上传 HF

## 微调铁律（来自官方文档/FAQ，不要违反）
- VoxCPM 2：`sample_rate=16000`（AudioVAE 编码器输入）、`out_sample_rate=48000`（仅推理）
- 数据 JSONL 字段：`audio`、`text` 必填；`ref_audio`（同说话人）30–50% 样本；时长 3–30s；尾静音 <0.5s（否则"生成停不下来"）
- LoRA：lr=1e-4、`enable_dit: true`（音质关键）、r=32（说话人）/64（语言风格）；全量：lr=1e-5（LoRA 的 1/10）
- 防过拟合忽略文本：`training_cfg_rate=0.1`（勿设 0）、`weight_decay=0.01`、1–3 epoch 即停
- 推理侧 LoRA 配置必须与训练完全一致；`load_lora` 返回的 `skipped_keys` 应为空
- Common Voice 22 官方已从 HF 撤架：数据源已切换到社区镜像 `fsicoli/common_voice_22_0`（非 gated，脚本式数据集，流式下载带 `trust_remote_code`）

## 来自 OmniVoice 生产实践的结论（/Users/dunxu.liu/workspace/others/OmniVoice）
其配音生产链路：基座模型 + 参考音频零样本克隆（不用 LoRA），强依赖 `reference_wav_path` 克隆与逐语言文本归一化。据此：
- **微调方式首选 LoRA**：全量微调更易损害参考音频克隆泛化（生产核心能力）；r=32 说话人适配 / r=64 语言风格适配，`enable_dit: true` 必开
- **数据必须保留克隆能力**：30–50% 样本带同说话人 `ref_audio`（本项目默认 0.4）；干净单人录音，杜绝多人/背景噪声样本
- **试听/推理默认参数**：`cfg_value=2.0`、`inference_timesteps=20`（生产基线 10，本项目默认 20 换更高音质）、`retry_badcase=True`（max_times=3、ratio_threshold=6.0）；音频坏例重试时降 CFG 至 1.2–1.6 并加步数
- 跨语言（中文→Tagalog/泰语）：目标语言数据为主 + 中文 10–20% 混入，防止中文音色能力退化

## 踩坑与约定（已修复问题的沉淀，勿回退）
- **Gradio 流式**：按钮必须直接绑定生成器函数；用 `lambda` 包一层会把生成器对象本身渲染进文本框
- **Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。任何运行后变化的列表（已加工数据集/配置/LoRA/上传目录）必须通过事件输出或 `Tab.select` 刷新
- **`.env` 加载顺序**：`paths.py` 必须**先** `load_dotenv()` **再**计算路径常量（`VOXFT_CKPT_ROOT`/`VOXFT_DATA_ROOT` 依赖此顺序）
- **大盘约定（远程）**：`VOXFT_DATA_ROOT`、`VOXFT_CKPT_ROOT`、`HF_HOME` 都指到 `/root/autodl-tmp/*`；系统盘小，下载/缓存勿落 `~`
- **HF 生态**：`datasets` 锁定 `<4`（5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突）；`hf`/`huggingface-cli` 不读项目 `.env`，命令行需手动 `export`；`snapshot_download` 的进度条不传给单文件，进度监控用缓存目录大小轮询；xet 下载分两阶段（downloading→reconstructing），进度"回退"属正常
- **训练默认**：`batch_size=2 + 梯度累积=8`（等效 16，官方示例）防 OOM；启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`save/valid_interval=250`；训练结束自动只保留最新 5 次 LoRA 运行
- **推理**：`load_denoiser=False`（去噪器依赖 modelscope，试听不需要）
- **数据**：无说话人列的数据源（FLEURS 全系 / thai20k / Porjai）加工时 `ref_audio` 比例必须 0，跨说话人配对会损害克隆；场景化配比见 README「数据策略」

## Submodule 升级
```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```
