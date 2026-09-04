# voxft — VoxCPM 2 微调工作台

基于 [VoxCPM 2](https://github.com/OpenBMB/VoxCPM) 的微调工作台：Tagalog/泰语高质量语料加工、跨语言（中文→目标语言）混合微调、wandb 监控、LoRA merge、HuggingFace 同步。

> 开发与代理协作约定见 [AGENTS.md](AGENTS.md)。

## 快速开始

```bash
git clone --recurse-submodules <repo>   # 必须带 --recurse-submodules
uv sync                # Mac 开发机（含 torch CPU/MPS 轮子）
uv run voxft-ui        # 打开微调工作台：http://0.0.0.0:6006
```

已经克隆过但 `third_party/VoxCPM` 是空目录的（训练预检报"官方训练脚本不存在"），补一条：

```bash
git submodule update --init --recursive
```

GPU 服务器（Linux）上同样 `uv sync`（自动解析 cu124 轮子）；大数据目录建议指到大盘：

```bash
echo 'VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data' >> .env
echo 'VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt' >> .env
# HF 缓存（基座模型 + 数据集分片）默认在 ~/.cache，系统盘容易满，也指到大盘：
echo 'HF_HOME=/root/autodl-tmp/hf_home' >> .env
```

> 命令行手动下载时 `hf`/`huggingface-cli` 不读 `.env`，需先 `export HF_HOME=/root/autodl-tmp/hf_home`。

## 密钥

复制 `.env.example` 为 `.env` 并填写：
- `HF_TOKEN` — 下载受限数据集与上传模型/数据集（默认仓库 `FrankLiuDundun/voxcpm-finetune-lora`）。Common Voice 22 官方已撤架，现走社区镜像，无需同意条款
- `WANDB_API_KEY` — 训练时自动把 TensorBoard 指标/验证音频桥接到 wandb
- `VOXCPM_BASE_PATH` — 本地基座目录（不填默认从 `openbmb/VoxCPM2` 拉取）

## 典型流程（中文 → 泰语/Tagalog）

1. **数据集页**：选数据源（泰语首选 CMKL Porjai 700h；Tagalog 首选 FLEURS+CV22）→ 下载 → 加工（16k/裁尾静音/响度归一化/3-30s/可选 UTMOS·Whisper 质检）
2. **混合**：目标语言为主 + 中文防遗忘（短剧泰语+Tagalog 场景见下方「数据策略」）
3. **训练页**：生成配置（默认 LoRA，r=32/64）→ 本机启动或复制命令到 GPU 机；中断后用同一配置重启即自动从 `latest/` 断点续训
4. **验证效果**：看 `loss/diff`、`val/loss` 曲线（wandb/TensorBoard）→ 听验证音频 → 试听页 A/B 对比各 checkpoint → 必要时 Whisper 回写核对文本贴合度；出现"忽略文本/停不下来"即回退更早 checkpoint
5. **模型管理页**：merge LoRA 导出完整模型 → 上传 HuggingFace

## 数据策略（短剧配音：泰语 + Tagalog）

场景：短剧翻译配音，音色靠**参考音频零样本克隆**（对齐 OmniVoice 生产），LoRA 只做语言风格适配，不能伤害克隆能力。

**数据源优先级**

| 语种 | 首选 | 补充 |
|---|---|---|
| 泰语 | `cv22_th`（量大、有 `client_id`）+ `fleurs_th`（干净锚点） | `thai20k`、Porjai 700h（录音棚级，体积大先小样本试） |
| Tagalog | `cv22_tl` + `fleurs_tl` | 其余候选见下 |

**关键规则**
- **无说话人列的数据源（FLEURS/thai20k/Porjai）：加工时 `ref_audio 比例` 必须设 0**——跨说话人乱配会教模型忽略参考音色，直接损害克隆；有 `client_id` 的 CV22 才用默认 0.4
- CV22 噪音多：加工必开 **UTMOS（≥3.5）+ Whisper 转写校验**（语言选 th/tl），先在 HF 页面同意条款并配 `HF_TOKEN`
- `tagalog_tts` / `filipino_emotion` 无文本列，暂不可直接用（需先 Whisper 转写）
- 一个 LoRA 覆盖两语种：混合权重建议 **泰语 45% + Tagalog 45% + 中文 10%**（中文仅防灾难性遗忘，不需要中文输出可降到 5%）
- **验收以克隆优先**：同一参考音频，基座 vs LoRA A/B 对比，音色还原度不降 + 目标语言发音更准才算通过；音色掉就减步数或降学习率

## 命令速查

```bash
uv run python -m voxft.data.download --source fleurs_th --max-samples 100
uv run python -m voxft.data.pipeline --source fleurs_th --max-items 50 --utmos-min 3.5
uv run python -m voxft.train.launcher configs/xxx.yaml 1   # 打印训练命令
uv run python -m voxft.lora.merge --lora-dir checkpoints/run/latest --out checkpoints/merged
uv run pytest
```

## 数据源许可速览

| 数据源 | 许可 |
|---|---|
| CMKL Porjai（泰 700h） | CC-BY-SA-4.0 |
| FLEURS | CC-BY-4.0 |
| hotdogs/thai-speech-20k | CC-BY-4.0 |
| Common Voice 22（社区镜像 fsicoli） | CC0（官方已撤架，镜像无需条款） |
| AISHELL-3 | Apache-2.0 |
| welyjesch/tagalog_tts、filipino-emotion-tts | 未知，商用前核实 |
| Speech-data/Filipino-Tagalog | CC-BY-NC-ND（非商用） |
