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

1. **数据集页**：选数据源（表现力主力：泰语 `thai_ser`、Tagalog `filipino_emotion`；锚点：FLEURS/CV22）→ 下载 → 加工（16k/裁静音/3-30s/质检/表现力指标/伪说话人聚类/按说话人响度对齐/控制前缀/ref 配对）
2. **混合**：目标语言为主 + 中文防遗忘（短剧泰语+Tagalog 场景见下方「数据策略」）
3. **训练页**：生成配置（默认 LoRA r=64/alpha=64/dropout=0.05，`max_grad_norm=1.0`）→ 本机启动或复制命令到 GPU 机；中断后用同一配置重启即自动从 `latest/` 断点续训
4. **验证效果**：看 `loss/diff`、`val/loss` 曲线（wandb/TensorBoard）→ 听验证音频 → 试听页 A/B 对比（可填情绪 prompt，种子固定）→ `voxft.eval` 看贴合度/截断率/语调起伏；出现"忽略文本/停不下来"即回退更早 checkpoint
5. **模型管理页**：merge LoRA 导出完整模型 → 上传 HuggingFace

## 数据策略（短剧配音：泰语 + Tagalog，目标：自然、有情绪、不像念稿）

场景：短剧翻译配音，音色靠**参考音频零样本克隆**（对齐 OmniVoice 生产）。**首要目标是自然、有情绪、不像念稿**，其次才是发音准。

> 先分清责任层：**同角色跨句跨集变声、角色选角不贴、翻译用词、非语言发声（笑/痛呼）、整体音量**都不是微调能解的，属于配音链路（固定 voice bank + 固定 seed + ref 去噪 + MT 层）。微调能解的是：**念稿感 / 目标语言发音与重音 / Taglish 句内英文词 / 词被吃掉 / 语速失控 / 响应情绪 prompt**。

**两类语料，作用不同（别混为一谈）**
- **朗读语料**（FLEURS / CV22 / Porjai）：教发音准、吐字清，是"锚点"。但它们本身是念稿风格，**占比过高会加重念稿感**。
- **情感/口语语料**（`thai_ser` / `filipino_emotion`）：教语速起伏、停顿、情绪、语气——**这才是去念稿感的主力**，应占目标语言混合的 30–50%。

**数据源优先级**（页面下拉里带【首选】标记）

| 语种 | 表现力主力（去念稿） | 发音/口语锚点 | 其余可选 |
|---|---|---|---|
| 泰语 | **`thai_ser`** 首选 — airesearch/thai-ser，2.8 万条/41h，200 名演员、5 情绪、有 `actor_id`；`turn_type=impro` 是即兴对话，最值钱。CC-BY-SA-4.0 | **`yodas_th`** 首选 — Chalermdej/yodas2_sidon_th_tts，14 万条/156h **YouTube 真实口语**（不是朗读腔），4199 说话人，DNSMOS + 三路 ASR 交叉校验分级，**CC-BY-3.0 商用友好** | `cv22_th`、`fleurs_th`、`porjai_th`（都是朗读腔，配比别高） |
| Tagalog | **`filipino_emotion`** 首选 — 1.1 万条、6 情绪、中位 3.0s，无文本→自动转写 | **`filswitch`** 首选 — 2.7K 条 Taglish，句内中英混杂，中位 8.5s（时长分布最贴 VoxCPM），教的是「RAW 该怎么念」 | `fleurs_tl`；`filipino_speech`（见下方警告） |
| 中文 | — | — | **`aishell3`** 首选（防遗忘，有说话人可 ref 配对）、`fleurs_zh` |

> 泰语两个首选是互补的，不是二选一：`thai_ser` 是**演员表演的情绪**，唯一能喂控制前缀（有情绪标签）；`yodas_th` 是**真人日常语流**，替掉 FLEURS/CV22 那类朗读锚点。锚点用朗读语料本身就在加重念稿感。

**扫过但没选的**（记录一下，别重复调研）
- Tagalog：`SilencioNetwork/tagalog-filipino-speech` —— 100% 自由说话、人工校验转写、**带词级时间戳**（能在真实停顿处切分），质量是全场最好的，但只有 90 条/2 小时且 **CC-BY-NC-4.0 禁商用**，只能做实验对照；`RidheshBhati/filipino-tts-10k-final` 是圣经/文学朗读，对念稿感是负作用；`Nexdata` 1100h 全双工对话只在 HF 放了无转写的 sample，要付费；YODAS2 的 230 个语种目录里**没有 tl/fil**，这块公开数据是真空。
- 泰语：`Saltywan/Thai-Duplex-Bench`（双工对话，但 viewer 关闭、是 bench 不是训练集）、`Nexdata` 211h 全双工（付费）、`thai_elderly_speech`（老年语音，属选角问题不是微调问题）、`dubbing-ai/vaja-thai`（需授权）。

**去念稿感三杠杆**（微调只是其一，需并行）
1. **情感语料 + 控制前缀微调**：把模型韵律先验往"有起伏"推，并让它在目标语言下听得懂中英文情绪指令。
2. **参考音频必须有情绪**：克隆输出韵律主要由参考音频决定，务必用**带情绪的台词**做参考，别用平淡朗读；每个角色一条固定 5–10s 干净 ref，全剧全集复用（这条同时解决"变声"）。
3. **推理参数放松**：`cfg_value` 偏高会更贴文本但更僵，试 1.2–1.6（生产 2.0）；seed 固定。

**关键规则**（加工页已**自动配置**，无需手选）
- **控制前缀**：VoxCPM2 的情绪控制就是文本前缀 `(控制指令)正文`，没有独立条件通道。加工会按情绪标签 + 实测语速/音量，给 25–50% 样本自动生成中英文前缀（如 `(愤怒地，语速快)`），其余保持裸文本。**训练文本全裸会把基座的情绪 prompt 能力冲掉**——这是"微调后情绪反而更不灵"的主因
- **ref 配对**：有说话人列的源自动 0.4 同说话人配对；无说话人列的源（FLEURS 全系 / `filipino_emotion`）走 MFCC **伪说话人聚类**后再配对；ref 候选限定 3–10s 且优先高信噪比，对齐线上参考音频分布
- **响度**：按说话人整体增益对齐到 −24 dBFS，不做逐条峰值归一（逐条归一会抹掉"音量=情绪强度"）
- **表现力指标**：每条样本落 `f0_std_st`（语调起伏，半音）/ `energy_std_db` / `rate` / `snr_db`，可用 SNR 与 f0 起伏门限筛掉噪声样本和平读样本
- 质检自动分档：干净源不质检；众包/未知源自动 Whisper 转写校验；无文本列的源自动 large-v3 转写 + 语种过滤（转写结果写回原始清单复用）
- 一个 LoRA 覆盖两语种：建议 **泰语 45% + Tagalog 45% + 中文 10%**；每个目标语言内部，情感语料占 30–50%；混合时小语料重复上限 3×
- **验收看三个数**（`voxft.eval`）：文本贴合度↑、**截断率↓**（"词被吃掉"）、**语调起伏↑**（"robotic"）；再用同一有情绪的参考音频做基座 vs LoRA A/B

## 命令速查

```bash
uv run python -m voxft.data.download --source fleurs_th --max-samples 100
uv run python -m voxft.data.pipeline --source thai_ser --max-items 200 --control-ratio 0.5 --min-snr-db 12
uv run python -m voxft.train.launcher configs/xxx.yaml 1   # 打印训练命令
uv run python -m voxft.lora.merge --lora-dir checkpoints/run/latest --out checkpoints/merged
# 客观评测：贴合度↑ / 截断率↓ / 语调起伏↑（需 --group qc）
uv run python -m voxft.eval base checkpoints/run/latest --lang tl \
    --ref-audio ref.wav --control "愤怒地，语速快"
uv run pytest
```

## 数据源许可速览

| 数据源 | 许可 |
|---|---|
| THAI-SER（泰语情感 41h） | CC-BY-SA-4.0（SA 有传染性，商用前确认权重分发口径） |
| YODAS2-Sidon 泰语 TTS 精选（156h） | CC-BY-3.0（署名即可，商用友好） |
| qwerttyuiiop/FilSwitch（Taglish） | 未声明，商用前核实 |
| CMKL Porjai（泰 700h） | CC-BY-SA-4.0 |
| FLEURS | CC-BY-4.0 |
| hotdogs/thai-speech-20k | CC-BY-4.0 |
| Common Voice 22（社区镜像 fsicoli） | CC0（官方已撤架，镜像无需条款） |
| AISHELL-3 | Apache-2.0 |
| welyjesch/tagalog_tts、filipino-emotion-tts | 未知，商用前核实 |
| sapinsapin/filipinospeechcorpus | MIT（但绝大部分是孤立单词，见「数据策略」警告） |
| Speech-data/Filipino-Tagalog | CC-BY-NC-ND（非商用） |
