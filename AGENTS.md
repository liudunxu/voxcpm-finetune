# AGENTS.md

## 项目简介
VoxCPM 2（OpenBMB TTS）微调工作台：Tagalog/泰语/越南语/印尼语/马来语高质量语料的下载与加工、跨语言（中文→目标语言）混合微调、LoRA/全量训练管理、wandb 监控、LoRA merge、HuggingFace 同步。**目标是五语种（th/tl/vi/id/ms）联合微调：一个 LoRA 同时提升这几种语言的配音质量，但验收必须分语种做**。Gradio 页面端口 **6006**。

## 环境
- Python 3.11（.python-version 已固定），依赖由 **uv** 管理：`uv sync`（本地开发）、`uv sync --group qc`（启用 whisper 质检 + PyAV 视频解码）。
- torch 平台分流（见 pyproject `[tool.uv.sources]`）：macOS → PyPI 轮子（CPU/MPS）；Linux → pytorch-cu124 index（CUDA 12.4）。训练只在 Linux GPU 机执行。
- **数据集下载只在远程 GPU 机器进行**：本地开发环境不下载数据集，也无需本地验证下载流程。
- 密钥走 `.env`（复制 `.env.example`）：`HF_TOKEN`、`WANDB_API_KEY`、`WANDB_PROJECT`、`VOXCPM_BASE_PATH`、`HF_ENDPOINT`（国内默认 `https://hf-mirror.com`，导入 voxft 即自动加载）。

## 远程 GPU 机调试约定（本轮实测沉淀，照这个方式来）

远程机是 autodl 容器（RTX 4090D 48G / 1TB 内存），项目在 `/root/voxcpm-finetune`，大盘在 `/root/autodl-tmp`。**系统盘只有 30G，任何下载/缓存/产物都不许落 `~` 或项目目录。**

### 接入与执行封装
- 密码登录先转免密：`brew install hudochenkov/sshpass/sshpass` → `sshpass -p '<密码>' ssh-copy-id -p <端口> -i ~/.ssh/id_ed25519.pub root@<主机>`。之后一律用公钥，别把密码写进每条命令。
- **`uv` 不在非交互 SSH 的 PATH 里**（在 `/root/.local/bin/uv`，`.bashrc` 不被加载），直接 `ssh host 'uv run ...'` 会 `command not found`。封装一个 wrapper，把 PATH、工作目录、按需的代理一次性带上：
  ```bash
  cat > /tmp/rsh <<'EOF'
  #!/bin/bash
  exec ssh -p <端口> -o StrictHostKeyChecking=no -o ServerAliveInterval=30 root@<主机> \
    "export PATH=/root/voxcpm-finetune/.venv/bin:/root/.local/bin:\$PATH; cd /root/voxcpm-finetune; $*"
  EOF
  chmod +x /tmp/rsh
  ```
  用 `$*` 直接拼命令时，别在本机用 `SSH="ssh ..."; $SSH '...'`——zsh 会把整串当成一个命令名，报 `command not found`。
  **PATH 必须同时带 `.venv/bin`**：`launcher.gpu_command` 生成的是裸 `python <train脚本>`（UI 走 `start_local` 时继承已激活环境所以没事），非交互 SSH 下只有 `uv` 可用、`python` 不存在，直接 nohup 会静默失败成一行 `python: command not found`。
- **代理按用途分流，不要无脑 `source /etc/network_turbo`**：它是海外学术加速，只对 github / pypi 官方源 / `huggingface.co` 直连有用。本项目 `.env` 默认走国内镜像 hf-mirror，**套上代理反而慢一倍以上**（实测 4.55 → 2.43 MB/s，劣化时 0.22 MB/s）。所以：下载数据集/模型 → 不开代理；`git pull` submodule、装非镜像源依赖 → 才开。详见「下载通道实测矩阵」。
- 需要交互式登录的命令（`gh auth login` 之类）让用户在会话里用 `! <命令>` 前缀自己跑，别代跑。
- **实例随时会没**：autodl 容器停机后表现为「网关 `ping` 通、SSH 端口 `Connection refused`」（`nc -z <主机> <端口>` 一秒就能确认，别反复重试 ssh）。重启后端口/主机名可能变。因此**代码与文档改动要边做边 `rsync` 上去，不要攒到最后一次性同步**——本轮就是在最后同步 AGENTS.md 时实例停了，远端因此落后一个版本。连不上先找用户确认实例状态。

### 长任务：一律 nohup 落日志，别占着 SSH
- 下载/加工/训练/eval 都可能几十分钟，SSH 断了任务就没了。固定写法：
  ```bash
  /tmp/rsh 'rm -f /tmp/x.log; nohup stdbuf -oL -eL uv run python -u -m <模块> ... > /tmp/x.log 2>&1 & echo started'
  ```
  然后**另起命令**轮询 `tail /tmp/x.log`。`python -u` + `stdbuf -oL` 缺一不可，否则日志憋在缓冲区里看着像卡死。
- **绝对不要把长任务管道给 `tail -N`**：`cmd | tail -25` 会等进程结束才输出，中途完全看不到进度——本轮因此误判过"下载没动静"。
- 多步循环写成一个脚本 `scp` 上去再 `nohup bash /tmp/x.sh`，比在 ssh 命令串里塞 `for` 循环可靠得多。

### `pkill -f` 会杀掉自己的 SSH 会话
`pkill -f "voxft.data.download"` 的模式会匹配到**承载这条命令的远端 shell 自身**（命令行里就含这个字符串），结果 ssh 直接 exit 255、任务状态不明。本轮踩了两次。两个安全写法：
- 拆开字符串让它匹配不到自己：`pgrep -f "vo""xft.data.download"`；
- 或者先 `pgrep -af` 拿到 pid，再按 pid `kill`。

⚠️ **拆字符串只保护 pkill 那一处，不够**：匹配的是整条远端命令行，所以**同一次 ssh 里任何位置**出现完整字面量都会自杀。本轮第二次就是这么栽的——`pkill -f "vo""xft-ui"` 拆对了，但同一条命令后面还有 `nohup ... uv run voxft-ui`，模式照样命中自己的 shell，整条命令输出全空、UI 也没起来。规矩：**杀进程和启动进程分成两次 ssh**，别写在一条命令里。

杀完务必复查 `pgrep -af`，确认没有残留进程还在写盘/占显存。

### 判断"卡死 vs 慢"的三板斧
现象一样（长时间无输出），成因完全不同，别靠猜：
1. `cat /proc/<pid>/io | grep rchar` 间隔 20–40s 取两次差值——**零增长才是真卡死**，在涨就是慢。
2. `ss -tnp | grep <pid>` 看有没有活连接；没有连接 + 零 IO = 阻塞在内部锁或非网络等待。
3. Python 侧用 `faulthandler.dump_traceback_later(90, exit=True)` 让进程自己吐栈。本轮就是靠它定位到流式下载卡在 `h11/_connection.py`（`load_dataset` 已返回，是取首行时挂的）。
辅助：`du -sb $HF_HOME` 轮询缓存增量、`ls -la blobs/*.incomplete` 看分片是否在长、`nvidia-smi` 看显存。

### 实测优先：两条反面教训
- **别拿突发速率当持续速率**。用 `curl -r 0-30000000` 测 30MB range 得到 2.06 MB/s，据此判断"直连比镜像快"，结果持续速率只有 1.0 MB/s（比镜像的 1.51 还慢）。要测就间隔取两次累计量算差值。
- **假设要被实验否掉就老实改掉结论**。本轮先假设"代理 + 镜像互相拖慢、该切直连"，切过去 xet 直接 CAS 断连失败；又假设"soundfile 0.14 改了 `sf.read` 返回类型、`>=0.12` 下界有风险"，装 0.12.1 实测发现它**一直**返回 `(data, sr)` tuple，根本没有这个 bug。两条都写进过草稿，都因实测被推翻——**低资源语种/依赖行为这类结论一律核实到原文或跑一遍，不要凭记忆**（与已有那条铁律同理）。

### 产物卫生
- 测试用的合成音频**绝不能留在真实数据源目录**里。本轮为测 `ingest` 用 FLEURS 音频拼了个 mp4，产物落进了 `raw/drama_vi/`——那是真实的越南语训练源，必须整目录删掉。`--max-items` 试跑不回写清单，所以污染只限该目录，但仍要清。
- 切换 `HF_ENDPOINT` 或 xet 模式后，旧的 `blobs/*.incomplete` 会变孤儿（后缀不同、不续传），重跑前手动清。
- 冒烟 run 的 `step_*` checkpoint 每份约 430MB，测完删掉；`merged_*` 每份 4.6GB，验证完立刻删。
- **不要 `cat .env`**：里面有 `HF_TOKEN`/`WANDB_API_KEY`，会原样进对话记录。要看配置就 `grep -v TOKEN` 或只看键名。

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
# 验收要的是「同一份多语种 case、base 与 checkpoint 同口径 A/B」，看报告 by_lang 有无语种退化：
#   cases.jsonl 每行 {"text": ..., "lang": "th|tl|vi|id|ms", "ref_audio"?, "control"?, "seed"?}
uv run python -m voxft.eval base /root/autodl-tmp/voxft_ckpt/<run>/latest \
    --lang th --texts-file cases.jsonl --seeds 42
# 大 case 集并行加速：N 个进程各跑一片（K 从 0 起），case_id 保持原始序号，跑完合并：
uv run python -m voxft.eval base --texts-file cases.jsonl --seeds 42 43 44 --shard 0/3
uv run python -m voxft.eval --merge <shard报告1.json> <shard报告2.json> <shard报告3.json>
uv run pytest                        # 测试（testpaths=tests，不会去收 third_party 的官方脚本）
# 训练（在 GPU 机器上，由页面生成的命令）：
cd third_party/VoxCPM && torchrun --nproc_per_node=N scripts/train_voxcpm_finetune.py --config_path <生成的yaml>
```
> 上面 download / ingest / eval / 训练 都是分钟到小时级：在远程**一律按「远程 GPU 机调试约定」用 `nohup ... > /tmp/x.log 2>&1 &` 落日志再轮询**，不要占着 SSH，也不要管道给 `tail`（会等进程结束才输出，中途看着像卡死）。

## 标准工作流
1. 数据集 Tab：下载或导入已审核 JSONL → 16k / 裁静音 / 3–30s / 质检 / 声学描述 → 按身份、会话、原音频隔离 train/val → 已验证说话人整体增益 → 可信控制前缀 → split 内 ref 配对
2. 素材导入 Tab（自备成片）：喂视频/音轨 → 自动切分转写 → 逐条试听淘汰 BGM 重的、标说话人与情绪 → 追加进原始清单 → 可选自动加工；钉一集进 `holdout.json` 当固定验证集
3. 混合：**联合口径为每个目标语种各 17 分 + 中文 10 + 英文 5**（回放全局共享，不按语种各配一份；实验起点），**按有效音频时长**采样；原始目标音频最多 3×（含嵌套混合），ref 复用另统计，验证集不重复，实际占比/曝光/联合覆盖写进 mix.json，**各语种「请求 vs 实际」占比写进 `language_shares`**——某语种吃不满会经 progress 告警，缺口不重分配给其他语种
4. 训练 Tab：填表单 → 生成 YAML → 启动；wandb 桥接自动转发指标
5. 试听 Tab：加载 checkpoint 试听；模型管理 Tab：merge LoRA / 上传 HF

## 每轮微调的固定动作（规范，别跳过）
1. **提交记录就是实验记录**：每轮微调的 commit message 必须写全——数据（源 id + 许可 + 时长 + 条数）、
   配比、超参（r/alpha/dropout/lr/batch×累积/epochs/实际步数/val loss 首末值）、
   **分语种验收数字（base → ckpt 的 CER/WER/疑似漏尾，逐语种列出）**、结论（通过 / 不通过 + 原因 + 下一步）。
   日后回溯不该还需要去翻 wandb 或远程日志。跑多轮时按时间顺序读 commit 就能还原全部实验。
2. **每轮都要顺手审视代码层面**：跑流程时遇到的别扭处当轮处理掉，不留"以后"——重复字面量、失效或误导的
   默认值、只在某种 shell 下能跑的命令、量纲不对的指标、只在 UI 路径下才成立的假设。
   能修的修（**并补对应测试**），不适合修的明确记进「踩坑与约定」并写清为什么不修。
3. **质量评估双轨，人工盲听是最终判据**：
   - 离线轨：`voxft.eval` 的 CER/WER/疑似漏尾 + `by_lang`，**必须 ≥3 个 seed**（原因见「单 seed 会给出相反结论」）
   - 人工轨：6006 工作台**「盲听评估」Tab** —— 选两份*同 case 集、同 seed 组*的报告（A 一般是 `base_*`，
     B 是 checkpoint），甲/乙顺序随机且不暴露来源，逐条打自然度/可懂度/截断/噪音，
     「汇总并写回」把评分落进两份报告各自的 `human_review`，并给出分语种 B 胜/平/负
   - 判据（**两轨都必须带最小差值门槛，否则单条波动就能否决整轮**）：
     盲听要求 `B 负 − B 胜 ≥ 2` 且 `B 负 ≥ 3` 才算该语种退化（`eval.BLIND_LOSS_MARGIN` /
     `BLIND_MIN_LOSSES`）；离线 CER 要求 `ΔCER > 0.05`（`runlog --noise`）。
     实测依据：本轮盲听 `id` 拿到 0胜/14平/1负、`zh` 拿到 0胜/2平/1负，按朴素的「负 > 胜」
     两条都算退化，但那只是 12-18 条样本里的单条听感波动；离线侧同一份配比跑两轮
     `ms` 的 CER 差出 0.042。**红线喊多了就等于没有红线**，但门槛也不能高到放过成片退化，
     真正的修法是把统计功效提上来（每语种 30+ case、5 seed），不是继续调门槛。
     两轨冲突时**以盲听为准**
   - **闭环**：盲听发现的失败形态必须落成新的 case 写进 `eval_cases/`（带 `note` 说明为什么加），
     让下一轮能自动复现——这是"根据反馈提升下一次微调"的唯一可靠机制，口头结论不算
   - 离线指标只作诊断的理由：ASR 误差不等于发音错误、疑似漏尾不等于真实截断、F0 不是越高越好、
     `rate` 的量纲还随语种不同（th 字符/秒、vi 音节级、tl/en/id/ms 词级），不能横向比

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
- 跨语言（中文/英文→Tagalog/泰语/越南语/印尼语/马来语）：**联合模型是目标，分语种验收是前提**（原「先分别训各语种 LoRA，不要一上来做联合模型」已作废）。基座无语种槽位，联合在训练侧就是普通多源混合，不改 submodule 也不改超参。归因靠四条硬规则：① 训练前先用同一份多语种 case 跑 `eval base` 拿分语种基线；② 看报告 `by_lang`，**任一语种相对基座退化即整轮不通过**，不许用「平均变好」掩盖；③ 至少给 TH 留一个分语种 LoRA 作对照，用来区分「联合不如单语」与「该语种数据不够」；④ 某语种退化或盲听判差 → 该语种退回独立 LoRA，其余继续联合。配比：每个目标语种各 17 分 + 中文 10 + 英文 5（全局共享，不按语种各配一份），**等权而非按数据量分配**，缺口不重分配给其他语种。数据准备顺序 TH → TL → ID → VI → MS（是准备顺序不是训练顺序）。vi/id/ms 首轮表演档为 0，只验发音与口语韵律，不声称情绪改善

### 线上真实调用口径（本轮从 OmniVoice 源码与归档 payload 逐条核实，微调要对齐这个）
OmniVoice 是 GPU 侧合成服务；配音编排在另一个仓库 `dubbing_intelligence_service`。`voxcpm/` 是**推理包的 vendored 副本**，与本项目 submodule 的 `load_lora_weights` 实现完全一致。
- **推理参数**：唯一实例化点 `api.py:3004-3021`，`VOXCPM_MODEL_ID` 默认 `openbmb/VoxCPM2`，`load_denoiser` 默认关、`optimize` 默认开。归档生产 payload（`results/voxcpm_quality_cases.json`）实测：`cfg_value=1.8`（**不是本项目默认的 2.0**）、`voxcpm_cfg_auto=True`、`inference_timesteps` 走服务端默认 20、`retry_badcase=False`、`retry_badcase_max_times=1`、`best_of=1`、`max_generation_attempts=1`、`trim_silence_vad=True`、seed 固定。**验收一律 `--cfg-value 1.8`**，用 2.0 测出来的数字不代表线上
- **seed 的精确语义（2026-09-15 读两个仓库源码核实）**：主调方每请求必传 seed（`dubbing_intelligence_service` `backends/voxcpm.py:812-822`）。默认 `VOXCPM_SPEAKER_STABLE_SEED=true` 时是 **per-speaker stable seed**——`sha256(speaker + model_id + cfg + steps + …)` 派生、**不含文本**（`backends/voxcpm.py:438-463`），**同一说话人的所有 cue 共享同一个 seed**；每个 speaker 首 take 只生活在一个 seed 上，跨 seed 方差在首 take 不可感知。**但重试路径大多换 seed**：模型内 retry_badcase **+1**（`voxcpm2.py:734`）、prompt-leak +1、speaker-mismatch +1、text-regen **+7**、best_of 候选 **+1009**、主调方整轮重试 crc32 重派生；**例外：metallic 质量重试不换 seed**，只降 cfg 加 steps（`api.py:9184-9196`，`api.py:7895` 那句"use a fresh seed"注释与代码不符，以代码为准）。实际生效 seed 经 `effective_seed` 回传、主调方记 `seed_drifted`。⇒ 跨 seed 方差的真实含义是「重试把好 take 换成坏 take」的风险，离线量它要用**相邻偏移 seed**（base/+1/+7）而不是无关大跨度 seed；首 take 稳定性该量的是**跨 cue 一致性**（同 speaker 同 seed 不同文本）
- **控制前缀线上根本不用**：60/60 份归档响应都是 `control_instruction: ""`、`control_instruction_applied: false`。只有独立的 voice-design 端点会用，且是**英文散文式**描述（`"a calm natural supporting-character voice"`），不是中文情绪标签。所以本轮训练 `with_control=0` 与线上一致，不是缺陷；`Options.control_zh_ratio=0.5` 与线上不符，等有可信标签时再改
- **参考音频**：按请求传 base64 或用 `voice_id` 命中缓存（`api.py:3054-3073`）。实测捕获到的 ref 是 **24kHz 单声道 4.47s**，来自主调方音色库 `group_f_fleurs.wav`（菲律宾语），**跨语言复用**——归档里那条泰语台词用的是同一条菲律宾语 ref。这正是 `pair_references` 三级优先里「跨语言 ref」那一档要复现的形态。cfg/steps 会按 ref 时长自适应（`api.py:7780-7799`）：<2s→28/1.9、2-4s→24/2.0、**4-5s→20/2.1**、>5s→22/2.2；ref 落盘前做 WPE 去混响 + 对齐到 −18dB active + 裁边，**不重采样**
- **文本形态**：一请求一个 cue、字幕长度（`"Hindi ko inaasahan na babalik ka pa."` 实测输出 2.395s），`MAX_TEXT_LEN=2000`，**不切分**，保留原始标点与大小写；行首 `(...)` 被保留给控制指令（`api.py:8118`）。→ **训练文本必须带标点与大小写**，这就是 `_TEXT_COLS` 把 `raw_transcription` 排在 `transcription` 之前的原因（FLEURS 的 `transcription` 是全小写去标点的归一变体，fil/vi/id/ms 两列 100% 不同，th 742/2602 不同；**两列的数字序列 100% 一致**，所以这个改动不碰数字，只补标点与大小写）
- **语种只喂文本归一化，不进模型**：`language`/`target_lang` 只传给 normalizer，没有 language token。TN 分派（`voxcpm/utils/text_normalize.py:364-403`）：zh→zh、en→en、**fil→filipino**（会把数字 verbalize、把 `’di→hindi`、`n’yo→ninyo` 展开）、**vi/id/ms/th→generic 原样透传**。⇒ **数字对 th/vi/id/ms 是原样进模型的，对 fil 是已展开的**，评测 case 要按这个分别设计（`eval_cases/omnivoice_prod.jsonl` 里 `tl_digit` 是压力 case、`tl_digit_prod` 才是线上形态）
- **线上已知 badcase**（`api.py:339-345` 的 severe 标签集）：`metallic_resonance`（金属音，有专门检测器 + 重试：cfg→`min(cfg−0.25,1.6)` 下限 1.2、steps→24-30）、`text_incomplete`、`source_script_residue`（跨语言配音漏源语言）、`gender_mismatch`、`duration_off_reference`。另有截断/跑飞（`_voxcpm_last_badcase`，`voxcpm2.py:738,1044` 注明「被标记的那条就是实际返回的音频」）与 AudioVAE V2 的 transient smearing / 「麦克风感」混响伪影——**steps 从 10 提到 20 就是为了压它**。⚠️ 这类音色伪影主要是 VAE/CFG/步数的产物，而 LoRA 只动 LM 与 DiT 的 q/k/v/o（`enable_proj: false`，`stop_proj`/`stop_head` 全冻结），**不要指望微调能修好金属音/塑料音，反而可能引入新的**；本项目侧目前也没有能自动测它们的指标（UTMOS 权重源已失效）
- **单一引擎**：Fish Speech 与 OmniVoice Cloud 合成链已移除（`README.md:3`），失败显式报错不切换（`docs/dubbing_vast_ai_api_zh.md:35`）。所以 VoxCPM2 既是默认也是唯一，微调质量直接等于线上质量
- **输出带宽是 16k 容器装 48k，模型侧定案（2026-09-15 实测结案）**：本项目自己的 `voxft.eval` **裸输出**（84 条 base、无 OmniVoice 后处理）`spectral_rolloff_99` 中位 **7262Hz**、16-24kHz 能量占比 max 0.0035%（砖墙），与 OmniVoice 后处理过的输出同一堵墙——**不是后处理链压的，是 VAE 16k 编码器决定的，cfg/steps 全网格无效、LoRA 结构上碰不到**。要 >8kHz 带宽只能靠推理后处理 BWE（独立立项，不占微调轮次）。微调能碰的只有「带内明亮度」`band_ratio_2_8k`：base 中位 0.162，r2 配对 Δ 均值 **-0.0143、71% 对子变暗**。后续实测（2026-09-15）：r1（纯 FLEURS）0.1642 ≈ base、r2 0.1524、r3 0.1550——**变暗幅度与 gigaspeech2 份额相关**；但训练源本身的 band_ratio 分布 gigaspeech2 与 FLEURS 几乎相同（各源 p50 都在 0.013-0.046，差异不在 2-8k 占比），⇒ **变暗机制不是源的可测谱倾斜，「按明亮度筛样本」没有可操作的靶子**，不要为此发明过滤规则。量级小（-8.6%）且盲听无人抱怨明亮度，真在意就调 gigaspeech2 份额（r3 证明这会伤 vi）或后处理 EQ，不占数据侧轮次
- **接入方式**：`api.py` 现在**不传任何 LoRA 参数**，所以两条路——① 零改动：把 merged 完整模型放在仓库根，`VOXCPM_MODEL_ID=FrankLiuDundun/voxcpm-finetune-lora` 即可；② LoRA：加一行 `lora_weights_path=<目录>`。`from_pretrained` 会**自动读同目录的 `lora_config.json`**（`core.py:48-58`）来对齐 r/alpha，所以 LoRA 目录必须**同时**含 `lora_weights.safetensors` 与 `lora_config.json`——只传 safetensors 会回落到 `LoRAConfig(enable_lm=True, enable_dit=True)` 的 **r=8 默认值**，形状不匹配、键被静默跳过。另：`core.py` 的 docstring 写「.pth 或 lora_weights.ckpt」是**过时的**，`voxcpm2.py:1310-1325` 实际优先读 safetensors

## 踩坑与约定（已修复问题的沉淀，勿回退）
- **Gradio 流式**：按钮必须直接绑定生成器函数；用 `lambda` 包一层会把生成器对象本身渲染进文本框
- **Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。任何运行后变化的列表（已加工数据集/配置/LoRA/上传目录）必须通过事件输出或 `Tab.select` 刷新
- **`.env` 加载顺序**：`paths.py` 必须**先** `load_dotenv()` **再**计算路径常量（`VOXFT_CKPT_ROOT`/`VOXFT_DATA_ROOT` 依赖此顺序）
- **大盘约定（远程）与实测磁盘图**：`VOXFT_DATA_ROOT`、`VOXFT_CKPT_ROOT`、`HF_HOME` 三个都指到 `/root/autodl-tmp/*`——**它已经是本实例唯一可写的大盘**，磁盘不够时不要再去找别的路径，要去找可回收的东西。实测挂载（autodl 4090D 实例）：`/root` 是 overlay **30G**（系统盘，下载/缓存勿落 `~`）；`/root/autodl-tmp` 是 `/dev/md127` **50G**；`/root/autodl-pub` → `AutoFS:fs1` **10T 但只读**（`touch` 直接 `Read-only file system`，只能读 autodl 预置的公共数据集，不能当产物盘）；`/root/autodl-fs`（网络文件存储）**本实例不存在**，`df` 里那个 877G 的 `ubuntu--vg` 是宿主机盘、只 bind 挂了 `nvidia-smi`，容器用不到。**回收手段按实测量级排序**：① `hf_home/hub/datasets--*` 的 parquet 缓存是最大头（FLEURS 五个 config 就 **13G**），加工完成后即可删，只有重新加工才需要重下；② `voxft_ckpt/*/merged`（单份 **4.6G**）验证/上传完立刻删；③ `voxft_ckpt/*/step_*`（单份 **415M**，官方脚本在最后一步必存，短 run 会多出冗余）只留 `latest`；④ `raw/<src>/audio`（每语种约 **0.9G**）在加工成功后可删——`processed/*/train.jsonl` 的 `audio` 指向 processed 自己的目录，只有 `origin_audio` 会失效，而它只被当字符串用于曝光计数与 ref 去重，不读文件；⑤ 许可不明/SA 排除源的 raw+processed。一轮五语种全流程实测峰值需求约 **20G**（parquet 13G + raw 7G + processed 7G + ckpt 2.5G + merged 4.6G，其中 raw 与 parquet 可在加工后回收），50G 盘够用但不宽裕，**开工前先 `df -h /root/autodl-tmp` 并按上面顺序清一遍**
- **HF 生态**：`datasets` 锁定 `<4`（5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突）；`hf`/`huggingface-cli` 不读项目 `.env`，命令行需手动 `export`（三个都要：`HF_ENDPOINT`/`HF_HUB_DISABLE_XET`/`HF_HOME`）——**优先用 `python -m voxft.data.prefetch`**，它导入 voxft 时就把这些处理好了；`snapshot_download` 的进度条不传给单文件，进度监控用缓存目录大小轮询；xet 下载分两阶段（downloading→reconstructing），进度"回退"属正常。**但走镜像必须关 xet**：hf-mirror 只代理 HF API，不代理 xet 的 CAS 服务器（cas-server.xethub.hf.co），reconstruction 阶段会直连并报 401。`paths._disable_xet_on_mirror()` 在 HF_ENDPOINT 非 huggingface.co 时自动 `setdefault("HF_HUB_DISABLE_XET","1")`；命令行用 `hf` 时要自己 export
- **下载通道实测矩阵（autodl + `/etc/network_turbo`）**：**hf-mirror 是国内镜像，机器也在国内，套上海外学术代理等于绕远路——下载不要 source 代理**。同一个 fleurs parquet 分片实测：`hf-mirror 不走代理 4.55 MB/s` ＞ `hf-mirror 走代理 2.43 MB/s`（代理劣化时掉到 **0.22 MB/s**，fleurs_ms 因此从 5 分钟变成 40 分钟下不完）＞ `huggingface.co 直连关 xet 1.0 MB/s`。`/etc/network_turbo` 自己的提示就写了"开启加速后对访问其他资源如 pip 源等会更慢"，hf-mirror 属于"其他资源"，只有 github / pypi 官方源 / huggingface.co 直连才需要它。**直连 + xet 会失败**：突发能到 2.94 MB/s，但 CAS 服务器 `Server disconnected without sending a response`，`_download_parquet` 3 次重试全挂。另：`.env` 默认（hf-mirror + 自动关 xet）就是最优组合，别去"优化"endpoint。**切换 endpoint 或 xet 模式会让已下载的分片变孤儿**——`blobs/<sha>.<后缀>.incomplete` 的后缀会变，新进程不续传而是从 0 开始，本轮为此白扔约 1.8GB，重跑前先 `ls blobs/*.incomplete` 手动清掉旧的
- **FLEURS 一个 config 就是一个约 1.9GB 的整片 parquet**：`--max-samples 60` 的试跑也要下完整片（`_download_parquet` 只在分片之间提前 break，片内是全量下载），不走代理约 5 分钟、走代理 17 分钟，四个语种就是 7.6GB 缓存。**别指望流式下载能省**：`load_dataset(streaming=True)` 对 fleurs 会在取首行时永久挂起（实测卡在 `h11/_connection.py`，进程零字节读入、无网络连接、无任何输出），`_download_stream` 已加 120s socket 超时把它变成可诊断报错，但它不是省流量的路子。磁盘紧张时按「每语种 2GB」预算
- **训练默认**：`batch_size=2 + 梯度累积=8`（等效 batch 还需乘 GPU 数）；页面按 1 epoch 自动算步数，换清单/卡数要重建配置，`.plan.json` 留审计；启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`save/valid_interval=250`；训练结束自动只保留最新 5 次 LoRA 运行。两点实测细节：① 官方脚本在 `step == num_iters-1` 和训练循环外各存一次，短 run 会多出一份冗余 checkpoint（61 步的 run 存了 step_0/60/61，`latest` 之外占 1.7GB），submodule 只读改不了，磁盘紧就手动删 `step_*`；② `save/valid_interval=250` 不影响短 run 落盘，因为上面那个"最后一步必存"兜住了。`python -m voxft.train.yaml_builder` 现在会像页面一样自动 `resolve_base_path`，不再写出 preflight 判死的 HF 仓库 ID
- **`max_grad_norm=1.0`、`num_workers=8`**：官方 v2 配置就是这两个值（早先本文写的"官方默认 0 = 不裁剪"是错的，已改）。情感语料动态大，不裁剪更容易出梯度尖峰
- **推理**：`load_denoiser=False`（去噪器依赖 modelscope，试听不需要）
- **Whisper 权重**：large-v3 约 3GB，国内直连 huggingface.co 常在 SSL 握手就超时。加载带 3 次重试并打印 endpoint；失败时报错里给了预下载命令。可用 `VOXFT_WHISPER_MODEL` / `VOXFT_WHISPER_MODEL_LARGE` 指向本地目录或镜像仓库。**项目用两个尺寸**：转写（结果会变成训练文本）用 `large-v3`，`ingest` 的整轨 VAD 与 `qc="whisper"` 质检用 `medium`（约 1.5GB）——`prefetch` 只预取 large-v3，**首次跑 ingest 或 whisper 质检会临时多下 medium**，磁盘和时长都要预留。另外权重下载期间**没有任何进度输出**，慢通道下十几分钟只显示一行「加载 Whisper medium（endpoint=…，第 1/3 次）...」，与卡死无法区分，别急着重启（要判断就 `du -sh $HF_HOME/hub/models--Systran--*` 看缓存在不在涨）
- **万级转写必须能断点续跑**：每 300 条及退出时原子保存完整原清单（包括坏例、尚未处理的行），重跑跳过已转写行。WhisperModel.transcribe 不接受 `batched`；ndarray 输入先转 16k。推理异常必须中止，不可当语料坏例吞掉。`--max-items` 试跑不回写原清单
- **数据源首选**：泰语 `thai_ser` 仅 impro / 审核后 `yodas_th`；Tagalog 自有真人 `drama_tl`；`filipino_emotion` 仅待审候选。`filswitch` 是新闻朗读，仅低比例补 Taglish 发音。越南语/印尼语表现力只有自建 `drama_vi` / `drama_id`，公开自然口语锚点首选 `gigaspeech2_vi/id`（Apache-2.0，**形态已核实可用，见下条**），`fleurs_*` 只当发音补充。马来语表现力同样只有自建 `drama_ms`，`fleurs_ms`（config `ms_my`）只当发音补充——**ms 的自然口语：yodas2_ms 已修复可下**（见下条②，YouTube 自发口语 + Sidon 降噪，CC-BY-3.0）——但口音必须抽听验证（可能是印尼内容互串）；自建 `drama_ms` 仍是表演档唯一解。中英文回放 `aishell3` / `fleurs_zh` / `fleurs_en` / 自备 `replay_en`。不能把朗读数据当去念稿感主力
- **自然口语源可用性实测（本轮逐个撞过，别重复调研）**：① **`gigaspeech2` 只有 th/vi/id**（cardData configs 就这三个，ms/tl/fil 全 400），gated:auto 需在页面同意条款；**它的 `refs/convert/parquet` 分支不存在**，但 parquet 索引 API 照样返回 200 和一串 URL，`_resolve_parquet_ref` 解析后下载必 404——真实布局是 `data/<lang>/<split>.tar.gz`（**每条一个 wav，不是长音频+时间戳**，早先的担心不成立）+ 同名 `.tsv`（`id\t全大写文本`），已加 `kind="hf_tar"` 走 `_download_hf_tar`。**dev 分片约 1GB/语种（8-9h）就够，train 单片 vi 3.4GB×240 / th 6.6GB×193 / id 1.7GB×592**；tsv 全大写要靠 `Source.sentence_case` 转句首大写（`.lower()` 对越南语变音符号安全，代价是句内英文专有名词被小写）。实测时长分布 **p50 只有 4.2-6.2s、vi 有 77% 落在 3-8s**，是五语种里最贴近线上 cue 长度的现货。session 数是 YouTube 视频 ID（dev 分片 vi 30 个 / th 21 个 / id 52 个），已按它做 train/val 隔离；② **`yodas2_ms` 已修复可下（2026-09-15）**：早先「`refs/convert/parquet` 404 = 不可用」是路径错了——该仓库根本没有这个分支，真正的自动转换 parquet 走 **`/api/datasets/sarulab-speech/yodas2_sidon/parquet/<config>/train/<n>.parquet`** 端点（ms000 6 片、vi000 4 片、id000 5 片，每片 ≤500MB，实测 Range 206 可下），`_download_parquet` 已加 404→API URL 直链回退。⚠️ 转换出的 parquet 列是 WebDataset 原样成员（`flac`/`metadata.json`/`__key__`/`__url__`），不是 audio/text 列；全量是 WebDataset tar（vi000/id000 各约 335GB），试跑取 1-2 片 parquet 就够。ms 自然口语因此从零变为有现货（YouTube + Sidon 降噪 + ASR 文本，CC-BY-3.0 需署名）；③ **`cv22_*` 已修复复活（2026-09-15，th 实测下载 100 条成功）**：早先「流式路径失效」的真相是加载脚本把数据 URL 硬编码到 huggingface.co（远端直连不通），仓库本身完好、镜像 `raw/main/n_shards.json` 200。已改为新 `kind="cv22"`：绕开脚本，按镜像 tree API 布局直拉 `transcript/<lang>/<split>.tsv` + `audio/<lang>/<split>/*.tar`（48kHz mp3，libsndfile 直接解码），**`client_id` 写进 speaker+session；r9 起重新评估为账号级持久身份（非聚类猜测），`has_speaker=True`、标 speaker_verified 并参与 ref 配对——只做匿名分组，不识别真人（CV 条款禁止 determine identity 与再分发数据本身）**。规模（release_stats 核实）：**th validated 173h / 7973 人、条均 4.19s（正对短 cue 分布）**，id validated 33.5h / 639 人，vi 6.3h / 354 人，全 CC0；tl/fil/ms 依旧没有。**`client_id` 配 ref 的路重新通了**（聚类仅供审计的纪律不变）；④ `filipino_speech`（MIT，有 `speaker_id`）能下但**行过滤后产出率约 1%**（首个分片扫 1813 行只写出 20 条），139 分片 5.8GB 换一两千条短切片，不值
- **自建短剧素材（已确认有：五语种、真人配音演员、自有版权、但是「成品混音」）的处理路线**：这是唯一能同时补上「表演语料 0 小时」「ref_audio 0%」「cue 长度错配」三个洞的来源，但**成品混音必须先分离人声**，否则 BGM 与音效会被当成语音学进去（模型会学会生成背景音乐）。
  ① **分离**：用 **Demucs `htdemucs`**（Meta，MIT，pip 可装，4090 上 GPU 加速）。这是从混音内容建 TTS 语料的业界标准做法——Emilia 那套 in-the-wild 管线就是「Demucs 分离 → VAD 切分 → ASR → DNSMOS 过滤」，与本项目 `ingest` 的流程一一对应。**分离不替代逐条试听**：分离会留伪影（频谱空洞、水声/相位感），仍要淘汰 BGM 泄漏严重的片段；分离也**解决不了混响**（配音通常干声录制、混音时按场景加混响，分离出的人声仍带场景混响，而 OmniVoice 对 ref 是做 WPE 去混响的）——混响重不重要先分离一集试听再决定。注意 OmniVoice 里那个 ModelScope ZipEnhancer 是**降噪不是分离**，压 BGM 不够用。
  ② **交付格式**：`ingest` 走 PyAV，mp4/mkv/mov/wav/mp3/m4a/flac 都行。⚠️ `decode_to_wav` **只取 `streams.audio[0]`**——配音视频常同时带原声轨与配音轨，给错整集白解，要么只给配音轨要么先说明音轨顺序。有台词本或带时间轴的字幕就一起给：演员实际念的台词本才是权威文本（不吃 ASR 错误），字幕时间轴还比 VAD 给出更准的 cue 边界。
  ③ **speaker 标注**：`pair_references` 只认 `speaker_verified=True`，这是把 ref 覆盖率从 0% 提到 30-50% 的唯一途径。流程是「自动 diarization 出候选簇（pyannote `speaker-diarization-3.1`，OmniVoice 自己在用，`api.py:238-240`，gated 需同意条款）→ 人工每簇听几条确认 → 6006「素材导入」Tab 勾选 `speaker_verified` + 标情绪」。AGENTS.md 的硬规矩照旧：**自动聚类仅供审计不能证明同人**，人工确认不能省；短剧里一个角色通常一个演员贯穿全剧，所以是"确认几个簇"而不是"标几千条"。跨集同一演员用同一个 ID。
  ④ ⚠️ **有了 verified speaker 之后的 split 陷阱**：`split_records` 的并查集已经把 `("speaker", ...)` 当分组键（`pipeline.py:640-641`），会自动做**说话人不相交**的 train/val 切分，`preflight` 也查泄漏——这部分不用改代码。但并查集会同时 union「同一演员」与「同一集」，而短剧主演通常只有 5-15 人：**某个演员出现在多集里，这几集会全部并成一个组**，整组要么进 train 要么进 val；极端情况 `len(groups) < 2` 会退化成「全部进训练、没有验证集」。对策是用 `--holdout <集ID>` 钉住固定验证集，加工后必须查 `stats.json` 的 val 条数与 `holdout_pinned_records`。
  ⑤ **先拿 1 集（挑 BGM 最轻的）走通全链路量产出率，别一上来全量**：要量的是「分离后有多少条能过试听」与「有几个可用说话人」。这个产出率完全未知，而 `tagalog_tts` 的先例是没先量分布就投产、150 条只留下 62 条
- **加工产出率实测（决定值不值得加工）**：FLEURS 四语种几乎全保留——`fleurs_vi` 60→60、`fleurs_ms` 60→60、`fleurs_th` 60→60、`fleurs_id` 60→59（只 1 条时长出界），`drop_*` 全 0；**`tagalog_tts` 只有 42%**：150 条转写后剩 146，其中 **84 条因不足 3s 被 `drop_duration` 砍掉**，与 registry 里「中位 1.6s」的警告一致。所以短切片源（`tagalog_tts` / `filipino_emotion` / `filipino_speech`）**先量时长分布再决定要不要加工**，别按原始条数估产能。另：加工进度文案已从「已产出 i 条样本」改成「已扫描 i 条，保留 N 条」——`i` 是 enumerate 的扫描序号，丢弃率高时会把产出说得严重虚高（tagalog_tts 末尾显示"已产出 100"，实际只留 62）
- **FLEURS 全量规模实测（整片 parquet 已在缓存时，扩全量是零下载成本）**：`th_th` 2602 条/8.49h、`fil_ph` 1884/7.71h、`vi_vn` 2994/9.08h、`id_id` 2579/9.09h、`ms_my` 2667/9.55h，五语种合计 **43.9h 全 CC-BY-4.0（可公开分发）**；加工产出率 **99.3%-99.8%**（drop 几乎全是 >30s），edge VAD 裁完各语种落到 6.3-7.8h。**时长 p50 是 10.4-14.1s、3-8s 只占 5.8%(fil)-24%(th)**——这是它与线上 cue 长度的根本错配，见下条。列取 `raw_transcription`（原正字法，带大小写与标点）而非 `transcription`（全小写去标点）；**两列的数字序列实测 100% 一致**，所以这个选择只影响标点与大小写，不影响数字
- **短 cue 出分布是微调最主要的失败模式（round 1 实测，最重要的一条）**：只用 FLEURS（p50 10-14s）训出来的联合 LoRA，在**线上那种 1-3s 的配音 cue 上会跑飞**——`vi_digit_3`（"Anh ấy sinh năm 1995."，输出约 1.2s）基座 4 个 seed 全对，LoRA 4 个 seed 全部崩成无关幻觉且疑似漏尾；vi 总体 CER 0.0686→0.5278、疑似漏尾 0→0.2778。但**同一份 LoRA 在 FLEURS 长度（5.7-21.7s）的 vi 探针上反而更好**：CER 0.0218→0.0175、疑似漏尾 0.0833→0.0。⇒ 退化不是"越南语变差"，是**长度出分布**；`step_500`（val loss 更优点）同样退化（vi 0.4299），所以**不是过训，退回去也没用**。对策是把训练数据的长度分布拉到 cue 量级：`gigaspeech2`（p50 4.2-6.2s）是现货，`drama_*` 成片导入是正解。注意 `process_dataset` 有 `3 <= min_dur <= max_dur <= 30` 硬守卫（官方区间），**不要为了塞进 1-3s 样本去拆它**——2-3s 段的收益远小于把 3-8s 自然口语加进来
- **离线质检指标口径（`voxft/qc/audio.py`，阈值一律移植 OmniVoice 生产口径，不自己发明）**：
  ① **数字类 case 必须单列**（`mean_cer_non_numeric`）。Whisper 自己会把口播数字词归一成阿拉伯数字或货币符号：实测输入 `Bayad ko ay isang libo't limang daan pesos.`，base 与 checkpoint 的输出**都**被转写成 `Bayad ko ay 1,500 pesos.`，两边 CER 同为 **0.588**——这不是 TTS 差异，是 ASR 假象。`id_digit_2`（`250 ribu rupiah` → `Rp250.000`）同理。判定用 `_is_numeric`：文本含阿拉伯数字自动命中，**已 verbalize 的要在 case 里显式写 `"numeric": true`**（自动检测抓不到）。剔掉数字类后实测 base 0.0022 / r2 0.0030，**两边都已饱和**——所以「非数字台词有提升」这个说法不成立，准确说法是「持平且无可测空间」；
  ② **多读与少读分开**：`suspected_truncation` 保持原语义 = 少读/漏尾（不要改定义，否则与前三轮已记录的数字不可比），新增 `over_read` = 归一化文本 >1.4× 参考（门限取 OmniVoice `api.py:6529-6536` 的长文本档）与 `len_ratio`；
  ③ **`metallic`**（金属音/玻璃音）：3-10kHz 内单点吃掉该频段 ≥20% 能量、且该频段占 100Hz-10kHz 总能量 ≥1%，在有声帧上连续 ≥5 帧且占比均值 ≥0.28。**不是谱质心也不是 HNR**；下限取 3kHz 而非 1.8kHz，因为 1.8kHz 会撞上普通 F2/F3 共振峰把干净人声误判。纯 numpy，40ms 帧 / 20ms hop / Hann，sr<12000 或 <0.45s 时返回 `None` 而**不是 False**（False 会被当成"检测通过"）。⚠️ **但这套阈值在本项目里没有区分力，只作参考值、不作门禁**：拿 84 条 48kHz 裸输出对照人工盲听，自动检出 4 条（base 1 / ckpt 3），**人工对这 4 条全判 `noise=False`、自然度 5/5**；反向人工唯一标 `noise=True` 的那条 score 只有 **0.0694**，远低于门限——**4 误报 / 1 漏报 / 0 命中**。原因是那套阈值是在 OmniVoice **后处理过**的音频上标定的（上线前有 peak ceiling 0.94、level match、可选 noise gate），频谱形态与裸输出不同；4 条误报的 score 全挤在 0.29-0.33 刚好压线也印证门限对这个分布太松。**2026-09-15 标定定案**：拿 168 条带盲听标注的样本（base_87eed5ce + lora_omni5_r2_latest_1e6447a9）做 ROC，`metallic_score` 对人工 noise 标注 **AUC=0.060（反相关）**、对自然度 ≤3/≤4 也只有 0.46/0.33——**永久降级为参考值，不是"暂无区分力"，不要再尝试重新标定，更不要拿它否决任何一轮微调**；
  ④ **`low_snr`**：有声 p85 与噪底 p15 的分隔 <18dB（50ms 帧 / 25ms hop）。是能量分位差不是真 SNR，取线上同一套帧长与分位点只为能和生产的 `quality_issues` 对照。**2026-09-15 标定定案**：同一批 168 条盲听样本 ROC **AUC=0.509（纯随机）**，且在 167 条人工判干净的样本上误报 48 条（**28.7% 误报率**）；与 OmniVoice 一致性核对（48 条生产后处理输出）：OmniVoice 自己 48/48 全判 `low_snr` + `noisy_reference`（**参考音频本身噪**，每条都中），我们判 31/48 且全是它判过的子集——方向一致、灵敏度更低，频繁触发的主因是 ref 噪底不是生成问题。**同样永久降级为参考值，不作门禁**；
  ⑤ **`speaker_sim`**：说话人嵌入余弦，**已换成 WavLM X-vector**（详见下面「说话人相似度已换成 WavLM X-vector」那条，含为什么废弃 MFCC 与两个加载坑）。报告里同时给 `speaker_sim_backend` 与 `speaker_sim_error`，这样 `None` 能分清是"没算"还是"算出来低"；
  ⑥ **`chars_per_sec` 必须配 `speech_ratio` 一起看，单看会得出完全错误的结论**。实测 r2 的 chars_per_sec 比 base 低 **12.7%**，乍看是"模型说慢了"；但拆开算**有声段字/秒只低 1.7%**（11.52→11.32），总时长 +12.3%、有声占比 0.94→0.84，再把静音分首/尾/内部三段，多出的 0.318s 里 **0.208s 是尾部静音**（0.084s→0.292s，最长 0.46s、p90 0.42s，**0/84 越过 0.5s 上限**）。⇒ 不是语速问题而是垫静音，成因大概是 32% 的 YouTube 自发口语比朗读多犹豫停顿；线上 `trim_silence_vad=True` 会裁掉所以基本无感，但不裁切的下游会拿到长 12% 的音频。**刻意没移植的**：`duration_off_reference` 在 OmniVoice 里是死代码（5 个调用点全传 `ref_duration=None`，理由见 `api.py:9037-9040`：VoxCPM 的参考音频只是音色锚，其时长与期望输出长度无关）；`RuleDurationEstimator`（200 行 / 600 语种 unicode 权重表，Apache-2.0）也没移植——A/B 比的是相对值，不需要绝对期望时长
- **首次人工盲听结论（r2 vs base，84 对全评）**：总体自然度 4.81→4.87、**12 胜 / 64 平 / 8 负**，按最小差值门槛**无语种退化**。分语种：`tl` 自然度 4.61→5.00 且可懂度 4.83→5.00、5胜0负（最清楚的赢家）；`vi` 可懂度 4.78→4.94；`ms` 可懂度 4.80→5.00；**截断 2 例→0 例、噪音 1 例→0 例**。但 **64/84 是平局**，所以「整体更自然」这个结论**不成立**，能声称的只有上面那几项具体改善 + 「无实质退化」。⚠️ 盲听还**交叉验证**了一条自动指标：8 条「人工判 B 更差」里 6 条是 **B 的音频明显变长而 CER 完全没变**，最极端的 `ms_manglish`(seed43) 从 1.76s 变 3.52s、文本反而更准（CER 0.094→0.000）但自然度被从 5 打到 3 —— 与「尾部静音 0.084s→0.292s」的自动测量指向同一个缺陷。**⇒ 时长类指标（`audio_sec` / `speech_ratio` / 首尾内部静音分解）是可信的，且大部分不需要母语者就能判**；真正只能靠母语者的只有「语调是否地道」「情绪是否对」两类，而这两类在零表演语料的现状下本来就不可能改善
- **验收 case 的 ref 语言必须覆盖线上真实分布，单一 ref 的结论不可外推（本轮实测推翻过一次验收）**：线上真实用法是**中文或英文 ref → th/tl/vi/id/ms 目标语种**，但我前三轮验收全部用 OmniVoice 归档 payload 里那条**菲律宾语** ref。补测 zh/en ref 后结论变了——同一份 r2 checkpoint：
  | ref | 总体 CER base→r2 | **非数字 CER** base→r2 | 退化语种 |
  |---|---|---|---|
  | tl（菲律宾语，前三轮用的） | 0.0774 → 0.0827 | 0.0022 → 0.0030 | id、vi |
  | zh | 0.1260 → **0.1128** | 0.0277 → 0.0268 | 仅 tl |
  | en | 0.1537 → **0.1179** | **0.0614 → 0.0045（−93%）** | 仅 id |

  ⇒ **同一份权重在不同 ref 语言下胜负语种完全不同**，用一条 ref 得出的"某语种退化"不能当定论。另外 base 自己在 zh/en ref 下就差得多（总体 0.0774 → 0.126 / 0.1537），说明**跨语言 ref 距离越远越难**，这是基座特性不是微调引入的。⚠️ 目前每种 ref 语言只有 **1 条**，ref 语言与 ref 说话人/录音质量是混淆的，要分离结论需每种语言 3-5 条 ref 重测
- **跨语言同人 ref 数据不存在，别再找（已确认死路）**：训练「zh/en ref → 目标语种」需要同一个说话人既有中/英录音又有目标语种录音。① 主调方确认**自有配音演员没有多语言版本**；② 公开语料也没有——FLEURS 是平行语料但各语种由不同众包说话人录制，Common Voice 的 `client_id` 理论上跨 locale 一致但 tl 的 `recordedHours=0`、ms 在 CV22 没有、CV23 起才有（2026-09-16 调研：CV26 scripted 29 人 3.65h + Spontaneous Malay 24 人 6.19h，CC0，需注册 MDC 下载，量小且几乎无 validated；cv22 下载器只能给同语种 ref 提供候选身份，跨语言同人依旧没有）。⇒ **`pair_references` 三级优先的第一级（中/英回放）永远是空的**，只能落到同语种 ref。同语种 ref 仍有价值（它至少让 `[103 ref 104][text][101 target 102]` 这条线上唯一在用的打包路径进入训练），但**对跨语言场景是否有正迁移是经验问题，必须实测不能预判**
- **音色一致性：实测有明确空间，但当前 LoRA 结构上改不动**：换成 WavLM X-vector 后实测「输出 vs 参考音频」的贴合度是 **0.86-0.94**（不是 MFCC 显示的 0.99），且**随 ref 语言显著变化**：tl ref 0.9421 / zh ref 0.9250 / **en ref 只有 0.8587**。同人上限约 0.997、跨说话人约 0.514，所以 0.86 意味着**离天花板还很远**。三轮微调几乎没动它（tl −0.0010 / zh +0.0013 / en +0.0086），原因是**架构性的**：`enc_to_lm_proj`（`voxcpm2.py:326`/`:1036`）与 `fusion_concat_proj`（`:339`/`:1073`）正是参考音频条件进入 LM 与 DiT 的通路，而 `enable_proj: false` 把它们全冻结在基座权重上（训练日志里 `enc_to_lm_proj.weight False` 就是）。**要提升音色必须 `enable_proj: true`，但那要求先有 ref_audio 数据**——在 `ref_audio` 覆盖率 0% 的数据上开 `enable_proj`，等于拿"根本没有 ref 出现"的样本去调整 ref 融合层，会把基座的克隆能力往坏里带。依赖链：`有 speaker 身份的语料 → ref_audio 30-50% → enable_proj: true`，顺序不能颠倒。附带一个跨 seed 一致性的小改善：r2 在三种 ref 下都比 base 略稳（+0.0018 / +0.0058 / +0.0152）
- **说话人相似度已换成 WavLM X-vector，MFCC 版废弃**：`qc/audio.speaker_sim` 现在用 `microsoft/wavlm-base-plus-sv`（VoxCeleb 上训的说话人验证模型，512 维），经 **transformers** 加载——它已是本项目依赖（官方训练脚本要用），**零新增依赖**，也不引 modelscope。可用 `VOXFT_SPK_EMB_MODEL` 指向本地目录或镜像仓库（与 `VOXFT_WHISPER_MODEL` 同一套约定）。换的理由：MFCC 余弦实测 84 条全挤在 0.985-0.996、跨 seed 一致性也 0.99+，动态范围不足；WavLM 的**尺子自检**是跨说话人 0.32-0.62（均值 0.514）vs 同人 0.995-0.998，**间隔 0.48**。⚠️ **两个坑**：① 必须用 `WavLMForXVector`，用 `Wav2Vec2ForXVector` 加载这个仓库会打印一大片 `MISSING` 并把 encoder **随机初始化**——不报错，只是嵌入全是垃圾（本轮踩过）；② 拿**随机噪声**验证动态范围是无意义的（噪声没有说话人身份，嵌入会塌到同一方向，实测两段不同噪声 cos=0.985），必须用真实语音的不同说话人做自检。与 OmniVoice 生产用的 modelscope ERes2NetV2（门限 `VOXCPM_SPEAKER_MISMATCH_MIN_SIMILARITY=0.45`）**刻度不可互换**，要对齐生产门限才需要换成它
- **离线指标的分辨率已经低于 run 间方差，别再对着它调配比**：`ms` 在 r2/r3 **配比完全相同**（都是 `fleurs_ms=17`）的情况下 CER 差出 **0.042**（0.0630 vs 0.1053）；`vi_digit_3` 同一条 case 在 base/r1/r2/r3 上是 `0.000 / 1.4375 / 0.000 / 0.958`，整条翻转且非单调。原因是混合用共享 RNG，改任何一部分的权重都会挪动后面所有部分的抽样，再叠加训练本身的非确定性 ⇒ **28 case × 3 seed 的差值在 ±0.04 以内一律当噪声**。要继续调参，先把统计功效提上来（每语种 30+ 条 case、5 个 seed），否则就是在噪声里挑好看的数字。本轮据此**停止了第 4 轮调参**
- **被实测否掉的假设要老实改掉（本轮第二条）**：r2 里 vi/id 数字 case 退化，我归因为「gigaspeech2/yodas 带数字样本 0.0%、FLEURS 20.6-24.1%，压缩 FLEURS 份额使带数字训练时长各降约 40%」。r3 据此把 vi/id 的 FLEURS 份额从 7/17 提到 11/17，带数字时长确实恢复了（vi 1.44h→2.30h，接近 r1 的 2.41h）——**但 vi CER 反而从 0.1176 恶化到 0.2839**。所以 r1→r2 的改善根本不是数字覆盖带来的，**把 vi 的 FLEURS（长朗读）份额从 100% 降到 41% 才是关键**，即上一条的长度分布。带数字时长这个变量与结果无因果关系
- **val loss 不能用来选 checkpoint**：r3 的 val loss 全程最优（末值 0.8823，r2 最优 0.9176），但 r3 的验收指标是三轮里最差的；r2 内部 `step_750`（val 最优 0.9176）在**所有语种**上都不如 `latest`（val 0.9767）。三轮一致 ⇒ 一律交付 `latest`，val loss 只用来看有没有发散
- **上传 HF 的通道与两个坑**：`hf-mirror` **只读不写**，上传必须 `source /etc/network_turbo` + `HF_ENDPOINT=https://huggingface.co` + **显式 `HF_HUB_DISABLE_XET=1`**（`paths._disable_xet_on_mirror` 只在 endpoint 非 huggingface.co 时才自动关，这里正好不触发）。实测远端直连 HF 下行 1.75 MB/s、**上行约 5 MB/s**，4.8GB 约 15 分钟。`load_dotenv` 用 `setdefault`，所以命令行 export 的 endpoint 不会被 `.env` 覆盖。⚠️ **`merge_lora` 会把基座目录里的非 safetensors 文件全拷过来，包括 VoxCPM2 自己的 `README.md`**——`upload_folder` 又只在 README 不存在时才生成卡片，两者叠加会把基座的模型卡当成我们的发布。必须在 merge 之后**显式覆盖** `README.md`（本项目用 `docs/model_card.md`）
- **远程脚本里 `cmd | tail` 会吞掉失败，本轮因此真丢了数据**：`uv run ... | tail -20` 的退出码是 `tail` 的 0，`set -e` 拦不住；`echo "exit=$?"` 拿到的也是 `tail` 的状态。结果加工明明抛了 `ValueError`，脚本却按"成功"分支把 `raw/<src>/audio` 和 tar 缓存删了，只能重下 3GB。规矩：**要么 `set -euo pipefail`，要么把命令输出重定向到文件再单独 `tail`，并且只有确认成功才删原始数据**（`if uv run ... > /tmp/x.log 2>&1; then ... rm -rf raw/...; else exit 1; fi`）
- **临时脚本里 `import huggingface_hub` 必须在 `import voxft.paths` 之后**：`HF_ENDPOINT` 是 `paths.load_dotenv()` 才写进 `os.environ` 的，而 `huggingface_hub.constants` 在自己被 import 时就读死了它。顺序写反 → 直连 `huggingface.co` → 远程报 `[Errno 101] Network is unreachable`（重试 5 次全是这个）。项目内模块靠"函数体内 import hf_hub"规避了，**手写的探针脚本没有这层保护**；纯 `requests` 打 `env("HF_ENDPOINT")` 最省事
- **`rate` 的单位按文字系统判，不按有没有空格判**：`audio_metrics` 原先 `" " in text` 就按词算，而泰文正字法本来没有词间空格，一个偶发空格能把整句切成 2"词"——实测泰语 `rate` 只有 **0.32/秒**（真实约 6 字/秒），差近 20 倍。已改为命中泰文/缅文区块（`_NO_WORD_SPACE`）一律按字符计。`rate` 只在 `log.py` 展示、不参与任何判定，所以**既有加工产物不需要重跑**，但看旧 stats 时要知道 th 的 rate 是失真的
- **运行记录用 `python -m voxft.train.runlog` 生成，不要手抄**：它从 `configs/<run>.yaml` + `.plan.json`、混合数据集的 `mix.json`、checkpoint 的 `train.log`、eval 报告 JSON 里抽字段，追加到 **`docs/runs.md`**（新记录在最上面）。`--eval` 传多份报告时**第一份当基线**，其余自动比对并按 ΔCER 分「红线」与「噪声级」两档（阈值 `--noise`，默认 0.005——每语种只有十几条样本，一个字符就能让均值动 0.002-0.003，阈值太紧红线就天天喊狼来了）。`--verdict/--next/--notes` 三个字段是给周报和下一轮看的，别留空
- **验收 case 集在 `eval_cases/omnivoice_prod.jsonl`（28 条，五语种 + zh/en）**：按线上形态设计——单句 cue、保留大小写与句末标点、统一用线上那条 **4.47s 菲律宾语参考音频做跨语言 reference-only**（`ref_audio: "prod_ref_fil.wav"`，相对 case 文件目录解析），并专门覆盖数字/货币（`vi_digit*`、`th_digit*`、`ms_digit_2` 的 RM、`id_digit_3` 的航班号）、Taglish/Manglish 英文借词、vi 句尾 nặng 调嘎裂声。**`prod_ref_fil.wav` 故意不入库**（`.gitignore` 的 `*.wav` 也挡着）：它是从 OmniVoice 归档生产 payload `results/voxcpm_quality_cases.json` 的 `reference_audio_base64` 解出来的主调方音色库素材，不能公开分发。新环境要跑这份验收，先从那个 payload 里解 base64 存成 `eval_cases/prod_ref_fil.wav`（24kHz 单声道 16bit，214638 字节）。`tl_digit` 是压力 case（裸数字），`tl_digit_prod` 才是线上 fil 的真实形态（TN 已 verbalize）
- **FilSwitch 下载**：转换 parquet 可以只有元数据，音频在原仓库的独立 FLAC 文件。`bytes=None` 不等于无音频；共享 `_load_audio` 支持内嵌 bytes、HF URL/路径和已解码数组，外链通过 hf_hub_download 保留原 revision、镜像、认证和缓存。不要把音频地址套到 `refs/convert/parquet` 分支；读取失败必须记录原因，不能静默跳过整包
- **code-switch 语种不能只认目标语种**：Tagalog 短剧台词是 Taglish、印尼语日常口语混英文、马来语是 Manglish，句内英文词多的样本 Whisper 会判成 en，只认目标语种会把最该保留的 code-switch 样本全部误杀。`Source.languages()` 查 `CODE_SWITCH_ACCEPT`，对 tl / id / **ms** 默认放行 `en`；**vi 默认从严**（混英以词内借词为主），实测 `drop_lang` 误杀再给该源加 `accept_langs=("vi","en")`，别提前放开。有权威文本的朗读源（`fleurs_id` / `cv22_id` / `fleurs_ms`）用 `accept_langs` **覆盖掉**默认放行——语种不符意味着错行，不是 code-switch
- **ms 与 id 高度互通，语种过滤挡不住两者互串**：马来语与印尼语是同一语言的两种标准化变体，Whisper 的语种检测和 YODAS 上游标签都会把印尼语内容判成 ms（反之亦然），所以 `drop_lang` 对 id/ms 互串**完全无效**。`yodas2_ms` 试跑时必须抽样听，确认拿到的是马来西亚口音；**ms 的盲听必须由马来西亚母语者做，不能拿印尼语听感代替**。收益是 id 语料对 ms 有正迁移，这是把两者放进同一个联合 LoRA 的主要理由；风险是口音与词汇差异被抹平。**离线先过一道词汇判据**：`fleurs_ms`（config `ms_my`）实测文本含 `ialah` / `bermaksud` / `amalan` / `dirujuk` 等马来语特征词（印尼语对应 `adalah` / `berarti` / `praktik`），不听音频也能先确认拿到的不是印尼语；**但这只验正字法与词汇，口音仍必须由马来西亚母语者盲听**
- **`yodas_th` 会话**：`utt_id.rsplit("-", 3)[0]` 保留可能含 `-` 的完整视频 ID。speaker_id 为视频级近似身份，不作 ref 依据；上游逐条峰值归一，不据此标音量。无原始连续时间关系就不拼接
- **数据身份**：MFCC 聚类仅供审计，不能证明同人，更不能调低阈值强凑 ref。身份未知默认不配 ref。同一演员跨源使用统一 speaker_namespace/ID；先隔离 train/val，再在集合内配 ref，混合与训练前再次检查泄漏
- **`filipino_speech`**：过滤 `machine` 与 `num_words<4`，只保留完整句；不再拼接孤立词或随机抖动停顿来伪造对白。行过滤缺列或无效数值时不放行
- **`thai_ser` 没有名为 `audio` 的列**（四路麦 `mic_clip/mic_con/mic_middle/mic_zoom`），必须靠 registry 的 `audio_cols` 映射，否则整个源在下载阶段被静默跳过；`mic_zoom` 是网络录音，不用
- **响度**：仅已验证说话人统一增益到 −24 dBFS；防削波时整个说话人共同回退，禁止单条峰值归一。未知身份不统一调响度；上游已抹掉的动态不能恢复
- **声学描述**：`f0_std_st` 含泰语声调和清浊音误差，不能代表自然度；`energy_range_db` 为能量分位差，不是 SNR。拒绝 min_snr_db/min_f0_std 硬筛，不把这些指标自动变成情绪指令（`--min-snr-db` 这个 CLI 参数已删：它唯一可能的结果就是触发 `process_dataset` 的抛错，留着只会误导；`Options.min_snr_db` 字段与守卫保留）。**`librosa.yin` 的 `frame_length` 必须按 sr 推导**：之前硬编码 1024 只在 16k 成立，而 `eval._prosody` 拿到的是 48k 合成音频，一帧装不下两个 60Hz 周期 → librosa 告警且 `f0_std_st` 失真（实测同一份 base 的 `mean_f0_std` 从 6.81 变成 6.21，差约 9%）。现在按 `2*sr/60` 向上取 2 的幂，16k 下算出来仍是 1024/256，**既有加工产物与 stats.json 不受影响**，只有 48k 的 eval 指标被修正
- **AISHELL-3**：约 85h；content.txt 的同一正文列交错汉字与拼音，必须剔除拼音。旧 processed 清单重新加工，不能直接混入
- **离线验收**：逐 case 固定 text/lang/ref_audio/ref_lang/control/seed；A/B 禁用自动坏例重试，普通试听保持原设置。CER/WER/疑似漏尾仅诊断，自然度/情绪/音色/真实截断需母语盲听，F0 不作通过门限。**实测参照锚点**（RTX 4090D，`eval base`，每语种 2 句、单 seed 42、无 ref、cfg 2.0 / 20 步）：CER `th 0.0`、`vi 0.0`、`tl 0.0119`、`id 0.0238`、**`ms 0.0814`**，疑似漏尾全 0——ms 基座最差，与「ms 是五语种里唯一输给竞品」的判断一致，也确实是提升空间最大的。同口径下一个五语种联合 LoRA（FLEURS 各 60 条、3.03h、1 epoch / 61 步）把总体 CER 从 0.0234 降到 0.0094，`ms 0.0814→0.0232`、`id 0.0238→0.0119`，th/vi/tl 持平、**无一语种退化**。⚠️ 这批数字只覆盖朗读语料、样本量极小、没做母语盲听，**只能当量级参照，不能当验收结论**；换 case 集或 seed 就会变
  - **单 seed 的 CER 会给出完全相反的结论，这是本轮最贵的一课**：vi 数字 case 在每 case 各自 seed（301-306）下 base CER **1.0686**、ckpt **0.2592**，看着像修好了一个 −76% 的大 bug；换成 `--seeds 42 43 44` 后 base 是 **0.0686**、ckpt **0.5278**——真相是 ckpt 严重退化。同一条 `vi_digit_3`（"Anh ấy sinh năm 1995."）在 4 个 seed 下 base 全对、ckpt 全崩成跑飞幻觉，**这种跨 seed 一致的才是真信号**。所以：**验收一律 ≥3 seed**（`--seeds 42 43 44`），base 与 ckpt 必须用同一组 seed 才叫配对；数字/货币类 case 方差极大，base 自己也会在某些 seed 上跑飞，**别把 base 的偶发崩溃当成微调的功劳**
  - **生产口径基线**（`eval_cases/omnivoice_prod.jsonl`，28 case × 3 seed = 84 条，cfg 1.8 / 20 步 / 线上那条 4.47s 菲律宾语 ref 跨语言 reference-only，RTX 4090D）：`base` 总体 CER **0.0774**、疑似漏尾 0.0952，分语种 `en 0.0 / zh 0.0 / th 0.0282 / ms 0.0852 / id 0.1036 / tl 0.1166 / vi 0.0686`。**基座的弱点全在数字与货币**：`vi_digit` 会跑飞成无关幻觉（某些 seed 下 CER 5.26）、`Giá vé là 250.000 đồng` 连声调符号都丢、`RM1,200 ringgit` 被念成印尼盾 `Rp`（ms/id 互串的实证）、`เที่ยวบิน 926` 念成 `916`。非数字的普通台词 base 基本全对 ⇒ **想靠微调把总体 CER 压下去，空间几乎只在数字类；普通台词只能验「不退化」**
- **重加工**：每次写新音频子目录，不覆盖旧清单引用的音频；旧产物不自动清理。原始 reference-only 与人工审核标记的 JSONL 格式、远程执行命令见 README
- **追加素材会让旧验证集泄漏**：`split_records` 的随机分组结果依赖清单长度，追加新素材后重新加工，上一轮的验证组会被整体重排进训练集，已训 run 的评测结论随之作废。挑一集写进 `data/raw/<source>/holdout.json`（`{"sessions": ["素材ID"]}`）钉住；钉住的分组不参与 shuffle，永远只进验证集，`stats.json` 的 `holdout_pinned_records` 可核对。矛盾组合（钉住了却 `val_ratio=0`、或全部素材都被钉住）直接报错，不静默把钉住的数据喂进训练。`ingest` 的 `session` 自动设为素材 ID，否则同一集的切片会各自成组跨 train/val
- **库函数不许 print**：UI 进程的 stdout 可能是已断开的 pty（启动 voxft-ui 的 SSH/tmux/JupyterLab 终端关掉后进程还在跑），`print` 抛 `[Errno 5] Input/output error`，会把一次**已经成功**的操作报成失败——filswitch 写完 2709 条清单后显示"失败"就是 `download_source` 结尾那句与 progress 重复的 print。一律走 `progress` 回调，CLI 侧传 `progress=print`（download/merge 已改，utmos 的 print 已删；`tb_wandb_bridge.start_bridge` 是最后一个残留——它由 UI 经 `launcher.start_local` 调用，未配 `WANDB_API_KEY` 时那句 print 就能把**已经启动成功**的训练报成失败，现已改为 `progress` 透传，`start_local(config, gpus, progress=)` 一路带到页面训练日志）。启动 UI 用 `nohup ... > ui.out 2>&1 &` 或 tmux，别把 stdout 挂在会断的终端上
- **视频容器解码走 PyAV**：soundfile 读不了 mp4/mkv，qc 组已显式声明 `av>=12`（本来就是 faster-whisper 的传递依赖）。不引入系统 ffmpeg 依赖，本地 macOS 与远程行为一致
- **首尾裁切按源分流**：`Options.edge_trim_ratio` 朗读 0.06（约 −24dB 相对有声电平）/ 表演 0.02。**RMS 门限单独用有个悬崖**：留白电平只要在有声电平 −24dB 以内，`rms > thr` 覆盖整条、一帧都裁不动，输出长度与输入**完全一致**——FLEURS「裁静音没生效」就是这个，不是没跑。所以朗读源改用 `Options.edge_vad`（Silero VAD，faster-whisper 自带、无需额外权重）定边界，不受电平影响，实测首尾误差 ±0.04s；表演源仍走 RMS 0.02（回落到 `peak×0.01`），抽气声是表演的一部分，裁掉模型就学不会换气。首尾低电平段还必须连续 ≥0.25s 才裁（`trim_silence(min_run=)`），否则提高门限会啃掉词首清辅音（/s/ 80–120ms）。两个必记常数：`min_silence_duration_ms` 用 **500** 而非 faster-whisper 默认 2000（默认给长音频分段用，会把不足 2s 的尾部底噪并进语音块，实测多留 1.0s）；喂 VAD 的数组必须 **float32**，float64 抛 ONNX `Unexpected input data type` 中断整轮加工。`stats.json` 记录 `edge_vad`/`edge_trim_ratio`/`max_tail_silence_sec`，旧产物能反推当时门限。**VAD/RMS 边界内的尾部静音再压一道硬上限**（`Options.max_tail_silence_sec=0.15`，`_cap_tail_silence`，帧口径与 `qc/audio.py` 的 `speech_ratio` 同一套：40ms 帧/20ms hop/gate=max(-52,p90-32)dB）：训练样本的尾静音会直接教会模型垫尾——FLEURS 加工样本尾静音 p50=0.30s，训出的 LoRA 输出尾静音 p50=0.32s（base 只有 0.084s），三轮一致复现，根因是 Silero VAD 的 `min_silence_duration_ms=500` 把更短的尾部停顿并进最后一个语音段、edge 裁切碰不到。只裁尾不裁头（头部没有缺陷证据），裁完跌破 `min_dur` 的按 `drop_duration` 丢弃，`stats.json` 另有 `tail_capped` 计数
- **听不清的按 ASR 置信度丢**：只对 `needs_transcribe` 的源启用（`filipino_emotion`/`tagalog_tts`/`thai_ser` impro）——转写结果就是训练文本，没有原文可比相似度。门限沿用 faster-whisper 解码器自己的默认（时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`），`--asr-min-logprob`/`--asr-max-no-speech` 可调，数量记在 `stats.json` 的 `drop_transcribe`，日志打印原因分类。这不违反"拒绝 min_snr_db/min_f0_std 硬筛"——被禁的是能量分位差与 F0 这类伪指标，不是 ASR 自己的置信度
- **Tagalog 无可商用的开源真人表演语料**（已核实，别重复调研）：Common Voice tl 官方 `recordedHours=0`（社区镜像也无 tl 音频）；YODAS/YODAS2 Sidon 的 224 个语种子集里没有 tl/fil；OpenSLR 无菲律宾语资源；HF 上 `modality:audio` 匹配 filipino/tagalog 的只有厂商 sample（`n<1K`，且多为 CC-BY-NC-ND 或 gated）；SEACrowd 的 23 个 th/tl/fil 数据集全是 text/图像，无音频；`liva-ai/yapdo-convo` 含 tl 但**完全没有许可声明**。`filipino_emotion` 连数据卡都没有。有规模的真人语料只能付费或用「素材导入」自建 `drama_tl`，逐项核实结论与询价模板见 `docs/corpus_sourcing.md`
- **MagicHub 别记错两个库**：`ASR-SFDuSC` 是 4.58h / 4073 条 / **10 人**的 **scripted monologue 朗读**，许可 **CC-BY-NC-ND 4.0**——NC 禁商用、ND 禁演绎（微调就是演绎），对本项目可用性为 0（早先本文写的"免费注册可用"是错的）。真正有价值的是 `ASR-BigFTagaCSC`（MDT-ASR-E076）：**1285h / 514 人自发对话**，16kHz WAV + TXT 转写，手机录、室内外，**专有授权需询价** `business@magicdatatech.com`。514 个真实说话人身份是 YouTube 抓取源给不了的，能同时补「自然口语锚点」和「ref 配对身份」，但**无情绪标签、不是表演**，只对应配比表的自然口语档
- **泰语源有 CC-BY-SA 红线**：`thai_ser` 与 `Porjai-central` 都是 **CC-BY-SA-4.0**（Porjai 的 `pattani`/`khummuang` 更是 **NC-SA**，直接排除）。SA 的触发条件是"向公众分享改编物"，模型权重算不算改编物**无判例**，CC 官方那句"应同许可发布"是保守建议不是法律要求。据此定的红线：**含 SA 数据训练的 LoRA 与 merge 后完整模型一律不对外分发**（不传 HF、不随客户交付、不开源），只通过 API 交付合成音频；要对外发布就向版权方谈商业授权（THAI-SER 出资方是 AIS + DEPA，有明确谈判主体）或只用 Apache-2.0/CC-BY 源。许可干净的泰语现货：Nexdata **1004h**（SKU 1687，商业买断，低噪，**带 speaker ID + gender**，WAR 98%）、`speechcolab/gigaspeech2` th（**Apache-2.0**，`gated:auto`，但短句为主无身份）
- **LAION DramaBox 那批"短剧配音数据"是 TTS 合成的，禁止用于补量**：`laion/dramabox-voice-acting-data-annotated` 看着完全对口（CC-BY-4.0、10万-100万条、标签带 `voice-acting`、数据卡还写了"同说话人跨情绪配对片段"），但数据卡 Models Used 明确源头是 `ResembleAI/Dramabox` 与 `gemini-2.5-pro-tts`，文件名 `{prompt_id}_seed{NN}_part1.mp3` 的 seed 就是生成采样。违反本项目"不用模型合成语音补量"的约定，且情绪标签是生成 prompt 不是真实表演标注。只有**标注 schema** 可参考。同理禁用 OpenSpeechHub 三个泰语集（无 license tag、无数据卡，同组织还挂动漫语音 rip）
- **低资源语种的调研结论必须核实到页面/API 原文**：本轮就出现过一篇编造的竞品论文（"JaiTTS arXiv 2604.27607，1万小时泰语，CER 1.94%"——arXiv API 查该 ID 与全文搜索均返回 0 条），以及把无许可的 `yapdo-convo` 说成 CC-BY-4.0。一条编造的"有现成大规模语料"足以让人跳过真正该做的自建工作
- **泰语转写可换 `typhoon-ai/typhoon-whisper-large-v3`**（SCB-10X，MIT，arXiv 2601.13044，约 11000h 泰语微调，自带泰语数字/重复标记归一化，Gigaspeech2/TVSpeech/FLEURS 泰语 SOTA）。**但不是即插即用**：`library_name: transformers`，不是 faster-whisper 的 CTranslate2 格式，要么 `ct2-transformers-converter` 转格式要么单开转写路径；模型卡在 MIT 之外另有一层 OpenTyphoon T&C 需商用前阅读；且**只有泰语**，对 TL 无帮助。优先级低于把数据搞到手
- **越南语/印尼语接入结论**（详见 `docs/vi_id_support.md`，每条都带 submodule 内的 `文件:行号`）：基座官方 30 语种**已含 vi/id** 且有实测分数（内部 30 语种基准 id WER 1.36% / vi 1.56%，**优于 tl 的 2.63%**），代码里无语言列表/language token/lang_id，tokenizer `byte_fallback=True` 对 vi/id 实测 **0 UNK** → **submodule 零改动**，别试图加语种信号，基座没有对应槽位。两者都**没有已核实的真人表演/情感语料**，处境同 Tagalog，首轮表演档为 **0%** 且验收只能声称发音/口语韵律/克隆不退化，**不能声称情绪表现力改善**。基座文本归一化是 zh/en 二分（`text_normalize.py:172`，非中文一律走英语规则），所以推理侧必须保持 `normalize=False`，**台词里的数字/货币念法是数据侧责任**，评测 case 保留阿拉伯数字时 CER/WER 会失真、只作盲听。vi 是 6 声调语言：`f0_std_st` 同样不代表自然度，**以 `ngã`/`nặng` 调收尾的句子句尾嘎裂声能量低、有被 RMS 裁掉的风险（未实测，验收重点听）**，`rate` 是音节/秒，**WER 是音节级**不与词级横向比。控制前缀守卫拦得住越南语专属字符，**拦不住印尼语**（纯 ASCII 与英文无法区分），只能靠标注纪律。许可上 vi/id 本轮全是 Apache-2.0 / CC-BY / CC0，**无 SA 红线，权重可对外分发**（后续核实进 SA/NC/ND 源则红线恢复）
- **马来语接入结论**（详见 `docs/ms_support.md`，每条都带 submodule 内的 `文件:行号`）：基座官方 30 语种**已含 ms**（`README.md:57`），内部基准 **ms WER 1.75%**（`README.md:546`）——**是五个目标语种里唯一输给竞品的**（同表 Fish S2-Pro 1.41%；id/vi/tl 都是基座赢），所以 ms 提升空间最大、最值得投入。但 **MLS 那两张 24 语种 WER/SIM 表都没有 ms**（`README.md:455-516`），克隆能力基线**没有官方数字可引用**，只能自己 A/B 实测。ms 是纯 ASCII 拉丁正字法，tokenizer 实测 **0.394 tok/char、0 UNK、0% byte-fallback**（id 0.373、vi 0.850、th 2.205），是五语种里文本侧最轻的 → **submodule 零改动**。ms 词间有空格，**`WER_LANGS` 含 ms**（与 id/tl/en 同量纲可横比），不像 th 只有 CER、vi 是音节级。**命名陷阱：FLEURS 的马来语 config 是 `ms_my`，结尾 `my` 是国家码马来西亚，不是缅甸语**（Burmese 的 ISO 码才是 `my`，基座也支持但我本轮一度认错）。已核实排除：`mesolitica/Malaysian-TTS` 是 **F5-TTS 合成**（数据卡原文「Malaysian Synthetic TTS dataset」，同 LAION Dramabox 的坑，禁用）；`espnet/floras` 是 **3TB 长音频基准**不可切分；`fsicoli/common_voice_22_0` 与 `speechcolab/gigaspeech2` 都**没有 ms**（别再照 th/vi/id 的模式去找 `cv22_ms`/`gigaspeech2_ms`）；`disco-eth/WorldSpeech` 与 `MERaLiON/sea_audiobench_*` 是 **NC/NC-ND**。`mesolitica`（Malaysia-AI）其余大批语料规模很大但**全线无许可声明**，拿到书面授权前一律不得使用（无声明 ≠ 开放，默认全权保留）。许可上本轮 ms 源是 CC-BY-3.0 + CC-BY-4.0 + 自有授权，**无 SA 红线**，但 CC-BY-3.0 要求署名
- **联合微调的四个可观测点**（缺一个就无法归因，别删）：① `dataset_summary` 的 `language_hours`——配比口径是**按有效音频时长**，按条数的 `languages` Counter 看不出占比；② `mix.json` 的 `language_shares`（各语种 `requested`/`actual`/`hours`），某语种实际低于请求值 90% 时经 `progress` 回调告警，**缺口不自动重分配**给其他语种（那正是大语种吃掉小语种的机制），也不靠 3× 重复强凑；③ `split_records` 的 **val 按语种配额**（`Options.val_min=16`，`n//5` 是 20% 硬上限），全局 `val_ratio` 会让小语种只摊到几条验证样本、评测无统计意义；④ eval 报告的 **`by_lang`** 分语种 CER/WER/漏尾，联合 run 必须逐语种与 `eval base` 的同一份 case 对比。另：`pair_references` 的跨语言 ref 池是**三级优先**（中/英回放 → 其他目标语种 → 同语种），联合数据才有的信号；`.plan.json` 记 `langs`，run 名默认带语种标记（≥3 语种用 `jointN`）；`_source_lang` 对未登记源返回 **`unknown` 而不是猜 `zh`**
- **网络受限时不得凭记忆写语料结论**：vi/id 接入那轮 HF API 全部被限流，处置是**只注册可由本仓库既有事实复现的源**（th/tl 已在用的仓库与 config 命名模式、`corpus_sourcing.md` 已核实过的 gigaspeech2 覆盖 th/id/vi），把 VIVOS / MagicHub / Nexdata / YODAS2-vi-id 等候选全部写进 `docs/vi_id_support.md §3` 的待核实清单并附远程核实命令，规模与许可一律标「未核实」不写数字。宁可留空也不要填一个看起来像结论的编造值

## Submodule 升级
```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```
