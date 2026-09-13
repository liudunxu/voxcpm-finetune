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
    "export PATH=/root/.local/bin:\$PATH; cd /root/voxcpm-finetune; $*"
  EOF
  chmod +x /tmp/rsh
  ```
  用 `$*` 直接拼命令时，别在本机用 `SSH="ssh ..."; $SSH '...'`——zsh 会把整串当成一个命令名，报 `command not found`。
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

## 踩坑与约定（已修复问题的沉淀，勿回退）
- **Gradio 流式**：按钮必须直接绑定生成器函数；用 `lambda` 包一层会把生成器对象本身渲染进文本框
- **Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。任何运行后变化的列表（已加工数据集/配置/LoRA/上传目录）必须通过事件输出或 `Tab.select` 刷新
- **`.env` 加载顺序**：`paths.py` 必须**先** `load_dotenv()` **再**计算路径常量（`VOXFT_CKPT_ROOT`/`VOXFT_DATA_ROOT` 依赖此顺序）
- **大盘约定（远程）**：`VOXFT_DATA_ROOT`、`VOXFT_CKPT_ROOT`、`HF_HOME` 都指到 `/root/autodl-tmp/*`；系统盘小，下载/缓存勿落 `~`
- **HF 生态**：`datasets` 锁定 `<4`（5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突）；`hf`/`huggingface-cli` 不读项目 `.env`，命令行需手动 `export`（三个都要：`HF_ENDPOINT`/`HF_HUB_DISABLE_XET`/`HF_HOME`）——**优先用 `python -m voxft.data.prefetch`**，它导入 voxft 时就把这些处理好了；`snapshot_download` 的进度条不传给单文件，进度监控用缓存目录大小轮询；xet 下载分两阶段（downloading→reconstructing），进度"回退"属正常。**但走镜像必须关 xet**：hf-mirror 只代理 HF API，不代理 xet 的 CAS 服务器（cas-server.xethub.hf.co），reconstruction 阶段会直连并报 401。`paths._disable_xet_on_mirror()` 在 HF_ENDPOINT 非 huggingface.co 时自动 `setdefault("HF_HUB_DISABLE_XET","1")`；命令行用 `hf` 时要自己 export
- **下载通道实测矩阵（autodl + `/etc/network_turbo`）**：**hf-mirror 是国内镜像，机器也在国内，套上海外学术代理等于绕远路——下载不要 source 代理**。同一个 fleurs parquet 分片实测：`hf-mirror 不走代理 4.55 MB/s` ＞ `hf-mirror 走代理 2.43 MB/s`（代理劣化时掉到 **0.22 MB/s**，fleurs_ms 因此从 5 分钟变成 40 分钟下不完）＞ `huggingface.co 直连关 xet 1.0 MB/s`。`/etc/network_turbo` 自己的提示就写了"开启加速后对访问其他资源如 pip 源等会更慢"，hf-mirror 属于"其他资源"，只有 github / pypi 官方源 / huggingface.co 直连才需要它。**直连 + xet 会失败**：突发能到 2.94 MB/s，但 CAS 服务器 `Server disconnected without sending a response`，`_download_parquet` 3 次重试全挂。另：`.env` 默认（hf-mirror + 自动关 xet）就是最优组合，别去"优化"endpoint。**切换 endpoint 或 xet 模式会让已下载的分片变孤儿**——`blobs/<sha>.<后缀>.incomplete` 的后缀会变，新进程不续传而是从 0 开始，本轮为此白扔约 1.8GB，重跑前先 `ls blobs/*.incomplete` 手动清掉旧的
- **FLEURS 一个 config 就是一个约 1.9GB 的整片 parquet**：`--max-samples 60` 的试跑也要下完整片（`_download_parquet` 只在分片之间提前 break，片内是全量下载），不走代理约 5 分钟、走代理 17 分钟，四个语种就是 7.6GB 缓存。**别指望流式下载能省**：`load_dataset(streaming=True)` 对 fleurs 会在取首行时永久挂起（实测卡在 `h11/_connection.py`，进程零字节读入、无网络连接、无任何输出），`_download_stream` 已加 120s socket 超时把它变成可诊断报错，但它不是省流量的路子。磁盘紧张时按「每语种 2GB」预算
- **训练默认**：`batch_size=2 + 梯度累积=8`（等效 batch 还需乘 GPU 数）；页面按 1 epoch 自动算步数，换清单/卡数要重建配置，`.plan.json` 留审计；启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`save/valid_interval=250`；训练结束自动只保留最新 5 次 LoRA 运行。两点实测细节：① 官方脚本在 `step == num_iters-1` 和训练循环外各存一次，短 run 会多出一份冗余 checkpoint（61 步的 run 存了 step_0/60/61，`latest` 之外占 1.7GB），submodule 只读改不了，磁盘紧就手动删 `step_*`；② `save/valid_interval=250` 不影响短 run 落盘，因为上面那个"最后一步必存"兜住了。`python -m voxft.train.yaml_builder` 现在会像页面一样自动 `resolve_base_path`，不再写出 preflight 判死的 HF 仓库 ID
- **`max_grad_norm=1.0`、`num_workers=8`**：官方 v2 配置就是这两个值（早先本文写的"官方默认 0 = 不裁剪"是错的，已改）。情感语料动态大，不裁剪更容易出梯度尖峰
- **推理**：`load_denoiser=False`（去噪器依赖 modelscope，试听不需要）
- **Whisper 权重**：large-v3 约 3GB，国内直连 huggingface.co 常在 SSL 握手就超时。加载带 3 次重试并打印 endpoint；失败时报错里给了预下载命令。可用 `VOXFT_WHISPER_MODEL` / `VOXFT_WHISPER_MODEL_LARGE` 指向本地目录或镜像仓库。**项目用两个尺寸**：转写（结果会变成训练文本）用 `large-v3`，`ingest` 的整轨 VAD 与 `qc="whisper"` 质检用 `medium`（约 1.5GB）——`prefetch` 只预取 large-v3，**首次跑 ingest 或 whisper 质检会临时多下 medium**，磁盘和时长都要预留。另外权重下载期间**没有任何进度输出**，慢通道下十几分钟只显示一行「加载 Whisper medium（endpoint=…，第 1/3 次）...」，与卡死无法区分，别急着重启（要判断就 `du -sh $HF_HOME/hub/models--Systran--*` 看缓存在不在涨）
- **万级转写必须能断点续跑**：每 300 条及退出时原子保存完整原清单（包括坏例、尚未处理的行），重跑跳过已转写行。WhisperModel.transcribe 不接受 `batched`；ndarray 输入先转 16k。推理异常必须中止，不可当语料坏例吞掉。`--max-items` 试跑不回写原清单
- **数据源首选**：泰语 `thai_ser` 仅 impro / 审核后 `yodas_th`；Tagalog 自有真人 `drama_tl`；`filipino_emotion` 仅待审候选。`filswitch` 是新闻朗读，仅低比例补 Taglish 发音。越南语/印尼语表现力只有自建 `drama_vi` / `drama_id`，公开锚点首选 `gigaspeech2_vi/id`（Apache-2.0，但**字段形态未核实，先 `--max-samples 20` 试跑**），`fleurs_*` / `cv22_*` 只当发音补充。马来语表现力同样只有自建 `drama_ms`，自然口语首选 `yodas2_ms`（`sarulab-speech/yodas2_sidon` config `ms000`，CC-BY-3.0，与 `yodas_th` 同上游家族；**gigaspeech2 与 Common Voice 22 都没有 ms**），`fleurs_ms`（config `ms_my`）只当发音补充。中英文回放 `aishell3` / 自备 `replay_en`。不能把朗读数据当去念稿感主力
- **加工产出率实测（决定值不值得加工）**：FLEURS 四语种几乎全保留——`fleurs_vi` 60→60、`fleurs_ms` 60→60、`fleurs_th` 60→60、`fleurs_id` 60→59（只 1 条时长出界），`drop_*` 全 0；**`tagalog_tts` 只有 42%**：150 条转写后剩 146，其中 **84 条因不足 3s 被 `drop_duration` 砍掉**，与 registry 里「中位 1.6s」的警告一致。所以短切片源（`tagalog_tts` / `filipino_emotion` / `filipino_speech`）**先量时长分布再决定要不要加工**，别按原始条数估产能。另：加工进度文案已从「已产出 i 条样本」改成「已扫描 i 条，保留 N 条」——`i` 是 enumerate 的扫描序号，丢弃率高时会把产出说得严重虚高（tagalog_tts 末尾显示"已产出 100"，实际只留 62）
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
- **重加工**：每次写新音频子目录，不覆盖旧清单引用的音频；旧产物不自动清理。原始 reference-only 与人工审核标记的 JSONL 格式、远程执行命令见 README
- **追加素材会让旧验证集泄漏**：`split_records` 的随机分组结果依赖清单长度，追加新素材后重新加工，上一轮的验证组会被整体重排进训练集，已训 run 的评测结论随之作废。挑一集写进 `data/raw/<source>/holdout.json`（`{"sessions": ["素材ID"]}`）钉住；钉住的分组不参与 shuffle，永远只进验证集，`stats.json` 的 `holdout_pinned_records` 可核对。矛盾组合（钉住了却 `val_ratio=0`、或全部素材都被钉住）直接报错，不静默把钉住的数据喂进训练。`ingest` 的 `session` 自动设为素材 ID，否则同一集的切片会各自成组跨 train/val
- **库函数不许 print**：UI 进程的 stdout 可能是已断开的 pty（启动 voxft-ui 的 SSH/tmux/JupyterLab 终端关掉后进程还在跑），`print` 抛 `[Errno 5] Input/output error`，会把一次**已经成功**的操作报成失败——filswitch 写完 2709 条清单后显示"失败"就是 `download_source` 结尾那句与 progress 重复的 print。一律走 `progress` 回调，CLI 侧传 `progress=print`（download/merge 已改，utmos 的 print 已删；`tb_wandb_bridge.start_bridge` 是最后一个残留——它由 UI 经 `launcher.start_local` 调用，未配 `WANDB_API_KEY` 时那句 print 就能把**已经启动成功**的训练报成失败，现已改为 `progress` 透传，`start_local(config, gpus, progress=)` 一路带到页面训练日志）。启动 UI 用 `nohup ... > ui.out 2>&1 &` 或 tmux，别把 stdout 挂在会断的终端上
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
- **马来语接入结论**（详见 `docs/ms_support.md`，每条都带 submodule 内的 `文件:行号`）：基座官方 30 语种**已含 ms**（`README.md:57`），内部基准 **ms WER 1.75%**（`README.md:546`）——**是五个目标语种里唯一输给竞品的**（同表 Fish S2-Pro 1.41%；id/vi/tl 都是基座赢），所以 ms 提升空间最大、最值得投入。但 **MLS 那两张 24 语种 WER/SIM 表都没有 ms**（`README.md:455-516`），克隆能力基线**没有官方数字可引用**，只能自己 A/B 实测。ms 是纯 ASCII 拉丁正字法，tokenizer 实测 **0.394 tok/char、0 UNK、0% byte-fallback**（id 0.373、vi 0.850、th 2.205），是五语种里文本侧最轻的 → **submodule 零改动**。ms 词间有空格，**`WER_LANGS` 含 ms**（与 id/tl/en 同量纲可横比），不像 th 只有 CER、vi 是音节级。**命名陷阱：FLEURS 的马来语 config 是 `ms_my`，结尾 `my` 是国家码马来西亚，不是缅甸语**（Burmese 的 ISO 码才是 `my`，基座也支持但我本轮一度认错）。已核实排除：`mesolitica/Malaysian-TTS` 是 **F5-TTS 合成**（数据卡原文「Malaysian Synthetic TTS dataset」，同 LAION Dramabox 的坑，禁用）；`espnet/floras` 是 **3TB 长音频基准**不可切分；`fsicoli/common_voice_22_0` 与 `speechcolab/gigaspeech2` 都**没有 ms**（别再照 th/vi/id 的模式去找 `cv22_ms`/`gigaspeech2_ms`）；`disco-eth/WorldSpeech` 与 `MERaLiON/sea_audiobench_*` 是 **NC/NC-ND**。`mesolitica`（Malaysia-AI）其余大批语料规模很大但**全线无许可声明**，拿到书面授权前一律不得使用（无声明 ≠ 开放，默认全权保留）。许可上本轮 ms 源是 CC-BY-3.0 + CC-BY-4.0 + 自有授权，**无 SA 红线**，但 CC-BY-3.0 要求署名
- **联合微调的四个可观测点**（缺一个就无法归因，别删）：① `dataset_summary` 的 `language_hours`——配比口径是**按有效音频时长**，按条数的 `languages` Counter 看不出占比；② `mix.json` 的 `language_shares`（各语种 `requested`/`actual`/`hours`），某语种实际低于请求值 90% 时经 `progress` 回调告警，**缺口不自动重分配**给其他语种（那正是大语种吃掉小语种的机制），也不靠 3× 重复强凑；③ `split_records` 的 **val 按语种配额**（`Options.val_min=16`，`n//5` 是 20% 硬上限），全局 `val_ratio` 会让小语种只摊到几条验证样本、评测无统计意义；④ eval 报告的 **`by_lang`** 分语种 CER/WER/漏尾，联合 run 必须逐语种与 `eval base` 的同一份 case 对比。另：`pair_references` 的跨语言 ref 池是**三级优先**（中/英回放 → 其他目标语种 → 同语种），联合数据才有的信号；`.plan.json` 记 `langs`，run 名默认带语种标记（≥3 语种用 `jointN`）；`_source_lang` 对未登记源返回 **`unknown` 而不是猜 `zh`**
- **网络受限时不得凭记忆写语料结论**：vi/id 接入那轮 HF API 全部被限流，处置是**只注册可由本仓库既有事实复现的源**（th/tl 已在用的仓库与 config 命名模式、`corpus_sourcing.md` 已核实过的 gigaspeech2 覆盖 th/id/vi），把 VIVOS / MagicHub / Nexdata / YODAS2-vi-id 等候选全部写进 `docs/vi_id_support.md §3` 的待核实清单并附远程核实命令，规模与许可一律标「未核实」不写数字。宁可留空也不要填一个看起来像结论的编造值

## Submodule 升级
```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```
