# AGENTS.md

## 项目简介
VoxCPM 2（OpenBMB TTS）微调工作台：Tagalog/泰语/越南语/印尼语/马来语高质量语料的下载与加工、跨语言（中文→目标语言）混合微调、LoRA/全量训练管理、wandb 监控、LoRA merge、HuggingFace 同步。**目标是五语种（th/tl/vi/id/ms）联合微调：一个 LoRA 同时提升这几种语言的配音质量，但验收必须分语种做**。Gradio 页面端口 **6006**。

## 当前生效边界（2026-09-22 核验）

> 本节只留**现在仍然生效**的决定，按主题归并，不再按日期追加。要去哪里找细节：
> **当前该做什么**看 `TODO.md`；**被本节收敛掉的逐轮流水原文**（commit SHA、回归条数、逐 cue 复盘、当时的判断依据）看 `docs/execution_log.md`；训练记录 `docs/runs.md`；成片复盘 `docs/video_review_*.md`；验收口径 `docs/qc_gates.md`；踩坑依据 `docs/pitfalls.md`；推理链 `docs/inference_chain_audit.md`；语料 `docs/corpus_sourcing.md`；各轮实验 `docs/*_20260917.md` / `docs/*_20260918.md`。
> 旧文档里过时的门槛与「下一步」不覆盖本节。

### 生产与训练状态
- **生产权重是 r8**，未变。r10/r11 都没拿到可晋级收益，**GPU 实验已暂停**：不为机器空闲继续微调。出现新的可复现坏例、或有明确增益假设的可追溯数据后，先冻结回归与单变量方案，再开 GPU。
- **连续严重坏例不得永久用“无需微调”结案。** 情绪/调用链修复不能解释或稳定消除时，冻结实际输入、核验基座与r8文件，先做≥3相邻seed的模型单变量诊断，再决定回退或定向微调；`model_id`请求或接口自报名称不等于换模型/权重验证。离线原始客户端ref对照不等于GPU预处理后生产重放，少量VI坏例不代表五语种验收。
- r8 与 r10/r11 共 5 份 LoRA + 配置 + 步数状态已备份本机 `checkpoints/gpu_pause_20260918/`（15 文件 SHA 全部通过，记录在同目录 `verification.json`）。**未备份**全量训练数据、历史评测音频与优化器状态 ⇒ 只能建议关机，不能据此释放实例或删数据盘。
- **源码 push 不等于部署。** 新成片已观察到 DIS 当前修复源码生效（核验范围与证据见 `TODO.md`），不能继续笼统说调用方全部未部署，也不能外推所有 worker/完整镜像已更新。OmniVoice 运行源码、渲染时权重 SHA 与部署 commit 仍未核验；元数据 `voxcpm.model` 是 DIS 客户端配置，**不是加载证明**。
- 旧实验数字一律保留：不重跑、不重评分、不挑低分覆盖报告，冻结的 WAV / 报告 / 评分器哈希不动。

### 成片反馈的处置顺序（反复验证过，别跳步）
1. **先排上下游，不自动重开微调。** 删尾/音色/口音类反馈，先比较同一 take 的「服务端输出 → 客户端裁边 → 变速 → 最终 segment」，再谈模型。已证实的删尾来自 DIS `clone_take_edge_hygiene` 与强制 containment（`content_fit.complete=false`），不是生成漏尾。FFmpeg退出码0也可能写空WAV；头裁切后预填静音再变速须校准采样时间戳，不能忽略产物时长探测失败。
2. **「同角色音色不稳」先验身份链是否真同一条。** 同名不等于同 speaker key / 同 character_id / 同 ref / 同模式；不同 ref、reference-only 与 combined 之间不是模型 A/B，也不能凭一组现象推定 LoRA 克隆能力不足。
3. **裁后必须重新做最终文本 QC。** 有能量 ≠ 有人声 ≠ 通过；硬裁/refit 会让裁前的 pass 失效，须查实际任务的执行与验收日志，不能声称「无 refit 消费者」。裁后漏尾检查不能要求前半句与目标逐token完全相等，ASR前部错字不应掩盖有序对齐末端的删除；此类弱证据只记uncertain，不当真人漏词结论、不触发提前裁词。
4. **系统顺序**：输入可用性与身份 → 完整候选风险 → 静音/额外发声/正文超窗分流 → 最终音频 QC。**不靠全局加步数、加重试或微调来兜。** 用户允许严重坏例多花少量预算：DIS本地策略为普通默认4次、触发现有强风险门后一次性共享增加2次模型生成，正常片段不扩；不是每层各加2次，也不是2次HTTP各自再多抽。弱情绪/低SNR提示不据此扩预算；部署状态见TODO。
5. **质检失败时选最佳可用 TTS，不自动回退原声**（此条覆盖旧「内容失败必须丢弃并保原声」的出片要求）。但 keep-best 带 severe 出片**不算质量通过**；无可用 TTS 就保持失败并抑制源人声，显式保留的笑声等不变。风险 take 不续写、不当干净缓存；重试必须改变实际生成条件：未换参考/控制/参数时要换seed，同seed且同有效参数才是缓存伪重试，不能把单变量控制A/B误当伪重试。
6. **弱证据记 uncertain**，不能认证 `false_alarm`/`aligned`，也不能用来提前裁词。生成音频的词边界走共用证据门 `timestamp_qc.trusted_word_bounds`；Whisper 词时间戳 warning 为空**不等于**可靠强制对齐。
7. **不凭同名跨身份借音**；参考全静音就是无候选，不能为此关掉保护。异常长音频不等于持续说话，不能凭短语音包络裁未知尾部。
8. **提前判退但可能出片的候选也要有完整观测**：DIS 长句 `output_vocalization` 会提前返回、跳过后面的 pitch 检查，观测缺失不是稳定通过；服务器其它 F0 估计不可直接混入本地口径。补测只用本地音区算法，不额外触发远端性别确认。
9. **锚点缓存不能早于元数据/分句/安全窗准备**（原预处理只读到目标 SRT 与临时 speaker，没有源文与风险标记）。参考窗扩长后padding必须重算且只收紧，VAD交集补边不越出输入窗；短参考无法做可靠声纹校验、或speaker_count=1，都不等于已验证单人。VN 音频源语配置 `en` 与中文 `source_text` 的对应要核对，不能直接强改 `zh`；meta 参考文字错误也不证明真实 prompt 错配。
10. **自动参考音的排序/切音/独立互证统一使用源音频窗**，复用DIS的`cue_reference_start/end`，不拿播放窗或扩展字幕时长充当有效源音时长；相同源窗不算独立证据。换参考失败时来源、锁定状态与参考质量须随已采纳音频回滚，失败尝试日志不能当实际生成来源；旧视频以已采纳请求hash核验字节。

### 素材与语料纪律
- **待QC参考音不增加视频数量**：用户已取消双视频方案；DIS保留唯一`DUBBED_VIDEO`及原契约，在原视频已有meta中增加按SHA去重的客户端ref/prompt。无需审核页改选视频类型；最终发布去meta需在真正发布入口处理，尚未接入，不提前多生成一份。QC rebuild不等于最终批准；新片参考不能凭同名代替旧参考，须本地字节SHA与旧请求记录一致才可认定找回旧客户端输入。恢复字节不等于取得GPU预处理结果/权重身份或完整重放证据。提取命令见DIS README。
- **合成成片不是训练语料**：不能抽它补真人语料、伪造同人 ref 或混入验证集。收到成片只能登记带时间点与反馈来源的开发坏例，并保留「语言质量未全面验证」。原始 take / ref 未取得时，不拿成片混音伪造因果 A/B、不宣称听感改善，也不猜内部 HTTP 路径去重抽。用户补原片后先核画面流与时间轴；原片混音切窗不等于实际分离/预处理后的 ref，无嵌入配音 meta 也不证明真人来源或训练授权。
- **四地区短剧来源链接已有（ID/VN/TH/PH），唯一登记处是 `docs/corpus_sourcing.md`。** 音轨语言、真人/合成来源、训练授权与质量**均未核验** ⇒ 只记录，不下载、不入库、不加训。旧「没有 drama」应理解为**尚无已核验可训练素材**，不要重复向用户索要链接。将来要用：先核授权与小样本质量，只在远端下载，按原音频/session/已知身份隔离 split，**不把角色名当配音员身份**。
- **drama 不是硬依赖，`enable_proj=true` 也不是必经步骤。** r9 同时改了源、gs2 份额、投影层和步数，只否定当轮组合，不判死整条 ref 路线；先保持投影关闭，数据有效后再单独对照。
- **同源切片 ref 仅为待验证实验**：须有单人证据、互不重叠的时间区间、准确 target 文本与原音频追溯。代码拒绝同 `origin_audio` 配对，**不能改名伪造不同原音频**；身份未知默认不配 ref。

### 验收口径
- **工程检查与语言验收分开。** 用户只做声学异常盲听（爆音、金属感、异常娃娃音、明显音色跳变、沙哑粗糙音），**不能要求其填五语种自然度/可懂度**；不确定可留空。口音、语言自然度、情绪一律标「未验证」——没有合格母语评审时，工程检查通过不等于语言验收通过，但这**不阻塞离线候选实验**。高频刺耳检测false不排除沙哑，人工时间点未定位时不把自动候选冒充已确认坏例。细则见 `docs/qc_gates.md`。
- ms 的盲听必须由马来西亚母语者做，**不能拿印尼语听感代替**；CER 或 Whisper 的 id/ms 标签不能当口音门禁。
- **联合模型是目标，分语种验收是前提**：任一语种相对基座退化即整轮不通过，不许用「平均变好」掩盖。
- **本项目离线评测仍是 Whisper large-v3**，不因线上 ASR 选型覆盖旧评分。生产不是全程固定 Qwen3（OmniVoice 合成文本 QC 默认 Qwen3，DIS 对可疑最终片段显式切另一后端复核，词时间戳按 aligner 支持回退）。`model=large-v3` 不等于切后端，对照要显式传 `asr_backend`；合成内置 Whisper 路径带期望文本 `initial_prompt`，不能和无提示分数直接比。**Whisper GPU 故障不等于需要微调或切 CPU**，也不默认重置设备或降级依赖。
- 分语种 ASR 首选需要同 WAV、同语言提示模式、无期望台词提示的对照，衡量误拒/漏检与 QC 总耗时，不凭 CER 最低或模型支持列表指定赢家。泰语已有「额外尾音被 Qwen 记录、Whisper 漏转」的反例 ⇒ 当前**没有**新的分语种 ASR A/B 结论。
- **当前优先级见 `TODO.md`**：明确坏例定位 → 验收口径/稳定性 → r8 LoRA 强度 → CFG/步数/ref 单变量对照 → 现有单人录音切片配 ref 小实验。

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
  /tmp/rsh 'nohup stdbuf -oL -eL uv run python -X utf8 -u -m <模块> ... > /root/autodl-tmp/<本轮唯一日志>.log 2>&1 & echo started'
  ```
  然后**另起命令**轮询该日志。`python -u` + `stdbuf -oL` 缺一不可，否则日志憋在缓冲区里看着像卡死。
- **多语种JSON读写显式`encoding="utf-8"`，长任务同时加`-X utf8`**：
  本轮原任务完成60条后默认编码变成ASCII，再读泰文plan失败；改变locale的来源未知，不猜测第三方库。
  共享`check_inputs`及本轮入口已修复，保留哈希守卫；不得跳过校验、覆盖成功报告或重跑已完成音频。
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
   - **适用范围**：完整语言质量结论仍需合格母语评审；当前用户只做声学异常盲听，不能要求其填写五语种自然度/可懂度。未评项保持未验证；只完成工程检查不算完整语言验收通过，不阻塞离线候选实验。
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
     两轨冲突时，在评审者能判断的维度**以盲听为准**；非母语者不能裁定声调、数字念法或口音正确性
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

> 这里每条只留**会改变下一步动作的规则**。实测数字、复现过程与调研明细在 **`docs/pitfalls.md`**（原节原文照录，未删减，按关键词搜即可定位），改代码或重跑流程前先查那里，别重复踩。

### 工程与环境
- **Gradio 流式**：按钮必须直接绑定生成器函数；用 `lambda` 包一层会把生成器对象本身渲染进文本框。
- **Gradio 下拉框**：`choices` 只在 `build_ui` 算一次；运行后会变的列表（数据集/配置/LoRA/上传目录）必须靠事件输出或 `Tab.select` 刷新。
- **`.env` 加载顺序**：`paths.py` 必须**先** `load_dotenv()` **再**算路径常量（`VOXFT_CKPT_ROOT`/`VOXFT_DATA_ROOT` 依赖此顺序）。
- **库函数不许 print**：UI 的 stdout 可能是已断开的 pty，`print` 抛 `[Errno 5]` 会把**已经成功**的操作报成失败。一律走 `progress` 回调，CLI 侧传 `progress=print`；启动 UI 用 `nohup ... > ui.out 2>&1 &` 或 tmux。
- **临时脚本里 `import huggingface_hub` 必须在 `import voxft.paths` 之后**，否则 `HF_ENDPOINT` 还没写进 `os.environ`，远程直连 huggingface.co 报 `[Errno 101]`。
- **`cmd | tail` 会吞掉失败**（退出码是 `tail` 的 0，`set -e` 拦不住），本轮因此真丢了 3GB 数据。用 `set -euo pipefail` 或重定向到文件再单独 `tail`，**且只有确认成功才删原始数据**。
- **视频容器解码走 PyAV**（`av>=12`，faster-whisper 的传递依赖）；soundfile 读不了 mp4/mkv，不引入系统 ffmpeg。
- **`rate` 的单位按文字系统判，不按有没有空格判**：泰文/缅文区块（`_NO_WORD_SPACE`）一律按字符计，否则 th 的 `rate` 会失真近 20 倍。`rate` 只在 `log.py` 展示、不参与判定，既有产物不需重跑。

### 磁盘、下载与训练执行
- **大盘约定（远程）**：`VOXFT_DATA_ROOT`/`VOXFT_CKPT_ROOT`/`HF_HOME` 都指 `/root/autodl-tmp/*`——**它是本实例唯一可写的大盘**，磁盘不够时不要去找别的路径，要去找可回收的东西（`autodl-pub` 只读、`autodl-fs` 不存在）。**回收顺序**：parquet 缓存（FLEURS 五 config 就 13G）→ `merged`（4.6G/份）→ `step_*`（只留 `latest`）→ `raw/<src>/audio`（`origin_audio` 只当字符串用，不读文件）→ 许可不明/SA 源。一轮五语种峰值约 20G，开工前先 `df -h /root/autodl-tmp`。
- **HF 生态**：`datasets` 锁定 `<4`（5.x 硬依赖 torchcodec 且与 cu124 torch 冲突）；`hf` CLI **不读项目 `.env`**，要手动 export 三个变量，优先用 `python -m voxft.data.prefetch`。**走镜像必须关 xet**（hf-mirror 不代理 CAS，reconstruction 阶段直连报 401），`paths._disable_xet_on_mirror()` 会自动 setdefault。进度监控用缓存目录大小轮询；xet 两阶段导致进度"回退"属正常。
- **下载通道**：代理分流与速率矩阵见「远程 GPU 机调试约定」。另三条：`.env` 默认（hf-mirror + 自动关 xet）就是最优组合，**别去"优化" endpoint**；**直连 + xet 必失败**；**切换 endpoint 或 xet 模式会让已下载分片变孤儿**（`.incomplete` 后缀变了不续传），重跑前先 `ls blobs/*.incomplete` 清掉。
- **FLEURS 一个 config 就是约 1.9GB 整片 parquet**，`--max-samples 60` 也要下完整片；**别指望流式省流量**（`load_dataset(streaming=True)` 对 fleurs 取首行时永久挂起，已加 120s socket 超时变成可诊断报错）。磁盘按「每语种 2GB」预算。
- **上传 HF**：`hf-mirror` **只读不写**，上传必须 `source /etc/network_turbo` + `HF_ENDPOINT=https://huggingface.co` + **显式 `HF_HUB_DISABLE_XET=1`**（此时自动关闭不触发）。⚠️ **`merge_lora` 会把基座的 `README.md` 一起拷过来**，而 `upload_folder` 只在 README 不存在时才生成卡片 ⇒ merge 后必须**显式覆盖** `README.md`（用 `docs/model_card.md`）。
- **训练默认**：`batch_size=2 + 梯度累积=8`（等效 batch 还需乘 GPU 数）；页面按 1 epoch 自动算步数，**换清单/卡数要重建配置**（`.plan.json` 留审计）；启动带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`save/valid_interval=250`；结束只保留最新 5 次 LoRA 运行。官方脚本"最后一步必存"会让短 run 多出冗余 `step_*`（submodule 只读改不了，磁盘紧就手动删）。
- **`max_grad_norm=1.0`、`num_workers=8`**：官方 v2 就是这两个值（早先写的"默认 0 = 不裁剪"是错的）。情感语料动态大，不裁剪更容易出梯度尖峰。
- **推理**：`load_denoiser=False`（去噪器依赖 modelscope，试听不需要）。
- **Whisper 权重用两个尺寸**：转写（会变成训练文本）用 `large-v3`（约 3GB），`ingest` 的整轨 VAD 与 `qc="whisper"` 用 `medium`（约 1.5GB）。`prefetch` 只预取 large-v3，**首次跑 ingest 或 whisper 质检会临时多下 medium**。可用 `VOXFT_WHISPER_MODEL`/`_LARGE` 指向本地目录。⚠️ 权重下载期间**没有任何进度输出**，与卡死无法区分，别急着重启（用 `du -sh $HF_HOME/hub/models--Systran--*` 判断）。
- **万级转写必须能断点续跑**：每 300 条及退出时原子保存完整原清单（含坏例与未处理行）。`WhisperModel.transcribe` 不接受 `batched`；ndarray 先转 16k。**推理异常必须中止，不可当语料坏例吞掉。** `--max-items` 试跑不回写原清单。
- **运行记录用 `python -m voxft.train.runlog` 生成，不要手抄**（从 yaml/`.plan.json`/`mix.json`/`train.log`/eval JSON 抽字段追加到 `docs/runs.md`）。`--verdict/--next/--notes` 不留空；`CER_NOISE=0.05` 各处共用；**不手改历史原始指标**。

### 数据源与语料（调研明细在 `docs/corpus_sourcing.md` 与 `docs/pitfalls.md`）
- **数据源首选**：th 用 `thai_ser` impro / 审核后 `yodas_th`；tl 用自有真人 `drama_tl`（`filipino_emotion` 仅待审、`filswitch` 只低比例补 Taglish 发音）；vi/id 表现力只有自建 `drama_vi`/`drama_id`，自然口语锚点首选 `gigaspeech2_vi/id`；ms 表现力只有自建 `drama_ms`，自然口语用 `yodas2_ms`；`fleurs_*` 一律只当发音补充；中英文回放 `aishell3`/`fleurs_zh`/`fleurs_en`/`replay_en`。**不能把朗读数据当去念稿感主力。**
- **自然口语源可用性（已逐个撞过，别重复调研）**：`gigaspeech2` **只有 th/vi/id**，走 `kind="hf_tar"`（`refs/convert/parquet` 分支不存在但索引 API 照样返回 200，解析后必 404），**p50 4.2-6.2s，是五语种里最贴近线上 cue 长度的现货**；`yodas2_ms` 走 sarulab-speech 的 API parquet 端点，**列是 WebDataset 原样成员不是 audio/text**；`cv22_*` 走 `kind="cv22"` 绕开加载脚本，**`client_id` 是账号级持久身份**可参与 ref 配对（**tl/fil/ms 依旧没有**）；`filipino_speech` 行过滤后**产出率约 1%，不值**。
- **FLEURS 全量规模**：五语种合计 **43.9h 全 CC-BY-4.0**，加工产出率 99.3%-99.8%，edge VAD 裁完各语种 6.3-7.8h。**时长 p50 10.4-14.1s、3-8s 只占 5.8%(fil)-24%(th)——这是它与线上 cue 长度的根本错配。** 列取 `raw_transcription`（带大小写与标点）而非 `transcription`；**两列数字序列 100% 一致**。
- **加工产出率决定值不值得加工**：FLEURS 几乎全保留，**`tagalog_tts` 只有 42%**（84 条因不足 3s 被砍）。⇒ 短切片源**先量时长分布再决定要不要加工**，别按原始条数估产能。
- **自建短剧素材的候选路线（授权与质量待核验，尚未执行）**：⚠️ **将来路线，不是当前能力**——本项目现在**不做声源分离/降噪**（无 demucs 依赖，见 README 与 `docs/finetune_playbook.md`）。真要做，五个要点：分离用 Demucs `htdemucs`，但**不替代逐条试听、也解决不了混响**；`decode_to_wav` **只取 `streams.audio[0]`**（双音轨给错整集白解）；台词本才是权威文本；**自动聚类与同角色名都不能单独证明同人**；**split 陷阱**——并查集同时 union「同一演员」与「同一集」，主演只有 5-15 人 ⇒ 多集并成一组、极端情况没有验证集，用 `--holdout` 钉住并查 `stats.json`。**先拿 1 集量产出率，别一上来全量。**
- **`thai_ser` 没有名为 `audio` 的列**（四路麦），必须靠 registry 的 `audio_cols` 映射，否则整个源在下载阶段被静默跳过；`mic_zoom` 是网络录音，不用。
- **`yodas_th` 会话**：`utt_id.rsplit("-", 3)[0]` 保留完整视频 ID。speaker_id 是视频级近似身份，**不作 ref 依据**；上游逐条峰值归一，不据此标音量；无原始连续时间关系就不拼接。
- **FilSwitch 下载**：转换 parquet 可以只有元数据，音频在原仓库的独立 FLAC，**`bytes=None` 不等于无音频**。**不要把音频地址套到 `refs/convert/parquet` 分支**；读取失败必须记录原因，不能静默跳过整包。
- **AISHELL-3**：content.txt 的同一正文列交错汉字与拼音，**必须剔除拼音**；旧 processed 清单要重新加工，不能直接混入。

### 数据加工规则
- **首尾裁切按源分流**：`edge_trim_ratio` 朗读 0.06 / 表演 0.02。**RMS 门限单独用有个悬崖**（留白在有声电平 −24dB 以内时一帧都裁不动，输出与输入完全一致）⇒ 朗读源改用 `edge_vad`（Silero），表演源仍走 RMS（**抽气声是表演的一部分**）。低电平段须连续 ≥0.25s 才裁，否则啃掉词首清辅音。两个必记常数：`min_silence_duration_ms=500`（不是默认 2000）；喂 VAD 的数组必须 **float32**。**尾静音另压硬上限 `max_tail_silence_sec=0.15`**（`_cap_tail_silence`，只裁尾不裁头）：训练样本的尾静音会直接教会模型垫尾，三轮一致复现。`stats.json` 记全部门限，旧产物能反推。
- **听不清的按 ASR 置信度丢**：只对 `needs_transcribe` 的源启用，门限沿用 faster-whisper 默认（`avg_logprob < -1.0` 或 `no_speech_prob > 0.6`）。这不违反"拒绝 min_snr_db/min_f0_std 硬筛"——被禁的是能量分位差与 F0 这类伪指标，不是 ASR 自己的置信度。
- **code-switch 语种不能只认目标语种**：`Source.languages()` 查 `CODE_SWITCH_ACCEPT`，对 tl/id/**ms** 默认放行 `en`；**vi 默认从严**，实测误杀再加 `accept_langs=("vi","en")`，别提前放开。有权威文本的朗读源用 `accept_langs` **覆盖掉**默认放行——语种不符意味着错行，不是 code-switch。
- **ms 与 id 高度互通，语种过滤挡不住互串**：Whisper 检测与 YODAS 上游标签都会互判，`drop_lang` 对 id/ms **完全无效** ⇒ `yodas2_ms` 必须抽样听，**ms 的盲听必须由马来西亚母语者做**。离线可先过词汇判据（`ialah`/`bermaksud` vs `adalah`/`berarti`），**但只验正字法，口音仍须母语者**。收益是 id 对 ms 有正迁移（这是两者进同一联合 LoRA 的主要理由）。
- **数据身份**：MFCC 聚类仅供审计，不能证明同人，**更不能调低阈值强凑 ref**；身份未知默认不配 ref。同一演员跨源用统一 speaker_namespace/ID；先隔离 train/val，再在集合内配 ref，混合与训练前再次检查泄漏。
- **响度**：仅已验证说话人统一增益到 −24 dBFS；防削波时整个说话人共同回退，**禁止单条峰值归一**；未知身份不统一调响度；上游已抹掉的动态不能恢复。
- **声学描述**：`f0_std_st` 含泰语声调与清浊音误差，不代表自然度；`energy_range_db` 是能量分位差不是 SNR。**拒绝 min_snr_db/min_f0_std 硬筛**，不把这些指标自动变成情绪指令（`--min-snr-db` CLI 已删，`Options` 字段与守卫保留）。**`librosa.yin` 的 `frame_length` 必须按 sr 推导**（`2*sr/60` 向上取 2 的幂），硬编码 1024 会让 48k 合成音频的 `f0_std_st` 失真约 9%。
- **重加工**：每次写新音频子目录，不覆盖旧清单引用的音频；旧产物不自动清理。JSONL 格式与远程执行命令见 README。
- **追加素材会让旧验证集泄漏**：`split_records` 的随机分组依赖清单长度，追加后重新加工会把上一轮验证组整体重排进训练集，**已训 run 的评测结论随之作废**。用 `data/raw/<source>/holdout.json` 钉住固定验证集；矛盾组合直接报错，不静默把钉住的数据喂进训练。`ingest` 的 `session` 自动设为素材 ID。

### 许可红线与禁用语料
- **泰语源有 CC-BY-SA 红线**（`thai_ser`、`Porjai-central`；Porjai 的 `pattani`/`khummuang` 是 NC-SA 直接排除）。据此定的红线：**含 SA 数据训练的 LoRA 与 merge 后完整模型一律不对外分发**（不传 HF、不随客户交付、不开源），只通过 API 交付合成音频；要对外发布就谈商业授权或只用 Apache-2.0/CC-BY 源。
- **合成语音语料禁止用于补量**：`laion/dramabox-voice-acting-data-annotated`（源头是 `ResembleAI/Dramabox` 与 `gemini-2.5-pro-tts`，文件名里的 seed 就是生成采样；**只有标注 schema 可参考**）、OpenSpeechHub 三个泰语集、`mesolitica/Malaysian-TTS`。
- **`mesolitica`（Malaysia-AI）全线无许可声明**，拿到书面授权前一律不得使用（**无声明 ≠ 开放，默认全权保留**）。`espnet/floras` 是 3TB 长音频基准不可切分；`disco-eth/WorldSpeech` 与 `MERaLiON/sea_audiobench_*` 是 NC/NC-ND。
- **Tagalog 无可商用的开源真人表演语料**（已核实，别重复调研）：CV tl `recordedHours=0`、YODAS 无 tl/fil、OpenSLR 无、HF 只有厂商 sample、SEACrowd 全是 text/图像、`yapdo-convo` 无许可声明。⇒ 只能付费或自建 `drama_tl`。
- **MagicHub 别记错两个库**：`ASR-SFDuSC` 是 4.58h/10 人朗读、**CC-BY-NC-ND**（ND 禁演绎，微调就是演绎），可用性为 0；有价值的是 `ASR-BigFTagaCSC`（**1285h / 514 人自发对话**，专有授权需询价），514 个真实身份是 YouTube 抓取源给不了的，但**无情绪标签、不是表演**。
- **低资源语种的调研结论必须核实到页面/API 原文**：本轮出现过一篇编造的竞品论文（"JaiTTS arXiv 2604.27607"，arXiv API 查无此 ID）和把无许可的 `yapdo-convo` 说成 CC-BY-4.0。**一条编造的"有现成大规模语料"足以让人跳过真正该做的自建工作。**
- **网络受限时不得凭记忆写语料结论**：只注册可由本仓库既有事实复现的源，其余候选写进待核实清单并附远程核实命令，规模与许可一律标「未核实」不写数字。**宁可留空也不要填一个看起来像结论的编造值。**

### 语种接入结论（逐条带 submodule `文件:行号` 的核实在 `docs/vi_id_support.md` / `docs/ms_support.md`）
- **vi/id**：基座官方 30 语种**已含 vi/id**（id WER 1.36% / vi 1.56%，**优于 tl 的 2.63%**），无 language token/lang_id、tokenizer **0 UNK** ⇒ **submodule 零改动，别试图加语种信号**。两者都**无已核实的真人表演语料**，首轮表演档 **0%**，**不能声称情绪表现力改善**。基座 TN 是 zh/en 二分 ⇒ 推理侧必须 `normalize=False`，**数字/货币念法是数据侧责任**。vi 是 6 声调语言，**`ngã`/`nặng` 收尾的句尾嘎裂声有被 RMS 裁掉的风险（未实测，验收重点听）**；`rate` 是音节/秒、**WER 是音节级**，不与词级横比。控制前缀守卫**拦不住印尼语**，只能靠标注纪律。许可全是 Apache-2.0/CC-BY/CC0，**无 SA 红线**。
- **ms**：基座已含 ms，**ms WER 1.75% 是五个目标语种里唯一输给竞品的**（Fish S2-Pro 1.41%），提升空间最大；但 **MLS 两张表都没有 ms**，克隆能力基线**无官方数字可引用**，只能自己 A/B 实测。纯 ASCII、tokenizer **0 UNK** ⇒ **submodule 零改动**；**`WER_LANGS` 含 ms**（与 id/tl/en 同量纲）。**命名陷阱：FLEURS 的 config 是 `ms_my`，`my` 是马来西亚国家码，不是缅甸语。** CV22 与 gigaspeech2 都**没有 ms**（别再去找 `cv22_ms`/`gigaspeech2_ms`）。无 SA 红线，但 CC-BY-3.0 要求署名。
- **泰语转写可换 `typhoon-ai/typhoon-whisper-large-v3`**（MIT，约 11000h 泰语微调），**但不是即插即用**：是 transformers 格式不是 CTranslate2，要转格式或单开路径；MIT 之外另有 OpenTyphoon T&C；**只有泰语**。优先级低于把数据搞到手。
- **跨语言同人 ref 数据不存在，别再找（已确认死路）**：主调方确认自有配音演员没有多语言版本；FLEURS 各语种是不同众包说话人；CV 的 `client_id` 理论上跨 locale 一致但 tl 为 0、ms 要 CV23 起。⇒ **`pair_references` 三级优先的第一级（中/英回放）永远是空的**，只能落到同语种 ref；同语种 ref 仍有价值（让线上唯一在用的 `[103 ref 104][text][101 target 102]` 打包路径进入训练），但**对跨语言场景是否有正迁移必须实测，不能预判**。

### 离线指标的口径与已定案的可信度（标定过程在 `docs/qc_gates.md`）
- **`metallic` 与 `low_snr` 已永久降级为参考值，不作门禁**（对人工标注 ROC **AUC=0.060 / 0.509**）：阈值是在 OmniVoice **后处理过**的音频上标定的，与裸输出频谱形态不同；`low_snr` 频繁触发的主因是 **ref 噪底**不是生成问题。**不是"暂无区分力"，不要再尝试重新标定，更不要拿它否决任何一轮微调。** 实现细节仍要记住：`metallic` 下限取 3kHz 不是 1.8kHz（1.8kHz 会撞 F2/F3 共振峰误判干净人声），sr<12000 或 <0.45s 时返回 **`None` 而不是 `False`**。
- **数字类 case 必须单列**（`mean_cer_non_numeric`）：Whisper 自己会把口播数字词归一成阿拉伯数字或货币符号，base 与 ckpt 会得到同一个高 CER——**那是 ASR 假象不是 TTS 差异**。已 verbalize 的要在 case 里显式写 `"numeric": true`（自动检测抓不到）。剔掉数字类后两边都已饱和 ⇒ 准确说法是「持平且无可测空间」。
- **多读与少读分开**：`suspected_truncation` 保持原语义 = 少读/漏尾（**不要改定义**，否则与前三轮已记录的数字不可比），`over_read` = 归一化文本 >1.4× 参考，另有 `len_ratio`。
- **`chars_per_sec` 必须配 `speech_ratio` 一起看，单看会得出完全错误的结论**：实测"说慢了 12.7%"拆开是**有声段字/秒只低 1.7%，多出来的几乎全是尾部静音**。线上 `trim_silence_vad=True` 会裁掉所以基本无感，但不裁切的下游会拿到长 12% 的音频。**刻意没移植** `duration_off_reference`（OmniVoice 里是死代码：ref 只是音色锚，其时长与期望输出长度无关）与 `RuleDurationEstimator`（A/B 比的是相对值）。
- **`speaker_sim` 已换成 WavLM X-vector，MFCC 版废弃**（MFCC 动态范围不足）。⚠️ **两个坑**：① 必须用 `WavLMForXVector`，用 `Wav2Vec2ForXVector` 会打印一大片 `MISSING` 并把 encoder **随机初始化**——不报错，只是嵌入全是垃圾；② 拿**随机噪声**验证动态范围毫无意义（噪声没有说话人身份，嵌入会塌到同一方向），必须用真实语音的不同说话人自检。可用 `VOXFT_SPK_EMB_MODEL` 指向本地目录。与 OmniVoice 生产用的 ERes2NetV2（门限 0.45）**刻度不可互换**。
- **时长类指标可信，且大部分不需要母语者就能判**：首次全评盲听按最小差值门槛**无语种退化**；「人工判 B 更差」的条目绝大多数是 **B 明显变长而 CER 完全没变**，与自动测到的尾静音增加指向同一缺陷。⇒ **真正只能靠母语者的只有「语调是否地道」「情绪是否对」**，而这两类在零表演语料的现状下本来就不可能改善。平局占多数 ⇒ **「整体更自然」不成立**。
- **离线指标的分辨率已经低于 run 间方差，别再对着它调配比**：配比完全相同的两轮 `ms` CER 能差出 **0.042**（混合用共享 RNG，改任何一部分权重都会挪动后面所有抽样，再叠加训练非确定性）⇒ **28 case × 3 seed 的差值在 ±0.04 以内一律当噪声**。要继续调参先把统计功效提上来（每语种 30+ case、5 seed），否则就是在噪声里挑好看的数字。
- **验收 case 的 ref 语言必须覆盖线上真实分布，单一 ref 的结论不可外推**（本轮实测推翻过一次验收）：线上真实用法是**中文或英文 ref → th/tl/vi/id/ms**，但前三轮全部用那条**菲律宾语** ref；补测 zh/en 后**同一份权重的胜负语种完全变了**。base 自己在 zh/en ref 下就差得多 ⇒ **跨语言 ref 距离越远越难，是基座特性不是微调引入的**。⚠️ 每种 ref 语言目前只有 **1 条**，与说话人/录音质量混淆，要分离结论需每种 3-5 条重测。
- **音色一致性：指标改善有限，不等于结构上改不动**：三轮微调的 WavLM「输出 vs ref」变化都在 ±0.009 内。**这些是观测，不能推出「必须开投影层」**——ref 条件还经过可训练的 LM/DiT，官方 v2 LoRA 默认 `enable_proj=false`。投影开关另作单变量实验，不盲开；30–50% 是混合后可信 ref 目标，**不为凑数伪造身份**。

### 实验结论（完整数字在 `docs/runs.md`）
- **短 cue 出分布是微调最主要的失败模式（round 1 实测，最重要的一条）**：只用 FLEURS（p50 10-14s）训出的联合 LoRA，在**线上那种 1-3s 的配音 cue 上会跑飞**，但同一份 LoRA 在 FLEURS 长度的探针上反而更好 ⇒ 退化不是"某语种变差"，是**长度出分布**；val loss 更优的 checkpoint 同样退化，**不是过训，退回去也没用**。对策是把训练数据长度分布拉到 cue 量级（`gigaspeech2` 是现货，`drama_*` 是正解）。`process_dataset` 有 `3 <= min_dur <= max_dur <= 30` 硬守卫，**不要为了塞进 1-3s 样本去拆它**。
- **被实测否掉的假设要老实改掉**：曾把 vi/id 数字 case 退化归因为"带数字训练时长下降"，据此提高 FLEURS 份额后带数字时长确实恢复了，**但 vi CER 反而恶化**。⇒ 真正的关键是**长度分布**，带数字时长与结果无因果关系。
- **val loss 不能用来选 checkpoint**：val loss 全程最优的那轮验收指标反而最差，val 最优的中间 step 在**所有语种**上都不如 `latest`。三轮一致 ⇒ **一律交付 `latest`**，val loss 只用来看有没有发散。
- **单 seed 的 CER 会给出完全相反的结论，这是最贵的一课**：同一条 case 在"每 case 各自 seed"下看着像修好了 −76% 的大 bug，换成 `--seeds 42 43 44` 后真相是 ckpt 严重退化。**跨 seed 一致的才是真信号** ⇒ **验收一律 ≥3 seed**，base 与 ckpt 必须用同一组 seed 才叫配对；**别把 base 的偶发崩溃当成微调的功劳**。
- **离线验收**：逐 case 固定 text/lang/ref_audio/ref_lang/control/seed；**A/B 禁用自动坏例重试**，普通试听保持原设置。CER/WER/疑似漏尾仅诊断，自然度/情绪/音色/真实截断需母语盲听，**F0 不作通过门限**。参照锚点（`eval base`，无 ref，cfg 2.0/20 步）：`th 0.0`、`vi 0.0`、`tl 0.0119`、`id 0.0238`、**`ms 0.0814`——ms 基座最差**，与「ms 是唯一输给竞品」一致。⚠️ 小样本朗读锚点**只能当量级参照，不能当验收结论**。
- **生产口径基线**（`eval_cases/omnivoice_prod.jsonl`，28 case × 3 seed，cfg 1.8/20 步/线上 4.47s 菲律宾语 ref）：`base` 总体 CER **0.0774**、疑似漏尾 0.0952。**基座的弱点全在数字与货币**（`vi_digit` 会跑飞、`250.000 đồng` 丢声调符号、`RM1,200` 被念成 `Rp`、`เที่ยวบิน 926` 念成 `916`），非数字台词 base 基本全对 ⇒ **想靠微调压总体 CER，空间几乎只在数字类；普通台词只能验「不退化」**。
- **验收 case 集在 `eval_cases/omnivoice_prod.jsonl`**：按线上形态设计（单句 cue、保留大小写与句末标点、跨语言 reference-only），覆盖数字/货币、Taglish/Manglish 借词、vi 句尾 nặng 调嘎裂声。**`prod_ref_fil.wav` 故意不入库**（主调方音色库素材，不能公开分发）：新环境先从 OmniVoice 归档 payload `results/voxcpm_quality_cases.json` 的 `reference_audio_base64` 解出来。`tl_digit` 是压力 case，`tl_digit_prod` 才是线上 fil 的真实形态（TN 已 verbalize）。
- **v2 不等于独立文本留出集**：r8 的 267 条训练记录覆盖 95/182 条 v2 评测文本——这是**文本暴露**，报告须分 `r8_seen_text` / `r8_unseen_text`。补充留出集 `eval_cases/fleurs_test_holdout_20260917.jsonl`（五语种各 30 条，只取文本）已冻结、不用于训练；**反复调参后应降级为开发验证集**，不能继续称最终独立留出。
- **联合微调的四个可观测点**（缺一个就无法归因，别删）：① `dataset_summary.language_hours`——配比口径是**按有效音频时长**，按条数的 Counter 看不出占比；② `mix.json.language_shares`（`requested`/`actual`/`hours`），低于请求值 90% 时告警，**缺口不自动重分配**给其他语种（那正是大语种吃掉小语种的机制），也不靠 3× 重复强凑；③ `split_records` 的 **val 按语种配额**（`val_min=16`，`n//5` 是 20% 硬上限），全局 `val_ratio` 会让小语种只摊到几条验证样本；④ eval 报告的 **`by_lang`**，联合 run 必须逐语种与 `eval base` 的同一份 case 对比。另：`pair_references` 跨语言 ref 池是**三级优先**；`.plan.json` 记 `langs`；`_source_lang` 对未登记源返回 **`unknown` 而不是猜 `zh`**。

## Submodule 升级
```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```
