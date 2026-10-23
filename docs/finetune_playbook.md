# VoxCPM 2 短剧配音 LoRA 微调流程（泰语 TH / Tagalog TL / 印尼语 ID）

面向的生产形态：**参考音频是中文或英文发音，输出是泰语、Tagalog/Taglish 或印尼语台词**，要求情绪可被中英文控制前缀驱动、发音清晰不漏词、音色跟着 ref 走。

本文是可以照着一步步执行的流程，三个语种共用。开发约定与踩坑记录见 [AGENTS.md](../AGENTS.md)，数据源取舍背景见 [README.md](../README.md)，**ID 特有的依据、语料核实状态与口径差异见 [vi_id_support.md](vi_id_support.md)**（同文件的 vi 部分与 [ms_support.md](ms_support.md) 已转存档）。

## 2026-10-22：r24 冷启动单变量轮（用户授权，已开训）

**授权与范围**：用户在看完 #8/#9 取证与 r19→r23 时长回归证据后决定「保持 r23，直接开训」。
本轮只做一个单变量实验；不追加 epoch、不改 CFG/步数、不动 HF 与生产。此前的暂停交接项（1.0/0.75 盲听包等）
保留在下一节，未取消、也未插队。

### 1. 开训依据（全部来自已冻结记录与本机复算，无新实验）

**成片取证：第43剧108集 #8/#9「同人音色差异」**（目标语实际是 **id**，不是 CDN 暗示的 VN）
- 同 speaker key `review_4f50387a`，但**参考字节不同源**：#8 = `8a4cc827…`（`episode_current`），
  #9 = `e92e2477…`（`episode_alternate` + `fallback=true`，锚点 cue 24）。本机 WavLM：`ref#8↔ref#9` 仅
  **0.6824**，而 `ref#9↔ref#24`=**0.9841**、`ref#9↔ref#10`=0.8912 ⇒ **#8 那条参考是该 speaker 参考池的离群值**
  （F0 134.1 vs 190.8/203.0/194.5Hz；DIS 另记录它头部 0.25s 被判「female against body male」并裁掉）。
- #9 换参考是**救援结果**：两次 seed 重试都跑飞（6.40s / 6.26s 对 1.2s 窗）+ severe `speaker_mismatch`
  → identity 预算用尽 → 改用 cue 24 参考。最终音频是从 6.26s 跑飞输出 `external silence trim 0.000–1.340s`
  **硬裁**出来的，DIS 自判 `tts_failed` → 源人声抑制 → **keep-best 出片（score −269.1）**，
  带 `unresolved_artifact=true`、`pitch_register_drift`、`runaway salvage rejected: untrusted_alignment`。
- **控制条件也不同**：#8 最终 take 带 style control「as a sharp but controlled outburst」（angry/shouting）；
  #9 最终 take `voxcpm_style_control_suppressed=true` + `style_retry_mode=reference_only`
  （原请求「openly angry, with full exclamatory force」那次因跑飞被拒）。⇒ **两条不是模型 A/B，是两组不同条件**。
- 后处理放大差异：tempo 1.500×（需 1.99× 被截）vs 1.314×；电平匹配 −0.77dB vs **+3.54dB**。
  输出 F0 交叉：#8 173.9Hz（高于自己 ref 40Hz）、#9 137.4Hz（低于自己 ref 53Hz）。
- DIS 自己已测到：`timbre_consistency.pitch_dispersion[review_4f50387a]` spread **2.192 / over_threshold=true**
  （drifted 含 #9 的 140.4Hz，ratio 1.527），`anchor_similarity` min **0.1354**、median 0.5052（生产门限 0.45）。
  两条 cue 的 `speaker_similarity_to_reference` = 0.3274 / 0.3826，**均低于 0.45**。
- ⚠️ **前提存疑**：该 speaker key 下 10 条 cue 的参考 F0 跨 142.9–333.3Hz，其中 cue 10/13/17/18 的参考被判
  **female**（置信 0.69–0.885）而请求 `declared_gender=male`，`speaker_status` 全为 `unverified`
  ⇒ 这个聚类可能混入多个真人，「#8/#9 是同一个人」在参考层面尚未证实。
- 冻结产物（本机 `work/video_review/20261022_43_108/`）：视频 SHA256
  `a6a24d9ada1f86d5d358e26fd8277442baa9a4c2b5f3333ec18956f9e7d1e7e8`（25,178,696B，与 CDN content-length 一致）、
  `embedded_meta.json`、`review-references/`（18 文件 3,865,972B，逐 cue SHA 校验通过）、`seg8/seg9_final_mix.wav`。
  DIS 客户端源码 SHA `42eeda758ba30ec0b095ac07a0ef2c50d8afa7e35f168cd7ecba05563a845875`、endpoint
  `220.130.209.122:41853`、`model_id="openbmb/VoxCPM2"`（**客户端配置，不是权重加载证明**；`git_commit=null`，
  `scope=publishing_process_not_tts_server_or_cached_take`）。本轮**未新增 TTS/ASR**，未重渲，未改生产。

**时长/停止回归的转折点在 r19，不在音色指标**
| 轮次 | 初始化 | 步数 | id p90 尾静音 | id speech_ratio | dramaprod id 跑飞 |
|---|---|---|---|---|---|
| base | — | — | 0.180s | 0.882 | — |
| r13x | 冷启动 | 1094 | +0.140s | — | 0/96 |
| r18 | 冷启动 | 1390 | +0.140s | — | — |
| r19 | 冷启动 | 1202 | **+0.640s（0.82s，越上限）** | **−0.145** | 8/96 |
| r23 | **r22 续训** | 2909 | **+0.840s（1.02s，越上限）** | **−0.147** | 4/96 |

r23 另有 id 非数字 audio_sec +24%。**r21 已证「把 thai_ser 全删也洗不掉」（runs.md 原话：续训没能洗掉权重残留）**；
r18 的 id 份额更高（47.5%）却只 +0.14s ⇒ 既不是 id 份额，也不是删数据能修的。
结构上一致：`enable_proj=false` 下 `stop_proj`/`stop_head` **全冻结**，停止行为只能被 LM/DiT 的 LoRA 漂移间接带坏
⇒ **累积 2909 步的续训链就是那个漂移量**。

**音色一致性本身没有可测退化**（本机复算既有冻结报告）
- r8 vs base（`checkpoints/holdout_eval_20260917/voice_stability/`，WavLM 两两 cosine，150 组 cross-seed /
  325 组 cross-cue 配对）：**r8 略优**，cross_seed Δmean id +0.0023 / th +0.0017 / tl +0.0001，
  cross_cue +0.0016/+0.0015/+0.0001，「更差」组数约一半 ⇒ 全在 ±0.009 run 间噪声内。
- 2026-10-22 四臂（各 1000 输出、200 case×5 seed）：sim-to-ref 三语种全部 ≥ base（id +0.0093~+0.0155）；
  同 ref 跨句离散度 **r23 比 base 更紧**（id 0.0072 vs 0.0136、tl 0.0096 vs 0.0122），
  未晋级的 off/on 反而更松（id 0.0192 / 0.0267）。Δsim(emo−plain)：r23 −0.0127，优于 r13x −0.0236。
- 边界：这两套离线集都是干净 FLEURS 参考 + 4–12s 朗读文本，**覆盖不到 #8/#9 那种 1.2s 窗 + 1.9s 低 SNR 参考 +
  跑飞后硬裁**的形态；不能用它们证明生产没有音色问题，只能证明「声纹相似度」这一维没有系统性退化。

### 2. 生产权重现实核验（与文档口径不一致，如实记录）

HF `FrankLiuDundun/voxcpm-finetune-lora` 当前发布的**是 r23**：`lora/lora_weights.safetensors` 的 LFS oid
`8ca1dc95dbbe3234967e09d3c459a8ebe6482193eded7934efe196997c8b0ef2`（144,749,240B）与远端
`voxft_ckpt/lora_omni5_r23_thtlboost_cont/latest`（`training_state.json` step 2909）**逐字节一致**，
README 亦写「当前版本为 r23」，仓库 lastModified `2026-10-21T14:21:34Z`（北京 22:21；r23 生成于 22:07）。
⇒ TODO/AGENTS 里的「生产指定 r8」是**指定口径，不是线上加载证明**；GPU 实例实际加载的权重仍未核验
（用户 2026-10-22 选择不追这条）。**用户决定：保持 r23，不回滚 r18**（r18 仍可从 HF revision `7e5ffab6` 取回，
远端 r18 checkpoint 已被「只留最新 5 次」清掉，现仅存 r8/r13x/r17/r23 + r23 的 step_2250/2500/2750）。

### 3. r24 冻结方案（唯一变量 = 初始化与总步数）

- **假设（可证伪）**：r19 引入、经 r20→r23 五轮续训继承的停止/时长失控，主因是**续训链累积的 LoRA 漂移**，
  不是 r23 的数据本身。
- **A 臂 `lora_omni5_r24_coldstart`**：r23 清单**原样**（`/root/autodl-tmp/voxft_data/processed/r23_thtlboost/`，
  train 7996 行 / val 1367 行 / 16.554h；id 3247、th 2833、tl 1535、zh 253、en 128）
  + **官方基座冷启动**（`pretrained_path` = VoxCPM2 snapshot `32279eff…`；save_path 为新目录，开训前已确认不存在 ⇒ 无 resume）
  + **3 epoch = 1500 步**（`ceil(7996×3/16)`）。
- 超参与 r23 逐字段相同，`diff` 只有 4 处：`num_iters` 2909→1500、`max_steps` 同、`warmup_steps` 120→150
  （builder 的 10% 规则）、`save_path`/`tensorboard` 路径。其余：LoRA r64/α64/dropout0.05、
  `enable_lm/dit=true`、`enable_proj=false`、lr 1e-4、batch2×累积8、wd 0.01、max_grad_norm 1.0、
  max_batch_tokens 8192、save/valid_interval 250、`lambdas` diff/stop 各 1.0。
- 配置由**同一份 builder 生成**（本地与远端 `yaml_builder.py` SHA `d422a939…` 一致、`launcher.py` `e76cd5f2…` 一致），
  `.plan.json` 留审计（epochs 3.0 / effective_batch 16 / train_samples 7996 / langs 同上）。
- **两头都有已知对照**：对 r23 只差初始化与总步数；对 r18（冷启动 1390 步、尾静音 +0.14s）只差数据。
- **判读（开训前写死，不看结果再改）**：
  - 达标 ⇒ 坐实续训链是主因，A 臂成为晋级候选，并说明 r19 的数据变更本身未必有害；
  - 仍 0.8–1.0s ⇒ 主因在 r23 数据，**下一臂才做时长卫生**（剔 >15s 目标 + 尾裁 0.15→0.06），仍冷启动、仍单变量。
- **晋级门**（全用既有门禁，不新造）：id/th/tl 的 p90 尾静音增量 ≤0.10s 且 id 绝对值 ≤0.5s；
  speech_ratio 降幅 ≤0.05；dramaprod id 跑飞 0/96；v2 三语种 CER 不劣于 r23 超噪声（±0.04）；
  Δsim(emo−plain) 不劣于 −0.013 量级。**任一语种退化即整轮不通过。**
- **预算**：训练 1500 步（实测约 1.75s/步 ⇒ 约 44 分钟）；评测 v2-118×3 seed + dramaprod 112×3 seed ≈ 690 输出。
  磁盘：开训前 14G 空闲；实测官方脚本只自留最后两个 `step_*`（1499/1500，各 415M）+ `latest`，
  收尾时远端仍剩 13G ⇒ 未删任何 checkpoint。
- **启动记录**：北京时间 **2026-10-22 21:26:58** 起（`step_0000000` 落盘时刻），单卡 GPU（28.9GB/49140MiB），
  日志 `/root/autodl-tmp/r24_coldstart_train.log`，val step0 `loss/total` **1.021114**（对照 r18 首值 1.0416）。
  完成判活看**进程退出 + `training_state.json` 步数到 1500**，不看 `latest/` 是否存在（官方脚本 step 0 就存一份）。

### 4. 明确不做

- 不为 1.2s 短 cue 跑飞开训练轮：`process_dataset` 的 3s 硬下限把它们挡在训练门外，**结构上学不到**，只能推理侧兜。
- 不做无假设的 lr / r / 强度 / `loss/stop` 权重网格（`loss/stop` 在 LoRA 下只能通过 LM 隐状态间接起作用，停止头冻结）。
- 不给 r23 继续加 epoch，不动 HF 与生产，不删 r23 的中间 `step_*`（是证据）。

### 5. 本轮顺手修掉的代码问题（含测试）

- `runlog._read_yaml` 缺配置时静默返回 `{}` ⇒ `finetune` 被写成 `full`、超参全 None。**这就是 r19 被错记成
  「全量微调」的原因**（远端 `configs/lora_omni5_r19_emotion_3ep.yaml` 实际是 LoRA r64、1202 步）。
  改为硬失败 `FileNotFoundError`，新增 `tests/test_runlog.py`（2 项）；`docs/runs.md` 加更正说明、不改历史指标。
- `launcher.preflight` 的 3-epoch 警告没给 `steps_for_epochs` 的向上取整留余量：1500×16=24000 > 3×7996=23988，
  正好 3 epoch 的配置也被报「超过 3 epoch」。改为留一步余量，补 2 项测试。
- 全套 **253 通过**（新增 2 项在改前失败）；两个文件已同步远端，SHA 双侧一致
  （`runlog.py` `2b69444f…`、`launcher.py` `92813372…`）。

### 6. 结果与判定（2026-10-22 22:53 完成：**不通过、不晋级，生产保持 r23**）

三套验收全部与 base/r23 同口径（cfg1.8 / 20 步 / 禁重试 / large-v3 / seed 42-44），base 与 r23 **未重跑**。
完整数字在 `docs/runs.md` 的 `lora_omni5_r24_coldstart` 段（由 `voxft.train.runlog` 生成，未手抄）。

| dramaprod（112×3=336） | base | r23 | **r24 冷启动** | 门禁 |
|---|---|---|---|---|
| id CER | 0.0688 | 0.0752 | **0.1298** | Δbase >0.05 ⇒ **越红线** |
| id p90 尾静音 | 0.18 | 1.02 | **0.78** | ≤0.5s ⇒ **越上限**（比 r23 好 0.24s） |
| id speech_ratio | 0.882 | 0.735 | **0.712** | 降幅 ≤0.05 ⇒ **越门** |
| id audio_sec | 2.53 | 3.13 | **3.61** | +43% vs base |
| id 跑飞 | 0/96 | 3/96 | **5/96** | 0/96 ⇒ **不达标** |
| th CER | 0.1125 | **0.0773** | 0.1183 | r23 的收益被丢掉 |
| tl CER | 0.0106 | 0.0139 | 0.0127 | 持平 |
| Δsim(emo−plain) | −0.0140 | −0.0127 | **−0.0129** | **保住** |

- **假设被否**：冷启动 + 更少步数（1500 vs 2909）在同一份数据上，尾静音只从 1.02→0.78s（仍远超 base 0.18），
  而 CER / 跑飞 / speech_ratio / audio_sec 全面变差 ⇒ 「续训链累积是主因」不成立。
- **v2 上 id CER 同样越红线**（base 0.0230 / r23 0.0177 / r24 **0.0876**），但拆开看全部来自 2 条短 cue 跑飞
  （`id_nat_02` audio 1.92→6.67s、`id_nat_20` 1.76→2.93s，各 1/3 seed 多读）；**剔最差 3 条后 id ΔCER −0.0030**、
  32 条里只 10 条变差 ⇒ 不是普遍性内容退化。逐 case 尾静音 106/118 比 base 高（均值 +0.111s）。
- **id 探针上的反向信号**：r24 mean CER **0.0042**、多读 0/54、SIM 0.8734，优于 r23 的 0.0672 / 1/54 / 0.8686
  （r23 有一条 `id_ms_colloquial_en` seed43 跑到 **20.8s**、CER 3.16）。⇒ **冷启动不是全面更差，而是把跑飞
  挪到了不同 case 上**（r23 炸在 `id_ms_colloquial_en`，r24 炸在 `id_nat_02/06/20/24`，后者 base 音频只有 1.6–2.8s）。
  历代 id 探针：base 0.059 / r8 0.022 / r13x 0.018 / r17 0.114 / r18 0.001 / r23 0.067 / r24 0.0042。
- **音色一致性没有变好**（用户最在意的维度）：同 ref 跨句离散度 id base 0.0154 / r23 **0.0132** / r24 0.0153，
  th 0.0158/0.0186/0.0183，tl 0.0102/0.0104/**0.0093** ⇒ r24 丢掉 r23 在 id 上的优势，**没有理由换**。
- **顺带学到的一条**：Δsim 在冷启动下保留（−0.0129 ≈ r23 的 −0.0127）⇒ **情绪前缀带来的音色稳定收益来自数据，
  不来自续训链**；而续训链在这份数据上是「内容更准 + 尾巴更长」的权衡，不是单纯过拟合。

**⇒ 按开训前写死的规则：本轮不通过、不晋级、生产保持 r23、HF 不动、不 merge 不推送。**
并且**不再为「修尾静音」开训练轮**——三次独立证据都指向生成侧而不是数据尾：r4（数据尾压到 0.15s，模型仍垫
~0.25s）、2026-10-22（6271 条训练目标尾全 ≤0.17s，生成仍 +0.18~0.22s）、本轮（冷启动仍 +0.60s）。
要降尾静音/跑飞只能走**推理侧**（DIS 长度门、重试策略、参考锁定、可选 VAD 尾裁）。
再开训练轮的前置是**新的可证伪假设**，不是继续调时长卫生或步数。

## 2026-10-22：r24 之后的微调计划（阶段 A–D）

**状态：已写死判据，尚未授权执行。** 生产与 HF 保持 r23。排序依据是 r24 换来的那条区分：

> ⚠️ **执行前置：远端实例已于 2026-10-22 23:0x 停机**（ping 通、SSH 51348 refused）。阶段 A 要读远端清单、
> 阶段 C 要读远端两份 LoRA 权重，**实例恢复前都动不了**。r24 的 `lora_weights.safetensors`（145MB）只在远端，
> 恢复后先拉回本地，否则释放实例就等于放弃阶段 C。

| 现象 | 量级 | 现状 | 处置 |
|---|---|---|---|
| **垫尾**（生成尾部多垫静音） | base p90 0.18s → 各轮 0.28–0.50s | 数据侧三次证明修不动（r4 / 2026-10-22 / r24），线上 `trim_silence_vad=True` 会裁掉 | **接受，永不再开训练轮** |
| **跑飞**（1.2s 窗生成 3.5–20.8s） | dramaprod id 3–5/96、id 探针单条 20.8s | r24 证明不是步数/初始化能修的（冷启动只把它挪到别的 case），**数据侧尚未排除** | 阶段 A 审计 → 有靶子才进 B |

### 阶段 A：跑飞的数据侧归因审计（零 GPU，只读，最先做）

对比 **r18**（垫尾仅 +0.14s、id 探针 0.0013、冷启动 1390 步、id 份额 47.5%）与 **r23**
（垫尾 id p90 1.02s、跑飞 3/96、续训 2909 步）两份清单：

- 清单：`/root/autodl-tmp/voxft_data/processed/r18_idaccent/train.jsonl`（7412 行/15.0077h）
  与 `/root/autodl-tmp/voxft_data/processed/r23_thtlboost/train.jsonl`（7996 行/16.554h，**已核实存在**）。
  ⚠️ r18 那个路径取自它的运行记录，**停机前未复核文件是否仍在盘上**（`processed/` 可能已为腾磁盘被清理）；
  恢复后先 `ls` 确认。若已被清理，改用 **r13x** 作"干净对照"（同为冷启动、垫尾只 +0.14s、dramaprod 跑飞 0/96），
  或按 r18 的 mix 配方重建清单——**不要凭记忆断言它还在**。
  **只读**：不重写清单、不重加工音频、不删任何产物；审计脚本与输出放
  `/root/autodl-tmp/r25_data_audit_20261022/`，结论回本地 `work/` 与本文档。
- 四个维度，全部**逐源（source_id）拆分**，因为 r19 是一次多变量变更（加 thai_ser 511 行 + th 份额
  27.7→32.9% + id 稀释 37.4→34.7% + 启用 20s 上限），必须拆开看：
  1. 目标时长分布：>15s / >20s 的行数与语种（已知 r23 有 65 条 >20s，r18 侧未知）；
  2. **文本长度 vs 音频时长错配**：字符/秒（th 按字符、tl/id 按词，量纲不横比）异常低的行占比——
     这是"教模型不停"的头号候选，因为低速率行意味着长停顿或慢速，而停止头在 LoRA 下是冻结的，
     模型只能从 LM 隐状态里学到"这种上下文还不该停"；
  3. 情绪前缀行（r19 新增 254 条 TH 可信前缀 + r20 的 id/tl 中性前缀）的时长与尾静音分布，
     与同语种裸文本行对比；
  4. 哪一源是 r19+ 才引入或加量的（thai_ser / 增量 gigaspeech2 / cv22 / fleurs_tl）。
- **判据（开跑前写死，不许看完结果再改）**：
  - 有靶子 ⇒ 某类行在 r18 里占比 <1%、在 r23 里 >5%，且形态与跑飞 case 同类（长音频/低速率/带前缀），
    进阶段 B，且 B 只删这一类；
  - 无靶子 ⇒ 两份清单在上述维度差异都在噪声内 ⇒ **判定"跑飞不可由数据修"，停止所有微调轮次、
    全部转阶段 D**。这本身就是结论，写进 runs.md，不再烧 GPU。

### 阶段 C：r23 × r24 权重平均（零训练，可与 A 并行）

假设：两者在**不同 case** 上跑飞（r23 炸 `id_ms_colloquial_en` seed43 → 20.8s；r24 炸
`id_nat_02/06/20/24`，其 base 音频只 1.6–2.8s）⇒ 平均增量可能同时压住两边极端而不丢内容。

- ⚠️ **不能直接把 lora_A / lora_B 各自取平均**：ΔW = B·A·(α/r) 是双线性的，分量平均 ≠ 增量平均。
  正确做法是**秩拼接**：`A = [A23; A24]`（128×d）、`B = [0.5·B23, 0.5·B24]`（d×128），
  配 `lora_config.json` 的 `r=128 / alpha=128`（保持 α/r = 1，与 64/64 同刻度），
  这样 `B·A·(α/r) = 0.5(ΔW23 + ΔW24)` 精确成立、无需 SVD 截断。
- 权重来源：`/root/autodl-tmp/voxft_ckpt/lora_omni5_r23_thtlboost_cont/latest/lora_weights.safetensors`
  与 `/root/autodl-tmp/voxft_ckpt/lora_omni5_r24_coldstart/latest/lora_weights.safetensors`
  （两者均 144,749,240B、r64/α64，形状可直接拼）。原权重只读，产物写新目录
  `/root/autodl-tmp/voxft_ckpt/lora_soup_r23r24/`。
- **评测前必须先过加载断言**：`load_lora` 返回 loaded 非空、skipped/missing 全空（AGENTS.md 铁律），
  并按秩拼接后的 r=128 校验形状；断言不过就停，不许"看起来能跑"就评。
- 评测与判据：dramaprod 112×3 + v2 118×3（≈690 输出，与 r24 同口径，base/r23/r24 都不重跑）。
  **通过** = id 跑飞 ≤3/96 且 id CER ≤ r23 + 0.04（噪声）且 id p90 尾静音 ≤ r23 的 1.02s；否则丢弃、记录阴性。
- ⚠️ 这不是官方 FAQ 的做法，属探索性；**不因为它便宜就降低判据**，正负结论都写 runs.md。

### 阶段 B：r25 单变量清洗轮（**仅当 A 找到靶子**，需另授权）

- `r25 = r23 数据 − A 指认的那一类行`，其余逐行不动；用 `scripts/incremental_mix.py` 审计继承与删除，
  保留真实 `source_id`，**禁止删旧清单重建来掩盖**。
- **从 r23 权重小步续训（约 400–600 步），不冷启动**：r24 已证这份数据上冷启动会把内容与跑飞一起搞坏
  （id CER 0.0752→0.1298、th 0.0773→0.1183），而我们要的是"保住 r23 的内容、只去掉跑飞"。
- 超参其余全同 r23（LoRA r64/α64/dropout0.05、lr 1e-4 接余弦尾、batch2×8、proj 关）。
- 门禁同 r24：id 跑飞 **0/96**、id p90 尾静音 ≤0.5s、三语种 CER 不劣于 r23 超噪声（±0.04）、
  Δsim(emo−plain) 不劣于 −0.013、id 探针无 >10s 输出。**任一语种退化即整轮不通过。**
- 预算：训练 ≤600 步（约 20 分钟）+ 评测 ≈690 输出；磁盘先 `df -h /root/autodl-tmp`（当前剩 13G）。

### 阶段 D：推理侧（不占微调轮次，但这是今天 #8/#9 的真修法）

- **同 speaker 锁定 canonical 参考**：从该 speaker 的参考池里选一条（时长最长 / 有声占比最高 /
  与池内中位 cosine 最高），`identity_locked=true`、全 cue 共用，不再逐 cue 取当前窗；
  **identity 重试不换参考**（今天 #9 就是换到 cue24 的 alternate 后偏到 140.4Hz）。
- **参考准入门**：duration<2s、`active_speech_ratio`<0.5、SNR<3dB、gender unknown 的参考不得作唯一锚点
  （#8 的参考三条全中：1.90s / 0.316 / 1.45dB / unknown，且头部 0.25s 被判「female against body male」）。
- **极短 cue（<1.5s 窗）跑飞后不 keep-best 出片**：#9 是 `tts_failed` → 源人声抑制 → keep-best
  （score −269.1）出片，DIS 自己标了 `unresolved_artifact`；这类应显式失败或走既有的失败分支。
- 前置：先核 `review_4f50387a` 聚类是否混入多个真人（10 条 cue 参考 F0 跨 142.9–333.3Hz、
  4 条判 female 而 `declared_gender=male`、`anchor_similarity` min 0.1354、`speaker_status` 全 unverified）。
  **聚类不干净时，任何"同人一致性"策略都跑在错的前提上。**

### 明确不做（避免白跑轮次）

- 不再为压 0.2–0.3s 的垫尾开训练轮（三次独立证据 + 线上会裁掉，收益为零）。
- 不做步数 / lr / r / 强度网格：r24 已把"步数"这个变量的方向测出来了——**内容与尾巴同向变化、没有甜点**
  （2909 步续训 = 内容更准 + 尾巴更长；1500 步冷启动 = 尾巴略短 + 内容更差）。
- 不为 1–3s 短 cue 开训练轮（`process_dataset` 的 3s 硬下限，结构上学不到）。
- 不动 HF 与生产（保持 r23）；不 merge、不推送权重，晋级另需用户确认。

### 用户 2026-10-20 两条目标的现状

- ①「加强对短 cue 的质量支持」⇒ **微调侧已三次证明无解**（round 1 长度出分布、r24 冷启动只挪位置、
  3s 硬下限），转阶段 D 由推理侧兜。
- ②「加情绪时音色别偏太多」⇒ **已达成且可复现**：Δsim(emo−plain) base −0.0140 / r23 −0.0127 / r24 −0.0129，
  且 r24 证明这份收益来自**数据里的情绪前缀**、不来自续训链 ⇒ 它是资产，后续任何轮次都必须保留可信前缀行。

### 阶段 A 结果（2026-10-23 执行完毕：**三个靶子全部被数据否掉，阶段 B 不启动**）

产物：远端 `/root/autodl-tmp/r25_data_audit_20261023/audit.log`，本机
`work/r25_data_audit_20261023/`（审计脚本 + 日志 + speaker 聚类核查脚本与输出）。全程只读、零 GPU。
两份清单都在盘上（**r18 = 7412 行 / 15.008h 已核实存在**，r23 = 7996 行 / 16.554h）。

1. **长目标（>15s / >20s）不是靶子**：r23 反而比 r18 少——id 74→46 行、th 17→15 行、**>20s 在 id/th 清零**；
   r23 剩下的 65 条 >20s **全在 `tl/fleurs_tl_r23boost`**，而 tl 恰恰是唯一没有跑飞的语种
   （tl 跑飞 0/96、p90 尾静音 0.28s）。⇒ 原计划里"剔 >15s 目标"这个动作**没有依据，删掉**。
2. **文本-时长错配（低 rate）不是靶子**：三语种的 rate 分位数在两份清单里几乎逐位相同
   （id p5 1.48/1.49、p10 1.64/1.67、p50 2.19/2.23；th p5 8.49/8.47、p10 9.31/9.34；tl 同）。
   "长音频 + 低 rate"错配行：id **1.44%→0.78%（反而变少）**、th 0.13%→0.14%、tl 1.04%→1.60%。
3. **⚠️ 我自己提出的"前缀行更长"假设也是错的**：先前从"前缀行 p90 14.89s vs id 全体 9.43s"推出的偏斜
   是**把 id 与 tl 混算造成的假象**（tl 本来就长）。分语种看：id 前缀行 p50 6.21 / p90 9.88 / 均值 6.77，
   裸文本 6.29 / 9.39 / 6.58；tl 前缀 12.03 / 16.83 / 12.02，裸文本 11.58 / 18.15 / 12.07
   ⇒ **分布几乎一致，无偏斜**。
4. **唯一"从无到有"的变化 = 373 条中性前缀行**（r18 **0 条** → r23 373 条 / 4.66%，其中 id 246 条 = id 的 7.6%、
   tl 127 条）。前缀只有 4 种、均布、**全是中性英文散文**：`(neutral, natural tone)` 94、`(calm and steady)` 94、
   `(speak naturally and clearly)` 93、`(relaxed and natural)` 92 ⇒ 与生产 voice-design 端点的形态一致，
   **不声称情绪**，这也解释了为什么全清单 `emotion_verified=true` 为 0 行。
   ⚠️ 口径要说清：铁律要求"只用 `emotion_verified`/`control_verified` 的可信标签"，中性 delivery 描述
   不主张表演、故不走情绪核验，但**这个解释必须写下来**，否则会被当成伪造标签。
   削弱它是靶子的证据：r24 的跑飞同时出现在 `_emo` 与 `_plain` 条件（`id_nat_24_nref_emo` 与
   `id_nat_24_nref_plain` 都炸），不是只在带前缀一侧。
5. **附带发现（与音色一致性直接相关）**：带 ref 的行 r18 1448/7412（19.5%）→ r23 1238/7996（15.5%），
   跨语种 ref 两轮都是 **0**，而 **tl 带 ref 两轮都是 0/990 与 0/1535** ⇒ 线上唯一在用的
   `[103 ref 104][text][101 target 102]` 打包路径**在 tl 上从未进入训练**。

**⇒ 按开跑前写死的判据：判定「跑飞不可由数据修」，阶段 B 不启动，停止为它开训练轮，转阶段 D。**
剩下唯一的数据侧单变量是"r23 − 373 条前缀行"，但 r24 已证**情绪下音色稳定的收益来自这些前缀行**
（Δsim −0.0129 在冷启动下保留）⇒ 那是拿一个已确认的收益去换一个机制不明的改善，**不做**。

### 阶段 C 的期望价值已下调（零训练，仍未执行）

r23×r24 权重平均的判据是 id CER ≤ r23+0.04 = 0.115，而 r24 的 id CER 是 0.1298、r23 是 0.0752
⇒ 线性插值预期约 0.10，**刚好卡在门内**，余量极小；跑飞是离散事件、平均可能同时压住两边极端，
但内容退化更可能按权重比例带进来。成本仍然很低（零训练 + ≈690 输出），要不要跑由用户定；
**不因它便宜就放宽判据**，不过就是阴性结论并归档。

### 阶段 D 前置已完成：`review_4f50387a` 聚类确实混了多个真人（2026-10-23）

用已提取到本地的 10 条参考做两两 WavLM cosine（45 对，零新增 TTS/ASR）：

- **min 0.4672 / 中位 0.7916 / max 0.9841，5 对低于 0.60**。结构很清楚：
  `cue9↔cue24` **0.9841**、`cue2↔cue10` 0.9543、`cue10↔cue13` 0.9451、`cue2/10↔cue17` 0.9324 是一个紧簇；
  而 `cue18↔cue20` **0.4672**、`cue20↔cue24` 0.4934、`cue13↔cue18` 0.5213、`cue9↔cue20` 0.5395、
  `cue8↔cue20` 0.5866 明显是另一批人。**cue8（今天 #8 用的那条）与 cue9 只有 0.6824。**
- 与 DIS 自己的读数同向：F0 跨 142.9–333.3Hz、4 条参考判 female（conf 0.69–0.885）而请求
  `declared_gender=male`、`anchor_similarity` min 0.1354、`pitch_dispersion` over_threshold。
- 对照：同集其它 speaker 的参考池是自洽的——`review_1ba2acde` 两条参考 cosine 0.9391，
  `review_07ea0705`（7 cue）与 `review_82dd7e8a`（3 cue）**各自只用一条参考字节**，不存在池内混人。
  ⇒ `review_4f50387a` 是这一集里唯一参考池内部就不自洽的 speaker。
- **结论**：用户"#8/#9 是同一个人"的前提**在参考层面不成立**（至少 cue8/cue18/cue20 的参考与 cue9/24 不像同人）。
  因此 #8/#9 的音色差异**首先是身份链问题，其次才是模型问题**；阶段 D 的"同 speaker 锁定 canonical 参考"
  必须**先修聚类**，否则锁定的是错的池子。
- 边界（不要过度解读）：1.58–4.84s、SNR 0.54–2.34dB 的短噪参考上 WavLM x-vector 是**弱证据**
  （AGENTS.md：短参考无法做可靠声纹校验）；我本地 `librosa.yin` 的 F0 与 DIS 的在两条低有声占比参考上
  差 50–150Hz（cue2 205.6 vs 156.9、cue6 296.8 vs 148.6，yin 在低浊音占比下会八度跳），F0 只当线索。
  但 0.98 与 0.47 这种量级差不是噪声能解释的。要坐实需人工听或更长更干净的参考。

## 2026-10-22：暂停交接与续做计划

**用户决定先记录、后续再做；此前“继续”授权暂停。** 本次只更新文档，不启动新试听包、TTS/ASR、
训练或大评测，不更换生产r8。下文的启动命令与时间线是历史记录，不是待执行队列。
强度对照已于北京时间2026-10-22 19:03:41完成；0.5不推进，0.75只保留局部诊断，两档均不晋级。
完整结果及哈希见[runs.md](runs.md)，当前待办见[TODO.md](../TODO.md)。

### 1. 恢复入口与证据保全

- 用户恢复后先公钥重连：`ssh -p 51348 -o BatchMode=yes -o ConnectTimeout=10 root@connect.bjb1.seetacloud.com`。
  若端口拒绝连接，向用户确认实例状态/新端口，不反复重试、不自行重启；不在命令或文档写密码。
- 只读核验`/root/autodl-tmp/strength_ab_20261022_v1/status.json`为`complete`，
  `plan.json`、四份报告、实际ref与WAV哈希仍匹配。计划SHA为
  `0c970e1dd10301a68eaf1f338bd1874c7539b721ebf125884ea9f321354eb803`，
  汇总`comparison.json` SHA为
  `d4d8c289652bdf3d0ca07a16bc93a442c559467bce1cd0a19029b2aa42790721`；
  其它报告SHA见运行记录。校验失败先查差异，**不重新生成来覆盖证据**。
- 原始大评测在`/root/autodl-tmp/control_ab_20261022_v4/`，旧包和反馈在
  `/root/autodl-tmp/review_20261022/acoustic_review_v1/`及其`feedback_review/`；
  候选/回归归档在`/root/autodl-tmp/work/for_train/ab_candidates/acoustic-5b297d423cc5/`。
  本地对应元数据在`work/strength_ab_20261022_v1/`、`work/review_20261022/acoustic_review_v1/`；
  这些目录**不含完整WAV/模型备份**，不能据此释放实例或删盘。
- 已完成的`prepare/run/collect`不重跑，不删除`run.started`或claim，不覆盖冻结执行器、旧报告和音频。
  本次不关机或管理旧试听服务/隧道；其当前在线状态未重查，历史GPU空闲不等于实例已停机。

### 2. 第一项：仅复用音频的6组强度盲听（尚未制作）

比较**同一个on权重的1.0与0.75强度**，各取下表同case/同seed的原始输出，共6组/12条。
来源为强度目录的`on_report.json`和`s0.75_report.json`；模型基座不是生产r8，不能写成r8 A/B。

| 用途 / 旧包代号 | case_id | seed | 语种 |
|---|---|---:|---|
| D01-B，尾空白 | `fresh_id_1786_plain` | 46 | id |
| D02-A，声音过小 | `regression_id_ms_colloquial_en_control` | 42 | id |
| D05-A，头空白 | `fresh_tl_1728_plain` | 45 | tl |
| D06-A，声音稍小且尚未改善 | `regression_sc_tl_1747_zh_plain` | 42 | tl |
| TH随机层对照 | `fresh_th_1806_plain` | 42 | th |
| TL随机层对照 | `fresh_tl_1781_plain` | 46 | tl |

TH/TL对照按原冻结`plan["selection"]`顺序，各取该语种首个random条目并保留原seed，不按结果排名。
前四组刻意复查已知异常；整包是开发诊断，不是新独立验收集，不能用其异常比例推算线上坏例率。
旧代号只用于追溯，不预设新包A/B顺序；case名称中的`ms`只是历史命名，实际目标仍为ID。

续做时的最小实现与验收：

1. 现有`scripts/control_ab_review.py`的`prepare/load_sources`固定原实验哈希和base/on来源，
   **不能直接把强度目录传入就当作1.0/0.75导出器**。补最小独立适配，复用现有页面、反馈、归档逻辑；
   不覆盖远端`acoustic_review_code_v3`或强度计划引用的冻结代码。
2. 补定向测试：精确6组且两侧case/seed/ref/生成条件一致、只差强度，SHA不符及覆盖旧产物均拒绝；
   验证公开页面不暴露模型映射、私有ref或自动分数，未知/未评反馈不转为通过。
3. 另建唯一包目录和包号，随机化A/B；复制完整原WAV字节并复验SHA，不裁剪、重采样、响度归一，
   不触发新TTS/ASR。私有记录保留实际ref/prompt是否存在、原字节/SHA、文本、有效参数、QC与旧反馈。
4. 现有`serve()`固定监听远端127.0.0.1:16006；该端口旧包是base/on 1.0，不是强度新包。
   上线试听前检查服务/隧道与端口冲突，只公开新包`public/`；不静默顶替旧服务。
   **目前没有新适配器、新包号或新试听URL，不能宣称已可打开。**

### 3. 收到反馈后，再决定是否值得开下一轮

- 用户只判断空白、声音过小、破音/杂音/明显变声等声学问题；未知或未评保留未知，
  不要求其判断三语种发音、口音或自然度。反馈写新副本，正常合成候选也仍为
  `development_only/training_eligible=false`，声学正常不清除自动内容风险或认证`false_alarm`。
- 对照必须保留D06和TH/TL，不能只汇报改善例。D02在0.75的原seed上RMS提高12.08dB，
  但CER为0.0541→0.1622；D06低电平未改善。0.75随机组ID/TH/TL的p90尾静音仍比base多
  0.22/0.18/0.18s。RMS不是LUFS、能量边界不是词对齐，不能据此承诺听感/内容改善。
- 优先复查已有证据。未来备料前检查两条TL极低电平目标
  `fleurs_tl/audio/run_mj9e85fn/0000585.wav`与`0000936.wav`；不改写旧训练数据，
  不预判它们解释全部退化，也不删掉两条就自动重训。0.5疑似续写保留原WAV/转写，不重抽挑优。
- 如果局部听感无稳定收益，停止推进0.75；即使听感改善，也只保留诊断结论，生产仍为r8。
  再开训练/大评测前须另冻可证伪的单变量假设、数据来源/隔离、case/ref/seed/参数、预算与停止条件，
  获得新授权后再执行；不自动追加epoch、改CFG/步数或扩大生成数。
- 这18条已用于选参，不能重新命名为独立验证集。未来晋级仍需独立样本的ID/TH/TL分语种验收、
  相应声学与合格母语评审；base比较不能冒充已核验的生产r8比较。没有母语评审可继续授权的离线诊断，
  但不能宣布完整语言质量通过。

## 2026-10-22：先修验收独立性，再做控制前缀单变量

复核范围：10 月 20–21 日提交、远端 r19/r22/r23 清单、既有 refrob 报告。
最初复核阶段没有训练、TTS/ASR、语料下载或生产变更；后续执行授权与实际状态见下节。
新工具与产物独立放在大盘，未覆盖远端已修改源码。

### 最新执行：小比例中英回放，中文为主的跨语言参考

**17:45结果更新**：base/r23/off/on各1000输出完成；off/on均已训练393步，GPU任务已退出。
on相对off的TH内容指标改善有配对证据，但ID/TL的CER区间仍跨零。
两臂均触发相对base的三语种p90尾静音增量门槛，**不晋级生产、不自动追加训练**；
当时优先对on做既有音频声学盲听、尾静音与历史风险归因；后续反馈及强度对照见下。
完整数字/哈希/限制见`runs.md`顶部。
下文12:05/14:14为调度发生时的快照，不代表当前仍待开训或仍在评测。

**18点后续诊断**：已用既有WAV完成18组base/on声学盲听包（36条/297.76秒），
12组元数据随机对照与6组风险样本分开；全部训练目标已做CPU尾静音复核。
6271条可测训练目标没有>0.17s长尾，另2条TL极低电平边界未知，不能当通过。
on的生成尾静音增量跨plain/control、中英ref和5seed存在；尚无根因或听感改善结论。
没有新合成/转写/训练，也不改变“不晋级”结论。详情见`runs.md`顶部。

#### 本轮盲听入口与反馈

- 以下为已收齐反馈的旧base/on 1.0包，当前服务/隧道状态未重查；不是计划中的1.0/0.75新包。
- 本机浏览器打开`http://127.0.0.1:16006/`，先听R组，有余力再听D组；全部原音频不裁尾。
  只填声学异常及时间点，不懂语言可选“不确定”。浏览器暂存不是提交，需导出文本发回。
- 包号`acoustic-5b297d423cc5`；远端产物根
  `/root/autodl-tmp/review_20261022/acoustic_review_v1/`。只有`public/`可被HTTP访问，
  `mapping.json`、`conditioning.jsonl`、参考音和私有报告不得公开。
- 构建器冻结在`review_20261022/acoustic_review_code_v1/`，带HTTP Range的只读服务/反馈入口
  在`acoustic_review_code_v2/control_ab_review.py`；没有修改旧`execution_code_v4`。
  服务复用已安装的Starlette/Uvicorn，只监听远端127.0.0.1:16006；不再用不支持Range的简单HTTP服务器。
- 断线后只重连SSH隧道，不重新生成包，不改变A/B映射：
  ```bash
  ssh -fN -p 51348 -o BatchMode=yes -o ConnectTimeout=10 \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 127.0.0.1:16006:127.0.0.1:16006 root@connect.bjb1.seetacloud.com
  ```
- 远端服务退出时，先确认端口空闲，再按nohup约定启动；日志在产物根旁：
  ```bash
  cd /root/voxcpm-finetune
  export PYTHONPATH=/root/autodl-tmp/review_20261022/execution_code_v4:/root/voxcpm-finetune/src
  export CUDA_VISIBLE_DEVICES=
  nohup stdbuf -oL -eL .venv/bin/python -X utf8 -u \
    /root/autodl-tmp/review_20261022/acoustic_review_code_v2/control_ab_review.py \
    /root/autodl-tmp/review_20261022/acoustic_review_v1 --serve \
    > /root/autodl-tmp/review_20261022/acoustic_review_range_http.log 2>&1 < /dev/null &
  ```
- 导入真人反馈用同一入口`<包目录> --feedback <反馈txt>`（继承上面的PYTHONPATH）：
  校验包号、试听代号、全部冻结产物和WAV哈希，拒绝重复或错包；写入新的`feedback_review/`，
  不覆盖原报告，三种抽样层分别汇总。自动化浏览器测试文件不得导入为真人反馈。
  已导入目录拒绝覆盖，后续修订反馈需另建可追溯副本。
- **本包36条反馈已完成导入**：随机组两侧均无明显声学异常；风险组4条异常全在on。
  只覆盖本包声学观察，母语与内容未验；on仍不晋级，后续强度对照已获授权但不自动晋级。
- 导入后通过独立v3入口`<包目录> --archive <私有归档目录>`收录候选/回归。
  本轮归档根`/root/autodl-tmp/work/for_train/ab_candidates/acoustic-5b297d423cc5/`；
  32条正常候选与4条回归分文件，实际ref原字节/参数/原始QC/用户原话一起留存，
  全部`development_only/training_eligible=false`；声学正常不清除内容风险。
  回归登记到`eval_cases/acoustic_feedback_20261022.jsonl`，不要反向将合成候选放进训练清单。

#### 反馈后的强度对照（19:03已完成，两档均不晋级）

- 独立工作目录`/root/autodl-tmp/strength_ab_20261022_v1/`；原18组×seed42–46，
  只生成on强度0.5/0.75共180条，原base/1.0共180条直接复用。未训练、未换生产模型。
  **已于北京时间19:03:41完成**，四份报告各90条与全部WAV/输入哈希已核验，GPU进程退出。
  0.5出现TL双seed强内容风险；0.75缓解部分头尾空白/低电平，但D06未改善，
  三语种随机组p90尾静音仍高于base。只保留0.75局部诊断，不晋级；数字见`runs.md`顶部。
- 新入口`review_20261022/strength_code_v1/control_ab_strength.py`，支持`prepare/run/worker/collect`；
  PYTHONPATH依次为该源码目录、`acoustic_review_code_v3`、`fast_eval_code_v1`、
  `execution_code_v4`、`/root/voxcpm-finetune/src`。前三个依赖目录为既有冻结实现，不要覆盖。
  本轮`prepare/run/collect`均已完成，原以nohup/stdbuf和`python -X utf8 -u`执行，**不要重复启动**。
- 准备日志`/root/autodl-tmp/strength_ab_20261022_v1_prepare.log`，调度日志同名前缀`_run.log`；
  进度读工作目录的`status.json`、`worker*_progress.json`和`worker*.log`。
  启动暖机时`status.json`曾暂留prepared，须结合进程与worker日志；不据此另起重复任务。
- 每条结果在`items/`，排他claim在`claims/`，强度参数核验在`scaling*.json`；
  最终自动写`s0.5_report.json`、`s0.75_report.json`、`levels.json`与`comparison.json`。
  原base/on子集带`reused_without_rescoring=true`，原报告/WAV不改。
- 运行拒绝已有`run.started`，计划与输入必须保持哈希；失败保留逐项结果，不自动重抽，
  不删除claim伪装续跑。分层、分语种复核后才能决定是否需要少量新盲听，不能据小样本晋级。

用户已授权“新独立基线→控制前缀 A/B→同口径复评”，并明确主要合成 ID/TH/PH，
ref 大部分中文、少部分英文。**目标回放比例与参考语言分布分别设置**：

- 目标回放按有效时长 zh 4%/en 1%，两臂使用相同 seed42 下采样与原行序；
  不删除 ID/TH/TL 目标，不改变可信同人 ref，不顺带重配三语种份额。
  实际 **6273 行/12.6414h**，TH/ID/TL/ZH/EN **2536/2443/1021/219/54** 行，
  时长占比 **33.60%/35.46%/25.94%/4.00%/0.99%**（四舍五入）。
- 254 条可信 TH 前缀、1340 条 ref 保留。ID/TL 情绪标签和跨语言同人 ref 仍缺，
  **中文参考占多数不等于可以把任意中文音频配到 ID/TH/TL 训练目标上**。
- 独立集为三语种各 30 条新文本，每句 plain/control、seed42–46；其中参考 zh/en **80%/20%**，
  各用 3 条可追溯、未命中已审计训练 PCM/已知身份/session 的 FLEURS 验证录音。
  另有数字/混英/短句/严重续写共 10 条历史压力文本，同样配对，单独汇总；合计每模型 1000 输出。
- 旧参考文件有两条 PCM 命中候选训练血统：`ref_en_03.wav` 与 `ref_tl_02.wav`。
  新独立集不使用任何旧参考；历史压力回归保留原参考，不伪称独立验证。旧文件、分数不改。
- 基线为 **base/r23**，不是已完成 base/r8 生产回放：远端缺 r8 的完整历史训练清单。
  冻结实际 base 权重、r23 的 step2909 文件、执行源码、输入音频与报告 SHA。
  中文/英文参考身份是否真正未见、录音授权、近似重复、基座预训练文本暴露仍未验证。
- 预算每臂 **393 步（约 1 epoch）**，沿用下述 r64/alpha64/lr1e-4/batch2×8/DiT 开/proj 关；
  两臂同基座，全新优化器和调度器，官方 Accelerator seed42 重复初始化抽样一致，
  不宣称逐位训练确定性，不自动续训或生产晋级。原 3 epoch 备料版留档不回写。

CPU 入口 `scripts/control_ab_gate.py`，分阶段执行器 `scripts/control_ab_run.py baseline|train|evaluate`；
冻结工作目录 `/root/autodl-tmp/control_ab_20261022_v4/`，源码
`/root/autodl-tmp/review_20261022/execution_code_v4/`。生成/训练一律 nohup 单独日志，
失败停止、不自动重抽，不覆盖成功 WAV/报告。训练前拒绝已有 checkpoint；共享 preflight
会创建空输出目录，执行器只移除自己刚创建的空目录，有意外内容即停止，防止误判 resume 或删数据。
完整性检查通过后才进入下一阶段，指标按语种与独立文本聚类；**运行状态以 TODO 与远端 status.json 为准**。
工程与语言验收分开，母语质量/声学盲听未完成，不能据此宣称实际配音质量提升。

**12:05 调度变更（用户要求 r23 提前并行）**：使用独立入口 `scripts/control_ab_overlap.py`
接管正在运行的两个 base 分片，不修改 `execution_code_v4` 或重跑其音频。
首先只加一个 r23 分片，总并发最多3；另一个分片等槽位与至少16GiB空闲显存。
接管前核 `/proc` 启动时刻、父子关系、模型/分片参数并暂停旧调度器，防止原队列重复启动；
成功后回收旧调度器，新队列继续原训练/复评，失败则停，不隐式重抽。
调度源码在 `review_20261022/overlap_code_v1/`，指纹和原进程身份另存
`control_ab_20261022_v4/overlap_schedule.json`；日志以 `overlap_pipeline.log` 为准。
当时双worker占约19.1GiB、仍有约28.3GiB可用，但GPU已100%忙，**显存富余不等于已证明提速**；
并行运行的 `gen_sec` 不作隔离延迟基准，生成条件/评分口径/每模型1000输出预算保持不变。
评测文本已登记 `eval_cases/control_ab_20261022.jsonl`，后续备料须继续排除这些文本。

**14:14 调度加速接续（用户要求缩短评测等待）**：

- 原500输出大分片在结束前不持久化逐条结果，不能中断后直接分摊余量。
  保留最后一个r23分片继续跑；新入口 `scripts/control_ab_fast_handoff.py`
  只暂停外层shell，核启动时刻/父子关系及源码指纹，待基线成功退出、全量报告/WAV核验后回收旧shell，
  接原冻结训练，再运行 `scripts/control_ab_fast_eval.py run`。不会重复启动训练或修改冻结v4源码。
- 后续off/on共2000输出改为共享逐case/seed队列，排他认领，最多4worker；
  逐个完成首项暖机并检查至少12GiB空闲显存才加下一worker。优先保留当前模型，结束本臂后可帮助另一臂。
  Whisper保持驻留，参考音频及相同PCM声纹缓存，生成目标的声纹仍独立计算；不改模型/精度/ASR参数。
  每条立即保存完整报告和WAV SHA，最终要求所有case/seed恰好一次、条件/权重一致。
  失败停止、保留已完成项，**未实现自动恢复/失败重抽**，不要删除claim伪装续跑。
- 缓存等价性只在现存base WAV做CPU声学重算，不触发TTS/ASR、不回写旧分数。
  TH/TL/ID各一个主集WAV，每轮各重复两次，4轮按原始/缓存/缓存/原始交叉顺序测试：
  原始两轮 **3.703s/2.893s**，缓存两轮 **2.672s/2.437s**，所有声学字段完全相等且SIM非空。
  相配比较约 **28%/16%局部耗时下降**；小样本且非隔离负载，不外推整体提速。
- 新源码独立存 `review_20261022/fast_eval_code_v1/`，计划与指纹为
  `control_ab_20261022_v4/fast_handoff.json`、`fast_eval_profile.json`；
  接续日志 `fast_handoff.log`，运行后逐worker日志/逐项报告在 `fast_eval/`。
  加速器SHA `4462b82b94b12472e0e9db82ac1d31c4ed1b21a461ba92e465251904aca0d400`；
  接续器SHA `6eec01f3d0eaf9f4fe69ecac46979b0bd46c41f4b7d46d824d3e12afb04ce7cc`。
  五seed、20步、CFG1.8、large-v3、每模型1000输出保持不变；ASR原有随机回退不保证逐位确定性。
  本地完整测试 **220 passed（107.58s）**，涵盖缓存等价/任务排他、逐项保留、报告与WAV篡改拒绝、
  接续父子身份和失败停止；`git diff --check` 通过。测试不等于已完成GPU四并发验收。
  **当前只验证缓存数值/局部耗时和安全接续，动态评测GPU全程吞吐/峰值显存尚未实测**。

### 实际数据问题

| 清单 | train/val 行数 | 训练同评测文本行 | 训练同验证文本行 | 目标 >20s |
|---|---|---|---|---|
| r19_emotion | 6407/1367 | 0 | 40 | 0 |
| r22_idboost | 6696/1367 | 0 | 40 | 0 |
| r23_thtlboost | 7996/1367 | 29 | 49 | 65 |

文本先去已记录的控制前缀，再沿用评测 `_norm` 归一化比较；两种同文检查可能重叠，
r23 合计 **76 条**文本风险行（29+49−2），加时长风险共 **141 条**不同训练行。
29 条评测同文行均为 TL，覆盖 v2 的 **21/30** 条 TL 文本、drama 两套各 **7/8** 条独立
TL 文本（各展开 28 个 case）、refrob 的全部 6 条 TL 文本。**这是训练见过评测文本，
不是已证明评测参考音频/身份泄漏；但不能把 r23 的 TL 分数当独立泛化收益。**
该暴露在 r23 增量中出现，不倒推污染此前 r12 A/B 的结果。

本次未发现元数据路径/已验证 speaker/session 交叉或目标原音频 >3 次重复；
这不等于 PCM、真人来源、授权或身份已核验。r23 的 ref 覆盖是 ID **625/3247**、
TH **613/2833**、TL **0/1535**，整体 **1238/7996（15.5%）**，远低于 30–50% 目标。
原注释“增量没有 ref”不符合 CV22 实际清单；保留已有可信配对，不给身份未知行强配。
r23 ID/TL 的 246/127 条控制行都是中性接种，不能据此声称非中性表演数据已齐全。

入口 `scripts/incremental_mix.py` 支持只读审计；r22/r23 构建器共享同一门禁，
包含继承清单检查、增量候选筛选、原 source_id/非空元数据保留与实际占比统计。
**旧 DST 存在就拒绝；旧父清单有风险也拒绝**，不要通过删除历史目录来让它运行。

### 第一版备料（当时未授权，原产物保留）

- 假设：在同样 THAI-SER 音频上，仅添加可信英文控制前缀，会否改善受控输出且不损害 ID/TL。
  两臂都保留 THAI-SER，**只隔离前缀作用，不隔离数据源作用**。
- 入口 `scripts/prepare_control_ab.py`：r19 原清单仅去掉 40 条验证同文行，保留 **6367 条/12.8539h**；
  TH/ID/TL/ZH/EN 分别 **2536/2443/1021/244/123** 条。
  on 臂仅 254 条已有可信 TH 标签带前缀（约 4.0%，不凑比例）；off 去同一前缀。
  两臂 **1340 条 ref（21.0%）**，跨语言同人 ref 仍为 0；不新增 ID/TL 情绪标签。
- target/ref/来源/顺序完全相同；验证集复制原字节。继承 r20 的 r64/alpha64/dropout0.05、
  lr1e-4、batch2×累积8、warmup120、DiT 开/proj 关，以原始基座新训，不续 r23。
  配置暂按 3 epoch 上限计算各 **1194 步**；这是备料预算，不是已执行或已获增益证明。
- 远端目录 `review_20261022/control_ab/` 包含 `off/on.jsonl`、`off/on.yaml`、`validation.jsonl`、
  `plan.json`；状态为 `prepared_not_authorized_for_training`。实文件逐行核验只差前缀，
  共同参数相同，checkpoint 目录不存在。**训练 seed42 目前只是计划，执行器 RNG 尚未验证。**
- 开训前：冻结每语种 ≥30 条独立新文本，含生产控制开/关及数字/混英/短句/严重失败回归，
  固定相同 case/ref、cfg1.8/20步、禁内部重试、large-v3、seed42–46；
  审计全部候选训练血统，核 target/ref PCM、身份/session、授权证据、实际基座 SHA。
  新文本需要新 base/稳定对照，不能复用旧集基线。然后核相同 RNG 与无隐式 resume。
- 每语种分别判断内容风险和声学异常；SIM 只作相应目标的诊断，不抵消乱读/重复。
  母语自然度、口音、情绪均未验证，不要求当前用户填这些分数。

冻结 SHA256（完整输入哈希在 `plan.json` 与审计报告中）：

| 文件 | SHA256 |
|---|---|
| r19 train 原件 | `1303c59f84496aea74af56563eee2c814f1c300f647be330a764f524b610aec1` |
| validation 原件/副本 | `9d60874edde09368196341a80292dd33c7f65c0d3ee0df9cff7b460e8df690bf` |
| off.jsonl | `5c369e7ae048e034879c3c66e8fd001722d23178d93797bf18fe298ccd0dcba6` |
| on.jsonl | `a81b498c0140ce29b27c5ea25516284fa67fc8fed3f3b56ab2afc80673968aaa` |

不直接延续 r20→r23 的另一个原因：官方训练脚本恢复 scheduler 后会按**新的总步数**
计算余弦 LR。本地按现有代码复现，1162→1531、1531→2159、2159→2909 时，
旧终点的 0 可分别回升到约 **1.59e−5/2.16e−5/1.68e−5**；这不是连续低 LR 尾部。
远端逐步 LR 日志未核，不把本地复现冒充实际轨迹。r16/r17 同时改数据/预算，
也不能推导“1500 步以上必过拟合”或通用最优数据量。

> **2026-10-19 目标语种收缩**：用户决定不再微调越南语（vi / VN）与马来语（ms / MY），
> 目标语种只剩 **th / tl / id**，另加 zh / en 防遗忘回放。`TARGET_LANGS`、eval 语种白名单
> （`SAMPLE_TEXTS` / `WER_LANGS`）、registry 的 7 个 vi/ms 源、UI 文案与混合配比已全部同步收缩，
> 现存 22 个源。本文里 vi/ms 的实测数字与调研结论**保留原文只加存档注记**，
> 语料与接入门槛的依据分别见 [vi_id_support.md](vi_id_support.md)（vi 部分）与 [ms_support.md](ms_support.md)。
> ⚠️ 含 vi/ms 行的 case 文件已不可复跑（撞语种白名单直接 `ValueError`，这是刻意守卫）。

> **2026-09-17 执行范围更新**：用户目前没有 drama 素材，也不能做三语种母语评审。
> 当前三语种实验按 [TODO.md](../TODO.md) 执行：先定位明确声学坏例，测 r8 强度和推理参数，
> 再评估现有单人录音的不重叠 ref/target 切片。该配对支持尚未实现，不得绕过身份/原音频隔离守卫。
> 本文的成片导入、表演采集和母语验收流程作为条件具备后的指南，不是当前实验前置；
> `enable_proj=true` 不是提升音色的必经步骤。工程检查与完整语言验收的区别见 [qc_gates.md](qc_gates.md)。

---

## 目录

- [1. 预期边界：微调能修什么，不能修什么](#1-预期边界)
- [2. 三个决策点（先读，决定后面所有步骤）](#2-三个决策点)
- [Phase 0 · 环境准备](#phase-0--环境准备一次性)
- [Phase 1 · 先建评测集](#phase-1--先建评测集在训练之前)
- [Phase 2 · 数据准备](#phase-2--数据准备)
- [Phase 3 · 逐源加工与检查](#phase-3--逐源加工与检查)
- [Phase 4 · 混合与检查](#phase-4--混合与检查)
- [Phase 5 · 训练](#phase-5--训练)
- [Phase 6 · 验收](#phase-6--验收)
- [Phase 7 · 迭代调参：症状 → 动作](#phase-7--迭代调参症状--动作)
- [清晰度专项 · 全链路控制点](#清晰度专项--全链路控制点)
- [Phase 8 · 导出与上线](#phase-8--导出与上线)
- [附录 A · 坑速查](#附录-a--坑速查)
- [附录 B · 全流程 checklist](#附录-b--全流程-checklist)

---

## 1. 预期边界

先对齐预期，否则后面所有的调参都是在追一个拿不到的目标。

**微调能改善的：**

| 维度 | 能改善到什么程度 | 靠什么 |
|---|---|---|
| 目标语言发音 / 重音 / 声调 | 明显改善，尤其是泰语声调、Taglish 句内英文词的读法 | 目标语料占比 + 发音锚点源（`fleurs` / `porjai_th` / `filswitch`） |
| **清晰度 / 可懂度** | 明显改善，但**上限由数据入口决定** —— 没有降噪链路，糊的录音进去就是糊的模型出来 | 三条战线：① 只收干净录音 + 训练文本必须与音频一致 ② 首尾裁切不啃掉词首清辅音 ③ 推理 `inference_timesteps=20`。详见[清晰度专项](#清晰度专项--全链路控制点) |
| 自然对白韵律、去念稿感 | 改善，上限由**真人表演语料的数量和质量**决定 | `drama_*` / `thai_ser` impro 占比；表演源不能裁掉换气 |
| 情绪指令响应 | 改善，但只对**训练里出现过的可信标签**生效 | `(控制指令)正文` 前缀样本占比 25–50%，且标签 `*_verified=true` |
| 漏词 / 吞字 | 改善 | 训练文本准确 + 尾静音 <0.5s + 不拼接孤立词 |
| 音色漂移 | 改善（保住基座克隆能力） | LoRA 而非全量；ref 覆盖 30–50%；中英回放 15% |
| 生成停不下来 / 尾巴拖长 | 改善 | 尾静音裁切生效（朗读源必须走 VAD，见附录 A） |

**微调不能修的（不要指望，也不要为它调参）：**

- 翻译错误、输入台词本身缺词 —— 那是上游文本链路的问题
- 剪辑截断、时间轴对不上
- 角色选角不合适（ref 选错人）
- BGM / 音效混入 —— 本项目**不做声源分离和降噪**（无 demucs 依赖，官方 zipenhancer 需 modelscope 且不在 lock 里）。有对白轨就喂对白轨，没有就在素材导入的试听环节把 BGM 重的条目判「丢弃」
- 上游已经逐条峰值归一化抹掉的动态范围 —— 恢复不了

**指标的正确用法：** CER/WER、疑似漏尾、F0 全部只是**诊断信号**。通过与否由至少两名母语评审盲听决定。`f0_std_st` 含泰语词汇声调误差，不能当自然度用；`energy_range_db` 是能量分位差，**不是 SNR**。禁止用 `min_snr_db` / `min_f0_std` 硬筛（代码里启用会直接报错）。

---

## 2. 三个决策点

### 决策点 1 · 表演语料从哪来：现成的补不上情绪，只能自建那一档

**逐项核实结论见 [docs/corpus_sourcing.md](corpus_sourcing.md)**（含厂商询价模板和自建录制脚本模板）。这里只给结论。

已核实、不要重复调研的死路：Common Voice tl 官方 `recordedHours=0`，YODAS/YODAS2 Sidon 的 224 语种里没有 tl/fil，OpenSLR 无菲律宾语资源，SEACrowd 的 23 个 th/tl/fil 数据集全是文本无音频，HF 上 `modality:audio` 匹配 filipino/tagalog 的只有厂商 sample（`n<1K`，多为 CC-BY-NC-ND、gated 或**完全没有许可声明**）。`filipino_emotion` 连数据卡都没有，实测中位时长约 1.6s，绝大多数低于 3s 下限。**泰语同样没有公开的情感/表演语料** —— THAI-SER 之后没有新的，Nexdata/Datatang 的泰语 SKU 里也没有现成情感库。

**两个陷阱**（都很容易踩，因为看起来完全对口）：

- `laion/dramabox-voice-acting-data-annotated` —— CC-BY-4.0、10 万–100 万条、标签写着 `voice-acting`、数据卡还讲「同说话人跨情绪配对片段」。**但它是 TTS 合成的**（源头 `ResembleAI/Dramabox` + `gemini-2.5-pro-tts`，文件名带 `_seed{NN}`）。违反「不用模型合成语音补量」，情绪标签也是生成 prompt 不是真实表演标注。只有标注 schema 可参考
- MagicHub **`ASR-SFDuSC`** —— 4.58h / 10 人**朗读**，**CC-BY-NC-ND**（NC 禁商用、ND 禁演绎，微调就是演绎）。可用性 0。别和下面那个库搞混

**现成能买的（补的是「自然口语 + 说话人身份」，不是情绪）**：

- **MagicHub `ASR-BigFTagaCSC`** —— **1285h / 514 人**菲律宾语自发对话，16kHz WAV + TXT 转写，专有授权需询价。514 个**真实说话人身份**是 YouTube 抓取源给不了的（那些 speaker_id 是视频级近似身份，不能做 ref），能同时补自然口语锚点和 ref 配对身份。**但无情绪标签、不是表演**，对应配比表的「自然口语 30%」档，不是「真人短剧/表演 45%」档
- **Nexdata 1004 Hours Thai**（SKU 1687）—— 商业买断、低背景噪声、16kHz mono WAV、**带 speaker ID + gender**、WAR 98%。许可干净的泰语自然语音，是 `yodas_th` 的升级替代
- `speechcolab/gigaspeech2` th —— **Apache-2.0**（无 SA/NC 污染），但短句为主、无说话人身份，`gated: auto`

**结论：表演/情绪这一档只能定制采集或自建。** 最省力的做法不是继续找数据，而是**一次录制同时解决三件事**（详见 sourcing 文档 §5）：

1. 真人表演对白语料（配比表 45% 主力档）
2. **跨语言同人 ref** —— 让同一演员在同一场次里既录目标语言台词、又录 20–30 条中文和英文短句。每人多花 10 分钟，就拿到了决策点 2 里那个「拿不到就无法验证」的硬约束的唯一解
3. **可信情绪标签** —— 按情绪脚本录，`emotion_verified` / `control_verified` 当场就是真的

规模：每语言 6–8 名演员 × 5–8h ≈ 30–60h ≈ 1–2 万条 ≈ 单卡 1 epoch 600–1300 步，几小时一轮。**台词直接从产品实际要配的短剧剧本里选**，分布和线上推理一致，比通用情绪脚本有效得多。

首轮 TL 表演语料目标：**5–10h 干净真人对白**，覆盖多名女声、男声和不同年龄段，重点补质疑、克制愤怒、担心、讽刺、哭腔、带笑说话、自然停顿。**每种情绪必须由 ≥3 名不同演员录** —— 否则模型学成「这个情绪 = 这个人的音色」。

TH 好一些：`thai_ser` 的 impro（即兴对话）子集有 `actor_id` 和情绪标签，是开源里唯一带可靠身份的泰语表演语料。**但它是 CC-BY-SA-4.0**，触发交付红线，见 Phase 8。

### 决策点 2 · 跨语言同人 ref 是硬约束，不是可以凑的字段

生产场景的 ref 是中文/英文，目标是泰语/Tagalog。要让 LoRA 学到「按 ref 的音色说目标语言」，训练集里必须有**同一个人**的 ref 音频和目标音频配对。

**红线（`preflight` 会拦，也不许绕）：**

- 不许拿另一个人的英文声音配这个人的泰语目标 —— 教出来的是错误的音色映射
- 不许用 MFCC 聚类相似度当身份证据。聚类只能辅助审计
- 不许因为 ref 覆盖率不够就调低阈值强凑。身份未知 → 默认不配 ref
- 中英文回放（`aishell3` / `replay_en`）**不是**跨语言同人 ref 的替代品，它只防中英能力遗忘

**`speaker_verified=true` 的含义是「人工或可靠原始元数据确认了这是同一个人」**，不是「有一列叫 speaker」。同视频、相同角色名、声音听起来像，都不够。同一演员跨数据源时使用统一的 `speaker_namespace` 和 ID。

**性价比最高的一步（如果预算允许）：** 找 5–10 名目标语言演员，每人录 3–10s 的中文或英文短句若干条（不需要标准，需要是同一个人），作为 `reference_only=true` 行导入。这一小批双语录音是唯一能真正建立「中/英 ref → 目标语言」通路的训练信号，几十分钟录音的收益远大于再加 50 小时无 ref 语料。

拿不到时怎么办：**不要声称已完成跨语言 ref 验证。** 走 R1/R2 把语言和风格调好，ref 通路靠「同语言同人 ref」保住机制不被 LoRA 破坏，然后在 Phase 6 用 A/B 实测中英 ref 的音色保持有没有退化 —— 基座本身有跨语言零样本克隆能力，LoRA 的任务是别把它弄坏。

### 决策点 3 · 联合模型是目标，分语种验收是前提

目标是**一个 LoRA 同时提升 th / tl / id 三种语言的配音质量**，不是各训一个。基座侧没有任何语种槽位（无 language token / lang_id，manifest 无语种字段，官方对多语种混合训练零特殊处理），所以「联合」在训练侧就是普通的多源混合，不需要改 submodule，也不需要改超参。

联合的原风险是**归因困难**：各语言的语料量、情绪覆盖、ref 覆盖问题会混在一起。这个风险不靠「先分开训」规避，而靠下面四条硬性验收手段：

1. **训练前先拿分语种基线**：用同一份多语种 case 文件跑 `voxft.eval base --texts-file eval/joint_holdout.jsonl`，记下各语种的 CER/WER/疑似漏尾。没有基线就没有「提升」可言。
2. **任一语种退化即算失败**：联合 run 的报告看 `by_lang` 段，逐语种与基线比。三个语种里只要有一个变差这一轮就不通过——**不允许用「平均变好了」掩盖单语种退化**。
3. **保留一个分语种对照点**：至少给 TH 单独训一个 LoRA（它数据最全、最便宜）。有了它才能区分「联合不如单语」与「这个语种的数据本身不够」——两者处置完全不同（前者调配比，后者补数据）。
4. **明确回退条件**：某语种 CER/WER 或疑似漏尾相对基线退化，或母语盲听判差 → 该语种退回独立 LoRA，其余语种继续联合。回退是正常操作，不是失败。

**数据准备顺序：TH → TL → ID。** 这是**准备顺序不是训练顺序**，联合模型一次训全部。

- **TH 先行**：有 `thai_ser` impro 这个开源表演源，数据准备成本最低，用它把完整链路、评测集和盲听流程建立起来。
- **TL 第二**：流程照抄 TH，但表演语料只能自建（`drama_tl`），这一轮会把「素材导入 → 人工标注」这条链路跑通。
- **ID 第三**：基座基线三个目标语种里最好（内部 30 语种基准 WER 1.36%、MLS 1.084），正字法纯 ASCII 无 byte-fallback，`gigaspeech2_id` 是 Apache-2.0 无许可红线。但**表演档为 0**，这一轮只验发音与口语韵律。
- **VI / MS（2026-10-19 已退出目标语种，以下为存档依据，重启时直接用）**：VI 排第四的理由是 MLS WER 3.307 偏高，且是 6 声调语言，句尾嘎裂声有被首尾裁切啃掉的风险（见 [vi_id_support.md §4.1](vi_id_support.md)）。MS 排最后的理由：基座 WER 1.75%，是当年五个目标语种里**唯一输给竞品**的（Fish S2-Pro 1.41%，其余三个都是基座赢），所以提升空间最大、最值得投入；但 MLS 两张表都没有 ms（克隆基线无官方数字可引用），`gigaspeech2` 与 Common Voice 22 也都**没有 ms**，自然口语只能靠 `yodas2_ms`；且 ms/id 高度互通，**盲听必须由马来西亚母语者做，不能拿印尼语听感代替**。详见 [ms_support.md](ms_support.md)。

**ID 首轮的验收目标必须降级**：表演档为 0 时只能声称发音准确度、口语韵律、克隆能力不退化、指令跟随不劣化，**不能声称情绪表现力有改善或去念稿感达成**。这不是保守措辞，是数据决定的上限。（VI/MS 退出前同样适用这条降级。）

---

## Phase 0 · 环境准备（一次性）

### 0.1 拉代码与依赖

```bash
git clone --recurse-submodules <repo>
cd voxcpm_finetune
uv sync --group qc          # qc 组 = faster-whisper + PyAV，转写和素材导入都需要
# 已克隆但 submodule 为空：
git submodule update --init --recursive
uv run pytest               # 应该全绿
```

Python 3.11（`.python-version` 已固定）。torch 平台自动分流：macOS 走 PyPI 轮子（CPU/MPS），Linux 走 pytorch-cu124 index。**训练只在 Linux GPU 机执行，数据集下载也只在远程做**，本地不下载语料。

### 0.2 远程 `.env`

复制 `.env.example` 为 `.env`。远程 GPU 机的关键四项：

```dotenv
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx
WANDB_PROJECT=voxcpm-finetune
VOXCPM_BASE_PATH=/root/autodl-tmp/models/VoxCPM2

VOXFT_DATA_ROOT=/root/autodl-tmp/voxft_data
VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt
HF_HOME=/root/autodl-tmp/hf_home
HF_ENDPOINT=https://hf-mirror.com
```

**系统盘小，`VOXFT_DATA_ROOT` / `VOXFT_CKPT_ROOT` / `HF_HOME` 三个都必须指到大盘**（`/root/autodl-tmp/*`），否则几十 GB 语料会把系统盘写满。

镜像相关不用手动管：导入 `voxft` 时 `paths.load_dotenv()` 先加载 `.env`，然后 `_disable_xet_on_mirror()` 在 `HF_ENDPOINT` 非 huggingface.co 时自动 `setdefault("HF_HUB_DISABLE_XET","1")`。**但独立 `hf` / `huggingface-cli` 命令不读项目 `.env`**，要用就得自己 export 三个变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/root/autodl-tmp/hf_home
```

优先用 `python -m voxft.data.prefetch`，它把这些都处理好了。

### 0.3 预取权重

```bash
uv run python -m voxft.data.prefetch --whisper large-v3      # ~3GB，转写标签用
uv run python -m voxft.data.prefetch --repo openbmb/VoxCPM2  # 基座
```

large-v3 国内直连 huggingface.co 常在 SSL 握手就超时。加载已有 3 次重试并打印 endpoint。下载完可以把路径固定进 `.env`，避免以后走网络：

```bash
echo 'VOXFT_WHISPER_MODEL_LARGE=/root/autodl-tmp/hf_home/...' >> .env
```

### 0.4 启动工作台

```bash
nohup uv run voxft-ui > ui.out 2>&1 &      # 端口 6006
```

**必须用 `nohup` 或 tmux，别把 stdout 挂在会断的终端上。** 库函数一律走 `progress` 回调不 print，但如果 UI 进程的 stdout 是已断开的 pty，仍然可能出问题。

页面 6 个 Tab：数据集 / 素材导入 / 训练 / 试听 / 模型管理 / 日志。下面所有 CLI 命令都有对应的页面操作，用哪个都行 —— **但页面下拉框的 `choices` 只在 `build_ui` 时算一次**，任何运行后变化的列表（已加工数据集 / 配置 / LoRA / 上传目录）要靠事件输出或重新点 Tab 刷新。

---

## Phase 1 · 先建评测集（在训练之前）

顺序反了会导致「训完了才发现没法判断好坏」，然后只能凭感觉调参。

### 1.1 评测集要求

- **未见演员、未见会话**。绝对不能从训练 ref 里取
- 每个目标语言 **80–100 条**
- 逐条固定 `text` / `lang` / `ref_audio` / `ref_lang` / `control` / `seed`
- 存成 JSONL，进 git，多轮实验共用同一份 —— 换了评测集，前后轮的结论就不可比了

⚠️ **2026-10-19 的一次例外，以及它的代价边界**：v2 `omnivoice_prod_v2.jsonl` 因 vi/ms 退出目标语种从
**182 条过滤到 118 条**（纯删行，留下的行字节未变）。⇒ 旧报告里含 vi/ms 的**总体聚合**
（如 base 总体 CER 0.0774）**不再可复现**，别再拿总体数字跨这次过滤对照；但 `evaluate()` 是逐 case
独立计算、`by_lang` 按语种聚合，所以 **th/tl/id 的分语种数字仍然有效、仍可跨轮对照**。
归档态、已不可复跑的 vi/ms case 文件清单见 [eval_cases.md](eval_cases.md)。

### 1.2 覆盖矩阵

按这几个维度交叉，别只测一种组合：

| 维度 | 取值 |
|---|---|
| ref 语言 | 中文 / 英文 / 目标语言本身 |
| 控制前缀 | 无 / 基本情绪 / 复合指令（哭腔、带笑、克制愤怒） |
| 角色 | 女主 / 其他女声 / 男声 |
| 语体 | 普通口语 / 强情绪 / code-switch 混英（TL、ID）/ 长句 / 短句 |
| **清晰度压力** | 长句（>20s）/ 多音节难词 / 数字与英文混排 / 快语速（`语速快`）/ 耳语（`轻声`）/ 句尾辅音收尾 |
| 回归 | 中文回放 / 英文回放（检查有没有灾难性遗忘） |

清晰度压力用例要占评测集的 **20–30%**，并且 `kind` 标成 `clarity`，报告里单独分组看。这类 case 平时不会暴露问题，但一旦模型吐字变糊，它们最先崩 —— 等用户反馈才发现就晚了。泰语重点测声调对立的最小对（同音节不同声调）和句尾塞音；Taglish 重点测句内英文词、品牌名、数字读法（对应线上「RAW 被念成英文」类反馈）；印尼语重点测混英品牌名与数字读法。（**存档**：越南语当年重点测声调最小对，以及以 `ngã` / `nặng` 调音节收尾的句子 —— 这两个声调带嘎裂声、能量低，首尾裁切有啃掉声调尾巴的风险，见 [vi_id_support.md §4.1](vi_id_support.md)。vi 已退出目标语种，重启时才需要。）

**id 的数字与货币要先和母语者确认念法**：基座的文本归一化是 zh/en 二分（非中文一律走英语规则，开了会把数字念成英文），本项目保持 `normalize=False`，所以念法完全由模型从训练数据里学。清晰度用例里保留阿拉伯数字（`3`、`10:30`、`Rp 5 juta`）正是为了测这一点，但**此时 CER/WER 会因为不存在唯一正确写法而失真 —— 这类 case 只作母语盲听，不进自动指标结论**。日常台词若已有确定念法，直接写成拼写形式（`lima juta rupiah`）更稳。

保留用户反馈里的难词和漏尾句，**但先让母语者确认台词与预期读法** —— 翻译改写不属于训练能自动修复的范围，拿错台词当 ground truth 会把结论带偏。

### 1.3 JSONL 格式

相对 `ref_audio` 路径按 JSONL 所在目录解析。

```jsonl
{"case_id":"th_female_surprise_enref","text":"เธอทำแบบนี้ได้ยังไง","lang":"th","ref_audio":"refs/female_en.wav","ref_lang":"en","control":"surprised, in disbelief","speaker":"heldout_f01","kind":"emotion","seed":42}
{"case_id":"th_female_plain_zhref","text":"ฉันไม่คิดว่ามันจะจบแบบนี้","lang":"th","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"","speaker":"heldout_f01","kind":"plain","seed":42}
{"case_id":"th_male_angry_zhref","text":"หยุดพูดซะที ฉันฟังมามากพอแล้ว","lang":"th","ref_audio":"refs/male_zh.wav","ref_lang":"zh","control":"愤怒地，音量压低","speaker":"heldout_m01","kind":"emotion","seed":42}
{"case_id":"tl_female_taglish_enref","text":"Hindi mo alam na buntis ka?","lang":"tl","ref_audio":"refs/female_en.wav","ref_lang":"en","control":"surprised, restrained","speaker":"heldout_f02","kind":"emotion","seed":42}
{"case_id":"th_clarity_final_stop","text":"อยากบอกให้รู้ว่าไม่อยากไปแล้ว","lang":"th","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"","speaker":"heldout_f01","kind":"clarity","seed":42}
{"case_id":"tl_clarity_numbers_taglish","text":"May 3 appointments ako bukas ng 10:30 ng umaga sa BGC.","lang":"tl","ref_audio":"refs/male_en.wav","ref_lang":"en","control":"语速快","speaker":"heldout_m01","kind":"clarity","seed":42}
{"case_id":"id_female_angry_zhref","text":"Aku tidak menyangka kamu akan berbuat seperti ini.","lang":"id","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"愤怒地，音量压低","speaker":"heldout_f05","kind":"emotion","seed":42}
{"case_id":"id_clarity_numbers_mix","text":"Besok aku ada 3 rapat jam 10:30 pagi di SCBD.","lang":"id","ref_audio":"refs/male_en.wav","ref_lang":"en","control":"语速快","speaker":"heldout_m03","kind":"clarity","seed":42}
{"case_id":"zh_regression","text":"你到底想怎么样，我已经解释过很多遍了。","lang":"zh","ref_audio":"refs/female_zh.wav","ref_lang":"zh","control":"烦躁地","speaker":"heldout_f03","kind":"antiforget","seed":42}
{"case_id":"en_regression","text":"Are you okay? I was worried about you.","lang":"en","ref_audio":"refs/male_en.wav","ref_lang":"en","control":"","speaker":"heldout_m02","kind":"antiforget","seed":42}
```

`kind` 是自定义标签，报告里原样保留，方便分组看结果。`lang` 支持 `th` / `tl` / `id` / `zh` / `en`（**2026-10-19 起 vi/ms 已退出白名单**，含 vi/ms 行的 case 文件再跑会直接 `ValueError`，这是刻意守卫）。只有 Taglish 走 Whisper 自动语种检测，其余强制指定解码语种让 CER 可比。⚠️ **仅历史集适用**：vi 的 WER 是**音节级**错误率（越南语正字法按音节空格分隔），不与 id/tl/en 的词级 WER 横向比 —— vi 已退出 `WER_LANGS`，但这条量纲事实对任何按音节空格正字法的语种都成立，重启 vi 时要重新纳入。

放这里：

```bash
mkdir -p eval/refs            # 参考音频
$EDITOR eval/th_holdout.jsonl eval/tl_holdout.jsonl eval/id_holdout.jsonl
```

### 1.4 先跑一次基线（R0）

**基座必须永久保留为固定评测基线**，后面每一轮都跟它比。

```bash
uv run python -m voxft.eval base \
  --texts-file eval/th_holdout.jsonl --seeds 42 43 44
```

这会为每条 case × 每个 seed 生成音频，跑 large-v3 转写，算 CER/WER/疑似漏尾，写报告到 `$VOXFT_CKPT_ROOT/eval/`（文件名带 uuid，多次运行不覆盖）。

三个 seed 是为了区分「稳定的问题」和「采样噪声」。只跑一个 seed 就把某个 case 判成坏例，很可能是运气。

---

## Phase 2 · 数据准备

### 2.1 首轮配比（实验起点，不是已验证最优值）

按**过滤后的训练音频时长**计算，不是按条数。

| 数据角色 | TH | TL | ID |
|---|---:|---:|---:|
| 真人短剧 / 即兴表演 | 45%（`thai_ser` impro + `drama_th`） | 45%（`drama_tl`） | **0%**（待自建 `drama_id`） |
| 自然口语 | 35%（审核后的 `yodas_th`） | 30%（自有自然 TL/Taglish） | 75%（`gigaspeech2_id`） |
| 发音补充 | 5%（`fleurs_th` / `porjai_th`） | 10%（`filswitch` / `fleurs_tl`） | 10%（`fleurs_id` / `cv22_id`） |
| 中文回放 | 10%（`aishell3`） | 10%（`aishell3`） | 10%（`aishell3`） |
| 英文回放 | 5%（`replay_en`） | 5%（`replay_en`） | 5%（`replay_en`） |

**ID 表演档为 0 是数据现状，不是待填的空格**（本轮未核实到任何可商用的开源真人情感语料，不用 TTS 合成补量）。因此该语种首轮验收只能声称发音/口语韵律/克隆不退化，**不能声称情绪表现力改善**；`gigaspeech2_id` 还需先试跑确认字段形态可用。详见 [vi_id_support.md](vi_id_support.md)（id 部分现行、vi 部分存档）与 [ms_support.md](ms_support.md)（存档）。

> **存档**：原 VI / MS 两列的角色比例是 VI 表演 **0%** / 口语 75%（`gigaspeech2_vi`）/ 发音 10%（`fleurs_vi` + `cv22_vi`），MS 表演 **0%** / 口语 76%（`yodas2_ms`）/ 发音 9%（`fleurs_ms`）。两语种已于 2026-10-19 退出目标语种，对应的 7 个源（`drama_vi`、`drama_ms`、`gigaspeech2_vi`、`fleurs_vi`、`cv22_vi`、`yodas2_ms`、`fleurs_ms`）已从 registry 删除。

### 2.1.1 三语种联合首轮配比

联合训练时中文/英文回放是**全局共享的 15%**，不按语种各配一份。所以三个目标语种各拿 **(100 − 15) / 3 ≈ 28.3 分**，语种内部再按上表的角色比例拆（上表的百分比是「占该语种 85% 目标部分」的比例）。原五语种口径是 `(100 − 15) / 5 = 17 分`，收缩后每语种拿到近 1.7 倍的目标语料配额。

| 语种 | 表演 | 自然口语 | 发音补充 | 合计 |
|---|---:|---:|---:|---:|
| TH | `thai_ser`=10 + `drama_th`=5 | `yodas_th`=11.7 | `fleurs_th`=1.6 | 28.3 |
| TL | `drama_tl`=15 | —（无自有自然语料时并入下方） | `filswitch`=8.3 + `fleurs_tl`=5 | 28.3 |
| ID | **0** | `gigaspeech2_id`=25 | `fleurs_id`=1.7 + `cv22_id`=1.6 | 28.3 |
| 回放 | — | — | — | `aishell3`=10 + `replay_en`=5 |

```bash
# 只在远程执行；名称均需先完成加工。这就是「混合」页 recipe 框可以直接粘的配方。
uv run python -m voxft.data.pipeline --out joint_v1 --mix \
  thai_ser_v1=10 drama_th_v1=5 yodas_th_v1=11.7 fleurs_th_v1=1.6 \
  drama_tl_v1=15 filswitch_v1=8.3 fleurs_tl_v1=5 \
  gigaspeech2_id_v1=25 fleurs_id_v1=1.7 cv22_id_v1=1.6 \
  aishell3_v2=10 replay_en_v2=5
```

三条配套纪律：

1. **等权而非按数据量分配。** th 有 200h+（`thai_ser` 41h + `yodas_th` 156h），id/tl 首轮可能只有十几小时可用。按可用量配权会让联合模型退化成「泰语模型 + 两个配菜」。某语种吃不满 28.3 分时，`mix.json` 的 `language_shares` 会把「请求 vs 实际」并列出来并告警——**缺口不重分配给其他语种**（那正是大语种吃掉小语种的机制），也**不靠 3× 重复强凑**（重复曝光会过拟合）。要么补该语种数据，要么在结论里写明实际占比。
2. **首轮总时长控制在 40–60h**（约 1.8–2.7 万条 → 单卡 batch2×accum8 下 1 epoch ≈ 1100–1700 步）。联合清单是单语种的 3 倍，先按首轮实验规模跑通再扩量；`epochs ≤ 3` 的硬上限与「3× 重复 × 3 epoch ≈ 9 次曝光」的警告在联合下更容易撞上。
3. **中英回放 15% 是下限不是余量。** 三个目标语种同时挤压基座，回放被砍会最先体现在中英文指令跟随与克隆能力上——那是生产链路的核心能力，不是可牺牲项。

`.plan.json` 会记录这个 run 的 `langs`（各语种条数），run 名默认带语种标记，事后能反查训了什么。
⚠️ 标记里的语种数 N 是**清单里出现过的全部语种，含 zh/en 回放**（`default_run_name` 数的是
`manifest_langs()` 的去重结果），不是目标语种数；≤2 个语种时用 `-` 连接**按字母序排序**的语种码。
实测：三目标 + 中英回放 → `lora_joint5_*`；只有三目标无回放 → `lora_joint3_*`；
两语种 th+id → `lora_id-th_*`（不是 `th-id`）；单语种 → `lora_th_*`。
（本文早先写的「≥3 语种 `lora_joint5_*`」在旧五目标口径下也是错的——五目标 + 中英回放实际是
`joint7`。2026-10-19 按代码实测改正。）

**小源不足时不强凑配比。** 混合器会记录请求时长和实际时长占比，差异看 `mix.json`。宁可某个角色少一点，也不要靠 3× 重复把小源撑到目标占比 —— 重复曝光会过拟合。

首轮规模建议：

| 阶段 | 训练总时长 | 条数量级（3–30s） | 1 epoch 步数（单卡 batch2×accum8） |
|---|---|---|---|
| 试跑（验证链路） | 1–2h | 500–1500 | 30–100 |
| 首轮实验 | 20–40h | 1–2 万 | 600–1300 |
| 扩量 | 80–150h | 4–8 万 | 2500–5000 |

步数公式：`ceil(条数 × epoch / (batch_size × grad_accum × GPU数))`。这是按清单条数的**近似**，官方 loader 的丢尾和长度过滤会有偏差，以日志里的实际样本数为准。单步耗时需要实测（音频序列长，`batch_size=2` 是官方示例值，激活显存大，**不要调大**）。

先从首轮实验规模开始，几小时能跑完一轮，才有迭代速度。

### 2.2 下载公开源

```bash
# 只在远程执行。先小样本试跑确认列映射和过滤条件生效，再全量。
uv run python -m voxft.data.download --source thai_ser --max-samples 200
uv run python -m voxft.data.download --source thai_ser        # 整包 12.7GB（含 4 路麦克风）

uv run python -m voxft.data.download --source yodas_th --max-samples 200
uv run python -m voxft.data.download --source yodas_th        # 整包 26.6GB

uv run python -m voxft.data.download --source fleurs_th
uv run python -m voxft.data.download --source filswitch
uv run python -m voxft.data.download --source aishell3        # ~20GB，Apache-2.0

# ID：全部先小样本试跑，确认 config 名与字段形态后再全量（本轮未核实规模）
uv run python -m voxft.data.download --source gigaspeech2_id --max-samples 20
uv run python -m voxft.data.download --source fleurs_id --max-samples 50
uv run python -m voxft.data.download --source cv22_id --max-samples 200
```

产物落在 `$VOXFT_DATA_ROOT/raw/<source>/manifest.jsonl`。

下载器已经按 registry 做了行级过滤，**过滤条件缺列或值无效时不放行**（宁可少数据也不让未知数据进训练）：

- `thai_ser`：`turn_type=impro` 且 `agreement ≥ 0.7`；音频列按 `mic_con → mic_clip → mic_middle` 顺序取，**不用 `mic_zoom`**（网络录音，质量差）。这个源**没有名为 `audio` 的列**，全靠 registry 的 `audio_cols` 映射 —— 映射错了整个源会在下载阶段被静默跳过
- `yodas_th`：`grade_avg ∈ {S+, S}` 且 `dnsmos_overall ≥ 3.2`。`utt_id.rsplit("-", 3)[0]` 保留可能含 `-` 的完整视频 ID 作为 session。`speaker_id` 是视频级近似身份，**不作 ref 依据**
- `gigaspeech2_id`：Apache-2.0 无许可红线，但 **`gated` 需先在页面同意条款并在 `.env` 配 `HF_TOKEN`**。**字段形态本轮未核实**：试跑后看时长分布，若音频是 `path`+`start`+`end` 的长音频切片形态，`_load_audio` 会整段读入而非按时间戳切，时长过滤会大面积丢弃 —— 此时**直接停用本源**，不要改 `_load_audio` 去猜时间戳语义（FilSwitch 的教训：外链音频要保留 revision/镜像/认证，猜形态会静默丢数据）
- `fleurs_id` / `cv22_id`：config 名按 FLEURS 的 `{lang}_{country}` 与 CV locale 惯例推得，**小时数未核实**，试跑确认存在后再放量。`cv22_*` 走 Whisper 校验（众包噪音大），且用 `accept_langs` 关掉了 id 的英文放行 —— 朗读语料有权威文本，语种检测不符意味着错行而不是 code-switch
- `filswitch`：转换 parquet 可能只有元数据，音频在原仓库的独立 FLAC 文件里。`bytes=None` 不等于无音频，下载器会继续按其中的 HF 地址下载（保留 revision、镜像、认证、缓存）。**不要把音频地址套到 `refs/convert/parquet` 分支**。如果日志出现「写入 0 条」，先看具体的缺文本/音频读取失败统计，读取失败会记录原因、不静默跳过整包
- `aishell3`：content.txt 的同一正文列交错汉字与拼音，下载器已剔除拼音。**旧 processed 清单必须重新加工**，不能直接混入
- `filipino_speech`：过滤 `speech_type=machine` 与 `num_words < 4`，只保留完整句。**不拼接孤立词、不用随机抖动停顿伪造对白**（中位 0.63s / num_words 中位 1，拼接会训出报菜名式念稿感）

进度监控：`snapshot_download` 的进度条不传给单文件，用缓存目录大小轮询。xet 下载分两阶段（downloading → reconstructing），进度「回退」属正常。

### 2.3 自建短剧语料（主力，也是唯一可信的情绪/ref 来源）

#### 2.3.1 准备原始清单

已经切好、有文本的真人对白，直接写 `data/raw/<source>/manifest.jsonl`。音频路径相对 JSONL 所在目录。

```jsonl
{"audio":"audio/actor01_th_001.wav","text":"เธอทำแบบนี้ได้ยังไง","lang":"th","speaker":"actor01","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording01","emotion":"surprised","emotion_verified":true,"control_zh":"惊讶地，语气克制","control_en":"surprised, restrained","control_verified":true}
{"audio":"audio/actor01_zh_ref.wav","lang":"zh","speaker":"actor01","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording02","reference_only":true}
{"audio":"audio/actor02_th_014.wav","text":"ฉันไม่รู้จะพูดอะไรแล้ว","lang":"th","speaker":"actor02","speaker_namespace":"cast_v1","speaker_verified":true,"session":"recording01"}
```

字段规则：

- `audio` / `text` 必填。`text` 是**裸台词**，不要自己加括号前缀 —— 控制指令放 `control_zh` / `control_en`，加工阶段按 `control_ratio` 自动拼
- `speaker_verified=true` 见 [决策点 2](#决策点-2--跨语言同人-ref-是硬约束不是可以凑的字段)
- `session` 是同一次录音/同一集的分组键，用于 train/val 隔离。**同一集的切片必须有相同 session**，否则会被拆到 train 和 val 两边造成泄漏
- `reference_only=true` 的行只作为该 split 内的 ref 候选，**不作为训练目标**，时长约束放宽到 3–10s，文本可空
- 没有可信标签就**不写** `emotion_verified` / `control_verified` —— 缺失会让该条保持裸文本，这是正确的降级；写个 `false` 也一样，但不要凭猜测写 `true`

#### 2.3.2 成片素材导入

素材是成片视频或音轨时，用导入链路代替手工切片：

```
PyAV 解码 → Whisper VAD 定边界(medium) → 隔 ≤0.7s 的相邻区间合并成 3–30s 候选
（超长的在最安静的一帧切开，不切在词中间）→ large-v3 逐条转写 + 语种过滤
→ 页面逐条试听标注 → 追加进 manifest.jsonl → 可选自动重新加工
```

```bash
# 需要 uv sync --group qc（faster-whisper + PyAV）；只在远程执行
uv run python -m voxft.data.ingest --input /root/autodl-tmp/drama/ep01.mp4 --max-items 20   # 试跑，不追加
uv run python -m voxft.data.ingest --input /root/autodl-tmp/drama/ep01.mp4                  # 切分+转写
uv run python -m voxft.data.ingest --input ep02.mp4 ep03.mp4                                 # 批量，素材 ID 取文件名
uv run python -m voxft.data.ingest --input ep01.mp4 --append --holdout ep01 --process --out drama_th_v2
```

产物结构：

```
data/raw/drama_th/manifest.jsonl        # 追加目标，加工读它
data/raw/drama_th/holdout.json          # {"sessions": ["ep01"]}：钉住的素材只进验证集
data/raw/drama_th/ingest/<素材ID>/
    source.wav                          # 解码后的 16k 单声道全轨（重切不必重解码）
    clips/0001_s0012.34_e0018.90.wav
    candidates.jsonl                    # 全量候选，含坏例与「丢弃/待定」；人工标注写回这里
```

`--source` 默认 `drama_tl`，做泰语/印尼语要显式 `--source drama_th` / `drama_id`。

要点：

- **code-switch 语种不能只认目标语种。** Tagalog 短剧台词是 Taglish、印尼语日常口语混英文，句内英文词多的样本 Whisper 会判成 en。`Source.languages()` 对 tl / id 默认放行 `en`（`CODE_SWITCH_ACCEPT`，2026-10-19 起只剩这两个语种），只认目标语种会把最该保留的 code-switch 样本全部误杀。有权威文本的朗读源反过来用 `accept_langs` **覆盖掉**这个默认放行 —— 语种不符意味着错行，不是 code-switch
- **转写可断点续跑**：每 100 条落盘，重跑跳过已转写行，坏例只排除不删除。`--max-items` 试跑不回写原清单。推理异常必须中止，不能当语料坏例吞掉
- **同一素材重切会替换它上次追加的行**（按 `ingest_video`），不会叠加成近似重复样本；再按音频绝对路径去重
- `session` 自动设为素材 ID
- **不做声源分离/降噪**，BGM 重的条目在试听环节判「丢弃」

#### 2.3.3 人工标注（这一步的质量直接决定情绪表现力上限）

在页面「素材导入」Tab 逐条试听并标注。标注入口的校验规则和加工阶段一致，别等加工时才炸：

- `verdict` ∈ 保留 / 丢弃 / 待定。**标「保留」必须有台词文本**
- 原始 `text` 必须是裸台词，以 `(` 或 `（` 开头会直接报错
- 勾了「已核实是本人」才写 `speaker_verified=true`，且必须提供真实 speaker ID（不能是空或 `default`）。**切分不出说话人，身份必须人工核实。** 只写 ID 不勾选 → 落成未验证身份 → 不配 ref、不调响度
- `control_zh` / `control_en` 只能中英文，含泰文或越南语专属字符（`ơ ư đ` 与带声调元音）会报错（`_assert_control_lang`）。**印尼语是纯 ASCII 拉丁字母，与英文无法区分，守卫拦不住，只能靠标注纪律**。**不要写目标语言的控制前缀** —— 线上 prompt 就是中英文

**情绪标签的两条路：**

1. `emotion` 填下表里的键 + `emotion_verified=true` → 自动从短语池随机取一条中/英文指令

   | emotion 键 | 中文 | 英文 |
   |---|---|---|
   | `neutral` | 平静地 / 语气自然 / 正常语调 | neutral tone / calm and natural / plain delivery |
   | `angry` `anger` | 愤怒地 | angry |
   | `happy` `happiness` | 开心地 | happy |
   | `sad` `sadness` | 伤心地 | sad |
   | `fearful` `fear` | 害怕地 | fearful |
   | `surprised` | 惊讶地 | surprised |
   | `frustration` `frustrated` | 烦躁地 | frustrated |
   | `disgust` | 厌恶地 | disgusted |

   另有 `rate_label ∈ {slow, fast}` 和 `volume_label ∈ {quiet, loud}`，需要 `control_verified=true` 才生效。

2. **短剧最需要的那些表演方式不在上面的池子里** —— 哭腔、带笑说话、讽刺、克制愤怒、咬牙切齿、气声。这些必须手写 `control_zh` + `control_en` 并置 `control_verified=true`，此时优先用你写的自由文本。

   ```jsonl
   {"control_zh":"带着哭腔，声音发抖","control_en":"tearful, voice trembling","control_verified":true}
   {"control_zh":"讽刺地，语速慢","control_en":"sarcastic, slow paced","control_verified":true}
   ```

   每条控制描述都必须有对应的真实录音支持，并且**经过试听核验**。「生气」不会被自动扩写成「大声喊叫」—— 那是伪造标签。

**不要凭能量/F0/空格数猜测表演。** 未知标签保持裸文本，比例不足就不足，不伪造。

### 2.4 钉住验证集

`split_records` 的随机分组结果依赖清单长度。**追加新素材后重新加工，上一轮的验证组会被整体重排进训练集** —— 已训 run 的评测结论随之作废。

挑一集写进 `data/raw/<source>/holdout.json`：

```bash
uv run python -m voxft.data.ingest --input ep05.mp4 --append --holdout ep05 --process --out drama_th_v2
```

或手写 / 页面填「钉进验证集的素材 ID」：

```json
{"sessions": ["ep05"]}
```

钉住的分组不参与 shuffle，永远只进验证集。`stats.json` 的 `holdout_pinned_records` 可核对。矛盾组合（钉住了却 `val_ratio=0`、或全部素材都被钉住）会直接报错，不会静默把钉住的数据喂进训练。

自有 acted / natural 两份清单如果共享演员或会话，**需在分源前统一安排 holdout** —— 混合器会拒绝跨源 train/val 泄漏。也可以先合并原始清单统一加工，接受两类语料内部的自然时长配比。

---

## Phase 3 · 逐源加工与检查

### 3.1 加工

先试跑，看丢弃统计是否符合预期，再全量：

```bash
uv run python -m voxft.data.pipeline --source thai_ser --max-items 200 --out thai_ser_probe
```

正式加工。**每次写新的音频子目录（`_v2` 后缀），不覆盖旧清单引用的音频** —— 旧产物不自动清理，确认无训练/混合清单引用后再人工清：

```bash
uv run python -m voxft.data.pipeline --source thai_ser      --out thai_ser_v2
uv run python -m voxft.data.pipeline --source yodas_th      --out yodas_th_v2
uv run python -m voxft.data.pipeline --source fleurs_th     --out fleurs_th_v2
uv run python -m voxft.data.pipeline --source aishell3      --out aishell3_v2

uv run python -m voxft.data.pipeline --source drama_th \
  --manifest /root/autodl-tmp/voxft_data/raw/drama_th/manifest.jsonl --out drama_th_v2
uv run python -m voxft.data.pipeline --source replay_en \
  --manifest /path/to/replay_en.jsonl --out replay_en_v2
```

`--source` 用 local 源（`drama_*` / `replay_en`）时必须配 `--manifest`。

加工链路：16k 重采样 → 裁首尾 → 时长过滤（3–30s）→ 质检 → 声学描述 → 身份/会话隔离切分 → 按说话人响度对齐 → 可信控制前缀 → split 内 ref 配对。

### 3.2 自动生效的策略（按 registry 分流，CLI 不暴露）

`options_for(source_id)` 按 registry 的 `expressive` 字段自动决定，**这是有意的，不要手工统一**：

| | 表演语料（`drama_*` / `thai_ser`） | 朗读语料（其余） |
|---|---|---|
| 首尾裁切 | 帧 RMS 门限 `0.02`（回落 `peak×0.01`） | **Silero VAD** 定边界（`edge_vad=True`） |
| 换气声 | **保留** —— 那是表演的一部分，裁掉模型就学不会换气 | 裁掉 |
| `control_ratio` | 0.5 | 0.25 |

其他默认：`ref_audio_ratio=0.5`、`ref_control_ratio=0.3`、`ref_min/max_dur=3/10s`、`val_ratio=0.02`、`val_max=200`、`target_dbfs=-24`、`control_zh_ratio=0.5`、`seed=42`。

CLI 可调：`--utmos-min` `--whisper-lang` `--control-ratio` `--asr-min-logprob` `--asr-max-no-speech` `--ref-audio-ratio` `--ref-control-ratio` `--val-ratio`。
`--min-snr-db` 只兼容旧参数，**启用会报错**。

### 3.3 检查 stats.json

产物：`$VOXFT_DATA_ROOT/processed/<out>/{train,val}.jsonl`、`stats.json`。

```bash
cat $VOXFT_DATA_ROOT/processed/drama_th_v2/stats.json | python -m json.tool
```

**丢弃计数**（`drop_*`）—— 数字异常大说明门限或数据有问题，不要直接往下走：

| 键 | 含义 | 异常时怎么办 |
|---|---|---|
| `drop_transcribe` | ASR 置信度不合格 | 只对 `needs_transcribe` 源生效。门限沿用 faster-whisper 解码器默认（时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`）。砍太多就 `--asr-min-logprob -1.5`。日志会打印原因分类 |
| `drop_decode` | 音频解码失败 | 检查容器/采样率；mp4/mkv 需要 PyAV |
| `drop_duration` | 不在 3–30s | 素材本身太碎，考虑调 ingest 的 `--min-dur` |
| `drop_lang` | 检测语种不符 | TL 源确认放行的是 `("tl","en")` |
| `drop_whisper` | 转写相似度 < `0.55` | 训练文本与音频对不上，该丢 |
| `drop_utmos` | UTMOS 音质分低 | UTMOS 权重源已失效，默认不启用 |

**覆盖统计**（`train` / `val` 各一份，来自 `dataset_summary`）—— 这几个数决定训练效果：

| 键 | 目标 | 不达标的后果 |
|---|---|---|
| `with_control / rows` | 25–50% | <25% 预检会警告，情绪指令跟随弱 |
| `with_ref_audio / rows` | 30–50% | <30% 预检会警告，克隆能力易被 LoRA 损害 |
| `with_ref_control / rows` | ~30% | ref + control 联合覆盖，最贴近生产形态的组合 |
| `cross_language_refs` | > 0 | 为 0 说明没有真正的跨语言同人 ref，见 [决策点 2](#决策点-2--跨语言同人-ref-是硬约束不是可以凑的字段) |
| `speakers` | 多名 | 只有 1–2 人会让模型把情绪绑到音色上 |
| `emotions` | 分布均衡 | 某情绪只来自某一名演员 = 学到的是音色不是情绪 |
| `max_exposure` | ≤ 3 | 超过说明混合重复过头 |
| `hours` | 对照 Phase 2.1 | 实际可用时长 |

联合覆盖目标（有足够已审核标签与身份时）：**ref+control 30% / ref+裸文本 20% / 无 ref+control 20% / 无 ref+裸文本 30%**。逐源加工后再混合，**最终覆盖不保证自动达到** —— 缺口用真实标注补，不要调低阈值凑。

`stats.json` 还记录本次用的 `edge_vad` / `edge_trim_ratio` 和完整 `options`，旧产物能反推当时门限。

### 3.4 响度对齐的边界

`target_dbfs=-24`：**仅对 `speaker_verified=true` 的说话人做整体增益**。防削波时整个说话人共同回退，保留条内与条间的相对动态。

- 禁止单条峰值归一化（会抹掉表演的强弱对比）
- 未知身份不统一调响度
- 上游已经逐条归一的（如 `yodas_th`）动态恢复不了，也不据此标音量指令

---

## Phase 4 · 混合与检查

### 4.1 混合

前提：所有参与混合的数据集名称都已完成 Phase 3 加工。

```bash
# TH
uv run python -m voxft.data.pipeline --out th_drama_v2 --mix \
  thai_ser_v2=25 drama_th_v2=20 yodas_th_v2=35 fleurs_th_v2=5 aishell3_v2=10 replay_en_v2=5

# TL
uv run python -m voxft.data.pipeline --out tl_drama_v2 --mix \
  drama_tl_v2=45 natural_tl_v2=30 filswitch_v2=10 aishell3_v2=10 replay_en_v2=5

# ID（首轮：表演档 0%，gigaspeech2_id 需先试跑通过；drama_id 素材到位后 drama=45 / 口语=30）
uv run python -m voxft.data.pipeline --out id_anchor_v1 --mix \
  gigaspeech2_id_v1=75 fleurs_id_v1=5 cv22_id_v1=5 aishell3_v2=10 replay_en_v2=5
```

权重是**目标时长占比**（相对值，不必凑满 100，但凑满便于对照）。`--mix` 必须配 `--out`，输出不能覆盖输入，输入不能重复。

混合规则：

- 按**有效音频时长**采样，不按条数
- 每条原始目标音频全局最多 **3×** 曝光（嵌套混合也检查），`max_repeat` 只能是 1/2/3
- ref 复用另行统计（`max_ref_exposure`），3× 上限只针对训练目标
- 验证集不重复采样
- 混合器会拒绝跨源 train/val 泄漏

**关键：目标重复 3× 再训练 3 epoch = 最多约 9 次曝光。所以先跑 1 epoch。**

### 4.2 检查 mix.json

```bash
cat $VOXFT_DATA_ROOT/processed/th_drama_v2/mix.json | python -m json.tool
```

- `parts` — 请求的权重
- `datasets[].actual_duration_share` — **实际时长占比**。和请求值差得多说明某个源不够量，被强行拉高了曝光
- `datasets[].max_exposure` — 该源的最大重复次数。接近 3 就是量不够
- `counts` / `train` / `val` — 同 Phase 3.3 的覆盖统计，这次是混合后的最终值

**这里的 `train.with_control` / `train.with_ref_audio` 就是预检要看的数**。不达标先回去补数据，不要直接开训。

---

## Phase 5 · 训练

### 5.1 生成配置

```bash
uv run python -m voxft.train.yaml_builder \
  --train /root/autodl-tmp/voxft_data/processed/th_drama_v2/train.jsonl \
  --val   /root/autodl-tmp/voxft_data/processed/th_drama_v2/val.jsonl \
  --base  /root/autodl-tmp/models/VoxCPM2 \
  --run   th_r1_e1 \
  --epochs 1 --gpus 1
```

输出 `configs/th_r1_e1.yaml` 和 `configs/th_r1_e1.plan.json`（记录卡数、epoch、等效 batch、训练条数，供预检比对）。

**`--base` 必须指向已下载的本地模型目录**，不能是 HF 仓库 ID —— 官方训练脚本要求本地目录。省略时用 `.env` 的 `VOXCPM_BASE_PATH`。

生成的 LoRA 配置（官方推荐 + 本项目默认，**不要随手改**）：

```yaml
learning_rate: 1.0e-4        # LoRA 用 1e-4；全量微调用 1e-5（1/10）
lora:
  enable_lm: true
  enable_dit: true           # 对音质至关重要，必开
  enable_proj: false
  r: 64                      # 语言+风格适配用 64；纯说话人适配 32 就够
  alpha: 64
  dropout: 0.05              # 情感语料体量小，0 容易几百步就过拟合到固定腔调
sample_rate: 16000           # AudioVAE 编码器输入，不是输出采样率
out_sample_rate: 48000       # 仅推理
batch_size: 2
grad_accum_steps: 8          # 等效 batch = 2 × 8 × GPU数
num_workers: 8
max_grad_norm: 1.0
weight_decay: 0.01
warmup_steps: <num_iters × 0.1>
save_interval: 250
valid_interval: 250
```

`epochs` 允许 1–3，`warmup_steps` 自动设为 `num_iters × 0.1`。**`epochs` 与显式 `num_iters`/`max_steps` 只能选一种**，同时给会报错。手动步数仅供受控实验。

**`training_cfg_rate=0.1` 不在训练 YAML 里。** 它属于基座 `config.json` 的 `dit_config.cfm_config`。往 overrides 里塞会直接报错，放在 YAML 顶层预检会拦。这是防过拟合忽略文本的关键项，确认基座里是 0.1。

**更改 GPU 数或训练清单后必须重新生成配置** —— `.plan.json` 会和实际比对，不一致预检报错。

### 5.2 预检

```bash
# 页面：训练 Tab → 预检
# 或直接在 Python 里调 launcher.preflight("configs/th_r1_e1.yaml", gpus=1)
```

预检会检查：官方训练脚本存在、清单每行合法 JSON、每条音频文件真实存在、每条有训练文本、ref 有已验证同人身份（`speaker_verified=true` 且 `ref_speaker == speaker`）、ref 不等于目标音频、**train/val 不共享音频/ref/说话人/会话**、基座 `training_cfg_rate=0.1`、`.plan.json` 与实际卡数条数一致、`save_path` 可写。

两条警告线：带前缀目标 <25%、同人 ref 覆盖 <30%。还有 `num_iters × effective_batch > 3 × 训练条数` 时警告超 3 epoch。

**警告不是可以忽略的形式主义。** 出现就回 Phase 2/3 补数据。

### 5.3 启动

预检通过后，在 GPU 机器上：

```bash
uv run python -m voxft.train.launcher configs/th_r1_e1.yaml 1
```

这条命令**打印**训练命令，复制到远程执行：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python /path/to/third_party/VoxCPM/scripts/train_voxcpm_finetune.py \
  --config_path /path/to/configs/th_r1_e1.yaml
```

多卡：

```bash
cd third_party/VoxCPM && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=4 scripts/train_voxcpm_finetune.py --config_path <yaml>
```

也可以从页面「训练」Tab 直接启动（`launcher.start_local` 起后台子进程，日志写 run 目录下 `train.log`）。已有任务在跑时会拒绝再启动。

wandb 桥接自动转发 TensorBoard 指标，沿用原配置。

### 5.4 监控

- **不同实验用不同 run 名，不覆盖旧 run。** 命名带上轮次和变量：`th_r1_e1`、`th_r2_ctrl50_e1`
- 同配置恢复会从 `latest/` 继续
- 每 250 步保存 + 验证，训练结束官方脚本也保存一次
- **不以 loss 最低作为选 checkpoint 的依据。** 情感语料的 loss 和听感经常不同向
- **训练结束会自动只保留最新 5 次 LoRA 运行**（`cleanup_lora_runs(keep=5)`，按 `latest/` 修改时间排序，全量微调目录不受影响）。**好的 run 及时备份出去**，否则会被后续实验挤掉

---

## Phase 6 · 验收

### 6.1 自动指标（诊断，不是结论）

```bash
uv run python -m voxft.eval \
  base \
  $VOXFT_CKPT_ROOT/th_r1_e1/step_750 \
  $VOXFT_CKPT_ROOT/th_r1_e1/latest \
  --texts-file eval/th_holdout.jsonl --seeds 42 43 44
```

第一个位置参数是 `base`（纯基座）或若干 LoRA checkpoint 目录。多个 target 会依次跑并在最后打印对比表。

报告写到 `$VOXFT_CKPT_ROOT/eval/<label>_<uuid>.json`，含逐条：`hyp`（ASR 转写）、`similarity`、`cer`、`wer`（`WER_LANGS`：tl/en/id）、`suspected_truncation`、韵律描述、`wav` 路径、`gen_sec`，以及待填的 `human_review`。

汇总看 `mean_cer`、`suspected_truncation_rate`、`mean_similarity`。

**读指标的正确姿势：**

- **ASR 误差 ≠ 发音错误。** Whisper 对泰语声调和 Taglish 本身就有误差。CER 只能看趋势和相对差异，不能当绝对门限
- **疑似漏尾 ≠ 真实截断。** 判据是 `len(hyp) < 0.6 × len(ref)` 或尾部 8 字符相似度 <0.5，ASR 同义转写也会触发。**必须听音确认**
- 泰语归一化保留声调/元音组合符（Unicode M 类），Taglish 不强制单一 ASR 语言（`language=None`），th/id 强制指定解码语种让 CER 可比
- ⚠️ **仅历史集适用**：**vi 的 WER 是音节级**（越南语正字法按音节空格分隔），不与 id/tl/en 的词级 WER 横向比。vi 已退出 `WER_LANGS`（现为 `{tl, en, id}`），但读含 vi 行的旧报告时仍要按这条量纲口径
- CER/WER 可以 >1（插入错误），不截断
- **F0/能量仅描述，不是越高越好**。泰语是声调语言，`f0_std_st` 含词汇声调，同样不代表自然度

评测时 `retry_badcase=False` —— 禁用自动换种子重试，避免掩盖差异。普通试听保持 `retry_badcase=True`。

### 6.2 A/B 对比

页面「试听」Tab。`synthesize_ab` 用**同一个模型**禁用/启用适配器，固定 seed，严格读 checkpoint 的 `lora_config`（官方 JSON 的嵌套层），检查 `load_lora` 返回的 `(loaded, skipped)` 和 missing keys：**loaded 非空、skipped 为空、missing 为空**，任一不满足直接抛错停止，绝不伪装成 LoRA 输出。

A/B 是验收「微调是否更听指令」的核心手段：同一 control 前缀，基座 vs LoRA。

### 6.3 母语盲听（唯一的通过判据）

至少两名母语评审，随机盲听同条件 A/B，四项分别 1–5 分：

| 字段 | 评什么 |
|---|---|
| `naturalness_1_5` | 自然度、去念稿感、对白韵律 |
| `emotion_fit_1_5` | 情绪匹配度、是否真的跟随了控制前缀 |
| `intelligibility_1_5` | 清晰度、发音准确、有没有吞字漏字 |
| `speaker_similarity_1_5` | 克隆音色与 ref 的相似度 |
| `cutoff` / `noise` | 真实截断 / 噪声，布尔标记 |

**分组看结果**：按 ref 语言（中/英/目标语）、角色（女主/其他女声/男声）、情绪类型、以及 `kind=clarity` 分别统计。整体均值会掩盖「男声崩了女声好了」「普通句没问题但长句和快语速糊了」这两种情况 —— 后者正是清晰度回归的典型形态，见[清晰度回归的判定](#清晰度回归的判定)。

**通过标准：自然度/情绪改善，且清晰度与音色不退化。** 两个条件都要满足。情绪涨了但音色漂移，不算通过。

自动报告**不生成「通过」结论**，结论由人写。

### 6.4 实验顺序

| 轮次 | 变量 | 目的 | 判据 |
|---|---|---|---|
| **R0** | 无 | 基座固定评测基线；旧 LoRA 若可用也测一次 | — |
| **R1** | 修正后的高质量目标语料 + 中英回放 | 1 epoch，确认发音/漏词**没有退化** | CER 不升、intelligibility 不降 |
| **R2** | 加入真人表演 + 可信中英控制标签 | 相同训练预算，重点比自然度和情绪 | naturalness / emotion_fit 上升 |
| **R3** | 补跨语言同人 ref 配对 | 比较中/英文 ref → TH/TL 的音色保持 | speaker_similarity 不低于基座 |

**每轮从同一基座开始，尽量固定总训练时长与采样曝光，只改变要检验的那一个数据因素。** 同时改两个变量就没法归因了。

R3 没有真实同人跨语言素材时**不要声称完成该验证** —— 记为「未验证」，而不是「通过」。

### 6.5 选 checkpoint

达标后才试第二个 epoch。出现内容服从、情绪或音色退化就**选择更早的 checkpoint**（`step_250` / `step_500` / …），不以 loss 最低为依据。

用同一份评测集把中间 checkpoint 都跑一遍再决定，不要只看 `latest`。

---

## Phase 7 · 迭代调参：症状 → 动作

**一次只改一个因素**，改完重跑 Phase 4→6。表里「先看什么」是归因步骤，跳过它很容易改错地方。

| 症状 | 先确认 | 动作 |
|---|---|---|
| 发音不准、重音/声调怪 | eval CER 趋势 + 盲听 `intelligibility` | 加发音锚点源占比（`fleurs_*` / `porjai_th` / `filswitch`）；确认**训练文本准确** —— ASR 转写错误会直接教坏发音，高价值子集要人工校对 |
| 念稿感、平读、不像对白 | 表演语料 `actual_duration_share` 是否到了 45% | 加 impro / 短剧占比；确认表演源 `edge_vad=False`（换气被裁掉就学不会呼吸）；朗读源占比过高会拉回念稿感 |
| 情绪前缀不跟随 | A/B 同 control 前缀 base vs LoRA；`with_control / rows` | <25% → 补可信标签（不是伪造）；检查 `with_ref_control` 是否接近 30%；确认线上推理走的是 reference-only 模式 |
| 带前缀变好但无前缀变差 | eval 里 `kind=plain` 的 case | `control_ratio` 太高，降到 0.25–0.35，保住裸文本推理路径 |
| 哭腔/带笑/讽刺学不出来 | 这些不在自动短语池里 | 必须手写 `control_zh`+`control_en` 并置 `control_verified=true`，且要有对应真实录音 |
| 某种情绪 = 某个人的音色 | `stats.json` 的 `emotions` × `speakers` 交叉 | 补其他演员的同类情绪；情绪覆盖必须跨演员 |
| 音色漂移、克隆变差 | 盲听 `speaker_similarity` + A/B | ref 覆盖是否 <30%；`enable_dit` 是否 true；epoch 是否过多；先回退到更早 checkpoint；仍不行则 `r` 从 64 降到 32 |
| 中文/英文能力退化 | 评测集里的 `antiforget` case | 回放占比提到 10% zh + 5% en；确认 `aishell3` 正文没混入拼音（旧清单必须重加工） |
| 生成停不下来、尾巴拖长 | `suspected_truncation_rate` + 听音确认 | 尾静音必须 <0.5s；朗读源确认 `edge_vad=True`（RMS 门限单独用有悬崖，见附录 A） |
| 漏词、吞字 | `cer`/`wer` + 盲听 | 训练文本质量；提高 ASR 置信度严格度（`--asr-min-logprob` 调到 -0.9）；针对具体难词补真实语料 |
| **吐字软、含糊、发闷**（不是漏词，是音不清） | 线上 `inference_timesteps` 是 20 还是 10；`cfg_value` 是否偏离 2.0 | 先修推理参数（最便宜）；再看是否过训练 → 回退更早 checkpoint；确认 `enable_dit=true` |
| **句尾辅音 / 尾音被截断** | 训练音频的 `tail_keep`；泰语句尾塞音、Tagalog 收尾辅音与喉塞音 | 确认 `tail_keep=0.3` 没被改；`suspected_truncation` 高的条目听音确认是数据问题还是生成问题；补句尾辅音清晰的锚点源（`porjai_th` / `fleurs_*`） |
| **长句后半段变糊** | eval 里 `kind=clarity` 长句 case 的 CER 分布 | `max_batch_tokens=8192` 的长度过滤让长样本在训练里欠曝光 → 提高长句（20–30s）在语料里的占比；推理加 `inference_timesteps` |
| **快语速时清晰度崩** | `语速快` 前缀的 case | 训练里缺快语速的清晰样本 → 补 `control_verified=true` + `rate_label=fast` 的真实条目，不要用加速音频伪造 |
| **某个 step 之后音质突然劣化** | train.log 的 loss 曲线有没有尖刺 | 梯度尖峰。回退到尖刺前一个 checkpoint；确认 `max_grad_norm=1.0` 没被关掉 |
| 有杂音、BGM | 盲听 `noise` | 素材导入阶段判「丢弃」。没有降噪链路，不要试图在训练侧修 |
| loss 降但听感变差 | — | 不以 loss 为准，选更早 checkpoint |
| 几百步就固定成一个腔调 | `dropout` 是否为 0 | 保持 0.05；减少 epoch；检查 `max_exposure` 是否接近 3 |
| 语料加了但效果没变 | `mix.json` 的 `actual_duration_share` | 新源量太小被稀释了；或者被 3× 重复撑起来的，看 `max_exposure` |

### 超参数调整顺序

按这个优先级动，前面的没排除不要动后面的：

1. **数据**（占比、标签质量、ref 覆盖、语料量）—— 90% 的问题在这里
2. **epoch / checkpoint 选择** —— 1 → 更早的 checkpoint → 2 → 3
3. **`control_ratio`**（0.25 ↔ 0.5）
4. **LoRA `r`**（64 ↔ 32）—— 语言+风格用 64，纯说话人适配 32 够；降 r 可以减轻对克隆能力的损害
5. **`dropout`**（0.05 ↔ 0.1）
6. **`learning_rate`**（1e-4，最多在 5e-5 ↔ 2e-4 之间试）
7. **`batch_size` / `grad_accum_steps`** —— 改了要重新生成配置，等效 batch 变化会连带影响 lr

`enable_dit: true` **不要关**，那是音质的关键。`training_cfg_rate=0.1` 不要动。

### 推理侧参数

生产/试听默认（OmniVoice 生产实践对齐）：

| 参数 | 值 | 说明 |
|---|---|---|
| `cfg_value` | 2.0 | 音频坏例重试时降到 1.2–1.6 |
| `inference_timesteps` | 20 | 生产基线是 10，本项目默认 20 换更高音质；坏例时加步数 |
| `retry_badcase` | True | `max_times=3`、`ratio_threshold=6.0`。**A/B 和批量评测必须关掉** |
| `load_denoiser` | False | 去噪器依赖 modelscope，试听不需要 |

**两个必须一致的地方：**

1. **推理侧 LoRA 配置与训练完全一致**（官方 JSON 的 `lora_config` 嵌套层）。`infer.get_model` 会读 checkpoint 里的配置，加载不完整直接抛错
2. **带控制前缀时必须走 reference-only 模式**：只传 `reference_wav_path`，**不要传 `prompt_text`**。combined 模式会拼成 `prompt_text + "(控制)正文"`，前缀跑到句中就失效了。`infer._gen_kwargs` 已按此处理（有 control 时不传 `prompt_text`），生产链路要照抄这个逻辑

控制前缀格式就是文本前缀，没有独立条件通道：`(控制指令)正文`。`clean_control` 会清掉指令里的括号，避免破坏格式。

---

## 清晰度专项 · 全链路控制点

清晰度不是一个开关，它在五个阶段各有独立的损耗点，任何一处漏了都会在最终音频上表现为「糊」。而且它和另外两个目标**直接冲突**：

- 与「裁静音」冲突 —— 门限调高会啃掉词首清辅音
- 与「保留换气」冲突 —— 表演源不裁，底噪和口水音就留在训练数据里
- 与「情绪强度」冲突 —— 喊叫/耳语动态范围大，要么削波要么听不清

所以要按阶段逐个控制点检查，**不要指望找到某一个参数把它调好**。

### 战线 1 · 数据入口（决定上限，权重最大）

没有降噪链路，糊的录音进去就是糊的模型出来。

| 控制点 | 位置 | 怎么做 |
|---|---|---|
| 只收干净录音 | 素材导入试听 | 本项目**不做声源分离/降噪**（无 demucs 依赖，官方 zipenhancer 需 modelscope 且不在 lock 里）。BGM、混响、口水音重的条目直接判「丢弃」，不要试图用训练修 |
| 上游音质分级 | registry `row_filters` | `yodas_th` 已有 `dnsmos_overall ≥ 3.2` + `grade_avg ∈ {S+, S}`。这是现成的清晰度过滤，**不要放宽** |
| UTMOS 音质过滤 | `--utmos-min 3.5` | 实现在 `qc/utmos.py`（移植自 OmniVoice，Apache-2.0）。**当前权重源已失效，默认不启用**（所有源 `qc` 都不是 `full`）。能拿到权重就给朗读源加一道；表演源慎用 —— UTMOS 偏好录音棚音色，会把有呼吸感的表演判低 |
| 训练文本与音频一致 | `drop_whisper` | 相似度门限 0.55。**文本错了就是教模型错误发音**，危害比音频糊一点更大 |
| 含糊音过滤 | `drop_transcribe` | `needs_transcribe` 源：时长加权 `avg_logprob < -1.0` 或 `no_speech_prob > 0.6`（faster-whisper 解码器自己的默认）。清晰度不达标就 `--asr-min-logprob -0.9` 调严 |
| 语种过滤 | `drop_lang` | 混入其他语种会教错音系。TL 放行 `(tl, en)` 是为保 Taglish，不是放松标准 |
| 不拼接孤立词 | `filipino_speech` 过滤 | 中位 0.63s / `num_words` 中位 1。拼接出的伪句词间韵律错乱，直接损伤可懂度 |
| 发音锚点源 | 混合占比 5–10% | TH 首选 `porjai_th`（录音棚级标准泰语，700h，TTS 专用）；`fleurs_*` 干净朗读；TL 用 `filswitch` 教句内英文词与数字读法。**锚点源是朗读体，占比过高带来念稿感** —— 5–10% 是上限不是起点 |
| 人工校对高价值子集 | 手工 | 表演语料的转写错误率高于朗读。至少把要进评测集的难词、以及情绪最强那批条目人工过一遍 |

### 战线 2 · 首尾裁切（清晰度最容易在这里被误伤）

`trim_silence` 的门限是三重保护，每一重都对应一个清晰度事故：

```python
thr = max(peak * floor,                    # floor=1e-3，约 -60dB 绝对下限
          min(noise * 3.0,                 # 上限：底噪(帧RMS p10) × 3
              max(voiced * edge_ratio,     # 相对有声电平(帧RMS p99)
                  peak * 0.01)))           # voiced 退化时的回落
```

- **`noise * 3.0` 这个上限是清晰度的保险丝**：几乎没有静音的条目上，p10 就是轻声语音本身。不设上限会把弱辅音整段裁掉
- **`min_run=0.25` 是第二个保险丝**：首尾低电平段必须**连续** ≥0.25s 才裁。词首清辅音 /s/ 只有 80–120ms、/h/ 只有 40–80ms，都比 0.25s 短，所以不会被误啃。**调低 `min_run` 或调高门限会直接损伤清晰度 —— 这两个值不是可以随手优化的旋钮**
- 边界按**帧 RMS** 判定而非逐采样点：一声口水音的单个尖峰不该把整段空白留下
- 众包语料底噪高，只用相对峰值的固定门限（−60dB）经常整条裁不动 —— 这就是要三重取大的原因

**`tail_keep=0.3`**：尾部最多留 0.3s（官方要求 <0.5s，防「生成停不下来」）。裁太狠会截断尾音 —— 泰语的句尾塞音、Tagalog 的收尾辅音与喉塞音（正字法里常不写出来，最容易被忽略）最先丢；留太多会训出拖沓。0.3 是这个权衡的落点，不要改。

**表演源 / 朗读源的分流本身就是清晰度决策**（由 registry 的 `expressive` 字段自动决定）：

| | 表演源（`drama_*` / `thai_ser`） | 朗读源 |
|---|---|---|
| `edge_vad` | False | True |
| `edge_trim_ratio` | 0.02（回落 `peak×0.01`） | 0.06 |
| 换气 / 呼吸声 | **保留** —— 裁掉模型就学不会换气 | 裁掉 |
| 代价 | 底噪和口水音也留着 | RMS 门限有悬崖（附录 A），必须走 VAD |

某个表演源底噪实在重时，正确做法是**在素材导入阶段丢弃那些条目**，不是提高全局裁切门限 —— 后者会啃掉所有条目的词首辅音。

### 战线 3 · 响度与削波

`apply_speaker_gain(target_dbfs=-24)`：

- 对同一说话人施加**同一个**增益，使其响度中位数落在 −24 dBFS，**条与条之间的相对强弱完整保留**
- 防削波：`gain = min(gain, 0.97 / peak)`，peak 取该说话人**所有条目**的最大值 —— 所以是整个说话人共同回退。0.97 的余量就是防削波用的
- **削波是不可逆的清晰度损失**，写进训练音频就修不回来
- 逐条峰值归一会把喊叫和耳语拉到同一响度，抹掉「音量 = 情绪强度」这条线索 —— 这正是情感语料训完仍然平淡的原因之一；而且耳语被放大时底噪一起被放大
- **只对 `speaker_verified=true` 的说话人生效**（未验证组可能混人）。未知身份不调响度
- 上游已经逐条归一的（如 `yodas_th`）动态恢复不了，也不据此标音量指令

### 战线 4 · 训练配置

| 项 | 值 | 与清晰度的关系 |
|---|---|---|
| `enable_dit` | **true** | 官方明确「对音质至关重要」。不要为省显存关掉 |
| `sample_rate` | 16000 | **AudioVAE 编码器输入**，不是输出采样率。与 `out_sample_rate=48000` 搞混会导致训练/推理不匹配，输出发闷发糊 |
| `max_grad_norm` | 1.0 | 情感语料动态大，不裁剪更容易出梯度尖峰。尖峰会让音质在某个 step 之后突然劣化 —— **看到 loss 尖刺就往前一个 checkpoint 回退** |
| `batch_size` | 2 | 官方示例值。音频序列长、激活显存大，**不要调大**（OOM 会逼你降 `max_batch_tokens`，那才是真正的损失） |
| `max_batch_tokens` | 8192 | 长度过滤。超长样本被官方 loader 丢弃/截断，**实际训练分布和你以为的不一样** —— 以日志里的实际样本数为准 |
| 曝光次数 | 先 1 epoch | 混合 3× 重复 × 3 epoch = 最多 9 次曝光。过训练会过拟合到固定腔调，同时伴随高频细节丢失（听感就是「糊」） |

### 战线 5 · 推理侧（不用重训就能拿到的清晰度）

最容易被忽略，但**收益最快**。

| 参数 | 默认 | 清晰度影响 |
|---|---|---|
| `inference_timesteps` | **20** | 生产基线是 10，本项目默认 20 换更高音质。**这是最便宜的清晰度提升**，代价是推理时间翻倍。先确认线上到底用的是 20 还是 10 |
| `cfg_value` | **2.0** | 过高 → 发紧、失真、金属感；过低 → 含糊、吐字软。坏例重试时降到 1.2–1.6 并加步数 |
| `out_sample_rate` | 48000 | 仅推理输出，与训练的 16000 不冲突 |
| `load_denoiser` | False | 去噪器依赖 modelscope，不在 lock 里。**别指望它修训练数据的问题** |
| `retry_badcase` | True（3 次 / ratio 6.0） | 生产兜底。但**评测时必须关掉**，否则换种子会掩盖清晰度问题 |
| ref 模式 | reference-only | 带 control 前缀时**不要传 `prompt_text`**。combined 模式前缀跑到句中失效，情绪不对会被感知成「不清楚」 |

**如果线上清晰度不如试听页**，先逐项比对这六个参数，再怀疑模型 —— 十有八九是 `inference_timesteps` 或 `cfg_value` 不一致。

### 清晰度回归的判定

Phase 6 的 `intelligibility_1_5` 必须**单独打分**，不能和 naturalness 混在一起。三种常见误判：

- **自然度涨了、清晰度掉了 → 不算通过。** 念稿感消失但吐字变软，是很典型的过拟合表现
- **只看 `mean_cer` 会漏。** Whisper 对糊音常常「脑补」出正确文本，CER 看不出来，人耳一听就知道
- **只看总均值会漏。** 按 `kind=clarity` 分组、按 ref 语言分组看 —— 中英 ref 通常比目标语 ref 更容易吐字不清

---

## Phase 8 · 导出与上线

### 8.1 merge LoRA

```bash
uv run python -m voxft.lora.merge \
  --lora-dir $VOXFT_CKPT_ROOT/th_r2_e1/step_1000 \
  --out /root/autodl-tmp/merged/th_r2_e1_s1000
# --base 省略时用 .env 的 VOXCPM_BASE_PATH
```

支持官方嵌套 `lora_config`。页面「模型管理」Tab 同样能做。

### 8.2 许可红线：含 CC-BY-SA 数据的权重不得对外分发

**这条在上传/交付任何权重之前先看。**

我们在用的两个最好的泰语源都是 **CC-BY-SA-4.0**：`thai_ser`（THAI-SER）和 `Porjai-central`。Porjai 的 `pattani` / `khummuang` 子集更差，是 **CC-BY-NC-SA**，直接排除。

- SA 的触发条件是**「向公众分享改编物」**。模型权重算不算「改编物」**在法律上无定论、无判例**；CC 官方说过「模型若基于 SA 内容训练且公开发布，建议以同许可发布」，但那是**保守合规建议，不是法律要求**
- **红线：含 SA 数据训练的 LoRA 与 merge 后的完整模型一律不对外分发** —— 不传 HF、不随客户交付、不开源。只通过 API 交付合成音频。SA 不追及模型输出（除非输出实质复现了原音频，TTS 不会）
- 确实需要对外发布权重时，两条路：① 向版权方谈商业授权 —— **THAI-SER 的出资方是 AIS + DEPA，有明确的谈判主体**（VISTEC / airesearch）；② 该实验只用 Apache-2.0 / CC-BY 源（如 `speechcolab/gigaspeech2` th、`aishell3`）
- EU DSM 指令第 4 条 TDM 例外、日本著作权法 30-4 条、美国 fair use 任一成立时 CC 条件可被架空，且**单纯挂 CC 许可本身不构成 TDM 保留**。但这属于法务判断，不要自己下结论

**以上是保守合规立场，不是法律意见。重大决策请咨询法务。**

**ID 不触发这条红线**：本轮注册的 id 源全是 Apache-2.0（`gigaspeech2_id`）+ CC-BY-4.0（`fleurs_id`）+ CC0（`cv22_id`）+ 自有授权（`drama_id`），**没有 SA，没有 NC/ND**，所以含这些源训练的 id 权重原则上可以对外分发。（**存档**：vi 侧当年同样不触发这条红线，许可组合一致；ms 侧是 CC-BY-3.0 + CC-BY-4.0 + 自有授权，也没有 SA，但 CC-BY-3.0 要求署名 —— 两语种已于 2026-10-19 退出目标语种，依据见 [vi_id_support.md](vi_id_support.md) / [ms_support.md](ms_support.md)。）两个前提别忽略：

- 后续若按 [vi_id_support.md §3](vi_id_support.md) 核实进任何 SA/NC/ND 源（MagicHub 免费档通常就是 **CC-BY-NC-ND**，NC 禁商用、ND 禁演绎 = 微调，可用性 0），红线立刻对该语种恢复
- `gigaspeech2` 是 `gated` 数据集，Apache-2.0 说的是**数据**许可，不代替仓库条款；商用前自己读一遍并在页面同意

### 8.3 上传 HF

```python
from voxft.hub import sync
sync.upload_folder("/root/autodl-tmp/merged/th_r2_e1_s1000", "<org>/<repo>", kind="model")
```

或页面上传功能。**只有训练与盲听实际通过后才发布新权重**，且必须先过 [8.2](#82-许可红线含-cc-by-sa-数据的权重不得对外分发) 的许可检查。

### 8.4 交付前最后一遍

- [ ] **训练数据里没有 CC-BY-SA / NC 源；如果有，权重不对外分发，只走 API**（见 8.2）
- [ ] 用**生产链路的实际调用方式**（不是本项目试听页）跑一遍评测集，确认参数一致 —— 特别是 reference-only 模式
- [ ] **`kind=clarity` 分组用生产参数复跑过，`intelligibility` 没有相对基座退化**（试听页达标不代表线上达标，多半是 `inference_timesteps` 不同）
- [ ] 中英 ref → 目标语言的音色保持已 A/B 实测，没有相对基座退化
- [ ] 中英回放 case 没有灾难性遗忘
- [ ] 选定的 checkpoint 已备份（`cleanup_lora_runs` 只保留最新 5 次）
- [ ] 评测报告、盲听评分、`stats.json`、`mix.json`、`.plan.json` 一起归档，下一轮要对照

---

## 附录 A · 坑速查

**首尾裁切的 RMS 悬崖**：RMS 门限是相对该条「有声电平」（p99 帧 RMS）的。留白电平只要在有声电平 −24dB 以内（朗读档 `0.06` 的门限），`rms > thr` 覆盖整条、**一帧都裁不动，输出长度与输入完全一致** —— 看起来就是「裁静音没生效」，其实是生效了但门限没跨过。FLEURS 那个问题就是这个。所以朗读源改用 Silero VAD（faster-whisper 自带，无需额外权重），不受电平影响，实测首尾误差 ±0.04s。

**`min_silence_duration_ms` 用 500，不是 faster-whisper 默认的 2000**。默认值是给长音频分段设计的，会把不足 2s 的尾部底噪并进语音块，实测多留 1.0s。

**喂 VAD 的数组必须 float32**。float64 抛 ONNX `Unexpected input data type`，中断整轮加工。

**首尾低电平段必须连续 ≥0.25s 才裁**（`trim_silence(min_run=)`）。否则提高门限会啃掉词首清辅音（/s/ 只有 80–120ms）。

**Gradio 流式**：按钮必须直接绑定生成器函数。用 `lambda` 包一层会把生成器对象本身渲染进文本框。

**Gradio 下拉框**：`choices` 只在 `build_ui` 算一次。运行后变化的列表必须通过事件输出或 `Tab.select` 刷新。

**库函数不许 print**：UI 进程的 stdout 可能是已断开的 pty，`print` 抛 `[Errno 5] Input/output error`，会把一次**已经成功**的操作报成失败（filswitch 写完 2709 条清单后显示「失败」就是这个）。一律走 `progress` 回调，CLI 侧传 `progress=print`。

**`hf` 走镜像必须关 xet**：hf-mirror 只代理 HF API，不代理 xet 的 CAS 服务器（`cas-server.xethub.hf.co`），reconstruction 阶段会直连并报 401。项目入口自动处理，命令行要自己 export `HF_HUB_DISABLE_XET=1`。

**`datasets` 锁定 `<4`**：5.x 硬依赖 torchcodec，且其库与 cu124 torch 冲突。

**Common Voice 22 官方已从 HF 撤架**：数据源已切到社区镜像 `fsicoli/common_voice_22_0`（非 gated，脚本式数据集，流式下载带 `trust_remote_code`）。

**`thai_ser` 没有名为 `audio` 的列**（四路麦 `mic_clip`/`mic_con`/`mic_middle`/`mic_zoom`），必须靠 registry 的 `audio_cols` 映射，否则整个源在下载阶段被静默跳过。`mic_zoom` 是网络录音，不用。

**越南语句尾嘎裂声可能被首尾裁切啃掉**（推断，未实测；**vi 已于 2026-10-19 退出目标语种，此条转存档**，但裁切机制对任何低能量声调尾音都成立）：`ngã` / `nặng` 调带 creaky voice、能量低，RMS 门限可能把它当静音。现有防线是 `min_run=0.25` + 朗读源走 VAD。**验收时重点听以这两个声调收尾的句子**；真出现就调低该源的 `edge_trim_ratio`，**不要动 `min_run`**（调它会啃词首清辅音）。

**控制前缀守卫挡得住越南语，挡不住印尼语**：`_assert_control_lang` 拦泰文区 + 越南语专属字符（`ơ ư đ` 与带声调元音），但印尼语是纯 ASCII 拉丁字母，与英文无法区分。素材导入页的控制描述框别往里填印尼语，这条只能靠标注规范。

**`yodas_th` 的 session**：`utt_id.rsplit("-", 3)[0]` 保留可能含 `-` 的完整视频 ID。`speaker_id` 是视频级近似身份，不作 ref 依据。无原始连续时间关系就不拼接短句。

**视频容器解码走 PyAV**：soundfile 读不了 mp4/mkv，qc 组已显式声明 `av>=12`（本来就是 faster-whisper 的传递依赖）。不引入系统 ffmpeg 依赖，本地 macOS 与远程行为一致。

**`third_party/VoxCPM/` 是官方 submodule，只读，禁止直接修改。** 训练入口是其内 `scripts/train_voxcpm_finetune.py`。升级：

```bash
cd third_party/VoxCPM && git pull origin main && cd ../..
git add third_party/VoxCPM && git commit -m "bump VoxCPM submodule"
```

**旧数据迁移**：所有旧 processed/mixed 清单重加工为新名称（`_v2`），不要混用旧伪说话人 refs、拼接音频或拼音污染正文。旧 THAI-SER manifest 缺 `turn_type` 会明确报错，需在远程重下。原能量差字段改名 `energy_range_db`，启用旧 `min_snr_db`/`min_f0_std` 硬筛会报错。

---

## 附录 B · 全流程 checklist

### Phase 0 环境
- [ ] `uv sync --group qc` 成功，`uv run pytest` 全绿
- [ ] submodule 已初始化（`third_party/VoxCPM/scripts/train_voxcpm_finetune.py` 存在）
- [ ] `.env` 的 `VOXFT_DATA_ROOT` / `VOXFT_CKPT_ROOT` / `HF_HOME` 都指向大盘
- [ ] large-v3 与基座已预取到本地
- [ ] UI 用 `nohup`/tmux 启动，stdout 没挂在会断的终端上

### Phase 1 评测集
- [ ] 80–100 条 / 语言，未见演员未见会话
- [ ] 覆盖矩阵齐全（ref 语言 × 前缀 × 角色 × 语体 + 中英回归）
- [ ] **`kind=clarity` 的压力用例占 20–30%**（长句 / 难词 / 数字英文混排 / 快语速 / 耳语 / 句尾辅音）
- [ ] 母语者已确认台词与预期读法
- [ ] R0 基线报告已生成并归档

### Phase 2 数据
- [ ] 每个源先 `--max-samples` / `--max-items` 试跑
- [ ] `thai_ser` 只留 impro + agreement ≥0.7，不用 `mic_zoom`
- [ ] `filswitch` 音频真的下载到了（不是「分片下载成功」就算）
- [ ] **ID**：`gigaspeech2_id` 试跑确认了 `audio` 列存在且时长落 3–30s（若是 `path`+`start`+`end` 长音频切片形态则停用本源，不改 `_load_audio` 猜时间戳）
- [ ] **ID**：`fleurs_id`、`cv22_id` 的 config 名试跑确认存在，`gated` 源已同意条款且 `HF_TOKEN` 有效
- [ ] 自建清单的 `speaker_verified` 是人工核实过的，不是「有一列 speaker」
- [ ] `session` 填了，同一集的切片同 session
- [ ] 中英同人 ref 以 `reference_only=true` 独立行导入
- [ ] 哭腔/带笑/讽刺等复合指令手写了 `control_zh`+`control_en` 且 `control_verified=true`
- [ ] 控制前缀只有中英文，没有目标语言
- [ ] holdout 已钉住（`holdout.json`）

### Phase 3 加工
- [ ] 输出用新名称（`_v2`），没覆盖旧音频
- [ ] `drop_*` 计数已逐项看过，没有异常大的
- [ ] 表演源 `edge_vad=false`、朗读源 `edge_vad=true`
- [ ] **抽听 10 条产物，词首清辅音（/s/ 80–120ms、/h/ 40–80ms）没被裁掉，句尾辅音完整**
- [ ] **响度对齐后抽听，喊叫条目没有削波（`gain ≤ 0.97/peak` 生效）**
- [ ] `with_control / rows` ≥ 25%
- [ ] `with_ref_audio / rows` ≥ 30%
- [ ] `cross_language_refs` 的实际值已记录（为 0 就明确标注「跨语言 ref 未验证」）
- [ ] `emotions` × `speakers` 没有「某情绪只来自某一人」
- [ ] `holdout_pinned_records` 与预期一致

### Phase 4 混合
- [ ] `actual_duration_share` 与请求占比的差异可解释
- [ ] `max_exposure` ≤ 3
- [ ] train/val 无跨源泄漏（混合器已拒绝，确认没报错）
- [ ] 混合后的 `with_control` / `with_ref_audio` 仍达标
- [ ] 发音锚点源在 5–10% 区间（够修发音，又不至于带回念稿感）

### Phase 5 训练
- [ ] 基座 `dit_config.cfm_config.training_cfg_rate == 0.1`
- [ ] `enable_dit: true`
- [ ] `sample_rate: 16000` / `out_sample_rate: 48000` 没被搞混
- [ ] 预检无 error，两条警告已处理或已记录原因
- [ ] run 名带轮次和变量，不覆盖旧 run
- [ ] `.plan.json` 与实际 GPU 数、清单条数一致
- [ ] 1 epoch 起步
- [ ] 好的 checkpoint 已备份（自动清理只留最新 5 次）

### Phase 6 验收
- [ ] 同一份评测集、同样的 seeds、`retry_badcase=False`
- [ ] LoRA 加载校验通过（loaded 非空、skipped/missing 为空）
- [ ] A/B 是同模型禁用/启用适配器，不是两次独立加载
- [ ] ≥2 名母语评审盲听，四项分数都填了
- [ ] 按 ref 语言 / 角色 / 情绪分组看过，不只看总均值
- [ ] **`intelligibility_1_5` 按 `kind=clarity` 单独分组看过，没有和 naturalness 混判**
- [ ] 中间 checkpoint 也评过，不只是 `latest`
- [ ] 结论是「自然度/情绪改善 **且** 清晰度/音色不退化」才判通过
- [ ] **ID**：表演档为 0 时，结论**没有**写成「情绪表现力改善」或「去念稿感达成」
- [ ] 读含 vi 行的**历史报告**时，没把 vi 的音节级 WER 和 id/tl/en 的词级 WER 横向比（vi 已退出 `WER_LANGS`，现行集不会产出该列）

### 清晰度专项（逐条对照正文五条战线）
- [ ] **入口**：BGM / 混响 / 口水音重的条目在素材导入阶段已丢弃，没有指望训练修
- [ ] **入口**：`yodas_th` 的 `dnsmos_overall ≥ 3.2` + `grade_avg ∈ {S+, S}` 没被放宽
- [ ] **入口**：高价值子集（评测集难词、强情绪条目）的训练文本已人工校对
- [ ] **裁切**：`min_run=0.25` 和 `tail_keep=0.3` 没被改动
- [ ] **裁切**：`thr` 的 `noise × 3.0` 上限还在（防止把轻声语音当底噪裁掉）
- [ ] **裁切**：（**存档，仅重启 vi 时适用**）抽听过以 `ngã` / `nặng` 调收尾的句子，句尾嘎裂声没被当静音裁掉（出现就调低该源 `edge_trim_ratio`，不动 `min_run`）
- [ ] **响度**：`target_dbfs=-24` 只对 `speaker_verified=true` 生效，未知身份没被统一调响度
- [ ] **训练**：`max_grad_norm=1.0` 没关，loss 曲线无未处理的尖刺
- [ ] **训练**：日志实际样本数与清单条数的差距可解释（`max_batch_tokens=8192` 长度过滤）
- [ ] **推理**：线上 `inference_timesteps=20`、`cfg_value=2.0`，与试听页一致
- [ ] **推理**：线上带 control 前缀时走 reference-only，没传 `prompt_text`

### Phase 7/8 迭代与交付
- [ ] 本轮只改了一个因素
- [ ] 生产链路用 reference-only 模式（带 control 时不传 `prompt_text`）已确认
- [ ] 生产链路六个推理参数逐项与试听页比对过
- [ ] merge 后的模型在生产调用方式下复测过
- [ ] 报告 + 评分 + stats + mix + plan 一起归档
