# 差参考音鲁棒性：阶段 0 冻结口径（2026-10-12，2026-10-19 按语种收缩修订）

> **2026-10-19 修订**：vi/ms 退出目标语种（只剩 th/tl/id + zh/en 回放），本文的抽样口径、
> case 文件名与生成量已同步改写；**冻结的实现口径（profile 参数、SIR 定义、守卫、确定性）
> 一个字未动**。GPU 仍是暂停状态，阶段闸门原文保留。

## 目标与边界

线上参考音不总是干净的：已观测到多人声参考（T02 听评）与低 SNR 参考（`mtg2i2m` 首句
4.27dB）。**目标：传入质量较差的参考音时，生成音频仍保持较好的内容与音质。**
用户（2026-10-12）明确允许为此牺牲部分参考音一致性——speaker_sim 全量记录但放宽门禁，
只记录、不作否决。

- **只动模型侧**：训练数据增强（降质 ref 配对干净 target）。训练打包
  `[103 ref 104][text][101 target 102]` 的 loss 只算 target
  （`third_party/VoxCPM/src/voxcpm/training/packers.py:process_tts_data_with_ref`），
  给 ref 加噪不会把噪声教进输出目标，反而教模型「条件脏也要出干净音」。
- **不改** OmniVoice/DIS 推理链路，不引入 ZipEnhancer，不动 `third_party/VoxCPM`，
  不开 `enable_proj`。
- 分阶段：阶段 0 全部本地 CPU（本轮完成）；阶段 1（GPU 基线量化）与阶段 2/3
  （A/B 训练验收）进入前都要用户确认。

## 用户圈定的两类退化（不加第三类）

1. **多人声/串音（babble）**：他人语音低电平混入参考，按 SIR（信干比）分两档。
2. **发闷/频带受限（lowpass）**：零相位低通，按截止频率分两档。
3. 另加一个**组合档**（先串音再低通）——真实差参考常两者叠加。

明确不加背景噪声/混响退化；用户没圈的不做。

## 冻结 profile 矩阵（`src/voxft/data/ref_degrade.py: PROFILES`）

| profile | kind | 参数 | severity |
|---|---|---|---|
| `babble_sir15` | babble | SIR=15dB | moderate |
| `babble_sir8` | babble | SIR=8dB | severe |
| `lowpass4k` | lowpass | 截止 4000Hz | moderate |
| `lowpass3k` | lowpass | 截止 3000Hz | severe |
| `combo_babble8_lp3k` | combo | SIR=8dB + 截止 3000Hz | severe |

冻结实现口径（改动必须连同本文档一起改）：

- **SIR 定义**：`20*log10(rms(ref_voiced) / rms(interferer_scaled))`。ref 侧取**有声段**
  RMS——帧 RMS（`pipeline._frame_rms` 同口径：25ms 帧、hop=帧长/2）p40 以上帧覆盖的
  样本；干扰侧取拼到等长后的整段 RMS。按有声段定标是因为参考首尾静音会稀释整段 RMS，
  同一标称 SIR 下短参考的实际串音会更响。干扰源确定性选 1–2 条，随机起点循环/截断到
  等长后按目标 SIR 缩放相加。
- **身份守卫**：干扰源**拒绝** speaker 等于 ref_speaker 或 origin_audio 等于 ref_origin
  的条目（过滤后为空抛 ValueError，不静默降级成无干扰）。训练侧干扰池只取同一
  train split 内 `speaker_verified=true` 的行，val 行绝不入池。
- **低通**：`scipy.signal.butter(4, cutoff, fs=sr)` + `filtfilt` 零相位（scipy 是
  librosa 传递依赖，不新增依赖）。
- **电平**：所有 profile 末尾把整段 RMS 对齐回原始 ref（只改「质」不改电平，生产侧
  本就有电平对齐），峰值 >0.99 时整体回退到 0.99 防削波。长度不变，`ref_duration`
  不失效。
- **确定性**：同一 (wav 内容, profile, seed) 逐字节一致；随机性全部来自
  `np.random.RandomState(seed)`。批量入口的 per-(文件, profile) seed =
  `sha256(f"{seed}:{src_sha256}:{profile}")` 前 8 字节转 int（`derive_seed`），
  不依赖调用顺序与进程状态。
- **sidecar**：批量产物追加写 `<out>/degradations.jsonl`（读旧合并后原子写回，utf-8），
  记录 source sha256 / profile / seed / 实际参数（sir_db、cutoff_hz、
  interferer_origins、level_gain_db、clipped）/ dst。原文件不动。

## 阶段 1 评测集口径（已冻结，wav 待远端生成）

- 来源：`eval_cases/omnivoice_prod_v2.jsonl`（2026-10-19 过滤后 118 条），筛 lang ∈ {th,tl,id}，
  seeded(42) 每语种抽 6 条 = **18 case**；ref 沿用各 case 原 ref（zh/en/tl 分布不变，
  抽样后实际覆盖 **9 条唯一 ref**，ref_lang 分布 zh 5 / en 7 / tl 6）。
- 条件矩阵：clean（对照）+ 5 个 profile = 6 份
  `eval_cases/refrob_20261019_<condition>.jsonl`。case_id 与原 case 相同，跨条件靠
  报告间 (case_id, seed) 配对；非 clean 条件的 `ref_audio` 指向
  `refs_degraded/<condition>/<原 basename>`（相对路径），并加 `ref_audio_original` /
  `ref_degradation={profile,severity}`；clean 条件仅追加 `ref_audio_original` 与
  `ref_degradation=null`，其余逐字段一致。
- 降质 wav 只落远端（CPU 即可）：在远端对 9 条被引用 ref 跑
  `python -m voxft.data.ref_degrade --manifest <refs.jsonl> --out eval_cases/refs_degraded
  --pool <干扰池.jsonl> --seed 42`（sidecar 留 `degradations.jsonl`）。
- 规模：18 case × 6 条件 × 3 seed × 2 模型（base/r8）= **648 次生成**（原五语种口径是
  30 × 6 × 3 × 2 = 1080），**低于**历史轮次的 990–1500 区间——省下的 40% 正是 vi/ms 那两档，
  不是把预算摊薄到更少样本上（每语种仍是 6 条）。评测口径沿用生产参数：
  cfg 1.8 / 20 步 / 禁重试 / `--shard` 并行后 `--merge`。
- speaker_sim **对原始干净 ref 计算**（衡量「脏参考下还保住多少身份」）。

### 旧 20261012 集已作废，且新集不是它的子集 ⚠️

- 旧 6 份 `eval_cases/refrob_20261012_*.jsonl`（30 case = th/tl/vi/id/ms 各 6）已删除，
  重建为 `refrob_20261019_*.jsonl`（18 case）。**阶段 1 从未跑过 GPU，旧集没生成过任何音频**
  ⇒ 没有对照损失；需要旧文件时从 git commit `b8a768e` 取回。
- **新 18 条不是旧 30 条的子集**：th 与 tl 的抽样与旧集完全相同（各 6/6 重合），但
  **id 只重合 2/6** —— `select_cases` 用同一个 `Random(42)` 顺序抽样，删掉 vi 那一档挪动了
  id 的随机流。⇒ **不能拿新旧两集互比**，任何跨版本对照都必须重新生成两侧音频。
- `scripts/build_degraded_ref_cases.py` 的 `LANGS` 已同步改为 `("th","tl","id")`。

## 阶段 2/3 训练侧单变量设计

- 数据：`scripts/degrade_train_refs.py` 基于 r8 同源配比生成 `<name>_refrob/`——
  train.jsonl 带 ref 的行按 seed 选 **50%** 把 ref 换成降质副本
  （profile 按行确定性分配，五档各约 20%），行内记 `ref_audio_original` /
  `ref_degradation`；**val.jsonl 逐字节复制**，mix.json 原样复制，统计写
  `refrob_stats.json`。行集、split、文本、target 音频、配比、曝光全部不变，
  原数据集目录一个字节都不改。
- 两组同预算训练（沿用 r10/r11 小预算口径：train_hours=12、1 epoch、从官方基座新训、
  r=64/alpha=64/dropout=0.05、enable_dit、proj 关闭、lr 1e-4、save/valid 250）：
  **A 组**干净 ref 控制组（同预算重跑 r8 配方）、**B 组** 50% ref 降质，r8 本身作
  额外参照。**不续训 r8。**
- 验收（全部同 case 集同 seed 组）：
  - **干净回归**：完整 v2（**118 case** × 5 seed）A/B/base/r8，分语种红线 ΔCER>0.05、
    盲听负−胜≥2 且负≥3；**红线只判 th/tl/id 三个目标语种**（zh/en 回放段照旧记录，
    用于防灾难性遗忘），任一语种退化即整轮不通过。⚠️ 过滤前是 182 case，旧报告里
    含 vi/ms 的**总体聚合**不可复现，只有 th/tl/id 的 `by_lang` 段可跨这次过滤对照。
  - **降质套件**：阶段 1 的 6 条件集上 B 相对 A/r8 的内容指标（CER/漏尾/多读）改善
    为主目标；配对抽样做声学盲听。
  - **一致性取舍**：speaker_sim 全量记录；干净条件下 B 相对 A 的 SIM 回退超过 run 间
    噪声（±0.009）要在结论里写明；降质条件下 SIM 下降属预期，只记录、不作否决。
    若 B 在降质条件下 SIM 反而优于 A 且内容指标改善，即是目标形态。

## 阶段闸门

- **基线无可测退化则不训练**：若阶段 1 实测降质 ref 下 base/r8 的内容/声学指标没有
  可测退化（相对 clean 条件的差异在噪声内），停止并向用户报告，不进入阶段 2/3。
- 每个 GPU 阶段进入前向用户确认；不通过则不晋级、保持 r8，产物与报告归档不改写。

## 明确不做

- 不改 `third_party/VoxCPM`；不动推理链路与生产参数；不开 `enable_proj`。
- 不加背景噪声/混响退化类型；不为凑比例伪造身份或用未验证说话人做干扰池。
- 不用合成成片 / `t_*` 测试数据做干扰源；不降质 target 音频；不动 val 集。
