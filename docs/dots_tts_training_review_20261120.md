# dots.tts 训练与微调方法对 VoxCPM2 的可迁移性（2026-11-20）

## 结论与边界

**可以借鉴训练目标和分阶段方法，不能直接搬训练脚本或超参。** 当前最佳仍是 r50，生产配置不变；
本次只完成官方论文、源码及本项目的静态对照，未运行 GPU、未训练、未生成音频，也未修改实验计划。
VoiceStudio 相关其他引擎的训练方法补充见
[voicestudio_training_transfer_20261120.md](voicestudio_training_transfer_20261120.md)。

对当前项目的候选优先级：

1. **EOS 停止头与 LM 解耦的小实验**：最贴近句末专名重复等已登记风险，但仍属待验证假设。
2. **SOAR 声学 DiT 自纠**：作为声学轨迹稳定性的独立研究分支，不解 EOS。
3. **MeanFlow 教师蒸馏**：仅在质量候选稳定后用于降步数，不作为当前质量修复。
4. 当前 vi/en 备料、准确转写、MMS-FA 切分和母语评审主线不变；不因本文重新扫描普通 LoRA 超参。

## 方法适配总表

| dots.tts 方法 | 与 VoxCPM2 的适配性 | 对当前项目的价值 | 处置 |
|---|---|---|---|
| 普通 SFT / Accelerate 训练循环 | 都是文本到 latent 的监督训练，但模型、模板和损失不同 | 增量有限；本项目已有训练、恢复和监控链 | 不搬代码 |
| SOAR 自纠 | 都有 flow-matching DiT 与干净 latent 监督，原理可迁移 | 可能降低声学轨迹累积误差；不直接修停止 | 先做无训练误差诊断，再决定是否立项 |
| CFG-aware MeanFlow 蒸馏 | VoxCPM2 已有 mean_mode、dt 通道及 JVP 目标，但实现不同 | 降步数、融合 CFG 分支，偏速度而非质量 | 质量稳定后单独立项 |
| 独立 EOS 头 | 两边均有停止头，但梯度路径和 LoRA 覆盖不同 | 可用于设计更聚焦的句末多读实验 | 只作为新假设，不宣称已验证 |
| 双 ASR / UTMOS / 带宽 / 声纹过滤 | 与模型结构无关 | 多数是现有能力或计划；不是新增收益证据 | 不重复建设，不新增未标定硬门 |
| AudioVAE 重训 / 全历史条件架构 | 需要改 codec 或主架构 | 超出 LoRA 微调范围 | 不混入当前项目 |

## SOAR：可迁移的是自纠目标，不是超参

官方论文 §2.6.3 / §3.2.3 的核心流程：

- 冻结 LLM、语义编码器、AudioVAE 和 speaker encoder，只更新声学 DiT。
- 保留普通 on-trajectory flow-matching loss。
- 用当前 CFG-guided DiT 做一次 **stop-gradient Euler rollout**，得到带自身推理误差的 off-trajectory 状态。
- 对这些状态重新加噪，并以干净 latent 终点构造辅助纠正目标。
- 目标是减少多步 ODE 推理时误差逐 patch 累积；不需要奖励模型、人类偏好或外部声学教师。

VoxCPM2 也通过 DiT 对 AudioVAE latent 做 flow matching，因此可以从冻结的 **base + r50** 出发，
冻结 LM、VAE、stop head，仅用新的 DiT LoRA 探索该目标。SOAR 本身不增加生产采样步数，但训练成本和质量收益均需实测。

不能直接复用的部分：

- dots 公式采用 noise→data；VoxCPM2 当前训练/推理的时间方向不同，还带 sway sampling 和 CFG-zero*。
  状态、速度符号、时间调度及 CFG rollout 必须按 VoxCPM 实际路径重写，不能复制公式代码。
- dots 的大池数据、50K 更新、全局音频 batch 和学习率不属于当前小数据 LoRA 配方，不能照搬。
- 本次官方开源树中找到普通 SFT 和 MeanFlow 训练入口，**未找到完整 SOAR 训练入口**；论文公式不是现成实现。
- SOAR 冻结 LM 和停止头，**不能直接解决句末专名重复或短 cue 不停**。即便官方 WER 有改善，也不能外推到当前病灶。

建议先做无训练诊断：在冻结 r50 上构造 off-trajectory 状态，比较其纠正误差与普通 on-trajectory 误差。
若没有可测差距，不开训练；若立项，冻结单变量、数据、训练 seed 和预算，按现行短句→开发集→一次性留出流程验收。

## MeanFlow：有现成基础，但不能只改配置

本项目 VoxCPM2 源码已包含：

- `mean_mode`；
- DiT 的 `dt` / `delta_time` 条件通道；
- 基于 JVP 的 MeanFlow 训练目标。

位置：

- `third_party/VoxCPM/src/voxcpm/modules/locdit/unified_cfm.py`（`sample_r_t`、`compute_loss`）；
- `third_party/VoxCPM/src/voxcpm/modules/locdit/local_dit_v2.py`（`delta_time_mlp`）。

但官方 VoxCPM2 配置未启用 mean mode，现有代码也是直接 MeanFlow 目标，**不是 dots 的冻结教师＋CFG 融合蒸馏**。
不能对现有 r50 只改配置就当作低步数模型使用。

dots 开源实现中可参考的工程点：

- 教师 frozen/eval/no-grad，学生从兼容权重初始化；
- 教师多步轨迹产生区间平均速度目标；
- CFG 融入教师目标，学生推理只需条件分支；
- anchor sample 用于稳定训练。

它适合将来的独立加速项目：以质量合格的 r50 或后续候选为教师，训练低步数学生，再测完整生成与 QC。
dots 自己的 MF4 相对 SOAR 已表现出一定相似度代价；r50 当前已有音色成本，因此不得把降步数与质量改动混成一轮，
也不得用厂商 RTF 代替 4090/3090 的完整链路测量。

dots 默认 YAML 是 LJSpeech smoke 配置，不是论文完整复现；其默认教师步数、预算等与论文设置并不完全一致。

## EOS：架构对照产生的新实验假设

dots 普通训练中对 EOS 使用 `eos_proj(llm_hidden.detach())`，停止头与 LM 表示在梯度上解耦。
VoxCPM2 当前 `third_party/VoxCPM/src/voxcpm/model/voxcpm2.py` 则由未 detach 的 `lm_hidden` 计算 `loss/stop`；
现有 LoRA 的 LM/DiT/proj 目标也不包含独立的 `stop_proj/stop_head`。

因此当前停止行为主要通过 LM 间接受影响。可针对句末专名续写设计一个独立小臂：

- 冻结声学 DiT；
- 使用准确文本、短句、句末专名及完整边界样本；
- 比较继续经 LM 反传 stop loss，与只训/解冻 stop_proj + stop_head、或 detach LM 表示的差异；
- 不再只增加 `loss/stop` 权重；
- 直接验收尾部多读、提前停止、正文完整性及多 seed 稳定性。

这是**待检验的新机制**，不是 dots 已证明的修复；dots 自己的公开短文本 issue 也存在 EOS/额外发声风险。
实验不能改生产 stopping 参数，也不能用改写台词或标点规避失败。

## 数据与训练工程：多数不是新增方向

dots 论文描述了双 ASR 一致性、UTMOS、有效带宽、片段内声纹变化、分离和说话人过滤。对照本项目：

- 双 ASR 一致性过滤已在当前计划中，伪标注源 CER≤0.05，继续执行即可。
- UTMOS 已有实现与使用记录，不因 dots 再新增全局硬门。
- 有效带宽、片段内声纹方差只能先作审计指标；短音频、跨语种和表演片段未标定前不得自动拒绝。
- dots 不做显式语种重加权的前提是大规模训练池，不能照搬到当前低资源联合 LoRA。
- AudioVAE 的 WavLM 语义教师及全历史条件属于 codec/架构项目，不是当前 LoRA 微调项。
- **不使用 dots 合成音频给 VoxCPM 补量。** Apache-2.0 许可层面没有本文所知的输出训练禁令，
  但合成音频仍不是经授权真人语料，且违反本项目现行数据纪律；不能以许可较宽松绕过身份、质量和来源审查。

## 建议的后续顺序

1. 不打断当前 vi/en 数据主线；先把 EOS 解耦和 SOAR 分别写成有明确假设、预算和停止条件的实验方案。
2. EOS 实验直接围绕已冻结的句末专名/短 cue 风险；SOAR 先做无训练 off-trajectory 误差诊断。
3. 任一训练臂均从固定基线出发，沿用现行分语种、≥5 seed、按句聚类、无语种退化及母语盲听要求；
   开发集反复择优问题仍须通过新开发集解决。
4. MeanFlow 只在质量候选稳定后单独立项；速度验收测完整 WAV + QC 的冷/暖启动、P50/P95 与峰值显存。
5. 所有新方向均不自动训练、不自动晋级、不切生产。

## 主要来源

- [dots.tts 官方仓库](https://github.com/studio-dots-ai/dots.tts)
- [dots.tts 技术报告](https://arxiv.org/html/2606.07080)（§2.6.3、§3.2.3、§3.2.4）
- [普通训练入口](https://github.com/studio-dots-ai/dots.tts/blob/main/scripts/train_dots_tts.py)
- [MeanFlow 蒸馏入口](https://github.com/studio-dots-ai/dots.tts/blob/main/scripts/train_dots_tts_meanflow.py)
- [dots EOS / 普通训练模型实现](https://github.com/studio-dots-ai/dots.tts/blob/main/src/dots_tts/models/dots_tts/core.py)
- [reference-only 短文本公开 issue #17](https://github.com/studio-dots-ai/dots.tts/issues/17)
- 本项目当前计划：`TODO.md`；音色边界：`docs/timbre_v1_20261119.md`。
