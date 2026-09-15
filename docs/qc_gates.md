# 质检指标与门禁总表

eval 报告（`voxft.eval`）里出现的全部指标，逐条标明：是 A/B **门禁**还是只作**参考**、
阈值、以及阈值是哪次实测标定的。原则：**能自动判的都已被盲听交叉验证过；没验证过的一律参考值**，
自然度/情绪/语调地道性永远以母语盲听为准（`eval.BLIND_LOSS_MARGIN=2` / `BLIND_MIN_LOSSES=3`，
12-18 条量级上挡单条波动用的最小差值门槛，实测依据：id 0胜/14平/1负、zh 0胜/2平/1负 都不算退化）。

## 指标 → 门禁/参考 → 阈值 → 标定依据

| 指标 | 角色 | 阈值 | 标定依据 |
|---|---|---|---|
| `cer` / `mean_cer` | **门禁**（A/B 相对） | ΔCER > `--noise` 红线（CLI 默认 0.005）；±0.04 以内一律当噪声 | 同一份配比（fleurs_ms=17）跑两轮 ms CER 差 0.042（0.0630 vs 0.1053）——纯 run 间方差；共享 RNG 下改任一权重挪动全部抽样 |
| `mean_cer_non_numeric` | **门禁**（数字类单列后看这列） | 同上 | Whisper 会把口播数字词归一成阿拉伯数字/货币符号：base 与 ckpt 输出都被转写成 "1,500 pesos"，两边 CER 同为 0.588，是 ASR 假象不是 TTS 差异。剔除后 base 0.0022 / r2 0.0030，已饱和 |
| `wer` / `mean_wer` | 参考 | 无 | 量纲按语种：tl/en/id/ms 词级可横比，vi 是音节级，th 词间无空格只有 CER |
| `suspected_truncation` | 诊断 | 少读/漏尾：归一化后 <0.6× 参考或尾部 8 字符不匹配 | 定义冻结（改了无法与前四轮数字可比）；ASR/同义转写也会触发，必须听音确认 |
| `over_read` / `len_ratio` | 诊断 | 归一化文本 >1.4× 参考 | 门限移植 OmniVoice `api.py:6529-6536` 长文本档；与少读分开，r1 的 vi 跑飞就是这档 |
| `metallic` / `metallic_score` | **永久参考值** | 不作门禁 | 168 条盲听标注样本定案：score 对人工 noise 标注 **AUC=0.060（反相关）**——唯一 noise=True 样本 score 0.0694，比 167 条干净样本里 157 条都低；对自然度 ≤3/≤4 也只有 0.46/0.33。阈值是在 OmniVoice 后处理音频上标定的，与裸输出频谱形态不同 |
| `low_snr` / `floor_separation_db` | **永久参考值** | 不作门禁（生产门限 18dB 仅作对照） | 同批样本 **AUC=0.509（纯随机）**；167 条人工判干净样本上误报 48 条（28.7%）。OmniVoice 自己 48/48 全判 low_snr + noisy_reference——主因是**参考音频噪底**，不是生成问题 |
| `speaker_sim` | 参考（同 ref 下 A/B 相对比较） | 无门禁 | WavLM X-vector（`microsoft/wavlm-base-plus-sv`）：同人 0.995-0.998 / 跨说话人 0.32-0.62（均值 0.514），间隔 0.48。实测输出-vs-ref 0.86-0.94（tl ref 0.9421 / zh 0.9250 / en 0.8587）。**与生产 ERes2NetV2 门限 0.45 刻度不可互换**。MFCC 版已废弃（0.985-0.996 饱和无动态范围） |
| `chars_per_sec` | 参考 | 无 | **必须配 speech_ratio 一起看**：r2 低 12.7% 乍看是说慢了，拆开有声段字/秒只差 1.7%，差的是垫静音 |
| `speech_ratio` | **门禁**（A/B 相对） | 下降 > 0.05 红线（`runlog.SPEECH_RATIO_DROP`） | r2 实测缺陷 -0.10（0.94→0.84），取一半作门槛 |
| `audio_sec` | **门禁**（duration_inflation） | mean 涨幅 >10% 且 \|ΔCER非数字\| ≤0.01 → 红线（`DURATION_INFLATION_RATIO` / `DURATION_INFLATION_CER_TOL`） | 盲听 8 条「B 更差」里 6 条是「B 音频明显变长而 CER 没变」（最极端 ms_manglish 1.76s→3.52s、CER 反降 0.094→0.000）；r2 总时长 +12.3% |
| `head_silence_sec` | 参考 | 无 | 头部没有缺陷证据，只量不设门（与 pipeline 只裁尾不裁头一致） |
| `tail_silence_sec`（`mean_/p90_tail_silence`） | **门禁** | p90 绝对增量 >0.10s，或 ckpt p90 >0.5s → 红线（`TAIL_SILENCE_P90_DELTA` / `TAIL_SILENCE_P90_MAX`） | r2 实测尾静音均值 0.084s→0.292s（p90 0.42s，0/84 越 0.5s）；0.5s 对齐官方训练数据尾静音上限。根因已归因并修复：FLEURS 加工样本尾静音 p50 0.30s 教会模型垫尾 → 输出 p50 0.32s（base 0.084s），pipeline 现加 `max_tail_silence_sec=0.15` 硬上限，r4 用修后数据重训 |
| `spectral_rolloff_99` | 参考 | 无 | 基座输出 16kHz 带宽装 48kHz 容器：中位 7262Hz、16kHz 以上能量 max 0.0035%（砖墙），cfg/steps 全网格无效、LoRA 结构上碰不到 |
| `band_ratio_2_8k` | 参考 | 无 | 带内明亮度，唯一可能随训练数据变的带宽相关量：base 中位 0.162，r2 配对 Δ 均值 -0.0143、71% 对子变暗（修在数据侧） |
| `f0_std_st` | 参考 | **不作通过门限** | 含泰语声调和清浊音误差，不代表自然度；F0 不是越高越好 |
| `similarity` | 诊断 | 无 | 文本相似度，与 CER 同源受 ASR 假象污染 |

时长类三条门禁（duration_inflation / tail_silence / speech_ratio）在 `runlog._duration_gates`，
overall 与 by_lang 都查；旧报告缺聚合字段时对应检查跳过、不误报（tail_silence 的 0.5s
绝对上限只依赖 checkpoint 自己的 p90，base 缺字段仍生效）。**duration_inflation 触发且
tail_silence 同向时基本可定案为垫尾缺陷，不需要母语者**；其余指标与盲听冲突时以盲听为准。

## 线上 seed 事实（2026-09-15 读两个仓库源码核实）

- 主调方每请求必传 seed（`dubbing_intelligence_service` `backends/voxcpm.py:812-822`）。
  默认 `VOXCPM_SPEAKER_STABLE_SEED=true` 时是 **per-speaker stable seed**：
  `sha256(speaker + model_id + cfg + steps + …)` 派生、**不含文本**（`backends/voxcpm.py:438-463`），
  同一说话人的所有 cue 共享同一个 seed——**首 take 时每个 speaker 只生活在一个 seed 上**。
- **重试路径大多换 seed**：模型内 retry_badcase **+1**（`voxcpm2.py:734`）、prompt-leak +1、
  speaker-mismatch +1、text-regen **+7**、best_of 候选 **+1009**、主调方整轮重试 crc32 重派生。
  **例外：metallic 质量重试不换 seed**，只降 cfg 加 steps（`api.py:9184-9196`）。
- 含义：**离线量的跨 seed 方差 = 线上重试把好 take 换成坏 take 的风险**；量它要用相邻偏移
  seed（base/+1/+7）而不是无关大跨度 seed。首 take 稳定性该量的是跨 cue 一致性
  （同 speaker 同 seed 不同文本）。
