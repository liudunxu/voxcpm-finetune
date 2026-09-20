# ASR 文本 QC：当前路由与分语种选择评估

核查日期：2026-09-20。源码：DIS `aebf175`、OmniVoice `d984c34`、
本项目 `b8d8d29`。这是本地源码与已有产物审计，不是线上配置核验。

## 结论

**不是所有 QC 都固定 Qwen3，也尚未根据分语种准确率配置首选后端。**
建议保留“一个主 ASR＋可疑项换后端复核”；先用已有 WAV 做分语种对照，
证实某语言的误报/漏报与耗时更合适后，再仅切该语言。
目前没有足够证据把 TH 全量改为普通 Whisper large-v3。
策略审计阶段未改运行时代码、默认后端、缓存、r8、部署或训练，也未新增 TTS/ASR 请求。
后续 TH 成片诊断暴露 Whisper GPU 故障，已另修生命周期保护；见第 5 节，不混作选型收益。

## 1. 当前实际策略

| 环节 | 当前默认与例外 | 源码入口 |
|---|---|---|
| OmniVoice 合成输出文本 QC | 全局默认 Qwen3；TH/VI/ID/MS/TL/EN 均在支持列表。不支持的语言回退 Whisper；全局参数可改为 Whisper。当前没有分语种首选或请求级 QC 后端选择。 | `api.py:628`、`api.py:6574` |
| 是否运行合成文本 QC | 请求 `output_text_qc` 可覆盖；否则根据语言 profile/env 和最少 3 个 QC token 判断，不是每条都跑。泰语按字符粒度，不能把 3 token 理解为 3 个词。 | `api.py:6562`、`language_config.py:18` |
| 合成端 prompt 泄漏复核 | 中文 prompt＋非 CJK 目标，且时长异常或多词时，追加同一 ASR 的无文本提示、自动语言转写；不是另一模型的独立确认。 | `api.py:6706` |
| DIS 后处理后的最终 QC | 仅检查短句、关键数字/否定、已有内容风险或明显缩短等触发项。首遍通常自动检测语言、继承服务端默认；服务端已排除 prompt 泄漏时可直接给目标语言提示。 | `render_dub.py:11165`、`render_dub.py:11223` |
| DIS 最终 QC 复核 | 首遍未干净通过时，显式换另一个后端：Qwen3→Whisper，或 Whisper→Qwen3；指定目标语言，不给期望台词。记录实际 backend/model，并计算是否真的不同后端。存在源语污染等提前返回分支，不是所有情况都双跑。 | `render_dub.py:11341` |
| 转写/词时间戳 | `/api/whisper/transcribe` 名称不保证 Whisper。可显式传 `asr_backend`；已知 TH/VI/ID/MS/TL 要词时间戳时回退 Whisper，因为 Qwen aligner 不支持，不是证明 Whisper 转写更准。 | `api.py:5996` |
| 本项目离线微调评测 | 使用 faster-whisper large-v3；TL 默认 auto，其余按目标语言。与线上 Qwen 主 QC 不是同一把尺子，旧报告不改模型重算冒充改善。 | `src/voxft/eval.py:26`、`src/voxft/eval.py:293` |

上述 API 路径位于 OmniVoice；`render_dub.py` 位于 DIS
`src/dubbing_intel/core/render/`。当前模型默认：
Qwen 为 `Qwen/Qwen3-ASR-1.7B-hf`，Whisper 为 `large-v3`，均可能被部署配置覆盖。

### 两个容易误判的参数

- `/api/whisper/transcribe` 的 `model=large-v3` **不负责切换后端**，要显式指定
  `asr_backend=whisper`。DIS 客户端已经支持，无需先开发新框架才能做离线对照。
- 合成 QC 的 `output_text_qc_model` 仅在 Whisper 分支生效；
  当前 `_build_output_text_qc` 直接看全局 `_ASR_BACKEND`，不会复用转写端点的请求级选择。
  不能只改这个模型名或给 synthesize 传普通 `asr_backend`，就宣称 QC 已切换。

## 2. 已有证据不足以直接换 TH

### 本项目已有同 WAV 对照

重新读取 DIS `work/audio_ab/20260909_qc_listening/` 的 10 份原 WAV、
Qwen 输出、显式 Whisper 转写与反馈；没有重转写、重合成或改旧报告。
只有 5 组台词，部分组改变数字写法，不是 10 条独立文本或均衡抽样。

具体反例是 `05_alternate`，对应当时试听 05-A：

- 用户原反馈：“A尾音有个奥”。
- Qwen 转写为 `บริษัทเราใกล้จะเจ๊งแล้ว เอ๊ะ`，保留了额外尾音。
- Whisper 转写为 `บริษัทเราใกล้จะจริงแล้ว`，没有该后缀。
- 原 Qwen QC 仍 `coverage=1`、`overread=false`、`decision=pass`。
  这里至少有“文本门禁未利用短尾音证据”的缺口，不能仅归因为 ASR 不够好。
  后续短尾音规则已另行修过，本轮不重复修改。

这支持保留互补复核，**不证明 Qwen 整体更准**。用户没有确认尾音的泰语字词，
试听 raw/fit 版本未指明；Whisper VAD 关闭，历史服务端完整解码版本未冻结，
所以也不是严格的“只换模型”因果实验。
没有母语真值，不计算准确率，也不把目标台词直接当成实际说出的内容。

白名单归档：`checkpoints/asr_qc_policy_20260920/archived_th_pairs.json`。
包含 10 对转写、原文件 SHA、反馈映射及限制；完整请求、音频不入 Git。
该归档 SHA256：`a10ffa1c8c7ec7370145d6ac9bf462fdc186efa4a9e856f804113a1a1c45e486`。

### 官方公开结果只作候选依据

Qwen3-ASR Technical Report，arXiv `2601.21337v1`，Table 5：

| FLEURS 范围 | Qwen3-ASR-1.7B | Whisper-large-v3 |
|---|---:|---:|
| 20 语种汇总 | 6.62 | 6.85 |
| 30 语种汇总 | 12.60 | 8.16 |

这些是论文报告的基准误差汇总，不是本项目 QC 的误报/漏报率。
20 语种包含 TH/VI/ID/MS；30 语种增加 FIL 等。
附录逐语种表没有逐语种 Whisper 对照，不能从汇总倒推出 TH 或 FIL 谁赢，
也不能将论文模型实现与当前 `-hf` 部署视为已核实完全相同。
原始来源：`https://arxiv.org/html/2601.21337v1`，本轮已读取原文。

## 3. 建议怎么做，不拖慢正常合成

1. **先比现有两个后端，不新增模型。** TH/TL 优先，随后覆盖 VI/ID/MS/EN；
   复用历史原始 take 和最终 segment，分阶段报告，不用带 BGM 的成片冒充裸输出。
2. **同一 WAV、同一语言提示模式、同一文本归一化。** 用现有转写端点显式指定后端，
   `word_timestamps=false`、不传期望台词；冻结模型版本、解码/VAD 参数和音频 SHA。
   固定语言的内容核对与 auto 的串语种审计分开比较。
3. **不能只挑 CER 最低的 ASR。** 重点统计已确认坏例的漏检、正常样本的误拒、
   数字/否定/尾音风险，以及暖机 p50/p95、二次复核率和每 cue 总 QC 耗时。
   未确认的 TTS 内容标未知，不把“更接近期望台词”自动算正确；
   工程性裁尾等可控扰动须与自然坏例分栏，不能代替真实语言验收。
4. **Whisper 比较时去掉答案提示。** 合成内置 Whisper 分支当前传
   `initial_prompt=expected_text[:200]`，DIS 最终 QC 则不传。
   含答案提示的低 CER 不能作为胜出依据；离线对照直接复用不带提示的现有端点，
   不为评估先改变生产行为。
5. **得到可信收益才落分语种默认。** 复用 `language_config.py` 的语言注册表，
   同时接通请求级覆盖、实际 backend/model 回显、DIS 最终复核和缓存失效。
   选择后端所依据的“目标语言”不等于强制 ASR 输出该语言；
   保留 auto 检查源语残留，不能为了选 Whisper 顺手丢掉这道保护。
6. **线上仍是单主模型＋有条件复核。** 不为每条正常 cue 固定双跑；
   分语种切换本身不增加请求数，但加载、显存与实际时延仍需实测。
   ASR 不能验证音色、金属音或口音；不以换 QC 代替 TTS 质量改进。

当前所有受支持语种保持既有首选；TH 的上述反例不支持立即切换，
其他语种也没有本轮均衡同音频真值对照，不编造胜者表。
本轮只完成策略审计与旧证据整理，新的分语种 ASR A/B 尚未运行。

## 4. 本地验证

- OmniVoice：`test_reference_identity_qc.py::OutputTextQcTest`＋
  `test_locale_quality.py`，58 项通过。
- DIS：`test_final_text_semantic_qc.py`＋`test_ph_th_quality.py`，49 项通过。
- 归档断言确认 10 对实际 backend 为 Qwen3/Whisper、5 组台词、
  05-A 映射与多余后缀差异；原文件只读，未触发模型请求。

107 项为路由与决策契约回归，使用测试替身，不是 ASR 准确率或线上性能验收。

## 5. 后续 GPU 故障（同日，未部署修复）

TH 新成片原计划四条片段各跑两后端，但首次显式 Whisper 请求 `41dd386a`
在 CUDA encode 返回 `cudaErrorInvalidDevice` / HTTP 502 后即停止。
**没有完成这轮 ASR A/B，也未执行其中的 Qwen 对照。**
另外单独冻结的两条 CPU/int8 Whisper 转写完成，用于硬裁后的内容诊断；
耗时 26.951 / 17.09 秒，不作为默认 CPU 回退或 GPU 恢复依据。

OmniVoice 已修复确定存在的服务生命周期缺口：取模型前获取共享门禁，
合成 QC 不再绕过；取消等待在途线程退出；CUDA 失败剔除匹配的 Whisper 缓存，
包括延迟迭代和尾部补转的故障。不改变后端选型、不自动重试、不升级/降级依赖，
更没有据此认定 GPU 底层故障已消失。服务端说明与验收步骤见
OmniVoice 的 `docs/whisper_runtime_20260920.md`（实际仓库为
`/Users/dunxu.liu/workspace/others/OmniVoice`）。
本地新增15项回归旧源码12失败，修后通过；完整530项通过，另5项缺fixture在旧源码同样报错。
应用实际仓库后150项相关回归通过；服务端修复已push `7f678b4`，未部署。
成片诊断见 `video_review_20260920_th.md`。部署后有界 GPU 复测通过，才恢复选型对照。
