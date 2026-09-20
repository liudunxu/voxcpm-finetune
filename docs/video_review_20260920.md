# 2026-09-20 成片复核：1:50 同角色音色差异

## 结论边界

用户反馈原文：「1min50秒的时候，同说话人的音色稍有不同」。
按**用户报告的局部音色差异**登记，不扩大成整条失真、娃娃音、印尼语口音异常，
也不声称已经重复合成复现。此次完成视频下载、元数据/源码核对、只读服务状态查询和
CPU控制流复现；未由助手听评、未重合成、未训练或部署。

1:50 在 cue22 内部，距起点2.84秒，不是已知的换cue或拼接位置。
cue19/22/23的角色ID、参考签名、锚点均相同；没有发现这三条换参考的元数据证据。
但cue22采用了改变seed、CFG及步数的重试候选，不能只凭同ref就当成同生成条件。
**优先调查重试候选的音区/音色变化及其漏检，不直接归因r8训练不足。**

## 产物与部署身份

- 视频：`id-11110136-6v3u0-mtbtg7qhj9j529.mp4`。
- SHA256：`d5951cfa306ec5c16ef487303611db127c5dd1822dbb43a211cb61647eef4099`。
- 内嵌渲染记录时间：`2026-09-20T01:56:08Z`，北京时间09:56:08。
- 视频约157.611秒，29条cue：24条TTS、5条保留原声；目标语为id，音频源语字段为auto。
- 与9月16日生成的旧Salju反馈视频不是同一素材，不能按相同cue编号或异常率直接作A/B。
- 新产物有 `confirmed_edge_v1`、Whisper低置信标记及24/24最终有效采样参数，
  证明相关新链路存在；**不证明完整checkout版本或SRT识别v13已部署**。
- 元数据的 `voxcpm.model=openbmb/VoxCPM2` 来自DIS客户端参数默认值，
  不是实际加载权重的证明。只读查询该产物所指服务的 `/api/status`，
  返回已加载 `FrankLiuDundun/voxcpm-finetune-lora`、CUDA、denoiser开启。
  这是查询时的状态；产物没有权重SHA、HF revision或部署commit，不能回填为渲染时的r8指纹。

原视频、完整meta、ffprobe结果、服务状态与检查时间在ignored目录
`checkpoints/video_review_20260920/`；完整meta含内部服务配置，不提交。
复用DIS的 `decode_embedded_dubbing_metadata`，没有另造解码器。

## 1:50 的具体链路

cue22窗口为107.160–120.900秒（1:47.160–2:00.900），正文：

> Jadi lukamu bukan karenaku? Demi menolongmu, botol bekas yang kupungut kutinggal.
> Uang makanku empat hari juga habis. Untuk membelikanmu obat luka.
> Ternyata bukan aku penyebabnya. Ya ampun! Aku benar-benar difitnah!

共有7个句子，一次生成完整take；报告中的成片词时间戳把110秒附近放在
`Demi menolongmu` 处，但来源是Whisper，不据此宣称精确定位到某个音素。
cue22没有内部停顿压缩记录，也没有硬裁记录；后处理是尾部外静音清理、
约1.055倍变速与增益等。尚无原始take，不能把后处理完全排除。

| 项目 | cue19，1:41.500 | cue22，1:47.160 | cue23，2:02.620 |
| --- | --- | --- | --- |
| 角色ID | c_21a8d7 | c_21a8d7 | c_21a8d7 |
| 参考锚点 | cue4 | cue4 | cue4 |
| 参考签名 | ref-v2:701d26d2a3656a1afb0b02d5 | 同左 | 同左 |
| 最终有效seed | 2105213648 | 595119419 | 2105213648 |
| 最终CFG / 步数 | 1.8 / 20 | 1.5 / 28 | 1.8 / 20 |
| 控制模式 | 惊讶/问句指令 | natural，无控制/续写prompt | 生气/强调指令 |
| 原take时长 | 2.140s | 16.320s | 4.665s |
| 成片segment时长 | 1.932s | 15.419s | 4.194s |
| local_pitch | 有 | 缺失 | 缺失 |

cue22的两次客户端候选记录：

| 候选 | 最终有效seed | CFG / 步数 | 原take时长 | 服务器speaker_identity输出F0中位 | 服务器prosody输出F0中位 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 首次 | 2105213648 | 1.8 / 20 | 19.495s | 320.5Hz | 370.42Hz | 未选 |
| 重试 | 595119419 | 1.5 / 28 | 16.320s | 210.7Hz | 264.86Hz | 选中 |

两个候选均带 `severe=output_vocalization`，重试耗尽后keep-best出片；
最终仍为 `unresolved_artifact=true`、`relative_best_after_retry_exhausted`，
不是质量通过。`output_vocalization` 是自动标签，尚未人工确认具体额外发声。
重试候选的两套音高估计均降低，支持进一步核对音区的方向；
但两套算法不可混用绝对值，F0也可能有倍频/半频误差，**不能替代音色听评或证明某参数致因**。
两候选同时改变多个参数，本身不是单变量实验。

与锚点的声纹相似度从0.6942升至0.7315，并没有挡住用户听到的差异。
同人相似度合格不等于同人每句话音区/音色一致。最终Qwen复核检测为ms、指定id的
Whisper转写较接近正文，也都不能证明真实口音为MY或ID。

## 已复现的检查缺口

DIS当前 `clone_output_problem` 在遇到长句severe `output_vocalization` 时提前返回，
后面的 `clone_gender_pitch_problem` / `clone_register_drift_problem` 未执行。
这能让候选先进入必要的发声重试，但keep-best最终仍采纳它时，
本片cue22没有补回 `local_pitch`。

`_cue_output_median_f0` 只读有效的 `local_pitch`，因此：

- cue22没有进入角色级 `mark_speaker_pitch_drift` 报告；
- 基于local/reference pitch的候选评分项也缺少这组输入；
- 不能把角色报告里“没列cue22”解释成它已检查且音色稳定。

修复前在DIS `900d243` 对真实cue22元数据执行当时函数，断言返回severe-vocalization问题、
pitch检查调用次数为0，且 `_cue_output_median_f0(cue22) is None`，通过。
此检查仅复现控制流，不拿成片混音冒充裸TTS做声学测量。
原 `checkpoints/video_review_20260920/check_pitch_gap.py` 作为修前证据保留，
该脚本未开启keep-best，不能验证新增补测入口；当前回归使用DIS
`tests/core/test_keep_best_pitch_and_silent_reference.py`，不修改旧检查来伪造修前结果。

相关源码：DIS `src/dubbing_intel/core/render/render_dub.py`：
`clone_output_problem`、`clone_result_quality_score`、
`final_segment_candidate_score`、`_cue_output_median_f0`、`mark_speaker_pitch_drift`。

## 本片其他发现

- 24条TTS没有 `take_edge_hygiene.trimmed=true` 或 `content_fit.hard_trim_applied=true`；
  cue9/23明确因Whisper对齐不可信跳过边缘裁切。旧危险尾裁保护已有产物证据，
  但外静音清理/变速仍在，不能写成“所有末音节已验证完整”。
- 5条有severe标签（7/9/15/22/23），6条有待复核原因（另含18）。
  不把这些自动标签当成6条人耳确认坏例。
- cue7（35.400–36.828秒）参考QC记录为全静音水平（peak/rms −140dB、语音占比0），
  最终带 `gender_mismatch/speaker_mismatch/too_quiet` 仍出片。
  该条为独白角色、缺角色ID，应优先核对原声切片/声道/角色绑定；
  不直接把同名独白与对白强并，也不将“脏但有人声”与“全静音”混为一类。
- cue6记录语音起点早1.030秒；cue22记录结束比源语音边界晚1.613秒，
  是局部节奏线索，不据此全局平移音轨。
- 当前长cue拆分入口遇 `reuse_matching_tts_cache=true` 直接跳过；
  本片命令有该开关，且cue22七句也超过默认最多三句的拆分条件。
  不能只移除一个开关就声称会拆开，更不能缺原声分句对应时机械切七段。

## 后续顺序

1. **先补keep-best候选的检查覆盖**：复用已有本地pitch/音区分析，使提前判退但最终可被
   选中的候选也有同口径观测；测不出保留未知。不混用不同F0估计器补数，
   不取消发声风险门，不新增全局F0否决阈值。
2. 优先取回本次cue22的两份原始候选和cue19/23、实际ref及后处理segment，
   按请求ID/哈希归档；先比较已有音频，不浪费GPU重抽。原始输入身份未齐时
   不把任意参考音生成结果称为复现。
3. 需要新合成时先固定文本/ref/控制/CFG/步数，仅对照seed；
   再单独试采样档位。短句/长句拆分另立对照并保留原声分句边界，
   不同时改模型、说话人、控制指令与速度。
4. 不立即追加微调；上述路径排除后，多个独立文本稳定出现同类问题再立单变量训练假设。
   当前生产r8不动，语言自然度/口音仍未验证。

用户反馈独立存于 `feedback.json`，开发回归文本为
`eval_cases/production_timbre_regression_20260920.jsonl`。
`context_98p5_130p5.wav` 是从最终视频解码的98.5–130.5秒混音上下文，
未归一化或分离；不是裁前take，不作训练数据。

## 2026-09-20 后续实施：限制时延的链路修复

用户授权继续，并要求识别和合成时间尽量不增加。DIS新增保护已经本地实现：

- 提前失败且允许keep-best的候选，在评分/快照之前补本地音区观测；后处理入口也覆盖
  旧缓存take和超短音频快速分支。保留原发声问题和severe，不以音区观测触发新重试；
  沿用现有评分权重、8秒分析上限，复用成功/测不出的profile。
- 补测只调用本地音区检查，不调用可能追加远端声纹确认的性别门。
  已有源切片profile仅复制，无额外源音频扫描。主合成循环同步QC移到线程执行。
- VoxCPM共用注册/请求路径检查真实条件音频，空或全声道峰值≤−120dBFS才拒绝，
  包括prompt-only；按完整字节缓存64项，同路径重写不会沿用旧结果。
  弱人声、反相双声道不按静音拒绝，本地解码未知交原服务端验证。
  明确静音由现有不可恢复错误流程留痕，不再无效换seed请求。

本机8秒16kHz PCM16合成正弦基准（不代表本片原take或生产Linux）：
已有参考时补测新候选中位148.78ms，两者均需测量295.87ms，
已有结果复用0.0065ms；静音检查首次0.0982ms、命中0.0001ms。
脚本与结果见ignored的 `benchmark_local_qc.py` / `benchmark_local_qc.json`。
识别/LLM/ASR/TTS请求未增加，r8、CFG/步数与重试预算未改；不能将单项CPU耗时外推整片提速。

原始音频取回仍待完成：现有生成服务会上传 `CUE_AUDIO_BUNDLE`，可取得已采纳raw/segment，
但不含所有被拒候选与实际ref。本机未找到此任务包，缺音频包描述符；
没有 `.env.spex.local` 或 `SP_SERVICE_KEY/SPEX_SDU/SPEX_GATEWAY` 查询配置，
未猜SPEX网关/内部workspace下载地址，也未重渲补造材料。
后续先拿现存产物，再做必要的最小单变量回放；不是重新开训的理由。

完整实现与回归说明见DIS
`docs/reviews/2026-09-20-keep-best-pitch-and-silent-reference.md`。
最终7条回归在旧源码为6失败/1健康对照通过，修后全过；相关254项扩展回归通过，
实际仓库应用后17项再次通过。补丁及测试/基准日志归档在同一ignored证据目录。
修复及回归已随DIS `aebf175`推送，用户原有 `docs/reports/` 未触碰。
本轮未部署、未听评新成片，不宣称1:50音色差异已消失。
