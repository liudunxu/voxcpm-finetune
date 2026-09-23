# 2026-09-23：质量自愈跨仓审查与严重坏例预算

## 决定与范围

用户要求：严重badcase宁可增加少量预算，也不要为了速度直接保留过差候选。
本轮先改DIS共享救援，不全局增加首轮best_of/步数，不自动重开微调。
默认普通cue上限4次，触发现有强风险门后一次性增加2次模型生成；所有救援层共享，不按HTTP数各自扩。
弱情绪/韵律提示、单纯低SNR、无独立确认的text_incomplete、短cue低可信身份提示不据此扩预算。

## 三个项目分别负责什么

| 优先级 | 项目 | 责任与状态 | 正常路径成本 |
|---|---|---|---|
| P0 | dubbing_intelligence_service | 已本地改：完整严重风险参与分流、一次性严重额度、严格剩余额度、旧候选回滚；复用既有去style/prompt与换参考机制 | 正常首轮请求预算不变，无新增ASR/TTS |
| P1 | OmniVoice | 本轮现有max_generation_attempts契约已足够，无需为扩预算修改/部署GPU代码；下一轮补reference-only跨语言韵律证据、内部候选原因摘要 | 优先复用现有QC；不新增逐cue大模型评分 |
| P1 | dubbing_intelligence_service | 接收上述证据，修复层次分流：输入/身份、生成声学、时序/后处理分别处理；不能拿ASR pass兜音质 | 仅疑似风险触发局部救援 |
| P2 | voxcpm_finetune | 冻结坏例、单变量对照与分语种回归，不改训练算法、不开GPU训练 | 离线成本，不进入生产热路径 |

FFmpeg处理在DIS内，不需要独立新增项目/服务；页面只需现有离线HTML，不需要先改审核页或新增视频。

## 本片暴露的系统问题

VI首句请求允许服务端3次生成，实际用完3次，调用方日志先记emotion_mismatch，
随后换参考时因普通额度用尽失败；同一已选候选还带severe speaker_mismatch。
默认总额4、证据预留1时，普通救援的可用额恰好是3，服务内多抽会先消耗掉后续换策略空间。
这不是“没有自愈”：两仓已有self-heal、换seed、去style、换参考、候选择优与回滚。
问题是预算分配和失败分流，不能再叠一个独立重试器。

源码位置（本轮读取/修改版本，行号后续可能变化）：
- DIS `core/render/render_dub.py`：clone_budget_used、clone_generation_limit、
  clone_severe_recovery_reason、_synthesize_with_retry_core、retry_clone_with_alternate_reference_if_needed。
- DIS `core/tts/backends/voxcpm.py`：_compose_voxcpm_style_instruction已支持reference_only，
  _gated_style_labels对schema>=2才应用新置信门，不能把legacy的0当实测置信。
- OmniVoice `api.py`：synthesize_voxcpm的self_heal_cap/候选池、_compare_prosody、
  _prosody_mismatch_is_corroborated和generation_attempts/effective_generation回传。

## 已落地的最小修复（仅本地）

1. 新增`CLONE_SEVERE_EXTRA_GENERATIONS`，默认2；第一次确认当前候选存在现有严重声学/身份门风险时
   扩额度，记录原因。后段才发现严重风险也在拒绝预算前检查；重新尝试、换参考、失败回滚不能再次
   扩额，最后证据救援仍使用共享总额。
2. 分流不是只看第一个错误串：补入severe声学/身份证据，避免emotion_mismatch遮蔽音色/金属音。
   对这类失败候选复用原有reference-only救援，身份救援尊重手动style锁与既有开关；正常首轮不变。
   改控制条件已改变缓存键时，不强迫额外改变所有参数；需要换seed的路径仍走既有派生规则。
3. 修掉剩余1次却给服务端max(2,...)的超额：API收到的生成额度不得超过可用剩余。
   超时/未知失败按已授服务端额度记账，不当作零成本重试。
4. 复用原候选择优、音频与实际控制条件回滚。新候选更差不覆盖旧候选；全部失败仍保留明确风险，
   不把keep-best认证为质量通过，不回退源人声，不让风险take进入干净续写/缓存。

未更改全局权重、CFG、steps、QC阈值、默认表演标签、识别API次数及成片数量。
这不意味着所有沙哑都能自动检测：现有metallic/high-frequency规则不是通用沙哑检测器。

## 下一轮OmniVoice/调用方契约

- `emotion_mismatch`实际来自F0范围、能量范围、活动比的韵律比较，不是母语情绪判定。
  现有cross_lingual分支依赖prompt_text含CJK；reference-only无prompt时缺这条证据，
  非CJK跨语言同样不在该启发式覆盖内。应显式传已核验的参考语言及来源，未知保持unknown，
  不从中文字幕或source_language配置硬猜真实参考语言，不按语种关闭全部QC。
- 服务内已有候选池，优先增加有限的候选摘要（实际控制、seed、生成计数、QC原因、采纳/拒绝），
  复用当前回包和DIS候选记录；不上传全量音频、不另建追踪服务。当前effective_generation不能代表所有失败候选。
- 后处理坏例先复用已有raw→segment证据：确认原take正常才局部重处理，不一律再抽TTS。
  没有可靠证据的沙哑指标先观测/人工核验，不直接转成新硬门或新增音频评分大模型。

## 验证与试听

首批13项回归修前全失败，其中6个实际复现场景是“服务端用完3次后没有第二次策略请求”；
其余包含新预算/证据契约与最终余量边界。修后13通过、8文件148通过、26文件540通过（12.51s）；
补齐后段才发现严重风险的入口后，新增回归共14项，最终26文件**541通过（12.78s）**。
覆盖严重风险换策略、劣化回滚、超时记账、弱提示不扩额、短cue身份弃权，
以及现有识别/参考/内容/候选链。没有把单测时长外推生产p95或声学收益。

情绪试听包`emotion-151930650565`在DIS `work/audio_ab/20260923_vi_emotion/index.html`。
固定实际参考、文本、seed1988468465、CFG1.8/28步，仅开关control_instruction，
2条各1次生成、无结果缓存命中、预降噪参考摘要一致；请求分别13.534s/15.944s。
这是服务输出探针，含服务器自己的处理，不含DIS变速/limiter/背景混音；
不是旧take复现，也不是新“4→6预算策略”的完整线上A/B。
用户已评：E01-A明确沙哑杂音，E01-B明确开头杂音，偏好B；揭盲A=原控制、B=无控制。
两者均未声学通过，不推广全局关情绪；17个原文件SHA保持不变，feedback.json另存精确反馈。
单seed、语言质量未验证；模型ID已查询，权重/运行源码SHA仍未知。

新的1:57坏例（mtg3tot cue27，目标VI）也被已知severe和4次额度阻断救援；不是新6次策略的失败验收。
原参考前静音线索只在前一条成立为低能量观测，新cue27参考没有同样的长低能量前导。
先固定两条做12音频base/r8归因，当前只完成计划/输入核验、0条生成；详见TODO与新片复盘。
随后灰度本地救援策略，统计强风险cue的修复/残留、
正常cue调用数、模型生成数和端到端p50/p95。只有正常路径未增调用且局部收益成立才推广。
DIS修复已随`bd2ada7`push并核验远端HEAD，待用户部署；OmniVoice源码未改；r8不变，
无新训练、业务视频或整片重渲。交付前DIS26文件367项、本仓全套152项通过，非声学收益验收。
