# 2026-09-23：紧邻同人音色与识别身份拆分

## 用户确认与证据边界

- `mtgg3m0`：cue1听似男声；cue1/2是同一人，音色明显不一致。
- `mtggf9f`：cue1/2/3是同一人；另有同人音色不一致，未单独指定其它异常配对。
- 用户要求：紧邻同人音色一致性优先于情绪、韵律、自然度；不抵消漏词、爆音等硬风险。
- 本轮未进行助手主观试听，下面是人工反馈与成片meta/本地源码核查，不是母语验收。
  原文保存在各自`work/video_review/20260923_<片号>/feedback.json`，这些大文件目录被git忽略。

## 两片身份冻结

| 字段 | mtgg3m0 | mtggf9f |
|---|---|---|
| 文件 | vn-11110136-6v3u0-mtgg3m02pg5c70.mp4 | id-11110136-6v3u0-mtggf9f0g9hd8a.mp4 |
| 实际目标语 | VI | VI（不能按URL的id判断） |
| 时长 / 字节 | 78.111995s / 33,817,583 | 136.24s / 72,468,271 |
| 剧 / 集 / 剧本版本 | 27 / 90 / 67 | 31 / 96 / 68，第7集 |
| 发布时间 UTC | 2026-09-23T08:24:03Z | 2026-09-23T08:33:11Z |
| 本地提取ref/prompt | 12份 / 3,275,876字节 | 17份 / 3,517,962字节 |

视频SHA256：
- mtgg3m0：`68d83fa03b8c1829702010fa2f848e7018afe973715e41bc43da2aaf59e0701d`。
- mtggf9f：`44b20f9e8c9614c549bc9accc74b147d45502e460ed4271483be8506fc724c09`。

任务分别为`generation-d4baa978-26b4-4451-8fde-763dfbd455b1`、
`generation-50a49da9-e881-413e-8a5b-94798eac3b9e`。
两片publisher与有记录的客户端请求源码摘要均为
`7533ae87b3478178ad83936deb4884eaa048b67440ad2c359fef83598f58b5e6`，
修改前本机DIS `7185f04`按相同算法逐文件核验一致。新片cue15无采纳音频，其请求不当成功take。
这证明上述任务的DIS包生效，不证明所有worker、GPU服务源码或加载权重；r8仍是指定生产基线。

两片分别与`mtg2i2m`、`mtg6jbj`的画面packet内容及PTS/DTS/时长摘要相同，
首片14cue时间/目标文字相同；后片28cue中首三句的时间、文字、speaker分配沿用旧版。
摘要分别为`f506cf133e9363bf3be6eb125eca1e0cae2df19ee6e6bb5ae331978731a499fd`、
`795d70b1ec8a14e7552b4ef29c4670e6c2b44961f6f32f94550c82ba979d94d1`。
生成重渲没有自动修复旧身份分配；不能由当前DIS包版本反推最初识别版本。

## mtggf9f：先纠正身份，不能只调TTS

| cue / 时间 | 实际speaker / 角色映射 | 已采纳参考 | 有效seed / CFG / steps |
|---|---|---|---|
| 1 / 1.800–4.298s | review_bd364edc / Bạn của Minh，c_cc4edb | canonical cue9，a21c6942… | 1809016040 / 1.5 / 22 |
| 2 / 4.720–7.320s | review_4d5495b4 / 配置映射Minh，c_a672a0 | 当前cue，c375e9f0…；最终reference_character_id为空 | 488068474 / 1.5 / 28 |
| 3 / 7.330–9.615s | review_bd364edc / Bạn của Minh，c_cc4edb | canonical cue9，a21c6942… | 2039707472 / 1.8 / 28 |

间隔为0.422s和0.010s。依据用户确认，这是同一声音被分配成A–B–A；但用户没有确认真实角色名，
不能直接把所有“Minh”或全剧同名角色强并。修改的是这三句的身份关系，不是强制合并字幕时间轴。

- 三句已采纳take均无control；cue1/2无prompt，cue3的prompt是cue1成功救援后的原始TTS。
  **因此这三句不能简单解释成情绪指令导致变声，cue1/3也不等于完全同条件。**
- cue2 canonical因`low_speech_share:0.424/0.820`被拒，换cue8参考仍未改善，回滚到当前参考；
  已记录5次模型生成，采纳take仍有severe `output_vocalization`，最终文本pass不解除声学风险。
- 相邻检查按speaker分组，天然不会直接比较这里的1→2和2→3；cue1/3对锚点约0.63/0.64且无风格，
  走健康跳过。提高候选音色权重不能修复错误的身份分组。
- 28条渲染记录均为`unverified`且reason为空。锁版cue没有身份状态/参考策略/换声边界字段，
  也未携带`recognition_task_id`；成片不足以证明声纹复核未执行，或断言是哪一层丢失了证据。
  应取原识别episode result→锁版cue→生成manifest逐段核对，不能把缺字段当已确认身份。

其它自动风险（不是新增人工异常真值）：
- **cue20/21/22，91.16–96.68s**：相同speaker，三个参考SHA不同；相邻相似度0.2644/0.2225，
  都低于当时0.35门槛。canonical因`ambiguous_gender:f0=166.7Hz`被拒，无合格锚点，未成功重合成。
- **cue15，74.60–75.953s**：已用完6次模型生成，最终文本incomplete，`tts_failed`、raw/segment时长0，
  源人声被抑制。这是明确配音交付失败记录，不把它误称为旧片相同沙哑take或新的空WAV根因。
- cue11/25/27仍有严重声学/身份风险及裁后uncertain；refit `rewrite_rejected`，changed_indices为空。
- 编码后峰值保护实际运行两次：0.12→−2.16dBTP；这是编码风险修复，不是身份/沙哑修复。

## mtgg3m0：身份相同，参考与救援仍不稳定

cue1（0–3.86s）与cue2（6.58–9.60s）都是SPEAKER_00，无锁定character_id；
已采纳seed均1988467456、CFG1.8/28步，reference-only，但实际ref SHA分别为
`b3b7da8c44ec65e1599a5385b4160a08caf6cb5fe5ef9643e80b5724ccdb07b4`、
`3317f3637b0f32fd4ad11aa217808ceeb40a8fc9bfb23e4e32f797135c1467a8`，控制与后处理也不同。
两句均severe speaker_mismatch，去控制/换参考救援被拒，keep-best保留初始take；不算质量通过。
相邻检查1→2为0.3186<0.35，已发现漂移，但canonical低语音占比被拒、无合格锚点，未成功重抽。
不关闭参考保护，也不把用户认为不对的首句作为干净续写基准。首句ref字节与已听R01相同，不重复索要评分。
F0未报性别漂移不反驳“听似男声”；少量VI坏例不能否定/认证联合模型的五语种能力。

## 最小改动与后续顺序

本轮DIS仅两处共享逻辑，已随`b5f4fd4`push至`feat/recognition-generation-mainline`，尚未部署：
1. `subtitle_refinement._continuous_turns_should_join`：自动词级音画审查自报高可信同声连续、
   无换声、间隔≤0.6s且总窗≤20s时，不再让不同speaker标签绕过结构校验。矛盾结果走原schema
   重审路径，缓存共用同一校验；不是按短间隔自动合人，更不凭角色名证明声学身份。
   真换声、低可信/缺连续性证据仍不触发。未获得本片原始音画审查结果，不能断言本片会命中。
2. `final_segment_candidate_score`：仅软emotion/prosody告警不再把身份权重从200压到24；
   内容不完整与其它声学/身份风险继续用原低权重，硬门、阈值、正常预算和严重共享4/6预算不改。
   复用已有身份分，不新增全片比对；不把对不同ref的相似度伪装成相邻segment实测。

新增3条回归在原代码全部失败，修后首组144通过、第二组162通过、预算/择优37通过，共343通过。都是本地无模型回归，
不是重识别、生产耗时或听感改善；未调用ASR/TTS、未训练/更换r8，旧24条模型诊断仍0生成。

交付前16个相关测试文件联跑343通过（8.09s），未覆盖原分组测试数字。
新提交Python包源码SHA为`4c96b6d61ebf0dd5b60a31d6b2408b59f97cacbbea08f147797f8b207362216e`；
用户计划部署后重识别第31剧第96集（第7集）。这是待验证动作，不预先认证前三句身份或音色已改善。

下一步：
- 优先取得第96集原识别结果与1.8–9.615s真实源人声，定位文本归因/音画精调/声纹复核/锁版传输。
  复用已有diarization/WavLM与源窗，强证据或人工身份订正后统一绑定；不另造全片第二遍识别。
- 用已确认同人且合格的共享源参考复核短话轮；若参考不合格，换可信源窗而非放宽QC。
- 相邻救援的未锁定路径仍待补与原compare_left同尺度的segment复量；当前仅锁定路径如此做。
  不用服务端raw-vs-ref分数冒充邻句一致性，不用风险前句强行做克隆锚点。
- 可靠输入下仍连续出现原始TTS严重坏例，继续冻结base/r8三seed模型诊断，不能永久“无需微调”结案。

## 可恢复与缺失

全部内嵌ref/prompt均已校验SHA。发现部分prompt字节恰好等于其它cue的raw SHA：
mtggf9f可恢复cue1/22 raw，mtgg3m0可恢复cue3/10 raw，关联存`recovered_stages.json`。
其余raw、最终segment、拒绝候选和GPU预处理后参考仍缺失；不笼统说所有raw均缺失，也不宣称完整重放。
所有成片与合成prompt只用于开发回归，training_eligible=false，语言质量未全面验证。
