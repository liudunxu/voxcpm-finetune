# 2026-09-23：mtgix3k源码更新，但音轨与上一版完全相同

## 冻结身份

- 文件：`vn-11110136-6v3u0-mtgix3kxn6698b.mp4`，78.111995s，33,817,499字节。
- SHA256：`52ec26289ca289bb3f3c77aa497a1ef66d6c803f7610aca1eaefa8cb96bbd5c8`。
- 实际目标VI，第27剧第90集/第4集，script revision67；**不是第31剧第96集/第7集**。
- 任务`generation-df1d355e-14ab-4191-b72e-237be0905cd8`，execution
  `c39e105b6c504efcb5cd565fdc493e48`，发布时间`2026-09-23T09:42:28Z`。
- 源语言配置仍en、源字幕中文；没有因复盘强改源语配置。
- 本地目录`work/video_review/20260923_mtgix3k/`被git忽略。保留完整inspection、
  ref/prompt、`media_comparison.json`、`source_identity_check.json`、`review_summary.json`。

## 版本核验：不能再归因“上次修复未部署”

publisher及14条client_request的包源码SHA均为
`633f9f6dd142af8d308fdb7e6a925886a39e78196b81116767c5a85da962c625`。
只读fetch发现远端新增识别模块重构`0cb9ded`及合并`ce01d0d`；对远端git归档内201份Python
源码按publisher同口径重新计算，**与本片完全匹配**。`b5f4fd4`是该合并提交的祖先，
两处已修源码相对`b5f4fd4`无差异，连续同声校验和软风格身份权重修复都在这个包里。

这里核实的是本次发布/请求进程；不代表所有worker、依赖、GPU服务或权重身份。
也不证明第96集已经用新策略完成识别。本机DIS工作区仍在b5f4fd4，本轮没有merge、改代码或push。

## 与mtgg3m0对照：新文件不等于新声音

基线视频SHA为`68d83fa03b8c1829702010fa2f848e7018afe973715e41bc43da2aaf59e0701d`。

| 检查 | 结果 |
|---|---|
| 锁版版本 | 都是67；未带recognition_task_id |
| 14条cue的时间、文字、speaker、cue_key | 全部相同 |
| 已采纳raw / 最终segment / reference SHA | 各14/14相同 |
| H264画面packet及PTS/DTS/时长摘要 | 相同 |
| AAC音轨packet及PTS/DTS/时长摘要 | 相同 |
| FFmpeg解码为原采样率/声道的s16le PCM | SHA完全相同 |

画面packet摘要：`f506cf133e9363bf3be6eb125eca1e0cae2df19ee6e6bb5ae331978731a499fd`。
音轨packet摘要：`9e69ddb35d2831aff223829a7f20f0c5c12dac735136800103e44592b2fe2876`。
PCM摘要：`a4f1e9298288d9c5730311f7b2c8f3ad9dca25cf758fff9a2ecfbfa4c65d636c`。

所以这份**不能算首句偏男声、1/2音色不一致已改善**，也不新增一票人工异常或一个独立失败样本。
不是助手新做的主观试听；相同音频直接沿用已确认的旧片反馈。
首两句请求ID和时间已更新、`tts_cache_reused=false`；没有服务器缓存命中字段。
**不能由音频相同就断言没有执行请求或全由缓存导致**，也不据此全局清缓存、扩大重试或微调。

## 首两句为什么仍未解决

- cue1/2仍为同一SPEAKER_00，seed均1988467456、CFG1.8/28步；仍使用各自不同的参考音和控制。
- canonical仍因`low_speech_share:0.365/0.626`及reference_event_risk被拒；
  1→2相似度仍0.3186<0.35，已发现漂移但无合格anchor，成功重合成仍0。
- 两句均有severe speaker_mismatch，救援未消除风险，选回相同初始take。
  上次只修“软emotion/prosody告警压低身份权重”；**不能用这个改动放过硬speaker_mismatch或坏参考**。
- cue2裁后仍uncertain；cue14仍critical_mismatch，refit requested=[2,14]、no_eligible_cues。
  编码峰值−3.18dBTP/−14.51LUFS与上版相同，不是本轮音色改善证据。

## 下一步

1. 第90集先处理合格同人参考缺口，取得首两句实际raw/segment与GPU处理后参考，做局部链路定位。
   不取消参考保护、不用坏首句当干净锚点、不以相同输入反复整片重渲充当优化。
2. 第96集继续验重新识别结果及新锁版身份：用户已确认前三句同人，需检查修正是否真的进入生成。
   这条第90集的链接不能替代该验收，也不能证明第96集识别没有执行。
3. 可靠输入下仍有原始TTS明显badcase，继续原base/r8诊断线；当前仍0/24生成，r8不变。

已校验提取12份客户端ref/prompt，共3,275,876字节；按prompt与raw SHA交叉匹配仍可恢复cue3/10。
首两句raw、最终segment及GPU预处理参考仍缺失，不能宣称完整重放。没有新增ASR/TTS/训练；
成片/合成prompt仅开发诊断，training_eligible=false，语言质量仍未验证。
