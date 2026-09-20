# TH 成片与 Whisper CUDA 诊断

2026-09-20。对象 `id-11110136-6v3u0-mtbwhene6sjnb1.mp4`，目标语 TH；
源语 meta 为 auto，不能从字幕文本替代判断原声语言。
本轮分析 meta、日志与两条显式 CPU ASR，不声称完成母语听评。

## 冻结证据

- 成片 104.385 秒；meta 时间 `2026-09-20T03:26:31Z`，北京时间 11:26:31。
- SHA256 `84bcda7af822b16a671ee088321f0ff57b5ce975a365c3baeeb276c643d51666`。
- 15 个 cue：14 条 TTS，cue7 笑声保留原声。7 条需复核（5/6/8/9/10/12/13），
  cue5 带 `output_vocalization`、cue9 带 `speaker_mismatch` 出片。
- 12 条有 final QC：10 pass、2 invalidated；cue6/13未运行，cue7非TTS。
  已保存的主后端均为 Qwen，未见成功的第二后端 Whisper 转写。
  meta 里的25条候选历史不是实际总模型调用数。
- 原片、meta、WAV、请求和响应在
  `checkpoints/video_review_20260920_th_mtbwhene/`，不入 Git，不作训练数据。

## 优先问题

| 位置 | 已确认事实 | 不能据此确认 |
|---|---|---|
| cue10，62.64秒起 | raw 8.685秒，最终4.659秒；`hard_trim_applied=true`、`complete=false`，要求refit；裁后QC失效。CPU转写尾部停在`องค์รัชทาย`。 | 不能把混音ASR直接当逐音节真值，也不能把裁后漏内容归因模型原始生成漏尾。 |
| cue12，68.599秒起 | raw 3.45秒，最终2.061秒；同样硬裁/未完成/refit。期望句尾`ไม่ได้แล้ว`，裁后CPU转写未保留。 | 不能拿裁前Qwen pass证明裁后内容完整；ASR遗漏仍须同take阶段音频确认。 |
| cue8/9，约52.7/56.7秒 | 历史候选有4.08/20.34/37.17秒失控长度；Whisper辅助对齐有OOM和invalid-device；cue9最终有身份风险。 | 时长/身份自动标签不等于人耳已经确认换人或口音。 |
| cue6，约41.88秒 | 参考音污染，被best-effort接纳，没有已知干净同人替代ref。 | 不能靠任意换参考人的A/B宣称模型克隆修复。 |

cue10 首个5.33秒候选的文本QC coverage=.896，但被长度保护按BPE比率判
`too_short`，后续候选更长再硬裁。这是泰语长度保护可能误报的线索，
不是该候选内容完整的证据；本轮不关闭长度保护、不全局增加变速或重试。

DIS 已有 `core/render/refit.py::collect_refit_issues` 与
`services/generation_adapter.py` 的 refit/重渲消费者。
对映射后的既有 meta 本机重放，cue10/12都能被收集；
没有实际任务日志，尚不能判断为何本轮未闭环，不能写成“没有检测/消费者”。

cue5/8/9/13未见 `local_pitch`，cue11控制文本仍有半短语的旧症状；
这些与已修过的链路缺口相符，不证明当前本地/已push版本已部署。
最终模型地址可查到微调仓库，但渲染时权重SHA/源码commit仍不能由此确认。

## Whisper GPU 排查结果

1. 用户确认服务地址后，显式 Whisper large-v3 / th、无答案提示、
   无VAD/词时间戳/尾部补转的首个请求 `41dd386a` 返回502，耗时1.791秒。
   错误来自 `list(segments_iter)` 内的 CTranslate2 encode。
   原4片段×2后端对照因此中止，没有把未执行条目计为完成。
2. 另做两条CPU/int8请求，只检查cue10/12裁后混音：
   分别26.951/17.09秒；不代表GPU恢复，不做CPU默认回退，也不据此评价后端整体准确率。
3. 只读服务日志/缓存与Vast元数据核对完成；现有单卡4090实例，失败CUDA模型仍在缓存。
   不重启、不卸载共享模型、不另开机器。
4. OmniVoice 本地完成共享门禁、取消等待、失败缓存剔除及尾部补转不吞CUDA异常；
   新15项回归旧源码12失败，修后通过。全量530通过，另5项缺fixture在旧HEAD同样报错。
   修复已随`7f678b4`推送，实际仓库相关150项复核通过，未部署。
   **只是工程保护，不是已在线修好GPU，也不是TTS音质收益。**

## 下一步

- 新版部署后先同一短WAV各做一次GPU文本/词时间戳检查；首个失败即停，
  保留request ID与运行环境，不盲目降级依赖。
- 获取cue10/12原take、最终segment及实际refit任务日志；先解决内容完整性，
  再评价cue9身份及cue6参考污染。不添加固定双ASR或扩大正常合成预算。
- 保持r8，不恢复训练；五语种自然度/口音仍未验证。
