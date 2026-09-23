# 按坏例准备训练材料（2026-09-23）

用户要求先搜集材料放`work/for_train`。已建立备料池，**不是已备好可训练音频，也没有开训**。
当前14个来源候选、16条回归/反馈条目、13份评测清单428行/393条去重文本排除；
已审可训练音频0、新下载真人音频0。既有本地`t_*`是测试素材，不纳入。

## 已落地

- `work/for_train/README.md`：准入边界、候选采集流程、下一轮r8实验条件与现有命令。
- `candidate_sources.jsonl` / `source_evidence/`：13个现有registry源配置，加1个待核VI源；
  查询6个公开Hub仓的元数据，保存revision/许可声明。未请求音频，不把元数据SHA当成音频版本。
- `badcase_inventory.jsonl` / `regression_exclusions.jsonl`：保留原反馈、来源行与文件SHA，
  成片和参考仅指向原诊断记录，不混入训练音频；这些文本还须在远端与val及PCM/身份/session共同隔离。
- `routing_feedback.jsonl` / `failure_modes.json`：漏对白/事件误判与TTS声学问题分开；
  不从爆音/沙哑/能量等推演情绪真值，不把相对偏好当成干净真人正样本。
- `collection_plan.json` / `validate.py`：原始采集条数上限、原清单哈希与准备阶段自检。

## 采集优先级

VI完整、干净自然短句 → TL停止/加词风险短句 → TH/ID/MS防退化与可靠同人ref。
先复用远端已处理train池及`scripts/training_pool.py`，不是重新下载一遍或把旧数据计为新增。
不足再小批取GigaSpeech2 VI、FLEURS VI/TL/MS、Filipino Speech Corpus、CV22 TH/ID。
所有音频仍在远端大盘下载，当前无可用独立执行入口，采集任务未启动。
五语种训练混合仍各17分+中文10+英文5，备料优先级不是更改混合比例。

公开仓库许可声明只作为来源筛选证据，逐条录音授权、真人来源与质量没有由此自动通过。
新候选VietSuperSpeech的README声明MIT而Hub card未填license，自动转写、原录音授权和身份未验；
隔离为metadata-only，不入registry/下载任务。CV22 VI不配未经验证的ref；YODAS MS不冒充已核马来西亚口音。
用户四地区短剧仍只引用`docs/corpus_sourcing.md`的唯一登记，不重复索要或下载未核授权素材。

## 训练决定

值得准备下一轮提升，但不是直接给r8加epoch。新片已验证额外救援生效仍有严重异常，
先做固定输入base/r8归因；r8更差则先评回退/强度，可靠参考下两者仍失败再做定向数据实验。
当前旧12条与新增12条模型计划均未生成；所有语种的历史数字、原模型与冻结音频不改。
新片细节与分层优先级见`docs/video_review_20260923_vi_mtg6jbj.md`。

## A/B正常片段候选（同日追加）

按用户新要求，已把refprep、leveldn、cleanup三个既有包里人工标“无明显异常”的10条WAV复制到
`work/for_train/ab_candidates/`：ID 2条、TL 8条，共39.02s、约3.6MiB。原WAV/反馈/评测未改，
新增真人录音仍0；这10条不是已验高质量训练正例。1条声学听评正常且已有自动文本pass/无severe，
其余9条存在自动风险或文本QC未收全，仍待复核；10条语言、授权及身份均未验。
emotion两侧明确异常，虽偏好B也不纳入正常候选。

导出器`scripts/collect_ab_candidates.py`读取既有`feedback.json`及其冻结文件SHA，
核对实际WAV与请求文本，按音频SHA去重；逐包/反馈版本写`annotations.jsonl`。
记录试听代号、文本/语种、时长、声学标签、配对备注/偏好、自动风险、有效seed/CFG/步数及证据SHA。
不复制完整请求、密钥或base64；请求model名不冒充加载权重身份。哈希错配失败，不覆盖旧快照。

```bash
.venv/bin/python scripts/collect_ab_candidates.py ../dubbing_intelligence_service/work/audio_ab/20260921_internal_cleanup
.venv/bin/python work/for_train/validate.py
```

后续每次A/B收到反馈后执行导出；仅归档已有本地片段，不增加生产合成/识别时间。
合成候选固定`training_eligible=false`、`split=development_only`，且使用`clip/target_text`避免被当作
训练`audio/text`直接加载。已听已调样本也不能改称独立盲测。真正真人候选仍需录音来源/授权、
准确转写、同人证据及与历史评测的身份/session/PCM/文本隔离，不能因放进for_train就自动获准。

`work/for_train/`已gitignore，候选音频/本地证据不随push发布；推送导出脚本、测试与此状态文档。
本轮导出器回归及微调仓全套153项通过；备料池自检核实10条候选、已批准训练音频0。
