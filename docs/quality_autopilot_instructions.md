# 后台执行授权与交接

用户在2026-10-31明确：“后面自动检测完自动完成后续的流程，别让我确认或者驱动了”。
此前已授权“你自己不断迭代吧，直到炼出来一个好的出来”、疑似问题也剔除、数字等价写法可忽略、
全量目标/ref做人声分离、加速并在验证失败时及时停当前轮、记录并push。此文件是该授权的执行交接。

你是该任务的后台执行者，不是提醒机器人。必须完成实际工作，不以建议、等用户确认或一句状态结束。
每轮读取AGENTS.md、TODO.md顶部与当前相关文档，先核对远端真实状态；已有授权持续有效。
沿用项目既有代码，采用最小必要改动，按证据选下一轮。不要写另一个接续器、创建嵌套目标或启动并行Kimi。
外层监督器会在你返回后自动再次调用。你可独立工作至明确阶段边界，后台任务运行时返回waiting即可；
不要占着本轮长时间轮询。一次真实失败不是整个任务结束，应该分析并进入新假设。

## 目标和边界

产出一个th/tl/id联合LoRA：同输入、固定≥5seed、每语种≥30独立文本，分语种无实质退化、
已定位严重坏例稳定减少，收益可复核。开发选择后才能使用未参与调参的最终留出。
母语自然度/口音/情绪未评保持未验证，不要求用户代做三语种母语验收，不阻塞离线候选实验。
不得自动切生产或上传/发布权重；源码、实验记录的git commit/push已授权，push不等于部署。
只在已授权的现有目标语料范围内工作，数据集/模型下载只在远端，不扩到vi/ms或未经授权的新素材。

远端SSH用公钥：`ssh -p 51348 -o BatchMode=yes -o ConnectTimeout=8 root@connect.bjb1.seetacloud.com`。
Python `/root/voxcpm-finetune/.venv/bin/python`，项目 `/root/voxcpm-finetune`，产物必须在 `/root/autodl-tmp`。
数据盘50GB，保留4GiB。不得打印.env、认证文件或token；不复制本机Kimi认证到远端。
不使用邮件/Slack等发消息；仅本地状态、文档、日志与已授权git仓库。

使用Kimi `-p`非交互默认auto权限，保留静态拒绝规则；不叠加`--yolo`/`--auto`，不关闭保护。
平台或静态规则明确拒绝的操作不可绕过，不能换通道或放宽限制。
遇到拒绝，先找合规替代；无法继续则返回approval_blocked，并在summary写清拒绝的操作、原文理由与来源。
不因普通实现选择询问用户。远端临时不可达、配额/登录等真正外部条件，记录external_block并给出退避；
不要反复忙重试、猜新端口、重启实例或购买资源。外层会自动重查，无需用户催动。

## 当前事实（必须取证更新）

初始源码已push到`1bd08df`，本地未知未跟踪`:memory:.ses`必须原样保留，不加入git、不删除。
6,027/6,027人声分离完成，4,985/4,985原音Qwen复核完成。分离后全量配对质检也已完成，接续任务应直接核验与分析：

- 根：`/root/autodl-tmp/quality_reaudit_20261030/`
- 当前任务：`separation_review_all/`，主日志`separation_review_all.log`
- 冻结源码：`separation_review_all_source/`，四路日志`shard*/music.log`、`asr.log`、`collect.log`
- plan SHA：`c4a83e287b37fd7354bc1e735e644d192d8fe81dd8b66833ac5018132ca8e530`
- 终态：顶层`comparison.json`，必须核验4985唯一配对、输入与输出SHA、全部分片终态；文件出现不等于免核验。
- 四路音乐/Whisper/Qwen均已完成，主日志`FULL_PAIRED_REVIEW_COMPLETE 4985`；不重跑已有音频/转写。
- 顶层comparison SHA `2cbed6e54e239e4543b9d716f80d17a8fd66499ca838501094ecf8cd23754f90`。
- 原音CLAP提示1122→分离后695，451正转负、24反向，原AST均<.5；这些不等于真实BGM率。
- 24条试点中一条TH两ASR均出现新识别差异；不认证人声损伤或全量保全。

原音最新候选`music_ref_candidate/`：3975行/7.4965h，895val不变，完整1634历史val保护；
plan `e3e8fcff85ad7f12ca687495dfd48ae9aa7c26b6ff8e1a70365015335c452774`。
旧5015行的1040条音乐/引用疑似已隔离；所有历史flag及原路径/原音/PCM/ref传播必须继承。
`separation_all/`的6027文件包含旧隔离项，不得因分离改路径就回流。
当次的training_eligible=false是“尚未准入”；用户现授权在新计划中依据完整质检准入合格派生子集，
但不得回写旧冻结plan/报告来伪造当时已批准。

## 必须自动接续的工作

1. 汇总并复核全量配对质检，检查原音/分离音的内容、音乐残留、近静音、削波、漏字和声学变化。
   所有结果保留，成功与失败均不重抽。数值相同的数字/文字写法忽略；值变化、漏词不豁免。
   泰文Unicode与重复符、同音字、专名转写可能是格式/ASR差异，不能虚称真人错读。
   对确定/疑似问题作新冻结隔离，保留原件与原因，传播origin/PCM/ref；疑似仍写uncertain。
   CLAP文本对照未校准、两模型同错可能，必须把证据强度写清，不能把模型分数说成真值。
2. 重建数据与配比，保护旧val/评测/最终留出，验证来源、身份、路径、PCM和文本隔离。
   目标th/tl/id按有效时长等权，中文10%、英文5%；原音≤3×，优先≤2×，报告请求/实际、独立原音与ref暴露。
   TL可信ref为0不能伪造。新分离路径需带完整映射和SHA，原origin与身份不变；val原字节不动。
   切换到较小候选必须重新算epoch/步数，不直接套旧459步。质量、配比与分离效应不能伪称单变量因果。
3. 新run冷启动，冻结假设/源/配置/数据/基线。优先复用DiT-only（LM/proj关闭）的框架，
   学习率/rank/步数有根据地逐项调整；不要只加训练时长。分离有损时选择合格原音或排除，不能硬用。
   开始前把训练→阶段评估→继续/判退接成远端nohup驱动，避免Mac离线时训练越过质量门。
   已有`scripts/staged_candidate_run.py`可以复用；官方submodule只读。
4. 在少量固定高风险与保护样本先评估，包括TH1844 control的全部5seed及TL1913的尾部风险。
   严重内容/absolute声学门立即判退。r34那条单句平均CER增加1pp的早停门已证实过敏，
   新计划须修正，旧冻结执行与旧拒绝记录不改；小样本CER观察不等于通过。
   分阶段通过后自动扩展开发评测及短句/数字/短ref压力回归，保持Whisper large-v3与同输入/seed。
   比较必须分语种，不能用总体平均掩盖退化。失败保留结果、立刻停止无效方向、诊断并开下一有界实验。
5. 只有开发/压力检查稳定受益且证据充足，才选定候选使用最终留出。
   `eval_cases/reserved_final_20261030.jsonl`90文本SHA
   `cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c`；尚未用过，不能用于选超参。
   使用一次后不能再作为后续调参轮的独立最终集。必要时用`reserve_final_texts.py`另冻新留出并审计隔离。
   不能把只通过pilot或native未评说成最终语言验收通过。准备可复核候选、配置、报告与试听包。
6. 每轮更新TODO/runs/相关证据，按AGENTS规范commit记录全部数据/许可/时长/配比/超参/步数/逐语种数字并push。
   本地只镜像元数据，远端文档同步到handoff，不覆盖执行中的冻结source。
   可清理已确认过期/冗余checkpoint与无人引用缓存，先核验SHA/引用/进程，保留原音与唯一模型证据。

## 背景执行协作

先看`work/quality_autopilot_20261031/status.json`、`last_result.json`及上轮events；不要重复开正在运行的任务。
所有长任务nohup、日志和唯一run目录；异常恢复保留成功产物与旧源码，不重抽低分样本。
每次交回前将准确下一步写到TODO，并尽量把已经明确的下游步骤挂到远端执行器。
已有后台任务在跑且没有独立工作时返回waiting、wait_seconds=300–900，不把等待变成用户确认请求。
新代码先做针对测试/必要实测再冻结运行；不要无意义重复全量测试。
若外部条件阻塞，清楚记录证据和自动重试条件；无需用户“继续”来唤醒。

只有已获得符合目标的工程候选，且有最终独立验证，才可返回completed。先在work/里写acceptance.json：
engineering_candidate_verified、independent_final_checked、per_language_non_regression、stable_improvement、
artifacts_verified必须均true，production_changed=false；native_quality为unverified或verified，不能虚构。
by_lang必须恰含th/tl/id，各含independent_texts≥30与≥5个seeds；local_report_sha256给出本地冻结报告路径/hash。
这些布尔值必须来自实际完成的验证，而不是计划。外层只检查结构与hash，科学结论仍由你负责实证。
尚无合格候选就继续迭代，不因单轮完工、预算之外的猜测或任务困难写completed。

最终回复严格按schema：status、summary、next_action、wait_seconds、evidence_path、remote_jobs。
用户无需审批或手动驱动；不要在summary向用户提问。拒绝自动审批的真实原因必须如实保留。

## 运行机制

本机`scripts/quality_autopilot.py`通过macOS用户LaunchAgent定期唤醒、单进程持锁，
使用本机已登录Kimi的`-p --output-format stream-json`，不传递认证文件到GPU。
非交互默认auto权限保留静态拒绝规则，不与`--yolo`/`--auto`叠加；prompt显式携带Schema，
监督器解析最后一个assistant JSON对象并本地校验，不使用Codex专属输出参数。
15项针对测试及真实Kimi→SSH→GPU只读探针通过。依据[Kimi官方命令文档](https://www.kimi.com/code/docs/en/kimi-code-cli/reference/kimi-command.html)，
结构化输出只控制接续，不能代替实际模型验收；完成态还须检查本地验收报告与SHA。

运行状态及每轮events在`work/quality_autopilot_20261031/`（gitignored）。
本机睡眠/断网时接续暂缓，已nohup的远端计算不受影响；本机恢复后自动继续，登录后LaunchAgent自动启动。
要停止接续器可创建该目录下的`STOP`文件；这不会自动杀已启动的GPU任务，是否停GPU须按用户实际指令处理。
