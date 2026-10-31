# TODO：配音质量改进（drama 非硬依赖）

## 当前最高授权：自主持续迭代至合格候选（2026-10-30）

用户2026-10-31再次明确：检测完要自动完成后续流程，不等确认、不等用户催动。
后台接续器`scripts/quality_autopilot.py`与[执行交接](docs/quality_autopilot_instructions.md)已迁移到Kimi Code，
15项测试及真实Kimi→SSH→GPU只读探针通过；非交互auto权限保留静态拒绝，不使用权限绕过或关闭保护。
每轮分析、过滤、训练、阶段判退、扩评与记录/push由后台接续，直到可复核工程候选；
本机LaunchAgent安装/运行状态以`work/quality_autopilot_20261031/status.json`为准。
未评母语项仍未验证，不自动部署；真正自动审批拒绝不绕过。工作状态与新结果持续回写本节。

用户明确「你自己不断迭代吧，直到炼出来一个好的出来」。允许按证据连续推进新的有界诊断和离线训练，
**不再逐轮索要确认**；覆盖此前“只诊断、不自动启动下一轮训练”的阶段约束，但不回写旧冻结plan的授权字段。
每轮仍单独冻结假设/数据/预算/基线；及时停止无效方向，继承所有历史隔离，记录并push方案与结果。
用户追加要求加速：先少量固定风险样本与保护样本，出现明确失败及时判退，不等全套跑完；失败产物保留。
目标为同一LoRA的th/tl/id分语种不退化、严重坏例稳定减少、≥5seed可复现收益；最终候选需另保留未参与调参的干净回归。
母语自然度/口音未评仍记未验证；源码push不等于模型部署，不凭均值或停止等待时间自动晋级。
102条诊断及80条LM/DiT拆分均已完成。超参数审查后r32已淘汰，r33仅DiT训练的三阶段小门均通过；
扩展开发验证在290条因新增疑似多读淘汰r33；r34仅将学习率减半，151次更新后被过敏的单句CER门停止。
原停止记录保留；补充诊断在35条因TL实际1.02s尾静音停止。r33/151也在新增5条后因两seed疑似多读停止。
用户随后要求审查数据质量、忽略等价数字写法，并尝试模型去背景音/杂音。
本轮6027个目标/ref音频已全部分离，4985对全量质检、物理检查与新候选构建均完成；
最新训练候选1881行/3.541h；r35/r36/r37均在40更新的第一声学门停止，原音对照与学习率减半均未消除失败。
r38在80更新新增疑似多读后停止；r39关闭dropout后40/80更新门通过，但118更新第3条又新增疑似多读，93条输出及终态已核验。
已接续80更新快照的300条有界诊断，复用45条、最多255新；无新训练，不改118更新拒绝。旧配比方案不再使用。
80更新诊断终态在197条因ID1973 plain的两条严重配对CER退化停止，505产物SHA和CPU声学已核验；103条不补抽。
`r39_punct_20261031`仅正文括号→逗号仍触发声学退化，不训练，不修改原拒绝或生产文本；终态已核验1301输入、259源码、89份产物SHA及40组配对CPU声学。
`r39_u40_20261031`在150条因TL1855 plain/s43新增疑似多读停止，105新+45复用、余150不补抽；368份产物SHA及150对CPU声学已核验。
40/80权重算术中点在首条ID1973 plain/s45即新增疑似多读，CER .0479→.6527、len_ratio .958→1.521，余29不补抽；1935输入及四视图CPU声学已核验。
u40推理侧强度0.5的三句诊断30/30完成并通过六组门与终态核验；TL平均CER仍6.36%→10.86%，ID有声比例下降，不能称三语种非退化。
同强度300条扩评在77条因ID1816 control/s45新增疑似多读停止，47新+30复用、余223不补抽；183份产物SHA及154份配对CPU声学已核验。
已接续同u40权重、仅推理强度0.5→0.25的四风险条件诊断，60旧端点复用/≤20新，plan `bdafe265…7cb`；99项测试及实际0.25 scaling检查通过。
ID开发缺口已另冻2文本/4条件行并核验隔离，原风险句和最终90文本不动；新文本未推理。
最终留出尚未使用。不恢复旧r31的900条任务。

## 2026-10-31：全量分离复核接续，构建有原音回退的候选

- [x] 实际SSH/GPU/磁盘核对完成，旧分离/两ASR任务均已退出；4985对顶层plan与comparison SHA符合交接。
  全量元数据出现125文件分离后RMS降低>6dB、20文件无能量边界（ID16/TH4），
  分离侧Qwen有2条未EOS（原音0条）；不能整体用分离音替换原音。CLAP1122→695仍只是未校准提示。
- [x] 复用既有核验、VAD、PCM/身份/文本隔离函数，新增有限候选构建入口
  `scripts/paired_quality_candidate.py`/`.sh`；22项最新针对测试、shell/compile/diff通过。
  全量重建四分片原始结果核对顶层内容，逐音频保留旧ASR；新增CPU VAD/削波/PCM检查，不重抽、不改标。
  数值全文等价和NFKC仅作诊断豁免；共同ASR差异/疑似漏字仍为uncertain，不认证错读。
- [x] 初版在候选目录创建前将2条未EOS误当作整个复核未完成而停止；原源码/日志保留。
  v2将它们保留为失败输出并拒绝对应派生音，不补转写；不是为了低分重抽。
- [x] **已完成并复核4998份产物SHA** `/root/autodl-tmp/quality_reaudit_20261030/paired_candidate_20261031_v2/`，
  日志同根`paired_candidate_20261031_v2.log`，冻结源码`paired_candidate_20261031_v2_source/`。
  顺序自动执行：全量SHA/四分片终态核验→4985对物理检查→新疑似隔离及origin/PCM/ref传播→
  原音/派生映射→时长混合→全部隔离断言→`completion.json`。出错保留成功产物，不能直接重新启动同目录。
  历史排除不回流；有损派生版本单列拒绝，只保留独立检查合格的原音；val895字节及1634完整旧val保护不动。
  新2144个源文件疑似传播后排除2028训练行；保留1947行原池（1601派生/346原音），拒绝1150派生版本。
  平衡训练**1881行/3.541h**，ID582/TH704/TL346/zh169/en80；实际时长份额28.304/28.341/28.334/9.992/5.029%。
  独立原音1637，原音最多2×；ref475次/435独立、最多3×，TL ref仍0。三份原池/派生池/混合隔离均通过。
  plan `60789ed48d2c4c4c4cec3126c3f7c787b1fb75dc52d186fe7b82e434f2781a82`；completion
  `ed647b04198e11da6ee2a7ce1ea1aac8d3c49a7672dad58b2d1d3d6199f56b28`。
  独立核验`paired_candidate_20261031_v2_verification.json`及本地`work/quality_reaudit_20261031/`镜像保留。
  任务已退出，磁盘剩余约5.86GiB；未开新训练、未使用最终90文本。
- [x] r35复用`staged_candidate_run.py`，新增`dit_paired`配置及入口`staged_paired_candidate.sh`；
  38项针对测试、shell/diff通过。远端根`/root/autodl-tmp/r35_dit_paired_20261031/`，
  2026-10-31已nohup连接SHA预检→冷启动训练→阶段评估，日志`driver.log`，源码冻结在`source/`。
  DiT-only、LM/proj关，保持r33的rank/alpha64、lr1e-4、dropout.05、batch2×8、wd.01、clip1、seed42。
  按1881行重算预算118更新/名义1.00372epoch、warmup12；40/80/118更新后原训练进程等待门禁。
  每门45条上限：TH1844 control全部5seed（44先）→TL1913 control全部5seed（43先）→原风险/保护样本。
  新计划小样本平均CER仅观察；新空转写/近静音/疑似多读、两条严重CER差异、absolute尾/时长/有声比例仍判退。
  失败不再更新、不重抽，保留4GiB；旧r33/34源码和拒绝记录不改。
  plan `c8535a2182342b4fea047328840f84836713a986b21a1e637fd2442663df972b`；18,447输入检查通过，
  203份本轮源码本地SHA对照通过。实际40次更新、val初值1.041568；CUDA bf16、loss/grad有限。
  本地`work/r35_dit_paired_20261031/launch_verification.json`保存启动证据；连接曾有间歇DNS失败，
  后续同步恢复，若复现则退避重查既有任务，不重复启动。
- [x] **r35已拒绝并核验**：TH1844 control的5seed后触发`stage_acoustic_regression`，剩余40条及后两门未生成。
  CER base .254%→0%，无疑似多读/少读；平均时长5.056→5.984s（+18.35%）、有声比例.88310→.81664，
  尾p90 .14→.18s，拒绝来自时长/有声比例，不是尾静音或CER。s43时长4.80→7.36s、内部低能量段最长.18→1.22s；
  s42时长5.28→6.72s、内部低能量段最长.78→1.46s；只是声学观测，不认证母语停顿好坏或分离因果。
  18,447输入、5WAV/claims/items及完整snapshot/latest SHA、原WAV采样数/声学重查通过；96个optimizer状态全40，
  6,684,672 DiT参数，未评分WAV0，GPU已释放。`run/verification.json`
  SHA `1237bce3abfa8abcb16564d68aefaa1ab9fbc29d5816d33fbb0ca4e0df475068`；不续训，不补剩余样本。
- [x] **r36原音对照已接续**：`/root/autodl-tmp/r36_dit_original_20261031/`，入口`staged_original_candidate.sh`。
  只把r35同1881行的目标/ref派生路径逐条换回已独立筛查合格原音；逐行与原池完整元数据相等才准入，
  行序/重复/文本/身份/ref配对/时长配比/优化器/118更新调度/质量门全部保持，不新增或回流历史隔离。
  每轮原音≤2×、ref≤3×，TL ref0；新训练仍冷启动。39项针对测试、shell/diff通过。
  远端nohup已连接失败证据核验→新plan→训练→阶段评估，CUDA bf16已启动、日志到step30。
  plan `d63c55bd5d982b8c509903405fdcf5edcbf9a4c1c5e4c5772ad7729bccbf48be`；
  PID driver90739/trainer90778，本地`work/r36_dit_original_20261031/launch_verification.json`记录逐行及源码SHA对照。
- [x] **r36终态也已拒绝**：40更新、TH1844 control五seed后同一时长/有声比例门失败，ID/TL未评。
  CER base .254%→0%，平均时长5.056→5.888s（+16.46%）、有声比例.88310→.81412、尾p90 .14→.18s。
  原音对照只证明未消除这一个风险文本的声学失败，不外推三语种效果，也不证明分离或学习率的因果。
  GPU0MiB；18,476输入、5WAV/claims/items、完整快照/latest SHA及CPU原WAV声学复查均通过，
  96个optimizer状态全40、6,684,672参数，未评分WAV0；日志`r36_dit_original_20261031/verification.log`。
  verification SHA `6d5ee04be14fe470725ce6ec89045796176e34c1103646dbb58b55a0a92dd7c2`，本地`work/r36_dit_original_20261031/analysis.json`。
- [x] **r37本地准备阶段（历史记录，启动见下项）**：现有`staged_candidate_run.py`新增`dit_original_lr5e5`与
  `staged_original_lr_candidate.sh`；仅lr1e-4→5e-5，其他行序/配比/DiT-r64/.05/batch2×8/118更新/warmup12/阶段门保持。
  显式绑定r36 plan及verification SHA；train/val/flagged/mix/jobs/base_items必须与r36字节相同，
  除路径与学习率外配置必须相同。32项针对测试、shell/diff通过；配置在`configs/lora_joint3_r37_dit_original_lr5e5_staged.*`，
  其中plan记录标明`remote_prepared=false`，不是远端训练plan。
  本轮SSH曾成功核验r36 verification SHA、GPU0MiB/无训练进程和磁盘5,779,718,144 bytes（约5.38GiB），
  随后3次SSH均报`Could not resolve hostname connect.bjb1.seetacloud.com: nodename nor servname provided, or not known`。
  这是域名解析外部阻塞，没有审批拒绝；未清理远端文件、未创建r37执行目录、未同步新源码或启动训练。
  证据`work/r37_dit_original_lr5e5_20261031/preparation.json`；下次自动重查，退避600秒，不需用户驱动。
- [x] **r37已恢复接入并启动**：沙箱内SSH一次DNS失败后，标准自动审批允许同一主机/端口SSH；没有审批拒绝。
  重新核r35/r36已拒绝、GPU空闲且r37不存在。先审计52份产物与进程/打开句柄，再将10对同字节checkpoint原子硬链接；
  回收160,653,312 bytes（153.21MiB），所有引用路径、SHA及唯一模型证据保留，未删原音。
  证据`/root/autodl-tmp/storage_dedupe_20261031_r35_r36/{audit,completed}.json`，本地镜像在r37工作目录的`storage_dedupe/`。
  新source同步250份文件并逐SHA核验；同步后可用5,937,025,024 bytes，超过5.5GiB启动预估，执行中4GiB底线不变。
  新plan `bcbe4df30084a64bc8e02e5e367efb54f36ea0f3c71b64cf349a190a0f96a0dc`，18,716项输入通过prepare/run；
  205份冻结源码本地对照通过，六份train/val/flagged/mix/jobs/base_items与r36字节相同，配置数值仅lr1e-4→5e-5。
  driver94746/trainer94793，实际40更新并停在gate39等待五seed评估，val初值1.041568；暂无质量通过结论。
  沿用上轮32项针对测试的相同源码；TensorBoard/训练日志保留，wandb桥接服务403 Forbidden，在线同步未完成。
  本地`work/r37_dit_original_lr5e5_20261031/launch_verification.json`保存启动与磁盘证据，旧本地预备记录不回写。
- [x] **r37终态已拒绝并核验**：40更新、TH1844 control五seed后时长5.056→6.112s（+20.89%）、
  有声比例.88310→.80196，触发相同声学门；CER .254%→1.012%、尾p90 .14→.18s，疑似多读/少读均0。
  s43时长4.80→8.16s、s42为5.28→6.72s；最长内部低能量段.18→.74s、.78→1.52s，不能只解释成多留尾部。
  18,716输入、5份WAV/items/claims、完整snapshot/latest SHA、CPU原WAV声学核验通过；96个optimizer状态均40、未评分WAV0，gate79/118不存在。
  verification SHA `e9a8ab4c5d47180800400f26732f59de835f32b365df37e43cbd49ac16469439`，本地`work/r37_dit_original_lr5e5_20261031/analysis.json`。
  训练/评估/两项CPU诊断均已退出，GPU0MiB；可用5,681,758,208 bytes（约5.29GiB），不再让本轮占用GPU。
  CPU复用`spectrum()`核三个停止快照各48个DiT模块：r36→r37更新Frobenius中位.081656→.043473（比值.5324），
  r37 rank95中位26.5、前32方向能量中位96.69%；幅度下降没有消除失败，不继续机械降lr。
  容量审计SHA `c9af5abb6a8b1923c6f56d3fb98f13bf2a187a21ea71e601b7631b93e43abba1`；无新音频或重评分，ID/TL未评，不认证三语种质量。
- [x] **r38已实现并启动后台驱动**：沿用r37工作目录`next_hypothesis.json`冻结假设；
  根`/root/autodl-tmp/r38_dit_original_lr5e5_rank32_20261031/`，日志`driver.log`，driver PID98271。
  固定同1881原音行序/配比/身份/ref/基线/jobs、lr5e-5、seed42、118更新/warmup12，只将DiT rank/alpha64→32、alpha/r保持1。
  复用`staged_candidate_run.py`与新入口`staged_original_rank_candidate.sh`，绑定r37 plan/verification及六份数据/任务字节；
  38项针对测试、shell/compile/diff与251份远端源码SHA全部通过；nohup已串联完整输入预检→新plan→冷启动→阶段门。
  plan `7cc5dde336ae8d5642c27e3309d142ced42d37e79a15c6dd3939b676458fdd17`；18,957项输入经prepare/run核验，
  206份冻结源码与本地SHA一致；六份数据/基线/任务文件与r37字节相同，配置数值仅rank/alpha64→32。
  trainer PID98329已实际40更新并暂停于gate39五seed评估；val初值1.029398，尚无分语种通过结论。
  r37真实终态、26份输出SHA和空闲GPU已复核；不重跑旧结果。输入预检中尚未生成run目录时，元数据同步一次返回目录不存在，未影响远端执行。
  rank改变会改变初始化RNG消耗，谱能量不证明低rank重训质量；旧r32是LM+DiT/旧数据/lr1e-4，不能据其失败否定此对照。
  3,342,336参数按FP32权重/Adam两moment保守计算：三阶段加latest四份共164,626,432 bytes预算（含序列化余量），
  另留768MiB给评估/运行时/元数据；启动门5GiB，实际同步后5,677,703,168 bytes，执行中4GiB底线不变，无需删除/硬链接。
  本地证据`work/r38_dit_original_lr5e5_rank32_20261031/launch_verification.json`；本轮SSH间歇DNS失败后标准自动审批接入成功，无审批拒绝。
- [x] **r38终态已拒绝并核验**：40更新45条原阶段门通过，80更新的TH1844 control第二条seed43触发新增疑似多读，余43条及118更新未执行。
  seed43在80更新时CER 0→.5443、文本长度比1→1.544、实际时长4.80→9.28s；是自动风险证据，母语未验证。
  首门45条CER ID/TH/TL base 0/5.740/2.460%→candidate 0/4.360/2.790%，仅1/4/1个独立文本，不能认证三语种收益。
  首门TH1844五seed平均时长5.056→6.336s，seed43实际10.88s、speech_ratio .954→.7901；
  组CER变化>.01导致“内容近似不变”的时长门不适用，平均speech_ratio下降.03672未到.05；原pass记录保留，不当稳定改善。
  18,957输入、47WAV/items/claims、两个完整snapshot及停止latest SHA、CPU时长/有声比例/边界重查通过；
  96个optimizer状态分别40/80，3,342,336参数，未评分WAV0、gate118不存在，GPU0MiB。
  verification SHA `d628f4239e51473532f16322cd646199404af46436b3a3f515686afd959de63c`，本地`work/r38_dit_original_lr5e5_rank32_20261031/`。
- [x] **r39单变量方案与代码已冻结**：仅DiT dropout .05→0；官方v2配置为0，实作在LoRA输出支路dropout，检验该训练随机项的敏感性，不预设因果。
  相同1881原音、rank/alpha32、lr5e-5、batch2×8、118更新/warmup12、40/80/118门及45个任务，绑定r38六份数据/基线/任务字节与终态SHA。
  新计划额外在完整风险组后检查逐seed非数字音频：时长>同输入base的2倍且speech_ratio下降>.10时判退，不依赖CER。
  原门与旧报告不改；这是评估协议补漏，不是第二个训练变量。回查旧首组只标出seed43，49项针对测试及shell/diff通过。
  198份产物、引用/打开句柄/停止进程审计后，r37/r38的10对重复快照原子硬链接回收120,545,280 bytes，全部路径/SHA保留；
  可用5,466,808,320 bytes，4GiB底线保持。证据`storage_dedupe_20261031_r37_r38/`；未删原音或唯一模型。
- [x] r39首次长目录启动遇到`OSError: AF_UNIX path too long`，数据加载器挂起，无已评分WAV/完整checkpoint。
  已对实际controller发SIGINT，由原finally停止trainer；失败源码/plan/日志保留，不重写。
  同实验改用短目录`r39_drop0_20261031_v2/`，训练参数/数据/质量门不变，启动前新增实际Linux socket bind自检。
- [x] **r39 v2已实际启动**：98字节Linux socket bind实测通过，256份源码SHA、19,346项prepare/run输入、172份r38父输出SHA均通过；
  六份数据/基线/任务字节与r38相同，配置数值仅dropout .05→0。plan `9b40d522071727eef4163ea70b5662098745c3c27617cd3103ee49074e631b5e`。
  driver105392/trainer105447，CUDA bf16已完成40更新，停在gate39五seed评估；启动前5,444,493,312 bytes可用，4GiB底线保持。
  本地`work/r39_drop0_20261031_v2/launch_verification.json`保存核验；无本轮质量通过结论。
- [x] **r39首门完成并核验**：40更新45条通过严重内容/absolute声学及新增逐seed极端声学门，训练自动继续至80更新并暂停评估。
  19,346输入、145份输出SHA（含45WAV/claims/items、完整快照）与CPU原WAV时长/有声比例/边界重查通过；
  96个optimizer状态均40、3,342,336参数，无未评分WAV；本地镜像95份元数据SHA通过。
  首门CER ID/TH/TL base 0/5.7364/2.460%→r39 .247/7.820/3.362%，尾p90 .14/.14/.12→.18/.16/.10s。
  仅1/4/1独立文本，五seed；疑似少读0/1/0→0/1/0、多读均0。CER上升是观察，不是非退化或收益证明。
  TH1844 control五seed均未触门，均时长5.056→5.312s、speech_ratio .88310→.84532；seed46仍4.80→6.08s、.8745→.7327，保留风险观察。
  verification `d31ea296deafbf4fc412c81b71d638a1ddbb58d9ae1ccb8f888ac4e48b4275a7`；
  本地`work/r39_drop0_20261031_v2/{verification_gate39,gate39_analysis}.json`。未重转写、未补抽，最终留出未用。
  07:30:48实查driver105392/trainer105447/evaluator109163，80更新门已26/45、暂无判退；两份快照SHA及256份冻结源码再核通过。
  可用5,091,758,080 bytes（约4.74GiB）；不清理在用文件，执行器继续守4GiB底线。
- [x] **r39终态已拒绝并核验**：40/80更新各45条通过；118更新第3条TH1844 control/s45出现新增疑似多读，
  5.12→10.08s、CER0→.8734、len_ratio1→1.873；s44也有额外转写及5.28→9.28s，保留疑似观察。
  45+45+3=93条，余42条不补抽，训练/eval退出；19,346输入、321份产物SHA、三个完整快照及96个optimizer状态40/80/118、
  全部实际WAV的CPU时长/有声比例/边界复查通过。终态verification
  `3a9ab7f53efdad4206fed2646b347ccb73e96d6ee87a826220c8e2b26f53f8ea`，本地216份镜像SHA通过。
  证据`work/r39_drop0_20261031_v2/terminal_analysis.json`；gate79/118 verification为
  `7eefcfb33912448be582d47384f3cd3ce5041228bcea53ac7b1b2a84cf932e1a` / `140db8ca1a326416a841154eee2e546306eeb9cf2670ad520ba646eefb5820a2`。
  80更新同45条CER ID/TH/TL base0/5.7364/2.460%→.247/5.1784/2.952%，尾p90 .14/.14/.12→.22/.14/.16s；
  疑似少读0/1/0→0/2/0，多读均0。最终门仅TH同一文本3seed，CER .4233→40.0833%，不能和完整总体混比。
  实际val日志首1.027584→末1.007363，均随机≤20条，不用它选模型；不认证语言错误或收益。
- [x] **已启动80更新快照诊断** `/root/autodl-tmp/r39_early_20261031/`，入口`expanded_r39_early_diagnostic.sh`，
  复用已有`expanded_candidate_eval.py`的新`r39_early_diagnostic` profile；冻结257份源码，47项针对测试及shell/diff通过。
  绑定r39父plan/终态verification与80更新权重`6cd84f6d2c56d067ae16a9d28c8a01f45c785514b1faa52cafb0cee7098107e5`。
  假设只检验早期快照能否避免晚期退化，不加训练、不重抽45条、不改旧拒绝；300条固定开发诊断（10文本/语种×两条件×5seed），最多255新。
  TH1844/TL1913全部5seed及原风险保留；均值CER仅观察，严重内容/absolute声学/逐seed极端声学仍判退。
  nohup driver116119、workers116146–116149已连接阶段屏障和失败后停止领取；包装shell116118已精确退出，不影响worker。
  新plan `f7e5ca1526e5e79e648f73292f0d61b33b8151e5a1c81a49ce6fb249675cdbef`，632项输入通过；启动日志`driver.log`。
  启动时5,802,303,488 bytes可用，4GiB底线保持。清理前审计74,429元数据、20,436受保护文件与进程/句柄，
  仅删除6个已终止run下36份无引用inductor/triton/numba缓存，释放占用约815MiB，原音/权重/结果SHA不变；
  证据`/root/autodl-tmp/storage_cache_20261031_r39/{audit,completed}.json`。未清理在用目录。
- [x] **r39_early进度快照已核验（184条，非终态）**：632输入、257源码及417份新产物SHA通过，139新+45复用，
  184对base/candidate原WAV的时长/有声比例/边界CPU重查通过；前六个完整阶段严重/absolute/逐seed极端声学门均未触发。
  同完成子集ID/TH/TL CER 4.62/4.45/2.77%→4.79/4.67/2.91%，疑似少读11/1/0→10/2/0，多读均0；
  仅7/6/6文本，其中ID第7文本未满5seed，不认证非退化或收益。verification SHA
  `80f6fcefa627135d3b910773b4172d962048f998c6c6311a763b5af8b29257a9`，本地`work/r39_early_20261031/verification_progress_20261031_0800.json`。
  08:02:55远端仍由driver116119和原四worker运行，188/300；可用5,367,046,144 bytes，4GiB底线保持，未清缓存/权重/数据。
- [x] **开发文本缺口已补齐准备**：复用`reserve_final_texts.py --id-supplement`，冻结
  `eval_cases/development_id_supplement_20261031.jsonl`（ID1783/1961，各plain/control，5seed预算20条/模型）。
  来自已缓存FLEURS test.tsv原文/原revision，CC-BY-4.0，不下载目标音频；承接旧ID1753/1905的参考/控制条件，旧风险句字节不动。
  183份清单/382旧报告、121,577种归一文本排除；574输入SHA及179,721条ID记录精确/≥40字符编辑距≤2检查通过，0冲突。
  28个原无编码风险ID文本+2新=30；只是开发覆盖准备，尚无新基线或候选输出，不是验收或母语证明。
  cases SHA `d1ea31c3974b21fc0674c5aafa00296794cba9cdb5f9ff129e5f97ba4809ae16`；verification
  `274ce4ec291fcd449828393350cf23dd8516fbfa96c582a9655396fda5df512d`，本地`work/development_id_supplement_20261031/`。
  新清单纳入全局训练文本排除；15项针对测试及diff通过，原fresh/最终90 SHA不变。远端依赖原冻结源码，新增脚本/输出在独立目录。
- [x] **r39_early终态已判退并核验**：`diagnostic_stopped/two_large_paired_cer_regressions`，197条=45复用+152新，
  第193条已满足门槛，另保留4条在途结果；余103条不补抽。632输入、257源码、505产物SHA及197对CPU声学通过，未评分WAV0，进程退出/GPU0MiB。
  ID1973 plain/s45 CER .0479→.3653、12.80→19.68s；s46 .2156→.4311、17.28→19.04s。base/s46本已有异常转写；
  同句control五seed base与80更新均CER≤.012，提示输入形态交互，但不认证括号因果或真人错读。
  同完成子集ID/TH/TL CER 4.99/5.57/2.77%→6.17/4.73/2.91%，疑似少读11/3/0→12/2/0，多读0；仅7/7/6文本且TH第7不完整。
  verification SHA `048735b2b6fdaeff2a6ea575ca05971cc4204bd20ba782a91fb5bec6699f7588`，本地`work/r39_early_20261031/terminal_analysis.json`。
- [x] **下一有界标点假设已接续** `/root/autodl-tmp/r39_punct_20261031/`：复用`r31_failure_probe.py`新增显式profile，
  只将ID1973正文括号换逗号，所有词/引号/ref/control/seed不变；原base与80更新各10条直接复用，最多20新（两条件×5seed×两模型）。
  串行先base再候选；同模型对原句、候选对同改句base均逐条严重门，完整条件组继承absolute/逐seed极端声学门，均值CER只观察。
  58项相关测试、shell/compile/diff通过，259份源码冻结；nohup driver129902，日志`driver.log`，计划/启动SHA见本地`work/r39_punct_20261031/`。
  实际1301输入SHA、259源码SHA、80更新权重与20任务/20旧端点核验通过；base的ASR/TTS均CUDA，启动时5,283,098,624 bytes可用。
  plan `e7d384385b525ece154da81d79c68ad7bfe95a50fbf339de4123703959cf8d2f`；launch verification
  `238568c48105f7750d9347e26702cf01d002203c86da644d54618a6c955577f6`。启动核验时0/20，不认证任何新评分。
  不续训、不重抽低分、不改原197条判退、不调整生产normalizer；即使改句通过，也不能认证原输入LoRA稳定。
- [x] **r39_punct终态已完整复核**：20/20新输出完成，20旧端点复用，无跳过任务、无新训练。
  `candidate_vs_original:stage_acoustic_regression`：候选control五seed有声比例原句.860→改句.804，下降.056。
  两条件合并CER：base原句5.51%→改句10.96%，候选12.70%→11.20%；单句改标点没有形成稳定修复，不据均值改判。
  1301输入、259源码、89份产物SHA及原句/改句、base/80更新四组共40份WAV的CPU时长/有声比例/边界全部通过；
  未评分WAV0、driver129902退出、GPU0MiB。verification SHA `938928fc954f1a04a05b35acf193593a23dc8744d6f474291e91463a2bc1b698`，
  本地`work/r39_punct_20261031/verification_terminal.json`；不是原输入模型通过或三语种验收。
- [x] **r39_u40终态已拒绝并核验** `/root/autodl-tmp/r39_u40_20261031/`：仅复用r39的40更新快照，无新训练。
  第150条`fresh_tl_1855_plain/s43`触发`new_suspected_overread`：CER .0695→1.0909、len_ratio .941→2.043、
  时长14.24→31.04s、speech_ratio .8622→.8162；最终105新+45复用，余150条不补抽。
  同完成子集各50条/5文本/5seed，CER ID6.92→6.59%、TH4.82→5.92%、TL2.96→4.95%；仅开发子集，不认证非退化或收益。
  ID1973 plain/s45、s46在40更新CER .0359/.1497，避免80更新.3653/.4311严重退化；但TH1689 control/s45在40更新CER .0238→.875，
  80更新同句.0119；TL1855在80更新.1765且无多读。两个快照在不同文本/seed上互补失败，均不晋级。
  634输入、260源码、368份产物SHA及150对base/candidate原WAV的CPU时长/有声比例/边界复核通过，未评分WAV0、GPU0MiB。
  初版核验器分别因文件名重排改变顺序敏感门、事后补算未记录阶段而断言失败，脚本/日志保留；v3按冻结result顺序及四个实际边界复算通过。
  verification SHA `ffa461073ce47bd5c9ce8c7c61553dad591b478da341ba779bec05f0b8bf56ca`，本地`work/r39_u40_20261031/terminal_analysis.json`。
- [x] **40/80权重中点已在首条拒绝并核验** `/root/autodl-tmp/r39_midpoint_20261031/`：
  对同轨迹全部96张量做Float32算术中点，配置不变；不是实际60更新checkpoint，无训练。
  `fresh_id_1973_plain/s45`首条即`new_suspected_overread`：CER .0479→.6527、len_ratio .958→1.521、
  时长12.8→23.68s、speech_ratio .7966→.7380；u40/u80同句CER为.0359/.3653，中点比两端点更差，余29不补抽。
  64项相关测试、1935输入、263源码、13份产物SHA及base/u40/u80/中点四视图CPU声学复核通过，未评分WAV0、GPU0MiB。
  plan `fd4e5626c4cf5df5e8efb91c988d098e6aff7cfdbb49df4a5bf2d4bd57194562`，中点权重
  `cbe95f2a8f55f7a250fefcd571a9b7da4ad5571b2087c33ba5d89b1381b648b0`；verification
  `f4d4a3047f828c7764b88fdb540be2120f7ff1509349104312596022c604f3ad`。插值方向淘汰，不改旧端点结果。
- [x] **u40半强度三句诊断已完成并核验** `/root/autodl-tmp/r39_u40_half_20261031/`：
  仅在推理侧使用0.5强度，原权重不变；30/30完成、60旧端点复用，六个完整条件组严重/absolute/逐seed极端声学门均通过。
  TL1855 plain/s43由全强度CER1.0909/len2.043/多读，变为半强度.1818/1.064/无多读；TH1689 control/s45由.875降为.0417。
  ID1973 plain/s45、s46半强度CER均.0359；但TL三句集均值CER base6.36%→半强度10.86%，ID平均speech_ratio .8422→.8078，
  不能因严重门通过就称三语种非退化或稳定收益。1168输入、266源码、69份产物SHA及三视图90份CPU声学复核通过，GPU0MiB。
  verification `28c88e1a65afeb1ecae5b585146ac1fb4097c5e0270721cd60062901d7e997a0`，本地`work/r39_u40_half_20261031/terminal_analysis.json`。
- [x] **同权重/强度300条扩评已启动** `/root/autodl-tmp/r39_u40_half_expansion_20261031/`：
  使用完全相同的300 jobs/base，复用仅30条半强度输出，最多270新；四worker各验0.5 scaling/恢复，旧全强度路径不变。
  94项相关测试、267源码、1446输入SHA通过；plan `25ac2ecc30c8e1fb1cdb3045cbe3c5c933ec2ea98bb025a767c73e7027ecbb99`，
  launch verification `08de3143d7be57c09b9e712bba22cef70c7f786a5e2314b1c301cd4fd0033fdb`，启动时34/300（30复用+4新），未触门。
  driver23541、workers23580–23583、pipeline32171；缓存清理审计40,614文件/704,167,936 bytes，1,885份受保护产物不动。
- [x] **u40半强度扩评终态已拒绝并核验**：77条=47新+30复用，第73条ID1816 control/s45新增疑似多读，余223不补抽。
  base→candidate CER .0536→.4821、len_ratio .973→1.402、时长7.68→8.00s；最终另满足两条严重配对CER退化。
  同完成子集ID/TH/TL CER 6.01/2.11/3.60%→7.09/2.04/5.22%，仅3/2/3文本，不能用TH小幅均值改善抵消ID/TL风险。
  1446输入、267源码、183份产物SHA及154份配对CPU声学通过，未评分WAV0、GPU0MiB；verification
  `740743f1f7f30d0c46ca5a55b7d4cf9ddb80cca20463f9847767592691fb6a14`，本地`work/r39_u40_half_expansion_20261031/terminal_analysis.json`。
- [ ] **u40四分之一强度诊断已自动接续** `/root/autodl-tmp/r39_u40_quarter_20261031/`：只改推理强度0.5→0.25，权重不变、无训练。
  ID1816 control优先，再接TL1855 plain、TH1689 control、ID1973 plain，各5seed；base/u40/half各20旧端点复用，最多20新。
  99项相关测试、269源码、1813输入预检通过，实际0.25 scaling及恢复纳入worker核验；plan
  `bdafe265217f7a0eb194eac28f98566b4ca923814e23b608aa8d7d88749ba7cb`。
  远端nohup已串联启动核验→严重/absolute/逐seed极端声学门→终态SHA和四视图CPU声学复核；失败不补抽，四条件通过也不直接晋级。
  下一步只核验既有任务；若通过才另冻同300开发扩评，稳定后再做≥30文本/语种和压力回归。最终90不参与调参，母语未验证，不部署或发布。

## 2026-10-30：数据质量复核、数字豁免与背景音分离

- [x] 重新连接5015训练行/6027目标与ref文件的旧声学、Whisper、AST证据，全部SHA通过；
  无旧已排除项回流。原始1417条CER>.05/537条>.10只是分歧计数，不能一刀切认定错标。
- [x] 独立CLAP复核1087条完成（176复用+911新增）；1031条AST偏高目标中926条两模型共同提示音乐，
  主要为917条GigaSpeech2_ID。40个低AST对照也有7条CLAP正向，故分类器分数不认证真配乐或干净。
  保守隔离926共同疑似、105单AST未解决项、7单CLAP项，并传播原音/PCM/ref，5015→3976行。
- [x] 显式检查1153个在用ref，补发现1个AST偏高ref，单独CLAP也正向；已连带排除对应训练行。
  **最新候选3975条/7.4965h**，只用`quality_reaudit_20261030/music_ref_candidate/`，
  原895val字节不变、完整1634旧val继续保护；累计新排除1040行，原件与所有旧flag保留。
  plan `e3e8fcff85ad7f12ca687495dfd48ae9aa7c26b6ff8e1a70365015335c452774`。
- [x] 用户数字豁免已实现：只忽略数值相同且全文剩余内容一致的数字/文字表达，
  不豁免数值改变、漏词或混合错误。初版91条、扩展明确千分位/中文时间后94条（其中77条原CER>.05）；
  未支持格式仍待复核，不自动删；原标签、旧ASR与评测分数不改。初版冻结报告不重写。
- [x] Qwen3-ASR-0.6B固定revision独立试点120条完成，无期望文本提示、120条均EOS；
  13条旧Whisper高CER被其完整转对，也有双方共同指出的漏词/增词/词序差异，不把任一ASR当真值。
  包装环境独立于训练环境，原large-v3验收不换。详情[数据质量复核](docs/data_quality_reaudit_20261030.md)。
- [x] 四路扩展全部4985个当前目标/ref文件（3975目标+1010仅ref）完成，120复用+4865新转写，全部EOS；
  4985份结果SHA核验通过，`transcript_all/results.json` SHA
  `8b5836e2b3101803c32864052fcb58be1afc2d9ecc1ffe1083b36d905bd70c7a`。
  仍混有泰文重复符/Unicode写法、专名与中文同音字差异，不凭两模型一致自动认证错标。
- [x] 用户要求模型保留原人声、移除BGM/杂音：HDemucs **24/24分离已完成**，
  12共同疑似（8 ID+3 TH+1 ID ref）+12低风险对照（各目标语言4）；峰值显存1118MiB。
  仅写独立实验WAV及被移除部分，输入SHA不变、长度不变；`training_eligible=false`，不回流训练。
- [x] 按用户要求四路完成本轮**6027/6027独立目标/ref文件**的分离（24复用+6003新）；
  覆盖5015行原训练子集，不是50321文件全池。`separation_all/` plan
  `4d93c149d0103076fc60f552c5a7ac29f426dee0fddc9f8a7ca1d5d1533845ed`；
  输入/输出SHA、长度和格式核验通过，全部仍为实验产物，不能用新路径绕过原隔离。
- [x] 24组固定前后对照完成，12音乐疑似中10条分离后不再触发CLAP三组音乐提示；
  12低风险对照的Whisper CER均不变。但1条TH疑似分离后两ASR均出现新的识别差异，
  ID ref两ASR变化方向相反，不能把分类分下降当人声保全证明。原音/人声/残差已保存，试听未完成。
  `separation_review/` plan `24bc3cd8d4d9cb71a33007fda003527d35a72c3e6c04d88c8ddf0a5150fc4e13`。
- [x] 扩展当前4985文件分离前后的音乐/两ASR核验全部完成，复用所有原输出；两条低风险对照新触发CLAP，
  作为新增疑似保留，后续隔离须连带原音/PCM/ref。未认证损伤或配乐；母语质量仍未验证。
  已冻结并四路启动`separation_review_all/`（每路依次音乐→两ASR→汇总），plan
  `c4a83e287b37fd7354bc1e735e644d192d8fe81dd8b66833ac5018132ca8e530`。
  4985原音ASR与12组在用试点结果全部复用；每文件中间结果落盘，失败不自动重抽、不自动入训。
  53项相关测试、py_compile、shell/diff检查通过；24组对照79份元数据SHA及最终留出原字节核验通过。
  音乐部分已全部完成（4985分离AST、9970前后CLAP视图），CLAP原音1122提示→分离后695，
  451条由正转负、24条反向；当前原AST均<.5，故这些是单模型未校准提示，不能当真实BGM率。
  2026-10-30 23:58 Whisper约1000/4985进行中，随后Qwen；GPU99%、约10GiB，不盲目增并发。
  部分报告`handoff_987d0f7/music_review_complete.json` SHA
  `d70b9c54a8ba9fe7af487d40b5c2ae5a0e984fc9689ec53eead8415e338068f2`。未开启新训练。
  四路均`ASR_COMPLETE`，顶层`FULL_PAIRED_REVIEW_COMPLETE 4985`；最终comparison SHA
  `2cbed6e54e239e4543b9d716f80d17a8fd66499ca838501094ecf8cd23754f90`，GPU已释放。
  下一步由后台接续器分析全量前后结果、冻结疑似隔离与新训练计划，不需要用户再驱动。
- [x] 再核Granary-v2官方说明：**仅标注清单、无音频，更不是去BGM成品库**。
  ID/TL主要已有FLEURS、TH另有CV26，旧接入与许可/隔离要求保持，不按46万小时启动全量下载。
- [ ] 清洗完成后重做语言配比/暴露审计及新训练计划；暂停使用旧4681行混合提案。
  当前空闲盘需继续保留4GiB；不能把有限显存占用当作必须开训的理由。
- [x] 核验并清理9份未使用THAI-SER parquet下载缓存，回收4,478,208,969 bytes；
  55,766份元数据无引用、无打开句柄，5份原raw manifest SHA不变；revision/hash保存在
  `storage_dedupe_20261030/thaiser_cache/`。分离完成后磁盘约6.19GiB可用。

## 2026-10-30：优化训练时长，检验r33较早checkpoint

- [x] r34补充诊断35/300停止（5复用+30新生成，265未生成），原因`stage_acoustic_regression`。
  TL1913 control/s43的实际尾静音.10→1.02s；末800ms RMS约−115.18dBFS，确认是近静音尾部，
  不是仅ASR标点差异。623输入/35WAV/30claims-items-new WAV SHA通过，无未评分产物。
  TL仅10输出，按冻结p90算法这一条决定1.02s，不能外推为全语种发生率。旧拒绝与原分数仍保留。
- [x] 同完成子集CER base→r34：ID10条0→.494%、TH15条1.713→1.892%、TL10条2.460→2.378%；
  疑似少读/多读均0→0；均值不能豁免声学门。等曝光r33/151的30条对照见runs，不认证因果收益。
- [x] 独立`/root/autodl-tmp/r33_early_risk_20261030/`已启动验证r33/151：同原数据/lr1e-4/459调度，
  只改变所评checkpoint，不重新训练。原300条开发文本全保留，TH1844两条件×5seed前置，
  复用40条、最多新增260条、四路；严重/absolute声学失败仍停，CER仅按已有口径记录观察。
  母语/最终留出/短句压力门仍待完成，诊断完成不等于晋级。
  初版准备目录`r33_early_diagnostic_20261030/`从未启动推理，0新claim/WAV；为前置风险另冻新目录，旧plan/source保留。
  有效plan `2340d073fb4b35ab69ac2870e341fe1cc9aacb0921f671469bf178c323e22770`；8项针对测试、shell/diff通过。
- [x] r33/151诊断45/300停止（40复用+5新，255未生成），新增TH1844 control/s42与s45疑似多读：
  CER0→.4304/.5316、len_ratio1→1.430/1.532，WAV5.28/5.12→7.68/8.00s；分别约多1.86/2.14s有声。
  605输入、45WAV与5claims/items核验；无未评分WAV，GPU已释放。不晋级，仍uncertain，不能将早期快照称为改善。
  当前配对CER ID10条0→0%、TH25条5.7364→8.5504%、TL10条2.460→2.788%；TH疑似少读1→2、多读0→2。
  结果SHA `f449b7c401d1689bf8801c443011f9c4bf7b0586313696481990f8260d7f4a03`。
- [ ] 下述语言混合提案已被最新质量复核暂缓，不能直接启动：以r33的DiT-only/r64/lr1e-4/dropout.05/batch2×8/459调度为对照，
  不同时改batch/LR。先远端重新审计完整历史排除/val/PCM，冻结新训练与早停方案，再启动。
  元数据提案4681行/9.4921h，ID/TH/TL各约28.3%+zh10%+en5%，原音最多2×、ref复用最大3；
  3589个独立原音，代价为训练中不采样1426条原ID录音（原件/候选池保留）、部分低份额语种重复，
  ref占比约19.85%，TL可信ref与控制仍0。不是新增多样性，也不是合格候选；proposal
  `79875a661b922ce3755e294ffed305778371323e23e03ca7481bbd650bcfe6bb`。
- [x] 发现microbatch不是纯提速旋钮：官方每微批按有效mask取均值，再等权累积，
  2×8与4×4可能改变长短样本相对系数。时长近似/100种排列估计TL系数10.117%→11.797%，
  不是实际梯度贡献；旧训练完全未改。row份额TL7.936%、时长份额14.210%，两者不能混称优化权重。
  证据`hparams_audit_20261030/handoff/batch_weight_proxy.*`；真正吞吐/OOM仍须另做GPU实测。

## 2026-10-30：r34停止核验与学习率补充诊断

- [x] r34实际151次更新、名义.481755epoch；11,738输入、5WAV/claims、完整snapshot/latest均SHA通过，
  96个optimizer状态全151，6,684,672参数；`run/verification.json`，无未评分WAV或后续门。
  原`stage_mean_cer_regression_th`拒绝记录和冻结源码保持；val只有初值1.047530，没有后测。
- [x] 只完成TH1844 control的5seed：CER base .254%→1.266%，增量1.012pp，
  来自seed43的少数字符差异；5条均无疑似多读/少读或声学门失败，平均时长同为5.056s。
  新增的单句1pp门过敏，不符合既有CER噪声/小样本限制，不能据此认定低学习率失败。
  ID/TL原训练门未评；不将这5条当全语种证据。
- [x] 新诊断入口`r34_diagnostic.sh`复用四路stage-barrier调度；每语种10文本×plain/control×5seed，
  复用5条、最多新增295；固定原300条开发选择，保留严重逐条门与absolute声学门。
  CER按现有`CER_NOISE=.05`记录红线/噪声观察，未越线也不证明不退化；终态只允许诊断完成/停止，
  不改变旧拒绝、不重启训练、不自动通过。另固定r33同151次更新的40条，避免与459步比较混淆学习率。
  22相关/447全量测试exit0（两条既有弃用警告），shell/diff通过。
- [x] 已冻结并启动独立`/root/autodl-tmp/r34_diagnostic_20261030/`，plan
  `c52fcf0436ff52769a4a0af49caa83d47b484f8c3f9a8bd0e2e5bec1a46ee48d`。
- [x] 终态35条/声学停止及完整核验见上节，下一项为既有较早checkpoint的有界诊断。
  后续仍需短句/数字/短ref压力回归、最终留出与试听；原90个最终文本未生成、母语未验证。

## 2026-10-30：r33扩评淘汰，r34学习率减半实验

- [x] r33扩评290/300（复用40+新增250）后自动拒绝`new_suspected_overread`，剩余10条未生成。
  `fresh_th_1844_control/s44`：base→r33 CER .0127→.5316、len_ratio .987→1.519、
  实际WAV 5.28→9.60s（1.818×），按既有有声比例多出约3.279s有声；不是仅尾部变长。
  仍记uncertain疑似多读，不认证母语内容。其它4seed未触发该标记。
- [x] 同完成子集CER base→r33：ID100条3.5509→3.3283%、TH100条4.5713→4.4271%、
  TL90条2.3000→2.7500%；均值不能覆盖新严重风险。WER ID4.8529→4.8434%、TL13.8727→14.7819%；
  疑似少读/漏尾11/3/0→11/3/0。绝对声学门此前各阶段均通过，但本轮仍不通过、不晋级。
  290候选WAV及失败base WAV SHA/物理时长、250claims/items一一对应，无未评分WAV；
  `run/failure_analysis.json`保留核验；不补剩余10条、不重抽失败音频。
- [x] r34同基座/5015训练行，仅相对r33将lr1e-4→5e-5；DiT-only、r/alpha64、dropout.05、
  warmup45、cosine459、wd.01、clip1、batch2×8和seed42不变。不能把一次失败当作lr过高的因果证明。
  新坏例control的5seed（44先）排在最前，再接旧40条；各阶段最多45、三阶段总上限135条。
  151/301/459次更新后暂停原训练进程，内容/绝对声学失败即不再更新。
- [x] 11,738输入SHA/预检通过；plan
  `90574851e38af8582a2b9339137d3f887f957e5c942284a700c73cd908e79cb8`。
  根`/root/autodl-tmp/r34_dit_lr5e5_20261030/`，入口`staged_dit_lr_candidate.sh`/共用driver。
  444全量测试exit0（两条既有弃用警告），shell/diff通过；已后台启动并核验CUDA bf16、
  loss/梯度有限，val step0=1.047530仅运行观测，尚无质量结论。
- [x] r34训练/门已停止并核验，详情见上节；仍须扩展开发与短句/数字/短ref压力回归，再使用最终留出。
  只读旧压力报告发现不少WAV缺失且无SHA，不能直接当新配对基线；后续使用新的相邻seed组并冻结新基线，
  旧报告不改。目前未启动压力基线或最终90文本生成。
- [x] 清理144处已停任务的生成式编译缓存，审计56,771个固定路径无交集；
  `storage_dedupe_20261030/compiled_cache_*`保留清单，回收约3.08GiB实际空间。
  权重、数据、checkpoint状态与WAV/报告均保留，磁盘回到约8.28GiB可用。

## 2026-10-30：r33仅DiT完成459次更新，扩展开发验证

- [x] 同r31数据/其余优化参数，仅关闭LM；实际96个DiT LoRA tensor、6,684,672参数，
  459次优化器更新，名义1.464407epoch。val 1.047530→1.008731仅为随机小批观测。
- [x] 151/301/459次更新的40条/5seed小门全部通过，内容与absolute声学门均检查。
  最终pilot CER base→r33：ID0→.370%、TH7.107→5.487%、TL2.460→3.362%；
  尾p90 .14/.18/.12→.16/.18/.24s；疑似少读/漏尾0/1/0→0/0/0。
  仅5个文本，TL增量.902个百分点接近1pp门，不能作完整语言验收。
- [x] 11,726输入、120份WAV、三个完整快照与latest全部SHA核验，96个optimizer状态均459，
  无未评分WAV；LoRA SHA `edfb125a2f19a3ed5a4af4eeb011ca1ca61afabff210d372d4716bc632293def`。
- [x] 已启动扩展到每语种10文本×plain/control×5seed=300条；复用最终checkpoint的40条，最多新增260条，
  4worker按阶段推进，失败后只完成已领取样本，不重抽。保留ID括号/缩写与TH风险文本，余项按ID哈希选择。
  入口`expanded_candidate_eval.py`/`.sh`，执行根`/root/autodl-tmp/r33_development_20261030_v2/`；
  初版启动在0新claim/0新WAV时停止，修复共享control JSON非原子写入；旧源码/plan保留，见初版`run/setup_stop.json`。
  最终90文本不参与本次扩评，原始输入/失败证据不改，未晋级/未部署。
  v2 plan `12e21245f775913548f262519df6da818a8a4eb2ceb2ce79985c92b3e01619e5`，
  602份推理输入SHA，保留r33全训练输入核验的父plan/verification；443全量测试exit0，两条既有弃用警告。
- [x] 扩评终止与核验见上节；r33不通过，不进入最终留出。
- [x] 磁盘整理只移除未被扫描的120份运行元数据引用的Whisper medium下载缓存，
  记录全部blob SHA/版本，可按固定revision恢复；回收1,530,572,613文件bytes。
  `storage_dedupe_20261030/`另存4000旧WAV重复审计（0命中，无修改）。

## 2026-10-30：r32 rank32因声学退化提前停止，补齐训练早停声学门

**接续r33已启动**：`/root/autodl-tmp/r33_dit_only_20261030/`，`staged_dit_candidate.sh`调用共用driver，
同r31数据/全部优化参数，仅关闭LM LoRA、保留DiT rank/alpha64（约668万参数）。
11,726输入SHA/预检与新最终文本隔离通过；plan `d1c19ebed1c74c165751bae49e3752894468698f3f0881ae8abb1f3bbf6efda1`。
仍在151/301/459次更新验证，但已在继续训练前检查绝对尾/时长/有声比例。7项最新针对测试exit0，shell/diff通过。
训练及三个小门已完成，最新扩评状态见上节；实际可训练96个tensor均为DiT LoRA、无LM，
CUDA bf16、loss/梯度有限。冻结source不再改，后续文档只进handoff。

- [x] `audit_training_hparams.py`读取真实optimizer/scheduler/LoRA，384参数状态均459步，最终lr0；
  47记录梯度均<.1（clip1），当前约3618万参数；LM更新谱前32方向能量中位99.50%，DiT95.62%。
  不能把谱压缩等同重训收益。官方rank32/dropout0与历史64/.05差异已记录，默认值未改。
- [x] 当前最长估算576token，microbatch2/4/8均不触发8192预算剔除；仍需另测显存/吞吐，不认证提速。
  固定生成验证替代随机约20条val loss选模型；参数优先级与限制见`docs/hparams_audit_20261030.md`。
- [x] 新入口`staged_candidate_run.py`/.sh，根`/root/autodl-tmp/r32_rank32_20261030/`；同基座/5015条，
  仅r/alpha64→32，保持比例1与其余优化参数，最多459步；完整checkpoint在151/301/459次更新后暂停原训练进程，
  每阶段最多40条5seed开发验证，失败不再更新，通过保留优化器/RNG/迭代器继续。官方submodule不改。
- [x] 11,724份输入核验，plan `9abcdb072759654dd8e5e3e869bf84ef7eecf1b694e548d9ce5d42777245f2f9`；
  435全量测试exit0（两条既有依赖弃用警告），shell/diff通过。已后台开训，CUDA bf16、5015/895条加载完成。
  旧r31过期step0/458/459已清理，回收1.214GiB，完整latest SHA不变；磁盘保底4GiB。
- [x] r32第151次更新完成40条内容pilot，原自动门记录pass，但只读补查发现TH尾静音p90 **.18→1.20s**、
  有声比例 **.9171→.8303**，TL有声比例.9901→.9329；超过既有absolute尾部/有声比例门。
  原自动门遗漏这两类检查，这是执行缺口，不能将内容pass当质量通过。已补齐本地driver，原冻结执行源码/内容判定不改。
- [x] **实际在301次更新时停止**：发现声学失败时第300标签checkpoint已完成、第二阶段尚无输出，
  按核验PID终止驱动/训练/eval/loader/bridge，GPU已释放。仅40条完成、无未评分WAV，150/300两个完整快照保留SHA。
  状态`stopped_quality_gate`，证据根`stopped_quality_20261030/`；未跑满459步，也未恢复旧r31复评。
  该pilot CER base→r32：ID0→.617%、TH7.107→5.945%、TL2.460→3.280%，均非完整语种验收。
- [x] 已启动下一方向冻结LM、仅训练DiT：旧拆分DiT-only的同40条绝对声学门也通过，
  ID/TH/TL尾p90 .22/.20/.10s；LM-only有声比例仍退化。它支持单独训练模块的假设，不证明新训练一定有效。
  最终回归已预留但未使用；当前仍无合格可晋级模型。
- [x] r32已停止，完整step300与latest五文件逐字节SHA相同，改为共享不可变文件，回收217,440,256 bytes；
  原路径/权重/optimizer/WAV/报告字节全部保留。证据`checkpoint_dedupe_20261030/`，禁止续训覆盖旧run。
- [x] 干净最终文本已预留，**尚未运行推理**：`eval_cases/reserved_final_20261030.jsonl`，每语种30文本、
  15 plain/15 control、5seed每模型450条；对164清单/382旧报告、121487种已知文本作精确排除，再查有界近文本。
  不用于调参、不修原台词；全局训练排除已保护。SHA `cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c`。

## 2026-10-30：LM/DiT拆分诊断，逐条判退

- [x] 新入口`scripts/r31_component_probe.py` / `.sh`，独立根`/root/autodl-tmp/r31_component_probe_20261030/`。
  同一r31权重分别仅启用LM或DiT适配器，实际模块逐个核验并恢复；不改官方源码或权重文件。
  两分支并行，每个先测TH1983/1994 control的5seed（已知严重seed优先），通过才跑三语种各一文本的
  plain/control×5seed保护样本；保护文本按ID哈希选择，不看分数。每支最多40条，总预算上限80。
- [x] 逐条早停：近静音、新增空转写或疑似多读立即判退；2条CER>.20且比同case/seed基座差>.10则判退；
  阶段末任一语种平均配对CER增量>.01则不扩评。疑似问题仍记uncertain，失败WAV/评分保留，不重抽凑分。
- [x] 47项相关测试exit0、shell/diff检查通过；80/80已完成，GPU释放。222输入与80WAV SHA及分组/恢复复核通过。
  LM-only因TL2.460%→4.264%淘汰；DiT-only3.444%仅小样本过线，未证明整语种收益。此次严重即时门未触发，
  LM-only到最终保护组才判退，本轮没有省掉预算。详情`docs/r31_component_probe_20261030.md`。
- [x] 核验实际分组/早停结果，下一轮试rank32。拆分已联合训练的适配器仅用于敏感性诊断，
  不能直接证明“只训某一部分”有效；小样本通过不等于三语种验收或母语质量通过。

## 2026-10-30：r31坏例诊断102条完成，半强度淘汰；后续采用分阶段早停

用户在CER核验后要求「继续」，授权执行已记录的有界诊断；旧r31的597条停止状态保持，不恢复900条复评。

- [x] 新入口`scripts/r31_failure_probe.py` / `.sh`，根`/root/autodl-tmp/r31_failure_probe_20261030/`。
  TH1983/1994、ID1816/1707四文本，各plain/control与seed42–46；同ref、CFG1.8/20步、large-v3、禁重试。
  固定复用base/r30/r31共118个已有结果，仅在新目录补2个缺失r31对照；r31强度.5新增40条，
  三个带正文括号的文本只改逗号、保留字词，base/r31强度1新增60条。合计102条，不混改强度与标点。
- [x] 340份输入SHA冻结，plan `5b5f48d3a0ff7bcf9857078071a47df1573aa1a579469085cfab9c583386e84d`；
  53项相关测试exit0，shell/diff检查通过。四路调度（3个r31、1个base），保留4GiB磁盘，已后台启动。
- [x] 102/102完成，GPU已释放。半强度ID CER13.62%→17.20%、TH17.19%→18.55%，疑似多读1→2，
  **淘汰该方向，不据此做降lr轮**。括号改逗号使ID1816 plain在base/r31都降至0.89%，但ID1707缩写句变差，
  不能全局机械替换。完整证据`docs/r31_failure_probe_20261030.md`；不代表整语种或TL/母语收益。
- [ ] 下一实验先拆分LM/DiT适配器的影响，采用逐条早停，过小样本门再扩评；后续训练按最高授权自主推进。
- [x] ID1753/1905的4个条件正文乱码已逐条对回固定FLEURS test.tsv并验原输入SHA：源文件本身已包含乱码。
  原始输入/分数不改，编码修复只能进入另版case与新基线；证据`input_text_audit.json`。本轮426全量测试exit0。

状态`run/status.json`为complete，日志根`driver.log`、`run/worker*.log`，冻结执行源码`source/`不改；后续文档只入`handoff/`。

## 2026-10-30：r31已停止复评；CER与严重坏例核验完成，不自动续训

用户要求定位“不改善”的原因并试用 Granary-v2；后续允许放宽尾静音，覆盖旧轮“不可放宽”的限制。
随后用户要求停下，并核验CER/评分脚本；当前停止状态与下一步覆盖下面的原定900条计划。

- [x] 按PID停止驱动和四路worker，GPU0MiB/0%；最终597/900条（ID300、TH272、TL25），
  597份JSON/WAV SHA及完整latest通过，另2份未评分WAV保留。状态`stopped_by_user`，不自动恢复。
- [x] **同已完成子集**base→r30→r31 CER：ID2.2746%→1.7111%→1.6828%；
  TH5.6666%→4.5307%→5.3608%；TL5.0936%→1.3836%→1.4708%（仅25条，不作整语种结论）。
  不能拿r31部分总体3.35%直接比base全量3.34%；ID相对r30无明确增益，TH部分均值上升。
- [x] 独立复算5097条CER/3325条WER，算术、聚合和条件配对不一致0；18源码路径、28份模型/ref/latest文件、
  11份运行记录核验一致，83相关测试通过。旧分数/评分器/WAV不改，详见`docs/r31_eval_audit_20261030.md`。
- [x] TH净增量约78.2%来自两条：1983_control/s45为29.92s近全静音、CER100%；
  1994_control/s43为疑似多读、CER83.52%。前者已直接核WAV，后者语言内容待听评；不移除失败凑分。
- [ ] 下一步先做冻结坏例的控制/强度/正文括号**分别单变量**诊断；本轮只登记方案，未发起新GPU任务。
  只有强度诊断支持后才考虑同数据仅降lr对照。数据配比/短句/可信控制与TL ref补齐另立实验，
  不继续盲目加步数或以Granary筛选代替能力覆盖。三语种完整验收与母语质量仍未完成，不晋级。

- [x] 新口径 `tail_policy=absolute`：p90绝对值>0.5s仍否决，增量>.10s仅观察；
  overall和分语种都保留。runlog支持显式`--tail-policy absolute`，默认legacy供历史复现。
  内容、时长膨胀、有声比例门与训练数据排除不变。旧轮报告和结论按当时口径保留。
- [x] r30冻结分数另存`work/granary_pilot_20261030/r30_tail_policy_assessment.json`；
  .34/.34/.36s不再构成尾静音否决，逐条>.5s仍记录（ID2/TH6/TL2），不自动晋级。
- [x] 原始FLEURS7065条完整metadata顺序对应；现用1504条（ID516/TL890/TH98）全部验证为原录音PCM精确裁切，
  三份完整archive流式SHA与固定revision LFS一致。原包不落盘，只在远端读取；没有新增训练音频。
- [x] 核对Curator固定commit `8a9a217c90127d424a8f503cb948698d7203002a`公开源码，
  采用转写视图区分、按语言检查ASR重复/异常字符速率、记录拒绝原因；不自动“修文放回”。
  官方HQ≠TTS认证，文本后处理可能用原reference，一致不等于独立ASR互证；细节`docs/granary_cleaning_20261030.md`。
- [x] 当前FLEURS1504行按HQ和`tn_raw|itn_raw`归一化一致筛选，保留731（ID304/TL398/TH29），
  排除773：721无HQ、5无标注、额外47条HQ但转写不一致；原因重叠总计转写分歧232。新ASR启发式0命中。
  只看tn_raw本会仅留657条，itn_raw支持额外74条；**不用删口头禅的itn_clean凑一致**。
  新排除均uncertain，保留原件/清单/原因，target/ref按路径/原音/PCM复用联动，原保留行字节与顺序不变。
- [x] r31候选 **5015条/9.4815h**，val仍895条/1.5965h，完整1634历史val保护。
  时长份额ID55.207%/TH20.463%/TL14.210%/zh6.260%/en3.860%；ref1218（24.29%），可信控制0、TL可信ref0。
  **TL份额显著下降，不能把这轮结果归因为单项清洗规则**；不为补量重复或恢复疑似录音。
- [x] 固定同基座冷启动、seed42、459步、batch2×8、lr1e-4、r/alpha64、dropout.05、LM/DiT开、proj关。
  实际1.464407epoch；只改变选择，复用base/r26/r28/r29/r30，新增900条四路复评，尾策略absolute。
  11,334份冻结输入SHA通过，67份src/官方训练推理源码仅判定记录模块runlog变化；模型训练/推理逻辑相同。
  plan SHA `68cb179fadf983ff130763b07e0956aee0240876062f9cabbc72970111cc7e58`。
- [x] **10:28:06启动训练**；10:30:32已到60/459步，CUDA bf16、loss/梯度有限，
  val step0=1.072322（≤10随机batch，不能据此判收益），wandb `kobfcz1t`。生产未切换。
  测试47项相关/422项全量均exit0，shell/diff检查通过；全量只有两条既有依赖弃用警告。
  已按既有授权删除r30的step0/458/459，回收1,303,443,919 bytes；完整latest SHA不变，步骤元数据/哈希另存。
- [x] **10:41:07正常完成459/459步**，耗时13分01秒；val loss
  1.072322(step0)→.962661(step250)→.899426(step458)，不作质量结论。
  LoRA SHA `065d98abc74f97ca32385a58ebf580d99ab5e8d8d02d19ea5576294cc9ed087e`，
  latest完整五文件已冻结SHA。32份固定小产物/源码与本机镜像一致，证据`local_verification.json`。
  随后自动启动四路评测，现已按用户要求在597条停止；状态以`run/status.json`为准。
- 原定900条复评**未完成并已停止**，不得自动补跑；CV26尚未接入，不宣称Granary是新增主力语料。
- 原因边界：r30−r29三语种CER配对区间均跨0；val仅≤10随机batch且验证集变化，
  loss下降不代表配音质量改善。TH时长17.7%、TL可信ref0、可信控制0仍是覆盖缺口；
  ID 1707/1816的正文括号/缩写与内容报警待定位，不能先断言推理解析或停止头有bug；新证据见顶部。

远端 `/root/autodl-tmp/granary_pilot_20261030/`；状态`run/status.json`、训练`run/train.log`、驱动根目录`driver.log`。
执行源码`source/`已冻结，更新文档只进`handoff/`。**后续混合必须继承本轮`run/flagged.json`的新隔离**；
原music_excluded仍为上游清洗池，不得绕过新增773条排除。原先training_approved=false阶段记录不回写。

## 2026-10-29：r30已完成，不通过、不晋级

用户明确「继续微调」，授权覆盖前一阶段仅清洗/不训练的范围；本轮独立根
`/root/autodl-tmp/full_clean_candidate_20261029/`，入口`scripts/full_clean_candidate_run.py` / `.sh`。

- [x] 固定方案：从同一原基座冷启动，使用最终`music_excluded_20261029/run/candidate/current/`，
  train5788条/12.0352h、val895条/1.5965h；保留原行/顺序、继承全部排除和完整1634行val保护。
  不从38,819候选池新抽行。ID/TH/TL/zh/en时长份额48.935/17.699/25.393/4.932/3.041%。
- [x] 沿用r29的459步、seed42、batch2×8、r/alpha64、dropout.05、lr1e-4、warmup45、wd.01、
  LM/DiT开、proj关。有效曝光约1.269epoch；保持优化器步数/调度，只改变清洗后的数据。
  清洗也改变语种/ref份额，单训练seed不能给出强因果结论；val已变化，不横比旧val loss。
- [x] 固定复评：180条件×seed42–46共900条，四路，CFG1.8/20步、large-v3、禁重试；
  复用base/r26/r28/r29冻结报告/WAV，新增r30−r29/r28/base对照，分语种检查内容及尾静音既有门槛。
  不自动切生产、加轮次、放宽门禁或重抽旧结果。可信控制0、TL同人ref0仍是局限。
- [x] 相关16项回归通过/exit0；shell与diff检查通过。62份直接剔除已提交并push：`b1525aa`。
- [x] 11,086份冻结输入SHA/预检通过，独立plan SHA
  `d5be461816214108423112315b916668fa2d7736b72b33cae2c86305d078644b`。
  已清理r29的step0/458/459，回收1,303,443,919 bytes（1.214GiB），完整latest五文件SHA不变；
  清理误传的本地venv后磁盘可用8.93GiB。训练已后台启动，20:54首个取证为CUDA bf16、5788/895条加载完成。
- [x] **20:53:42开训，21:07:33正常完成459步**，耗时13分51秒；loss/梯度有限、CUDA bf16。
  val loss **1.085739(step0)→.969930(step250)→.898114(step458)**，wandb `a8gdknd1`；
  新val与旧轮不同，不直接横比。LoRA SHA `06944982432581880a1aaebad6bd82d3fd7c4af73d8945d42a8ac0833570fc6b`。
  11份核心产物本机/远端SHA一致，67份训练/推理源码与r29相同，完整latest仍核验通过；
  证据`work/full_clean_candidate_20261029/{startup_snapshot,verification}.json`。
- [x] **22:30:17完成900/900复评**，约83分钟。r29→r30 CER：ID **1.60%→1.71%**、
  TH **4.68%→4.69%**、TL **1.98%→2.03%**；逐语种配对95%区间均跨零，无明确内容收益。
  疑似少读/漏尾ID4→9、TH12→9、TL0→0；都是ASR诊断，不能直接认证漏词或误报。
- [x] **不通过、不晋级**：尾静音p90 ID.34s、TH.34s、TL.36s，相对base+.18/+.16/+.18s，
  全部超过既定+.10s门。TH较r29少20ms，其余两语种p90不变；不因小幅改善或val loss下降放宽门槛。
  900条无空ASR/缺失尾指标/自动多读标记；未做声学盲听或母语验收，生产未切换。
- [x] 完成后核验11,086份输入、完整latest、5模型4500份WAV及6份ref；19份完成文件本机/远端SHA一致。
  只读汇总`run/postcheck_20261029.json`、逐seed分解`run/severity_20261029.json`，原报告/WAV未改或重跑。
  GPU已释放（0MiB/0%），磁盘余约6.3GiB；全量音频/优化器仍主要留远端，不据此释放实例或数据盘。
- [ ] 后续先定位冻结失败样本的停止/尾部行为；ID报警集中于`fresh_id_1707/1816`（正文括号/缩写）。
  弱证据保持uncertain，不以ASR豁免；新的单变量假设与回归冻结后再决定实验，不自动加步数或扩池。

完成状态以`run/status.json`及`checkpoint/latest/training_state.json`为准，日志`run/train.log`、
`run/driver.log`、`run/eval/worker*.log`。旧审计的`training_approved=false`不回写；新plan单独记录授权。

## 2026-10-29：全量清洗完成，音乐疑似62份全部剔除（当前候选）

用户明确「直接剔除就行了」，随后要求继续。**该授权覆盖此前“先核验52份”的限制：
冻结名单的62份全部保守隔离，不以试听/母语确认作前置。仍记疑似，不认证音乐污染，不把分数扩成通用新门。**
入口 `scripts/quarantine_music_candidates.py prepare|watch|finish` / `.sh`，独立根
`/root/autodl-tmp/music_excluded_20261029/`。原始音频、旧清单/评分和主清洗执行源码保持不变。

- [x] 先从r29旧候选移除62训练行/437.89s，生成7190条/14.5086h的`current_music_only/`中间快照，
  原1148行val不变，未额外命中ref。**这份中间快照现已被最终候选替代，不用于训练。**
- [x] 主清洗于**19:42:13**完成全部50,321文件：CPU完成50,321，large-v3完成49,121；
  8447文件命中已有/本轮风险，重叠原因含ASR文本差7102、头静音697、VAD尾314、ASR非语音概率283。
  文件风险数不等于删除行数；全部旧验证行继续受保护，原音频没有物理删除。
- [x] **19:42:39自动完成音乐排除合并**：62份中10份已被主清洗排除，额外排除52行/365.02s。
  当前候选 **7252 − 1412（主清洗）− 52（新增音乐）= 5788条 / 12.0352h**；
  val **1148→895条 / 1.5965h**，音乐阶段未额外改val。与旧val loss不再同口径，旧指标保留。
- [x] 当前训练按时长：ID48.935% / TH17.699% / TL25.393% / zh4.932% / en3.041%。
  ref1218行/5788（21.04%）：ID868、TH350、TL0；不同ref1153，目标最大曝光1、ref最大复用3。
  可信控制标签仍0；不伪造身份/情绪补覆盖，也不因GPU空闲加epoch/proj。
- [x] 10源完整候选池合计 **38,819训练行 / 67.4032h**，同样叠加62份的路径/原音/PCM/ref复用排除；
  只有`gigaspeech2_id/train`和`current/train`相对主清洗结果各少52行，两者是不同视图，不累计为104份录音。
  当前5788行及整个候选池的评测文本、完整1634行受保护val、身份/session/path/PCM检查通过，风险复用命中0。
  保留行原字节/原split不变；原保护val不会因删坏行回流训练。
- [x] 相关18项回归通过/exit0，shell/diff检查通过；44份最终候选及排除清单、5份报告共49份本机/远端SHA一致。
  全量复核50703份父输入、235份本轮输入SHA，全部候选字节保留和62份复用排除通过；证据`verification.json`。
  后台清洗、音乐扫描、候选合并均已结束，GPU0MiB，**尚未启动新训练或切生产**。
- [x] 下一轮数据/配比/步数及同口径评测方案已按最新授权冻结为r30，见顶部。当前TH份额已降至17.7%，
  “保持r29子集做清洗对照”与“从全池重新平衡”是不同方案，不能混称单变量；不要直接增加步数。
  音乐扫描只覆盖原当前训练目标/ref的8312文件，**没有扫完整38,819候选池**；若从池中补新行，需补对应覆盖。

**后续混合/训练只读取 `music_excluded_20261029/run/candidate/<source>/train.jsonl|val.jsonl`，
当前子集是其中`current/`；同时继承 `run/flagged.json`、原主清洗排除记录和完整旧验证保护。**
不要回用`full_corpus_clean_20261029/parallel8/run/candidate/`或`current_music_only/`遗漏音乐排除。
计划/结果在`run/plan.json`、`run/summary.json`，状态`run/status.json`，日志`driver.log`；
本地镜像`work/music_excluded_20261029/`。清洗通过不等于已经获得模型收益或母语质量验收。

## 2026-10-29：新增音乐旁路、近重复文本审计与Granary-v2核验

用户再次要求利用空余显存并行提升质量，并追加评估`nvidia/Granary-v2`。
本节记录旁路诊断阶段；主清洗与后续62份排除已完成，当前候选见顶部。各自独立source/run，未启动新训练。

- [x] CPU近重复文本检查 `scripts/audit_near_text.py`：7252训练行对冻结评测文本和完整受保护val的2059个不同文本，
  4.17s完成，**0命中**。同文全长度检查；近似仅同语种、双方至少40个归一化字符、Levenshtein距离≤2。
  三字片段倒排索引及有界编辑距离，经独立完整DP/重复字符/泰文测试核验；没有做语义或音频近重复认证。
  产物 `/root/autodl-tmp/near_text_20261029/run/{plan,summary}.json`，空`exclusion_overlay.jsonl`。
- [x] 声音事件入口 `scripts/audit_music.py fetch|prepare|run` / `.sh`，独立目录
  `/root/autodl-tmp/music_audit_20261029/`。固定MIT AST AudioSet revision `f826b80d28226b62986cc218e5cec390b1096902`，
  BSD-3-Clause、346MB权重SHA核验；只在远端大盘下载，使用既有torch/torchaudio/transformers，未加依赖。
  首次HF下载数分钟未写入权重，按PID停掉；同固定URL改curl下载135s后验SHA再安装缓存，原日志与零字节partial留存。
- [x] 冻结当前训练目标/ref **8312文件**，跳过127个已有声学报警文件；16k单声道、10s窗/5s步长且覆盖结尾，
  float32、单条推理、单CPU线程、每窗暂停100ms、PyTorch内存上限10%。采用多标签sigmoid，Speech/Music可同时高分。
  **100条小样21.26s完成，PyTorch峰值404MiB**（nvidia-smi进程约820MiB）；同期主ASR42592→42641，
  约20s推进49条，只能证明仍推进，非严格吞吐A/B。小样有1条ID gs2的Music=.8567，Speech=.9073。
- [x] **19:14:01完成8312/8312音乐审计**，小样100条复用，接续8212份1361.99s、PyTorch峰值404MiB。
  62个高分核验文件**全部来自gigaspeech2_id，均是当前训练目标，不涉及ref**；ID4445份中62个，
  TH2465/TL939/en152/zh311份中均0个。该数不是62个已确认配乐，其他语种低分也不认证无配乐。
  对照主ASR：62份均已转写，其中10份已命中`asr_text_mismatch`排除规则，另52份无本轮ASR报警。
  这些是文件级规则命中，最终候选行数及原音/PCM连带排除仍待主清洗汇总。
- [x] 此前计划核验剩余52份；已被用户「直接剔除」覆盖，冻结62份现全部隔离，见顶部。
  原旁路>=.8只作为核验优先级的报告仍保留；新授权的隔离单独记录，不把未标定分数认证为BGM真值。
  ASR通过不豁免本次排除；没有改变旧报告或扩为任意新数据的音乐自动门。
  完成报告`run/summary.json`、逐文件`run/items/`；交叉证据`priority_asr_snapshot.json`、`postcheck.json`。
  8450份输入及全部8312文件的分窗/汇总一致性复核通过，未重跑ASR/声音事件模型或改旧分数。
- [x] Granary-v2 revision `81402b751b451b5fe4f4c29e8b40f92892f1e440`四份目标清单全部下载到远端并验SHA，
  **仅标注、没有音频下载/入库**。ID和Filipino只有FLEURS；TH另有CV26。详见`docs/corpus_sourcing.md`新节。
  TH CV26 33004条/37.631h、HQ且3–30s为10846条/14.064h；其中32914条至少一个转写版本与现有CV22同文。
  同文不等于同PCM，尚未恢复原音频ID映射；不把重新标注当新语料、不直接换原文/补情绪或speaker标签。
  **暂不作为新增主力；优先考虑已有录音的标注复核线索。** 三语已有验证/评测同文也必须继承排除。
- [x] 新逻辑11测试通过；全量在单线程原生库环境下 **409 passed / exit0**，两条既有依赖弃用警告。
  默认线程环境首次409断言通过后仍exit134（macOS原生库recursive_mutex退出异常），保留此失败；
  单线程这次成功不等于已确认退出异常根因或彻底修复。
- [x] 18:57快照：音乐2075/8312（25.0%），主ASR43688/49121（88.9%），两任务均持续运行。
  复核音乐8450、近文本141、Granary43份输入SHA通过；12份冻结报告/脚本本机远端SHA一致，
  证据 `work/music_audit_20261029/verification.json`。数据和报告留远端大盘，原始音频未改。
- [x] 19:17快照：音乐已完成释放显存；主ASR46121/49121（93.9%），八路均持续更新，未开启新训练。
  接续音乐期间主ASR42824→45741，约1360s推进2917条≈2.14条/s；任务构成不匹配，不据此宣布无吞吐影响。

## 2026-10-29：现用10源全量清洗已完成（执行记录）

用户在r29复评后要求「整体清洗一波微调数据呢」，范围扩为现用10个来源的全部已加工train/val，
加当前r29目标/ref与历史受保护验证集：**50,321个不同文件，约86.44h**，ID和当前训练数据优先。
沿用「有问题及疑似有问题都排除」；原始录音、源清单、旧评测及训练产物保留。

- [x] 新入口 `scripts/full_corpus_clean.py` / `.sh`；独立目录
  `/root/autodl-tmp/full_corpus_clean_20261029/`，14:24:38后台启动。最终全量 **389 passed**，shell/diff检查通过。
  复用现有来源、隔离、PCM、能量/VAD和ASR工具；不下载语料/模型，不修改官方submodule。
- [x] 四路CPU扫完50,321文件：1200个带旧排除/本轮风险标记，49,121个进入ASR；不是1200个已确认坏录音。
  主要重叠原因：头部静音697、VAD尾部314、旧排除路径143、乱码68、能量无边界49、无VAD语音43、短文长音13。
  检查解码、16k单声道/3–30s、时长一致性、PCM、削波和首尾静音。
  继承全部旧排除、验证文本/路径/身份/session；无可信控制前缀的原行排除，不伪造标签或换ref。
- [x] 用户要求增加GPU并行后，使用 `scripts/full_corpus_parallel.py` / `.sh` 接续到
  **8个独立CUDA ASR进程×2 CPU线程**。旧四路实测GPU多在35–67%、约3GiB；CPU额度20核。
  保留全部 **3759条**已落盘转写，仅接续剩余45,362条；原进程PID34531已退出，显存释放后再启动新驱动。
  旧nohup进程忽略SIGINT，按PID终止；未落盘的在途工作可重新分配，成功结果按ID/路径/SHA排重。
  旧 `source/`、plan和ASR文件不改，新执行代码、计划、交接证据另冻在 `parallel8/`。
  同时显式继承旧plan内22条乱码剔除及完整165个历史PCM指纹；复核无本轮声学阶段漏标的新别名。
- [x] 17:43状态33,502/49,121（68.2%），八路均正常；15:00–15:08实测4.48条/s，
  后段约2.5–3条/s，不能按前段速度固定推算或宣称并发倍速。当时声纹旁路同时运行，GPU99%、约20.4GiB。
- [x] 18:18只读快照：ASR已39,367/49,121（80.1%）；ID和英文扫描已完成。
  ID共13,244文件，CPU标记427，剩余12,817个ASR全完成，其中1,070个ASR疑似、11,747个无本轮ASR标记。
  TH剩6161、TL剩798、zh剩2795个ASR任务。此为文件级快照，**不等于最终剔除/保留训练行数**，
  target/ref/原音/PCM联动与最终隔离尚未汇总；证据 `work/codec_padding_20261029/corpus_progress.json`。
- [x] 49,121文件全部完成八路CUDA large-v3无台词提示复核：beam5/temperature0、
  不沿用前段文字、本轮数据诊断关闭VAD；TL自动识别，其余指定语种。旧TTS评测口径不变。
  **CER>0.15、时长加权avg_logprob<−1.0、no_speech_prob>0.6、空ASR均隔离**；
  首/尾能量或VAD静音>0.5s、连续满幅≥1ms也隔离。ASR语种标签不单独判口音或错语种。
  阈值为本轮保守筛选政策，不是母语正确性门槛，允许误删，逐条保留原文/转写/原因。
- [x] 已自动生成 `parallel8/run/candidate/<source>/train.jsonl|val.jsonl` 与 `current/` 对应清洗候选，
  target/ref按原路径、原音和PCM连带隔离；训练精确PCM去重、同原音最多3次曝光。
  保留行原字节、原split；即使验证行被排除，完整旧验证集仍受保护，不回流训练。
  原始主清洗报告已冻结；后续已叠加用户授权的音乐62份排除，最终候选位置与数量见顶部。

**当前只看 `parallel8/run/status.json`**，日志 `parallel8/driver.log`、`parallel8/run/worker*.log`；
旧 `run/status.json` 在交接后不再更新。CPU明细在两阶段各自 `run/acoustic.jsonl`，
已完成旧ASR保留为 `parallel8/run/retained_asr.jsonl`，八路新增在 `parallel8/run/workers/*/items.jsonl`；
已合并新 `parallel8/run/asr.jsonl` 并落 `summary.json`，19:42:13状态complete。磁盘保留4GiB，失败不隐式重启/覆盖。
**当前仅授权清洗候选；未开始新训练/切生产。** 自动检查不能完整认证近重复、多人/BGM、音色或母语质量。

## 2026-10-29：并行声纹与训练补齐诊断

用户要求「还有哪些可能提升微调质量的事情，看看能不能并行做了」，并指出GPU显存空闲。
现有八路ASR已99–100%忙，空闲显存不等于同等比例的算力余量；只加一个有界的声纹进程，
每条间隔50ms、单CPU线程、PyTorch显存上限约4.8GiB。源码和输入单独冻结，原清洗规则/源码不改。

- [x] 入口 `scripts/audit_training_quality.py` / `.sh`，独立目录
  `/root/autodl-tmp/training_quality_20261029/`；复用已缓存WavLM，无模型/语料下载。
  从当前7252训练行核对1429对可信元数据ref，54对已有声学报警跳过，实际审计1375对/2525不同音频。
  不用未知身份的TL数据强配ref，不重写原清单。
- [x] 64份GPU小样约7.27s，PyTorch峰值588MiB，已完成结果直接复用。
  小样期间ASR状态增加28条/10s；窗口太短，不据此认定无吞吐影响。
- [x] **17:44:49完成2525份声纹、1375对配对评分**，接续部分2461份用时220.56s，PyTorch峰值608MiB。
  期间主ASR由33135到33764，增加629条/220.04s≈2.86条/s，持续运行；不是严格同任务吞吐A/B。
  `run/reference_summary.json`、`pair_scores.jsonl` 与 `gpu_2525.json` 已镜像本机。
  ID925对中位cosine .9469，TH450对 .9430；名义不同ID的p99仍为.9481/.9462，分布明显重叠。
  因此最低20对/来源仅是复核优先级，不是40对确认错配或新自动排除；不同声明ID非已验证负例。
  不套用其他声纹模型的0.45阈值，不自动换ref或新增身份标签；高相似度也不能认证真人身份。
- [x] CPU审计7252行的训练打包：实际块长160ms，平均补齐ID77ms/TH80ms/TL73ms；
  原始尾静音p90三语均150ms，`原尾+补齐量`的算术估算p90为ID260/TH260/TL280ms。
  原文/音频和旧评测未改；这不是重新测量补齐WAV，也不是已证实的生成尾静音因果。
  以形状模拟VAE执行官方packer，两条路径均在最后目标块给stop标签，EOS无loss；未发现额外监督一整块的错位。
  不能据此改官方submodule、硬裁低声尾字，或把既有验收门放宽。stop头仍冻结，开proj也不解冻它。
- [x] 全量 **394 passed / exit0**，两条既有依赖弃用告警。新增shape-only检查在独立子进程跑，
  避免macOS混合原生运行时在退出时abort；首次394断言虽过但exit134，不计为成功验证。
- [x] 主清洗和音乐排除完成后已核新候选配比、时长和ref覆盖，见顶部；后续加量仍要避免稀释ref。
  当前可信控制标签0、TL可信同人ref0仍是结构缺口，不能靠声纹相似度自动补身份/情绪或合成补量。
  有新可追溯语料再做单变量方案；不因剩余显存自动加epoch/proj或开启新训练。

旁路仅诊断，不生成TTS或重评分旧报告；不声称已验证多人/BGM、授权或母语质量。
本地镜像 `work/training_quality_20261029/`，源码冻结后文档和补充证据另进 `handoff/`。

## 2026-10-29：真实VAE编解码补齐对照已完成

用户「继续」后，沿用上节提出的固定样本编解码诊断，不开启训练。
入口 `scripts/audit_codec_padding.py` / `.sh`，独立根 `/root/autodl-tmp/codec_padding_20261029/`。

- [x] 冻结96条当前训练目标：ID/TH/TL各32，3–15s、声学及已落盘ASR均通过；
  每语种按额外补齐0/40/80/120ms各8条，不能把这份分层小样p90当全量训练分布。
  比较原音、40ms/160ms补齐输入、相应48kHz VAE重建，保留192份独立重建WAV及SHA。
- [x] **18:13:27完成96/96**；先12条4.83s，再接续84条24.08s，旧结果未重跑。
  CUDA float32、严格加载冻结VAE权重，峰值1526MiB；源码、267份输入及192份输出SHA核验通过。
  主ASR持续运行；接续窗口20s仅推进20条，窗口太短且任务构成未匹配，不据此声称无算力竞争。
- [x] 160ms训练对齐相对40ms原生对齐的**配对尾静音均值差**：ID+56.25ms、TH+60ms、TL+60ms。
  160ms补齐输入→VAE重建的配对均值差仅ID+0.625ms、TH+1.25ms、TL约0。
  小样重建尾p90（40ms→160ms）：ID180→260ms、TH178→260ms、TL180→300ms。
  没看到VAE把补齐扩大成180ms额外尾长；这只量化编码变换，**仍未证明r28/r29相对base退化的原因**。
- [x] 零额外补齐的24对尾指标全部相同。22对PCM完全一致，另2对PCM最大差约.001735/.001582，
  恰为两次进程首条，但原因未确认，不能宣称逐采样确定性或擅自归因CUDA/JIT。
  初次把两臂WAV文件SHA当相等性检查的assert失败；保留原检查脚本，后续回读PCM记录差异，
  不抹掉这2对、不重评分或挑结果。各输出自身冻结SHA均通过。
- [x] 全量 **398 passed / exit0**，shell/diff检查通过；保留原件和旧评测，重建结果仅development diagnostic。
- [ ] 等完整清洗后核配比/ref覆盖；下一轮仍需数据/训练单变量方案。
  不把本诊断当理由改官方packer、硬裁尾字、放宽尾静音门或自动开始新训练。

产物 `run/summary.json`、`run/items/*.json`、`verification.json`；本地镜像同名work目录。
原始录音、VAE重建WAV均主要保留远端；没有新增ASR/TTS或完整语言/听感验收。

## 2026-10-29：r29已完成，不通过、不晋级

用户「接下来可以启动训练了吧」已授权本轮，随后授权「过期的checkpoint，该删掉就删掉」。
新 run 的 `training_authorized=true`；旧审计候选的 `training_approved=false` 保留为历史阶段记录。

- [x] 清理4个已完成run的16个无报告/计划引用的中间或重复 `step_*`，回收 **6.47 GiB**；
  清理后可用11.56 GiB。各run完整 `latest`（含优化器）逐文件SHA前后相同；原数据、冻结报告/WAV保留。
  删除清单、逐文件SHA和执行结果在 `work/suspect_candidate_20261029/checkpoint_cleanup_*.json*`。
- [x] 从原基座冷启动 `lora_joint3_r29_suspect_excluded_459`，**10:41:56（北京时间）已开训**。
  仅改r28的数据剔除：**7252条/14.6302h，459步≈1.013epoch**，seed42、r/alpha64、dropout.05、
  lr1e-4、batch2×累积8、warmup45、wd.01、LM/DiT开/proj关；未追加第2epoch或续训r28。
  10:43:58启动取证到40/459，loss/梯度有限，CUDA bf16，GPU99%/25239 MiB；
  wandb run `ggxkkng4`；val loss 1.028783(step0)→0.944497(step250)→0.965453(step458)。
- [x] 冻结输入与配置，11份核心清单/清理证据本地远端SHA一致。全量测试 **379 passed**，
  shell语法和diff检查通过。新入口复用已有训练/评测驱动，源码冻结后不再覆盖。
- [x] **10:54:36训练正常结束，459/459步**，约13分钟，latest状态与权重文件核验一致；
  LoRA SHA `3d124aa95c8b6feaa45023393e8552fe848cc8e3a700f4408bdc7e9eca6a9424`。
- [x] **12:17:01四路复评完成900/900**，评测约82分钟；原180条件×seed42–46，复用base/r26/r28，
  CFG1.8/20步/禁重试/large-v3口径不变；新增 `r29−r28` 分语种配对比较，保留近静音回归。
  19份完成产物本地/远端SHA一致，4模型共3600份既有WAV、6份ref及最新权重/优化器SHA复核通过。
- [x] **不通过、不晋级**：ID/TH/TL尾静音p90为0.34/0.36/0.36s，与r28相同，
  相对base均+0.18s，超过0.10s门。CER r28→r29：ID1.48%→1.60%、TH4.70%→4.68%、TL2.21%→1.98%；
  三语种配对区间均跨零，没有明确的整体清洗收益，不把点估计当显著改善/退化。
  r28那条TL近静音在同case/seed45下变成10.24s、speech_ratio.9804、CER.0656；本次未复现，不等于稳定修复。
  900条无空ASR或尾部指标缺失；疑似少读/漏尾ID4/TH12/TL0，仍是自动诊断，未做语言验收。
- [x] 只读汇总 `run/postcheck_20261029.json`，原分数与失败条目不修改；清洗无明显收益不撤回已隔离数据。
  用户已要求全量清洗现有来源，下一步见顶部，不直接加epoch/proj或凭GPU空闲开新训练。

远端 `/root/autodl-tmp/suspect_candidate_20261029/`，本地镜像同名 `work/` 目录。
配置 `configs/lora_joint3_r29_suspect_excluded_459.yaml`；日志 `run/train.log`、`run/driver.log`、
`run/wandb.log`，状态 `run/status.json`。运行中不重启、不覆盖 `source/`，文档交接另放 `handoff/`。
本轮合计约95分钟，已结束；**生产未切换，声学盲听及母语质量仍未验证**。
可信控制标签0、同人ref19.705%（TL为0）仍是数据限制，不能通过伪造标签或强配ref补齐。

## 2026-10-29：全部已发现的异常/疑似样本已从新候选剔除（审计阶段已完成）

用户先要求继续审计，随后明确「原则就是把这些有问题的，以及疑似有问题的数据都删掉」。
**此决定覆盖先前“确认后才排除”的限制：所有本轮报警都隔离，不等人工确认；部分 ASR 对齐不自动放回。**
“剔除”落在新训练清单及 target/ref 复用链，原件、冻结清单与旧评分保留，疑似不冒充已确认坏录音。
远端 `/root/autodl-tmp/training_silence_20261029/`，本地报告镜像 `work/training_silence_20261029/`。

- [x] 四路 CPU 扫完 **8530 个不同文件**：7333 个训练目标、1445 次参考引用（1369 个不同 ref）。
  能量口径可测8524个，**尾静音>0.5s为0，p90=0.15s，最大0.37s**；6个低电平无边界保留None。
  VAD长尾53、无语音23、能量无边界6，重叠后 **80 个待审文件**，不能叫80条已确认尾静音。
- [x] 四路 CUDA Whisper large-v3 对80文件做无台词提示诊断，另7次内存增益复核；
  `vad_filter=false`仅用于本次数据诊断，防止同一VAD漏检再次抹掉输入，未改变冻结评测口径。
  80条原幅度转写均非空，但存在“感谢观看”等疑似幻觉；76个文件有目标文本，28个CER>0.3，
  **这些只是诊断分层，不是新增自动剔除阈值或母语结论**；7条极低电平也不都等于纯静音。
- [x] 按用户最终原则隔离全部80个报警及原音路径/精确PCM/ref复用，共剔除 **81训练行 / 531.82s**：
  **ID55 / TH13 / TL9 / en3 / zh1**；76行目标命中、6行ref命中，其中1行两者都命中。
  之前隔离的79行仍未回流；相对最初7412行，累计剔除160行。
- [x] 新候选 **7252条 / 14.6302h**，路径 `candidate_all_flags/`；验证集1148行原字节不变，
  仍保护完整旧验证集1149行。保留训练行逐字节一致，未换ref、修文本、补样本或改控制前缀。
  评测/验证文本及身份/session/path门通过，train/val PCM冲突、同target/ref PCM、目标>3×曝光均为0；
  保留集对本轮疑似路径/原音/PCM/ref复查命中0。未知身份、稿音及母语质量仍未认证。
- [x] 按时长实际配比 ID47.492% / TH22.681% / TL22.259% / zh5.029% / en2.539%；
  ref1429行（19.705%）、1353个不同ref，目标最大曝光1、ref最大复用3；可信控制标签仍0，TL同人ref仍0。
  审计、ASR、隔离分别冻结在 `source/`、`recheck_source/`、`quarantine_source/`，不覆盖既有执行证据。
  两个新入口及原隔离相关测试 **16 passed**；原始音频/输入SHA前后校验通过，
  15份核心报告/清单/执行源码的本地与远端SHA一致（`verification.json`）。
- [x] 下一轮单变量方案已冻结并启动为r29，见顶部；只测试“排除本轮所有疑似样本”这一数据变化：沿用r28原基座冷启动、seed42、
  r/alpha64、lr1e-4、batch2×累积8、warmup45、LM/DiT开/proj关，固定459步（新集约1.013epoch），
  与r28相同180条件×seed42–46复评，复用原base/r26/r28报告，保留近静音坏例。
  不同时加到2epoch、改前缀/proj或替换评测文本；不能把本轮审计说成模型已有收益。

新train SHA `27ed1d5ff21d8c31067b09429d2c9ba78b7fa71556b2d23b5ec540c52196e0a5`；
排除清单 `excluded.jsonl` 与完整指纹在同目录plan/audit，后续增量混合必须继承这些排除。
审计阶段的 `training_approved=false` 表示该入口仅产出候选；后续r29启动见顶部。
本审计阶段新增87次ASR诊断、TTS/下载/训练/生产变更均0，r28仍未晋级。

## 2026-10-28：r28 已完成、不通过；下一步审计训练尾静音（当前生效）

用户明确「直接去掉有问题的训练集，然后按照你建议来训练吧」，并要求继续。
本节覆盖此前仅评测/不训练的执行边界：授权本轮清理后候选训练及同口径复评，**不含生产切换**。
用户最新要求「记录好todo，先push下」：当前先归档并提交推送，尾静音清理与新训练留作后续，未自动启动。

- [x] 原 r26 的 7412 行中先前已隔离 58 行（49 同文、8 条 ID 待审目标、1 条引用行）；
  再排除 21 条明显乱码文本（ID 13 / TL 8），新训练集 **7333 条 / 14.778h**。
  验证集另排除 1 条 TL 乱码，**1148 条**；完整旧验证集仍用于训练隔离检查。
  原清单、录音、历史报告保留；不凭总词速批量删除整个源，不修写未知台词。
- [x] 清理后尚存 **497 条未核验的自动中性前缀**，只移除前缀及其控制字段，保留正文/音频；
  另外 3 条前缀行已随乱码样本排除。可信控制标签为 0，不伪造补齐；控制跟随能力需观察。
  因前缀与训练轮数也变化，本轮是清理后的候选，**不是证明清洗收益的因果 A/B**。
- [x] 新入口 `scripts/clean_candidate_run.py` / `.sh`；全量 **363 passed**（两条既有依赖弃用警告）。
  同步独立目录 `/root/autodl-tmp/clean_candidate_20261028/source/`，原冻结源码与远端脏工作树不动。
- [x] 远端预检通过，**21:08:37 已开训**；首批实测到 60/459，CUDA bf16、GPU 93%、27199 MiB，
  loss/梯度有限，wandb 已同步（run `kosuky1m`）。run 名 `lora_joint3_r28_clean_1ep`；基座冷启动、seed42、
  **1 epoch / 459 步 / batch 2×累积8 / lr 1e-4 / r=alpha=64 / dropout .05 / wd .01**；
  LM+DiT 开、proj 关，warmup45。拒绝续训/覆盖，长任务落大盘日志。
- [x] **21:21:38 训练完成，459/459 步**，约13分钟；正常退出且 latest 状态一致，
  LoRA SHA `7499cee1c20a216e42a7f8f4a9d2dfce545ad81a3b4137935eb19d0ee8cd77d7`。
  val loss 1.014285(step0)→0.941851(step250)→0.966215(step458)；每次至多10个验证batch，
  不能凭这三个小批次数字认定欠训/过拟合或替代分语种评测。
- [x] **22:45:18 四路评测完成 900/900**，`stage=complete`，459训练步已核验；GPU已空闲。
  原 180 条条件 × seed42–46 = **900 条新增输出**；
  复用已有 base/r26 报告与 WAV，不重跑。CUDA/哈希/磁盘守卫不变，失败不自动重抽。
  分语种及 plain/control 验收；冻结评测里的乱码与专名报警单独保留为限制，不篡改旧分数。
  新文本集已用于开发诊断，不永久冒充独立最终验收；母语质量仍未验证。
- [x] 最终报告、plan、状态与权重 SHA 本地/远端核对一致；逐条件/seed覆盖完整。
  `run/postcheck_20261028.json` 仅汇总冻结评分，不重抽、不重评分、不修改旧报告。
- [x] **结论：不通过、不晋级。** 三语种尾静音p90相对base均增加 **0.18s**，超过既定0.10s门槛。
  TL `fresh_tl_1913_control` seed45生成 **65.92s近乎静音**，ASR空、speech_ratio=0；
  字节核验 peak=0.001068、RMS=6.396e-6，仅3276/3164160采样点非零。
  已登记 `eval_cases/acoustic_regression_r28_20261028.jsonl`，实际输出/ref字节、SHA、条件与原评分
  保存在 `work/clean_candidate_20261028/regression/`；合成输出固定development_only，不入训练。
  该条尾静音指标为None（未测），不能记0或声称900条全部正常；所有分数仍含该失败条目。

| 语种（每模型300条） | CER base → r26 → r28 | 疑似少读/漏尾 base → r26 → r28 | 尾静音p90 base → r28 |
|---|---:|---:|---:|
| ID | 2.27% → 0.79% → 1.48% | 13 → 1 → 3 | 0.16 → 0.34s |
| TH | 5.65% → 4.43% → 4.70% | 14 → 6 → 10 | 0.18 → 0.36s |
| TL | 2.10% → 1.66% → 2.21% | 1 → 0 → 1 | 0.18 → 0.36s |

r28相对base的ID/TH CER区间支持改善，TL区间跨零；相对r26三语种点估计均更高，但ID/TH差值区间跨零。
带控制时对参考的嵌入相似度下降更明显，属于声学诊断，不等于已验证情绪/口音错误。
完整分层、置信区间、WER与源表见 `docs/runs.md` 和 `docs/finetune_playbook.md`。

- [x] **只读审计本轮训练目标和参考音的真实尾静音/近静音：10-29已完成，见顶部。** 按源、语种统计，
  优先核实尾静音>0.5s与近乎全静音样本，保存文件SHA、时长、能量/VAD边界和原因；
  正常短尾静音保留，低声尾字与内部停顿不冒充可删尾部。生成垫尾不能倒推训练音频一定有问题。
- [x] **从新候选排除问题目标及其ref引用：10-29用户扩大为全部疑似，已执行，见顶部。**
  同时查原路径/原音路径/PCM回流，保留原件和排除清单；
  不全局硬裁，不伪造文本，不按语速强删整个源。重新统计时长配比、曝光和ref覆盖，复查评测隔离。
- [x] 审计后已冻结r29数据单变量方案并启动，见顶部；可信控制标签仍为0，不能以自动中性前缀冒充真人控制标签。
  尚未追加第2epoch。加步数能否改善尚无结论；若另测累计约918步，应保留1epoch对照、单独规划
  近零学习率之后的调度，不同时改数据/前缀/proj或覆盖冻结run。

启动状态只认新 `run/status.json`，训练完成同时核 `checkpoint/latest/training_state.json`。
新清单、配置与 plan 已镜像至 `work/clean_candidate_20261028/run/` 并逐份核对输入 SHA；
配置另登记 `configs/lora_joint3_r28_clean_1ep.yaml`。实际按时长配比 ID 47.556%、TH 22.634%、
TL 22.265%、zh 4.987%、en 2.558%，沿用父候选比例，不冒充三语等权。
训练完成状态、loss、分语种评测与结论已补齐；完整WAV/优化器仍在远端，本地镜像不是全实例备份。

## 2026-10-28：新留出 base/r26 基线已完成

用户「现在有gpu了，继续吧」后，恢复**已约定的新基线评测**，不扩成训练或生产切换。
**20:15:13（北京时间）完成 1800/1800**，总耗时约 2 小时 38 分钟；
20:37 复核 RTX 4090 D 49140 MiB，GPU 利用率 0%、显存占用 0 MiB。

- [x] 入口 `scripts/fresh_baseline_eval.py`：**179 份输入**已冻结并复核，包含新源码、权重、
  六份参考音与评分器资产；本地全量 **338 passed**（两条既有依赖弃用警告）。
  独立目录 `/root/autodl-tmp/fresh_baseline_20261028/`，不覆盖远端脏工作树或旧 CPU 快照。
- [x] **已完成** base 与历史 r26 × 180 条条件 × seed 42–46 = **1800 条**；
  CFG 1.8、20 步、禁重试、Whisper large-v3（th/id 指定语言，tl 沿用自动识别，无期望文本提示）。
  ID 优先调度，不改 case 文件；每条报告/WAV SHA 落盘，失败停本轮，不重抽成功条目。
  `run/status.json` 已为 `stage=complete`，两模型各 900 条，训练/生产变更标记均为 false。
  启动核验快照：**44 份完整逐条报告（base 33 / r26 11，均为 ID）**条件/种子/指标字段匹配，
  不是最终总数或质量结论。两路 TTS/ASR 均实测 CUDA，large-v3 实际精度 `int8_float16`；
  base 为 0 个 LoRA 参数张量、r26 严格加载 384 个。此为离线进程取证，**不是生产加载证明**。
- [x] 用户随后明确「直接改成4路」：新增独立调度 `scripts/fresh_baseline_parallel.py`，
  先让旧两路在当前条目保存后退出，再按每模型两路接续未做项；不修改原冻结源码/plan，
  不改变 1800 条预算、参数或评分。全量 **357 passed**（两条既有依赖弃用警告）。
  **保留旧两路 274 条，四路补齐 1526 条**；四路启动至合并完成约 2 小时 13 分钟（含加载），
  没有相同任务分布的持续吞吐对照，不宣称提速倍数。
  新调度记录在 `run/parallel4/`，屏障退出不代表生成失败，不得按旧日志异常自动重抽。
- [x] 两份完整报告的条件/种子/评分参数逐条核验通过；最终报告、comparison、plan、status 的
  本地/远端 SHA 一致。分语种、plain/control、seed 已复核，补充分析独立落在
  `work/fresh_baseline_20261028/run/postcheck_20261028.json`，不改原报告或重评分。
  置信区间以每语种 **30 个独立文本**聚类，不能把 seed/控制变体当新增独立句子。
- [x] **历史 r26 未使用本次隔离清单训练**，本轮只能建立新文本基线，不能证明清洗收益。
  8 条跑飞回归、48 条阅读探针均不额外追加到本轮预算；8 条待审训练录音仍隔离。
  不开 proj/rank/数据对照、不 merge/上传/晋级，不改变门禁；训练须另定单变量方案。
- [x] 用户问「接下来呢」后完成只读报警定位：r26 的 **7 条疑似少读/漏尾仅涉及 3 个文本**，
  ID `1707` 一条、TH `1695` 一条、TH `1925` 五条；末尾缩写/专名的 ASR 转写差异是待查线索，
  **尚未听音，不能认证误报或真实漏词**。另发现 ID `1753` 原文本含 `bien Ã©levÃ©` 疑似编码异常，
  ID `1816` 的 base plain 五个 seed 转写均缺开头姓名与括号句；输入处理、生成与 ASR 尚未区分。
  `run/triage_20261028.json` 单独保留上述 5 文本的两条件×5 seed×两模型，共 **100 条既有输出**
  的路径、WAV/ref SHA、条件和转写；不下载音频、不重评分、不改冻结文件。
- [ ] 下一步先核上述输入/音频；若修正文，另建版本并补同条件基线，不能修后沿用旧分数。
  用已有 base/r26 WAV 做声学异常盲听；用户不承担母语验收，口音/自然度/情绪仍未验证。
  然后据确认的问题冻结下一轮假设；隔离后的候选不是自动批准训练集，8 条 ID 不回流。
  若做 1 epoch 候选，历史 r26 的 3 epoch 只能作参照，不能据此声称清洗的因果收益。

| 语种（每模型各 300 条） | CER base → r26 | 疑似少读/漏尾 base → r26 | ΔCER 95% 区间（百分点） |
|---|---:|---:|---:|
| ID | 2.27% → 0.79% | 13 → 1 | [−3.33, −0.19] |
| TH | 5.65% → 4.43% | 14 → 6 | [−1.92, −0.61] |
| TL | 2.10% → 1.66% | 1 → 0 | [−1.22, +0.14] |

ID 收益主要在 **plain（3.53%→0.77%）**，TH 主要在 **control（6.31%→4.17%）**；
另外两组及 TL 两组的分层 CER 区间均跨零。三语种合并两种条件后 5 个 seed 均下降，
但 TL 仍不确定；区间为探索性逐项区间，未作多重比较校正。
r26 平均尾静音各语种增加约 **0.10–0.11s**，不能宣称全面改善。
这些是阅读文本上的 ASR 内容诊断，不是短剧情绪/口音验收；**r26 不晋级，生产不切换**。
报告镜像在 `work/fresh_baseline_20261028/run/`；完整 WAV 保留远端，不代表已完成音频备份。

本轮顺手修复：空历史回归分组的 F0 均值原会除零，现记 `None`（未测，不是 0/完美），
补测试；非空评分公式不变，不回写旧报告。执行源码快照已冻结，启动后不再覆盖。

## 2026-10-28：此前无 GPU 阶段，CPU 审计、新留出与 ID 隔离已完成

用户确认实例以无 GPU 模式运行，本轮只做现有文件审计及验收准备。
**没有 TTS、ASR、训练、数据集下载或生产切换；不把本次授权扩成恢复 GPU 实验。**
完整范围、输入哈希、后续闸门见 `docs/finetune_playbook.md` 顶部。

- [x] r26/r27 原清单只读审计：分别 7412/12312 行，共用原 val 1149 行；
  各有 **49 条训练/验证同文**，现有评测文本暴露 0；旧清单与报告未回写。
- [x] 两轮都命中 **同一批 8 条 ID 短文长音待审**：cv22_id 2、fleurs_id 2、gigaspeech2_id 4；
  r27 新增行没有新命中。只记待审，不据词速判坏、自动压缩或删除。
- [x] 检查 **14856 份**训练/验证及候选参考音：解码/格式失败 0；两轮 train/val 精确 PCM
  重叠、同条 target/ref 相同 PCM、PCM 曝光 >3 均为 0；原始来源元数据追溯异常 0。
  这不认证未知身份、录音授权、近似重复或稿音正确性。
- [x] 从远端已有 FLEURS TSV 缓存冻结 **th/tl/id 各 30 个新文本**，plain/control 共 180 条条件；
  已登记 `eval_cases/fresh_text_holdout_20261028.jsonl`，后续混合自动排除。新集尚无合成/评分，
  计划 seed 42–46、CFG 1.8、20 步、禁重试、large-v3；旧分数不能当新基线。
- [x] 10-26 阅读探针中 **48 条中英 ref 条件**已冻结实际参考字节；**24 条缺母语 ref 仍 blocked**。
  没有静默无参考回退，也没有把 48 条说成全套 72 条通过。
- [x] 基座 model/AudioVAE SHA 与缓存 blob 名一致；r26 `65d79023…`、r27 `f88db81e…` 与历史记录一致。
  仅为磁盘文件核验，**不是生产实际加载证明**；AGENTS/TODO 的 r8/r23 生产口径冲突仍待实际服务取证。
- [x] 用户「继续，尤其是 ID」后，已派生 `quarantine/r26` 与 `quarantine/r27`：
  各隔离 **58 行 = 49 条验证同文 + 8 条 ID 待审目标 + 1 条复用待审音频的 ID 参考行**。
  ID 合计各隔离 **25 行 = 16 条同文 + 8 条待审 + 1 条引用行**；不是新增确认 25 条坏音频。
  r26 **7412→7354**、r27 **12312→12254**，ID 分别 **3878→3853 / 6391→6366**。
  旧清单/音频不动，剩余行与 val 保留原始字节；不补抽样、不换 ref、不改前缀或词速阈值。
- [x] 对同一批 8 条 ID 加工音频完成 **CPU VAD**（非 ASR），14856 份音频 SHA 前后核验一致。
  例：cv22_id `0009469` 共 7.674s，VAD 估计语音 2.010s、内部非语音间隔 5.568s；
  这说明低总词速可能含大量非语音间隔，不证明漏标台词。全部继续隔离、不自动批准或裁切。
  **8 条 `origin_audio` WAV 当前均不在盘上**，未完成原音/加工音对照；不下载补回、不拿加工音冒充原音。
- [ ] 人工稿音复核仅作为将来**放回候选**的条件，不再作为隔离或其他 CPU 准备的前置。
  用户不承担 ID 发音/文本准确性验收；新清单均为 `training_approved=false`，
  工程门通过不等于稿音、未知身份、授权或母语质量通过。
- [ ] r26 的 502 条、r27 的 826 条前缀均为历史 `control_inoculation`，**人工 verified control 为 0**；
  tl 同人 ref 仍为 0。不能把中性接种当已核验真人情绪，不能伪造 ref/控制标签补比例。
- [ ] 恢复 GPU 前另行确认：核妥工作清单与参考，先跑新基线，再决定清洗对照或 proj 对照；
  proj 两臂同数据、同基座、先各 1 epoch。旧 r26 的 3 epoch 不冒充 1 epoch 因果对照。

产物：远端 `/root/autodl-tmp/finetune_readiness_20261028/`；
本地 `work/finetune_readiness_20261028/`（仅报告/清单，不含新增数据集音频）。
新产物在 `quarantine/`；执行源码另存 `quarantine_source/`，未覆盖旧 `source/` 或远端工作树。
全量离线测试 **319 passed**（两条既有依赖弃用警告），`git diff --check` 通过。

## 2026-10-23 17:30：id 母语盲听口音包已交付（零 GPU，微调线暂停状态不变）

**这是唯一能把"印尼语地道性"从`未验证`变成有结论的路径**，也是决定 2 里 `enable_proj=true` 值不值得开的前置。
包已建好，**下一步只剩把它发给一位印尼语母语者并把导出的反馈文本贴回来**（不需要 GPU、不需要动 DIS）。

- [x] **内容**：4 个模型（`base` / `r18` / `r23` / `r26`，按你「主要看这四个就行」收窄，原计划 7 个）
  × 6 条 zh-ref 印尼语台词 × seed 42 = **24 条**；同文本、同参考音、同种子，只有模型不同。
  文本取自 `id_prosody_probe_20261020.jsonl` 的 zh-ref 行（`eval_cases/id_accent_pack_20261023.jsonl` 已冻结登记）。
- [x] **盲化**：组内代号 A–D 顺序随机（`--shuffle-seed 20261023`，可复现）；页面不出现任何模型名/轮次名。
  **两道防泄漏**：构建器自带禁词扫描（命中即 `rmtree` 整包回滚）+ 事后独立复扫 `public/`，均**无泄漏**。
  `private/mapping.json`（代号 → 模型 / 自动分数）**不得外发**；可外发的只有 `public/`。
- [x] **页面文案全英文**（听者是印尼语母语者）：标题、三问说明、每条的两个下拉
  （`Non-Indonesian rhythm/accent`、`Obvious generation defect`）、Q1「哪条最像本地印尼人」、导出按钮。
  `data-*` 属性与导出字段名（`ACCENT_BLIND_FEEDBACK v1` / `best`/`accent`/`defect`/`rank`/`note`）未改，解析口径稳定。
- [x] **顺带证明了一件更重要的事：固定 seed 的推理可复现**。base 与 r23 的探针音频已被远端 `auditions/` 轮转删掉，
  重新生成的 **12 条与冻结报告在 `audio_sec` / CER / SIM 上完全一致** ⇒ 这个盲听包不是在拿"另一次抽样"冒充历史结果。
- [ ] **交付路径**：`work/audio_ab/20261023_id_accent/`
  - 外发用：`id_accent_blind_20261023.zip`（5.3 MB / 26 文件 = `index.html` + `audio/` 24 wav，已确认包内无 mapping）
  - 自留：`pack/private/mapping.json`、`plan.json`、`raw/`（原始 wav）
  - `work/` 已 gitignore；脚本 `scripts/build_accent_blind_pack.py` + 6 条守卫测试入库（全量 **286 passed**）。
- [ ] **回来后怎么判**（预注册，别到时候现编）：以 Q1「最像本地人」的票数为主，`accent=yes` 比例为辅；
  ① 若 base 与 r18/r23/r26 **票数无差**且 `accent=yes` 普遍高 ⇒ 地道性不是微调能碰的（属基座先验），
  **`enable_proj` 不开**，把结论写进 `docs/qc_gates.md` 的"口音"行并停在这；
  ② 若某个 LoRA 明显更接近 base 或票数最高 ⇒ 该轮的配方差异（r18 无前缀 / r23 内容向 / r26 音色向）
  就是可追的线索，据此再决定 proj 单变量轮；
  ③ 若反馈是"**像中文/英文的节奏**"⇒ proj 是唯一机制对症的杠杆（四个投影正是跨语言 ref 的韵律泄漏通路）；
  若是"**印尼语内部口音像马来语**"⇒ proj 无效，只能靠新语料。
  ⚠️ 6 组 × 4 条是小样本，**只够定方向、不够当门禁**；`metallic`/`low_snr` 那两条 AUC 教训不要重演。

## 2026-10-23 16:40：r27 已完成 = 不通过；**微调线暂停，等母语盲听与生产选型决定**（当前生效）

r27（25.01h / 12,312 行 / 2309 步，r26 配方等比放大）三套跑完，按预注册六条门禁**不通过、不晋级，生产保持 r23**。
完整数字与归因见 `docs/runs.md` 的 `lora_omni5_r27_scale25h` 段与[续做手册](docs/finetune_playbook.md#2026-10-23r27-结果与归因不通过微调线暂停)。

- [x] **退化的形状不是"内容系统性变差"**：v2 id CER 均值 0.0204→0.0675 **完全由 5 条 case 驱动**
  （`id_nat_20` s44 10.56s/2.50×、`id_nat_06` s42+43、`id_nat_08`、`id_nat_17`、`id_nat_07` audio 2.72→8.16s），
  32 条里 14 条变差但**中位 Δ=0.0000**、剔最差 3 条后均值 Δ 仅 **+0.0062**（噪声内）；
  同期 **id 探针 mean CER 0.0009 = 历代最佳**（超 r18 的 0.0013）、0/54 多读、最长输出 3.2s；
  **dramaprod id p90 尾静音 0.60→0.48 反而更好**。唯一方向一致的系统变化是多读 item 4/246→**12/246**
  （Fisher 双侧 **p=0.072，未达显著**）。⇒ 准确表述是"**无可测收益 + 坏例频次上升迹象**"，
  我已把先前"规模是负向的/内容全面变差"的过头措辞在 AGENTS.md 与本文件改准。
- [x] **归因排除清单（都有数据）**：隐藏构成漂移（新增 4900 行 vs 继承 7412 行在 duration/rate/f0_std/energy/
  >15s/>20s 上全部同分布，差异 ≤3.4%，逐源逐语种同样）；前缀稀释（比例保持 id 9.8%/tl 12.3%）；
  经典过拟合（epoch 仍 3、`origin_audio` 曝光 max=1、>3× 条目 0）；步数导致内容退化
  （借 r23 vs r24 反证：同数据下更多步数是**改善**内容的）。
- [x] **发现一个设计时漏掉的混杂变量**：等比放大把**带 ref 的行稀释**了（19.54% → **15.36%**，相对 −21%；
  新增行只有 9.04% 带 ref，因为同人配对在源加工阶段就定死、剩池里更少），而三套验收全跑在 reference-only 下。
  ⇒ **下次任何"加量"轮必须先按 ref 覆盖率配额补，否则又是多变量轮。** 但它不能单向解释 Δsim 退步
  （r10 的 ref on/off A/B 方向相反：带 ref 的 B 臂跨 seed/跨 cue 均值更低）。
- [x] Δsim 退步来自 **th/tl 而非 id**（r27 id −0.0103 略优于 r26 的 −0.0113；th −0.0091 vs −0.0057、
  tl −0.0078 vs −0.0025）；th 在所有轮次都没有前缀行（接种只在 id/tl），th 数据变多拉大了"有前缀 vs 无前缀"的分布差。
- [ ] **决定 1：生产选型（零 GPU，需要你拍）**。按现行门禁 **r25/r26/r27 都不够格晋级**，r23 是历史上唯一
  被接受过的（且当时是用户以"跑飞可接受、音色地道性优先"的口径放行的），所以"保持 r23"是**没有更好选择**
  的结果、不是"r23 好"。若要换，实质是**你主动越过门禁**，我把代价摆平：
  - **r26（我推荐）**：音色一致性轴全面最好——Δsim −0.0088、id SIM 0.9166、同 ref 跨句 id 0.0129、
    常设跑飞回归 **0/8 唯一清零**、探针 0/54；且 **id p90 尾静音 0.60 vs r23 的 1.02**——这一条超出噪声、
    正对你 10-22 反馈的"尾空白/声音变小"那一族。与 r23 的内容差（id 0.0850 vs 0.0752、th 0.0818 vs 0.0773）
    **都在 ±0.04 噪声带内**。⚠️ 诚实说：Δsim 对 r23 的优势（0.0039）也在 ±0.009 SIM 噪声内，
    **真正站得住的是尾静音与"无灾难级跑飞"**（r23 探针有一条 20.8s、r25 有一条 11.84s，r26 都没有）。
  - **r25**：时长与 v2 内容最好（v2 总体 CER 0.0357、三语种 0 跑飞），但 dramaprod 有 7.97×/11.84s 的灾难条、
    Δsim 只有 −0.0145（比 r23 差）。
  - **r23（现状）**：dramaprod 内容最好，但 id p90 尾静音 **1.02s 历轮最差**、探针有 20.8s 跑飞、
    且有 **29 行评测文本暴露**（分数偏乐观）。
- [ ] **决定 2：要不要开 `enable_proj=true` 单变量轮**（微调线唯一还没排除的杠杆）。
  **建议先做零 GPU 的 id 母语盲听口音包再定**：base/r18/r23/r24/r25/r26/r27 的 id 探针音频
  （各 18 case × 3 seed）**已经全部生成在远端**，同文本/同 ref/同 seed，可直接做盲标——
  听感若是"**带中文/英文的节奏**"⇒ proj 是唯一机制对症的杠杆（它放开 `enc_to_lm_proj`/`fusion_concat_proj`/
  `lm_to_dit_proj`/`res_to_dit_proj` 四个投影，即跨语言 ref 的韵律泄漏通路）；
  若是"**印尼语内部口音像马来**"⇒ 属基座先验，proj 无效，只能靠新语料。
  注意 `stop_proj`/`stop_head` **不在** `target_proj_modules` 内、开了也仍冻结 ⇒ **proj 不是修跑飞的杠杆**。
  开的话必须单变量 + **5 seed**（±0.009 的 SIM 变化在 3 seed 下判不出来），且权重目录必须带 `lora_config.json`
  （`core.py:57` 缺它会回落到 `enable_proj=False, r=8`，键不匹配被静默跳过）。
- [ ] **明确不再做**：任何"加数据/加步数"的轮次（r17 4.6× 与 r27 1.67× 两个独立点都无收益）；
  rank（r=128）如要测必须单独一臂、绝不与 proj 合并（r9 组合拳教训），且当前分辨率判不出其效应量级；
  为压 0.2–0.3s 垫尾开轮；为 1–3s 短 cue 开轮（3s 硬下限）；无假设的 lr/强度网格。
- [ ] **三个结构缺口都不是调配方能补的**（都卡在你这边）：① 1–3s 短 cue（`process_dataset` 3s 硬下限，
  今天 #8/#9 的成因）；② tl 同人 ref（`fleurs_tl` 1833 行**全部 `speaker=unknown`**，已用数据确认是死路）；
  ③ 表演/情绪语料（已核验为 0）。⇒ 只有 **drama_tl / 授权表演语料**能同时补上，四地区短剧链接你给过，
  但音轨语言、真人来源、训练授权、质量**四项全未核验**。
- [x] 待办（不需要 GPU）：id 母语盲听口音包 **已做出**（见上面 17:30 段；按你的收窄只含 base/r18/r23/r26 四个模型）。
- [ ] 待办（不需要 GPU）：`review_4f50387a` 那类 speaker 聚类混人问题的复核清单（属 DIS/识别侧，你说过不动 DIS，
  所以只出清单不改代码）。

## 2026-10-23 14:37：r27 已开训

用户「适度增加训练集和 step」→「清理磁盘并开 r27」。生产与 HF **仍保持 r23**。
完整依据、冻结方案与**预注册门禁**见[续做手册](docs/finetune_playbook.md#2026-10-23r27--r26-配方按-语种源-等比放大到-25h用户授权已开训)。

- [x] 磁盘回收：删 r24/r25/r26 的 `step_*`（只留 `latest`，三个 latest 权重 SHA 复核在位），3.8G → **11G**；
  **r23 的 5 个 step 目录未动**（是证据）。用户明确不做本地权重备份。
- [x] 构建 `r27_scale25h`：继承 r26 的 7412 行**逐行原样** + 按每个 (语种,源) 等比补足
  （scale=1.6658）**新增 4900 行** ⇒ **12,312 行 / 25.010h**，SHA `6058acd8…`。
  **份额逐位不变**（id 47.5/th 22.4/tl 22.4/zh 5.1/en 2.6），**前缀比例未被稀释**（新增 id/tl 行按同一套
  4 短语 10% 接种，合计 id 9.8%/tl 12.3% ≈ r26 的 9.7%/12.7%）。守卫：评测文本命中 **0**、曝光>3× **0**、
  验证文本命中 49（**全部继承**、新增 0）、继承行逐字段不变、val 逐字节拷贝、源池不足只记 `shortfalls` 不重分配。
- [x] **时长上限用 30s 而不是 incremental_mix 默认的 20s**：继承的 r18 系清单本来就有 83 行 >20s（最长 29.63s），
  卡 20s 会让新增数据的时长分布与继承数据不一致（实测新增行 >20s：20s 版 0% vs 30s 版 1.1% = 继承的 1.1%）
  ⇒ 等于偷偷多改一个变量。为此给 `row_issues/eligible_additions` 加了 `max_dur` 参数（**默认仍 20.0**）+ 1 项测试。
- [x] 配置与 r26 的 diff 只有清单路径 / `num_iters`+`max_steps` 1390→**2309** / `warmup` 139→**230** / 输出路径；
  其余超参逐字相同；冷启动、save_path 无 `latest` ⇒ 无 resume。14:37 启动，ETA 训练 ~15:45、评测完 ~16:35。
- [x] 评测驱动改用**轮询 `training_state.json` 的 step** 判完成（不再 pgrep——r26 那次被 ssh 包装 shell 永久命中、
  空等 20 分钟），step 到位后再等显存 <5GiB；桥接用绝对路径启动。三套 744 输出、全程离线、其它模型不重跑。
- [ ] **预注册门禁（全部满足才算晋级候选）**：① dramaprod id CER ≤**0.0800**、th ≤0.0818、tl ≤0.0125；
  ② Δsim(emo−plain) ≥**−0.013**；③ 常设跑飞回归 8 条中 `len_ratio≥2` **= 0**；
  ④ id p90 尾静音 ≤0.5s **且 dramaprod 与 v2 两套都要**（r26 的教训：只看一套会误判）；
  ⑤ v2 总体 CER ≤0.0839；⑥ zh 侵蚀 ≤0.1337。任一不满足 ⇒ 不晋级、保持 r23。
- [ ] 判读预案：内容改善但 Δsim 回吐 ⇒ 规模与情绪稳定也是权衡；两者都保住 ⇒ **首个同时过两轨的轮次**，
  交用户做生产选型；内容没改善 ⇒ "规模不是杠杆"，与 r17（77h/4177 步过拟合）合流，**停止在现有数据池开轮次**。
- [ ] 未决（等用户表态）：**生产选型**。r23/r25/r26 三轮已把构成这条前沿画完，没有任何一轮全面占优——
  内容（dramaprod）r23 最好、时长与 v2 内容 r25 最好、情绪音色稳定与"无灾难级跑飞"r26 最好。
  三者权重都在远端（r26 `65d79023…`、r25 `d3da75c4…`），切换可逆。

## 2026-10-23 10:35：r25（A 臂）与 r26（B 臂）都已完成 = 均不通过

用户授权「先push，然后r25开训」。生产与 HF **保持 r23**（是否切回 r18 用户未表态，权重已双备份可随时切）。
完整依据、冻结方案与预注册门禁见[续做手册](docs/finetune_playbook.md#2026-10-23r25--r18-配方--233-行中性前缀移植用户授权已开训)。

- [x] 构建 `r25_r18_prefix_graft`：**r18 清单逐行原样**（7412 行/15.008h，份额与 r18 完全相同）
  + 把 r23 中 audio 路径重合的 **233 行**（id 162 / tl 71）改成 r23 的前缀文本，四个短语均布。
  守卫全过（行数/audio 序列不变、非移植行逐字段不变、裸文本严格匹配、val 逐字节拷贝、拒绝覆盖）；
  `train.jsonl` SHA `98cc7227…`；**与三套验收集文本暴露 0 条**（r23 是 29 行覆盖 v2 的 21/30 条 TL 文本）。
- [x] 配置与 r18 的 yaml **diff 只有 4 行**（两个清单路径 + save_path/tensorboard），`num_iters 1390`、
  `warmup 139`、LoRA r64/α64、lr 1e-4、batch2×8、proj 关全部逐字相同；冷启动自官方基座、save_path 开训前不存在。
  val 集与 r18 逐字节相同由**构建脚本的 SHA 断言**保证（不是靠 val loss 比对——官方验证只取 10 个 batch 且
  val 也 shuffle，是约 20 条随机样本的估计，跨轮不可比；见 AGENTS.md 的机制记录）。
- [x] 10:35:16 启动，日志 `/root/autodl-tmp/r25_train.log`，ETA ~11:10；评测驱动
  `/root/autodl-tmp/eval_r25_20261023.sh` 自动接力（等进程退出 + step==1390，已有报告拒绝覆盖，
  **全程 HF_HUB_OFFLINE=1**），三套共 744 输出，base/r18/r23/r24 一律不重跑。
- [x] wandb 桥接：开训时漏了（远程 `nohup` 裸启不走 `start_local`，而 `start_bridge` 是它的进程内线程），
  10:45 补起并**回补 step 0→370**（run `m7czdoex`）；r24 那轮也漏了，已用新增的
  `scripts/wandb_backfill.py` 一次性回补 **776 条 / max_step 1499**。约定已进 AGENTS.md。
- [x] **判定：不通过、不晋级，生产保持 r23**（两项门禁未达）：① dramaprod id CER 相对 r18 **+0.0504**
  （越 +0.04 门），全部由单条灾难 item 驱动——`drama_prod_id_nat_17_sref_emo` seed44 生成 **11.84s**
  （len_ratio 7.97、CER 6.97），剔掉这一条后 id CER **0.0645 是五方最好**；② id 探针 mean CER **0.0284**
  越 ≤0.005 门（最长 8.96s 在 `id_ms_colloquial_en` seed43，**与 r23 炸到 20.8s 的是同一 case/seed**）。
  收益轨 Δsim **−0.0145** 未达 r23 的 −0.0127（id 子项 −0.0163 vs −0.0089），改善 +0.0083 在 ±0.009 噪声带边缘。
  ⚠️ 我预注册时把 Δsim 门禁方向写反了（越负越差却写成"≤−0.013 达标"）；按正确方向判仍未达，结论不变、记录已更正。
- [x] **门禁全过的项多为历轮最好**：id p90 尾静音 **0.46**（r18 0.48 / r23 1.02 / r24 0.78）、overall p90 **0.34**、
  id speech_ratio 0.817、th CER **0.0765**（历轮最好）、dramaprod id 跑飞 2/96、v2 三语种 **0 跑飞**、
  v2 总体 CER **0.0357**（历轮最好）、zh 回放侵蚀从 r18 的 0.0758 回到 **0.0470**。
  ⚠️ 但 v2 上 r25−r18 多数差值在 ±0.04 噪声带内，只能说"没变差"。
- [x] **Δsim 剂量-反应坐实（本轮最大产出）**：前缀 0 行(r18) −0.0190 → 233 行(r25) −0.0145 →
  373 行冷启动(r24) −0.0129 → 373 行+续训(r23) −0.0127；id 子项 −0.0246 → −0.0163 → −0.0085 → −0.0089。
  单调、与初始化无关 ⇒ **A 臂只拿到一半收益正对应一半剂量**。
- [x] **中性前缀被排除为时长失稳源**：r25 带前缀却是时长轨历轮最好 ⇒ r23 的 id p90 1.02s 跟着
  **r23 的数据配方**走。阶段 A 审的是形状维度、**没按源审** ⇒ 下一个数据侧假设指向源构成与份额。
- [x] **跨轮新发现：极端跑飞按 seed 聚集**（5 模型×3 套×id 246 条=1230 条）：len_ratio≥2 共 10 条，
  **seed44 占 8 条**（均匀零假设 p=**0.0034**）、seed43 占 2、seed42 **0**；seed44 的 1.2–2.0× 也是 10 vs 5/5，
  p95 1.133 vs 1.042/1.065。**但"跑飞由微调引入"不成立**：base 0/246 vs 微调 10/984，
  在总体率下 base 出 0 次的概率 0.134 ⇒ 不显著。⇒ 用跑飞条数横比轮次噪声很大，1–2 条差异别过度解读。
  已落：AGENTS.md 验收口径新增"必须报逐 seed 的 len_ratio 分布"；常设回归集
  `eval_cases/runaway_regression_20261023.jsonl`（8 条，`scripts/build_runaway_regression.py` 自动生成、拒绝覆盖）。
- [x] 记录已落 `docs/runs.md`（`runlog` 生成）+ playbook 第 6–9 节；权重 SHA `d3da75c4…`。
- [x] **B 臂 r26（`lora_omni5_r26_r18_inoc10`）已完成 = 不通过、不晋级，但收益轨首次达标**：
  r18 清单逐行原样 + 按 `build_r20_emofix.py` 原样逻辑 10% 接种 **502 行**（id 376=9.7% / tl 126=12.7%），
  剂量为 A 臂 2.15×、id 剂量高于 r23 的 7.6%。**Δsim(emo−plain) −0.0088 是历轮唯一超过 r23（−0.0127）的**，
  id SIM 0.9166 与同 ref 跨句一致性 id 0.0129 也都是历轮最好，常设跑飞回归 **0/8 唯一清零**、探针 0/54 无跑飞。
  未达的三项全在 id 时长/多读轴（dramaprod id p90 0.60s 越 0.5 上限、多读 4/96 越 ≤3/96 但**全是中等级
  ≤2.00×**、探针 mean CER 0.0109 越 ≤0.005），**且这三项只在 dramaprod 出现、另两套反向**
  （v2 id p90 r26 0.30 优于 r25 0.34；探针 r26 0/54 优于 r25 1/54；常设回归 0/8 vs 2/8）⇒ 不改门禁，
  但不能读成"r26 时长控制比 r25 差"这一稳定结论。剂量-反应补齐到 5 点：0 行 −0.0190 → 233 行 −0.0145
  → 373 行 −0.0129/−0.0127 → **502 行 −0.0088**（总体单调，id 子项非单调）。
- [ ] 踩坑两条已进 AGENTS.md：① **`preflight()` 会 mkdir save_path** ⇒ 判"无 resume"要看有没有 `latest/`，
  不是看目录是否存在；② `A && export PATH=… && cmd &` 把 export 关进子 shell，同一 ssh 的后续段落拿不到 PATH
  （r26 的桥接第一次就是这么死的：训练正常、`nohup: failed to run command 'python'`），起多个后台进程一律用绝对路径。
- [ ] 未决（等用户表态，不自动执行）：**生产是否从 r23 切回 r18**。r25 的结果让这件事更需要重新看——
  r18 与 r25 都属"r18 配方"系，时长轨干净、v2 内容最好，但都各有 1–2 条极端跑飞；r23 的强项是
  dramaprod 内容与 Δsim。切换可逆（r18 权重本地+远端双备份，SHA `2c2b4af0…`）。

## 2026-10-22 23:00：r24 已推送，接下来的微调计划（阶段 A–D，**均未授权执行**）

已推送 `c2ca772`（runlog/preflight 修复）+ `4af548f`（r24 记录与取证）。生产/HF 保持 **r23**，未 merge 未推送权重。
完整依据与判据见[续做手册](docs/finetune_playbook.md#2026-10-22r24-之后的微调计划阶段-ad)。

> ✅ **阻塞已解除（2026-10-23）**：实例恢复（同一端口 51348），远端 checkpoint / 清单 / 报告全部存活，
> 仓库里上一轮同步的 runlog/launcher 修复也在（SHA 双侧一致）。已按文档要求把只在远端的权重拉回本地：
> **r24**（`ecd19f05…`，145MB）与 **r13x**（`1c4092cc…`，145MB，此前无任何异地备份）都在
> `checkpoints/<run>/latest/`。r23 在 HF 有备份、r8 在 `checkpoints/gpu_pause_20260918/`。
> r17（4177 步过拟合轮，已判不通过）未备份，判断为无保留价值。**r18 清单已核实仍在盘上（7412 行/15.008h）。**

**先分清两个被混为一谈的现象**（这是本轮最大的收获，也是排序依据）：
- **垫尾 0.2–0.3s**：base p90 0.18s → 各轮 0.28–0.50s，**每个微调轮都有、量级小**；数据侧三次证明修不动
  （r4 数据尾压到 0.15s 模型仍垫 0.25s；2026-10-22 训练目标尾全 ≤0.17s 生成仍 +0.18~0.22s；r24 冷启动仍 +0.60s），
  且线上 `trim_silence_vad=True` 会裁掉 ⇒ **接受它，永不再为它开训练轮**。
- **跑飞/不停（1.2s 窗生成 3.5–20.8s）**：r19 起变多，是**真问题**（今天 #8/#9、id 探针 20.8s、dramaprod 3–5/96 都是它）。
  r24 证明它**不是步数/初始化能修的**（冷启动只是把它挪到别的 case 上），但**还没排除数据侧**。

- [x] **阶段 A 已执行完毕（2026-10-23，只读零 GPU）：三个靶子全部被数据否掉 ⇒ 判定「跑飞不可由数据修」。**
  产物在 `work/r25_data_audit_20261023/`（脚本 + 审计日志 + 聚类核查）与远端同名目录。
  ① **长目标不是靶子**：r23 的 >15s 行比 r18 更少（id 74→46、th 17→15，**>20s 在 id/th 清零**），
  剩下 65 条 >20s 全在 `tl/fleurs_tl_r23boost`，而 tl 是唯一没跑飞的语种（0/96、p90 0.28s）；
  ② **文本-时长错配不是靶子**：三语种 rate 分位数两份清单几乎逐位相同，"长音频+低 rate"错配行
  id 还从 1.44% 降到 0.78%；
  ③ **我自己提的"前缀行更长"也是错的**——那个 p90 14.89s 是把 id 与 tl 混算的假象，分语种看
  前缀行与裸文本的时长分布几乎一致（id p50 6.21 vs 6.29、tl 12.03 vs 11.58）。
  唯一"从无到有"的变化是 **373 条中性前缀行**（r18 0 条 → r23 4.66%，id 子集 7.6%），
  4 种均布的**中性英文散文**（`(neutral, natural tone)`/`(calm and steady)`/`(speak naturally and clearly)`/
  `(relaxed and natural)`），不声称情绪 ⇒ 全清单 `emotion_verified=true` 为 0 行是自洽的，
  **但这个口径解释必须留着**，否则会被当成伪造标签。削弱它是靶子的证据：r24 的跑飞在 `_emo` 与
  `_plain` 两侧都出现（`id_nat_24_nref_emo` 与 `_plain` 都炸）。
- [x] **阶段 D 前置已完成：`review_4f50387a` 聚类确实混了多个真人。** 本地对 10 条参考做两两 WavLM cosine
  （45 对，零新增 TTS/ASR）：**min 0.4672 / 中位 0.7916 / max 0.9841，5 对 <0.60**——
  `cue9↔cue24` 0.9841、`cue2↔cue10` 0.9543 是紧簇，而 `cue18↔cue20` 0.4672、`cue20↔cue24` 0.4934 明显另一批人；
  **今天 #8 用的 cue8 参考与 #9 的 cue9 参考只有 0.6824**。同集其它 speaker 的参考池都自洽
  （`review_1ba2acde` 0.9391；`review_07ea0705`/`review_82dd7e8a` 各自只用一条参考字节）。
  ⇒ **用户"#8/#9 是同一个人"的前提在参考层面不成立**，#8/#9 首先是身份链问题、其次才是模型问题；
  阶段 D 的"锁定 canonical 参考"**必须先修聚类**，否则锁的是错池子。边界：1.58–4.84s、SNR 0.54–2.34dB
  的短噪参考上 WavLM 是弱证据，本地 yin 的 F0 与 DIS 在低有声占比参考上差 50–150Hz（八度跳），只当线索。
- [ ] **阶段 B 不启动**（A 没找到靶子）。唯一剩下的数据侧单变量是"r23 − 373 条前缀行"，但 r24 已证
  **情绪下音色稳定的收益就来自这些前缀行**（Δsim −0.0129 在冷启动下保留）⇒ 拿已确认收益换机制不明的改善，**不做**。
- [ ] **阶段 C（r23×r24 权重平均，零训练）期望价值已下调，等用户决定**：判据要求 id CER ≤ r23+0.04 = 0.115，
  而 r24 是 0.1298、r23 是 0.0752 ⇒ 线性插值预期约 0.10，**刚好卡在门内、余量极小**。
  跑飞是离散事件、平均可能同时压住两边极端，但内容退化更可能按权重比例带进来。成本仍低（≈690 输出），
  跑不跑都不放宽判据，不过就是阴性结论并归档。方法仍是**秩拼接**（`A=[A23;A24]`、`B=[0.5B23,0.5B24]`、
  `r=α=128`），评测前先过 `load_lora` 断言。
- [ ] **阶段 D（推理侧，现在的主线，本机可做）**：① 先修 speaker 聚类（`review_4f50387a` 这类池内不自洽的
  不能当一个说话人用）；② 同 speaker 锁定 canonical 参考、identity 重试**不换参考**；
  ③ 参考准入门（duration<2s、`active_speech_ratio`<0.5、SNR<3dB、gender unknown 不得作唯一锚点）；
  ④ 极短 cue（<1.5s 窗）跑飞后**不 keep-best 出片**。
- [ ] **顺带发现（与音色一致性直接相关，待决定要不要开轮）**：带 ref 的行 r18 19.5% → r23 15.5%，
  跨语种 ref 两轮都是 **0**，且 **tl 带 ref 两轮都是 0**（0/990、0/1535）⇒ 线上唯一在用的
  `[103 ref 104][text][101 target 102]` 打包路径**在 tl 上从未进入训练**。这解释不了 id 的跑飞，
  但它是"同 ref 跨句稳定性"在 tl 上无法从训练侧改善的结构性原因。
- [ ] **明确不做**：不再为垫尾开训练轮；不做步数/lr/r/强度网格（r24 已测出"内容与尾巴同向恶化、没有甜点"）；
  不为 1–3s 短 cue 开训练轮（3s 硬下限，结构上学不到）；不动 HF 与生产（保持 r23）。
- [ ] 用户 2026-10-20 的两条目标现状：①「短 cue 质量支持」⇒ 已三次证明**微调侧无解**，转阶段 D；
  ②「情绪下音色别偏」⇒ **已达成并保持**（Δsim base −0.0140 / r23 −0.0127 / r24 −0.0129，
  且 r24 证明该收益来自数据里的情绪前缀、不来自续训链，所以它是可复现的资产，不是运气）。

## 2026-10-22 21:26：r24 冷启动单变量轮（已完成，不通过）

用户看完 #8/#9 取证与 r19→r23 时长回归证据后决定：**保持 r23（不回滚 r18）、直接开训**。
暂停已解除，但只授权这一个单变量轮；不追加 epoch、不改 CFG/步数、不动 HF 与生产。
完整依据、冻结方案与判读规则见[续做手册](docs/finetune_playbook.md#2026-10-22r24-冷启动单变量轮用户授权已开训)。

- [x] 成片取证（第43剧108集 #8/#9）：同 speaker key 但**参考字节不同源**（#8 `episode_current`，
  #9 是 identity 预算用尽后换的 cue24 `episode_alternate`），WavLM `ref#8↔ref#9` 仅 0.6824 而
  `ref#9↔ref#24` 0.9841 ⇒ #8 那条参考是池内离群值；两条控制条件也不同（#8 style control 生效、
  #9 `style_control_suppressed` + reference-only）；模型侧失败形态是**1.2s 窗短 cue 跑飞**
  （3.52s / 6.40s / 6.26s），#9 最终音频是从 6.26s 跑飞硬裁前 1.34s、keep-best 出片。
  冻结在 `work/video_review/20261022_43_108/`（视频 SHA `a6a24d9a…`）。**未新增 TTS/ASR**。
- [x] 核验生产权重现实：HF 上发布的**是 r23**（LFS oid 与远端 step2909 逐字节一致、README 写 r23），
  「生产指定 r8」只是指定口径不是加载证明；GPU 实例实际权重仍未核验（用户选择不追）。
- [x] 音色一致性复算：r8 vs base 无系统性退化（Δmean +0.0001~+0.0023，全在 ±0.009 噪声内）；
  r23 的同 ref 跨句离散度比 base 更紧（id 0.0072 vs 0.0136），未晋级的 off/on 更松（0.0192/0.0267）
  ⇒ 用户听到的问题量在**时长/停止**轴上，不在声纹相似度轴上。
- [x] 开训 `lora_omni5_r24_coldstart`：r23 清单原样 + 官方基座**冷启动** + 3 epoch/1500 步，
  与 r23 的 `diff` 只有 num_iters/max_steps/warmup/路径 4 处；北京 **21:26:58** 启动，
  日志 `/root/autodl-tmp/r24_coldstart_train.log`，val step0 1.021114，约 44 分钟。
- [x] 本轮顺手修：`runlog._read_yaml` 缺配置静默返回 `{}`（**这就是 r19 被错记成全量微调的原因**）改硬失败；
  `preflight` 的 3-epoch 警告补 `ceil` 余量。新增 4 项测试，全套 **253 通过**，两文件已同步远端。
- [x] 训练完成核验：进程退出 + `training_state.json` **step=1500**（22:08 落盘，`lora_weights.safetensors`
  144,749,240B + `lora_config.json`）；val loss 首 **1.021114** → 末 **0.919980**、最优 0.906632(step1000)，
  按铁律一律交付 `latest`，val loss 只用于确认无发散。
- [x] 评测：v2-118、dramaprod-112、id 探针-18 各 seed 42/43/44（共 744 输出），由
  `/root/autodl-tmp/eval_r24_20261022.sh` 自动执行（等训练进程退出且 step==1500 才开跑、已有 r24 报告拒绝覆盖、
  shard 非零退出即停）；base/r23 **未重跑**。
- [x] **判定：不通过、不晋级，生产保持 r23、HF 不动、不 merge 不推送。** id CER 0.0688→**0.1298**
  （Δbase +0.0610 越红线，v2 同向 +0.0646）、id 跑飞 **5/96**（r23 3/96、门禁 0/96）、id p90 尾静音
  **0.78s** 仍越 0.5s 上限、speech_ratio −0.170、audio_sec +43%、th 丢掉 r23 的收益（0.0773→0.1183）；
  只保住 Δsim(emo−plain) −0.0129 与 tl 持平。**「续训链累积是主因」被否。**
- [x] 结论已落 `docs/runs.md`（`runlog` 生成，未手抄）+ playbook 第 6 节；commit message 待写全
  数据/配比/超参/分语种验收/结论。**晋级与 HF 推送都要另获用户确认，本轮不触发。**
- [ ] **新认识（影响后续所有轮次）**：冷启动不是全面更差，而是**把跑飞挪到不同 case 上**
  （r23 炸 `id_ms_colloquial_en` 20.8s；r24 炸 `id_nat_02/06/20/24`，后者 base 音频只 1.6–2.8s）
  ⇒ 1–3s 出分布是结构性的（3s 硬下限），微调只能改变它落在哪些 case 上、不能消除。
  且 Δsim 在冷启动下保留 ⇒ **情绪前缀的收益来自数据，不来自续训链**。
- [ ] **不再为「修尾静音」开训练轮**：r4（数据尾 0.15s、模型仍垫 0.25s）、2026-10-22（训练目标尾全 ≤0.17s、
  生成仍 +0.18~0.22s）、本轮（冷启动仍 +0.60s）三次独立证据都指向**生成侧**。要降尾静音/跑飞只能走推理侧：
  DIS 长度门与重试策略、**同 speaker 锁定 canonical 参考**（今天 #8/#9 的直接成因）、可选 VAD 尾裁。
  再开训练轮的前置是新的可证伪假设，不是继续调时长卫生或步数。
- [x] 磁盘卫生：r24 目录实测只剩 `step_0001499`/`step_0001500`（官方脚本自留最后两个，各 415M）+ `latest`，
  远端仍剩 13G ⇒ **本轮未删任何 checkpoint**（不必要时不做不可逆操作）。r23 的 `step_2250/2500/2750/2908/2909`
  是证据，保持原样。
- [ ] 仍未做、也未被本轮取代：`review_4f50387a` 这个 speaker 聚类是否混入多个真人
  （参考 F0 跨 142.9–333.3Hz、4 条判 female vs `declared_gender=male`、`anchor_similarity` min 0.1354）；
  #8/#9 登记为开发坏例（`training_eligible=false`）。这两条属 DIS/识别侧，不靠微调解决。

## 2026-10-22：暂停交接（暂停已由用户解除；下列未勾选项仍有效，不自动插队）

用户要求“先将todo和后续的规划写到文档中吧，我后续再搞”。**后续执行暂停，本节覆盖此前的“继续”授权**；
本次只整理文档，不制作新试听包、不启动TTS/ASR/训练或新GPU评测，不变更生产。

- 已完成：base/r23/off/on各1000输出、off/on各393步训练；旧声学包36条反馈及归档；
  on强度0.5/0.75新增180条于北京时间**2026-10-22 19:03:41**完成，输入/报告/WAV哈希复核通过。
- 当前结论：**生产仍指定r8；0.5不推进，0.75仅保留局部诊断，均不晋级**。
  0.75缓解D01尾空白、D05头空白和D02低电平，但D06未改善，三语种尾静音风险仍在。
  新音频尚未人工听评，语言质量未验证；base对照不等于已完成生产r8权重核验。
- **尚未制作on 1.0/0.75新试听包**，没有新包号或可用入口；旧`acoustic-5b297d423cc5`是base/on 1.0，
  已评完，不要误认成新包或要求用户重复提交。

用户恢复后按顺序处理，具体输入、路径和操作边界见[续做手册](docs/finetune_playbook.md#2026-10-22暂停交接与续做计划)：

- [ ] 公钥重连，核验既有完成状态、冻结计划/报告/ref/WAV哈希；只读检查，不重跑已完成任务。
- [ ] 补最小强度试听导出适配及定向测试，再用**既有WAV制作6组/12条on 1.0 vs 0.75盲听**：
  D01/D02/D05/D06原seed，另加按原抽样顺序选定的TH/TL对照；不新增合成/ASR、不裁边或归一化。
- [ ] 收声学异常反馈，分诊断/对照记录，写入新副本并保留原ref、参数、SHA、QC和未验证项；
  合成候选仍为`development_only/training_eligible=false`，不充当独立验收或训练语料。
- [ ] 根据听评与既有五seed证据再决定是否值得下一轮；先复核2条极低电平TL训练目标，
  不把它们直接认作根因。新训练/大评测须另冻假设、回归、预算并获授权，不自动追加epoch。
- [ ] 长期停用前另核备份：**本机只有本轮元数据/报告，WAV与模型未完整备份**。
  本次不关机、不释放实例、不删盘；此前GPU空闲不等于实例已关闭或停止计费，旧试听服务/隧道当前状态未重查。

## 2026-10-22：两日提交复核后的执行顺序

本节为已执行记录，覆盖下方旧条目的“缺 GPU 入口”与待开训顺序；独立基线→前缀 A/B→分语种复评的历史授权不延伸为新任务，**后续以顶部暂停交接为准，不自动晋级生产**。
10 月 20–21 日历轮训练与分数不改写；r23 是离线候选版本，不能据此认证线上加载权重。

- [x] 远端公钥接入；审计/执行源码在 `/root/autodl-tmp/review_20261022/`，新执行目录
  `/root/autodl-tmp/control_ab_20261022_v4/`；未覆盖脏工作区或旧清单。
- [x] 核对 r19/r22/r23 实际清单：r23 有 **29 条训练行覆盖 v2 的 21/30 条 TL 文本**，
  也覆盖 drama 的 7/8 条 TL 独立文本；不是独立泛化收益证据。另有 65 条目标时长 >20s。
- [x] 增量脚本补齐评测/验证文本、音频/身份/session、时长、曝光与来源字段守卫；已有输出拒绝覆盖。
- [x] refrob 改按原目标补做 B−A 配对：重度组合降质下 TH/ID 有局部内容收益，SIM 不再误作降质侧否决门。
  只加独立复核报告，历史“不晋级”决定不变；详见 `docs/ref_robustness_stage2_20261020.md`。
- [x] 控制前缀单变量 A/B 已备料：从 r19 剔除 40 条验证同文行，两臂各 **6367 行/12.85h**，
  只切换 254 条可信 TH 前缀；这是先前冻结的 3 epoch 备料版，未执行、不覆盖。
- [x] 按用户最新方向生成两臂一致的降回放版：**6273 行/12.6414h**，ID/TH/TL 目标全保留，
  中英目标时长占比 **3.9989%/0.9938%**；254 条可信 TH 控制、1340 条同人 ref。
  每臂 **393 步（约 1 epoch）**，同基座、全新优化器/调度器、seed42；不自动扩到旧 3 epoch。
- [x] 独立新文本每语种 30 条，中文/英文参考 **80%/20%**，每文 plain/control × seed42–46；
  另 10 条历史压力文本同样配对，仅作回归。每模型共 **1000** 输出，不能把两个控制条件算独立文本。
  基座权重内容 SHA、训练/验证 PCM、已知身份/session、原始来源与 RNG 初始化检查通过。
  未知说话人、近似音频重复、录音授权与预训练文本暴露仍未认证。
- [x] 旧 `ref_en_03/ref_tl_02` 与候选训练血统中的 PCM 重合；新独立集不再复用旧参考，
  改用可追溯当前 FLEURS 验证音频（中英各 3 条）。旧评测和历史 WAV 不重写。
- [x] **12:05（北京时间）按用户要求提前启动 r23 评测**：保留两个 base worker，先增开一个
  r23 worker，总并发上限 3；另一 r23 分片等槽位释放，启动前须留有 16GiB 空闲显存。
  旧调度器暂停防重复启动，冻结源码/case/seed/权重/总生成数不变，不重跑已生成音频。
  接管记录 `overlap_schedule.json`，当前队列日志 `overlap_pipeline.log`；
  两模型全部完成并校验后仍接原 off/on 各393步训练和复评。并发耗时不作孤立模型速度比较。
- [x] **14:14 评测加速接续已就绪**：base 已完成1000输出，r23 已完成719/1000；
  最后一个旧分片仍运行，不强停丢失其内存中的逐条结果，也不重复合成已完成音频。
  `fast_handoff.log` 确认仅暂停外层 shell9080，基线调度器9081/worker31925不动；
  接续进程37603等基线成功及SHA校验后执行原393步训练，再切新动态评测。
  新评测最多4worker、启动逐个暖机且至少12GiB空闲显存；每个case/seed排他认领并立即保存报告/WAV SHA。
  参考PCM声纹缓存的三语种小样本声学指标完全一致，声学评分局部耗时下降约16%–28%；
  **不是端到端提速比例**。五seed、20步、CFG1.8、Whisper large-v3与目标输出数均未改；
  完整GPU吞吐与四并发峰值显存尚待复评实测。源码/记录见playbook。
- [x] **17:45 全部完成**：base/r23/off/on各1000输出（900主集+100历史回归），off/on各393步；
  四份报告SHA、case/seed完整性与条件已核对，GPU进程退出。WAV/模型仍在远端，本地仅备份报告与日志。
  主集CER（ID/TH/TL）：base **0.0122/0.0537/0.0216**；r23 **0.0071/0.0344/0.0193**；
  off **0.0251/0.0523/0.0275**；on **0.0072/0.0363/0.0226**。
  on相对off的TH内容改善有配对证据，ID/TL的CER差异区间仍跨零；不能认证情绪/母语质量改善。
  **两臂均不晋级**：相对base，三语种p90尾静音增量均越过0.10s门槛；历史回归另有有声占比风险。
  r23也有ID/TL尾静音门槛与历史ID内容风险，不能因主集CER低直接升级。完整记录见`docs/runs.md`顶部。
  r8 的完整历史训练清单在远端缺失，本轮不冒充已完成 base/r8 生产诊断。
  方案、入口和限制见 `docs/finetune_playbook.md` 的 2026-10-22 节。
- [x] 既有WAV分层声学盲听包已生成：**base/on 18组、36条、297.76秒**，
  12组元数据随机对照（每语种4组，中文/英文参考3/1）+6组风险诊断；不混算坏例率。
  远端`review_20261022/acoustic_review_v1/`，包号`acoustic-5b297d423cc5`；
  原始WAV/ref逐字节校SHA，私有目录保留映射、实际生成参数、评分和所选文本全部5seed证据。
  本机SSH隧道试听入口`http://127.0.0.1:16006/`，只服务`public/`，不公开映射。
- [x] CPU核完6273条训练目标：6271条可测尾静音最大0.17s、各语种p90均0.15s；
  另2条TL极低电平，边界未知，不当作零静音或有效语音通过。
  on主集尾静音增长在plain/control、中英ref、5个seed中均存在，不只由少数极长输出拉动；
  暂不支持“训练目标原本有大量长尾”解释，也不能就此证明stop/EOS机制是根因。
  历史ID内容与TH疑似漏尾、ID/TL有声占比风险已定位到逐文本记录，仍非人工确认缺词。
- [x] 用户已评完36条，按随机/诊断分层回写`feedback_review/`新副本，原报告不变：
  R组24条全为无明显异常；D组4条明确异常**全部来自on**，对应base均无明显声学异常。
  ID：D01-B末尾空白、D02-A声音过小；TL：D05-A开头空白、D06-A声音稍小；TH诊断对照无明显声学异常。
  D02/D06的整段RMS相对base低16.53/3.73dB；不是LUFS，也不替代母语/内容验收。
  32条声学正常候选+4条回归连同原ref/参数/SHA/QC归档在远端
  `/root/autodl-tmp/work/for_train/ab_candidates/acoustic-5b297d423cc5/`；
  全部`development_only/training_eligible=false`，不是新增训练集。
  4条回归登记`eval_cases/acoustic_feedback_20261022.jsonl`，保留用户原话；D05确认的是**开头**空白。
- [x] 用户授权继续后，已冻结并启动on强度诊断：保留现有18组固定输入与seed42–46，
  对on做LoRA强度0.5/0.75单变量对照；base/1.0复用既有180条同条件WAV，仅新增180条。
  工作目录`/root/autodl-tmp/strength_ab_20261022_v1/`，独立源码`review_20261022/strength_code_v1/`；
  新计划SHA为`0c970e1dd10301a68eaf1f338bd1874c7539b721ebf125884ea9f321354eb803`。
  本地47项、远端17项定向测试通过，原权重/参考/报告和已验证缓存实现均已核验；
  最多4worker逐个暖机、逐项排他落盘，失败停止，不隐式重跑。运行状态见该目录`status.json`和worker日志。
- [x] **19:03:41完成180条新输出与分层复核**；四模型各90条、351份冻结输入及报告/WAV SHA均通过，
  GPU任务退出。0.75相对1.0：D01尾空白2.40→0.38s，D05头空白1.80→0.02s，
  D02整段RMS提高12.08dB；D06小声未改善。这些是工程指标，尚无新音频真人反馈。
  0.5在D05同文本seed43/46出现疑似续写/多读（17.44/29.60s，CER0.8242/2.3407），不推进。
  0.75随机组ID/TH/TL的p90尾静音仍比base多0.22/0.18/0.18s，**两档均不晋级**；
  不能以局部坏例缓解替代整轮验收。结果、分层数字与限制见`docs/runs.md`顶部。
- [ ] 暂停后的下一步：用户恢复时仅用既有WAV对0.75做少量声学盲听，包含改善例、未改善的D06与TH/TL对照；
  0.5新增的内容风险保持疑似，不拿ASR自动标记冒充母语确认。暂无新GPU任务或追加训练。
  不裁旧WAV过门槛、不扩epoch、不替换r8；后续备料先复核2条极低电平训练目标。
- [ ] 本包声学盲听已收到，但全体样本声学异常率、母语质量仍未验证；
  不要求用户判断三语种自然度，不把D组富集样本当线上坏例率，也不把声学正常当ASR误报认证。

## 当前边界（2026-09-17，用户更正后）

**指定生产基线 = r8**（HF `FrankLiuDundun/voxcpm-finetune-lora`），r9 未动生产。
实际线上实例的代码/权重身份尚需核验，HF发布成功不等于实例已加载；见 `docs/inference_chain_audit.md`。

**2026-10-19：目标语种收缩为 th / tl / id（用户决定）。** 不再微调 VN（vi）与 MY（ms），优先 id / PH(tl) / th。
registry 的 7 个 vi/ms 源、eval 语种白名单、UI 文案与配比全部同步收缩；混合配比改为 **th/tl/id 等权各 28.3 分 + 中文 10 + 英文 5**。
`omnivoice_prod_v2.jsonl` 182 → **118** 条、`fleurs_test_holdout_20260917.jsonl` 150 → **90** 条、`acoustic_regressions_20260917.jsonl` 3 → **2** 条（纯删行）。
refrob 阶段 0 的 6 份 `refrob_20261012_*.jsonl`（30 case）已删除，重建为 **`refrob_20261019_*.jsonl`（18 case，th/tl/id 各 6，9 条唯一 ref）**，阶段 1 生成量 1080 → **648**。
⇒ 旧报告的**总体聚合**（base 总体 CER 0.0774、model_card 的 910 条生成）不再可复现；但逐 case 独立、`by_lang` 按语种聚合，**th/tl/id 的分语种数字仍有效可对照**。
下面所有 `[ ]` 条目里残留的「五语种」「VI」措辞按此口径读；已勾选 `[x]` 的历史记录不改写。

- 用户已于2026-09-20提供ID/VN/TH/PH短剧来源链接，**尚无已核验可训练的drama素材**；
  来源登记见`docs/corpus_sourcing.md`（**VN 那行只是档案**：vi 已退出目标语种，链接从未下载/核验，不启动采集）。用户不能评价三语种的发音、口音与语言自然度；能听出
  爆音、金属感、异常娃娃音、明显音色跳变等 badcase。不再把交成片或完成母语评分作为实验前置。
- 当前目标：**减少明确声学坏例、提高跨 seed / 跨 cue 稳定性，并检查各语种内容指标是否回退**。
  不声称情绪、口音或语言自然度提升。评价分工见 [qc_gates.md](docs/qc_gates.md)。
- drama 曾被用于同时补「表演示范、短对白、同人 ref」三个缺口，但后三者不是同一问题：
  短句和同人录音不必来自短剧。真实表演数据保留为后续选项，不是当前唯一数据路线。
- 以下为执行计划，**未勾选的能力与实验均未完成**。不因计划更新自动替换生产模型。

## 最高优先级：原始TTS明显badcase（2026-09-23用户更正）

**先降低模型原始输出的明显坏例率；读稿感、口音、韵律、情绪自然度与混音润色全部放到最后。**
漏词、乱读、重复、无声等明显内容/生成失败仍属最高优先级，不因自然度优化后移而放过。
本节覆盖下方旧路线的执行顺序，暂不新增润色实验或索要相应评分；必要的完整性/安全检查不撤销。

- [ ] 先处理已有沙哑/杂音/破音/异常变声等明确反馈，不等待用户补新的读稿感时间点或母语评分。
  核同take原始TTS与实际生成条件；只拿到成片的case仍标生成层待定位，不把混音异常直接算模型失败。
- [ ] **紧邻同说话人优先保音色一致性。** 身份分配→可信共享参考→候选音色排序先于情绪/自然度；
  不把不同speaker按同名强并，也不让相似度收益抵消漏词/伪影。局部实现与未决项见下方新两片复盘。
- [ ] 优先执行已冻结的24条base/r8诊断：核基座/r8文件及实际ref、文本、控制、CFG/步数，
  保留原计划各3seed、禁内部重试，直接检查每条原始WAV，而不是只听筛选后的最佳成片。
  当前仍0生成，需要独立Linux GPU入口；不改旧seed组，不以请求model_id冒充权重验证。
- [ ] r8更差则单独验证LoRA强度/基座回退；可靠输入下两者均反复失败，则按坏例类型准备定向微调。
  必要时增加VAE重建诊断，尚未执行。不把持续严重坏例永久归为“无需微调”，也不直接续训r8。
- [ ] 验收分别记录固定case×seed下的原始TTS明显坏例率、同4/6总预算下严重坏例残留率及调用耗时；
  声学与内容异常分开，缺证据/未评单列，不算通过。ASR pass、keep-best出片或压背景都不等于原始TTS修复。
  同参考风险簇不充当独立身份样本；少量单语种诊断不替代三语种回归。

## 需要你配合的事项（2026-09-24更新）

以下为用户侧清单，未约定截止时间；工程取证、参数对照和结果归档由助手处理。

- [x] **本轮CFG反馈已收到，不需再补听。** 9组均已反馈，G01-B保留“未评”及长空白备注。
  R组6条全部异常；G组长空白已定位原API首句，非试听拼接额外插入；P组6条无明显异常。
  默认CFG保持1.8；语言未验，不要求外语自然度评分。
- [x] **2026-10-22 已提供独立 Linux GPU 入口并验证公钥接入**，不再索要入口。
  已完成的新独立基线/控制前缀 A/B 与后续暂停交接见顶部；旧 24 条 base/r8 生产诊断仍未执行，
  不自动新租机器、重启生产 GPU、直接续训或把 r23 离线文件当线上加载证明。
- [ ] **有空：补听旧包尚未收到反馈的3个片段。** I01（ID约64秒）、P01/P02（TL约85.48/87.27秒）；
  只标成片/参考有无明显声学异常及时间点，不确定可留空。入口为DIS仓
  `work/qc_review_20260922_batch/index.html`；无需重听整片，其余片段选听。
- [ ] **本轮新修复待部署和局部验证。** `mtgg3m0`/`mtggf9f`已核实DIS `7185f04`源码生效，
  不再要求为该旧版本重复部署；新增连续同声校验/软风格不压低身份权重已随DIS `b5f4fd4`推送至
  `feat/recognition-generation-mainline`。新片`mtgix3k`已核含该修复的`ce01d0d`包生效，
  但这是第27剧第90集；第31剧第96集（第7集）的重新识别与新锁版结果仍待验收。
  保持r8与局部背景平衡默认关闭；旧锁版1/2/3身份错误须订正或重识别，单纯重渲不会修正。
- [ ] **最后阶段再做：标出“读稿感 / AI味”的具体位置。** 暂不需要补充；明显TTS坏例主线完成后再排。
  不作为当前严重TTS坏例诊断的前置条件，不要求判断外语是否地道。

已完成、无需重交：M01/M02混音听评、E01情绪对照、R01参考定位、TH七条听评及四地区来源链接。
暂不需要你做三语种自然度评分、重新搜短剧链接或决定微调超参；局部背景平衡继续默认关闭。

## 当前待办（按顺序执行）

### 2026-10-12：差参考音鲁棒性（分阶段，先本地后GPU）

用户 2026-10-12 提出：传入差参考音（多人声串入/发闷频带受限）时仍保持输出质量，
允许牺牲部分参考一致性换差参考下的输出质量（speaker_sim 全量记录、放宽门禁）。
只动模型侧数据增强，不改推理链路。冻结口径见 `docs/ref_robustness_plan_20261012.md`。

- [x] 阶段 0（本地 CPU）完成：确定性降质器 `src/voxft/data/ref_degrade.py`
  （5 个冻结 profile：babble_sir15/babble_sir8/lowpass4k/lowpass3k/combo_babble8_lp3k，
  含 CLI 与 sidecar）、评测集构建 `scripts/build_degraded_ref_cases.py`、
  6 个条件文件 `eval_cases/refrob_20261019_*.jsonl`（**18 case**，case_id 跨条件相同）、
  训练侧 50% ref 降质脚本 `scripts/degrade_train_refs.py`、测试 `tests/test_ref_degrade.py`
  （全套 163 通过）。降质 wav 不落本仓，待远端生成。
  - 2026-10-19 随语种收缩重建：原 `refrob_20261012_*.jsonl`（30 case = 5 语种 × 6）已删除，
    `LANGS` 改为 `(th, tl, id)` 后按同一 seed 42 重生成。**新 18 条不是旧 30 条的子集**——
    th/tl 抽样与旧集完全相同（各 6/6 重合），id 只重合 2/6（`select_cases` 共用一个
    `Random(42)`，删掉 vi 那一档挪动了 id 的随机流），所以新旧两集不能互比。
    旧集从未生成过音频（阶段 1 未跑），无对照损失；需要时从 git `b8a768e` 取回。
- [x] 阶段 1（GPU 基线量化）：**2026-10-20 完成**（用户提供 autodl 入口并确认继续）。
  远端已生成 **9 条被引用 ref**（全部按 SHA 回溯到 `fleurs_{zh,en,tl}_tc` val 行，
  ref_lang 分布 zh 5 / en 7 / tl 6）× 5 个降质档 = 45 条降质 ref（干扰池 =
  `fleurs_{th,tl,id}_tc` train 6818 行，与 val 派生 ref 无重叠，sidecar 在
  远端 `eval_cases/refs_degraded/degradations.jsonl`）。base/r8 × 6 条件 × 3 seed
  （42/43/44，cfg 1.8/20 步/禁重试）跑 `eval_cases/refrob_20261019_*.jsonl`
  （**648 次生成全部完成**，10:18–11:03，12 份合并报告各 54 条）。
  本轮代码改动：speaker_sim 改对原始干净 ref 计算（commit `2b1ab8b`）。
  **闸门结论：有可测退化**——SIM 4/5 条件显著下降且随 severity 单调（两模型同形态，
  最重 combo：base −0.040 / r8 −0.050）；CER 方向一致为正但 n=18 功效不足；
  串音类条件输出显著变短；单点坏例 r8/lp4k/id_nat_02/seed42 垫 15.16s 尾静音。
  完整数字与阶段 2/3 修订建议见 `docs/ref_robustness_baseline_20261020.md`。
- [x] 阶段 2/3（A/B 训练验收）：**2026-10-20 完成，B 不晋级，保持 r8**。
  修订：r8 配方（五语种含 ms/vi 各 17%）随 vi/ms 数据删除已不可重建，且实测
  joint_omni8 共 32307 行 **0 条 ref**（B 臂无 ref 可降）⇒ A/B 改用新三语种配比 +
  cv22 可信配对：th/tl/id 各 28.3 分 + zh 10 + en 5（按有效时长、12h 预算、seed 42），
  th=fleurs:yodas:gs2:cv22(2:3:9:3)、id=fleurs:gs2:cv22(4:10:3)、tl=fleurs、zh/en=fleurs；
  源用 r10 的 v2 排除别名，另剔 tl_short/holdout 文本（实测剔除 0 行，已断言）。
  A=`r12a_refrob_clean`（5705 行/12.01h，ref 939 行全为 cv22 th/id，3-8s 占比
  th 68.6%/id 61.6%/tl 9.8%/zh 37.1%/en 36.2%——gs2 主力语种短句占比自然过半，
  tl 依旧最短缺）；B=`r12b_refrob_deg50`（唯一变量：50% ref 降质=470 行，
  五档 86-109 均布，val 与 A 逐字节相同，sidecar `refrob_stats.json`）。
  训练：`lora_omni5_r12a_refrob_clean` / `lora_omni5_r12b_refrob_deg50`，各 357 步
  1 epoch 从官方基座新训（r64/alpha64/dropout0.05/enable_dit/proj 关/lr 1e-4），
  串行执行各带 wandb 桥接（项目 liudunxu22-b/voxcpm-finetune，连通性已实测）。
  构建 `scripts/refrob_stage2_build.py`、驱动 `scripts/refrob_stage2_train.sh`。
  **验收结果**：v2 红线全过（th/tl/id 相对 base 无退化，干净 SIM 0.921/0.920 ≥ r8 0.918）；
  但降质套件主目标未达成——B 的 SIM 退缩与 A 同量级（combo −0.059 vs −0.056），
  仅有局部内容收益（combo ΔCER 从 A 的 +0.045 压回 +0.009、时长退缩最浅）。
  **阴性结论：当前 ref 覆盖（16.5%，仅 cv22 th/id）下 ref 降质训练不修身份退缩，
  不据此判死路线但也不扩大重试**；重试前置是把可信 ref 覆盖做上去。
  全文 `docs/ref_robustness_stage2_20261020.md`；base/r8 的 v2-118×3seed 基线报告
  已沉淀，后续轮次验收只需跑新候选。
  干净回归走完整 v2（**118 case**）分语种红线（ΔCER>0.05、盲听负−胜≥2 且负≥3，
  **th/tl/id 任一语种退化即整轮不通过**）；降质套件看内容指标改善为主目标；允许牺牲部分
  speaker_sim 但全量记录，干净条件下 SIM 回退超 run 间噪声（±0.009）须写进结论。
  不通过则保持 r8、归档不改写。

### 2026-10-20：用户新增两条微调目标（登记，不插队当前阶段 1）

用户在看继续微调时补充：**① 加强对短 cue 的质量支持；② 添加情绪（控制前缀）时
音色别偏离太大。** 两条分别对应已知缺口，进入后续轮次的方案设计约束，不改动正在跑的
refrob 阶段 1；每条仍按「可复现坏例/明确增益假设 → 冻结单变量方案 → 开训」推进。

- [ ] **短 cue**：round 1 已坐实「短 cue 出分布」是微调最主要失败模式（FLEURS p50
  10-14s 训出的 LoRA 在 1-3s 线上 cue 跑飞）；r11 的 FLEURS TL 内部短句增配
  （7.98%→25.03%）实测**无明确收益**、不扩大。**2026-10-20 短 cue 门已跑完**：
  FLEURS 平行句探针（13 句 × th/tl/id × zh/en/tl ref = 117 条，全对训练集未见），
  base/r8/r12a × 3 seed 同口径——**三语种 ΔCER 区间全部跨零，无可测退化或缺口**
  （tl/id 上 r12a 的 SIM 还略高）；gs2 主力语种的 3-8s 覆盖（r12a 实测 th 68.6%/id 61.6%）
  已把 round-1 的长度出分布问题压到噪声内。**门结论：不开短 cue 训练轮**。
  边界：探针文本产出 4-5s 音频，1-3s 真实 cue 由 `process_dataset` 3s 硬下限挡在
  训练门外、结构上学不到，只能由推理侧兜（重试/长度门）；tl 短句真缺口仍待 drama_tl。
  另：本次事故一条——**删目录前必须确认无进程在写**（误删 auditions/ 中断了一个
  评测分片，已按片补跑恢复，指标无损）。又：训练完成判活**不能以 latest/ 文件存在为准**
  （官方脚本 step 0 就存一份），要看进程退出 + training_state 步数到位。
- [ ] **情绪下音色稳定**：训练侧只有 `emotion_verified=true`/`control_verified=true`
  的可信标签可进控制前缀（现状：th 有 thai_ser（CC-BY-SA，只 API 交付）；
  tl 的 filipino_emotion 仍待审；**id 为 0，不伪造**）。验收新增固定配对口径：
  同 ref/同文本/同 seed 下 control on/off 的 speaker_sim 差与声学盲听，
  控制开时 SIM 回退超 run 间噪声（±0.009）即记负；线上真实口径注意——归档生产
  60/60 `control_instruction=""`，但 DIS 成片确有过 pained/whisper 生效实例，
  验收 case 须覆盖「控制开」一侧而不是只测裸文本。与 refrob 不同变量，不合并进
  同一训练轮次。
- [ ] **超参数只按证据动（用户 2026-10-20 提出「看看要不要调超参」）**：现行默认全部
  来自官方 FAQ 或已被实验固定（lr 1e-4、r=64/alpha=64/dropout 0.05、enable_dit、
  batch2×累积8、1 epoch、training_cfg_rate=0.1、weight_decay 0.01），**不做无假设的
  网格扫描**——离线指标分辨率低于 run 间方差（同配比两轮 ms CER 差 0.042），
  28 case × 3 seed 下 ±0.04 以内全是噪声。可调杠杆与各自的触发证据：
  ① `enable_proj`（唯一还没做过单变量的结构开关，r9 是组合拳不能归因）；② 步数/epoch
  （r8 2020 步 vs r9 3130 步，更多没更好，暂无加预算证据）；③ lr/r 值（官方默认，
  无变更依据）。任何一条要动，前置是**把统计功效提上来**（每语种 30+ case × 5 seed）
  并冻结单变量方案；推理侧 cfg/步数是生产冻结值（1.6 与 28 步都已测过无统一收益），
  不属于微调超参。
- [ ] **继续降低 id/th/tl 的 WER（用户 2026-10-20 提出）**：先明确可动空间在哪——
  生产口径基线实测基座的弱点几乎全在**数字/货币**（非数字台词 base 基本全对，
  普通台词只能验「不退化」），所以降 WER 的杠杆排序：① 数字/货币形态的训练文本覆盖
  （注意 TN 分流：fil 数字已词化进模型，**th/id 数字是原样进模型的**，数据侧负责
  覆盖这两语种的数字/货币念法形态；评测仍按 `numeric` 单列，防 Whisper 归一化假象）；
  ② 短 cue 与 ref 鲁棒性两轮顺带稳住内容指标（漏词/跑飞是最大的非数字 CER 来源）；
  ③ tl 的官方 WER 基线（2.63%）差于 id（1.36%），提升空间最大但语料缺口也最大。
  红线不变：分语种 ΔCER>0.05 判退化，且先把统计功效提到每语种 30+ case × 5 seed
  再谈「降低了多少」，不接受噪声级数字当收益。
- [ ] **验收口径（用户 2026-10-20 定案，两段式）**：
  - **数据内容轮的晋级口径**：th/tl/id 各自相对 base 有可测改善（配对 ΔCER 95% 区间
    整体 <0 或 Δ<−0.05）；持平=不达标。zh/en 只观察遗忘不作门禁。
  - **口音修复轮的目标结构（MY 腔专用）**：主门 = **id 母语者听判**探针包
    （`work/audio_ab/20261020_id_prosody/`，6 文本 × base/r8/r12a/r13 盲标）；
    安全网 = 内容不大幅退化（th/tl/id 相对 base ΔCER 上限 < +0.05）+ 音色不大幅回退
    （speaker_sim 相对 base Δ ≥ −0.02，约 2 倍 run 间噪声）。**口音归母语者、内容/音色
    归指标，不用 CER 判口音，也不拿「必须更好」的卡尺挡口音修复**。
  - 按此口径回读：r12a/r12b 未达「比基座强」（区间跨零）；r13（358 步）安全网全过，
    等口音主门；r13x（同数据 3 epoch 1094 步，用户指示的步数放大）训练中。
- [ ] **id 发音地道性（「像马来西亚腔」，用户 2026-09-18 首提、2026-10-20 重申并提高优先级）**：
  **归因已修正（2026-10-20 用户确认）：MY 腔样本来自基座合成、经 id 母语者实听反馈
  ——是基座预训练自带行为，不是 r8 的 ms 训练数据导致**（r8 含 17% ms 的假设降级；
  新配比零 ms 仍然该做，但它不是修复手段，真正的杠杆是**用更多地道 id 数据把韵律往
  id 拉**——用户明确提出加大 id 微调份额）。口音验收可走 native 评审闭环（用户有
  本地 id 母语反馈渠道）。
  进展（2026-10-20）：
  ① 词表代理扫描 r12a 训练集 1852 条 id 文本：ms 标志词（ialah/bermaksud/sahaja）仅
  3 条（0.16%）且属正式印尼语可用形态，**文本侧无可见污染**（只验文字不验口音）；
  ② 已重建纯 id 韵律探针 `eval_cases/id_prosody_probe_20261020.jsonl`
  （6 条 id 诊断文本 × zh/en/tl 3 ref = 18 行，剔除归档探针的 ms/tl/en 对照行）；
  ③ 待做：短 cue 门跑完后用探针对 base/r8/r12a 同口径出音频（之后并入 r13 验收），
  产出母语评审试听包；口音以 native 评审为准，离线指标不裁定口音。
  ④ 设计变更：id 份额上调作为独立单变量轮次（候选方案 id 28.3→38.3、从 zh/en 回放扣，
  不碰 th/tl）；与 thai_ser 泰语轮分开训，不合并变量。

### 2026-09-24：CFG 1.8 / 1.6 与同角色跨句稳定性

- [x] 用户确认优化目标：优先降低明显声学坏例、保持同ref/同说话人产出一致，不追求高度复刻原片音色。
- [x] 冻结5条VI输入、3份实际参考SHA；1/3/9同ref用于跨句诊断，7/24保留原坏例参考风险。
  两档CFG、3个相邻seed、统一28步、单候选且无隐藏重试，共30次请求。不是旧take重放或五语种验收。
- [x] 补齐现有API A/B工具：保留参考处理的可选模式、明确关闭self-heal/text-regen、核实际采纳参数，
  拒绝缓存/隐藏生成，冻结请求/输出SHA；不改首轮1.8、伪影救援1.5或身份QC门。
- [x] 30/30生成完成，15对请求仅CFG不同，实际参数/GPU参考指纹通过；132份产物SHA复核通过。
  原声纹均值0.721→0.611保留；听评确认有长静音/噪音混杂，不作为纯音色稳定性结论。
- [x] 精简盲听包已交付：首批固定7对/14条（5组），其余16条折叠；无新生成、无按分数挑样本。
- [x] 9组反馈归档并揭盲：G01偏好1.8、P03偏好1.6，其余持平但包含双方异常，不晋级1.6。
  G01-B共28.020秒，其中首句原API WAV有20.847秒低电平长尾；拼接只额外加0.5秒。
- [x] 新增cue1/cue7两组开发回归，保留请求/ref/输出SHA；P组6条收为开发候选，非训练样本。
- [x] 冻结音频离线过当前DIS候选门，11/11异常均拦截；不是此门漏检，不重复加硬门或盲裁。
- [x] 修A/B报告漏项，加入真实解码停止原因观测且随已采纳take保留；8项负向失败、197项联合通过。
  该观测不改变音频/重试预算；OmniVoice `24c09dd`已push，仍未核部署，旧实验产物和数字不改写。
- [x] 按用户要求移除DIS无标签超短句自动“读完即停”控制、专用环境变量及风格复用豁免；
  正常缓存门拒绝带旧指令的take，保留情绪/重试控制和坏例QC。5项负向失败、257项联合通过，DIS `613c677`已push。
- [x] 两仓全部本地源码改动复核并推送：DIS终态3175项通过，OmniVoice全量673项及3个subtest通过。
  含CPU声纹/辅助VAD与伪影自愈等既有改动；未部署/重渲/新听评，Linux完整镜像构建仍待验证。
- [ ] 部署并核运行/权重后，cue7固定同ref/reference-only/1.8/28步，验证移除控制后的噪音/多读及停止原因；
  因果验证仍需3配对seed仅切控制，不拿旧部署baseline冒充严格对照；不跨身份借音，持续失败保留base/r8诊断。
  纯尾静音和内部多读继续分开处理；生产默认仍1.8，本轮新增TTS/ASR均0，语言未验。
  方案、结果和交付边界见 `docs/cfg_stability_20260924.md`。

### 2026-09-23：mtgix3k新版源码生效，音轨未变化

- [x] 冻结第27剧第90集/第4集新片，视频SHA与源码核验见`docs/video_review_20260923_vi_mtgix3k.md`。
  publisher及14条client_request源码SHA均匹配远端`ce01d0d`（含`b5f4fd4`，两处修复未丢失）。
  不外推全部worker、最初识别版本或GPU权重；本轮仅fetch取证，没有merge远端重构或新push。
- [x] 与`mtgg3m0`相比，仍为锁版67，14条cue时间/文字/身份及raw/segment/ref SHA全部相同；
  实际AAC包与解码PCM也完全相同。未见音色改善，不重复计为新人工反馈或独立坏例。
- [ ] 本片首两句仍有speaker_mismatch、邻句相似度0.3186且无合格共享anchor；先补可靠同人参考
  和实际阶段音频，不能靠软风格加权绕过参考/身份硬风险，也不盲目清缓存或按同输入整片重渲。
- [ ] 继续等待第96集新识别结果/新锁版进入生成，验前三句同人绑定；第90集这条链接不替代该验收。

### 2026-09-23：紧邻音色与识别A–B–A错误

- [x] 新两片冻结视频、29份客户端ref/prompt并校验SHA。`mtgg3m0`首句听似男声、1/2同人音色跳变；
  `mtggf9f`用户确认1/2/3同人，实际却为A–B–A，第二句使用另一参考。两片实际目标语均VI。
  复盘见`docs/video_review_20260923_adjacent_identity.md`；新增开发回归
  `eval_cases/production_adjacent_identity_20260923.jsonl`，不可训练、语言未验。
- [x] 两片publisher/客户端源码与DIS `7185f04`一致。第二片1/2/3身份沿用旧版；生成请求未携带
  身份审查状态和recognition_task_id，不能由成片反推哪一层识别失败或断言全部声纹复核未执行。
- [x] 最小修复已push：自动音画审查的高可信同声连续不因不同标签绕过校验；软emotion/prosody
  告警不再将最终候选身份权重200降到24。原代码新增3回归全失败，修后三组144/162/37共343通过。
  交付前16文件联跑343通过（8.09s）；DIS `b5f4fd4`已push，硬门、坏参考保护、共享4/6预算不变。
  不按A–B–A形状自动并人；第90集已核新包生效，但第96集重识别/新听评仍待确认，不能声称修复该片。
- [ ] 优先核第96集原识别episode result→锁版cue→生成manifest及首三句真实源人声；
  按人工已确认同声关系或可靠声学证据订正这一话轮，不猜真实角色名、不把全剧同名角色强并。
- [ ] 同身份再统一可信源参考；`mtgg3m0`1→2以及`mtggf9f`20→21→22均实测邻句低相似度，
  canonical被拒，无合格锚点不能靠加权/更多seed兜底。未锁定相邻救援仍待同compare_left复量。
- [ ] 第二片cue15（74.60s）已6次生成后TTS失败、源人声抑制；单独定位原始候选/后处理与最终文本，
  不把它当旧片相同沙哑take。11/25/27仍有严重/裁后风险，keep-best不算通过。
- [x] 内嵌prompt与raw SHA交叉匹配恢复第二片cue1/22、首片cue3/10原始TTS；其余raw/segment及
  GPU预处理参考仍缺失。没有新增TTS/ASR，保持r8，旧24条模型单变量诊断仍0生成。

### 2026-09-23：新增“读稿感 / AI味”反馈

- [x] 登记用户总体反馈并核查文本口语化、话轮/停顿、有效表演控制、变速与电平链路。
  系统策略见`docs/natural_dialogue_strategy_20260923.md`，不将ASR通过、没有爆音等同自然对话。
- [x] 联网核对VoxCPM2官方限制/公开坏例及微调边界，见`docs/voxcpm2_known_limits_20260923.md`。
  区分官方确认、社区报告与待验假设；未新生成或训练，VAE重建只列为可选诊断、尚未执行。
- [ ] 先定位同take的raw→segment→mix，再做文本/控制/模型单变量；新读稿片段尚未指定，
  不把沙哑case或自动低韵律提示冒充人耳读稿真值。局部实验允许推进，母语自然度保持未验证。
- [ ] 新模型/控制对照使用≥3相邻seed，控制开关两侧固定reference-only模式；
  不改旧24条沙哑计划、不全局加情绪/重试或给r8直接续epoch。

### 2026-09-23：续接局部混音与编码后峰值

- [x] DIS补实际AAC解码峰值检查，超标从原PCM有界重编码一次；保持原视频数量、视频包与字幕。
  初始/最终LUFS和峰值单独记入`encoded_audio_qc`，不覆盖旧PCM口径；正常不重抽TTS。
- [x] 局部背景平衡候选已实现，默认关闭；只在有效配音窗内补剩余背景差额，不抬人声。
  同时修复侧链开启且渐变为0时已计算背景衰减不生效；旧源码1失败21通过，相关132通过。
- [x] 两组同一泰语TTS/背景的离线盲听包已冻结，含实际ref/请求参数/SHA：
  `work/audio_ab/20260923_mix_peak/pack_v2/index.html`。原包未改，18份产物SHA，新TTS/ASR均0。
  两条电平压力测试编码峰值−0.44/−0.19 → −1.72/−1.98dBTP；不作为VI沙哑修复证据。
- [x] 已收到局部背景盲听反馈：M01/M02均“差不多”，备注为空，原文存于上述`pack_v2/feedback.json`。
  本轮未显示偏好收益，局部背景平衡保持默认关闭；持平不等于无明显异常或自然度通过，不收为正常候选。
  生产仍r8，未部署/未重渲/未训练，24条模型诊断仍0生成。详见DIS
  `docs/reviews/2026-09-23-local-mix-and-encoded-peak.md`，DIS实现提交`7185f04`。

### 2026-09-23：整体音质复查与r8下一轮路线

- [x] 9条本地成片逐条探测：44.1/48kHz、AAC约243–296kbps；4条另测响度/真峰值。
  最新VI为-14.63 LUFS/-0.05dBTP，不宜全局再加音量；局部对白遮盖仍需分轨，未认定沙哑是编码引起。
- [x] 正常A/B候选10/10补实际ref和conditioning侧录（2个去重WAV）；回归16条中的9条补到
  历史SHA匹配ref（7个去重WAV）。未找回项、无prompt、无GPU预处理音频均显式记录；不可训练。
  本轮微调仓离线回归154通过（12.75s），候选/参考关联与SHA自检通过。
- [ ] 按`docs/quality_roadmap_20260923.md`推进局部对白平衡/编码后峰值、可信声学风险救援与
  目标语文本/停顿A/B，不增加正常全片ASR或全局生成预算。
- [ ] 微调转为并行准备定向实验，不再笼统排除；原24条模型诊断仍0生成，且风险参考不等于
  模型归因。按结果决定r8 warm-start还是基座/强度对照，不直接续epoch、不热切生产模型。
- [x] 上一批push已完成：微调仓`40b8954`，DIS保留远端新停顿/并发提交后为`f30291c`，
  20相关文件824通过（10.28s）；OmniVoice仍`11302c7`，三仓远端SHA已核对，未部署。

### 2026-09-23：新片残留、系统补识别与训练备料

- [x] `mtg6jbj`已下载冻结，17个参考恢复并核SHA；22个TTS请求和publisher均匹配DIS `bd2ada7`包。
  严重风险6次策略已生效，1:16/1:53仍有用户确认沙哑，不能再以未部署解释；复盘见
  `docs/video_review_20260923_vi_mtg6jbj.md`。1:53无控制仍异常，同ref风险簇先排参考/模型。
- [x] 定位18–20s附近wailing/keepOriginal及24–27s的短句TTS延长；不是所有异常都归于识别或模型。
  DIS本地补长空隙有界扫描（4窗/每窗≤6s/阶段最多12次不变），旧4回归失败、修后相关483通过。
  开关仍默认false，未部署本轮新代码、未重跑识别；旧锁定脚本不会被重渲自动修复。
- [x] `work/for_train`建立候选池：14来源、16回归条目、428条评测记录/393去重文本排除，
  6仓元数据证据。已审训练音频0、新真人下载0；不收合成成片或本地`t_*`测试数据。
  详见`docs/training_material_pool_20260923.md`，大体积音频仍只在远端采集。
- [x] 新cue15/25独立12条base/r8诊断计划已冻结，复用旧执行器；原12条计划与旧音频不动。
- [x] A/B正常片段候选收录：新增可复用导出脚本，已保存10条/39.02s（原音频SHA不变），含文本、
  试听代号、原反馈、参数与自动风险；9条仍需自动QC复核，全部语言未验/不可训练，不因相对偏好洗白。
  正式训练音频仍0；后续每次A/B听评后按同规则收录，不给生产增加ASR/TTS调用。
- [x] DIS共享重试seed加入累计请求序号，修复拒绝救援回滚后的重复探索（含缓存恢复旧attempt）；
  首take及软风险锁seed不变，不扩大4/6生成预算，不改变拒绝较差候选的回滚规则。
- [x] 本轮交付复核：微调仓全套153通过（107.70s），DIS17个相关文件698通过（9.44s）；
  备料池自检通过，旧试听包/模型计划/冻结音频不修改；这是工程回归，不是新听评或模型验收。
- [ ] 取得原识别/声轨证据后，对有界补识别做单集灰度：短粗口恢复、真哭笑保留、脚步不生字、邻句不重复。
- [ ] 同参考跨cue反复失败仍待核实际GPU输入；新的重抽听感与生产耗时收益需部署/局部试听验证，
  不以seed变更或单测通过认定沙哑已消失。
- [ ] 独立GPU可用后先跑模型归因：原12+新增12均0生成；可靠参考下确认失败再定向微调。
  先复用远端已验train池，不直接续训r8几个epoch、不让训练材料准备冒充开训或质量改善。

### 2026-09-23：当前交付与用户部署

- [x] 已审查本地源码与测试；DIS `bd2ada7`已push到`feat/recognition-generation-mainline`，
  `ls-remote`与本地HEAD一致。包含参考源窗统一、失败参考元数据回滚、强风险共享额外2次生成及严格余量。
- [x] 本次交付前微调仓全套**152通过（26.61s）**；DIS26个相关文件**367通过（9.46s）**。
  早先541/357/70等计数属于不同轮次/集合，保留不覆盖；测试通过不等于听感改善。
- [x] OmniVoice没有本地改动或待push提交，当前`11302c7`；这轮自愈修复无需重建GPU服务镜像。
  本仓交付诊断入口、冻结计划、坏例与runs记录；ignored WAV/视频/权重不入Git。
- [x] 新片核验DIS部署范围见上：严重救援策略已执行但仍有残留；cue15 raw改变、cue25相同，
  不能把装配cache_reused当作所有候选复用旧结果。仍未核验GPU源码/权重，源码生效不等于坏例修复。
- [ ] 12条基座/r8模型诊断仍0/12，需要独立GPU执行入口；不能以本次调用方重渲冒充模型A/B。

### 2026-09-23：严重沙哑进入模型归因，不直接开训

- [x] 冻结新片`mtg3tot` 1:57反馈（重复消息只记一次）。实际目标语是VI，cue27 116.54–118.48s
  片段；已提取并核验16个参考文件。生成已报严重风险，2+2次生成后reference-only救援被4次额度阻断；
  最终ASR pass不撤销声学异常。详见`docs/video_review_20260923_vi_mtg3tot.md`。
- [x] E01用户听评已收：A沙哑杂音，B开头杂音，偏好B但两者明确异常；不再标待听、不全局去情绪。
  首句客户端ref确有前段低能量（20ms RMS首次超过−60dBFS在1.22s），但GPU已记录预处理；
  新cue27参考首次超过同门槛在0.02s，不能把两次坏例统一解释为前置空白或按弱词时间戳切头。
- [x] 准备模型单变量诊断：两条VI × 相邻3seed × base/r8 = **12条**，当前**0/12生成**。
  计划`eval_cases/roughness_model_ab_20260923.json`，入口`scripts/roughness_ab.py`；复用严格LoRA
  加载及官方开关，固定原客户端ref/文本/控制/各cue有效CFG与步数，禁重试、禁覆盖既有输出。
  已核验本地r8三文件与两个参考SHA；输入包在ignored `work/audio_ab/20260923_model_attribution/`。
  新执行器及既有加载/开关回归21项通过；DIS预算/参考救援70项通过（含短cue advisory与严重发声共存）。
- [ ] ~~获取独立Linux GPU执行入口并核验基座文件，才运行12条。~~ **2026-10-19 作废**：vi 已退出
  目标语种，这 12 条（两条 VI × 相邻 3 seed × base/r8）的唯一目的是「决定回退或定向微调」，
  不再有微调决策要支撑 ⇒ 降级为**已归档开发诊断**，不排 GPU。计划文件
  `eval_cases/roughness_model_ab_20260923.json` 与输入包 `work/audio_ab/20260923_model_attribution/`
  字节保留（0/12 生成，从未产出音频），成片复盘 `docs/video_review_20260923_vi_*.md` 照旧不改写。
  原取证纪律仍然有效并适用于其它语种：现有生产API只自报HF模型名，无权重SHA、无已发布SSH口；
  Vast命令接口明确拒绝对运行实例执行；不为取证停服务/重租/切权重；此类诊断不用API去噪/参考清理
  /DIS后处理，不冒充生产重放或完整排除参考风险。
- [ ] ~~若r8在匹配条件下更差，先验证基座回退；若两者在可靠输入下稳定失败，再准备核验真人数据的
  定向微调小实验。~~ **2026-10-19 随上一条作废**（vi 退出目标语种）。保留下来的通用纪律：
  仅两条同语种坏例不足以晋级、退役联合模型或声称整轮收益；不再一概说无需微调。

### 2026-09-23：严重坏例优先质量，局部增加自愈预算

- [x] 用户明确允许严重badcase多花少量预算。DIS共享重试入口本地改为普通默认4次、严重风险
  一次性扩到6次模型生成；服务器内抽样/客户端换seed/换参考/后段救援共用额度，预留位仍保留。
  剩余1次不再强行授予服务端2次；超时按已授额度记账，不因失败/回滚重新发额度。
- [x] 重试分流补入完整severe证据，避免首个emotion_mismatch遮住严重音色/伪影风险；复用原有
  去表演/续写救援，正常首轮不变、弱提示不扩预算、不全局关情绪，救援更差则恢复旧候选及条件。
  14项新增回归通过（含后段才发现严重风险），当轮26文件联跑**541通过（12.78s）**；
  现已随`bd2ada7`push，部署及声学收益待确认。
- [x] 完成首句情绪开关新配对`emotion-151930650565`：同一实际ref、文本、seed1988468465、
  CFG1.8/28步/预处理，仅control_instruction不同；2条各生成1次、无缓存命中、预降噪参考SHA相同。
  DIS离线页`work/audio_ab/20260923_vi_emotion/index.html`，用户已评E01，两者异常，不重复索要。
- [x] E01反馈落盘到同目录`feedback.json`，17个原冻结文件未改；偏好B不是通过。
  仅换seed的旧试跑计划暂停，先做上节模型归因；语言质量仍未验证。
- [ ] 跨仓后续：DIS继续按证据分配救援；OmniVoice优先补reference-only跨语言韵律QC证据与
  候选摘要，而不是再造重试器；微调仓启动归因准备、尚未开训。完整范围/ROI见
  `docs/quality_self_healing_20260923.md`。

### 2026-09-23：VI首句沙哑，自动识别源窗及回滚证据修复

- [x] 冻结`mtg2i2m`及用户首句沙哑反馈；从MP4校验提取13个ref/prompt。发布进程与首句客户端源码
  SHA `f76c13f…`匹配本轮修改前的DIS `d794f16`包内容，先前修复已见于此任务；不能外推全部worker或GPU权重。
- [x] 查明原canonical被低语音占比门拒绝、首句实际用回自身源窗参考（4.06s、SNR4.27dB）；
  speaker_mismatch/legacy风格/0.9倍速/limiter仍是混杂，尚不能判沙哑发生在哪一层。
- [x] DIS本地修复自动参考候选/切音/源音区互证统一源窗，不再把播放窗当源窗；失败换参考的来源与
  参考质量元数据随已采纳音频回滚。新增回归修前14失败1通过，联跑357通过，无新增远端调用。
- [x] 定位包`vi-first-mtg2i2m` R01补听完成：实际参考无明显异常、成片首句沙哑；不重复计票，
  语言质量未验证。不能用低SNR或ASR pass否认反馈，也不据此认证参考适配性或模型病因。
- [x] 首句服务端确认fearful/shouting指令生效；旧raw/segment仍缺，已完成局部新配对，见上节。
  不冒充旧take复现，不全局关情绪、不多跑全片ASR、不盲开微调；严重风险预算遵循用户新要求。
- [ ] 自动识别下一优先级：可信源窗/身份与参考质量，再核VI两词长窗及旧风格证据；复用已有整集
  ASR/VAD，不全片重复识别。新旧冻结时间轴相同，不能声称只重渲就已应用识别修复。
- [ ] 同片cue14片尾硬裁后否定词风险、cue7三词长音频仍待处理。本轮新增修复已push、待用户部署，
  情绪新配对已听、两者异常，不宣称修复成功；细节见`docs/video_review_20260923_vi_mtg2i2m.md`。

### 2026-09-22：本地改动统一审查与交付

- [x] 包含非本轮编写的GPU语言路由与两份历史周报，均审查后收口；修复故障救援跨语言/能力池、
  请求失败绕过控制面冷却两处缺陷，新增5项回归在修前均失败。DIS已push `d794f16`，
  已保留并整合远端字幕校正/编码配置改动；合并后527项回归通过。
- [x] 本仓归档TH/ID/TL/VI复盘、2条仅供开发诊断的TH case及AGENTS约定；
  OmniVoice `11302c7`无本地改动、与远端一致。音视频、完整meta、试听HTML、密钥未纳入提交。
- [ ] 补齐GPU运行源码/权重及定点音频；9月22日当轮仅源码交付，未代部署/重渲；9月23日已在新片
  核验此任务DIS包源码，范围见上节。仍保持r8。
  下文各轮“未push”仅指当时取证状态，本节交付状态优先；不改写历史数字或推断质量已改善。

### 2026-09-22：追加VI/VI/ID三片，裁后质检新漏检已修

- [x] 冻结`mtf7fu9`（实际VI）、`mtf7ft5`（VI）、`mtf7jqh`（ID）；均与对应旧片画面包/时间轴一致。
  前两片译文不变，最后ID仅122.62s缩译，已不硬裁；147.66s旧空片段恢复非空，不等同听感验收。
- [x] VI `mtf7ft5` 6.58s裁后ASR缺后半问句却pass：前部姓名/发音错字绕过精确前缀漏尾门。
  真实文本回归旧逻辑1失败5通过；改公共裁后入口为有序token尾部删除检查，仅标uncertain不裁词，
  不增加远端调用、不改音色参数。与参考窗等13文件合跑186通过，现已随`d794f16`push、未部署。
- [x] ID `mtf7jqh` 35.4s参考已取回，32k单声道46400帧逐样本全零，与旧诊断SHA一致；
  仍失败且不回原声。该输入未成功请求，request_hash_verified=false如实保留，不冒充已核验TTS参考。
- [ ] 先核ID35.4s源人声/分离与真实身份再换参考；不能关静音门或跨身份借音。
  VI6.58s的补丁只修错误认证，内容仍需核源窗/保义短表达；VI24.34s三词约5秒为可选V01定位点。
- [ ] 统一入口DIS `work/qc_review_20260922_batch/index.html`：优先仅I01、P01/P02三片段，V01选听。
  不重评已提交TH；声学与语言验收分开。详细证据见`docs/video_review_20260922_reference_batch_followup.md`。

### 2026-09-22：TL新片mtf75lr，87秒拆成两句检查

- [x] 与旧G07及用户原片画面包/时间戳一致，33条cue时间/译文不变；旧47.893s空片段恢复3.065s，
  但raw/seed已变化，不冒充同take后处理A/B。新版仍7条最终uncertain、4条severe，非空不等于通过。
- [x] 新MP4提取14个ref/prompt共4,493,274字节，37个当前使用关联hash通过；旧33条ref全部找回，
  旧prompt仅cue5/17找到，旧cue24 prompt及raw/segment仍缺。相同参数/时长不证明相同take。
- [ ] 优先试听P01/P02（85.48/87.27s）：前句仍1.684倍速，后句prompt/变速/limiter均变化且身份风险未消。
  DIS页`work/qc_review_20260922_mtf75lr/index.html`给新版混音、实际ref/prompt及旧片/原片选听，非模型A/B。
  P03漏句恢复点选听；旧51秒反馈保持不确定、不重复计票。先定位raw→segment→mix，不全局调参/微调。
- [ ] 另核cue26（93.739s）硬裁与refit拒绝：输入1.43s/预算0.968s，最终uncertain未解决。
  详见`docs/video_review_20260922_tl_mtf75lr.md`；本轮无新远端请求、训练或部署。

### 2026-09-22：ID新片mtf6spn，旧64秒参考已找回

- [x] 与旧G08及已提供原片核对画面包/时间戳一致，24条cue时间和最终译文不变。
  新片15条TTS均非空；自动风险不代表新版已有人耳异常结论。
- [x] 从新MP4提取9个参考，共2,205,374字节；15条关联均匹配当前请求及旧G08请求hash，
  包括原64秒沙哑case17。旧客户端ref现在可取回，但raw/segment、服务端实际参考及权重仍缺，
  不把开发case升级成完整可重放、不要求从已销毁容器取回同一参考。
- [ ] 优先听新版64秒I01：同参考/文本及pain/whisper控制，换seed+1009、变速由1.107降为1.022；
  仍有output_vocalization，不能因speaker_mismatch标签减少就算沙哑已修复。
  DIS页`work/qc_review_20260922_mtf6spn/index.html`包含新/旧/原片同窗和实际参考，非模型A/B；
  旧版已有反馈，不重复打分。I02/I03同角色、I04短句选听。
- [ ] 新版若仍有异常，先分raw/segment/mix；raw确认后才用已找回ref做固定输入、仅控制前缀变化的
  局部A/B。不新增全片重试，不重开微调。详见`docs/video_review_20260922_id_mtf6spn.md`。
  本轮没有新ASR/TTS或生产代码修改；边界补丁已随`d794f16`push、未部署。

### 2026-09-22：TH新片mtf4ok1，先处理58–61秒

- [x] 新旧G06画面包/时间戳、23条cue文本与时间轴一致；cue4同一返回TTS的后处理由空WAV
  恢复为1.345646s，15条TTS均有非空segment。8条显式原声均为笑声/哀叫，不是失败回原声。
- [x] 从单个新MP4提取19个ref/prompt，共3,101,706字节；22条使用关联全部核验请求hash。
  DIS本地定位页`work/qc_review_20260922_mtf4ok1/index.html`，优先T01/T02；不是模型A/B。
- [x] 收到完整7条听评，覆盖此前局部提交、不重复计票：T02成片异常/参考不确定（多人声、急促），
  T06成片杂音/参考无明显异常；其余5条两侧无明显异常。两个开发坏例已入`eval_cases/`，
  但不作训练数据；人工声学结论不清除原文本/身份QC，语言质量仍未验证。
- [x] 参考窗扩长后旧padding、VAD补边越界已本地复现并补最小护栏；旧代码新增回归2失败2通过，
  修后8个相关测试文件共123项通过，不加远端请求。边界补丁已随`d794f16`push、未部署、未做效果A/B。
- [ ] **优先T02**：重取受边界约束的真实参考，核窗内是否仍混人；
  不宣称这就是多人声的完整成因、不以speaker_count=1认证单人，参考仅1.53s不强跑不可靠声纹。
  同时追溯缺失source_text、refit未入选和硬裁，不把裁后forced pass当完整通过。
- [ ] **优先T06**：取真实任务音频包，按同take的raw→segment→mix定位杂音；meta有段级limiter及
  .946倍速，均未获因果证明。不全局关降噪/限幅、不重开微调；拿到替代方案再做局部A/B。
- [ ] T01/03/04/05/07降低声学排查优先级；未判自动QC误报，不需用户重填当前试听包。
  详见`docs/video_review_20260922_th_mtf4ok1.md`及DIS `docs/reviews/2026-09-22-th-reference-listening-feedback.md`。

### 2026-09-22：待QC参考音持久化，保持原视频数量

- [x] 按用户最新决定撤销双视频方案；保持唯一`DUBBED_VIDEO`、原文件名与五类产物契约，
  在原有一次meta封装内增加按SHA去重的ref/prompt，不另建审核副本或追加清理remux。
  参考音同步放入QC音频包、重建恢复路径；单个待QC MP4可提取并校验原字节。
  不增加TTS/ASR请求；普通带字幕仍2份业务视频、无字幕1份，对外只上传1份。
  单输出方案重新完成354项回归；真实FFmpeg确认PCM/编码包/时间戳与字幕不变，
  内存负向对照确认额外MP4会被数量断言检出，不沿用双输出草案的验收结果。
- [x] 新TH单个下载文件已验证参考音可取回且hash匹配，见上节；不据此认证所有worker完整版本。
- [ ] 确认原manual review页面仍读取原视频（本轮未操作审核页面）。
  当前这份视频带诊断/参考音；最终发布如仍需去掉，应在已有发布入口清理，尚未接入。
  相关实现前轮已push；当前单片证明归档功能已执行，整体部署/权重仍待核验。
  `generation_review_video`记录原视频体积增量及去重参考音字节数。
  详见DIS `docs/reviews/2026-09-22-review-reference-persistence.md`。

### 2026-09-22：四片沙哑反馈，先绑定时间点与处理阶段

- [x] 下载并冻结用户新四片（G05–G08），实际目标依次VI/TH/TL/ID，不能按CDN地域认语种。
  均对应DIS `bb19a51`，实际API为`58.8.189.113:58651`；本批未见上一批Whisper OOM，
  但仍有18条最终uncertain、1条fail，不认证全部QC健康。
- [x] TH22.32s、TL47.893s均已有TTS却因后处理空WAV漏句；本机按记录参数包含停顿压缩复现，
  旧链均为0样本，已有共用时间戳修复后恢复有声文件。TH仍超安全窗，不能把恢复样本当完整通过。
  不再按语种加补丁，先整合远端后部署已有修复并局部重渲；本轮未push/部署。
- [x] 用户补充沙哑时间点：第3条TL约51s、87s，第4条ID约64s；绑定G07 cue11、
  cue23尾部/cue24边界、G08 cue17，整数秒不当精确起点。三条开发反馈登记于
  `eval_cases/production_roughness_feedback_20260922.jsonl`，原ref/raw未取得，不能直接重放。
  DIS本地定位页：`work/qc_review_20260922_global/roughness_review/index.html`，8个优先候选
  加4条完整视频/全部cue跳转；其他候选不自动标坏，混音不是裸TTS、训练语料或因果A/B。
- [x] TL51s未变速、无段级limiter，不能把这批沙哑统一归因于强变速。TL87s附近参考重配、
  combined与51s的reference-only不一致；ID64s实际应用pained/whisper控制、约1.107倍速，
  变速前已有低频谱质心比，但以上均不是沙哑成因的听感证明，不据此全局关控制或换参考。
- [x] 取得用户提供的TL/ID原片（`mtcoctauzp4x8d` / `mt6bz8fpgqo313`），大小与输入记录匹配，
  编码视频流及逐包PTS/DTS/时长均与对应成片一致，三处可直接对同一时间窗，不是错片/时间轴偏移。
  DIS本地对照页`work/qc_review_20260922_global/source_comparison_20260922/index.html`：
  S01 TL51s、S02 TL87s邻句上下文、S03 ID64s，共6条对照片段+4条可选锚点原片窗口；
  保留原采样率/声道，无降噪/变速/归一化。只用于补评原片是否也沙哑，不是裸TTS或模型A/B。
- [x] 已收到`roughness-source-20260922`复听：S01选择不确定、备注两边都不沙哑，暂降优先，
  保留初评，不认证false_alarm/修复成功；S02两边都有但成片更重；S03原片无异常而成片新出现。
  当前顺序**S03 → S02**，S02仍保留cue23/24边界；语言质量未验证。
- [x] S03风格来源核查：schema v1、未手动锁定、置信字段均0、出处空，却启用了pained/whisper。
  与成片DIS版本一致的共用门控函数纯代码复现：v1保留标签，同字段v2抑制，v2手动锁定保留。
  这是旧版兼容路径的风险线索，不证明耳语就是根因；不能把schema号改大冒充修复或覆盖人工锁定。
- [ ] 通过GetEpisodeGenerationTask返回的CUE_AUDIO_BUNDLE文件描述取回同take的raw/segment，
  这批旧包原ref需另取，容器已销毁，不能再要求去旧worker打包；新版持久化不补回旧资产。
  原片已到不等于取得分离/裁切/预处理后的真实ref，不能拿含BGM源窗冒充。
  旧实现音频包在视频上传后生成，MP4无bundle标签不等于没生成。当前本机缺SPEX配置/
  调用凭据，也无可用集成，尚未发任务查询，不猜HTTP路径或S3 key。先分阶段对比再做局部单变量A/B。
  优先S03核真实ref/返回take/segment；若原take有问题且证据指向控制，首个A/B只去掉whisper短语、
  保留pained和其它有效条件，不同时换ref/CFG/steps/降噪，不增加生产调用预算；S02随后核源参考与变速。
  84/85参考触发低SNR自动降噪只作线索；
  高频刺耳检测false不排除沙哑，不全局关降噪/加低通/调CFG步数/加重试。
- [ ] 局部复核TH59.80s短句跑飞后硬裁且最终fail、TL93.739s/103.034s裁后风险。
  ID25.03s已有一次refit完成，但文本通过不解除音色风险。继续暂停微调。
  详见DIS `docs/reviews/2026-09-22-four-videos-roughness-review.md`。

### 2026-09-22：ID 漏句，先修后处理与恢复 QC 可用性

- [x] 冻结`mteq9pnflssh8b`，实际ID、157.611s、DIS源码对应`bb19a51`。
  29句中22条有最终TTS、2条失败、5条显式笑声保原声；29/29源文/译文/时间同旧N01。
- [x] 定位147.66s漏句：已有4.44s TTS，后处理时长读取失败后整句为空，不是模型未生成。
  本机真实FFmpeg用相同时间参数的合成信号复现：头裁切→预填静音→rubberband产生异常PTS，
  后续按时间裁切输出空WAV。DIS共用变速入口加样本时钟校准，正片/试听一并覆盖，
  不换音色参数、不加ASR/TTS/LLM调用。旧代码2失败2通过，修后4通过，相关241项通过。
- [ ] 整合远端后发布此修复并局部重渲cue28；原raw若仍可取回，优先只重做后处理与最终QC。
  当前未取得原take、未制作语音A/B、未push/部署，不能宣称线上漏句已恢复。
- [ ] 核查当前API实例的Whisper显存/驻留模型/并发：本片18条词对齐明确报CUDA OOM。
  另7条最终QC复核不可用，尚未逐条证明同因；不默认改CPU、不加重试或重启生产服务。
- [ ] 修35.40s的源参考输入：与旧片同一SHA的全静音参考仍被正确拒绝；
  不关闭静音门、不按同名借音。122.62s金额句仍硬裁且refit因length_bounds拒绝，需局部核词尾。
  继续暂停微调；详见DIS `docs/reviews/2026-09-22-id-empty-segment-followup.md`。

### 2026-09-22：VI 短窗缺词，先核源边界与压缩预算

- [x] 冻结 `mteo0qamnd3558`，实际 VI、110.466s，DIS 源码指纹匹配 `3f3572d`。
  51.88s / 103.64s 裁后转写缺目标后缀，新保护已从 pass 改为 uncertain，**不代表内容已恢复**。
  81.04s 长尾现已收容；15条有TTS、5条显式笑声保原声，仍有身份/声学风险。
- [x] 同版本本地复现 refit 预算冲突：cue7 可容4单位、保守下限5；cue18 可容8、下限10。
  虽记录 eligible，实际在 LLM 请求前跳过，不是已完成缩译。
  DIS 已本地补 `same_language_rewrite` 原因/预算/未尝试标记，不改声音、不加请求；
  新2回归旧源码失败，相关145项通过，尚未push/部署。
- [ ] 取源人声与原始脚本核对两处短窗；文字/边界与旧成片20/20相同，不先断言源窗一定错。
  原窗正确再做两条局部保义短表达或节奏/候选择优对照，不放宽全局裁尾/文本保护、不恢复原声。
- [ ] 核查 cue7/14/20 的参考与身份；需改变音色时用原 take/ref 做 DIS `work/` HTML A/B。
  当前原 take 未在本机，未重合成或训练；详情见 DIS `docs/reviews/2026-09-22-vi-boundary-refit-followup.md`。
- [x] 补查第27剧第90集 `mtepajbwym0x49`：源码摘要匹配远端新提交 `bb19a51`，
  但14/14句的源文/译文/时间仍同旧片。6.58s裁后缺第二问句；74.44s片尾有否定词风险、
  最终ASR复核不可用，均未认证完整。两句实测缩译预算12/14低于估算下限18/19，collector未入选。
- [ ] 对该集先核cue2/14源窗与保义短表达，查最终ASR复核日志；核源资产后再改cue13人名长窗。
  cue13 raw/ref/segment与旧片相同，不能靠整集重跑修锁定脚本；cue4/5/6参考风险另行处理。
  本轮仅补复盘、未新增源码/推送/部署；原有未提交诊断补丁保留，远端只fetch未合并。
  详见 DIS `docs/reviews/2026-09-22-vi-episode90-followup.md`，继续暂停微调。

### 2026-09-22：TH 70 秒慢感，先做局部节奏对照

- [x] 冻结 `mtenissziolfbe`，实际目标 TH；用户反馈 70s 慢，对应 cue13（70.68–79.12s）。
  API 返回 8.925s → segment 8.86s、tempo=1；不是 DIS 为填窗减速。
  首轮 6.46s 因模型长度门 too_short 被退，重试的 seed/CFG/steps/守卫开关同时变化，
  不能把两条历史候选当单变量对照，也不能仅凭 ASR pass 断言误拒。
- [x] 成片 publisher 与14条实际请求的 DIS Python 源码摘要均匹配 `3f3572d`；
  cue8 已按文件终点 containment。仅确认本片相关进程的源码，不外推全量部署或权重身份。
  实际 API 为 `115.75.223.236:54386`，未据此改默认地址；保持指定 r8、不启动训练。
- [ ] 取回 cue13 已采纳 raw/segment，只对同一 take 做约1.09倍速的轻量节奏 A/B，
  复用现有拟合能力、不全片重合成，试听 HTML 放 DIS `work/`。当前无音频包标签、原 take 未在本机，
  尚未产出 A/B；不拿成片混音代替。人工通过后再决定是否收窄表演保护，不全局提速。
- [ ] 另取首轮候选核验长度门临界误拒；重试关闭守卫后的空 verdict 不算同口径通过，
  不全局取消短句保护、不增加每句 ASR/重试预算。
  完整证据与 request_id/音频 SHA 见 DIS `docs/reviews/2026-09-22-th-cadence-feedback.md`。

### 2026-09-20：三片新反馈，按共享策略修复，不按 cue 特判

- [x] 冻结 `mtcc8ms/mtcc0wu/mtcciyf`，实际目标分别 VI/TH/ID；用户68秒疑似坏例对应ID11。
  先查参考可用性→完整候选风险→时间拟合→最终音频QC，系统方案及证据见
  `docs/video_review_20260920_mtcc_system_policy.md`，不直接启动微调。
- [x] DIS共享评分归一化所有已知问题标签，并独立计算真实runaway风险；
  原阈值/权重/生成预算不变，无语种/句子/序号特判。
  新6回归旧源码失败，修后相关270项通过；短候选仍有声学风险，不冒充听评改善。
- [x] 生成音频诊断/裁边/salvage/containment/句首/孤立尾音/chunk共用可信词边界验证；
  弱证据不认证false_alarm/aligned，原坐标/词/警告保留。既有独立ASR佐证例外、源SRT策略不改。
  旧风险裁切缓存窄失效、句首旧判定重验；正常缓存保留，实际DIS应用后524项回归通过。
- [x] 分块前检查当前后端强制对齐能力，不支持的目标直接逐cue，避免白跑chunk后再回退。
  不加全片ASR/TTS预算；相对旧成功分块的逐cue耗时仍待实测，不承诺全片等时。
- [x] 本轮DIS已普通推送至`a7a0240`，快进保留远端`2f000ad`；
  并行Drive线程隔离/逐文件计时独立提交`62633f6`，整合后585项回归通过。
  OmniVoice无待推送修改，`7f678b4`与远端master一致；未部署/新合成，保持r8。
- [ ] 部署后用既有候选核验新排序；优先取已有`CUE_AUDIO_BUNDLE`及任务`result.refit`，
  不拿最终混音当原take A/B，也不以导出时刷新的QC重算冒充历史初选分数。
- [ ] 补齐共享时间处理规则：纯静音尾、额外发声、真实正文超窗分流；
  原take未取得前不放宽尾裁、不撤销可信对齐保护，低可信对齐不能认证false_alarm。
- [ ] 统一最终产物验收顺序，仅对变过的音频复核；保留refit拒绝/残留和best-effort风险。
  收集器可识别VI7/18与ID23，不等于线上执行成功，不新增第二套refit循环或全片双ASR。
- [ ] 输入不可用只走可信同身份参考救援/源切片修正，不对全静音反复换seed或回原声。
  原因分流与预算约束作为跨语言回归不变量；验证正确条件下裸TTS仍稳定失败后再考虑模型对照。

### 2026-09-20：调用方修复落地（不增加正常路径模型请求）

- [x] 源身份/稀疏窗风险不会因文本或take成功而丢标；旧风险bool独立阻止干净缓存，任务warning可见。
- [x] 最终文本失败才复核最多一条早期已生成候选；复用评分/后处理，不加TTS预算或放宽长度门。
- [x] salvage与edge hygiene共享可信对齐校验、保留原WAV/QC、更新裁后坐标并失效旧指标；
  原severe不改成通过，裁后最终QC不可省略。正常cue零新增模型请求。
- [x] 既有单轮refit后再封装一次字幕，使用最终cues；任务审计记录跳过/失败/完成/残留，
  成功发布时透传`result.refit`，重渲失败走既有task error。
  新21回归旧代码全失败，修后扩展531通过；未部署/新听评，详见DIS对应复盘。
- [x] DIS已推送：功能`e5377d6`合并远端识别修复`2ba92b7`，最终`07d3c45`；
  合并后551项回归通过，无强推。仅推送代码，未执行部署、训练或新合成。
- [ ] 部署后局部重放TH cue3/8/11、VN cue12/13；取得真实raw/segment/ref后按需要导出本地A/B。
  不把旧混音当原take对照；VN源时间窗仍在新脚本版本修正，保持旧锁定版本证据。

### 2026-09-20：TH新成片 mtc8im7（ID CDN，实际目标th）

- [x] 15条脚本与旧`mtc4l3g`一致；cue3已有最佳TTS且原内容fail保留，
  不回原声策略已出现在新产物，最终修复有换seed。详见`docs/video_review_20260920_th_mtc8im7.md`。
- [x] cue10成片局部Qwen/Whisper无提示对照完成，仅2次请求；两者均th、均有词形偏差，
  词对齐不完整，不用于裁音频或宣称语言通过。没有新TTS/训练/部署。
- [ ] 取已有cue3候选核验“初始内容转写正确但长度门拒绝→较短候选最终critical fail”的选择链；
  不直接关闭长度门、按ASR改台词或增加正常重试预算。
- [ ] cue8/11原take18.8/17.84秒，salvage后0.94/0.8秒；检查裁前裁后而非沿用原take的QC。
  同步核验cue6脏参考、cue10内容与refit。稳定裸TTS问题再冻结base/r8对照，保持r8。

### 2026-09-20：VN新成片 mtc7xi1

- [x] 完成与审核27/90旧片逐cue对比：14条源译文/时间窗/speaker未变；
  cue2不再硬裁，cue13已出现稀疏窗保护，但8.16秒源窗未修。
  详见`docs/video_review_20260920_vn_mtc7xi1.md`，不把保护标记等同完整部署身份或听评改善。
- [x] 5次局部ASR诊断完成并冻结；新片64–74.4秒无可用转写，不冒充静音或正确发音证据。
  源29–33/56–60秒未支持既有`Thank you.`，尚不足自动删cue；不扩同片ASR网格。
- [ ] 优先新脚本版本校正cue13，并取已有raw/segment/voice-only核对生成与混音；
  核验cue8/12源文和参考身份，cue4/14仍需缩译闭环。
- [x] 修复源身份风险的复核标记被候选/文本成功清理掉：新片cue12风险bool仍true但reason空，
  共享修复与缓存/任务warning回归已完成；不新增合成或ASR调用，未部署。保持r8，不启动微调。

### 2026-09-20：地区短剧来源登记（储备，不立即开训）

- [x] 记录用户提供的四条YouTube原始链接及视频ID，候选语言id/vi/th/tl；
  统一保存在`docs/corpus_sourcing.md`，不复制为训练清单。实际语言、真人来源和授权待核验，无MS新增来源。
- [ ] 后续确需微调时先核验授权/音轨与单人对白质量，再在远端小量取样、冻结划分并审计同人ref；
  合成语音不补量、同源不跨split，不自动扩大下载或加训。当前保持r8。

### 2026-09-20：VN审核27/90短文本长窗口

- [x] 用户导出的审核JSON、源片、成片/meta已冻结；episode90为第4集、revision67/version8已锁定。
  不再等待登录；见`docs/video_review_20260920_vn_27_90.md`。
- [x] 第13条两词但8.16秒是脚本窗，裸TTS1.05秒/最终1.072494秒，无补静音。
  单次无提示源ASR指向人名在72.96–73.28秒，meta配音约65.204秒起；不把同音异字转写当严格对齐真值。
- [x] DIS本地共享保护：稀疏源文异常长ASR跨度不认证、可走已有SRT局部对齐；
  渲染/试听不强行减速填空白，粗源边界只报待核验，原数值/身份风险保留。
  新断言旧源码10失败/11通过，修后21通过；扩展342通过，另3项旧fixture失败复现，不修无关测试。
  补丁应用实际DIS目录后159项定向通过；功能`79b7af4`保留远端6条提交合并为`3dc9477`，
  合并后343项相关回归通过，已push；未部署，之前最佳TTS策略也一并提交。
  正常路径无新增模型请求；异常原本误判adjusted的SRT条目可能多一次局部对齐。
- [ ] 部署后核验代码身份，在新脚本版本校正并仅回放第13条；保留version8原证据。
  校正应基于可信词级边界，下一cue安全窗容纳自然TTS，不把约1秒越南语硬压进0.32秒源词窗。
- [ ] 核对源语en配置与源片局部ASR zh冲突、初始识别缓存/编辑历史；
  优先第2条硬裁/refit及4/5/6/7/13身份/弱参考，取回已有take/ref包，不全片盲重识别。
- [ ] 正确文本、时间、角色、ref冻结后仍有稳定裸TTS坏例再做base/r8对照；保持r8，不开训练。

### 2026-09-20：最佳TTS策略、TH/MS反馈与审核页27/90

- [x] TH `mtc4l3g7puyr55` cue3已翻译，23.519秒原声来自最终双ASR失败后的混音回退；
  “修复”请求同seed/CFG/steps、generation_attempts=0，确认是缓存回放而非新生成。
- [x] 按用户最新裁定在DIS本地改为保留最佳可用TTS并明确风险，不自动回退原声。
  QC原fail保留、风险take不续写/干净缓存，无可用音频仍标失败；显式笑声保留不变。
  同时修复有证据的最终文本重试换seed、无音频/静音候选的择优回滚，不新增重试预算。
  DIS相关458项通过，旧源码快照15失败/87通过；已随`3dc9477`推送，未部署。
- [x] 冻结新MS片 `ph-...mtc4nbj347wj6c`：目标ms，13条合成+1条笑声，零tts_failed；
  参考混声/音乐、身份/非词发声、晚结束仍需核对，不能把ASR pass当语言或同步通过。
  见`docs/video_review_20260920_th_ms.md`，保持r8，不重开GPU。
- [ ] 核验部署后仅重放原TH cue3，确认真实新seed/生成次数与最终采用TTS；本轮未部署/重渲。
- [ ] MS优先取cue7/8/12/14同take/ref及源声，先查参考选择和识别边界，再查裸TTS；
  对cue1/2/11/13查源节奏/译文时长，不全局加步数、强裁或增加ASR调用。
- [x] 审核页27/90：此前数据接口401；用户随后导出数据，已取回独立源片/成片并诊断，
  重点为第13条源窗和第2条硬裁，后续与验收边界见上节，不能套用另一视频结论。
- [ ] 仅当正确文本/身份/ref/后处理固定后裸TTS仍稳定失败，做同条件base/r8小对照；
  有明确数据增益假设再考虑训练，不把当前暂不微调说成模型已经全面达标。

### 2026-09-20：Whisper GPU 故障与泰语成片

- [x] 冻结 TH `mtbwhene6sjnb1` 成片/meta；cue10/12 有最终硬裁，且显式 CPU
  Whisper 转写遗漏尾部内容，仍只是诊断，不是母语听评。见 `docs/video_review_20260920_th.md`。
- [x] 指定服务 Whisper GPU 请求 `41dd386a` 返回 invalid-device / 502，
  故障出在延迟迭代的 encode；停止原计划对照，未完成分语种 ASR A/B。
  两条 CPU 诊断成功不代表 GPU 恢复，未改生产默认。
- [x] OmniVoice 共享 ASR 门禁覆盖转写/词对齐/合成 QC；等待在途线程结束后再释放，
  CUDA 故障剔除匹配缓存、尾部补转不吞 CUDA 错误。无同请求重试、无默认 CPU 回退，
  无依赖降级、无部署/重启；正常请求不增加模型调用数。
  新增15项旧源码12失败/3通过，修后通过；全量530通过，另5项缺fixture在旧HEAD同样失败。
  OmniVoice已推送`7f678b4`，应用实际仓库后150项相关回归通过；push不代表实例更新。
- [ ] 部署后核验源码/依赖/设备，复用一条短WAV分文本与词时间戳各测一次GPU；
  首个失败即停，不拿health或CPU成功当恢复。当前修复是生命周期保护，
  尚未证明底层CUDA故障消失；服务端说明见OmniVoice `docs/whisper_runtime_20260920.md`。
- [ ] 取回TH cue10/12原始take与refit任务日志，核对既有refit执行/验收链；
  检测器本机重放已能识别这两条，不能再声称没有refit消费者。
  不靠更大变速/裁切、增加全局重试或微调掩盖问题。

### 2026-09-20：ASR QC 分语种选型

- [x] 核对实际链路：OmniVoice合成文本QC默认Qwen3，DIS可疑最终片段显式换后端复核；
  需要词时间戳时另按aligner支持情况回退Whisper。本项目离线评测仍为large-v3。
  当前没有按语言准确率选择首选；端点名或`model=large-v3`不能证明使用了Whisper。
- [x] 复核已有泰语10对转写/5组台词：用户听到的05-A额外尾音被Qwen记录、
  Whisper未转出；不能据此认定Qwen全面更准，也不支持立即将TH全量切Whisper。
  原音频/报告不改，白名单与SHA归档在`checkpoints/asr_qc_policy_20260920/`。
- [x] 两仓路由/决策107项回归通过；策略审计阶段无新增ASR/TTS、训练或部署，
  默认后端与r8不变。分析及官方基准边界见`docs/asr_qc_routing_20260920.md`。
  后续TH成片诊断调用及GPU失败见上节，不把中止的对照计为完成。
- [ ] 复用现有WAV与`asr_backend`请求覆盖，优先TH/TL做无期望文本提示的同音频对照；
  分开内容识别与auto串语种审计，不把旧Whisper评分或更低CER当发音真值。
- [ ] 在可核验服务测分语种误拒/漏检、二次复核率和QC总耗时；
  有证据胜出再收口分语种首选/请求覆盖/有效回显/缓存，不固定让每条双跑，
  不为这次选型先新开训练机或安装第三种ASR。

### 2026-09-20新增ID/VN：先收敛参考与失败闭环

- [x] 用户要求推送：DIS修复与两轮回归已推至现有分支，提交`aebf175`；
  OmniVoice无本轮改动，`d984c34`已与远端同步。本项目复盘/待办随本次文档提交归档，
  不含音频、权重、密钥和既有`docs/reports/`；push不等于部署。
- [x] 冻结ID `mtbup5vzqs5d4a`、VN `mtbv0gkefabl3c`原片/meta；
  ID末句有译文，是超读后双ASR失败回退；VN14条全TTS但4条severe、7条待复核。
  详见 `docs/video_review_20260920_id_vn.md`，不把自动标签当人工/母语验收。
- [x] DIS修复四个可复现缺口：尾部超读分类漏罚、控制指令半短语截断、
  锚点缓存早于源文/身份/安全窗载入、身份冲突后弱参考救援借回同speaker。
  新回归旧源码6失败/2对照通过，修后通过；最终整合446项通过，实际DIS目录84项复核通过，
  冻结ID末句旧评分复现，超读项仅减少400分。
  不加模型请求/重试预算，不改r8和阈值；本地修复未部署，原有改动保留。
- [ ] 核验部署后身份/实际参考与最终采样参数，取回现有任务音频包、原ref/原声；
  优先ID37秒/末句、VN14.68–19.34秒/65.16秒同take裸输出与segment，不任意换ref伪复现。
- [ ] 核对VN音频源语en与中文source_text的来源，保留真实英语源支持，不凭文本强改zh。
  VN开头及24秒先复核粗源边界/节奏；无硬裁不等于自然。
- [ ] ID41.8秒、52.98秒缩译/refit尚未闭环且有硬裁，先补原声/take证据，
  不用更大加速/更激进裁切掩盖；77秒、91秒另有自动性别漂移风险待听。
- [ ] 只有链路修复后同条件仍稳定复现、多个独立文本指向模型时才立新训练假设；
  目前不启动微调，不把等待原始音频当理由让GPU空转。

### 2026-09-20成片反馈：1:50同角色音色稍有不同

- [x] 冻结新视频/meta及用户反馈，定位为cue22内部；同角色cue19/22/23的ID/ref签名相同，
  但cue22重试改变seed、CFG、步数后带 `output_vocalization` 风险出片。不是单变量对照。
- [x] CPU复现检查缺口：长句severe提前返回，cue22缺 `local_pitch`，角色音高报告漏掉它；
  不将服务器F0估计或声纹相似度当成人耳音色通过。实际线上查询模型地址为微调仓库，
  meta中的基座名称只是客户端配置；渲染时权重SHA仍未知。
- [x] 用户反馈与混音上下文留在 `checkpoints/video_review_20260920/`，
  新增开发回归文本；未重合成、未训练、未部署。详情见 `docs/video_review_20260920.md`。
- [x] 复用既有本地音区检查，补齐keep-best评分/快照与缓存take后处理入口；
  不取消发声门、不增加远端确认/重试次数，不变更阈值。本机8秒音频基准：
  新候选且参考已测约149ms、两者均未测约296ms、已测结果复用约0.007ms；
  不是生产E2E耗时承诺，识别流程不改。DIS本地修复，尚未部署。
  最终新增回归旧源码6失败/1对照通过，修后7项全过；扩展254项通过，
  应用实际仓库后17项复核通过。修复已随DIS `aebf175`推送，既有未跟踪文档不动。
- [ ] 取回cue22两份原始候选及cue19/23、实际ref/segment；优先复听已有产物，
  需要新合成时先固定其余条件只改seed，再单独比较采样档位。
- [x] cue7静音参考路径增加共用请求前保护：只拒绝空/全声道峰值≤−120dBFS的实际音频，
  reference/prompt-only均覆盖，按内容缓存、不把弱人声或反相双声道当静音；
  现有失败流程留痕，不为静音换seed消耗GPU。原始cue7参考未取得，不宣称已重现其全部根因。
- [ ] 已定位既有 `CUE_AUDIO_BUNDLE` 可取采纳raw/segment；本机没有该任务包，
  也无已配置SPEX只读查询，缺音频包描述符/原始ref；不能猜workspace下载URL或拿混音替代。
- [ ] 长cue拆分需解决缓存模式跳过及
  7句超默认上限，缺原声边界时不机械拆句。先修链路，不立即重启微调。

### 2026-09-18新增反馈：SRT串语种与ID像MY的韵律

- [x] DIS本地增加EN/ID/MS/TL/VI/PT的SRT对比复核与引文验证；歧义不免翻译，
  音频源语不再代替字幕文本源语。PH与MY兼容别名仍归一到tl/ms。
  英语源/目标别名及ASR/TN配置已补齐，未新增第六个训练目标。
- [x] ID既有本地化审稿补印尼语句法断句与ID/MS语体对比，不加共享词黑名单；
  用户已切r8，本轮不改权重、CFG/步数或参考音选择。代码推送不代表服务部署。
- [x] 本轮代码已按用户要求push：DIS `900d243`、OmniVoice `d984c34`。
  相关回归415/58项通过，DIS另2项既有规范失败；未部署、未调用真实LLM/GPU、未启动微调。
- [x] 新建 `eval_cases/id_ms_prosody_probe.jsonl`：10条待验证诊断文本，
  包含共享短句、ID否定/从句/问句、MS/TL/EN对照；不是已复现坏例或验收集。
- [ ] 冻结实际ID投诉的模型/ref/最终文本与裸输出，区分译文、参考模式、后处理与模型韵律。
  SRT识别结果不会自动把TTS目标id改成ms；ASR检测为ms也不能证明口音是ms。
- [ ] 先同文本/ref/seed诊断；若链路已排除、多个独立ID句仍稳定出现同类问题，
  再做ID数据质量或份额单变量LoRA小实验，保留全三语种与克隆回归，不盲目加epoch。
  诊断显式传同一 `--ref-audio`，至少3 seed，稳定性按既有5 seed口径；
  缺drama/母语评审不阻塞离线实验，但口音收益未评不得声称改善或自动晋级。
- 具体链路证据与检查顺序：DIS仓
  `docs/reviews/2026-09-18-language-detection-and-id-prosody.md`。

### 已完成实验与运行边界

- [x] **2026-09-18暂停GPU实验，不继续盲目加训**：r10/r11没有明确晋级收益，
  数字125条及ASR复核已完成，保持r8/strength1/CFG1.8/20步。不是质量全面通过，
  也不否定后续微调；有新的可复现坏例或可追溯数据与明确增益假设时，
  先冻结独立回归和单变量方案，再开GPU。无需为等待线上身份核验保留空闲训练机。
- [x] **关机前关键权重备份完成**：r8及r10/r11 A/B共5份latest LoRA、
  配置与步数状态已备份本机`checkpoints/gpu_pause_20260918/weights/`，15文件SHA全部通过；
  r8与数字实验冻结权重指纹一致。GPU无计算任务，未代用户执行关机。
  校验见同目录上层`verification.json`；训练数据、全量历史评测音频及优化器状态未完整备份，
  **可以关机，但不要据此释放实例或删除数据盘**。权重/音频不入Git。
- [x] **两侧代码已push**：OmniVoice `master`→`88d9dd2`；
  DIS `feat/recognition-generation-mainline`→`be6e735`，保留本轮两次修复并合并
  远端`25592cd`，无强推，既有`docs/reports/`未动。合并后69项相关测试通过；
  本项目本次定向3项及全套151项测试通过。
  push不等于部署；OmniVoice既有镜像构建不代表实例已加载新代码或r8。
- [x] 2026-09-18执行29条数字输入审计：实际DIS payload再经过服务端normalizer，
  true/false开关下29/29目标文本相同，不能用只切开关做A/B；四语种已在上游词化，
  `spoken_text_normalized=true`当前仅TL/fil设置。
- [x] 已修复两侧`RM1,200 ringgit`额外添加币种词的问题，覆盖四语种的同类货币前缀。
  服务端缓存版本与DIS分语种文本签名同步更新；两侧先失败后通过，扩展138/247项测试通过。
  提交DIS `1a1e309`、OmniVoice `ac1e51d`已随后续修复push；
  不改数值/时钟规则、不部署服务、不宣称GPU发音已改善。
- [x] **r8数字输入125条小诊断已完成（2026-09-18 11:05:09）**。8条数字＋4条固定随机对照×5 seed×raw/words，
  另5条保留修前MS输入；CFG1.8/20步/同ref，单worker、禁重试。
  20/20非数字控制WAV与ASR相同；共同CER raw→words：TH .053480→.046337、
  VI .421026→.049487、ID 0→0、MS .172332→.302217。每语种仅2独立数字文本，
  不能直接比较两臂原始CER，也不能仅据MS分数判定退化。详见`docs/numeric_probe_20260918.md`。
- [x] **MS币种ASR敏感性复核16/16完成（11:08:34）**：复用8条WAV，
  两组零温度，仅指定MS/auto不同；5/8转写变化，其中2条RM→Rp。
  不能据auto检测ID认定串语种，也不能确认实际币种读错。原分数与全局ASR默认不改。
- [x] **编码故障已恢复并修复**：原任务60条后默认ASCII读泰文JSON失败，原60条与报告保留，
  只恢复余65条。两脚本7处JSON读取显式UTF-8，回归先失败后通过；
  本机相关28项、远端全套143项通过。125WAV已备份本机并验SHA；
  冻结源码、失败证据、恢复/收尾校验均归档，不改旧plan或重跑已完成任务。
- [x] **未知币种/数字分隔符纯文本修复完成（2026-09-18）**：两侧共享词化入口不再局部展开
  连续数字串；当前已识别标记在目标语种不受支持时保留原样，不把Rp映射成RM。
  合法币种空白与重复名称仍正常处理，币种后带符号不猜读法；不是完整ISO币种/时钟解析器。
  两侧各21项回归，旧代码各19失败/2通过，修后全过；扩展DIS 198项、OmniVoice 254项通过。
  29条实际payload复核仅`id_digit_3`改变：`06.15`不再拆成两段词化，未宣称时钟发音已正确。
  提交DIS `9a961b2`、OmniVoice `88d9dd2`已push、未部署；缓存/文本签名已递增并实测失效。
  旧冻结评分器、报告和125WAV哈希不变；没有重评分或新GPU任务。详见`docs/numeric_probe_20260918.md`。
- [ ] **下一步：有可核验服务时做工程上线核对**，先确认部署/权重、最终文本与采样参数、缓存版本，
  再决定最小真实调用验证。没有新坏例/数据收益证据时不扩大网格、不加训；
  当前保持r8与既有CFG/步数，不全局切词化，语言质量仍未验证。
- [x] 2026-09-18补齐`docs/runs.md`的r10/r11：配置/清单指纹、来源许可/时长/条数、
  配比/超参/实际步数/损失、逐语种CER/WER/疑似漏尾和不晋级结论；旧轮次历史数字保留。
  已随本轮代码/配置/测试/评测case归档push到`origin/main`：`4d8e092`，不包含密钥、权重或音频。
- [x] OmniVoice最终采样参数追踪已push：`30211d9`，保留远端已有ASR提交，相关66项测试通过。
  push触发既有镜像构建，不代表服务部署或GPU质量验收；DIS无本轮改动，不提交其既有`docs/reports/`。
- [x] 随后完成`number_verbalization` HTTP与prompt缓存全路径透传：OmniVoice `26829e1`已push。
  请求覆盖值进入目标/续写normalizer、prompt缓存key及最终take追踪；结果缓存版本8→9，
  不继续复用修复前忽略开关的结果。省略/null及上游已归一化语义不变，四语种17项定向、
  扩展223项测试通过；真实模型发音/线上质量未测，未部署、未追加训练。
- [ ] 有可核验服务时验证实际部署/权重与最终参数记录；数字对照先确认DIS是否已词化，
  不把仅传开关当端到端A/B。没有新坏例/数据收益证据时，仍不扩大r10/r11或加epoch。

### 2026-09-18：r11完整990条已结束，无明确收益，不扩训

- [x] 对照r8实际训练清单与现有候选池，按语种统计3–8秒的**时长占比**、独立录音及曝光上限；
  核对训练文本与raw清单的原音频关联。原文一致不等于音频/文字已正确对齐。
- [x] 固定随机抽查TL/MS候选各三个时长层，每层5条，用现有large-v3检查30条录音的内容；
  不按ASR单分数自动删样本或改转写，不扩大TTS/音色指标扫描。
- [x] r11已冻结并后台启动：**只提高FLEURS TL内部短句抽样**，其余来源记录逐条相同，
  不改ref/投影。A/B各约12.025h、5164/5282条、322步；TL短句时长份额7.98%→25.03%。
  从同一基座新训，不续训r8；固定更新预算、名义epoch约.998/.975，
  不冒称实际消费小时数或样本顺序完全相同，单训练seed的pilot不证明纯时长因果效应。
- [x] 已冻结14条新的较短TL test文本×三种ref，另选五语种15句冒烟对照；
  5 seed、CFG1.8/20步、禁重试，计划990次正式pilot生成。
  14条独立文本不能按42个ref条件扩大样本量，也不代表已充分覆盖1–3秒对白。
- [x] A/B均完成322步，checkpoint状态与完成标记一致；不重复训练、不续训r8。
  val loss/total首末：A 1.015644→0.877042，B 1.018281→0.883851，不据此判断合成收益。
- [x] 首条评测SIGSEGV已复现定位至旧Numba缓存下的F0计算；同WAV换新缓存冷/热两次通过。
  旧失败状态、日志、缓存保留；独立eval-only恢复已启动，先确认一路出结果再开第二路，
  **09:27已确认两路均产出完整单条指标**，不是仅启动进程或990条已经完成。
  每job隔离缓存，990条口径不变；进度看 `short_cue_ab_20260918/eval_recovery_20260918/`，
  原目录 `status.json` 的failed是历史故障，不能误判成当前进度。
- [x] 10:02按用户要求提前并行150条冒烟：现有两路短句不重启，新增一路先跑A的75条，
  空出名额后跑B的75条，总并发上限3。新调度器 `scripts/short_cue_overlap.py` 复用原worker，
  原控制器暂停防重复调度；冻结输入/源码/990条预算不改，记录见 `overlap_schedule.json`。
  不要恢复旧控制器或重跑原run；当前进度仍看同一份status与各job进度文件。
- [x] **10:20:18完整990条生成/评分结束**（短句840+冒烟150），六报告全部校验，两个控制器均已退出；
  原控制器不得恢复，原训练/评测不得重跑。完成标记不等于质量通过。
- [x] 已按14条源文本聚类比较A/B/base/r8，而非把42个ref条件当独立文本。
  B−A CER −0.000472，95%区间[−0.003751,+0.002768]；时长+0.104381s，
  区间[+0.060190,+0.144762]。**没有明确短句增配收益**，不扩大25%剂量、不加epoch、
  不追加候选全量评测；保持r8。A/B相对base尾静音p90均0.20→0.36s，保留门禁风险。
- [x] 五语种冒烟与原base/r8同条件各75条精确配对；每语种只有3独立句，
  B的TL CER 0.0363（A 0.0217/r8 0.0156）。未越ΔCER>.05不能说无退化，
  MS单条疑似漏尾也不是已确认截断；口音/自然度/情绪及本轮声学盲听均未验证。
- [x] 复用12条WAV完成24次ASR语言敏感性诊断，两组均零温度，仅auto/指定TL不同。
  A的异常中文转写CER .5893→0、B .7857→.1071；不能据中文转写判模型串语种，
  也不能以指定TL低分冒充发音正确。不修改原报告、全局ASR默认或删除异常样本。
- [ ] 后续只在有新的可追溯短对白文本或稳定声学坏例证据时，冻结独立回归与下一轮单变量实验。
  r8在这14条较短TL朗读文本的CER也略高于base，已记录局部风险，不拿平均收益掩盖；
  暂不扩大ASR/WavLM扫描、不开下一轮训练。缺drama/母语评审不是暂停原因，当前缺的是收益证据。
- 数据与执行细节见 `docs/r11_short_cue_20260918.md`，入口 `scripts/short_cue_ab.py`；
  已完成的恢复入口为 `scripts/short_cue_eval.py`，**不要重跑**；配对统计为`scripts/short_cue_results.py`，
  结果在恢复目录`paired_results.json`，ASR诊断在`asr_language_probe/`，均已备份到本机。
  r8的TL短句时长份额仅9.40%，已有TL短句库存只够小预算，不按3×以上重复强行扩训。
- 推理身份/在线参数追踪降为独立配套，不阻塞离线数据实验；不等待drama或母语评分。
  D03复听不作为开训理由，生产r8保持不变。本节优先于下文旧的“当前优先”顺序。

### r10 已完成：可信 ref 开/关小预算对照

用户已同意继续微调。先复核现有 CV22 th/id 的同人跨文件配对，不等 drama、不开发同源切片。
复用 r8 的源配比，仅将 TH/ID 各3分 FLEURS 替换为已配对 CV22，保留所有 GigaSpeech2 份额。
固定12小时目标音频预算；A/B 同初始化、target、顺序、步数和超参，唯一训练输入差异是 ref 字段。
两组从同一官方基座新训1 epoch，**不是继续训练 r8**；投影均关闭。
先审计与预检，通过后串行训练，再按完整 v2 的182 case × 5 seed 比较 A/B/base/r8。
这是小预算方向实验；低 ref 覆盖与单个训练 seed 限制阴性结论，不自动转正。
状态与复现见 `docs/ref_ab_20260917.md`；旧推理参数实验不重跑。
预检已通过并启动后台流程：两组各5472条/11.9866h/342步，B有575条ref；本地124项、远端116项测试通过。
A/B均已完成342步，完整1820条生成/ASR、两两音色分析已完成，18:40写入 `done`；
配对统计和30对声学试听报告均已生成，人工评分未做。B在五语种的跨seed/跨cue均值均低于A和r8，
本轮不转正、不扩大预算，也不自动追加A/B的新文本生成。详见 `docs/ref_ab_20260917.md`。
训练池的评测文本已在新清单中排除；r8已见95/182条v2文本，后续报告分已见/未见组。

### 新文本对照已完成：先核对评分重复性，不扩大训练预算

- [x] 150条新文本的输入/参考音频审计与执行预检完成；固定五语种各30条，
  seed42/43/44/45/49，CFG1.8、20步、禁重试，共1500次生成。
- [x] 2026-09-17 18:07 已后台启动 `scripts/holdout_eval.py run`；两组worker现已正常结束。
- [x] 基座/r8完整同条件报告、case配对统计和两两音色分析已完成。
  工作目录 `holdout_eval_20260917`，2026-09-17 20:59:30已写 `done`：
  **1500条生成/ASR、1500条音色分析全部完成**；追加24次ASR复核也已完成。
  内容诊断ID/MS有改善信号，TL/VI均值改善但区间跨零，TH不明确；
  五语种音色均值略升，但VI/TH等局部组下降，不当作全面稳定性通过。
  结果见 `docs/holdout_eval_20260917.md`。r10已收尾；不改冻结输入，不恢复8核亲和性限制。
- [ ] 新文本结果与r10结果一起决定候选复核/下一轮数据配比。暂不启动r11，不自动加epoch或启用投影。
  r10 B在旧集的MS整体CER风险集中于 `ms_nat_16`（含数字/括注）：seed43生成34.4s且转写大幅偏题，
  不能直接解释成数字念错，更不能只剔除这条后宣布通过。详见实验记录。

### 新对照收尾后的执行顺序（不提前启动r11）

1. [x] **先给五语种结论表**：内容诊断与跨seed/跨cue稳定性分别列出收益信号、不明确与局部风险，
   对齐case/seed并查看case配对区间、时长、疑似少读/多读，不以全局平均掩盖单语种问题。
   本轮只比较base/r8，不能选出r11；非数字朗读文本也不能替代旧数字/括注压力或生产短cue验证。
2. [ ] **整理r8本身的修复清单**：结合新报告、已定位的ID/TL跨cue异常与已有声学坏例，
   先取10–20条诊断样本，另保留固定随机对照；尽量复用现有音频，不先重跑全量评测。
   明确区分已确认声学异常与ASR/WavLM疑似问题；r10 B的坏例不能冒充r8线上坏例。
   用户可选标明确异常，不等待drama或母语评分。样本用于调参后标为开发回归，不再冒充最终留出。
   - [x] 已复用新集音频生成15个诊断case+15个固定随机对照，分别输出base/r8小报告；
     只标疑似CER/参考相似度问题，没有伪造人工评分。随机抽样不排除与诊断组重合的case。
   - [x] 前三条内容异常已完成原VAD开启/关闭的24次ASR复核，检查评分器敏感性；
     复用12条已有音频，不重跑TTS、不覆盖原报告。r8 `th_1990/42` CER .5536→.0179，
     部分base泰语反而升高；`vi_1695/42` .8973保持，不能把全部差异归因VAD或直接改训练数据。
   - [x] 冻结任务结束后修复共享门禁浮点边界并补回归测试，另存 `gate_fix/comparison.json`；
     overall/MS恰好100ms的误报消失，TH/ID真实越界保留，历史报告和阈值不改。
   - [x] **ASR解码重复性小实验**：6条TH/VI既有音频×默认/零温度×3次，36/36完成；
     默认2/6条转写波动，零温度6/6三次一致。记录模型/音频/源码哈希与实际解码参数，
     不改变通用ASR默认值，不挑低分覆盖原报告；这不证明ASR准确或五语种普遍稳定。
   - [x] **VI重复转写词级对齐复核**：仅给同case的base/r8两条零温度转写加
     `word_timestamps=True`，2/2完成。r8的12s音频原末段到29.98s；
     对齐后重复段46个词级条目中42个为11.98s零时长，CER仍.897260。
     不把时间戳归位当文本修复，也不把转写异常直接认定为模型复读。
   - [x] **评分器固定随机对照**：原15条随机case（五语种×三种ref语言），base/r8共30条WAV，
     默认/零温度各3次，共180/180完成。两组均30/30稳定，彼此转写一致，也与历史原报告
     归一化转写一致；默认组此次均temperature=0，不能据此判定取消困难样本回退更好。
     与诊断富集组重合的2个case如实保留；不扩全量ASR、不切默认值、不覆盖旧分数。
   - [x] **局部音色裁边对照**：固定TH `1739/1707`、VI `1856/1874` ×5 seed×base/r8，
     40条既有WAV只做原样/内存裁边，共80/80次CPU嵌入；原样20组数值按6位精度全部复现。
     TH42跨cue的r8−base −.046010→−.044534，VI43 −.063893→−.074993；
     首尾静音不能充分解释这两组下降，不把WavLM结果当人工确认。原音频/内部停顿/ref未改。
   - [x] **句首/中段/句尾定位**：上述40条WAV各取3个3秒窗口，120/120次CPU嵌入于22:10:05完成，
     复用前轮整句嵌入，不重跑TTS/ASR、不写裁片音频。TH句首仅2/5 seed跨cue下降，
     中段/句尾各4/5；VI43三处Δ为−.030848/−.067725/−.098840。
     但 `1874/43` 的自身跨seed短窗偏离并非每处最低，不能只归责它或确认单段跳变。
     另已生成20对原始完整cue声学试听报告，无人工评分；窗口不是词级对齐，也不能当新门禁。
   - [x] **同模型双cue材料**：22:26:02完成10对/20条拼接试听，每条内部模型/ref/seed固定，
     原40条WAV的PCM逐样本保留，仅在句间加0.5秒静音。20条文件经6006读取与磁盘哈希一致，
     不重启UI、不重跑模型；报告前缀 `holdout_dual_cue_base` / `holdout_dual_cue_r8`。
     `review_only` 报告不复制单句自动分数或人工标签，自动汇总拒绝；本地139项、远端相关32项测试通过。
     导出时人工未评，后续离线反馈见下条；组内甲乙固定，跨条仍随机。
     旧20对单句报告继续保留，不冒充连贯双句材料。
   - [x] **本机离线双句试听**：用户浏览器拉取6006音频超时，已将20条WAV原样下载到本机，
     SHA256全部与冻结产物一致。入口 `checkpoints/holdout_eval_20260917/dual_cue/offline/index.html`，
     10组代号 `D01`–`D10`，各有A/B；映射另存 `dual_cue/offline_key.json`，不能重新洗牌覆盖。
     无外部资源/JS fetch/音频转码，原生播放器直接读相邻 `audio/`；只标声学异常，可导出反馈文本。
     未评不算无异常，填写不自动保存；未写回原报告。本地140项测试通过，未冒称用户浏览器播放通过。
   - [x] **离线声学反馈已归档**：`dual-4883215cdc38` 的20条反馈已按冻结映射还原，
     16条无明显异常、1条明确异常、3条不确定。**D03-B = r8 / TH / seed44，备注「模糊」**，
     同组base无明显异常；D07两侧（VI43）均不确定，D09-A（base/VI45）不确定，不强判胜负。
     标注另存 `dual_cue/feedback_review/` 两份报告副本，冻结报告、原单句、网页和映射不改；
     自然度10对均未评。双句回归成员已入 `eval_cases/dual_cue_regression_20260917.jsonl`，
     标明仅组级确认、单句/时点未定位，不把两个成员分别标为已确认坏例。
     导入当轮本地140项测试通过、SSH端口探测未通过；后续已恢复并补同步评分/回归清单/代码文档。
     本机归档完整，不重听、不重新合成；远端仅归档，不冒称已发布6006评分报告。
   - [x] **D03原单句/ref工程核对**：已下载4条原单句与共用ref，不从双句重编码；
     5个文件SHA256通过，4段PCM与原双句逐样本一致，间隔严格24000个零样本。
     无PCM满幅样本，但不能排除所有失真或自动定位模糊。ref精确匹配FLEURS TL验证集，
     16kHz/5.11秒，身份未验证，没有可直接认定为同人的替代ref。
     新入口 `dual_cue/d03_localization/index.html`，只补4条原句+ref，代号D03-A/B各加1/2；
     导出时单句未评，后续反馈见下条。`verify.py`可只读复查；本地140项、远端相关33项测试通过。
   - [x] **D03补听反馈：4条原句和ref均无明显异常**。原文「试听了，都没有异常」，
     按上述补听页范围另存 `dual_cue/d03_localization/feedback.json`；
     原双句D03-B「模糊」仍作历史记录，单句复听未复现，不声称已修复或原双句复评通过。
     开发回归成员保留并补最新状态；暂不作为稳定可复现坏例，不再扩试听、指标或以它驱动训练。
     声学判断不扩展为语言质量通过，VI不确定项也不改变。
   - [x] **推理链路只读审计落档**：见 `docs/inference_chain_audit.md`。
     已核对数字词化、auto步数/CFG、ref处理、续写prompt、API输出及DIS段级处理；
     明确记录参数透传/最终参数回显缺口。审计当时两侧代码未改；后续最小本地改动见下条，未部署。
   - [x] **最终采样参数追踪已本地实现**：OmniVoice新增`adaptive_params.effective_generation`，
     在真实生成调用处记录CFG/步数/实际seed等，仅随被接纳音频切换；
     已覆盖异常兜底、质量/漏音/身份/文本重试的采纳/拒绝、次选复用及新旧缓存。
     旧回显保留初始含义；缺记录不补猜，不改变采样参数或候选选择。
     DIS原有`adaptive_params`透传可保留新字段，不改主调方代码；轻量表达式检查通过，非端到端实测。
     OmniVoice相关241项测试通过，故意置空字段时8项新断言失败，恢复后通过。
   - [x] **部署可用性只读核对**：本机现有Vast凭据下查询返回0实例，
     环境未设置`DUBBING_API_URL/VOXCPM_CLOUD_URL`。仅代表该凭据/环境，不推断其它账户或固定地址已停机。
     未新开实例、加载模型或调用合成；实际线上权重与部署版本仍未验证。
   - [ ] **当前优先：实际模型身份与新字段的在线验证，不继续扩指标扫描**。
     D03单句/ref补听已完成且无明显异常，不再等待用户补时间点或要求重听。
     本地追踪代码已就绪；有可核验服务时先确认部署版本/权重，再验证最终参数记录与缓存语义。
     不能把“继续审查源码”当已有同条件pre/post音频，也不为了避缓存换seed后声称单变量。
     本包未经OmniVoice/DIS后处理，不能把这条反馈直接归因为生产后处理；
     生产对应链路仍需继续核对，
     只有同text/ref/seed/实际cfg/steps的裸输出与处理后音频才能作归因对照；配不齐就记录具体缺口。
     用户可选只标明确异常及时间点；拼接边界问题回原WAV确认，不能把单句异常自动传给整组两句。
     不按WavLM直接标娃娃音、改训练比例，不把未评语言维度当通过或设为离线前置。
     VI重复转写仍须原音频片段或独立证据，不手删文本降分；生产r8不变，不自动开r11。
     用户补充的链路仓库：服务端 `/Users/dunxu.liu/workspace/others/OmniVoice`，
     主调方 `/Users/dunxu.liu/workspace/dubbing/dubbing_intelligence_service`。审计及OmniVoice本机改动已落档，
     DIS仍未修改；实际部署与同次生成pre/post证据未补齐，源码默认、旧归档与实际部署必须分开。
3. [ ] **按证据选择修法，不同时展开所有路线**：
   内容/特定语种回退 → 检查相应训练文本对齐、来源和cue长度覆盖，只选一个已证实的数据缺口；
   爆音/金属感/音色跳变 → 比较同条件裸输出、ref及生产后处理，先定位发生环节；
   结果不明确 → 保持r8，先处理可复现坏例，不靠堆小时数或重复参数网格制造结论。
   怀疑联合干扰时，补TH或问题语种的单语小对照作归因，最终目标仍是三语种联合LoRA。
4. [ ] **满足条件才开r11小预算A/B**：须有可复现问题、预先写明的改善目标及经过审核的干预数据。
   两组同初始化/预算，只改一个数据变量，再与r8比较；不直接续训r8，不同时改源、ref比例、
   投影和步数。维持投影关闭及冻结的生产推理参数，不因候选库存已准备好就全部送训。
   暂不扩大r10 ref路线，也不预先指定下一轮该补哪个语种。
5. [ ] **候选有收益才扩大验证**：三语种逐项检查，并补中英文回放、旧压力/坏例回归；
   任何门禁回退都不自动转正。用于选择修法的样本与后续未用验收样本区分，
   语言自然度、口音和情绪仍保留未验证，不要求用户填不懂的维度。

### 等待期间并行：统计与下一轮数据候选

- [x] `scripts/ref_ab_diagnostics.py audit` 审计现有源验证集：逐条比较训练 target/ref、
  归一化文本、原音频、可信身份、会话，以及验证损失使用情况；音频头检查全部通过。
  TH20条/ID12条仅在 r8/r10 范围未见，补查 r9 后也全部已用于验证；
  **当前不能从这些源验证集凑出五语种全新独立测试集**，未知说话人仍是未知。
- [x] 配对统计脚本与回归检查已实现，CPU sidecar 已写出 r10 的 `paired_diagnostics.json`。
  先在 case 内平均5 seed，再按 case 重采样；缺失值不补零，不把910次生成当910条独立文本。
  分语种、数字/非数字、r8已见/未见分别输出；区间不是新门禁，也不代替多训练 seed。
- [x] 下一轮候选池重扫：`scripts/training_pool.py` 已完成。
  仅复用现有15个源，不新增下载、不改原音频/原清单、不改本轮冻结配置。
  隔离评测与源验证文本/身份/会话，检查真实波形、去精确重复，疑似削波/长零尾单列待复核。
  扩充可信 TH/ID 跨文件 ref 的**候选库存**，不把候选全配对当训练要100% ref。
  CV22 VI 与 YODAS MS 单列，身份/口音不补猜测标签；实际下一轮配比等 r10 结果决定。
  实测61624条→61025条/108.6713h，598条文本重合被排除，1条疑似削波隔离；
  原TH/ID源清单13000条配对→23656条可选配对，未新增真实录音。详见数据候选记录。
- [ ] 新外部数据仅按缺口定向补：**TL 的新文本/短句与可信同人录音**（id/th 同类缺口按需）。
  先核实授权、身份、原始文本与未用测试切分，再设下载小预算；不以囤小时数代替质量提升。
  现有v2未见非数字文本TL仅4条，须补独立评测，不能靠增加TH/ID数据掩盖。
  （原条目还列了 MS/VI 的缺口与「v2 未见非数字文本 MS 仅 1 条」——2026-10-19 起 ms/vi 退出
  目标语种，这两个缺口不再补；MS 那 1 条已随过滤从 v2 移除，TL 的 4 条仍在。）
- [x] 新文本诊断集已冻结：`eval_cases/fleurs_test_holdout_20260917.jsonl`，
  五语种各30条（**2026-10-19 过滤掉 vi/ms 后为 th/tl/id 各 30 = 90 条**），
  来自固定版本官方FLEURS test，仅在远端取文本清单，未下载源音频。
  与现有raw、各版本processed train/val、候选池和历史评测文本做归一化排除；
  不证明基座预训练未见或新说话人泛化，不加进当前r10的182条冻结case。
- [x] 新文本同条件评测：base/r8的1500条生成/ASR和音色分析全部完成；5 seed含42→43/49。
  r10候选不自动追加。
  不拿该集补训练，不删差句或挑好句当验收；抽出的修复样本单列开发回归。
- [x] 预先抽取15个case×seed42/43，共30对声学试听：五语种×中/英/菲ref分层随机，
  不按B结果挑样本。`scripts/acoustic_review.py collect --wait` 已复用现有音频生成两份小报告；
  人工声学评分尚未进行，仅可选标明确异常，不要求母语打分，不阻塞离线实验。
- [x] 盲听防错配守卫：拒绝重复/缺失case、文本/ref/control不一致；
  写回时核对会话音频与所选报告，防止切换下拉框后把旧评分写到另一模型。已补回归测试。

产物与当前结果见 `docs/data_candidates_20260917.md`；不因完成候选池就自动开下一轮训练。

执行记录见 [2026-09-17 实测与实验状态](docs/quality_20260917.md)：
CER 默认值与稳定性输出已实现；声学标注入口已支持未评/不确定；3 条历史明确坏例已入库；
24 case × 5 seed × 4 强度 GPU pilot 已完成，端点校验通过；未发现统一占优方案，保留 r8。
14 case × 5 seed 的三组参数对照也已完成，共690条；没有候选通过本轮筛查，保持现有生产设置。
修复时长/CER样本口径错配，本地121项、远端108项测试通过。条件尚未具备的后续项不勾选，也不启动无依据训练。

1. [ ] **明确坏例集与分阶段定位**：优先整理已有归档的坏例，并加入固定随机样本作对照。
   同 text/ref/seed 比较 base 与 r8、模型裸输出与生产处理后输出，记录异常发生在哪一阶段。
   用户仅判「有明确异常 / 无明显异常 / 不确定」及类型、时间点；不填不懂语言的自然度/可懂度分。
   坏例回归集与随机抽样结果分别统计，不把刻意富集坏例的比例说成线上坏例率。
2. [ ] **验收口径与稳定性输出**：统一 `runlog` 的 `0.005` / `0.05` 默认值并补测试；
   **统一与测试已完成**，历史报告数字不手改。复用 base/r8 的 v2 五 seed 报告，
   按 case 输出 `speaker_sim`、`audio_sec` 的 std/极差与 CER std，再按语种汇总。
   补线上相邻重试偏移（seed、seed+1、seed+7）与同 speaker/同 seed/不同文本的跨 cue 检查；
   不把已有 42–46 组误称为已覆盖 +7。
   - [x] `--summarize` 另存报告；case 内 std/极差及分语种聚合；base/r8/r9 完整报告已重算。
   - [x] 声学标注与可选母语评分分开；未评项不当无异常或平局，重新载入保留既有标注。
   - [x] 生成音频两两音色距离：base/r8 完整 1820 条已重算；TL 跨 cue 略差，ID/TL 具体异常组已定位。
   - [x] +7 重试偏移：完成42/43/44/45/49的210条固定参数对照，覆盖42→43/49。
3. [ ] **r8 LoRA 强度对照，先不重训**：`W = W_base + strength * ΔW_r8`，
   试 `strength=0/0.5/0.75/1`（0=base，1=r8）。固定 cfg=1.8、steps=20、ref 与 seed 组，
   禁用自动重试。严格按训练配置加载，再额外缩放增量；不改训练 r/alpha，不修改 submodule。
   在报告与实验标识中保存 strength；验证 0/1 端点和开关切换后的强度，避免缩放被静默重置。
   先跑覆盖三语种的固定子集，候选再跑完整 v2 集；收益不保证单调，所有结论都要对照 base/r8。
   - [x] 480 条固定 pilot 与真实 GPU 0/1 端点验证；分语种 CER/时长门禁已分析。
   - [ ] 若出现明确候选，再做完整验证；当前有语种和尾静音权衡，不按平均分转正。
4. [ ] **推理参数与 ref 单变量对照**：固定选中的 strength 后，分别测试 CFG、步数、
   同一说话人的不同干净 ref，其他条件不动；记录实际生效 cfg/steps/seed，避免服务端自适应混淆。
   比较坏例、内容回退、稳定性与耗时；best_of 留作后续筛选实验，不能把筛选收益当模型本身更稳定。
   CFG/步数实验针对声学异常和稳定性，不重复拿它们验证已做过的带宽扫描。
   - [x] 固定生产strength1，分别试CFG1.6和28步；均未得到统一收益，不更改生产默认。
   - [ ] 同人不同干净ref对照：目前未确认可用的同人替代ref，不拿不同人冒充。
5. [ ] **现有单人录音切片配 ref：先审计，再小实验**。优先核查现有较长真人录音
   （如 FLEURS 候选）的单人证据、可用时长、切点和文本对齐；不假定同视频或所有长录音都是同人。
   将确认单人的录音切成不重叠 ref/target，ref 可先取 3–5s、target 保持 3–30s，
   不截词，文本只对应 target。保留原音频、起止时间与审核依据；局部同人证据不伪装成全库身份。
   同原音频所有切片在同一 split，已知同人的跨文件/跨会话仍须隔离；ref/target 重叠必须拒绝。
   **当前 `pair_references` 拒绝同 `origin_audio` 配对**：明确时间区间表示并补防泄漏测试后，
   才能显式支持这类配对；不改文件名或伪造 origin 绕守卫。产物写新数据集，不污染旧训练源。
   - [x] 现有清单审计完成：FLEURS未登记可信片内单人证据/词级对齐；CV22 th/id已有跨文件身份配对，
     优先复用后者。尚未生成经过审核的同源ref/target切片，不放松守卫。
6. [ ] **有合格配对后再训练**：同一份裁切后 target，先做 ref 开/关对照，
   保持 `enable_proj=false`；有效后才在同数据上单独比较 `enable_proj=true`。
   固定三语种配比、回放与训练预算，统计原音频实际曝光（最多 3×）、ref 复用和混合后覆盖率。
   30–50% 是可信 ref 的混合后目标，不为凑比例放宽身份或过采样。用独立录音、跨语言 ref 验收，
   不把同录音切片上的改善当跨录音/跨语言泛化。**此路线在本项目未验证，不预告有效**。

共同要求：分语种报告、同 case/seed 的配对对照，稳定性实验用至少 5 seed。
工程检查完成后只写「工程检查通过，语言自然度/口音/情绪未验证」；未评项不是通过，
自动指标和非母语听感都不能替代完整语言验收。明确失败形态写入 `eval_cases/`（带 `note`），
用于调参的坏例与最终留出的验证样本分开。

### r9 结论与已完成项

r9 **维持不通过**：`enable_proj=true` + cv22_th6/cv22_id6（50% 配对，
全局 ref 覆盖 10.2%，2249 人）下，speaker_sim 0.9121→0.9074；
en ref 0.8272→0.8159 低于基座，id ΔCER +0.0435、漏尾 0.075、多读 0.031，vi 同向恶化。
zh 0.0775→0.0494、th 略好、ms 保持 0.0115。
**只能说明这一组合未奏效**：cv22 加入同时 gs2_th 9→5、gs2_id 10→7，
训练也从 r8 的 2020 步变为 3130 步，不能单独归因给 CV22 或投影层，更不能推出必须改用 drama。
历史运行记录与原始指标保留在 [runs.md](docs/runs.md)，旧「下一步」不代表当前排期。

已完成（2026-09-15~17）：r5-r7 裁定；r8 复现 r2 配方成功并转正生产仓库；
r9 训练+验收（不通过，runlog 已归档）；eval `--shard/--merge` 并行；
`--mix` 重复参数修复；`repair_refs` 重配对工具；cv22_th/id 的 client_id 身份上调为可信。

### 备选项（不阻塞当前实验）

- **drama**：将来有授权素材或需验证表演能力时再做一集试产；成品混音需分离并复核伪影，
  字幕/台词本校对、同人确认仍要做。现在不催用户提供三语种成片。
- **CV22**：仅考虑固定数据预算的小份额、可信配对对照，不再因单轮混合实验一概判死整条路线。
- 以下保留此前数据调研登记，执行前复核可用性与授权，不作为这轮的数据前置。
- **ms 数据跟进**：CV26 scripted（29 人 3.65h）+ Spontaneous Malay（24 人 6.19h）
   需注册 MDC 下载；BabelSpeech 50h Malay/Indonesian（Apache-2.0，manual gated，
   申请后先确认 `audio_info.json` 有无 speaker 字段）；yodas2_ms 口音抽听
   仍需马来西亚母语者，用户的声学异常标注不能替代。
- **vi/tl 同说话人语料**：PhoAudiobook（941h/735 人，缺许可）给作者发邮件；
   tl 仍是 BigFTagaCSC 询价或 drama_tl 自建。
- **Nexdata 询价**（待用户决定）：200h ms（SKU 1280）/ 621h th 对话（622 人带 ID）。
- **母语验收**：将来有合适评审时先评当前 r8/候选，不要求用户完成。旧 r4 报告仅备查：
   A=`base_1759722cf7884b88bf66c32f06dd1d66`，
   B=`lora_omni5_r4_latest_0859d1d75d6f46c99e3efb735dd21800`。
- **BWE**：带宽扩展单独验证，不混入 LoRA 训练；本轮不承诺通过微调补出高频带宽。

---

目标（用户 2026-09-14 提出）：
1. 同 ref wav + 同文本，**跨 seed 合成稳定**（用户已确认是这个含义，不是"音色贴合 ref"）
2. 按 qc 规则提升合成质量，符合 qc 要求
3. 提升人声质量——当前有些像 16k 音质，不够清晰明亮

## 历史诊断 A–E（保留实测，不作为当前执行顺序）

A/C/D 已完成的带宽诊断、指标标定和验收扩容继续复用；B 的稳定性输出仍需补齐。
E 保留旧轮次的尾静音实验过程，不据此再次自动启动尾裁训练。
原「音色改善硬依赖 drama → ref 30–50% → enable_proj=true」结论撤回：
需求是可信同人条件，不是素材必须来自短剧；投影层开关是待单独验证的变量。

---

## A. 目标 3：带宽诊断（已完成大半）

材料：OmniVoice 那次 base 模型 cfg×steps 扫描的 48 条输出
（`../OmniVoice/results/voxcpm_quality_20260909/case*_steps{20,32}_cfg{1.6,1.8,2.0,2.4}_seed*_n1.wav`，
48kHz 单声道，**已过 OmniVoice 后处理链**，无 LoRA = 纯基座行为）。4096 点 FFT / 1024 hop。

| steps | cfg | 99%滚降 | 有声99%滚降 | 质心 | 0-4k | 4-8k | 8-12k | 12-16k | 16-24k |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 1.6 | 7494 | 5887 | 1227 | .950 | .043 | .0057 | .0005 | **.0000** |
| 20 | 1.8 | 7609 | 5844 | 1226 | .953 | .040 | .0061 | .0004 | **.0000** |
| 20 | 2.0 | 7578 | 6012 | 1246 | .951 | .043 | .0058 | .0004 | **.0000** |
| 20 | 2.4 | 7533 | 5105 | 1237 | .953 | .042 | .0054 | .0003 | **.0000** |
| 32 | 1.6 | 7596 | 5941 | 1226 | .956 | .037 | .0063 | .0004 | **.0000** |
| 32 | 1.8 | 7559 | 5766 | 1210 | .956 | .038 | .0057 | .0004 | **.0000** |
| 32 | 2.0 | 7578 | 5533 | 1237 | .951 | .043 | .0058 | .0004 | **.0000** |
| 32 | 2.4 | 7619 | 5824 | 1258 | .949 | .045 | .0060 | .0004 | **.0000** |

**三条结论：**
1. **基座输出就是 16kHz 带宽装在 48kHz 容器里。** 99% 能量滚降 7.5kHz（≈16k 采样的 Nyquist），
   16kHz 以上**恰好为 0**（砖墙，不是衰减到很小）。用户"像 16k 音质"的听感**准确且可测**。
2. **cfg 与 steps 完全不是杠杆。** 8 个网格点滚降 7494-7619Hz，极差 <2%，在 seed 噪声内。
   ⇒ "提 steps 改善音质"只对 VAE 的 transient smearing 成立，**对带宽无效**。
3. **成因在 VAE 编码器侧，微调结构上碰不到。** `AudioVAEConfig.sample_rate=16000` /
   `out_sample_rate=48000`（`audio_vae_v2.py:366-369`）；编码一律 16k（`voxcpm2.py:478`）；
   解码虽带 `sr_cond` 分档（`sr_bin_boundaries=[20000,30000,40000]`），但 `decode()` 在
   `sr_cond=None` 时**已默认取 `out_sample_rate=48000`**（`audio_vae_v2.py:452-472`），
   调用点 `voxcpm2.py:748/1053` 正是不传 ⇒ **已经在最高带宽档，没有"传错档位"这种便宜 bug 可捡**。
   LoRA 只动 LM 与 DiT 的 q/k/v/o，VAE 全程不训。

⇒ **目标 3 只有"带内明亮度"这半属于微调**（2-8kHz 的共振峰/辅音占比，实测只占 3.7-4.5%）：
若 r2 比 base 暗，那是训练数据引入的回归（32% YouTube 自发口语 + FLEURS 录音质量参差），修在数据侧。
8kHz 以上带宽只能靠推理后处理（带宽扩展 / 谐波激励），是独立产品决策，**不占微调轮次**。

**剩余动作**
- [x] 给 `src/voxft/qc/audio.py` 加 `spectral_rolloff_99` 与 `band_ratio_2_8k`，进 `analyze()`（2026-09-15 完成，
      含合成信号单测 4 个）。
- [x] **关键分叉已结案（2026-09-15，远端实测）**：用我们自己的 `voxft.eval` **裸输出**（84 条 base、
      48kHz、无 OmniVoice 后处理）复测——rolloff_99 中位 **7262Hz**（p75 7450）、16-24kHz 能量
      占比 max 0.0035%（砖墙）、与 OmniVoice 后处理过的输出同一堵墙。⇒ **坐实是模型（VAE 16k 编码器），
      不是后处理链**，目标 3 的带宽半彻底结案，只能靠推理后处理 BWE（独立立项）。
      附带量到「带内明亮度」：base `band_ratio_2_8k` 中位 0.162，**r2 配对 Δ 均值 -0.0143、
      71% 对子变暗**——r2 训练数据确实引入了轻度变暗（32% YouTube 自发口语 + FLEURS 录音质量参差），
      这是目标 3 里微调能碰的那半，修在数据侧（归 E）。
- [ ] ~~需远端实例~~（实例在线，已完成上述复测）

---

## B. 目标 1：跨 seed 稳定性（基座量级已免费量到）

材料：同一批 48 条 + `voxcpm_quality_bestof_20260909/` 的 best_of=2 对照（6 条）。
指标取 OmniVoice 自己逐条记录的 `audio_qc.speaker_similarity_to_reference`
（**ERes2NetV2 口径，门限 0.45**——与本项目 WavLM X-vector **刻度不可互换**，只看相对方差）
与 `audio_duration_seconds`。每 cell 3 seed（101/202/303）。

**基座跨 seed 表现（16 个 cell）**
| 量 | 极差均值 | 极差最大 |
|---|---|---|
| 音色相似度 | **0.0870** | 0.1867 |
| 韵律相似度 | 0.2644 | 0.5255 |
| 时长 | **0.547s** | **1.625s** |

- **不稳定的主因是时长，不是音色本身。** 片子只有 ~2s，跨 seed 时长极差均值 0.547s（≈27%）、
  最差 1.625s。音色相似度跟着时长走：模型多垫一段/换一种停顿，嵌入就变了。
- **稳定性强烈依赖 case**：case1 音色极差 0.070-0.187（很飘），case2 只有 0.010-0.092（稳）。
  两者都是短 cue，差别在文本 ⇒ 不能拿一条 case 的结论外推。
- **cfg 越低越稳（case1 上一致）**：cfg 1.6 的音色极差 0.0697/0.1344、均值 0.6998/0.7052，
  都优于 cfg 2.0 的 0.1819/0.1867、均值 0.6715/0.6870。case2 上规律不明显（cfg 2.4 反而最好）。
  ⇒ **与"提高 cfg 换音质"的直觉相反**，至少对 case1 类内容，降 cfg 同时改善均值与方差。
  n=3 seed × 2 case，**只能当假设，不能当结论**。
- **best_of=2 是真杠杆，但它是筛选不是让模型变确定**：
  case1 音色极差 0.1258→0.0672（腰斩）、时长极差 0.915→0.235（降到 1/4）；case2 改善很小
  （0.0282→0.0183 / 0.575→0.440）。代价：耗时 5.0s→6.8s，且 case2 在 n2 下**新增了一个
  `metallic_resonance` severe**（n1 没有）。⇒ 提均值、压方差是真的，但会引入新伪影，且 2× 成本。
- ⚠️ 基座里已有 severe：`output_vocalization` ×2（case2 cfg1.6，steps20/32 都有）、
  `metallic_resonance` ×2（case1 cfg2.0/2.4）。**这些是基座自带的，别记到微调账上。**

**剩余动作**
- [x] ~~**先问用户一个 scoping 问题**~~ **已查实（2026-09-15，读 OmniVoice + dubbing_intelligence_service 源码）**：
      线上默认 **per-speaker stable seed**——`backends/voxcpm.py:438-463` 用
      `sha256(speaker + model_id + cfg + steps + …)` 派生（**不含文本**），`VOXCPM_SPEAKER_STABLE_SEED`
      默认 true，**同一说话人的所有 cue 共享同一个 seed**。⇒ 跨 seed 方差在线上首 take 不可感知，
      每个 speaker 只生活在一个 seed 上；**但重试路径大多会换 seed**（模型内 retry_badcase +1、
      prompt-leak +1、speaker-mismatch +1、text regen +7、best_of +1009、主调方整轮重试 crc32 重派生；
      例外：metallic 质量重试**不换 seed 只降 cfg 加 steps**，`api.py:9184-9196`）。
      ⇒ **正确量法是两件事都量**：① 跨 seed 方差 = "重试把 seed 从好 take 换到坏 take 的风险"，
      用相邻 seed 扰动（base、base+1、base+7 这类线上真实偏移）而不是无关大跨度 seed；
      ② **跨 cue 一致性**（同 speaker 同 seed 不同文本）才是首 take 线上可感知的稳定性。
      另：`reference_identity_locked` 的 cue 对伪影重试保 seed 防邻句音色断层——跨 cue 音色断层
      是线上真实顾虑。
- [ ] `voxft.eval` 把跨 seed 方差做成一等输出：每 case 报 `speaker_sim` 的 std/极差、
      `audio_sec` 的 std/极差、CER 的 std。**seed 数提到 ≥5**（估方差比估均值需要更多样本，
      现在的 3 seed 只能给出上面这种量级）。
- [ ] 用 WavLM 口径量 base vs r8 及 strength 候选：LoRA 是让它更稳还是更飘？
      已有弱信号（r2 在 zh/en/tl ref 下比 base 稳 +0.0018/+0.0058/+0.0152），但那是 3 seed、
      且只有 delta 没有绝对值，不足以判断。
      ⚠️ WavLM 的同人上限是 0.995-0.998，动态范围与 ERes2NetV2 完全不同，**两批数字不能混着比**。
- [ ] 按当前待办先测 strength，再逐项测 cfg/steps/ref；best_of 另作筛选实验，
      样本量按 D 的口径来。零训练成本不等于零推理/验收成本，不一次铺开大网格。

---

## C. 目标 2：qc 尺子标定（已完成，2026-09-15 定案）

结论：`metallic`（AUC=0.060 反相关）与 `low_snr`（AUC=0.509 纯随机、28.7% 误报）
**永久降级为参考值、不作门禁**，已写进 `docs/qc_gates.md` 与 eval 报告 note；
时长类三条门禁（duration_inflation / tail_silence / speech_ratio）已进 `runlog` 并配测试。

- [x] ~~84 对盲听标注做 ROC 标定~~ → 标定定案：两指标在本项目分布下无区分力
- [x] 时长类指标升级为门禁（runlog `_duration_gates`，overall + by_lang 双口径）
- [x] 「指标 → 门禁/参考 → 阈值 → 标定依据」总表 → `docs/qc_gates.md`

---

## D. 验收统计功效扩容（已完成，2026-09-15）

- [x] `eval_cases/omnivoice_prod_v2.jsonl`：182 条 case（th/vi/id/ms 各 30+，zh/en 各 12），
      来自各源 val 集无泄漏；5 seed（42-46）= 910 条/轮
- [x] ref 覆盖线上真实分布：zh/en/tl 各 4 条新 ref（远端 `eval_cases/refs/`）+
      线上 prod_ref_fil，13 条轮换——基线实测 ref 说话人对 CER 影响 0.014→0.109，多 ref 必要
- [x] base 新基线已跑：`base_8de890a3af74491b948c1ae0449967b0`（总体 CER 0.0568、漏尾 3.85%）
- [x] eval 并行：`--shard K/N` + `--merge`，910 条的下轮验收可 ~3× 提速
- [ ] 同配置重跑量 run 间方差（暂未做；v2 样本量已是旧集 6 倍，红线 0.05 够用）

---

## E. 历史训练：修时长不稳（同时服务目标 1 与目标 2）

已被人工盲听确认的唯一真实缺陷：尾部静音 0.084s → 0.292s，总时长 +12.3%，
而有声段字/秒只差 1.7%（**不是语速问题，是垫静音**）。B 节独立量到的基座跨 seed 时长极差
0.547s 曾提示需要拆分停顿与语速。最初归因 gigaspeech2 犹豫停顿的假设已被下面的实测修正。

- [x] 数据侧修：**r4 已执行并验收（2026-09-15）**。实测推翻原假设：尾部垫音不是 gigaspeech2
      的犹豫停顿——**长尾来自 FLEURS**（训练样本尾静音 p50 0.30s；gigaspeech2 p50≈0s），根因是 Silero VAD
      `min_silence_duration_ms=500` 把 <500ms 尾停顿并进语音段。r1（纯 FLEURS）输出尾静音 0.305s 与
      r2/r3 相同，与数据分布数字级吻合。修法：pipeline 加 `max_tail_silence_sec=0.15` 硬上限（只裁尾），
      远端后裁 11 源得 `<src>_tc`（FLEURS 50-87% 行被裁、147 条跌破 3s 丢弃），按 **r2 配比**重建
      `joint_omni4`（76.5h）重训 `lora_omni5_r4`（同 r2/r3 超参，2055 步）。
      明亮度顺带实测：变暗与 gigaspeech2 份额相关（r1 0.164 / r2 0.152 / r3 0.155），但训练源的
      band_ratio 分布两源几乎相同 ⇒ 无可操作的筛选靶子，本轮不动（详见 AGENTS.md「输出带宽」条）。
      **r4 结果**：尾静音部分修复（mean 0.292→0.228、p90 0.42→0.36，base 为 0.084/0.174），
      数据尾已压到 0.15s 但模型仍垫 ~0.25s（剩 ~0.1s 成因未明）；离线轨红线全在数字彩票区。
- [x] r5 续上（2026-09-15）：全量尾裁数据 + 短句分布修复（cv22 三源 + yodas2_ms），
      结果见 runs.md——ms 大胜、th 改善、id 红线待盲听。尾静音进一步收敛到 0.189/0.28。
- [ ] 注意 AGENTS.md 已实测否掉的两条：`vi` 退化主因是**长度出分布**不是数字覆盖；
      val loss **不能**用来选 checkpoint（一律交付 `latest`）。
- 验收：D 扩容后的口径下，尾部静音 p90 回到 base 量级、跨 seed 时长极差下降，
  且盲听 B 负−B 胜 < 2（`eval.BLIND_LOSS_MARGIN` / `BLIND_MIN_LOSSES`）。

---

## ref 与投影层的依据和边界

- 官方打包逻辑 `third_party/VoxCPM/src/voxcpm/training/packers.py:process_tts_data_with_ref`
  接收 ref、target 与 target text，只在 target 上计算损失；并不要求 drama 来源。
- `third_party/VoxCPM/conf/voxcpm_v2/voxcpm_finetune_lora.yaml` 默认 `enable_proj=false`；
  `voxcpm2.py:forward` 中 ref 条件还经过可训练的 LM/DiT，投影冻结不能推出音色结构上改不动。
- `third_party/VoxCPM/src/voxcpm/modules/layers/lora.py:LoRALinear` 已有增量缩放；
  但本项目的 strength 参数、实验报告和同源区间配对支持仍须实现与测试，不能当现成功能使用。
- 同录音切片只提供同语种、同录音条件，不等于有跨语言同人数据；跨录音和 zh/en ref 到
  三语种的泛化仍需实测。没有数据证据时，不声称已解决克隆或情绪问题。

---

## 明确不要做的事（避免白跑轮次）
- **不要**为了扩展带宽重复调 `cfg_value` / `inference_timesteps`；针对明显伪影/稳定性的对照仍可做。
- **不要**为了目标 3 去开 `enable_proj` 或动 VAE：瓶颈在 16k 编码器，LoRA 够不着。
- **不要**把带宽扩展（BWE）混进微调任务：它是推理后处理的独立立项，有自己的伪影与延迟代价。
- **不要**默认"提高 cfg 更稳"：B 节实测在 case1 上正好相反（cfg 1.6 的均值与方差都优于 2.0）。
- **不要**把基座自带的 severe（`output_vocalization` / `metallic_resonance`）记到微调账上。
- **不要**拿 `metallic` / `low_snr` 否决任何一轮微调：C 已将其定案为参考值。
- **不要**把未评的母语质量写成通过，也不要拿 F0 高低直接判娃娃音。
- **不要**用不可信同人配对或重叠音频提高 ref 覆盖率，不用 TTS 合成语音补训练量。
- **不要**在没有固定对照与留出验证的情况下继续调数据配比。
- **不要**混用 ERes2NetV2 与 WavLM 两套相似度刻度（B 节的 0.087 极差是前者，与我们的 0.86-0.94 无关）。
