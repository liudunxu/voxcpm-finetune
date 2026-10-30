# 当前语料复核与人声分离试验（2026-10-30）

用户要求自主提升质量，明确疑似也剔除、数字等价表达可忽略，并提出模型去除背景音/BGM、保留原人声。
本轮先把训练池中的问题拆开核验；未开启新训练、修改旧评测或部署。

## 最新可追溯候选

远端根`/root/autodl-tmp/quality_reaudit_20261030/`，当前候选`music_ref_candidate/`，
3975行/7.4965h；原895条验证集字节不变、完整1634历史val及全部评测文本继续保护。
所有保留行逐字节来自旧5015行，未改标、未新增录音、未恢复旧排除。
plan `e3e8fcff85ad7f12ca687495dfd48ae9aa7c26b6ff8e1a70365015335c452774`。
文本一致性还在复核，不能立即把它称为已验收训练集。

| 来源 | 许可 | 行 | 小时 |
| --- | --- | ---: | ---: |
| cv22_id | CC0 | 863 | 1.1114 |
| cv22_th | CC0 | 350 | .4472 |
| fleurs_en | CC-BY-4.0 | 150 | .3659 |
| fleurs_id | CC-BY-4.0 | 304 | .8092 |
| fleurs_th | CC-BY-4.0 | 29 | .0943 |
| fleurs_tl | CC-BY-4.0 | 397 | 1.3436 |
| fleurs_zh | CC-BY-4.0 | 259 | .5902 |
| gigaspeech2_id | Apache-2.0 | 759 | 1.3582 |
| gigaspeech2_th | Apache-2.0 | 503 | .8907 |
| yodas_th | CC-BY-3.0 | 361 | .4858 |

## 音乐疑似与覆盖缺口

先把旧清洗证据重新连接至实际5015训练行、6027目标/ref文件，SHA全部通过，旧旗标未回流。
旧AST只按.8提示强疑似，而当前1031个目标的Music分数≥.5，集中在GigaSpeech2_ID。
这里的.5只是有限审查分层，不是借用Granary的CNN14阈值或经校准的音乐概率。

采用[CLAP](https://huggingface.co/laion/clap-htsat-unfused/tree/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a)
三组“干声/人声伴随音乐”文本对照，10s窗/5s步长，所有三对差值正向才记共同提示。
1087结果=176原试点复用+911新判别，冻结源与旧结果不动；原始余弦、窗口及聚合已重算核验。

| 分层 | 条数 | CLAP三对共同偏向音乐 |
| --- | ---: | ---: |
| 当前AST≥.7 | 220 | 207 |
| 当前.5≤AST<.7 | 811 | 719 |
| 低AST对照 | 40 | 7 |
| 旧排除对照 | 16 | 16 |

926共同疑似中ID917/TH9；旧排除对照也不是人工真值，两个模型可能共享偏差，不能认证926条都有BGM。
按用户保守原则隔离926共同疑似、105单AST未解决项、7单CLAP项，共1038文件；路径/原音/PCM/ref传播
共移除1039训练行，形成3976行中间候选，plan `1128ca0d327abf365f2d829d72094463574187c90243bb7531c8a1d06e225f45`。
所有状态仍uncertain，低分未获“干净认证”。

随后显式检查1218次ref使用/1153独立文件，发现一个CV_ID ref的AST=.7174未进入目标扫描。
独立CLAP单条也三对正向（plan `58eab68d9ce9b1301bcf80edc96dd719ba52eb5b42264c7f9bbd9a402c6957b3`），
再排除其唯一使用行，得到当前3975行；累计1040行。原件不物理删除，历史flag和保护val全部继承。

## 数字表达与独立ASR

原5015行有1417条Whisper CER>.05、537条>.10。抽查既有数字写法、专名/拼写差异，也有疑似漏词/增词。
因此不直接降低旧.15阈值。用户明确数字写法可以忽略，`quality_numeric.py`只豁免数值相同、
剩余内容全文一致的成对表达；保留值变化、漏词、混合错误。支持有限整数、明确地区千分位与中文时间；
未支持格式保持待复核，不能以未识别等价为由自动剔除。首版91条纯数字差异，扩展格式后94条，
其中77条原CER>.05；这些是新的清洗诊断视图，原标签/Whisper分数/旧模型评测都不改。
数字豁免不会覆盖另有音乐/声学风险的同一录音。

[Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0)
使用固定revision、Apache-2.0、全部权重/配置SHA核验。独立overlay为qwen-asr0.0.6、transformers4.57.6、
torch2.6.0+cu124；训练原环境未改，旧离线验收继续large-v3。
试点80个按来源/哈希固定的Whisper分歧样本（排除首版纯数字差异）+40个低分歧对照；
context为空，无期望台词提示；ID/TH/zh/en给语言提示、TL自动检测；bf16/SDPA/batch4、512token上限。

120/120完成、全部见EOS，峰值2792MiB，约38s（含加载）。13个Whisper高误差样本被Qwen完整转对；
9条两ASR归一化一致但不同于标签，仍混有数字、同音字或重复符表达，不能全部当错标。
有意义的疑似包括ID共同多出`yang besar`、FLEURS_ID共同缺少句末`cukup hangat`、英语共同词序变化。
Qwen也有低Whisper误差对照上退化的样本，不能替代Whisper作为真值。
plan `de667f522d4d051537b68b20b0838ea25db07e0588c141d111bcab0eff8c489a`；
summary `d5ee1d9d5cab25ce54f5b4cca884e27a1921a9df1330bf191ebbe8c8e40c3aaf`。
四路扩展至4985个当前文件（3975目标+1010仅ref）**已完成**，120复用+4865新增，全部EOS；
plan `034d813a04e244324e8b0f4b22a5cb562cfa3913bc52853f77183ce3917887f1`，
results SHA `8b5836e2b3101803c32864052fcb58be1afc2d9ecc1ffe1083b36d905bd70c7a`，4985份结果SHA核验通过。
不按Qwen结果选择输入、不自动改标或剔除。两ASR均CER>.05：当前目标ID303/1926、TH465/1243、
TL26/397、zh39/259、en17/150；仅ref ID64/670、TH74/340。这些仍是原始分歧，不是错标率，
包含数字、泰文重复符、`ํา`/`ำ`写法、同音字与专名拼写差异。原label与旧评分不改。

## 去背景音试验

用户提出模型移除BGM/杂音，已执行[torchaudio HDemucs](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.html)
24条固定试验：8个ID/3个TH共同疑似+上述1个ID ref；另各目标语种4个低风险对照。
模型针对音乐声源分离，未认证所有背景噪音都可去除；对照也不冒充已确认干净真值。
使用现有torchaudio2.6.0，官方334,697,255-byte模型SHA
`a004b2790d73ffeaa535db458a1a79b539dfdbafbccc31f275d07e632ebd7816`，安全权重加载、strict state dict。
16k源在内存升至44.1k，10s窗/2s重叠加权后还原原增益、再回16k；不裁词、压缩或峰值归一化。
只纠正重采样最多1sample长度差，保留原长度；同时记录纯重采样误差，避免把它误归模型。

`separation_pilot/` **24/24已完成**，峰值1118MiB；每条保存vocals与original−vocals残差，原音SHA不变。
plan `d0c9f2915cfe065b9a3acbe87e5a722d16a74f937b19a0e17a9f65f6e36da83a`。
所有派生音频`training_eligible=false`，继承原来源及历史风险，不通过换路径回流训练。

用户随后要求全部做人声分离，已四路完成**6027/6027**（24试点复用+6003新文件），覆盖本轮5015行
训练目标/ref的独立文件，非全池50321文件。`separation_all/` plan
`4d93c149d0103076fc60f552c5a7ac29f426dee0fddc9f8a7ca1d5d1533845ed`，
results SHA `010ab0bae9c59008c0b60df475566f654ef4c752b50d9f7cb4431dcfc5022ffc`。
每条保存FLOAT vocals WAV，原件/长度保留；新WAV共2,468,979,756 bytes。所有产物仍未获训练准入。

24组固定前后对照`separation_review/`已完成，plan
`24bc3cd8d4d9cb71a33007fda003527d35a72c3e6c04d88c8ddf0a5150fc4e13`。
Whisper large-v3 int8_float16、无VAD/期望文本提示、beam5/temperature0、不传递上文；Qwen同原设置，
复用已有原音转写，音乐与ASR并行。12条音乐疑似含11目标与1ref：

| 固定分组 | n | AST Music均值 前→后 | CLAP三组音乐提示条数 前→后 | Whisper CER均值 前→后 | Qwen CER均值 前→后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 音乐疑似目标 | 11 | .58138→.30629 | 11→2 | .00965→.01907 | .02197→.02668 |
| 音乐疑似ref | 1 | .71736→.00684 | 1→0 | .05556→.16667 | .05556→0 |
| 低风险对照，非干净真值 | 12 | .02129→.00801 | 2→1 | .00185→.00185 | .04293→.04326 |

12音乐疑似有10条分离后不再触发CLAP提示，但不认证真实配乐全部去除。TH `0006365.wav`分离前两ASR
都与文本一致，分离后Whisper CER .03448、Qwen .05172（各自差异位置不同），需保留疑似人声损伤风险；
ID ref的两个ASR变化方向相反，也不能认证哪一边是真值。12低风险对照Whisper逐条CER不变，但其中
ID `0001824.wav`与TL `0001237.wav`原音新触发CLAP提示，后者分离后仍触发。后续隔离须继承这些证据。
下一步对当前4985文件全量复核，不能凭24条将6027文件整体准入。原音/人声/残差试听仍待进行；
静音变多/分类器分下降不自动证明保留完整原人声，母语质量未验证。

已冻结并后台启动`separation_review_all/`，plan
`c4a83e287b37fd7354bc1e735e644d192d8fe81dd8b66833ac5018132ca8e530`。
四路按原SHA次序分片，每路依次AST/CLAP→Whisper/Qwen→配对汇总；已有4985原音ASR、在用12组试点
及原CLAP全部复用。覆盖原音与分离后的音乐残留，每文件中间结果落盘；不因失败重抽、不自动隔离/准入。
新汇总检查精确覆盖、重复输出、原音/分离音SHA、文本与语种条件，避免错误配对。53项相关测试通过，
py_compile、shell/diff通过；旧24组79份元数据SHA通过，comparison SHA
`e4190c685d032b7f621d321a27ae859e667c83554d698b6ef627cebaf7551231`。最终留出原字节未动。

四路音乐阶段已完成：4985个分离音频AST、9970个前后CLAP视图（均包含复用），逐窗聚合与覆盖复核通过。
原音CLAP提示1122/4985→分离后695/4985；451由正转负、24由负转正，仍671两侧均提示。
当前原音AST全部<.5，分离后6个ID目标≥.5。特别是原TL157/397得到CLAP单模型提示，
这些并非确认配乐或干净真值；可能包括模型/提示文本偏差，不能报告成“音乐污染率”或“去除率”。
保留新增疑似供下一隔离决策，不能仅因分离分数下降自动放回历史排除。部分报告在
`handoff_987d0f7/music_review_complete.json`，SHA
`d70b9c54a8ba9fe7af487d40b5c2ae5a0e984fc9689ec53eead8415e338068f2`。
23:58四路Whisper约1000/4985进行中、之后Qwen，GPU99%/约10GiB；显存剩余不等于计算空闲。
原音4985复核已完成与分离后4985复核未完成是两个不同状态；不将前者冒充后者。

Granary-v2的[官方数据卡](https://huggingface.co/datasets/nvidia/Granary-v2)明确只发布标注清单，没有音频，
更不是分离后的干声库。ID/TL是现有FLEURS、TH另有CV26，不能用全库463,062h替代三语种净新增估计。
已经借鉴的多视图文本与独立证据/拒绝原因记录继续使用；详见[既有接入记录](granary_cleaning_20261030.md)。

## 2026-10-31 自动接续：全量结果与派生子集

远端已确认原四路计算退出、GPU空闲；顶层4985唯一配对的comparison SHA
`2cbed6e54e239e4543b9d716f80d17a8fd66499ca838501094ecf8cd23754f90`，plan SHA
`c4a83e287b37fd7354bc1e735e644d192d8fe81dd8b66833ac5018132ca8e530`，与交接一致。
本地只镜像JSON元数据，原音/模型未下载；已有转写与旧评分不改。

| 原音语种（目标+仅ref） | 文件 | 分离后RMS降低>6dB | 分离后无能量边界 | CLAP提示 前→后 |
| --- | ---: | ---: | ---: | ---: |
| ID | 2596 | 94 | 16 | 807→466 |
| TH | 1583 | 21 | 4 | 110→68 |
| TL | 397 | 4 | 0 | 157→121 |
| zh | 259 | 0 | 0 | 24→20 |
| en | 150 | 6 | 0 | 24→20 |

例如review_id2835（ID ref）RMS −15.694→−81.639dBFS，原音两ASR CER均0，分离后
Whisper .7755、Qwen .9184；该结果足以否定“所有分离文件可直接替换”，不能用CLAP下降认证人声保全。
这些是文件级诊断，不等同真实背景音乐率或三语种母语质量。原音没有Qwen未EOS，分离侧2条
（review_id2874 ID、3068 en）未EOS，均保留为失败结果；不补转写、不从完整覆盖计数中偷偷移除。

新的`paired_quality_candidate.py`复用既有`quality_separation_review`、`measure`、`inspect_audio`、
`exclude_flagged`、`audit_mix`和`pcm_conflicts`，只准备新候选，不启动训练。
执行根`/root/autodl-tmp/quality_reaudit_20261030/paired_candidate_20261031_v2/`，冻结源码为同根
`paired_candidate_20261031_v2_source/`，nohup日志`paired_candidate_20261031_v2.log`。
先核验全部输入/输出SHA、四分片终态，并从原结果重建顶层行；随后4线程CPU检查4985个派生音的
VAD/能量边界/削波与原音PCM，成功检查逐文件保存，保留4GiB底线。

准入按新冻结policy执行：任何一侧AST≥.5或CLAP提示均为未校准疑似，保守连带原音/origin/PCM/ref隔离；
不能用分离后下降回收旧排除。内容诊断仅对NFKC视图与全文数字等价作格式处理，不改原台词。
两ASR残余CER均>.05、相同但非标签转写、共同疑似删除≥2字符或共同数值未解决都记uncertain隔离；
泰文重复符、同音字和专名仍可能导致保守误排，不认证真人错读，多个条件不当独立证据计数。
派生音新增转写分歧、未EOS、VAD/边界失败、>1ms削波、RMS损失>6dB或有声比例下降>.05时不采用；
仅在原音单独合格时回到原音，`rejected_derivatives.json`保留失败版本及SHA/PCM，不能再换路径混入。
金属分/频谱变化仅观察，不作门禁。

新候选保留历史flag，原895val字节不变、1634完整旧val和全部评测文本保护；
`audio_mapping.json`保留原路径/原音/派生SHA与PCM，身份、来源、台词及原ref配对不变，TL不造ref。
按时长请求ID/TH/TL各28.333%+zh10%+en5%，选可满足的有界总时长，最多2×原音；
`mix.json`报告请求/实际、独立原音和ref暴露，检查原池/派生池/训练混合三份隔离。
数量/小时/许可继承明细与步数建议以完成后的冻结报告为准；不能先把计划称为完成或质量通过。
数据筛选、语言配比和部分分离同时改变，后续实验只验证组合候选，不作单变量因果结论。

首版`paired_candidate_20261031_source/`在候选写入前因将未EOS错误当作缺失工作而停止，
源码和`paired_candidate_20261031.log`保留。核对发现原始行重建完全一致，失败来自两条已完成未EOS，
因此v2将其归入派生失败证据；没有任何新ASR/TTS抽样，也未改原comparison或两条失败输出。
22项最新针对测试通过（包括SHA变更、分片未完成、原始结果不一致、未EOS保留、数字值变化、
Unicode、共同漏字、原音回退、2×暴露）；此前数字/旧隔离测试通过，shell/compile/diff通过。

**终态已完成**：4985对原始结果重建与SHA核验通过，4985派生物理检查完成，4998份产物SHA独立复核通过。
新增2144源文件疑似，经原音/PCM/ref传播后3975→1947行；1150派生版本单列拒绝，原音声学复查0新增flag。
分离输出另有65个峰值超出可用幅度范围，未进行削顶修复，不采用这些派生版本。
新训练1881行/3.541h，ID/TH/TL各约28.3%+zh10%+en5%，1637独立原音、最大2×；
ref475次/435独立、最大3×，TL ref0；目标1522派生+359原音、ref408派生+67原音。
三份隔离审计全部通过、val原字节不变，许可/逐源小时详见`runs.md`对应节。
plan SHA `60789ed48d2c4c4c4cec3126c3f7c787b1fb75dc52d186fe7b82e434f2781a82`；
completion SHA `ed647b04198e11da6ee2a7ce1ea1aac8d3c49a7672dad58b2d1d3d6199f56b28`。
独立复核在同根`paired_candidate_20261031_v2_verification.json`，本地元数据镜像
`work/quality_reaudit_20261031/candidate/`；计算进程已退出、剩余约5.86GiB，未清理任何原音或唯一模型证据。

接续r35已使用既有分阶段driver，入口`staged_paired_candidate.sh`，执行根
`/root/autodl-tmp/r35_dit_paired_20261031/`。独立`source/`、`driver.log`和新plan，
先重核准入completion全部输出SHA、所选原音/派生音字节和原origin映射，再冷启动；不重复ASR或分离。
保留DiT-only/rank64/alpha64/dropout.05/lr1e-4/batch2×8/wd.01/clip1；
按1881行缩到118更新/名义1.00372epoch，warmup12，40/80/118更新后等待同输入五seed质量门。
官方保存回调只在声明阶段写完整快照，非阶段调用不写盘；官方submodule未改。
每阶段TH1844 control全部5seed和TL1913 control全部5seed前置，原风险与保护样本保留，45条上限。
小样本平均CER按原CER_NOISE=.05记观察；新空转写/近静音/疑似多读、两条严重CER差异、
absolute尾部/时长/有声比例门仍有效。38项针对测试、shell/diff通过，实际执行终态与记录见TODO/runs。
清洗、混合、部分分离共同改变，不认证单变量因果。最终90文本未使用，母语未验证，无生产变更。

r35实际40更新后在TH1844 control五seed处因时长/有声比例门拒绝，未跑到TL1913。
平均时长5.056→5.984s、有声比例.88310→.81664；CER .254%→0%、尾p90 .14→.18s均不豁免声学失败。
CPU重查原WAV采样数、同算法声学指标、96个optimizer状态全40与5份输出/完整snapshot/latest SHA通过，
18,447冻结输入复核；verification `1237bce3abfa8abcb16564d68aefaa1ab9fbc29d5816d33fbb0ca4e0df475068`。
s43/s42内部低能量段延长，不是只多留尾部；低能量观测不当语言自然度判断，也不证明分离导致失败。

下一有界r36仅改变训练音频版本：按`audio_mapping.json`把同1881行目标/ref换回已筛查合格原音，
逐行与`original_pool.jsonl`完整元数据比对，保留原行序、重复次数、文本、身份、配对与有效时长；
所有原音在准入阶段均独立通过，旧疑似隔离不回流。不改变r35的超参/118步调度/阶段门或推理条件。
目的为验证分离敏感性，不预判原音一定更好；单训练seed不认证普遍因果。
入口`staged_original_candidate.sh`、根`r36_dit_original_20261031/`，39项针对测试通过；终态见TODO/runs。

r36也在40更新、TH1844 control五seed后拒绝：CER .254%→0%，但时长5.056→5.888s、有声比例.88310→.81412；
尾p90 .14→.18s。原音回退未消除该文本的失败，不能据此认证整池分离无影响或原音更差。
18,476输入、5WAV/claims/items、完整快照/latest与CPU原WAV声学核验通过，96个optimizer状态均40；
verification `6d5ee04be14fe470725ce6ec89045796176e34c1103646dbb58b55a0a92dd7c2`，未评分WAV0，GPU已释放。
下一计划固定r36原音清单和118更新调度，仅减半学习率；r34旧数据5e-5的TL尾失败仍须保留风险前置，
不因某个TH样本通过跳过TL。r37尚未启动；最终留出未用，无生产变更。

r37本地准备已完成：现有driver新增`dit_original_lr5e5`，绑定r36终态与六份输入的字节一致性，
配置除路径与lr1e-4→5e-5外必须相同；32项针对测试通过。SSH先成功核验r36终态与空闲GPU，
后续3次因主机DNS解析失败退出。未执行checkpoint回收、未远端冻结r37/开训；本地候选配置明确标为预备态。
自动退避600秒后重查SSH，恢复后先核磁盘与重复checkpoint的SHA/打开句柄，保留路径和唯一证据再启动。

## 执行卫生

所有长任务独立source/plan、nohup日志、原结果不覆盖；本地只镜像元数据，不下载数据集音频或模型。
隔离器准备曾遇到rsync未结束即启动和四位输出ID字典序问题，均在0候选产物时失败；旧源码/日志保留，
等待传输完成并按数值校验完整ID集合后另冻v2执行。ASR首版在0次模型推理时遇到CUDA API命名错误，
另冻v2修复，原plan不改；并非丢弃低分后重跑。对应边界、数字反例、PCM/ref传播、重叠加权有测试。

为保留4GiB磁盘底线，核验9份未使用THAI-SER parquet缓存与官方revision
`90440337b57fbaea18c54f762de1ac47c72948b3`的字节SHA、55,766份元数据零引用与零打开句柄后，
仅删除这9份下载blob和链接，回收4,478,208,969 bytes。5份raw manifest SHA不变，未删原音、模型或checkpoint；
证据与可恢复版本/hash在`storage_dedupe_20261030/thaiser_cache/`。全量分离后约6.19GiB可用。
扩展复核的临时/缓存目录已在模型工作前显式建于大盘；仓库入口也补齐mkdir，避免新实例因TMPDIR不存在
回落系统盘。已执行的冻结入口字节保持，运行以plan内SHA为准。
