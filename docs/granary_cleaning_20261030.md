# Granary-v2：可借鉴的清洗逻辑与 r31 数据试验

用户要求实际试用 Granary-v2、允许放宽尾静音，并明确可以借鉴其清洗逻辑。
数据卡固定 revision `81402b751b451b5fe4f4c29e8b40f92892f1e440`；公开 Curator
固定 commit [`8a9a217c90127d424a8f503cb948698d7203002a`](https://github.com/NVIDIA-NeMo/Curator/tree/8a9a217c90127d424a8f503cb948698d7203002a/tutorials/audio/granary_v2)，Apache-2.0。
本项目只复用思路和既有依赖，不安装整套 Curator，不修改官方 VoxCPM submodule。

## 原文核对与取舍

| 上游实际逻辑 | 本项目采用方式 / 限制 |
|---|---|
| metadata：原音频ID、分段起止、RTTM说话人数、声音事件、逐模型LID、拒绝原因各自保留 | 恢复FLEURS原文件名与现用录音的PCM裁切对应，保留来源/录音哈希。模型说话人数不认证同人，不凭caption建ref |
| SED的0.5仅标记声音事件，不自动过滤 | 沿用已经完成的AST音乐排除；不将CNN14的0.5搬到AST，也不据此认证无BGM |
| 非Indic语言通常要求SpeechBrain和Whisper同意；英语有单模型特例 | 借鉴保留各证据与分歧。此次不新增LID模型、不搬英语特例、不以LID当口音门 |
| ASR幻觉：词汇多样性≤0.4、长词、常见幻觉短语、字符速率>40；TH等无空格语种跳过词检查 | 在本轮FLEURS接入范围检查已完成的无台词提示ASR：不同于原文且≥8词、低多样性，或>40字/s标疑似。TH/zh不按空格判重复/长词；不直接拉黑“谢谢观看”等可能真实台词，不使用未校准长词阈值 |
| SelectBestPrediction可用fallback或reference清除幻觉标志；跨模型一致默认80%，TH走CER | 保留原始证据；**不自动清除历史隔离**，不将80%一致直接当TTS合格。未重新调用第二ASR |
| `tn_raw`是口语数字表达，`itn_raw`保留用词但改成书面数字/缩写 | 两种视图分别比对原文，记录哪一种一致；保留原始TTS正文，不靠全文LLM重写“纠错” |
| `itn_clean`删填充词、重复词、假起音；caption/QA/translation是派生任务 | 不用这些字段认证稿音一致或充当TTS正文，避免音频仍说了而标签删了 |
| 文本管线只读文字，实体恢复可使用reference；模型改写有守卫和回退记录 | 不把Granary与原文一致称为独立ASR互证；不将文字改写当已听懂音频 |

源码定位：`metadata/README.md`“What each stage does”、`qwen_omni_asr/pipeline.yaml`、
`text_pipeline/README.md`，以及 `nemo_curator/stages/audio/text_filtering/{whisper_hallucination,select_best_prediction}.py`。
公开教程没有给出可完整重建此发布版 `high_quality` 的最终统一逻辑，不能把本项目规则声称为官方门槛复刻。
Qwen教程显式模型为30B、配置2 GPU，支持列表没有TH/TL，不能假定本机48GB可直接覆盖三语种。

## 已取得的接入证据

- 三语种原始FLEURS train TSV共7065行；按原始文件名字典序排列，与旧raw manifest的全部文本及序号一致。
  Granary行按文件名连接；同文本不单独证明同录音。固定FLEURS revision为
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`，原音频CC-BY-4.0、Granary标注OpenMDW-1.1。
- 三语种各3条原始音频抽样，现用处理后PCM均是原始PCM16的精确裁切；随后启动三份完整archive的流式SHA与
  当前1504条FLEURS音频逐条PCM核验：**ID516/516、TL890/890、TH98/98全部通过**，三份完整archive
  SHA均与固定revision的LFS声明一致。只在远端流式读取，完整archive不落盘，不加入新音频或回流旧排除。
  ID核验240.9s；之后TL元数据请求503，旧成功产物保留，显式接续TL108.3s、TH133.8s，未重做ID。
- 当前ID516/TL890/TH98条中，Granary高质量标记分别322/426/30，共778条。
  仅按`tn_raw`归一化，657相同，121有分歧；与既有Whisper比较9条误差少>.02、105多>.02、7相近。
  **该Whisper已经用于筛选现有候选，存在选择偏向；数字表达也会影响误差，这不是标注优劣认证。**
  因此最终规则把保留用词的`itn_raw`单列比对，不能将上述121条全部直接称作错标。

## 冻结的试验范围

入口 `scripts/granary_candidate_run.py` / `.sh`，远端独立根 `/root/autodl-tmp/granary_pilot_20261030/`。

1. 只从最终 `music_excluded_20261029/run/candidate/current/` 选择；继承全部路径/原音/PCM/ref排除，
   保护完整1634条历史val以及全部评测文本。当前验证895条保留字节不变，后续不因Granary标注回流。
2. 当前FLEURS行要求官方HQ=true、原文与`tn_raw`或`itn_raw`归一化一致、原录音PCM精确裁切验证通过；
   ASR新启发式异常也排除。其他来源沿用当前候选，不从全池扩样。保留行原字节、原顺序、真实source_id不变。
   所有新排除均记uncertain，不认定已确认坏音频；无新speaker/ref/control标签，无新台词，无新音频。
3. 沿用r30原基座冷启动、seed42、459步、batch2×8、lr1e-4、r/alpha64、dropout.05、LM/DiT开、proj关。
   选择变化会改变语种份额与有效epoch，**这是Granary筛选组合试验，不是每项规则的独立因果实验**。
4. 复用冻结base/r26/r28/r29/r30，新增r31的180条件×seed42–46=900条，四路；同CFG1.8/20步/large-v3，禁重试。
   新尾静音策略absolute：p90>0.5s仍否决，较base+.10s只观察；内容/多读/语种分开验收。
5. 核验与预检不全则不启动。无自动续训/重试/生产晋级；实验结果不等于母语验收。

当前结论仍是：Granary可以作选择与复核信号，不能把46万小时视作三语种新增真人主力；
ID/TL来自已有FLEURS，TH的CV26访问条件、录音重合和质量尚未核清。

## 本轮执行取证

候选保留5015条/9.4815h，FLEURS731条（ID304/TL398/TH29）；隔离773条，原件保留。
原因重叠计数：HQ非true721、标注缺失5、转写分歧232；其中新增于非HQ/缺失之外的分歧为47条。
`itn_raw`使74条在书面形式下可对上，不使用itn_clean取巧；ASR新增启发式0命中，不能据此声称它改善质量。
数据时长份额ID55.207%、TH20.463%、TL14.210%、zh6.260%、en3.860%；ref1218、TL可信ref0、控制0。
同459步对应1.464407epoch；计划SHA `68cb179fadf983ff130763b07e0956aee0240876062f9cabbc72970111cc7e58`。
11,334输入SHA通过，保留行/val字节一致；67份核心源码仅runlog判定记录模块变化，模型逻辑未变。
10:28:06启动、10:41:07完成459步，wandb `kobfcz1t`，val1.072322→.962661→.899426；900条评测已启动。
LoRA SHA `065d98abc74f97ca32385a58ebf580d99ab5e8d8d02d19ea5576294cc9ed087e`，不凭loss下降晋级。
47相关及422全量测试exit0，保留两条既有依赖弃用警告。远端`startup_verification.json`另冻结
元数据获取、原音PCM核验脚本和报告SHA；本地镜像在`work/granary_pilot_20261030/`，无源音频。

## 当前停滞的证据边界

r30并非各方面都没有进步：相对base，TH CER从5.65%降至4.69%；但相对r29，三语种CER配对区间均跨零，
这次更彻底清洗没有带来明确增益。旧自动否决主要来自+.10s尾静音增量，按新偏好已降为观察。
val loss来自≤10随机batch，且r29/r30验证集不同，不能据它断言配音效果必然继续改善。
TH时长17.7%、TL可信ref0、可信控制0仍是覆盖不足；未证明它们就是唯一根因。
LoRA没有直接训练停止头，但梯度可经LM传递，不能由“停止头冻结”断言无法学停止；
96条训练对齐诊断多留约56–60ms也不足以解释完整180ms输出差异。暂不改停止逻辑或加步数。

只读复核还发现ID1816的seed42/43/44在base、r29、r30的ASR里都缺括号正文，
base的CER均.4375；不能将这类现象全部归因于r30清洗或尾静音。正文括号可能与控制形式冲突，
但当前代码核对没有发现直接删括号正文的解析步骤，ASR也未独立证实真人漏词。
后续需固定字词、只改括号标点的多seed对照；旧输入/报告不因此修改，不能先按猜测修模型。
