# 推理链路核查：源码、归档与实际部署分开

## 范围与结论

初始只读检查以下两个本机checkout；下列HEAD/干净状态与源码哈希为审计起点。
后续OmniVoice最小改动见「最终采样参数追踪」「数字词化透传修复」，均已push、未部署。
随后数字输入实测发现重复币种词，已在DIS/OmniVoice两侧本机修复，
独立r8诊断125/125及追加16次ASR复核均于2026-09-18完成；
见`numeric_probe_20260918.md`，不把初始只读状态当当前状态。
重复币种修复提交DIS `1a1e309`、OmniVoice `ac1e51d`尚未push或部署；
20/20非数字控制WAV一致，VI有局部改善信号，MS受币种转写与共同评分边界混淆，
不能据此确认退化或正确发音。不是HTTP端到端或完整语言验收，不改指定r8或追加训练。
后续币种/分隔符边界修复也已本地提交：DIS `9a961b2`、OmniVoice `88d9dd2`，
均未push或部署。两侧原21项输入各19失败，修后全过；扩展198/254项通过。
旧29条payload只有`id_digit_3`停止局部词化`06.15`，不是正确时钟发音的证明；
原125条报告/音频及冻结评分器未改，没有重评分。工程与数据证据见同一诊断文档。

- OmniVoice：`/Users/dunxu.liu/workspace/others/OmniVoice`，
  HEAD `b92fb28f894071f0bf06ffe1e13bab4654c30a28`，工作区干净。
- DIS：`/Users/dunxu.liu/workspace/dubbing/dubbing_intelligence_service`，
  HEAD `638779e1e3792552f3b05887eca28d44278ff173`，仅既有未跟踪 `docs/reports/`。
- OmniVoice `api.py` SHA256：
  `a268801e4d38a470d1e35fe4094e506a2997f0c24c96454f4857ac7abdcf5526`。

**D03是本项目模型输出的保真双句拼接，不经过这两仓的生产处理。**
因此不能用D03模糊证明API/DIS后处理有错；也不能把生产链路差异当作需要微调的证据。
两仓当前源码不等于线上镜像版本。指定生产基线继续r8/strength1/CFG1.8/20步，
但线上实例的实际权重身份未核实，HF发布成功不能代替加载证据。

## 当前源码对齐

以下OmniVoice行号均指`api.py`；DIS后端指`src/dubbing_intel/core/tts/backends/voxcpm.py`，
渲染指`src/dubbing_intel/core/render/render_dub.py`。

| 环节 | 核查结果 | 对本项目的影响 |
| --- | --- | --- |
| 加载模型 | OmniVoice:3004的唯一加载函数按服务端`_VOXCPM_MODEL_ID`调用`from_pretrained`，未传LoRA路径 | 请求中的model_id/HF仓库状态不等于加载了r8；还需部署版本和权重哈希 |
| 数字文本 | DIS后端:790、1538在normalize开启时调用`normalize_clone_numbers`；`tts_text.py:471`对vi/th/id/ms已有词化。API helper:8130也有数字归一化 | 旧“这四语种数字原样入模型”不能作为当前主调方事实；旧压力case保留，不改冻结评测文本或历史数字 |
| 请求步数 | DIS后端:922起普通首轮请求10步、弱ref20步，并传`voxcpm_steps_auto=True` | 这是请求值，不是实际采样值，不能因此把本项目20步降10 |
| 自适应 | OmniVoice:7753使用`min(requested_cfg, profile_cfg)`和`max(requested_steps, profile_steps)`，显式override另受保护 | CFG1.8不会被旧profile升到2.1；auto步数可上升，部署cap和后续重试还会影响结果 |
| reference | OmniVoice:8693–8800有quiet/peak、条件WPE、active-level、非语音/边缘裁剪，helper另传模型VAD | 请求ref字节与最终模型输入ref不必相同；须分别取哈希和处理记录，不能只比请求路径 |
| 续写 | DIS渲染:14855设置前一条成功take的prompt；后端:1557读取其音频/文本并抑制style | 不是始终reference-only；独立生成两条再拼接的D03不覆盖续写模式 |
| API输出 | OmniVoice:9440附近至9524的finalize有裁静音、可选speed、max-duration、noisegate、active-level和peak ceiling，再输出PCM16 | API的“raw WAV”不是模型裸输出，不能与本项目裸WAV直接做后处理归因 |
| DIS输出 | 渲染:17600起有短cue保真分支、裁边/停顿压缩、rubberband/atempo、时间窗裁剪、局部spike修复、条件高通/高频shelf、动态增益和条件limiter，后续还有轨道/整片混音 | 需把API返回、段级输出和最终混音分开保存；代码有某滤镜不等于本次实际启用 |

轻量执行直接取当前`_voxcpm_adaptive_params`函数，固定一个CFG2.1/20步的profile替身：
输入1.8/10、允许auto时输出1.8/20；将两项都设为显式override时保持1.8/10。
这验证分支语义，**不是对D03 ref或某线上请求实际profile的测量**。

续写还受key、间隔、链长、超短句、情绪/强度边界等门控，并非每句启用。
`_record_voxcpm_take_continuation`（渲染:14927）记录`cue.raw_path`及实际说话文本；
它是API返回后、DIS段级处理前的take，不是模型内部裸输出。
严重/伪影标签、伪影重试、时长与音色相似度等会阻止入链。
只读确认这些门存在，不宣称能拦住全部伪影。

## 初始定位的两个契约缺口

1. **HTTP漏传`number_verbalization`**。helper在OmniVoice:8134、8149读取它，
   但`synthesize_voxcpm`的`gen_kwargs`（:8869）只传`spoken_text_normalized`，没有该字段。
   AST核查确认缺失：直接测helper支持False不代表HTTP调用能关掉数字词化。
   D03文本无数字，不能拿此缺口解释这次模糊；对后续数字单变量实验必须先接通并验证。
2. **旧回显CFG/步数可能不是被接纳take的参数**。质量重试接纳分支（原:9235）
   更新`gen_kwargs["cfg_value"]`/`["inference_timesteps"]`；
   返回的`adaptive_params`（:10032）却仍引用初始局部`cfg_value`/`inference_timesteps`。
   因此有重试时不能把这两个回显值当最终有效参数；`effective_seed`的存在也不补齐CFG/步数。
   后续已新增独立追踪字段覆盖候选接纳/复用分支，保留旧字段的初始参数含义；见下节，尚未部署。

上述审计为静态调用链与轻量函数检查，没有为复核执行线上合成、下载模型或跑新网格。

## 最终采样参数追踪：已提交推送，未部署

本轮先使用现有`OmniVoice/scripts/vast.py`的GET封装，只读查询当前环境Vast凭据下实例清单：
返回实例总数0，`omnivoice-api`前缀实例0。本机环境未设置`DUBBING_API_URL`或`VOXCPM_CLOUD_URL`。
这是该凭据/环境的当次观察，不证明其它账户、页面显式地址或其它部署不存在。
没有可据此核验的在线模型，未新建/启动/停止实例、未warmup或调用合成，线上权重身份继续标未知。
脱敏观察归档在`checkpoints/holdout_eval_20260917/inference_tracking/availability.json`。

为避免继续把初始请求参数当最终take的参数，仅在OmniVoice增加记录：

- 响应新增`adaptive_params.effective_generation`，含`cfg_value`、`inference_timesteps`、`seed`、
  `normalize`、`denoise`、`trim_silence_vad`、`retry_badcase`和`generation_attempt`。
  它描述生成调用参数，不冒称是处理后ref、词化后文本、实际去噪执行证明或权重指纹。
- 复用已有每请求`generation_budget`，在互斥内成功生成后记录实际调用参数与有效seed，
  失败不继承上一条记录；每条候选保存自己的快照，仅在音频被采纳时切换最终记录。
  质量、漏音、身份、文本重试拒绝后仍保留原音频的记录；异常兜底也从真实调用处记录，
  不拿外层`candidate_kwargs`猜兜底参数。
- 次选复用取该次选快照，不使用“最后一次调用”的参数。新缓存保留原音频的记录；
  `generation_attempt`是原请求内的调用序号，含失败调用，不是命中缓存时新花费的次数。
  旧缓存缺字段或记录为null即未知，不为了填字段重新生成或伪造为请求值。
- 旧`adaptive_params.guidance_scale/num_step`保持初始参数含义，兼容现有消费者。
  DIS原有`server_generation`保留完整`adaptive_params`，无需改代码；已直接执行其原始
  字典表达式验证新字段保留、旧响应不补值。此检查不是在线端到端请求。
- 不改变CFG、步数、seed策略、候选评分、重试预算或缓存键，不加载模型、不改r8。
  当时没有把`number_verbalization`透传或模型身份混入记录改动；前者后续单独修复，后者仍未验证。

验证：

- 改动前相关基线19项通过；新增后重点86项通过，扩展相关集**241项通过**。
- 将返回字段临时置空，8个新路由断言全部失败；恢复后上述241项通过。
  覆盖质量/漏音/文本重试采纳与拒绝、异常兜底、三候选中复用非最后一条，
  以及既有身份重试采纳/拒绝、新缓存记录保留和旧缓存缺记录。
- 测试通过受控模型/QC替身核对参数归属与生成次数，不是GPU音质、完整语言验收或线上部署验证。

OmniVoice改动限定`api.py`、两份既有测试和接口文档。
2026-09-18按用户要求提交并push到`origin/master`：`30211d9`；
整合远端已有的两条ASR提交，没有覆盖或强推。整合后相关66项测试通过。
该仓库master push触发既有GPU镜像构建流程，**不等于部署到GPU服务或已在线验收**；
没有启动/替换实例，生产指定模型仍为r8。数字开关透传随后单独提交，见下一节。
本机补丁与测试记录归档至`inference_tracking/`，只作证据备份，不安装到微调服务。

## 数字词化透传修复：26829e1，已push、未部署

2026-09-18先完成两仓已有改动的push，再继续该已定位契约缺口。
根因不止HTTP `gen_kwargs`丢字段：prompt缓存构建也没有传数字覆盖值，
voice注册表的prompt缓存参数亦未区分该开关。只补HTTP一行会留下续写路径错误。

- 请求`number_verbalization`按既有布尔解析统一为true/false/null，透传目标文本与续写prompt；
  缓存与非缓存生成走同一normalizer约定，切换开关不会复用另一种数字读法的prompt缓存。
- 只影响`normalize=true`下vi/th/id/ms的现有数字词化开关；省略/null仍沿用normalizer配置，
  zh/en/tl规则不改。`spoken_text_normalized=true`仍不再改目标文本，不能撤销DIS已做的词化；
  prompt仍正常归一化。`normalize=false`不执行该步骤。
- 结果缓存算法版本8→9，隔离修复前忽略开关生成的音频；没有删除旧产物或更换seed作伪对照。
- 被接纳take的`effective_generation`追加`number_verbalization`与`spoken_text_normalized`；
  null表示沿用normalizer默认，不是false，也不冒充实际词化执行证明。
- 不改变默认CFG/步数/seed/候选选择，不改DIS，不加载真实模型或启动训练。

先补失败回归，修复前已实测vi/th显式true被丢弃、输出仍是原始数字。
修复后17项定向检查通过；扩展合成/locale/身份/缓存/契约集**223项通过**。
其中16种组合覆盖四语种、缓存/非缓存与两种默认配置，每组覆盖true/false/字符串false/null/省略，
核对目标/续写实际模型输入、开关回显、prompt缓存切换与结果缓存复用。
使用真实normalizer与模型替身，不是GPU音质、数字发音或母语验收。

修复已到OmniVoice `origin/master`，提交`26829e1`。既有镜像构建流程会触发，
没有部署到Vast/AutoDL服务，也没有核验线上实际权重或效果；生产指定基线保持r8。
后续若验证线上数字A/B，仍须先对齐实际部署、是否上游已词化、同次输入/ref和采样参数，
不能仅传此开关便宣称完成端到端声音对照。

## 旧归档不能拼成D03单变量对照

既有归档`results/voxcpm_quality_20260909`与`results/voxcpm_quality_bestof_20260909`：
前者48响应、后者6响应，共54响应记录60次生成；54主WAV仅53个独立SHA。
109个WAV包含54主文件、54响度调整试听副本和1个`preview_rubberband_1.2x.wav`，
不能把每个文件都算独立生成或当成模型裸输出。

旧请求仅两条文本，与当前150条新文本、局部40条音频及D03均无文本交集。
旧ref SHA256 `7645c946617f844c2bb78d4eea0ec41aebe70a4847e2f829199021aa6d1fc781`
匹配当前`prod_ref_fil.wav`，不是D03的`ref_tl_03.wav`。
旧归档有ref增益/输出裁剪/peak-limited记录，但缺部署代码/权重哈希、最终模型ref及同次生成pre/post波形。
故不能复用它们声称“模糊由某次后处理引入/由r8修复”。

## 下一步：先补可归因记录，不扩大训练

1. 优先补**实际模型身份和新参数记录的在线验证**：部署commit/模型文件哈希、
   实际text/normalization/prompt模式、原ref与处理后ref哈希、最终seed/CFG/steps、候选ID及cache状态。
   本地最终调用参数追踪已完成，DIS无需改动；部署验证、最终输入追踪与权重核验仍为待办。
2. 最小阶段对照必须取**同一次生成**的模型裸WAV→API最终WAV→DIS段级WAV；
   各阶段保留哈希、采样率及实际滤镜记录。不要换seed重生成再称只改后处理，
   也不要仅改request_id以为能绕过缓存。先核验服务端版本与实际缓存策略。
3. D03的4条原单句/ref补听已反馈均无明显异常；初次双句模糊记录保留，原双句未报告复评。
   后续r10/r11离线小实验已完成，未得到可晋级候选，详见`runs.md`；
   不再等待补听、不追加训练、不换全局CFG/步数；
   不要求drama或母语评审来完成上述工程记录。
