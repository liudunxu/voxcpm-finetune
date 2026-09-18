# 微调运行记录

每轮一行摘要 + 一段明细，**按时间倒序追加**。字段从 configs/、mix.json、训练日志、
冻结清单和 eval 报告自动抽取；单模型可用 `python -m voxft.train.runlog`，同轮 A/B 合并记录。
不要手改原始数字或把不同 case 集混成一张验收表。

**2026-09-18收尾：暂停GPU实验，不开启r12。**
r10/r11均未得到明确晋级收益，现有数字诊断和ASR复核也已结束；保留r8，不加epoch或扩大网格。
有新的可复现坏例，或可追溯数据与明确增益假设后，再冻结回归和单变量实验；不是永久停止微调。
11:35:55（北京时间）实查无训练/评测进程，GPU显存0MiB、利用率0%；未代用户执行关机。
两侧工程修复已push：OmniVoice `88d9dd2`；DIS保留远端`25592cd`并合并至`be6e735`，
合并后69项相关测试通过，无强推，既有未跟踪文档未动。本项目本次定向3项及全套151项通过。
尚未部署或完成线上权重验收；后续工程核验不需要保持这台空闲训练GPU运行。

2026-09-18 r11之后继续的是**r8数字输入诊断，不是r12微调**：
29条实际payload审计定位重复币种词，两仓修复通过138/247项测试，
**11:05:09完成125/125条**（raw60/words60/legacy_ms5），20/20非数字控制WAV与ASR一致。
r8 strength1/CFG1.8/20步/同ref/禁重试，seed42/43/44/45/49；每语种仅2独立数字文本。
共同参考CER raw→words：TH .053480→.046337、VI .421026→.049487、
ID 0→0、MS .172332→.302217；不比较参考文本不同的原始CER。
MS重复币种输入修前→修后共同CER .613793→.275862，但转写不直接证明音频重复词消失。
11:08:34追加16次ASR复核完成：同8WAV、零温度、指定MS/auto；5/8转写变化，其中2条RM→Rp。
另发现冻结评分器对MS `Rp45,000`部分词化，MS共同CER不能当金额/币种发音的裁决。
**VI有局部改善信号，MS保留风险但不可据此判退化；不扩网格、不加训，生产指定r8不变。**
编码故障发生于60条后，仅恢复65条；UTF-8回归修复后本机相关28项、远端全套143项通过。
125WAV与冻结源码/失败恢复证据已备份本机；原报告/分数未改。
随后已完成币种/分隔符纯文本修复：两侧各21项回归，旧代码各19失败，修后通过；
扩展DIS 198项、OmniVoice 254项通过。原29条payload仅`id_digit_3`改为保留`06.15`，
未新增时钟读法，旧冻结评分器、报告及125WAV未改、未重评分，没有新GPU/ASR或微调。
两仓提交DIS `9a961b2`、OmniVoice `88d9dd2`已push，缓存/签名失效已验证，仍未部署。
明细见`numeric_probe_20260918.md`；下一步为可核验服务的版本/权重/最终输入与缓存检查，
不以输入工程修复代替模型或母语验收，无新收益证据不扩训。

用途：下一轮微调的起点参照、周报汇总、以及「这个结论是哪一轮、用什么数据得出的」回溯。

**当前口径（2026-09-18）**：指定生产基线保持r8；r10/r11均不转正、不扩训。
当前CER门槛是Δ>0.05，工程检查不等于完整语言验收；没有drama或母语评审不阻塞离线实验。
下方r9及更早记录中的0.005、必须等drama、投影路线定论等是历史表述，不能覆盖当前约定。
本次补录重新校验配置、训练清单及历史基线指纹；结构化抽取备份为大盘
`runlog_backfill_20260918.json`，原始报告未改。

## r11：lora_omni5_r11_control / lora_omni5_r11_tl_short

- 实验完成：**2026-09-18 10:20:18**；补录：2026-09-18。
- 结论：**不通过候选晋级：短句增配无明确收益；不加epoch、不扩大该剂量，生产保持r8。**
- A为原抽样，B仅提高FLEURS TL源内3–8秒录音时长份额7.98%→25.03%。
  非TL的4581条训练记录逐条相同；从同一官方base新训，不续训r8。
- 下一步：保留r8较短TL内容/停顿风险；只在有可追溯短对白或稳定坏例证据时冻结下一轮单变量实验。
  随后数字归一化HTTP/prompt缓存透传缺口已在OmniVoice `26829e1`修复并push，223项相关测试通过；
  未部署、未改变默认生产参数。这是工程契约修复，不是新的微调轮次或模型质量验收。

### 数据、配比与超参数

| 项目 | A：control | B：tl_short |
| --- | ---: | ---: |
| 目标录音条数 / 小时 | 5164 / 12.0257 | 5282 / 12.0251 |
| 实际更新步数 / 名义epoch | 322 / 0.997676 | 322 / 0.975388 |
| val loss首→末（step0→321） | 1.015644→0.877042 | 1.018281→0.883851 |
| 最优val loss | 0.869957 | 0.883851 |
| target最大曝光 / ref / 控制前缀 | 1 / 0 / 0 | 1 / 0 / 0 |

共享：LoRA r64/alpha64/dropout0.05、lm/dit开启、proj关闭，lr1e-4，
batch2×累积8×1GPU=16，warmup32、weight_decay0.01，训练seed42。
固定更新预算不保证shuffle后实际消费音频时长/顺序相同；单训练seed仅支持抽样政策pilot。
目标时长权重：TH=FLEURS5/YODAS3/GS2 9，VI=FLEURS7/GS2 10，ID=FLEURS7/GS2 10，
TL=FLEURS17、MS=FLEURS17，ZH=FLEURS10、EN=FLEURS5，回放全局共享。

实际来源（target小时数；许可沿用本轮已登记来源）：

| source_id | 许可 | A条数 / 小时 | B条数 / 小时 |
| --- | --- | ---: | ---: |
| fleurs_en | CC-BY-4.0 | 262 / 0.604211 | 262 / 0.604211 |
| fleurs_id | CC-BY-4.0 | 300 / 0.843067 | 300 / 0.843067 |
| fleurs_ms | CC-BY-4.0 | 668 / 2.041878 | 668 / 2.041878 |
| fleurs_th | CC-BY-4.0 | 218 / 0.604258 | 218 / 0.604258 |
| fleurs_tl | CC-BY-4.0 | 583 / 2.042881 | 701 / 2.042275 |
| fleurs_vi | CC-BY-4.0 | 350 / 0.841353 | 350 / 0.841353 |
| fleurs_zh | CC-BY-4.0 | 503 / 1.204017 | 503 / 1.204017 |
| gigaspeech2_id | Apache-2.0 | 672 / 1.201408 | 672 / 1.201408 |
| gigaspeech2_th | Apache-2.0 | 613 / 1.080425 | 613 / 1.080425 |
| gigaspeech2_vi | Apache-2.0 | 735 / 1.202044 | 735 / 1.202044 |
| yodas_th | CC-BY-3.0 | 260 / 0.360161 | 260 / 0.360161 |

### 分语种离线诊断

固定seed42/43/44/45/49、CFG1.8、20步、禁重试、large-v3。
以下单元格依次为 **CER / WER / 疑似漏尾率**；`—`表示不适用，不是0或验收通过。

短句：14条TL文本×3ref×5seed，各模型210条，合计840条：

| 语种 | base | r8 | A | B |
| --- | --- | --- | --- | --- |
| TL | .0156 / .0732 / .0000 | .0215 / .0909 / .0095 | .0250 / .1045 / .0000 | .0246 / .0974 / .0000 |

按14条源文本聚类、5000次配对bootstrap：B−A CER −.000472，
95%区间[−.003751,+.002768]；时长+.104381s，区间[+.060190,+.144762]s。
短句均时长base/r8/A/B=4.5691/4.8693/4.8770/4.9813s；
A/B相对base尾静音p90 .20→.36s，触发现有相对门禁。
对ref相似度的提高不是两两跨cue音色稳定性；未追加WavLM全量扫描。

五语种冒烟：每语种3句×5seed，A/B新生成各75条，base/r8复用**相同case/seed**各75条：

| 语种 | base | r8 | A | B |
| --- | --- | --- | --- | --- |
| TH | .0219 / — / .0000 | .0344 / — / .0000 | .0320 / — / .0000 | .0341 / — / .0000 |
| TL | .0540 / .1410 / .0667 | .0156 / .0955 / .0000 | .0217 / .0986 / .0000 | .0363 / .1144 / .0000 |
| VI | .0211 / .0235 / .0000 | .0191 / .0216 / .0000 | .0173 / .0206 / .0000 | .0161 / .0167 / .0000 |
| ID | .0036 / .0158 / .0000 | .0000 / .0000 / .0000 | .0073 / .0320 / .0000 | .0060 / .0225 / .0000 |
| MS | .0022 / .0135 / .0000 | .0065 / .0158 / .0000 | .0088 / .0131 / .0000 | .0042 / .0131 / .0667 |

三句/语种只作冒烟；未越ΔCER>.05不证明无退化。B/MS一次尾部启发式命中尚非真实截断确认。
A相对base在TH/VI/ID/MS、B在五语种均有尾静音相对风险；r8在该小样本TH/TL/ID也触发。
相对r8及B−A没有新增时长类红线，不据此宣称完整通过。ZH/EN回放本轮**未评**。

24次复用原WAV的ASR诊断确认两条分数对语言解码敏感：
A/1902_zh/44的CER .5893→0、B/1902_zh/42为.7857→.1071（auto与指定TL，均零温度）。
不把中文转写当作TTS串语种，也不以指定TL低分证明发音正确；原评分/全局ASR默认未改。
本轮声学盲听、口音、自然度与情绪均**未验证**，不要求用户作五语种母语评审。

复现与证据：`docs/r11_short_cue_20260918.md`；
大盘`short_cue_ab_20260918/eval_recovery_20260918/{plan.json,paired_results.json,asr_language_probe/}`。
训练/评测完成后没有再次提交任务；原SIGSEGV与缓存诊断证据保留。
工程检查：远端全套141项通过、退出0；不是GPU音质验收。

---

## r10：lora_omni5_r10_ref_off / lora_omni5_r10_ref_on

- 实验完成：**2026-09-17 18:40**；补录：2026-09-18。
- 结论：**不通过候选晋级：B未取得希望的音色稳定性收益，A/ZH和B/MS存在CER回退风险。**
  保持r8，不扩大本剂量；仅否定当轮组合，不否定全部ref路线。
- A/B从同base新训，target/文本/顺序完全相同，B仅保留可信同人、同split、不同原音频的ref字段；
  不伪造身份、没有同源切片，额外ref输入不冒充相同FLOPs。
- 下一步的base/r8新文本对照已完成，随后r11也已结案，见文首；不重复启动历史“下一步”。

### 数据、配比与超参数

两组各5472条/11.9866h，1epoch/342次更新、target最大曝光1次。
B为575条ref（10.51%）、565个独立ref、最大复用2次，A无ref；两组无控制前缀。
共享LoRA r64/alpha64/dropout0.05，lm/dit开启、proj关闭，lr1e-4、
batch2×累积8×1GPU=16，warmup34、weight_decay0.01，训练seed42。
val loss：A 1.039289→.909620，最优.896916；
B 1.039289→.899584，最优.899584（step0→341）。

时长权重：TH=FLEURS2/CV22 3/YODAS3/GS2 9，ID=FLEURS4/CV22 3/GS2 10，
VI=FLEURS7/GS2 10，TL=FLEURS17、MS=FLEURS17、ZH=FLEURS10、EN=FLEURS5。
实际来源两组完全相同（target小时，不计ref）：

| source_id | 许可 | 条数 | 小时 |
| --- | --- | ---: | ---: |
| cv22_id | CC0 | 290 | .358867 |
| cv22_th | CC0 | 285 | .359339 |
| fleurs_en | CC-BY-4.0 | 265 | .601386 |
| fleurs_id | CC-BY-4.0 | 174 | .480539 |
| fleurs_ms | CC-BY-4.0 | 690 | 2.034381 |
| fleurs_th | CC-BY-4.0 | 81 | .241183 |
| fleurs_tl | CC-BY-4.0 | 587 | 2.036917 |
| fleurs_vi | CC-BY-4.0 | 346 | .840903 |
| fleurs_zh | CC-BY-4.0 | 496 | 1.198933 |
| gigaspeech2_id | Apache-2.0 | 650 | 1.195783 |
| gigaspeech2_th | Apache-2.0 | 613 | 1.079764 |
| gigaspeech2_vi | Apache-2.0 | 732 | 1.197703 |
| yodas_th | CC-BY-3.0 | 263 | .360931 |

### 分语种离线诊断

完整v2，182case×seed42/43/44/45/46，CFG1.8、20步、禁重试、large-v3；
A/B新生成各910条，共1820条。r8见过其中95条文本，不能把全量称外部独立留出。
单元格为 **CER / WER / 疑似漏尾率**：

| 语种 | base | r8 | A/ref off | B/ref on |
| --- | --- | --- | --- | --- |
| TH | .1142 / — / .0688 | .0957 / — / .0437 | .1049 / — / .0688 | .1024 / — / .0688 |
| TL | .0184 / .0742 / .0000 | .0146 / .0762 / .0000 | .0182 / .0811 / .0000 | .0186 / .0791 / .0000 |
| VI | .0730 / .0926 / .0563 | .0647 / .0835 / .0250 | .0691 / .0867 / .0437 | .0617 / .0830 / .0375 |
| ID | .0230 / .0958 / .0000 | .0194 / .0960 / .0063 | .0222 / .1062 / .0000 | .0250 / .1061 / .0063 |
| MS | .0681 / .1146 / .0938 | .0116 / .0479 / .0000 | .0293 / .0665 / .0312 | .0661 / .0989 / .0375 |
| ZH | .0580 / — / .0000 | .0775 / — / .0000 | .1468 / — / .0333 | .0644 / — / .0167 |
| EN | .0150 / .0248 / .0000 | .0173 / .0268 / .0000 | .0173 / .0331 / .0000 | .0162 / .0287 / .0000 |

A相对base/r8触发ZH的ΔCER>.05；B相对r8触发MS，异常集中于含数字/括注的`ms_nat_16`，
seed43输出34.4s、CER5.8375，不能删掉后宣布通过，也不直接等同于母语确认的漏读。
B相对base另有ID非数字时长膨胀、有声占比下降及多语种尾静音风险，详见原自动run记录。
WavLM两两音色分析中，B的跨seed/跨cue均值在五目标语种均低于A/r8；不是人工音色结论。
30对声学小报告已生成但未评分；口音、自然度、情绪均**未验证**。
单训练seed、10.51%全局ref覆盖限制阴性结论，不启用投影或加epoch补救。

复现与完整统计：`docs/ref_ab_20260917.md`；
大盘`ref_ab_20260917/{plan.json,comparison.json,paired_diagnostics.json,voice_stability/}`。
原两份自动run记录仍保留，补录没有改原评测/音频或重跑训练。

---

## lora_omni5_r9

- 生成时间：2026-09-17 10:39:12
- 结论：**不通过：一等指标 speaker_sim 没升反降（0.9121→0.9074，en ref 0.8272→0.8159 低于基座）；id ΔCER +0.0435 逼近红线且漏尾 0.075/多读 0.031（r8 为 0.006/0.000），vi 同向恶化——cv22 把 r5 时代的韵律毛病带回来了；唯一亮点 zh 0.0775→0.0494、th 略好、ms 保持 0.0115**
- 下一步：生产保持 r8。ref 配对路线要么 cv22 100% 配对率+小份额重试，要么等 drama 真实同人语料；enable_proj 在 10% 覆盖+众包朗读 ref 下证伪

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/True |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 3130 |
| warmup / weight_decay / max_grad_norm | 313 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni9/train.jsonl |
| 样本数 / 语种条数 | 50067 / {"en": 2313, "id": 10722, "ms": 6136, "th": 11379, "tl": 5366, "vi": 9640, "zh": 4511} |
| val loss 首→末 | 1.0491 (step 0) → 0.9733 (step 3129)，最优 0.8922 |

### 数据配比（按时长）

- 总时长 **107.6708 h** / 50067 条，max_repeat=3.0
- 带 ref_audio：5104，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 5.3848 |
| id | 0.17 | 0.17 | 18.3054 |
| ms | 0.17 | 0.17 | 18.3032 |
| th | 0.17 | 0.17 | 18.3047 |
| tl | 0.17 | 0.17 | 18.3014 |
| vi | 0.17 | 0.17 | 18.304 |
| zh | 0.1 | 0.1 | 10.7673 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、182 个 case × seed = 910 条样本

| 语种 | base CER | lora_omni5_r9_latest CER |
|---|---|---|
| en | 0.015 (WER 0.0248) | 0.0168 (WER 0.0308) |
| id | 0.023 (WER 0.0958) | 0.0665 (WER 0.1414) |
| ms | 0.0681 (WER 0.1146) | 0.0115 (WER 0.0443) |
| th | 0.1142 | 0.0895 |
| tl | 0.0184 (WER 0.0742) | 0.0132 (WER 0.0766) |
| vi | 0.073 (WER 0.0926) | 0.0965 (WER 0.1136) |
| zh | 0.058 | 0.0494 |
| **总体** | 0.0568 | 0.053 |
| 疑似漏尾 | 0.0385 | 0.0363 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r9_latest` 红线：id 0.0230→0.0665；vi 0.0730→0.0965
  - 噪声级（未触发红线，照实记录）：en 0.0150→0.0168

时长类门禁（audio_sec 涨幅 >10% 且 |ΔCER非数字| ≤0.01；p90尾静音增量 >0.1s 或 >0.5s；speech_ratio 降 >0.05；标定依据见 docs/qc_gates.md）：
- `lora_omni5_r9_latest`：id p90尾静音 0.180→0.300s（+0.120s）；ms p90尾静音 0.180→0.300s（+0.120s）；th p90尾静音 0.180→0.280s（+0.100s）；tl p90尾静音 0.180→0.300s（+0.120s）；vi p90尾静音 0.180→0.300s（+0.120s）；id speech_ratio 0.902→0.826（-0.076）

报告文件：`base_8de890a3af74491b948c1ae0449967b0.json`、`lora_omni5_r9_latest_a2f9e99f614245f0842c37a076c381bf.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

enable_proj=true + cv22_th6/cv22_id6（50% 配对，全局 ref 覆盖 10.2%，2249 人）；变量混淆提示：cv22 加入的同时 gs2_th 9→5、gs2_id 10→7 被挤占，id/vi 退化是 cv22 还是 gs2 份额下降所致未分离

---

## lora_omni5_r8

- 生成时间：2026-09-16 19:41:11
- 结论：**通过（离线轨）：r2 原配方（全非 _tc、无 cv22/yodas2）用当前代码重建复现成功——id 0.023→0.0194 红线清除、ms 0.068→0.0116 史上最好、th 0.114→0.096 持平 r2 优于基线、vi/tl/en 持平，zh 0.058→0.0775 略升但 Δ<0.05 未触线；漏尾率 0.0132 为各轮最低、多读 0.0011 与基座持平、尾静音 0.185 远好于 r2 的 0.346。r4-r7 的 nat 探针崩溃确认是 _tc 尾裁数据+新增源导致，非配方或环境漂移**
- 下一步：母语盲听 r8（A=base_8de89 B=r8 f1315988，重点 id_nat/ms/th），通过后 merge + 传 HF 替换 r2 转正；之后精力转 drama 自建数据（ref 覆盖率 + enable_proj 修音色）

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 2020 |
| warmup / weight_decay / max_grad_norm | 202 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni8/train.jsonl |
| 样本数 / 语种条数 | 32307 / {"en": 1608, "id": 5934, "ms": 4302, "th": 6837, "tl": 3747, "vi": 6716, "zh": 3163} |
| val loss 首→末 | 1.0128 (step 0) → 0.9456 (step 2019)，最优 0.8436 |

### 数据配比（按时长）

- 总时长 **75.2049 h** / 32307 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 3.7609 |
| id | 0.17 | 0.17 | 12.7833 |
| ms | 0.17 | 0.17 | 12.7831 |
| th | 0.17 | 0.17 | 12.7872 |
| tl | 0.17 | 0.17 | 12.7847 |
| vi | 0.17 | 0.17 | 12.7851 |
| zh | 0.1 | 0.1 | 7.5208 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、182 个 case × seed = 910 条样本

| 语种 | base CER | lora_omni5_r8_latest CER |
|---|---|---|
| en | 0.015 (WER 0.0248) | 0.0173 (WER 0.0268) |
| id | 0.023 (WER 0.0958) | 0.0194 (WER 0.096) |
| ms | 0.0681 (WER 0.1146) | 0.0116 (WER 0.0479) |
| th | 0.1142 | 0.0957 |
| tl | 0.0184 (WER 0.0742) | 0.0146 (WER 0.0762) |
| vi | 0.073 (WER 0.0926) | 0.0647 (WER 0.0835) |
| zh | 0.058 | 0.0775 |
| **总体** | 0.0568 | 0.0423 |
| 疑似漏尾 | 0.0385 | 0.0132 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r8_latest` 红线：zh 0.0580→0.0775
  - 噪声级（未触发红线，照实记录）：en 0.0150→0.0173

时长类门禁（audio_sec 涨幅 >10% 且 |ΔCER非数字| ≤0.01；p90尾静音增量 >0.1s 或 >0.5s；speech_ratio 降 >0.05；标定依据见 docs/qc_gates.md）：
- `lora_omni5_r8_latest`：id p90尾静音 0.180→0.300s（+0.120s）；ms p90尾静音 0.180→0.280s（+0.100s）；th p90尾静音 0.180→0.280s（+0.100s）；tl p90尾静音 0.180→0.280s（+0.100s）；vi p90尾静音 0.180→0.300s（+0.120s）

报告文件：`base_8de890a3af74491b948c1ae0449967b0.json`、`lora_omni5_r8_latest_f1315988633448b1bd151f4fd999323b.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

r8 与 r2 唯一已知差异：yodas_th 取已下载 32K 条的前 2500 条加工（~3.3h，对齐 r2 时代池子）；2020 步/1 epoch，val loss 末 0.9456。zh 回放漂移需盲听留意（zh_nat_01 0.282、zh_nat_10 0.124，均为已知难 case）

---

## lora_omni5_r7

- 生成时间：2026-09-16 16:35:57
- 结论：**不通过：id 换回非 _tc 数据后修复未果（0.0812，仍 3.5× 基线，id_nat_20/17 照旧崩）；th 0.195/vi 0.121/zh 0.074 相对 r6 全面变差——而这三个语种的训练数据与 r6 完全相同，只有 id 源替换+共享 RNG 抽样挪动。_tc 单因论被推翻， run 间方差比既有认知大，r2 仍是唯一全语种通过的联合模型**
- 下一步：r8 复现性实验：用当前代码重建 r2 的 joint_omni2 配方（全非 _tc、无 cv22/yodas2），区分「数据配方」vs「代码/环境漂移/运气」；同时启动 r2 在 v2 集上的母语盲听（A=base_8de89 B=r2 afbdd4ec）

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 2977 |
| warmup / weight_decay / max_grad_norm | 297 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni7/train.jsonl |
| 样本数 / 语种条数 | 47630 / {"en": 2233, "id": 8230, "ms": 6946, "th": 10217, "tl": 5190, "vi": 10463, "zh": 4351} |
| val loss 首→末 | 1.0820 (step 0) → 0.9269 (step 2976)，最优 0.9085 |

### 数据配比（按时长）

- 总时长 **104.196 h** / 47630 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 5.2101 |
| id | 0.17 | 0.17 | 17.713 |
| ms | 0.17 | 0.17 | 17.7139 |
| th | 0.17 | 0.17 | 17.714 |
| tl | 0.17 | 0.17 | 17.7133 |
| vi | 0.17 | 0.17 | 17.7129 |
| zh | 0.1 | 0.1 | 10.4189 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、182 个 case × seed = 910 条样本

| 语种 | base CER | lora_omni5_r7_latest CER |
|---|---|---|
| en | 0.015 (WER 0.0248) | 0.0183 (WER 0.0347) |
| id | 0.023 (WER 0.0958) | 0.0812 (WER 0.1574) |
| ms | 0.0681 (WER 0.1146) | 0.0184 (WER 0.052) |
| th | 0.1142 | 0.195 |
| tl | 0.0184 (WER 0.0742) | 0.0141 (WER 0.0743) |
| vi | 0.073 (WER 0.0926) | 0.1212 (WER 0.1419) |
| zh | 0.058 | 0.0741 |
| **总体** | 0.0568 | 0.0815 |
| 疑似漏尾 | 0.0385 | 0.0385 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r7_latest` 红线：id 0.0230→0.0812；th 0.1142→0.1950；vi 0.0730→0.1212；zh 0.0580→0.0741
  - 噪声级（未触发红线，照实记录）：en 0.0150→0.0183

时长类门禁（audio_sec 涨幅 >10% 且 |ΔCER非数字| ≤0.01；p90尾静音增量 >0.1s 或 >0.5s；speech_ratio 降 >0.05；标定依据见 docs/qc_gates.md）：
- `lora_omni5_r7_latest`：vi audio_sec 4.08→4.69s（+15%），ΔCER非数字 +0.0022——变长但内容没变；id p90尾静音 0.180→0.280s（+0.100s）；ms p90尾静音 0.180→0.280s（+0.100s）；th p90尾静音 0.180→0.280s（+0.100s）；tl p90尾静音 0.180→0.300s（+0.120s）；vi p90尾静音 0.180→0.300s（+0.120s）；id speech_ratio 0.902→0.835（-0.067）；th speech_ratio 0.902→0.847（-0.055）

报告文件：`base_8de890a3af74491b948c1ae0449967b0.json`、`lora_omni5_r7_latest_3ea9989b4a36467887d516c5d2f64f6e.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

r7 与 r6 的 th/vi/zh 数据一字不差但 th CER 0.122→0.195、vi 0.064→0.121；崩溃 case 每轮换一批（r6 的 vi_nat_30 正常、r7 崩到 1.539；r6 崩的 id_nat_08 0.544、r7 降到 0.150）——nat 探针崩溃是训练 run 间不稳定的固有表现。尾静音：r7 保持 _tc 改善（0.190，r2 是 0.346）

---

## lora_omni5_r6

- 生成时间：2026-09-16 12:37:47
- 结论：**不通过：id 退化未修复且略恶化（CER 0.023→0.1373、漏尾 9.4%、多读 6.2%，红线 Δ>0.05），砍掉 cv22_id 没有解决，归因被推翻；th 小幅退化（漏尾 6.9%→10.6%、多读 3.1%）；ms 大胜保住（0.068→0.015、漏尾 9.4%→0）、tl/vi 持平略好、zh 回到基线（0.086→0.056）**
- 下一步：查 r2 的 id 数据口径与 _tc 尾裁差异；失败集中在 id_nat/th_nat 自然口语探针且 r5/r6 同一批 case，怀疑 _tc 尾裁或 gigaspeech2 系数据破坏停止行为；r7 候选：id 退回非 _tc 数据或独立 LoRA

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 2978 |
| warmup / weight_decay / max_grad_norm | 297 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni6/train.jsonl |
| 样本数 / 语种条数 | 47646 / {"en": 2233, "id": 8240, "ms": 6946, "th": 10220, "tl": 5190, "vi": 10465, "zh": 4352} |
| val loss 首→末 | 1.0791 (step 0) → 0.9446 (step 2977)，最优 0.9182 |

### 数据配比（按时长）

- 总时长 **104.2082 h** / 47646 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 5.2101 |
| id | 0.17 | 0.17 | 17.7154 |
| ms | 0.17 | 0.17 | 17.7139 |
| th | 0.17 | 0.17 | 17.7184 |
| tl | 0.17 | 0.17 | 17.7133 |
| vi | 0.17 | 0.17 | 17.7173 |
| zh | 0.1 | 0.1 | 10.4199 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、182 个 case × seed = 910 条样本

| 语种 | base CER | lora_omni5_r6_latest CER |
|---|---|---|
| en | 0.015 (WER 0.0248) | 0.0242 (WER 0.0364) |
| id | 0.023 (WER 0.0958) | 0.1373 (WER 0.1876) |
| ms | 0.0681 (WER 0.1146) | 0.0146 (WER 0.0486) |
| th | 0.1142 | 0.1216 |
| tl | 0.0184 (WER 0.0742) | 0.0147 (WER 0.0732) |
| vi | 0.073 (WER 0.0926) | 0.0638 (WER 0.0815) |
| zh | 0.058 | 0.056 |
| **总体** | 0.0568 | 0.067 |
| 疑似漏尾 | 0.0385 | 0.0451 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r6_latest` 红线：en 0.0150→0.0242；id 0.0230→0.1373；th 0.1142→0.1216

时长类门禁（audio_sec 涨幅 >10% 且 |ΔCER非数字| ≤0.01；p90尾静音增量 >0.1s 或 >0.5s；speech_ratio 降 >0.05；标定依据见 docs/qc_gates.md）：
- `lora_omni5_r6_latest`：id p90尾静音 0.180→0.300s（+0.120s）；ms p90尾静音 0.180→0.280s（+0.100s）；th p90尾静音 0.180→0.280s（+0.100s）；tl p90尾静音 0.180→0.280s（+0.100s）；vi p90尾静音 0.180→0.300s（+0.120s）；id speech_ratio 0.902→0.840（-0.061）

报告文件：`base_8de890a3af74491b948c1ae0449967b0.json`、`lora_omni5_r6_latest_e5c65c1f92be4f3a872dbecea45bbab8.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

失败 case 高度集中：id_nat_20 r6 CER 2.41（5 seed 全崩）、id_nat_08 0.544、id_nat_02 0.376；th_nat_05 0.350、th_nat_20 0.306。eval 分 3 片并行跑（新 --shard/--merge，已修字符串 case_id bug）

---

## lora_omni5_r5

- 生成时间：2026-09-15 20:28:54
- 结论：**离线轨不通过：id 非数字 CER 0.0237→0.1130 过红线（Δ>0.05），集中在 id_nat 自然口语 case（id_nat_20/02 跨 seed 全崩）；ms 大胜（CER 0.068→0.016、漏尾 9.4%→0.6%）、th 改善（0.114→0.106）。待母语盲听终裁**
- 下一步：若盲听确认 id 退化：r6 砍掉 cv22_id 份额（怀疑众包朗读韵律带偏自发口语），其余语种维持联合配比不变；尾静音继续观察（0.228→0.189 在收敛但仍差于 base 0.102）

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 3267 |
| warmup / weight_decay / max_grad_norm | 326 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni5/train.jsonl |
| 样本数 / 语种条数 | 52271 / {"en": 2394, "id": 10125, "ms": 7426, "th": 10929, "tl": 5496, "vi": 11205, "zh": 4696} |
| val loss 首→末 | 1.0968 (step 0) → 0.9368 (step 3266)，最优 0.9134 |

### 数据配比（按时长）

- 总时长 **111.1782 h** / 52271 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.0501 | 5.5723 |
| id | 0.17 | 0.1704 | 18.9418 |
| ms | 0.17 | 0.1704 | 18.9503 |
| th | 0.17 | 0.1704 | 18.944 |
| tl | 0.17 | 0.1681 | 18.6865 |
| vi | 0.17 | 0.1704 | 18.9417 |
| zh | 0.1 | 0.1002 | 11.1415 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、182 个 case × seed = 910 条样本

| 语种 | base CER | lora_omni5_r5_latest CER |
|---|---|---|
| en | 0.015 (WER 0.0248) | 0.0191 (WER 0.033) |
| id | 0.023 (WER 0.0958) | 0.1016 (WER 0.1739) |
| ms | 0.0681 (WER 0.1146) | 0.0156 (WER 0.0483) |
| th | 0.1142 | 0.1055 |
| tl | 0.0184 (WER 0.0742) | 0.0134 (WER 0.0726) |
| vi | 0.073 (WER 0.0926) | 0.0764 (WER 0.0953) |
| zh | 0.058 | 0.0864 |
| **总体** | 0.0568 | 0.0618 |
| 疑似漏尾 | 0.0385 | 0.033 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r5_latest` 红线：id 0.0230→0.1016；zh 0.0580→0.0864
  - 噪声级（未触发红线，照实记录）：en 0.0150→0.0191；vi 0.0730→0.0764

时长类门禁（audio_sec 涨幅 >10% 且 |ΔCER非数字| ≤0.01；p90尾静音增量 >0.1s 或 >0.5s；speech_ratio 降 >0.05；标定依据见 docs/qc_gates.md）：
- `lora_omni5_r5_latest`：id p90尾静音 0.180→0.280s（+0.100s）；ms p90尾静音 0.180→0.280s（+0.100s）；th p90尾静音 0.180→0.300s（+0.120s）；tl p90尾静音 0.180→0.280s（+0.100s）；vi p90尾静音 0.180→0.300s（+0.120s）；id speech_ratio 0.902→0.841（-0.061）；th speech_ratio 0.902→0.848（-0.054）

报告文件：`base_8de890a3af74491b948c1ae0449967b0.json`、`lora_omni5_r5_latest_3f898e8e9f84428eb3bef9e9e2ce7742.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

新增 cv22_th 25.2h/cv22_id 7.2h/cv22_vi 1.4h/yodas2_ms 1.1h 四个源；v2 case 集 182×5seed 首用；zh 回放漂移 0.058→0.086 未过线但需留意；eval 新增 --shard/--merge 并行

---

## lora_omni5_r4

- 生成时间：2026-09-15 12:06:43
- 结论：**不通过（离线轨：vi +0.246 过红线，但全部红线集中在数字彩票区——vi_digit_3 在 seed43/44 崩、tl_digit seed44 崩；非数字 CER 0.0022→0.0039 饱和持平）**
- 下一步：尾静音修复部分成立（p90 0.42→0.36，数据已压到 0.15s 但模型仍垫 0.25s，剩余 ~0.1s 另有成因）；数字 case 稳定性等 D 扩容（omnivoice_prod_v2，182 条）后用 5 seed 重判；下一轮考虑句中停顿（gigaspeech2 内部犹豫）与 EOS 时序

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 2055 |
| warmup / weight_decay / max_grad_norm | 205 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni4/train.jsonl |
| 样本数 / 语种条数 | 32867 / {"en": 1644, "id": 6023, "ms": 4373, "th": 6941, "tl": 3813, "vi": 6860, "zh": 3213} |
| val loss 首→末 | 1.0647 (step 0) → 0.8811 (step 2054)，最优 0.8811 |

### 数据配比（按时长）

- 总时长 **76.4681 h** / 32867 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 3.825 |
| id | 0.17 | 0.17 | 12.9986 |
| ms | 0.17 | 0.17 | 12.9994 |
| th | 0.17 | 0.17 | 12.9992 |
| tl | 0.17 | 0.17 | 12.9996 |
| vi | 0.17 | 0.17 | 12.9988 |
| zh | 0.1 | 0.1 | 7.6475 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni5_r4_latest CER |
|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.1339 (WER 0.2389) |
| ms | 0.0852 (WER 0.1762) | 0.1186 (WER 0.1873) |
| th | 0.0282 | 0.0337 |
| tl | 0.1166 (WER 0.0944) | 0.1729 (WER 0.1468) |
| vi | 0.0686 (WER 0.1) | 0.3147 (WER 0.35) |
| zh | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.1544 |
| 疑似漏尾 | 0.0952 | 0.1429 |

退化语种（红线阈值 ΔCER > 0.05；任一语种触发即整轮不通过）：
- `lora_omni5_r4_latest` 红线：tl 0.1166→0.1729；vi 0.0686→0.3147
  - 噪声级（未触发红线，照实记录）：id 0.1036→0.1339；ms 0.0852→0.1186；th 0.0282→0.0337

报告文件：`base_1759722cf7884b88bf66c32f06dd1d66.json`、`lora_omni5_r4_latest_0859d1d75d6f46c99e3efb735dd21800.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

数据侧唯一改动：全部源尾静音硬裁 0.15s（FLEURS 50-87% 行被裁，147 条跌破 3s 丢弃）；配比与超参与 r2 完全一致。尾静音 mean 0.292→0.228、p90 0.42→0.36、max 0.46→0.44；audio_sec 2.91→2.83。混合 RNG 因时长变化漂移，数字 case 崩哪条是抽签（r2 崩 vi_digit_2、r4 崩 vi_digit_3）。待母语盲听裁定。

---

## lora_omni5_r3

- 生成时间：2026-09-14 16:25:48
- 结论：**不通过，且整体劣于 round 2：vi 0.0686→0.2839、ms 0.0852→0.1053、总体 0.0774→0.1259；只有 th 0.0141、tl 0.0998 与 r2 持平且优于基座**
- 下一步：停止调配比。已证明离线指标分辨率不足：vi_digit_3 在 base/r1/r2/r3 上是 0.000/1.4375/0.000/0.958（非单调、整条翻转），ms 在配比完全相同的 r2/r3 之间差 0.042。交付 round 2，数字类残余退化写进模型卡并交盲听裁定

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 1984 |
| warmup / weight_decay / max_grad_norm | 198 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni3/train.jsonl |
| 样本数 / 语种条数 | 31739 / {"en": 1632, "id": 5445, "ms": 4404, "th": 7013, "tl": 3826, "vi": 6216, "zh": 3203} |
| val loss 首→末 | 1.0212 (step 0) → 0.8823 (step 1983)，最优 0.8823 |

### 数据配比（按时长）

- 总时长 **77.6356 h** / 31739 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 3.8813 |
| id | 0.17 | 0.17 | 13.1992 |
| ms | 0.17 | 0.17 | 13.1966 |
| th | 0.17 | 0.17 | 13.198 |
| tl | 0.17 | 0.17 | 13.1986 |
| vi | 0.17 | 0.17 | 13.199 |
| zh | 0.1 | 0.1 | 7.7629 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni5_r3_latest CER |
|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.128 (WER 0.2278) |
| ms | 0.0852 (WER 0.1762) | 0.1053 (WER 0.1746) |
| th | 0.0282 | 0.0141 |
| tl | 0.1166 (WER 0.0944) | 0.0998 (WER 0.0903) |
| vi | 0.0686 (WER 0.1) | 0.2839 (WER 0.3556) |
| zh | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.1259 |
| 疑似漏尾 | 0.0952 | 0.1429 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r3_latest` 红线：id 0.1036→0.1280；ms 0.0852→0.1053；vi 0.0686→0.2839

报告文件：`base_c04dd9d80a164c91a0731f9d29fbf510.json`、`lora_omni5_r3_latest_7f53befe6a7d4649b14393820a8128ed.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

本轮假设（提高 FLEURS 份额以恢复带数字训练时长）被实测否掉：vi 带数字时长从 r2 的 1.44h 恢复到 2.30h（接近 r1 的 2.41h），但 vi CER 反而从 0.1176 恶化到 0.2839。所以 r1→r2 的改善不是数字覆盖带来的，把 vi 从 100% FLEURS 降到 41% 才是关键（短 cue 长度分布）。r3 唯一确定收益是 val loss 更低（0.8823 vs r2 的 0.9176），再次说明 val loss 与验收指标不同向，不能用来选 checkpoint

---

## lora_omni5_r2

- 生成时间：2026-09-14 15:01:15
- 结论：**不通过（id 0.1036→0.1280、vi 0.0686→0.1176 过红线），但相比 round 1 大幅好转：th 0.0282→0.0141、ms 0.0852→0.0630、tl 0.1166→0.0998 三语种优于基座，vi 疑似漏尾 0.2778→0.0，总体 0.0774→0.0827 基本持平**
- 下一步：已定位并启动 round 3：gigaspeech2/yodas 的带数字样本实测为 0.0%，FLEURS 是 20.6-24.1%，round 2 把 vi/id 的 FLEURS 份额压到 7/17 导致带数字训练时长各降约 40%（vi 2.41h→1.44h、id 2.17h→1.30h）。round 3 只改 vi/id 为 fleurs 11 + gs2 6，th/tl/ms/zh/en 配比不动

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 2068 |
| warmup / weight_decay / max_grad_norm | 206 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni2/train.jsonl |
| 样本数 / 语种条数 | 33082 / {"en": 1632, "id": 6121, "ms": 4404, "th": 7013, "tl": 3826, "vi": 6883, "zh": 3203} |
| val loss 首→末 | 1.0775 (step 0) → 0.9767 (step 2067)，最优 0.9176 |

### 数据配比（按时长）

- 总时长 **77.6308 h** / 33082 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：3

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 3.8813 |
| id | 0.17 | 0.17 | 13.1965 |
| ms | 0.17 | 0.17 | 13.1966 |
| th | 0.17 | 0.17 | 13.198 |
| tl | 0.17 | 0.17 | 13.1986 |
| vi | 0.17 | 0.17 | 13.197 |
| zh | 0.1 | 0.1 | 7.7629 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni5_r2_latest CER | lora_omni5_r2_step_0000750 CER |
|---|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.128 (WER 0.2389) | 0.1633 (WER 0.2833) |
| ms | 0.0852 (WER 0.1762) | 0.063 (WER 0.1151) | 0.1158 (WER 0.1873) |
| th | 0.0282 | 0.0141 | 0.0435 |
| tl | 0.1166 (WER 0.0944) | 0.0998 (WER 0.0903) | 0.1277 (WER 0.1194) |
| vi | 0.0686 (WER 0.1) | 0.1176 (WER 0.1389) | 0.1405 (WER 0.1944) |
| zh | 0.0 | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.0827 | 0.1135 |
| 疑似漏尾 | 0.0952 | 0.0952 | 0.1429 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni5_r2_latest` 红线：id 0.1036→0.1280；vi 0.0686→0.1176
- `lora_omni5_r2_step_0000750` 红线：id 0.1036→0.1633；ms 0.0852→0.1158；th 0.0282→0.0435；tl 0.1166→0.1277；vi 0.0686→0.1405

报告文件：`base_c04dd9d80a164c91a0731f9d29fbf510.json`、`lora_omni5_r2_latest_aab2ce1c1ace4d768b8d3ae31b451265.json`、`lora_omni5_r2_step_0000750_b11ca19f51604fdea7ce66e22243b767.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

残余退化全部集中在 3 条数字 case（vi_digit +0.098、vi_digit_2 +0.216、id_digit_2 +0.190），非数字 case 无退化。同时修好一批：ms_digit_3（RM45,000 基座念成印尼盾 Rp，r2 全对）、id_digit_3（GA 876 + pukul 06.15）、tl_digit、ms_digit_2。⚠️ id_digit_2 的 r2 输出被 Whisper 转写成 Harganya Rp250.000.（基座是 250 ribu rupiah），高度疑似 ASR 把口播金额归一成符号形式而非 TTS 念错——CER 分不开这两者，需盲听裁定。step_750（val loss 最优 0.9176）在所有语种上都比 latest 差，再次说明 val loss 不是可用的 checkpoint 选择依据

---

## lora_omni_joint5

- 生成时间：2026-09-14 13:08:40
- 结论：**不通过：vi 在生产短 cue 上 CER 0.0686→0.5278（3 seed × 18 样本），疑似漏尾 0→0.2778；step_500 同样退化(0.4299)，非单纯过训**
- 下一步：补 cue 长度数据：gigaspeech2 dev 分片（p50 4.2-6.2s 自然口语，Apache-2.0）；FLEURS p50 10-14s 与线上 1-3s cue 严重错配

### 超参数

| 项 | 值 |
|---|---|
| finetune | lora |
| LoRA r/alpha/dropout | 64/64/0.05 |
| enable_lm/dit/proj | True/True/False |
| learning_rate | 0.0001 |
| batch × 累积 × GPU | 2 × 8 × 1 = 16 |
| epochs / num_iters | 1.0 / 1274 |
| warmup / weight_decay / max_grad_norm | 127 / 0.01 / 1.0 |
| save/valid_interval | 250/250 |
| max_batch_tokens | 8192 |
| 基座 | /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 |
| 训练清单 | /root/autodl-tmp/voxft_data/processed/joint_omni/train.jsonl |
| 样本数 / 语种条数 | 20370 / {"en": 1126, "id": 3079, "ms": 3013, "th": 4774, "tl": 2620, "vi": 3582, "zh": 2176} |
| val loss 首→末 | 1.0074 (step 0) → 0.8165 (step 1273)，最优 0.8165 |

### 数据配比（按时长）

- 总时长 **53.3127 h** / 20370 条，max_repeat=3.0
- 带 ref_audio：0，带控制前缀：0
- max_exposure：2

| 语种 | requested | actual | hours |
|---|---|---|---|
| en | 0.05 | 0.05 | 2.6658 |
| id | 0.17 | 0.17 | 9.0616 |
| ms | 0.17 | 0.17 | 9.062 |
| th | 0.17 | 0.17 | 9.0634 |
| tl | 0.17 | 0.17 | 9.0617 |
| vi | 0.17 | 0.1701 | 9.0661 |
| zh | 0.1 | 0.1 | 5.3321 |

### 验收（离线轨）

- case 集与口径：cfg=1.8、steps=20、retry_badcase=False、ASR=large-v3、28 个 case × seed = 84 条样本

| 语种 | base CER | lora_omni_joint5_latest CER | lora_omni_joint5_step_0000500 CER |
|---|---|---|---|
| en | 0.0 (WER 0.0) | 0.0 (WER 0.0) | 0.0 (WER 0.0) |
| id | 0.1036 (WER 0.2111) | 0.126 (WER 0.2278) | 0.1653 (WER 0.2944) |
| ms | 0.0852 (WER 0.1762) | 0.0856 (WER 0.1429) | 0.119 (WER 0.1706) |
| th | 0.0282 | 0.0287 | 0.0465 |
| tl | 0.1166 (WER 0.0944) | 0.098 (WER 0.0833) | 0.0998 (WER 0.0903) |
| vi | 0.0686 (WER 0.1) | 0.5278 (WER 0.5944) | 0.4299 (WER 0.4611) |
| zh | 0.0 | 0.0 | 0.0 |
| **总体** | 0.0774 | 0.176 | 0.1709 |
| 疑似漏尾 | 0.0952 | 0.1548 | 0.1786 |

退化语种（红线阈值 ΔCER > 0.005；任一语种触发即整轮不通过）：
- `lora_omni_joint5_latest` 红线：id 0.1036→0.1260；vi 0.0686→0.5278
  - 噪声级（未触发红线，照实记录）：ms 0.0852→0.0856；th 0.0282→0.0287
- `lora_omni_joint5_step_0000500` 红线：id 0.1036→0.1653；ms 0.0852→0.1190；th 0.0282→0.0465；vi 0.0686→0.4299

报告文件：`base_c04dd9d80a164c91a0731f9d29fbf510.json`、`lora_omni_joint5_latest_f4c83666f59149d79dc52b0db84bbb39.json`、`lora_omni_joint5_step_0000500_3eed4673dfb34abcbd92004a5c96f6c6.json`

### 人工盲听

（在 6006「盲听评估」Tab 做完后把分语种 B 胜/平/负 与要点粘到这里；**指标与盲听冲突时以盲听为准**）

非数字类 case 15/15 完全无变化；退化全部集中在数字/货币短句。FLEURS 长度(5.7-21.7s)的 vi 探针上 LoRA 反而更好：CER 0.0218→0.0175、漏尾 0.0833→0.0，证明问题是短 cue 出分布而非越南语能力下降。

---
