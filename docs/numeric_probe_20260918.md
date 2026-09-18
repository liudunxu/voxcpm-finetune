# 2026-09-18：数字输入诊断，不开新微调

## 先核对输入，再用GPU

r10/r11未产生可晋级候选，指定生产基线继续r8。本轮不加epoch、不改CFG/步数、不替换生产权重。

实际调用本机DIS的`_build_payload`，覆盖旧`omnivoice_prod`和v2中的29条TH/VI/ID/MS数字文本，
再送入真实OmniVoice `TextNormalizer`：

- **29/29目标文本在服务端数字开关true/false下相同**。原因是DIS已先完成数字词化，
  不是开关再次失效；该结论只覆盖这29条目标文本，不外推续写prompt或所有输入。
- 当前DIS只对TL/fil设置`spoken_text_normalized=true`；四个SEA语种不设此字段，
  服务端仍会清理文本，但不能靠关闭开关还原上游已展开的数字。
- 因而不跑“只切服务端开关”的伪A/B。旧数字压力case和历史成绩保留，
  它们的原始数字形态不能继续冒充当前DIS的最终模型输入。
- 另发现`id_digit_3`的`pukul 06.15`展开为`pukul nol enam.lima belas`。
  保留为时钟/标识符规则边界，不在本轮猜测目标读法或按CER自动修正。

输入审计、两仓源文件SHA、修前源码快照在
`checkpoints/numeric_probe_20260918/input_audit.json`与`source_identity.json`。
记录的HEAD为改动前父提交，实际实验输入以文件SHA为准。

## 已确认并修复的输入错误

`ms_digit_2`：`Harganya RM1,200 ringgit.`被两侧词化器各自展开成
`Harganya seribu dua ratus ringgit ringgit.`，**额外添加了一个币种词**。
这是确定的文本构造错误，不依赖ASR、GPU或非母语听感才能确认，也不表示r8自己发生复读。

- 修复DIS与OmniVoice各自的共享货币前缀展开分支：后面已有完整同币种名称时不再添加；
  保留原有名称的大小写与空白。拉丁文字检查词边界，泰语保留无空格形式。
- 不全局删除重复词，不改数值或币种、不猜测时钟读法；EN/TL规则不变。
- 服务端结果缓存算法版本9→10；DIS已有分语种文本签名版本TH 2→3、VI/ID/MS 1→2。
  使正常缓存复用路径不继续服务修复前的旧take；显式仅复用旧音频不等于重新合成。
- 两侧各8个回归样本：修复前各6失败/2通过，修复后通过；覆盖四语种、
  大小写、泰语黏连、无重复名称及拉丁词边界，DIS还核对真实payload，
  OmniVoice核对真实normalizer与false开关不改输入。
- 扩展验证：DIS **138项**、OmniVoice **247项**通过。
  这是输入/契约测试，不是线上部署或母语发音验收。

修复已本地提交：DIS `1a1e309`、OmniVoice `ac1e51d`，本次未push或部署。
另直接执行修前/修后的DIS签名函数，确认仅TH/VI/ID/MS旧签名失效，TL/EN保持一致。
在29条审计文本中，仅`ms_digit_2`的目标输入被此次修复改变。
原文已词化的`ringgit ringgit`不会被全局删词；已有失败产物保留。

## 已完成的125条GPU诊断

入口：`scripts/numeric_probe.py`，只调用既有`voxft.eval`，单worker串行，不引入新评分模型。
远端目录：`/root/autodl-tmp/numeric_probe_20260918`。

| 组 | 文本 | 生成数 |
| --- | --- | ---: |
| raw | 四语种各2条已有数字压力文本＋各1条固定随机非数字对照，原样入模型 | 60 |
| words | 同12条文本，使用修复后DIS实际payload文本；其他条件不变 | 60 |
| legacy_ms | 仅`ms_digit_2`修复前的重复`ringgit`输入，定位该修复的声音影响 | 5 |

- seed固定42/43/44/45/49，即基础seed及+1/+2/+3/+7；r8 strength1、CFG1.8、20步，
  reference-only，禁重试，无控制指令、去噪或额外模型文本归一化。
- 共用既有`prod_ref_fil.wav`，SHA256
  `7645c946617f844c2bb78d4eea0ec41aebe70a4847e2f829199021aa6d1fc781`。
  这是冻结评测ref，不冒充当前在线服务经过处理的参考音。
- 数字文本：`th_digit/th_digit_2`、`vi_digit_2/vi_digit_3`、
  `id_digit/id_digit_2`、`ms_digit_2/ms_digit_3`。每语种只有**2条独立数字文本**，
  五个seed不能把样本量变成10条独立文本。
- 固定抽样seed20260918，非数字对照为`th_prod_2/vi_prod_1/id_prod_1/ms_marker`。
  raw/words对照组输入逐字相同，完成后**20/20对WAV SHA256一致、原始ASR转写也一致**。
  这验证本轮相同输入的重复性，不外推所有模型输入。
- 运行前冻结case、ref、base/r8权重、推理/评测源码及独立数字评分器副本SHA；
  不复用历史音频冒充本轮重生成，不覆盖旧报告。
- 开始时大盘剩3.2GiB，GPU空闲；产物与新Numba/编译缓存只写大盘，
  启动要求剩余≥2GiB、逐条要求≥1GiB，关闭在线模型下载，不清理历史证据。

2026-09-18 **11:05:09 +08:00全部125条生成与评分完成**，三份报告及`done`已校验。
原任务完成raw 60条后发生编码异常；保留这60条，只恢复剩余65条，详见下文。
原进程与恢复进程均已退出，**不得重启run或为修复编码而重跑已完成音频**。

## 评分口径

- 保留两臂原始ASR、各自原始CER及WAV；**不直接比较两套不同参考文本的原始CER**。
- 另用同一冻结数字词化器把两臂ASR和同一原始`source_text`映射到共同参考，
  生成`common_cer`、疑似漏尾/多读及分语种配对时长；不改通用评分器或旧报告。
- 此口径仍可能混淆ASR表记、地区数字习惯和发音，币种/时钟边界尤其需要保留原始文本。
  不用自动分数宣布数字念法正确、语言自然度提升或完整验收通过。
- `comparison.json`分别保留raw/words/legacy_ms；legacy_ms只与同case/seed的words比较，
  不混入四语种均值；随机对照单列，数字样本不进入非数字时长门禁。
- 新共同参考评分检查本机与远端各1项通过，验证相同内容的数字/词形式得分相同，
  额外币种词仍有错误，缺少共同源文本直接失败。
  内存中故意改回两臂各用自身输入作参考，新增检查确实失败；未修改冻结源码。
  启动阶段本机相关21项、GPU机全套142项测试通过；收尾编码回归见下文。

## 结果：VI有改善信号，MS不能凭此分数定性

下表来自`comparison.json`与`results_summary.json`；每语种仅2条独立数字文本×5 seed。
CER为共同参考口径，时长差为words−raw，不进入非数字`duration_inflation`门禁。

| 语种 | raw共同CER | words共同CER | 平均时长差 |
| --- | ---: | ---: | ---: |
| TH | 0.053480 | 0.046337 | +0.208s |
| VI | 0.421026 | 0.049487 | −0.128s |
| ID | 0.000000 | 0.000000 | +0.032s |
| MS | 0.172332 | 0.302217 | −0.096s |

- **VI**：`vi_digit_2`由0.560000→0.006667，5/5 seed共同CER改善；
  词化后4条转写对应250.000金额，1条剩`vé/về`差异。
  `vi_digit_3`为0.282051→0.092308，3改善/2平，但words seed43仍有异常转写。
  这是两条文本的内容诊断信号，不是语言质量通过，也不构成追加微调的证据。
- **TH**：`th_digit`为0.076190→0.061905，2改善/2平/1差；
  words seed44仍异常。`th_digit_2`五对均平，不宣布统一收益。
- **ID**：两条文本共10对共同CER均为0。原始CER受数字词/数字/货币符号表记影响，
  不能用两臂各自原始CER推出发音改善。
- **MS**：两条case均为1改善/2平/2差，均值上升，但币种符号与评分边界存在混淆；
  下述复核不足以确认实际币种读错，也不能排除内容风险。
- **非数字对照**：20/20对输入、WAV字节和原始ASR一致，不存在本轮控制组漂移。

### 重复币种词修复子组

`legacy_ms`输入为`Harganya seribu dua ratus ringgit ringgit.`，
修后words输入为`Harganya seribu dua ratus ringgit.`。
同case、同5 seed共同CER均值0.613793→0.275862；修前5条ASR均写`Rp`，
修后3条写`RM`、2条写`Rp`。原转写没有直接写出`ringgit ringgit`，
因此只能确认**输入不再额外添加币种词**，不能声称音频中的重复词已被听证或已消失。

### 16次MS币种ASR敏感性复核

2026-09-18 **11:08:34 +08:00完成16/16次转写**。复用8条原WAV，
同large-v3、VAD开启、两组均temperature=0，唯一差异为指定MS或auto。
样本/权重/源码SHA与解码参数在`asr_currency_probe/plan.json`，
原结果与配对摘要在`results.json`、`summary.json`；没有新TTS或独立ASR模型。

- 指定MS的8条均复现原报告转写；auto检测8/8为ID，**这不证明TTS串语种**。
- 同WAV有5/8条转写变化，其中2条从`RM`改写为`Rp`：
  words `ms_digit_2/42`为`RM1,200`→`Rp 1.200`；
  raw `ms_digit_3/43`为`RM45,000`→`Rp45.000`。
  另外3条主要为空格/千分位表记变化。
- 币种符号受ASR语言提示影响，不能仅据上述共同CER宣布MS退化或币种发音错误；
  也不能将所有MS风险归咎评分器。原报告、全局ASR默认与语言未验证状态均保留。

### 共同评分器的已知边界

直接执行本轮冻结的`locale_text.py`，MS下：

```text
Kereta itu berharga Rp45,000.  → Kereta itu berharga Rp45,kosong kosong kosong.
Kereta itu berharga Rp 45,000. → Kereta itu berharga Rp empat puluh lima ribu.
```

MS词化器未识别`Rp`前缀，数字正则会部分展开逗号后的数字。
**仅空格差异就能改变共同CER的惩罚量**，故本轮MS共同CER也不是可靠的语义金额校验。
本轮不修改冻结评分器、重算挑分或覆盖原报告；上述原值与原转写全部保留。
后续将“未知币种整串保留、数字分隔符边界”作为独立版本的纯文本回归，
先明确处理边界再最小修复，不顺便猜测时钟规则或将不同币种视为等价。

## 编码故障、恢复与归档

- 原PID42929完成60/125后，`holdout_eval.check_inputs`以默认ASCII读取含泰文的JSON，
  抛出`UnicodeDecodeError`。原`failure.json`、`run.log`及恢复前状态均保留；
  没有证据确定哪个库改变了进程locale，不猜测原因。
- 实测`LC_CTYPE=C`下`python -X utf8=0`读取原plan失败，`-X utf8=1`成功。
  一次性`resume.py`以`python -X utf8 -u`运行，先校验已完成raw报告/60WAV与冻结输入，
  只补words 60＋legacy_ms 5。`recovery.json`记录保留文件SHA与恢复脚本SHA。
- 收尾`closure.json`记录：**85项冻结输入通过、原报告＋60WAV共61文件未变、
  125输出哈希通过**。三报告、计划、对照、失败/恢复证据和125条原WAV均已备份本机
  `checkpoints/numeric_probe_20260918/`，WAV在`audio/`，逐文件SHA验证通过。
- 永久修复仅给`holdout_eval.py`与`numeric_probe.py`共7个JSON读取点指定UTF-8；
  新回归模拟ASCII默认编码，修前失败、修后通过，且冻结输入变更仍拒绝。
  本机相关**28项**、远端全套**143项**通过，远端退出0；不冒称本机全套通过。
- 两个脚本的旧版已按原plan SHA归档到`frozen_sources/`；TTS与ASR诊断均结束后，
  才同步永久修复到远端项目。映射见`postrun_source_archive.json`。
  因而旧plan中的两个项目源码路径现已不同，**不能改plan凑哈希或再调用旧run/collect**；
  历史执行源码与完成前校验由归档、`recovery.json`和`closure.json`共同追溯。

## 下一步与停止条件

1. 保留已确认的重复币种输入修复；不全局启停数字词化、不调CFG/步数、不加epoch、不晋级新模型。
2. 下一项只做未知币种/分隔符的纯文本边界回归，另行版本化；
   不扩TTS/ASR网格、不改旧分数，暂不启动新训练。
3. 有可核验服务时独立核对部署commit、权重与最终采样参数；缺服务不妨碍上述文本修复。
   金额发音、口音、自然度与情绪继续未验证，不要求用户作五语种母语判断。

本轮不部署OmniVoice、不改DIS线上实例，也不是HTTP开关的端到端验收；
实际线上模型身份与最终参数追踪仍待有可核验服务时独立验证。
