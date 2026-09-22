# 2026-09-22：追加三片复核与裁后QC漏检修复

本轮处理三条追加视频，前两条ID/TL另见`video_review_20260922_id_mtf6spn.md`、
`video_review_20260922_tl_mtf75lr.md`。本批共五条的定位入口是DIS
`work/qc_review_20260922_batch/index.html`，产物只在本机ignored work，不可从Git取回。

## 身份冻结

| 新文件 | 实际目标 / 时长 | 旧片对应 | 发布时间UTC+8 | ref/prompt文件 / 字节 |
|---|---|---|---|---|
| `id-11110136-6v3u0-mtf7fu924077d7.mp4` | VI / 136.240s | G05 `mtes3aa2zri9b9` | 19:32:18 | 15 / 3,833,610 |
| `vn-11110136-6v3u0-mtf7ft5ipv5t06.mp4` | VI / 78.111995s | G03 `mtepajbwym0x49` | 19:32:10 | 11 / 3,148,758 |
| `id-11110136-6v3u0-mtf7jqhjk2dd42.mp4` | ID / 157.611s | G04 `mteq9pnflssh8b` | 19:34:53 | 10 / 2,734,724 |

均为2026-09-22的发布记录。CDN地域不是目标语种：第一条虽然URL含ID，实际是VI。
三片与各自旧片的编码画面包、PTS/DTS/时长摘要相同，cue时间轴也相同；
前两片译文不变，最后一片仅cue23缩译。没有本轮新取得的原片，不把旧成片当原声。

文件SHA256分别为：
- mtf7fu9：`c0d6d2b86834c4fe078baf52c1ed8c8d59333d9fda78350e7f31ead4e75e8ab0`（72,456,522字节）。
- mtf7ft5：`f4b8cbf3ff8f09e8a8dfd9c9e9143d3eba9464cff092e8b034ff955c48be3bdc`（33,825,817字节）。
- mtf7jqh：`697a389477fe467df11c60f53ee644761e85958102bf9f6e78cb508247b21ebd`（80,445,857字节）。

publisher源码摘要均为`9adcde15a4aa2989bfe51842c6db1349d5429eba5dffc43cacbb5a5944192cda`。
这是发布进程指纹，不是GPU源码、旧缓存生成版本或r8权重加载证明；本轮新补丁尚未push/部署。

## P0：VI裁后转写未覆盖后半问句，却标pass

mtf7ft5 cue2（6.58s）的目标是：
`Bác sĩ, chẳng phải bác sĩ nói Hạo Nam tỉnh rồi sao? Sao anh ấy vẫn nằm đó?`

最终segment为2.567007s，meta记录`hard_trim_applied=true`、`complete=false`、正文溢出0.892s。
裁后ASR记录：`Bác sĩ, chẳng phải bác sĩ nói hỏng nam tình rồi sao?`，没有后半问句，
但`final_text_qc`为`pass`且`post_containment=true`。这里只确认**转写证据不足却放行**，
不把ASR当母语听评或声学强制对齐，也不宣称已凭视频混音证明每一个词真实被截掉。

根因已回当前DIS代码复现：`finalize_segment_boundaries`原来反向复用“精确前缀后有多词”的
检测器检查漏尾。ASR前半句将`Hạo Nam tỉnh`识成`hỏng nam tình`，精确前缀条件不成立，
整段缺失的后半问句便未被记录，宽松coverage的pass直接留下。

最小修复只在**发生最后边界裁切、裁后ASR拟放行**的公共入口：
- 复用已有token规范化与`difflib.SequenceMatcher`，查看目标/转写有序对齐末端是否为删除。
- 前部错字不再掩盖尾部缺失；保留原ASR文本和原pass记录，改为`uncertain`、保留身份风险。
- 完整句仅有前部错字仍可沿原口径通过；未裁切句不进入新检查，不改全局ASR门槛或额外发声检测。
- 不新增ASR/TTS/LLM调用，不换音色、seed、CFG、steps，不额外裁词，也不回退原声。
- 该修复是**阻止错误认证**，不是把缺失内容补回。当前cue2仍需核源窗、缩译预算与局部重渲。

回归使用真实文本及转写、伪ASR和本地合成测试波形：旧代码1失败5通过；修后原组合6通过，
三个直接相关测试文件75通过；再与参考窗、缩译、FFmpeg和发布链共13文件合跑，**186通过（7.22秒）**。
断言裁后仍只调用一次ASR、音频保留、原身份和severe标记保留；不是线上E2E或听感验收。

同片另有：
- cue7（24.34s）文本`Sống tiếp đi.`，raw6.88s→segment5.026145s，`output_vocalization`，
  refit请求后无eligible；不因最终文字pass或长窗有能量就认证正常。V01提供新版混音与实际参考，选听。
- cue13仍是65.16–73.32s字幕窗，仅`Tri Hạ.`；segment1.745011s未变，源身份风险仍在。
  不能把字幕长窗等同持续朗读，也不再次为填满字幕窗拉长声音。
- cue14（74.44s）新版无硬裁、最终ASR pass；没有新母语结论，不能直接认证否定词/语言自然度。
- 源音频语言配置仍`en`、源字幕有中文，此前已登记；不据此直接改音频源语为`zh`。

## P0：最后一条ID的35.4秒仍缺TTS，原因已能直接验证

mtf7jqh cue7文本`Dia kayaknya nggak bohong.`；raw/segment均0，错误为
`voxcpm invalid reference audio: empty or effectively silent`，角色ID为空。

从MP4恢复参考`04f74e3ff4a398bbe2ee5500505cacd657bd44c9c31787978cce2d126481fb9a`：
32kHz、单声道、46,400帧（1.45s），逐样本绝对峰值**0**，与旧片诊断资产SHA相同。
该关联的`request_hash_verified=false`：是被拒输入，不冒充成功TTS请求已核验的参考。
因此不关静音保护、不跨身份借音、不重试相同零参考；需要重新核源人声窗/分离结果，取得可信同人参考。
单个成片没有源人声全轨和已知同身份候选，当前不能安全自动补配。

已确认的变化：
- cue28（147.66s）旧空segment→2.909458s，新版最终forced pass。未取得旧raw来做同take对照，
  不把本轮差异全部归功于某一个后处理补丁。
- cue23（122.62s）本次refit确实完成：`Biaya pengobatan. Rp38.680. Kembalikan uangku.`
  → `Obat Rp38.680. Kembalikan uangku.`，金额字符串不变；raw7.3→4.94s，约1.161倍速后
  不再硬裁、content_fit完整。最终segment文本QC字段未记录，语义、实际金额念法仍未全面验证。
- 24条需TTS，23条非空；5条显式保留均为笑声，不能与35.4秒失败抑制原声混为一谈。
  新severe为4/9/11/15/16/18/22，仍有风险；自动uncertain变少不等于完整验收通过。

## VI mtf7fu9：先查局部输入和时长，不训练

- 28个cue，22条TTS均非空；6条显式原声为笑声/哀叫。
- cue7（22.28s）`Đi đi.`的segment仍4.198188s，有speaker_mismatch，自动forced pass不是拖长/音色通过。
- cue13（69.52s）segment从3.214667到1.3s，但仍speaker_mismatch且最终uncertain。
- cue15（74.6s）短文本segment3.132s，content_fit尚有0.082s溢出；cue17（78.32s）保留自动gender_mismatch。
  不把这些自动标签当已确认换性别，也不凭成片F0指定参考性别。
- cue25/27仍有output_vocalization和speaker_mismatch；全片5个最终uncertain保持待查。

## 恢复与下一步

三片当前输入核验关联依次为25、16、27个；最后ID另有1个未成功请求的零参考。
恢复旧请求ref依次21/22、14/14、23/23条；恢复旧prompt依次3、0、0条。
这是客户端输入恢复，不是服务端预处理结果，更不是完整模型重放；原raw/segment仍优先从真实`CUE_AUDIO_BUNDLE`获取。

优先顺序：发布前先整合并部署已验证的裁后QC/参考窗修复；恢复ID35.4s真实参考输入；
ID64s与TL87s新版做必要声学听评；再针对局部原始take做单变量对照。
五条视频没有提供足以启动新一轮微调的隔离证据。继续保持r8，不加全局重试/步数，不重跑旧评测。
用户语言质量仍未验证；本轮没有新远端ASR/TTS调用、训练、提交或部署。
