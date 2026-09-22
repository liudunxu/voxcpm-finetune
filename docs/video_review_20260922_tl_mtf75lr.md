# TL 新片 mtf75lr：87秒拆句定位与旧参考恢复

## 产物与证据边界

- 用户成片：`ph-11110136-6v3u0-mtf75lrr9csh47.mp4`，目标`tl`、源语配置`auto`，114.923s。
- 文件55,609,354字节，SHA256：`c87f777ee7dd96e78b4d8a2d493c86b9f5612f9af3605395b2d0febed939f806`。
- 发布记录2026-09-22 19:24:00（UTC+8）；publisher源码摘要为
  `9adcde15a4aa2989bfe51842c6db1349d5429eba5dffc43cacbb5a5944192cda`，与同批TH/ID一致。
  Git commit、GPU服务端代码和加载权重仍未核实，不能证明本地未提交参考窗补丁已部署。
- 与旧G07 `mtesojt2hiwz27`及用户原片`mtcoctauzp4x8d`的编码画面包、PTS/DTS/时长摘要一致：
  `b05d80a7d883130779eb4528776ea45985355153c8019b550ee5ea0f9ecbcd58`。
  新旧33条cue的时间轴与最终目标文本完全一致，可作同窗定位，不是模型/单变量A/B。
- 本轮只做离线取证、参考提取和定位页；没有新ASR/TTS请求、训练或部署，没有代替用户听评。

## 已确认的工程变化

- 33条均走TTS，新片无空segment；旧cue10（47.893s）从0恢复为3.065s。
  但新旧该句的raw、seed与tempo都变了，**不是同一take只修FFmpeg的效果对照**。
  新版还留`pitch_register_drift`；非空不等于音质、语言或完整性通过。
- 旧severe cue为2/9/13/20/24/28，新为9/16/17/24。风险计数减少不认证听感改善。
- 新版最终文本QC：pass 14、forced pass 5、compact pass 1、uncertain 7、未记录6。
  uncertain在cue11/12/14/17/19/26/31；不能把其余cue全部算人工验收通过。
- 单MP4提取14个去重ref/prompt，共4,493,274字节；33个ref关联和4个prompt关联均核验当前请求hash。
  **旧33条请求的客户端ref全部可由新片字节恢复**；旧prompt仅cue5/17的hash在新片资产中找到。
  特别是旧cue24 prompt尚未找回，不能拿新prompt替代。裸TTS、最终segment与服务端预处理参考仍缺。

## 87秒：不能把相邻两句混成一个原因

此前S02反馈是“原片和成片两边都有，成片更重”，覆盖cue23尾部和cue24边界；
并未确定究竟哪句、哪阶段新增沙哑。新版未听评，不继承旧版异常结论。

| 项目 | cue23：85.48s | cue24：87.27s |
|---|---|---|
| 新旧ref | 同SHA `511902ee…` | 同SHA `fc4e401f…` |
| 模式 | 无prompt、reference-only | ref + prompt；prompt字节已变化 |
| 实际控制 | angry / controlled outburst，仍应用 | 无控制 |
| 新旧实际seed/CFG/steps | 304134999 / 1.5 / 28，相同 | 671272075 / 2.0 / 28，相同 |
| raw时长 | 均4.33s，但SHA不同 | 7.595→5.985s，SHA不同 |
| tempo | 均1.684 | 1.348→1.122 |
| segment | 均1.719042s，但SHA不同 | 4.742083→4.655167s |
| 新版风险 | output_overread，keep-best；forced pass不能排除沙哑 | output_vocalization + speaker_mismatch，源身份风险仍在 |
| 段级limiter | 未应用 | 旧未应用，新应用 |

cue23压缩停顿1.33s后仍需约1.684倍速。它的相同参数/时长**不证明音频相同**，
raw及segment哈希已否定“就是旧take原样复用”的断言；`tts_cache_reused=true`也不指定复用了哪一轮。
cue24的prompt、raw、变速及limiter均有变化，不能由其中一个变化直接推导沙哑成因。
参考SNR较低只是输入风险，不能自动判多人或沙哑，也不据此全局关降噪/控制/limiter。

另有两个检查点：
- 51.2s旧S01复听为“不确定，原片与成片都不沙哑了”，降低优先级、不认证false alarm。
  新cue11的模式/控制/CFG/seed都变化且最终QC uncertain，不能沿用旧听评，也不强迫用户重复评分。
- cue26（93.739s）仍硬裁：输入1.43s、安全预算0.968s、记录正文溢出0.462s，最终QC uncertain。
  refit请求14/26，eligible仅26，changed为空、`rewrite_rejected`。需要核源窗与保义短表达，
  不因请求过refit便宣称已缩译，也不以更激进裁切或回原声“修好”。

## 交付与下一步

- 试听包`tl-ref-mtf75lr`：DIS `work/qc_review_20260922_mtf75lr/index.html`。
  P01/P02拆开87秒两句，分别给新版混音与实际ref；P02另给实际prompt。
  旧版/原片同窗折叠选听，不重复计旧反馈。P03是47.9s非空恢复点，选听。
- P02为保留完整句尾截到92.20s，包含约0.17s下一句起头，页面明确说明，不误标为多读。
- 7个切窗WAV保留三份视频原有48kHz双声道，未降噪、未归一化；全部非空、hash及11个媒体路径通过。
  JS语法校验通过；未作浏览器播放或人工听评验收。全部大产物仅本地ignored work保存，不能从Git取回。
- 优先只听P01/P02和对应输入；如果仍异常，取实际任务`CUE_AUDIO_BUNDLE`，按同take
  raw→segment→mix定位，再分别设计控制或变速单变量A/B。不猜内部下载路径、不要求恢复已销毁容器。
- 继续保持r8与暂停微调；本轮不改生产声音参数、不增加合成时延、不根据自动风险数晋级模型。
