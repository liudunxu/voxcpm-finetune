# VI 1:57严重沙哑：生成风险、救援额度与模型归因

## 反馈与证据边界

视频`id-11110136-6v3u0-mtg3totfowzlc4.mp4`，实际metadata目标语为`vi`，不是由CDN地区推断ID。
本地SHA256：`b70fba6bcc1cbf95169f0452bc321a913ae1c42531b10c5d74891184987da03c`。
用户反馈“1:57左右有明显的沙哑音，沙沙的，都听不清了”；重复消息只记一次，语言质量未全面验证。
视频/原始inspection在ignored `work/video_review/20260923_mtg3tot/`；共136.24s、28个cue。
用DIS既有`cue_audio_bundle`提取16个去重参考文件，总3,452,798字节并核验SHA。
未取到实际raw/segment或GPU编码器输入，不能把成片混音冒充裸TTS对照。

## 定位：cue27，而非邻句或原声笑声

- 目标`Tất cả các người phải chết.`；字幕116.54–117.74s，源窗116.24–118.04s。
- 最终segment 1.94s、placement=0，物理片段约116.54–118.48s；`actual_end=118.42`
  对应记录的语音包络终点，不用它替代文件终点。1:57落在本句内部。
- speaker=`review_ed98686f`；参考锚点来自cue13/27，身份锁定。cue26属于另一speaker；
  cue28在118.7s开始显式保留笑声，不是此句质检失败后回退原声。
- 选中请求ref与嵌入字节SHA一致：`315f69f37a30ec92576f8686d69f1177d1b68a1f3a231074143b627978e604cb`。
  4.14s、32kHz mono，未使用prompt。按20ms RMS首次超过−60dBFS在0.02s；
  前0.5s峰值−29.10dBFS，**没有上一条VI参考同样的长低能量前导**，但这不等于已证明干净单人。
- 参考QC仍有SNR1.52dB及Gasp风险；speaker_count=1不是已验证单人。服务记录denoise=true、
  ref level−3.02dB、无dereverb/内部发声移除，pre-denoise时长仍4.14s；最终编码器输入未知。

## 生成已报风险，自愈未完成

两条客户端候选各消耗服务端2次生成，共4次，不是只有两次抽样：

| 候选 | 时长 | 主要证据 | 选择 |
|---|---:|---|---|
| 1 | 3.35s | 1.2s短窗runaway，severe output_vocalization/plosive | 放弃，score−788.996 |
| 2 | 2.04s | severe output_vocalization/speaker_mismatch | keep-best，score−612.0988 |

第二条有效seed=`1504565771`（请求`1504564762`），CFG1.5、22步（不是请求的28步）。
控制确实生效：`subtly as a sharp but controlled outburst`；legacy schema1的angry/shouting，
置信0和空来源并不是重新识别后的表演证据，不能将沙哑视为“符合愤怒表演”。
日志明确记录reference-only救援失败：`per-cue generation budget exhausted (4)`。
后续prosody/timbre尝试也未采纳；缓存复用和重复notes不当成独立成功实验。
最终仍`unresolved_artifact=true`、`relative_best_after_retry_exhausted`。

Qwen最终转写与文字一致、final_text_qc=pass，只支持文本覆盖，不推翻用户声学失败。
短cue身份QC advisory也不撤销独立的output_vocalization；自动标签不是人工沙哑病因真值。
同ref的cue25已走无控制救援仍留harsh_high_freq/plosive/text_incomplete，说明不能保证去情绪就好；
那是另一文本/seed，**不是本句因果A/B，也不是额外用户听评**。

## 后处理与部署状态

raw SHA=`3a258bca42fa06752c340240b6f4139aa1253911a8c58aca59846a693b9138bd`；
segment SHA=`d2a4f8563d9c408606a8bfd91c1dbf773255209a0c0c66b0becdde21f47d4c16`。
服务已裁头0.165s、尾0.035s；客户端边缘语义裁切因Whisper低可信被拒，后续0–1.94s片段。
tempo=1，content_fit无硬裁，段级limiter=false；quiet boost+3.19dB和后续匹配+5dB，
均记为透明增益，记录峰值到−2.75dBFS。可能放大已有噪声，但无阶段WAV不能指认它制造沙哑；
同样不能把所有后处理/最终混音排除。

publisher与本cue请求源码指纹均`f76c13fffc389e8a5a090bf944342d327dc8313fd3e5a3607442918a595d997e`，
对应此前核验的DIS d794f16包内容。此视频仍记录4次上限；本地新6次策略未在本片得到验收。
API只读查询自报已加载`FrankLiuDundun/voxcpm-finetune-lora`，不等于已验证r8权重SHA。
代码`_load_voxcpm_model_sync`用进程级`_VOXCPM_MODEL_ID`，**逐请求改model_id不能成为base/r8 A/B**。

## 模型诊断：已准备，尚未生成

连续严重坏例与E01两边失败，已足以启动模型归因准备，不代表可以直接开训或全局回退。
唯一计划：`eval_cases/roughness_model_ab_20260923.json`；入口`scripts/roughness_ab.py`。
两条VI各3个相邻seed、相同输入下禁用/启用同一严格加载的r8适配器，共12条。
复用`infer.synthesize_ab`，禁内部重试；SHA不符即停止，既有输出目录拒绝覆盖。
模型文件指纹来自9月18日冻结plan；本地r8三个文件与两个ref已验证。
这轮保留客户端原始ref、不运行API去噪/参考清理/电平匹配或DIS后处理；
只测这一共同输入条件下的模型变量，不声称生产重放、干净参考验收或完整排除预处理。

现有Vast实例52148024运行生产服务，仅发布8000映射、未返回SSH入口。
官方execute接口实测HTTP400：仅允许停止实例；**没有为取证停服务、开新租赁或切换权重**。
目前0/12生成，不生成空的“试听完成”HTML；有真实音频后再复用离线页供用户声学盲听。
原始stage音频仍缺，生产case保持`replay_ready=false`，合成输出不进入训练集。

独立GPU就绪后，将ignored输入包与r8备份同步到数据盘，基座使用同SHA文件。执行：

```bash
VOXFT_DATA_ROOT=/root/autodl-tmp VOXFT_CKPT_ROOT=/root/autodl-tmp/voxft_ckpt \
HF_HOME=/root/autodl-tmp/hf_home PYTHONUNBUFFERED=1 \
nohup stdbuf -oL -eL uv run python -X utf8 -u scripts/roughness_ab.py \
  /root/autodl-tmp/roughness_model_ab_20260923 \
  --base /root/autodl-tmp/hf_home/hub/models--openbmb--VoxCPM2/snapshots/32279effe8c19989596f05d353d1447f51d9e915 \
  --lora /root/autodl-tmp/voxft_ckpt/lora_omni5_r8/latest \
  --output /root/autodl-tmp/roughness_model_ab_20260923_output \
  > /root/autodl-tmp/roughness_model_ab_20260923.log 2>&1 &
```

先加`--check-only`做完整输入预检；不是让当前生产实例执行。
若r8更差，先做回退候选验证；两模型在可靠输入下仍持续失败，才准备授权真人数据的定向微调。
两条VI是诊断富集，不足以裁决五语种联合模型，不新增母语质量结论。

本轮本地验证：执行器及既有LoRA加载/开关测试21项通过（1.67s）；DIS预算/参考救援70项通过
（1.88s），其中补验短cue身份advisory不会掩盖output_vocalization且不会重复扩额。
执行器测试使用假模型验证12个产物、种子、SHA拒绝及禁止覆盖，不算GPU生成或声学收益。
随后交付检查：本仓全套152项通过（26.61s）、DIS26文件367项通过（9.46s）；
DIS `bd2ada7`已push并核验远端，待用户部署，OmniVoice源码未改。两仓diff whitespace检查通过。
取证后再次查询Vast完整实例记录（未做列裁剪），`image_runtype=args`，ssh_host/ssh_port均null，
依然仅发布8000端口；不会为取SSH入口重启当前生产服务。只读结果保存在原视频work目录。
