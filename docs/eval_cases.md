# 离线评测 case 集

## omnivoice_prod_v2.jsonl（2026-09-15 扩容，182 条）

`omnivoice_prod.jsonl`（28 条，每语种 4-6 条）的分辨率低于 run 间方差（实测同配比两轮 ms CER 差 0.042），v2 把每语种扩到 30+ 条。

### 设计口径

- **来源零泄漏**：文本全部取自远端 `/root/autodl-tmp/voxft_data/processed/<src>_tc/val.jsonl`——train/val 已按说话人/会话隔离，val 行未进任何训练。th 混 `fleurs_th/yodas_th/gigaspeech2_th`，vi/id 混 `fleurs_*` + `gigaspeech2_*`，tl/ms 只有 `fleurs_*`，zh/en 用 `fleurs_zh/fleurs_en`。文本原样保留（标点、大小写、转写错误都不修），只剔空串/单字符行与行首 `(...)`（会被线上当控制指令，`api.py:8118`）。
- **长度优先 2-6s**（线上 cue 约 2.4s），短句不够时向 10s 以内补。实际分布：th/vi/id/zh p50 3.2-4.6s（24-28 条 ≤6s）；**tl p50 11.1s（仅 1 条 ≤6s）、ms p50 7.8s**——fleurs_tl/ms 的 val 就这么长，长句当长文本探针用，不要误读为分布达标。en 的 4 条 numeric 全部 7.6-19.3s（val 里只有这 4 条带数字）。
- **显式覆盖**（每条 `note` 标注）：
  - 数字：各语种 4-6 条，含阿拉伯数字的自动标 `"numeric": true`（剔出 CER 汇总，只走盲听）。
  - code-switch：tl 4-5 条（`mag-book`、`Sea to Sky corridor`、`Yahoo! at Microsoft` 等）、id 4 条（`oke`、`like komentar dan subscribe`、`link` 等）、**ms 只有 2 条**（`hotel`、`Konami`，fleurs_ms val 里没有真 Manglish，混英文覆盖仍靠 v1 的 `ms_manglish` 与 drama_ms）。th 的 4 条是泰文夹英文专名，不是真 code-switch。
  - vi 句尾 nặng 调 4 条（`vi_nat_14/26/27/28`，嘎裂声截断风险形态）。
- **ref 轮换**：ref 池 = `refs/ref_{zh,en,tl}_01..04.wav`（12 条，从 `fleurs_{zh,en,tl}_tc` val 挑的干净 wav，zh/en 3-6s、tl 窗口放宽到 2.5-8s）+ 原有 `prod_ref_fil.wav`，全局 round-robin，每条 case 带 `ref_lang`。五个目标语种的 zh/en/tl ref 覆盖各约 9-13 条。ref 文件只在远端 `/root/autodl-tmp/eval_cases/`，不入库。
- **schema**：与 v1 相同（`case_id`/`lang`/`text`/`ref_audio`/`seed`/`note`，可选 `numeric`），新增 `ref_lang` 字段。seed 1100 起、与 v1 不重叠；`case_id` 用 `<lang>_nat_NN`。

### 统计

| lang | n | numeric | code-switch | 时长 p50 |
|---|---|---|---|---|
| th | 32 | 5 | 4（专名） | 3.2s |
| tl | 30 | 4 | 5 | 11.1s |
| vi | 32 | 4 | —（nặng 尾 4） | 3.4s |
| id | 32 | 4 | 4 | 3.6s |
| ms | 32 | 5 | 2 | 7.8s |
| zh | 12 | 4 | — | 4.6s |
| en | 12 | 4 | — | 4.5s |

### 后续动作

1. GPU 空下来后先跑 `eval base` 拿 v2 基线（cfg 1.8 / 20 步 / ≥3 seed，同 v1 口径），case 文件在远端 `/root/autodl-tmp/eval_cases/omnivoice_prod_v2.jsonl`。
2. **同一配置（同 base）重跑一次基线**，量 v2 自己的 run 间方差——新噪声门槛要按这个实测值定，不要沿用 0.005。
3. 每语种 30+ 条后 ΔCER 的可信分辨率仍待实测确认；盲听仍按 `BLIND_LOSS_MARGIN`/`BLIND_MIN_LOSSES` 门槛。
4. 按 ref_lang 分组看结果（zh/en/tl ref 各自的分语种 CER），验证「ref 语言改变胜负语种」的结论在 12 条 ref 池上是否仍成立。
