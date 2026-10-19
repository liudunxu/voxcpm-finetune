# 离线评测 case 集

> **2026-10-19 语种收缩**：vi/ms 退出目标语种，多份 case 文件已按语种过滤（详见文末
> [「2026-10-19 过滤记录与归档清单」](#2026-10-19-过滤记录与归档清单)）。本文的统计表与构成明细
> **以过滤后的文件现状为准**；被删掉的 vi/ms 行只以存档注记形式保留说明。

## omnivoice_prod_v2.jsonl（2026-09-15 扩容 182 条 → 2026-10-19 过滤为 118 条）

`omnivoice_prod.jsonl`（28 条，每语种 4-6 条）的分辨率低于 run 间方差（实测同配比两轮 ms CER 差 0.042），v2 把每语种扩到 30+ 条。

### 设计口径

- **来源零泄漏**：文本全部取自远端 `/root/autodl-tmp/voxft_data/processed/<src>_tc/val.jsonl`——train/val 已按说话人/会话隔离，val 行未进任何训练。th 混 `fleurs_th/yodas_th/gigaspeech2_th`（8/8/16），id 混 `gigaspeech2_id`(27) + `fleurs_id`(5)，tl 只有 `fleurs_tl`(30)，zh/en 用 `fleurs_zh/fleurs_en`（各 12）。文本原样保留（标点、大小写、转写错误都不修），只剔空串/单字符行与行首 `(...)`（会被线上当控制指令，`api.py:8118`）。
- **长度优先 2-6s**（线上 cue 约 2.4s），短句不够时向 10s 以内补。实际分布：th/id/zh p50 3.2-4.6s（≤6s 的条数：th 24、id 26、zh 12）；**tl p50 11.1s（仅 1 条 ≤6s）**——fleurs_tl 的 val 就这么长，长句当长文本探针用，不要误读为分布达标。en 的 4 条 numeric 全部 7.6-19.3s（val 里只有这 4 条带数字）。（**存档**：过滤前 vi p50 3.4s、ms p50 7.8s。）
- **显式覆盖**（每条 `note` 标注）：
  - 数字：各语种 4-5 条，含阿拉伯数字的自动标 `"numeric": true`（剔出 CER 汇总，只走盲听）；过滤后共 21 条（th 5 / tl 4 / id 4 / zh 4 / en 4）。
  - code-switch：tl 5 条（`mag-book`、`Sea to Sky corridor`、`Yahoo! at Microsoft` 等）、id 4 条（`oke`、`like komentar dan subscribe`、`link` 等）。th 的 4 条是泰文夹英文专名，不是真 code-switch。（**存档**：ms 原只有 2 条 Manglish（`hotel`、`Konami`），vi 原有句尾 nặng 调 4 条（`vi_nat_14/26/27/28`，嘎裂声截断风险形态）——**这些行已随 2026-10-19 过滤从文件里移除**，形态描述只作历史记录的读法参考。）
- **ref 轮换**：ref 池 = `refs/ref_{zh,en,tl}_01..04.wav`（12 条，从 `fleurs_{zh,en,tl}_tc` val 挑的干净 wav，zh/en 3-6s、tl 窗口放宽到 2.5-8s）+ 原有 `prod_ref_fil.wav`，全局 round-robin，每条 case 带 `ref_lang`。**按 2026-10-19 过滤后口径**，三个目标语种的 ref_lang 覆盖各 9-12 条：th zh 10 / en 10 / tl 12，tl zh 10 / en 9 / tl 11，id zh 10 / en 10 / tl 12。ref 文件只在远端 `/root/autodl-tmp/eval_cases/`，不入库。
- **schema**：与 v1 相同（`case_id`/`lang`/`text`/`ref_audio`/`seed`/`note`，可选 `numeric`），新增 `ref_lang` 字段。seed 1100 起、与 v1 不重叠；`case_id` 用 `<lang>_nat_NN`。

### 统计（2026-10-19 过滤后，共 118 条）

| lang | n | numeric | code-switch | 时长 p50 |
|---|---|---|---|---|
| th | 32 | 5 | 4（专名） | 3.2s |
| tl | 30 | 4 | 5 | 11.1s |
| id | 32 | 4 | 4 | 3.6s |
| zh | 12 | 4 | — | 4.6s |
| en | 12 | 4 | — | 4.5s |

存档（过滤前 182 条口径，行已删除、数字不再可复现）：vi 32 条（numeric 4、nặng 尾 4、p50 3.4s）、ms 32 条（numeric 5、Manglish 2、p50 7.8s）。

### 后续动作

1. GPU 空下来后先跑 `eval base` 拿 v2 基线（cfg 1.8 / 20 步 / ≥3 seed，同 v1 口径），case 文件在远端 `/root/autodl-tmp/eval_cases/omnivoice_prod_v2.jsonl`。⚠️ **基线要按 118 条重新跑**：旧报告里含 vi/ms 的总体聚合不能直接沿用，只有 `by_lang` 的 th/tl/id 段可以跨轮对照。
2. **同一配置（同 base）重跑一次基线**，量 v2 自己的 run 间方差——新噪声门槛要按这个实测值定，不要沿用 0.005。
3. th/id 各 32 条、tl 30 条，ΔCER 的可信分辨率仍待实测确认；盲听仍按 `BLIND_LOSS_MARGIN`/`BLIND_MIN_LOSSES` 门槛。
4. 按 ref_lang 分组看结果（zh/en/tl ref 各自的分语种 CER），验证「ref 语言改变胜负语种」的结论在 12 条 ref 池上是否仍成立——三个目标语种各约 9-12 条/ref 语言，分组后单格样本量偏小，结论要按这个量级收着说。
5. 需要 vi/ms 结论时**只读归档报告**，不要试图重跑归档 case 文件（语种白名单会直接 `ValueError`）。

## 2026-10-19 过滤记录与归档清单

用户决定不再微调越南语（vi / VN）与马来语（ms / MY），目标语种收缩为 **th / tl / id**（外加 zh/en 防遗忘回放）。`eval` 的语种白名单（`SAMPLE_TEXTS` / `SAMPLE_BY_LANG`）同步收缩为 `th/tl/id/zh/en`，`WER_LANGS = {tl, en, id}`。

**已过滤的文件（纯删行，留下的行字节未变）：**

| 文件 | 变更 | 过滤后构成 |
|---|---|---|
| `omnivoice_prod_v2.jsonl` | 182 → **118**（删 vi 32 + ms 32） | th 32 / tl 30 / id 32 / zh 12 / en 12 |
| `fleurs_test_holdout_20260917.jsonl` | 150 → **90** | th / tl / id 各 30 |
| `acoustic_regressions_20260917.jsonl` | 3 → **2**（删 ms 1） | th 1 / id 1 |
| `refrob_20261012_*.jsonl`（6 份） | **删除并重建**为 `refrob_20261019_*.jsonl` | 6 条件 × 18 条，见 [ref_robustness_plan_20261012.md](ref_robustness_plan_20261012.md) |

⇒ **旧报告的总体聚合不再可复现**（含 base 总体 CER 0.0774 这类跨语种均值、以及「r8 的 267 条训练记录覆盖 95/182 条 v2 文本」里的 182 —— 那是**过滤前口径**，历史事实不改写）。但 `evaluate()` 逐 case 独立、`by_lang` 按语种聚合，**th/tl/id 的分语种数字仍然有效、仍可跨轮对照**。

**保持字节冻结、已归档且不可复跑的 case 文件**（含 vi/ms 行，再跑会直接 `ValueError`；这是刻意守卫，不改成静默跳过）：

- `omnivoice_prod.jsonl`（v1，28 条，含 vi 6 + ms 5）——生产口径基线 CER 0.0774 就出自这份
- `id_ms_prosody_probe.jsonl`
- `cfg_stability_feedback_20260924.jsonl`
- `production_adjacent_identity_20260923.jsonl`
- `production_vi_mtg6jbj_feedback_20260923.jsonl`
- `production_vi_roughness_feedback_20260923.jsonl`
- `roughness_model_ab_20260923*.json`（两份报告）

这些文件与其报告**照旧引用、不重跑、不重评分**；引用时注明「vi/ms 时期口径」。
