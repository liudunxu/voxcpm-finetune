<!-- 由 scripts/eval_report_index.py 生成；报告文件在 gitignored 的 work/rescore/（本机）与 /root/autodl-tmp/voxft_ckpt/eval/（远端）。 -->
<!-- 重新生成：uv run python -X utf8 scripts/eval_report_index.py work/rescore > docs/eval_reports_index.md -->

# 评测报告索引：work/rescore（合并件 35 份，分片 0 份）

分类只依据报告内容（case_id 前缀 / 语种 / 条数 / seed），不依据文件名。

| 模型 label | drama_chain | drama_chain_prod | id_prosody_probe | omnivoice_prod_v2 |
|---|---|---|---|---|
| `base` | 336条/3seed `base_0225cfdff93c4de591f456c…` | 336条/3seed `base_dramaprod.json…` | 54条/3seed `base_c529323baaf34af3a90845f…` | 354条/3seed `base_b34db7c942ec4464ae04543…` |
| `lora_omni5_r13x_idemph_3ep_latest` | 336条/3seed `lora_omni5_r13x_idemph_3ep_l…` | 336条/3seed `r13x_dramaprod.json…` | — | 354条/3seed `lora_omni5_r13x_idemph_3ep_l…` |
| `lora_omni5_r18_idaccent_3ep_latest` | 336条/3seed `lora_omni5_r18_idaccent_3ep_…` | 336条/3seed `lora_omni5_r18_idaccent_3ep_…` | 54条/3seed `lora_omni5_r18_idaccent_3ep_…` | 354条/3seed `lora_omni5_r18_idaccent_3ep_…` |
| `lora_omni5_r19_emotion_3ep_latest` | — | 336条/3seed `lora_omni5_r19_emotion_3ep_l…` | 54条/3seed `r19_idprobe.json…` | 354条/3seed `r19_v2_merged.json…` |
| `lora_omni5_r20_emofix_3ep_latest` | — | 336条/3seed `r20_dramaprod.json…` | 54条/3seed `r20_idprobe.json…` | 354条/3seed `r20_v2.json…` |
| `lora_omni5_r21_nothaiser_cont_latest` | — | 336条/3seed `r21_dramaprod.json…` | 54条/3seed `r21_idprobe.json…` | 354条/3seed `r21_v2.json…` |
| `lora_omni5_r23_thtlboost_cont_latest` | — | 336条/3seed `r23_dramaprod.json…` | 54条/3seed `lora_omni5_r23_thtlboost_con…`（另有 1 份） | 354条/3seed `lora_omni5_r23_thtlboost_con…`（另有 1 份） |
| `lora_omni5_r24_coldstart_latest` | — | 336条/3seed `lora_omni5_r24_coldstart_lat…` | 54条/3seed `lora_omni5_r24_coldstart_lat…` | 354条/3seed `lora_omni5_r24_coldstart_lat…` |
| `lora_omni5_r25_r18_prefix_graft_latest` | — | 336条/3seed `lora_omni5_r25_r18_prefix_gr…` | 54条/3seed `lora_omni5_r25_r18_prefix_gr…` | 354条/3seed `lora_omni5_r25_r18_prefix_gr…` |
| `lora_omni5_r26_r18_inoc10_latest` | — | 336条/3seed `lora_omni5_r26_r18_inoc10_la…` | 54条/3seed `lora_omni5_r26_r18_inoc10_la…` | 354条/3seed `lora_omni5_r26_r18_inoc10_la…` |
| `lora_omni5_r27_scale25h_latest` | — | 336条/3seed `lora_omni5_r27_scale25h_late…` | — | — |

## 逐份明细（合并件）

| 文件 | label | 验收集 | items | cases | seeds | 语种 | cfg/steps |
|---|---|---|---|---|---|---|---|
| `base_0225cfdff93c4de591f456cd6ebb7119.json` | base | drama_chain | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 str0.0 |
| `lora_omni5_r13x_idemph_3ep_latest_6140b10f7c23455084f7d9e59c033a37.json` | lora_omni5_r13x_idemph_3ep_latest | drama_chain | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r18_idaccent_3ep_latest_335be653f1304797bc5fd7573114a60e.json` | lora_omni5_r18_idaccent_3ep_latest | drama_chain | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `base_dramaprod.json` | base | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 str0.0 |
| `r13x_dramaprod.json` | lora_omni5_r13x_idemph_3ep_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r18_idaccent_3ep_latest_744c4228c9a74a2884b8e476c1986349.json` | lora_omni5_r18_idaccent_3ep_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r19_emotion_3ep_latest_f89e8d6684e94f8e94cb7ae09f2b4064.json` | lora_omni5_r19_emotion_3ep_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `r20_dramaprod.json` | lora_omni5_r20_emofix_3ep_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `r21_dramaprod.json` | lora_omni5_r21_nothaiser_cont_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `r23_dramaprod.json` | lora_omni5_r23_thtlboost_cont_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r24_coldstart_latest_9462d1a6a265417bb931a267c69bd78a.json` | lora_omni5_r24_coldstart_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r25_r18_prefix_graft_latest_533cc495a29a465dbc7b58ab6a0fc055.json` | lora_omni5_r25_r18_prefix_graft_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r26_r18_inoc10_latest_12d7250beeee45bdbf67a4aef61e1a6a.json` | lora_omni5_r26_r18_inoc10_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `lora_omni5_r27_scale25h_latest_fb0ba87e322e4659bcbbc039f5ecfba1.json` | lora_omni5_r27_scale25h_latest | drama_chain_prod | 336 | 112 | 42,43,44 | en/id/th/tl | 1.8/20 |
| `base_c529323baaf34af3a90845fa87ad36bb.json` | base | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 str0.0 |
| `lora_omni5_r18_idaccent_3ep_latest_9385e915bf4d46e2b640b549446e5fe8.json` | lora_omni5_r18_idaccent_3ep_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `r19_idprobe.json` | lora_omni5_r19_emotion_3ep_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `r20_idprobe.json` | lora_omni5_r20_emofix_3ep_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `r21_idprobe.json` | lora_omni5_r21_nothaiser_cont_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `lora_omni5_r23_thtlboost_cont_latest_dd5f5c56755149bebed5e4921e98559f.json` | lora_omni5_r23_thtlboost_cont_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `r23_idprobe.json` | lora_omni5_r23_thtlboost_cont_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `lora_omni5_r24_coldstart_latest_bec165693e9b4edea8ae40d8c425b5b9.json` | lora_omni5_r24_coldstart_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `lora_omni5_r25_r18_prefix_graft_latest_f6da45869f394e019b4fb6362aefa35d.json` | lora_omni5_r25_r18_prefix_graft_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `lora_omni5_r26_r18_inoc10_latest_561b62ba50d24ee291bc5c9f1acfeb4e.json` | lora_omni5_r26_r18_inoc10_latest | id_prosody_probe | 54 | 18 | 42,43,44 | id | 1.8/20 |
| `base_b34db7c942ec4464ae04543c02bb8f9a.json` | base | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 str0.0 |
| `lora_omni5_r13x_idemph_3ep_latest_2bbd5a05aa8b4ed4a55073b011871c95.json` | lora_omni5_r13x_idemph_3ep_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `lora_omni5_r18_idaccent_3ep_latest_14e9edd24d5a441288a77e1614b8e547.json` | lora_omni5_r18_idaccent_3ep_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `r19_v2_merged.json` | lora_omni5_r19_emotion_3ep_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `r20_v2.json` | lora_omni5_r20_emofix_3ep_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `r21_v2.json` | lora_omni5_r21_nothaiser_cont_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `lora_omni5_r23_thtlboost_cont_latest_4dd161ef2217403d8d415a7eb534c312.json` | lora_omni5_r23_thtlboost_cont_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `r23_v2.json` | lora_omni5_r23_thtlboost_cont_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `lora_omni5_r24_coldstart_latest_556be769563640aa9bf96689218edd70.json` | lora_omni5_r24_coldstart_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `lora_omni5_r25_r18_prefix_graft_latest_3ee97cda229642bca1c6c8d97e1028c7.json` | lora_omni5_r25_r18_prefix_graft_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |
| `lora_omni5_r26_r18_inoc10_latest_cb544bbe936c44f3b91e0f7baa936b58.json` | lora_omni5_r26_r18_inoc10_latest | omnivoice_prod_v2 | 354 | 118 | 42,43,44 | en/id/th/tl/zh | 1.8/20 |

⚠️ 同一 (label, 验收集) 有多份合并报告（引用前必须确认用哪一份）：
  - lora_omni5_r23_thtlboost_cont_latest / id_prosody_probe: ['lora_omni5_r23_thtlboost_cont_latest_dd5f5c56755149bebed5e4921e98559f.json', 'r23_idprobe.json']
  - lora_omni5_r23_thtlboost_cont_latest / omnivoice_prod_v2: ['lora_omni5_r23_thtlboost_cont_latest_4dd161ef2217403d8d415a7eb534c312.json', 'r23_v2.json']
