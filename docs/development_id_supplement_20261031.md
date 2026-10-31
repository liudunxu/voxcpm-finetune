# 印尼语开发文本补充（尚未推理）

原fresh开发集的ID1753、ID1905含编码疑似，剩28个无编码风险独立文本。
新增清单`eval_cases/development_id_supplement_20261031.jsonl`只补两个开发文本，不修改原句、原报告或最终90文本。
ID1783和ID1961分别沿用这两个旧slot的参考音与plain/control条件，各5seed时为20输出/模型。
需要独立生成新文本的base；原编码风险句仍作为单列压力样本，不静默从历史统计删除。

复用`scripts/reserve_final_texts.py --id-supplement`，按已有确定性ID/原文字节哈希顺序选取，不读取候选分数挑句。
只读取远端缓存的FLEURS `id_id/test.tsv`，revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`、
TSV SHA `824cad6367283d95f545911eec051c488c182baa13a63227c6330af3c8f80741`，许可CC-BY-4.0。
未下载目标音频，未做ASR/TTS或训练，不将这些文本对应录音认证为可训练音频。

183份train/val/raw/eval清单、382份旧报告共121,577种归一文本参与排除。
574个输入SHA全部核验；独立遍历179,721条ID记录，精确匹配及≥40字符编辑距离≤2均0冲突。
新文本原字节/来源字段与TSV逐条一致；参考字节、ref_lang、control和condition与旧slot一致。
两个参考分别是既有中文和英文条件，不声明跨语言同人身份；新清单被全局训练文本排除器收录。
语义近重复、预训练暴露、说话人独立性和母语质量未认证。

| 产物 | SHA-256 |
| --- | --- |
| cases.jsonl | `d1ea31c3974b21fc0674c5aafa00296794cba9cdb5f9ff129e5f97ba4809ae16` |
| plan.json | `8a14f31ecd6776d66d38350f1337d4908d09efba76412815617b685c0d6b496b` |
| verification.json | `274ce4ec291fcd449828393350cf23dd8516fbfa96c582a9655396fda5df512d` |
| reserve_final_texts.py | `33aa3bf0ab86eeed9719509e431854385a0e27f7c7c40320719b54882b59609a` |

远端`/root/autodl-tmp/development_id_supplement_20261031/`保存脚本、启动日志及run产物，依赖已冻结的r39_early源码。
本地`work/development_id_supplement_20261031/`仅镜像元数据。原fresh清单SHA
`10566052db1254a68d8c2e295248b0283cde4e22b410e4762b9231339df55267`、最终90清单SHA
`cc7da8b05f6e78c92b5a7789e9cc95b0eadfb5047caf3287f3acf47b6b55fd0c`不变。

15项针对测试覆盖文本选择/条件继承/旧句不改/最终排除/新清单训练隔离及既有扩评协议。
这一步只补开发覆盖；当前r39_early完成前不改其冻结300任务，未产生工程候选或最终验收结论。
