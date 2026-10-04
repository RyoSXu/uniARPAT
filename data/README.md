# 当前数据

Q1 是唯一当前数据池；train/valid/test 为 18,706 / 2,313 / 2,287 条。
`train4ARPAT/` 保留机器本地数组与掩码，仅根部 manifest/stats 纳入 Git。
`reference/` 保存固定划分、网格、元素表、价电子/截断参考与排除清单。
`quarantine_q1.json` 是兼容链接，保留原 Q1 manifest 的旧清单路径。

旧数据与旧网格位于 [archive/data](../archive/data/)，不能替代 Q1。
输入契约和恢复限制见 [docs/data.md](../docs/data.md)，工具见 [tools/data](../tools/data/README.md)。
