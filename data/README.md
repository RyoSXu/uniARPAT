# 数据目录

当前训练数据是 `datasets/dos_v1/`（2026-11-18 重建）。训练视图在 `datasets/dos_v1/views/dos_v1/`，说明见该目录 README。

`crystal_q1_v1/` 是历史修复版：存储 train/valid/test 为 18,706/2,313/2,287，有效训练视图 18,704。两条质量隔离记录仍在磁盘。
历史 `train4ARPAT/` 仅供旧模型复核，当前服务器未保留完整旧数组。
两套缓存的数组均为机器本地文件；Git只保留获准的清单/统计等小型元信息。
`reference/` 保存固定划分、网格、元素表、价电子/截断参考与排除清单。
`quarantine_q1.json` 是兼容链接，保留原 Q1 manifest 的旧清单路径。

旧数据与旧网格位于 [archive/data](../archive/data/)，不能替代 Q1。
输入契约和恢复限制见 [docs/data.md](../docs/data.md)，工具见 [tools/data](../tools/data/README.md)。
当前训练目标、总量与选点口径见[项目状态](../docs/status.md)，不由本目录的历史侧车决定。
