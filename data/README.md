# 当前数据

当前默认数据为完整结构修复版`crystal_q1_v1/`；存储train/valid/test为18,706/2,313/2,287条，
有效训练视图为18,704条，两条质量隔离记录仍保留在磁盘。谱、mask与划分未因新loss重建。
历史`train4ARPAT/`仅供旧模型复核，当前服务器未保留完整旧数组，不能自动替代默认输入。
两套缓存的数组均为机器本地文件；Git只保留获准的清单/统计等小型元信息。
`reference/` 保存固定划分、网格、元素表、价电子/截断参考与排除清单。
`quarantine_q1.json` 是兼容链接，保留原 Q1 manifest 的旧清单路径。

旧数据与旧网格位于 [archive/data](../archive/data/)，不能替代 Q1。
输入契约和恢复限制见 [docs/data.md](../docs/data.md)，工具见 [tools/data](../tools/data/README.md)。
当前训练目标、总量与选点口径见[项目状态](../docs/status.md)，不由本目录的历史侧车决定。
