# 历史实验产物

`reports/` 放历史正式摘要与机器本地明细，`runs/` 放权重/有效配置/中间结果，
`figures/` 放历史图像。全部 checkpoint 保留；原始载荷和运行目录由 Git 忽略。
只有 `.gitignore` 精确开放的有限报告、汇总表和图像可进入新提交，单文件上限 2 MiB。

历史记录内部的旧路径和哈希保持原文，查找时使用[迁移登记](../metadata/repository-migration-2026-10-04.json)。
原来已被忽略的报告继续保持本地状态。复跑需要原输入、划分、checkpoint 和独立授权。
