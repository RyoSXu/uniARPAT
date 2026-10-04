# 历史实现与复现入口

`model/` 保存周期多体旧候选；`utils/` 保存 G1、宏晶格、pair 辅助和旧日志实现。
`tools/eval/` 保存旧诊断与实验工具，`tools/legacy/` 保存退役训练入口及 notebook；
`tests/` 保存相应数学和调用契约。模块使用 `archive.code...` 显式导入。

历史工具的运行版本已适配目录迁移；原文通过[迁移登记](../metadata/repository-migration-2026-10-04.json)
的 SHA-256 与快照/Git 提交恢复，不把适配后的运行文件当成当时源码原文。
独立执行脚本时使用其当前完整路径；先查看帮助和对应历史设计，确认输入及新输出位置。
涉及训练、资源测量或数据写入的旧命令仍需独立授权，不能仅因留存而自动执行。

`thermo_props.py` 保留旧五组批处理入口，但其默认输入在整理前已经缺失；
当前可导入的 `ThermodynamicCalculator` 位于 [utils/thermodynamics.py](../../utils/thermodynamics.py)，计算公式保留。
