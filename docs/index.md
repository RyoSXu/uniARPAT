# 文档导航

先读[当前状态](status.md)。本页只提供入口；结论在状态页维护，过程、成本和验证记录在各实验日志维护。

## 日常说明

| 需要了解什么 | 入口 |
|---|---|
| 当前目标、参考模型、结论和下一步 | [status.md](status.md) |
| 已确定的数据、评估和对称性规则 | [decisions.md](decisions.md) |
| 数据版本、划分和字段 | [data.md](data.md) |
| 研究与实施流程 | [workflow.md](workflow.md) |
| 目录职责、兼容路径和 Git 边界 | [repository.md](repository.md) |
| 指标、模型和实验术语 | [glossary.md](glossary.md) |

## 模型设计

| 主题 | 入口 |
|---|---|
| 总体研究方向 | [路线图](design/design-2026-09-19-structural-refinement.md) |
| ZP 元素入口 | [初始化设计](design/design-2026-10-02-element-initialization.md) |
| 完整晶体输入与周期邻居 | [结构输入](design/design-2026-10-04-crystal-structure-input.md)、[物理记录](design/design-2026-10-03-periodic-neighbor-records.md) |
| A/B+ 局部消息与接口 | [候选公式](design/design-2026-10-04-periodic-message-candidates.md)、[模块边界](design/design-2026-10-04-periodic-message-module-boundaries.md) |
| 局部—全局 Encoder 与当前 I1 对照 | [接入设计](design/design-2026-10-04-local-global-encoder.md)、[投影对照](design/design-2026-10-13-attention-projection-attribution.md) |
| 最终输出的 E(3) 要求 | [对称性设计](design/design-2026-10-04-e3-output-invariance.md) |

## 当前问题的证据

| 问题 | 主要日志 |
|---|---|
| 路线/编排首轮表现与同数据 baseline | [四组对照](logs/log-2026-10-07-local-global-four-arm-s42.md)、[baseline](logs/log-2026-10-08-lg-baseline-q1v1.md) |
| 局部更新与新旧骨架的贡献 | [局部内容对照](logs/log-2026-10-09-local-structure-controls.md)、[无局部对照](logs/log-2026-10-12-global-only-run.md)、[旧/新骨架对照](logs/log-2026-10-13-legacy-backbone-run.md) |
| 三项全局设置的影响 | [归一化位置](logs/log-2026-10-13-global-norm-run.md)、[注意力 dropout](logs/log-2026-10-13-attention-dropout-run.md)、[投影初始化/学习](logs/log-2026-10-13-attention-projection-run.md) |
| 结构谱差的幅度、方向与可读性 | [泛化诊断](logs/log-2026-10-10-local-structure-attribution-diagnosis.md)、[逐阶段读出](logs/log-2026-10-11-stage-readout-diagnosis.md) |
| 小样本拟合与困难材料对 | [小样本拟合](logs/log-2026-10-12-local-structure-small-fit.md)、[困难对归因](logs/log-2026-10-12-hard-pair-attribution.md)、[损失隔离](logs/log-2026-10-12-local-structure-loss-isolation.md) |

其他设计与证据见 [design/](design/)、[logs/](logs/)；早期研究见 [archive/docs/](../archive/docs/)。历史待办不自动成为当前执行指令。
新文档使用[设计模板](design/template.md)或[日志模板](logs/template.md)。
