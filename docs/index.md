# 文档导航

更新：2026-11-03。接手先读[状态](status.md) → [交接](handoff.md) → 与任务相关的契约与证据。
当前事实只维护在常用说明页；带日期文件记录当时协议与结果，不是执行队列。

## 常用说明

| 要了解什么 | 入口 |
|---|---|
| 当前目标、工作点、结果边界、待决定问题 | [status.md](status.md) |
| 新 agent／新机器接手、产物恢复、最小检查 | [handoff.md](handoff.md) |
| 当前模型、loss、选点与历史兼容 | [decisions.md](decisions.md) |
| 数据版本、字段、质量视图 | [data.md](data.md) |
| 数据处理、来源、发布与许可快照 | [中文数据卡](dataset-card-zh.md)、[English](dataset-card-en.md)、[来源声明](dataset-sources.md) |
| 方案、长时实验交接与结果核验 | [workflow.md](workflow.md)、[AGENTS.md](../AGENTS.md) |
| 代码调用关系、目录与 Git 边界 | [repository.md](repository.md) |
| 指标和候选简称 | [glossary.md](glossary.md) |
| 可执行工具和测试依赖 | [评估](../tools/eval/README.md)、[数据](../tools/data/README.md)、[测试](../tests/README.md) |

## 模型设计

| 主题 | 当前规范／关键结果 |
|---|---|
| 总体研究方向 | [路线图](design/design-2026-09-19-structural-refinement.md) |
| ZP、完整晶体与周期几何 | [初始化](design/design-2026-10-02-element-initialization.md)、[输入契约](design/design-2026-10-04-crystal-structure-input.md)、[邻居记录](design/design-2026-10-03-periodic-neighbor-records.md) |
| A/B+ 公式、模块边界及选型 | [公式](design/design-2026-10-04-periodic-message-candidates.md)、[接口](design/design-2026-10-04-periodic-message-module-boundaries.md)、[阶段选型结果](logs/log-2026-10-16-local-module-run.md) |
| 局部宽度、内部/外层 norm、完整/增量读出 | [W1](logs/log-2026-10-20-local-width-fresh-run.md)、[LN1](logs/log-2026-10-21-local-internal-norm-run.md)、[LN2](logs/log-2026-10-21-local-outer-norm-run.md)、[LR1](logs/log-2026-10-21-local-increment-run.md) |
| 当前完整 A 的 LR2＋E＋S | [设计](design/design-2026-10-30-spectral-residual.md)、[运行、独立核验与采用](logs/log-2026-10-30-spectral-residual-run.md) |
| 等权 blind E 与谱形辅助 | [E 协议](design/design-2026-10-23-balanced-blind-objective.md)、[E＋S 设计](design/design-2026-10-25-spectral-shape-supervision.md)、[原 A64 对照](logs/log-2026-10-26-spectral-shape-aux-run.md) |
| staged 与 midpoint 编排 | [设计](design/design-2026-11-01-local-global-layout.md)、[结果：保持 staged](logs/log-2026-11-01-local-global-layout-run.md) |
| 训练随机性、无结构参照与单任务 | [设计](design/design-2026-11-03-variance-and-reference-controls.md)、[准备与预检](logs/log-2026-11-03-reference-controls-preparation.md)、[完整运行、独立核验与判读](logs/log-2026-11-03-reference-controls-run.md) |
| 结构利用与数据上限 | [第一阶段诊断方案](design/design-2026-11-05-structure-use-and-data-ceiling.md)、[几何普查、噪声地板与描述符基线结果](logs/log-2026-11-05-structure-use-audit-run.md) |
| 结构分支泛化正则 | [第二阶段方案：几何扰动 G 与局部分支丢弃 D](design/design-2026-11-05-structure-generalization.md)、[实施与预检](logs/log-2026-11-05-structure-generalization-preparation.md) |
| E(3)、晶胞一致性与截断边界 | [输出要求](design/design-2026-10-04-e3-output-invariance.md)、[完整模型有限验收](logs/log-2026-11-02-crystal-model-invariance.md)、[全量边界统计](logs/log-2026-11-02-cutoff-boundary-census.md) |

## 当前问题的证据

| 问题 | 主要结果 |
|---|---|
| 局部更新是否有价值、为何曾弱于旧骨架 | [局部内容对照](logs/log-2026-10-09-local-structure-controls.md)、[无局部对照](logs/log-2026-10-12-global-only-run.md)、[旧/新骨架](logs/log-2026-10-13-legacy-backbone-run.md) |
| 三项全局骨架归因 | [norm](logs/log-2026-10-13-global-norm-run.md)、[注意力 dropout](logs/log-2026-10-13-attention-dropout-run.md)、[投影](logs/log-2026-10-13-attention-projection-run.md) |
| I1/I2/I3 与局部直连 I4 | [完整 B+](logs/log-2026-10-14-i1-bplus-run.md)、[完整 A](logs/log-2026-10-15-i1-a-run.md)、[直连读出](logs/log-2026-10-16-local-readout-run.md) |
| R²、blind/oracle、总量与标签语义 | [谱形/总量诊断](logs/log-2026-10-15-shape-total-diagnosis.md)、[科学口径审查](logs/log-2026-10-22-spectral-contract-audit.md) |
| 同组成谱差、阶段表示及数据可比性 | [材料对与几何](logs/log-2026-10-14-i1-pair-audit.md)、[冻结读出](logs/log-2026-10-27-shape-aux-stage-readout.md)、[组比较](logs/log-2026-10-27-shape-gap-comparability.md) |
| decoder、物理头、数据扩展与误差互补 | [双 decoder](logs/log-2026-10-28-split-decoder-run.md)、[物理先验](logs/log-2026-10-29-phonon-prior-run.md)、[结构盘点](logs/log-2026-10-28-structure-pool-inventory.md)、[已有模型平均](logs/log-2026-10-28-error-structure-ensemble.md) |

其余按日期/主题查 [design/](design/)、[logs/](logs/)；更早阶段在 [archive/docs/](../archive/docs/)。
历史本地产物并非都随 Git 提供：保留路径不等于文件已恢复或实验可立即重跑，见[交接边界](handoff.md)。
新记录使用[设计模板](design/template.md)与[日志模板](logs/template.md)，保留协议和结果之间的链接。
