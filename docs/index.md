# 文档导航

更新：2026-11-18。接手先读[状态](status.md) → [交接](handoff.md) → 与任务相关的契约与证据。
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
| 学习曲线（数据量是否为瓶颈） | [方案](design/design-2026-11-06-learning-curve.md)、[实施与预检](logs/log-2026-11-06-learning-curve-preparation.md) |
| phDOS 输出头深度 | [方案](design/design-2026-11-06-phdos-head.md)、[实施与预检](logs/log-2026-11-06-phdos-head-preparation.md)、[运行与结果](logs/log-2026-11-08-phdos-head-run.md) |
| MACE 冻结特征参照 | [方案](design/design-2026-11-08-mace-features.md)、[实施与预检](logs/log-2026-11-08-mace-features-preparation.md)、[运行与结果](logs/log-2026-11-09-mace-features-run.md) |
| MACE 收益归因 M0（仅元素先验） | [方案](design/design-2026-11-09-mace-m0.md)、[实施与预检](logs/log-2026-11-09-mace-m0-preparation.md)、[运行与结果](logs/log-2026-11-09-mace-m0-run.md) |
| phDOS 的 uMLIP＋phonopy 基线（路线阶段 A） | [方案](design/design-2026-11-09-umlip-phonon-baseline.md)、[运行与结果](logs/log-2026-11-09-umlip-phonon-baseline-run.md) |
| 由 DOS 推出的物理量（热容、熵、带隙、金属判断） | [结果](logs/log-2026-11-09-physical-quantities.md) |
| 多来源、部分标签的谱数据扩充 | [方案与用户决定](design/design-2026-11-10-partial-label-data.md)、[P0 谱数据库设计](design/design-2026-11-10-spectral-store-p0.md)、[建库与核验](logs/log-2026-11-10-spectral-store-build.md)、[补充标签平移检查](logs/log-2026-11-10-fallback-shift-check.md)、[标签契约 v2（精确积分分箱）](design/design-2026-11-11-label-contract-v2.md)、[eDOS 标签质量（分辨率与 Static）](logs/log-2026-11-11-edos-label-quality.md)、[异常谱筛除规则（已批准）](logs/log-2026-11-11-spectrum-anomaly-rule.md)、[第 2 步：旧材料 v2 标签导出](logs/log-2026-11-11-label-v2-export.md)、[第 4 步：范围、去重与切分方案（已批准）](design/design-2026-11-11-splits-v2.md)、[切分 v2 构建、视图与训练侧部分标签支持](logs/log-2026-11-12-splits-v2-build.md)、[v3 上的 h2 基线](logs/log-2026-11-13-s1-v3-baseline-run.md)、[视图 s1_v3_1（统一声子标签规则）](logs/log-2026-11-13-s1-v3-1-build.md)、[视图 s1_v3_2（结构偏离规则与标签来源表）](logs/log-2026-11-14-s1-v3-2-build.md)、[谱数据库 v1（只有来源事实）](logs/log-2026-11-15-spectral-db-v1.md)、[数据集 dos_v1](logs/log-2026-11-15-dos-v1.md)、[PhononDB 原胞缩放](logs/log-2026-11-18-dos-v1-phonondb-norm.md) |
| Mat2Spec 外部基线 | [方案](design/design-2026-11-10-mat2spec-baseline.md)、[运行与结果](logs/log-2026-11-10-mat2spec-baseline-run.md) |
| E(3)、晶胞一致性与截断边界 | [输出要求](design/design-2026-10-04-e3-output-invariance.md)、[完整模型有限验收](logs/log-2026-11-02-crystal-model-invariance.md)、[全量边界统计](logs/log-2026-11-02-cutoff-boundary-census.md) |

## 当前问题的证据

当前数据是 2026-11-18 重建的 dos_v1，基线尚未重训：[重建](logs/log-2026-11-18-dos-v1-phonondb-norm.md)。
11-17 的 100 轮用的是修正前标签，checkpoint 不能复用：[方案](design/design-2026-11-16-dos-v1-h2-e100.md)、[准备](logs/log-2026-11-16-dos-v1-h2-e100-preparation.md)、[运行](logs/log-2026-11-17-dosv1-h2-e100-run.md)。更早的 35 轮准备因 OOM 停在预检：[方案](design/design-2026-11-16-dos-v1-baseline.md)、[预检](logs/log-2026-11-16-dos-v1-baseline-preparation.md)。

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
