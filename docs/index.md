# 文档导航

进度同步：2026-10-27。先读[当前状态](status.md)。本页只提供入口；结论在状态页维护，过程、成本和验证记录在各实验日志维护。

## 日常说明

| 需要了解什么 | 入口 |
|---|---|
| 当前目标、参考模型、结论和下一步 | [status.md](status.md) |
| 迁移到学校平台、实验断点与必带文件 | [2026-10-18迁移交接](logs/log-2026-10-18-platform-migration-handoff.md) |
| 已确定的数据、评估和对称性规则 | [decisions.md](decisions.md) |
| 数据版本、划分和字段 | [data.md](data.md) |
| 数据发布：中英文来源、处理、单位与许可 | [中文](dataset-card-zh.md)、[English](dataset-card-en.md)、[来源声明](dataset-sources.md) |
| 研究与实施流程 | [workflow.md](workflow.md) |
| 目录职责、兼容路径和 Git 边界 | [repository.md](repository.md) |
| 指标、模型和实验术语 | [glossary.md](glossary.md) |
| 历史联合blind R²选点与双任务报告 | [实现、验证与兼容边界](logs/log-2026-10-15-joint-blind-selection.md) |
| 当前总量与约束选点定义、纯E已完成对照 | [设计与科学定义](design/design-2026-10-23-balanced-blind-objective.md)、[准备与短检](logs/log-2026-10-23-balanced-blind-preparation.md)、[正式运行与核验](logs/log-2026-10-23-balanced-blind-run.md) |
| 当前研究工作配方：E＋独立谱形辅助（已完成） | [单候选设计](design/design-2026-10-25-spectral-shape-supervision.md)、[准备与短检](logs/log-2026-10-25-spectral-shape-preparation.md)、[正式运行与核验](logs/log-2026-10-26-spectral-shape-aux-run.md) |

## 模型设计

| 主题 | 入口 |
|---|---|
| 总体研究方向 | [路线图](design/design-2026-09-19-structural-refinement.md) |
| ZP 元素入口 | [初始化设计](design/design-2026-10-02-element-initialization.md) |
| 完整晶体输入与周期邻居 | [结构输入](design/design-2026-10-04-crystal-structure-input.md)、[物理记录](design/design-2026-10-03-periodic-neighbor-records.md) |
| A/B+ 局部消息与接口 | [候选公式](design/design-2026-10-04-periodic-message-candidates.md)、[模块边界](design/design-2026-10-04-periodic-message-module-boundaries.md) |
| A/B+阶段选型（暂定完整A，非总体胜出） | [复用范围、判据与预算](design/design-2026-10-16-local-module-selection.md)、[入口与短检](logs/log-2026-10-16-local-module-preparation.md)、[正式运行与结果](logs/log-2026-10-16-local-module-run.md) |
| 完整 A 内部容量 | [64/128与128/256方案](design/design-2026-10-16-local-a-width.md)、[迁移后准备](logs/log-2026-10-19-local-width-fresh-preparation.md)、[两组结果与核验](logs/log-2026-10-20-local-width-fresh-run.md) |
| 完整 A 内部归一化（已完成） | [pre/post-norm设计](design/design-2026-10-20-local-internal-norm.md)、[准备与短检](logs/log-2026-10-20-local-internal-norm-preparation.md)、[结果与独立核验](logs/log-2026-10-21-local-internal-norm-run.md) |
| 完整 A 外层512维归一化（已完成，未达采用条件） | [LN2设计](design/design-2026-10-21-local-outer-norm.md)、[准备与交接](logs/log-2026-10-21-local-outer-norm-preparation.md)、[正式运行与核验](logs/log-2026-10-21-local-outer-norm-run.md) |
| 完整 A 完整状态与增量读出（未达采用条件） | [LR1设计](design/design-2026-10-21-local-increment-readout.md)、[准备与交接](logs/log-2026-10-21-local-increment-preparation.md)、[运行与核验](logs/log-2026-10-21-local-increment-run.md) |
| 完整 A 内部残差LR2（旧协议满足数值条件，未并入新配方） | [LR2设计](design/design-2026-10-21-local-internal-residual.md)、[准备与交接](logs/log-2026-10-21-local-internal-residual-preparation.md)、[正式运行与核验](logs/log-2026-10-22-local-internal-residual-run.md) |
| 历史局部—全局接入与I1骨架对照 | [接入设计](design/design-2026-10-04-local-global-encoder.md)、[投影对照](design/design-2026-10-13-attention-projection-attribution.md) |
| I1性能参照上的完整 B+ 升级 | [实验设计](design/design-2026-10-14-i1-bplus-encoder.md)、[准备与短检](logs/log-2026-10-14-i1-bplus-preparation.md)、[运行与核验](logs/log-2026-10-14-i1-bplus-run.md) |
| 修正联合骨架上的完整 A/B+ 比较 | [完整 A 设计](design/design-2026-10-15-i1-a-encoder.md)、[准备与成本](logs/log-2026-10-15-i1-a-preparation.md)、[运行与核验](logs/log-2026-10-15-i1-a-run.md) |
| 最终输出的 E(3) 要求 | [对称性设计](design/design-2026-10-04-e3-output-invariance.md) |
| 完整 B+ 接口干预 | [局部状态直连读出](design/design-2026-10-15-local-readout-interface.md)、[实现与短检](logs/log-2026-10-15-local-readout-preparation.md)、[运行与核验](logs/log-2026-10-16-local-readout-run.md) |

## 当前问题的证据

| 问题 | 主要日志 |
|---|---|
| 路线/编排首轮表现与同数据 baseline | [四组对照](logs/log-2026-10-07-local-global-four-arm-s42.md)、[baseline](logs/log-2026-10-08-lg-baseline-q1v1.md) |
| 局部更新与新旧骨架的贡献 | [局部内容对照](logs/log-2026-10-09-local-structure-controls.md)、[无局部对照](logs/log-2026-10-12-global-only-run.md)、[旧/新骨架对照](logs/log-2026-10-13-legacy-backbone-run.md) |
| 三项全局设置的影响 | [归一化位置](logs/log-2026-10-13-global-norm-run.md)、[注意力 dropout](logs/log-2026-10-13-attention-dropout-run.md)、[投影初始化/学习](logs/log-2026-10-13-attention-projection-run.md) |
| 总体收益与结构谱差的幅度、方向、可读性 | [I1 收益与泛化诊断](logs/log-2026-10-14-i1-benefit-diagnosis.md)、[早期泛化诊断](logs/log-2026-10-10-local-structure-attribution-diagnosis.md)、[逐阶段读出](logs/log-2026-10-11-stage-readout-diagnosis.md) |
| 材料对可比性、标签几何与训练环境支持 | [材料对审查与几何替换](logs/log-2026-10-14-i1-pair-audit.md) |
| 完整局部模块的收益是否被总量预测掩盖 | [谱形与总量交叉诊断](logs/log-2026-10-15-shape-total-diagnosis.md) |
| R²、失败率、eDOS尺度上限与谱段覆盖的科学含义 | [只读语义审查与下一步](logs/log-2026-10-22-spectral-contract-audit.md) |
| 小样本拟合与困难材料对 | [小样本拟合](logs/log-2026-10-12-local-structure-small-fit.md)、[困难对归因](logs/log-2026-10-12-hard-pair-attribution.md)、[损失隔离](logs/log-2026-10-12-local-structure-loss-isolation.md) |
| 谱形辅助 checkpoint 上同组成谱差停在哪一层 | [冻结逐阶段读出设计](design/design-2026-10-27-shape-aux-stage-readout.md)、[正式结果](logs/log-2026-10-27-shape-aux-stage-readout.md) |
| 验证组是不是比训练成功组更小的结构差 | [比较设计](design/design-2026-10-27-shape-gap-comparability.md)、[结果](logs/log-2026-10-27-shape-gap-comparability.md) |
| 共享 decoder 拆成两套是否提高联合误差 | [方案](design/design-2026-10-28-split-decoder.md)、[结果：不采用](logs/log-2026-10-28-split-decoder-run.md) |
| 本机还有没有 Q1 以外的晶体结构 | [只读盘点](logs/log-2026-10-28-structure-pool-inventory.md) |
| 误差主要在哪、单模型随机差异有多大 | [误差结构与已有模型平均](logs/log-2026-10-28-error-structure-ensemble.md) |
| 声子谱改由学习弹簧加精确晶格动力学给出 | [设计](design/design-2026-10-28-phonon-lattice-head.md)、[实现与短检](logs/log-2026-10-28-phonon-head-preparation.md)，正式训练未启动 |

其他设计与证据见 [design/](design/)、[logs/](logs/)；早期研究见 [archive/docs/](../archive/docs/)。历史待办不自动成为当前执行指令。
新文档使用[设计模板](design/template.md)或[日志模板](logs/template.md)。
