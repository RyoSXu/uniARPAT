# 文档导航

本索引只列入口，不枚举每份历史设计和实验文件。历史材料统一保留在 `archive/` 的 `docs/code/data/results` 分类下，不默认构成当前执行指令。

当前先读 `status.md`，再按记录规范、候选公式、模块边界定位接口；执行证据按阶段查下方日志。
局部 CPU/GPU 验收已通过，完整模型接入和训练仍待另定方案。

| 目的 | 文档 |
|---|---|
| 当前基线、最优指标参考和下一步 | [`status.md`](status.md) |
| 目录职责、路径迁移、设计日期和 Git 忽略规则 | [`repository.md`](repository.md) |
| 设计与日志模板 | [`design/template.md`](design/template.md)、[`logs/template.md`](logs/template.md) |
| 大方向路线图 | [`design/design-2026-09-19-structural-refinement.md`](design/design-2026-09-19-structural-refinement.md) |
| 当前 ZP 元素初始化、代码职责与使用限制 | [`design/design-2026-10-02-element-initialization.md`](design/design-2026-10-02-element-initialization.md) |
| 当前周期邻居记录规范与审查边界 | [`design/design-2026-10-03-periodic-neighbor-records.md`](design/design-2026-10-03-periodic-neighbor-records.md) |
| 最终谱输出的 E(3) 目标、两条 Encoder 路线与证据边界 | [`design/design-2026-10-04-e3-output-invariance.md`](design/design-2026-10-04-e3-output-invariance.md) |
| 6.0 Å A/B 与增强 B+ 的公式、实际原型接口与取舍 | [`design/design-2026-10-04-periodic-message-candidates.md`](design/design-2026-10-04-periodic-message-candidates.md) |
| 已提取的物理记录/A/B+ 接口、职责边界与后续验收 | [`design/design-2026-10-04-periodic-message-module-boundaries.md`](design/design-2026-10-04-periodic-message-module-boundaries.md) |
| 协作与实验流程 | [`workflow.md`](workflow.md) |
| 长期决策和评估约定 | [`decisions.md`](decisions.md) |
| 数据、缓存及标签口径 | [`data.md`](data.md)；[`data/reference/edos-label-audit.md`](../data/reference/edos-label-audit.md) |
| 完整原子、FP32 存储与矩阵输入契约 | [`结构输入设计`](design/design-2026-10-04-crystal-structure-input.md)、[`实施日志`](logs/log-2026-10-04-crystal-structure-input.md) |
| 术语与指标 | [`glossary.md`](glossary.md) |

## 当前路线相关证据

- 周期消息工作按阶段归档、正式/中断目录与提交边界：[2026-10-04 归档记录](logs/log-2026-10-04-periodic-message-archive.md)
- A/B+ 的 GPU 数值、局部对称性、梯度与固定 batch CPU/GPU 成本：[2026-10-03 GPU 验收](logs/log-2026-10-03-periodic-message-gpu.md)
- A/B+ 核心提取、共同接口与梯度/依赖验收：[2026-10-03 解耦验收](logs/log-2026-10-03-periodic-message-modules.md)
- 增强 B+ 与 A/B 的参数、CPU 成本与受控分辨能力：[2026-10-03 高阶角度对照](logs/log-2026-10-03-periodic-message-high-order.md)
- 元素入口实验： [A100/B100](../archive/docs/logs/log-2026-09-30-element-identity-control.md)、
  [Z100](../archive/docs/logs/log-2026-10-01-eid-zonly-baseline.md)、[ZP100](logs/log-2026-10-02-eid-zonlyproj.md)
- 纯 Z 训练差距诊断：[协议与完成状态](../archive/docs/design/design-2026-10-02-element-identity-diagnosis.md)、
  [诊断报告与迁移导航](../results/reports/eid_diagnosis_s42/20261002T095909Z/README.md)；解释边界见当前元素初始化文档
- 周期多体 encoder 单候选 Q1 valid 判决：[`logs/log-2026-09-29-periodic-manybody-encoder.md`](../archive/docs/logs/log-2026-09-29-periodic-manybody-encoder.md)
- 当前 6.0 Å 几何验收与边界：[`logs/log-2026-10-03-periodic-geometry-acceptance.md`](logs/log-2026-10-03-periodic-geometry-acceptance.md)
- 两套局部消息的 CPU 预检、资源与表达限制：[`logs/log-2026-10-03-periodic-message-precheck.md`](logs/log-2026-10-03-periodic-message-precheck.md)
- eDOS 误差与收益上界：[`logs/log-2026-09-28-model-accuracy-upper-bound.md`](../archive/docs/logs/log-2026-09-28-model-accuracy-upper-bound.md)
- 结构条件谱形支持与同组成材料对：[`logs/log-2026-09-26-edos-spectral-support.md`](../archive/docs/logs/log-2026-09-26-edos-spectral-support.md)
- eDOS-only 本地候选审计：[`logs/log-2026-09-28-d3a-edos-support-adjudication.md`](../archive/docs/logs/log-2026-09-28-d3a-edos-support-adjudication.md)
- gamma 校准与尺度头候选复核：[`logs/log-2026-09-28-h1-gamma-calibration-gate.md`](../archive/docs/logs/log-2026-09-28-h1-gamma-calibration-gate.md)、[`logs/log-2026-09-28-h1-next-candidate-review.md`](../archive/docs/logs/log-2026-09-28-h1-next-candidate-review.md)

历史实验按主题在 `archive/docs/logs/` 中检索；只有 `status.md` 明确列为当前工作的设计才是活跃执行依据。

本次仓库整理与验证见[整理日志](logs/log-2026-10-04-repository-cleanup.md)，历史材料入口见[archive](../archive/README.md)。

提交与推送前核验见[提交准备日志](logs/log-2026-10-04-repository-publish.md)。
