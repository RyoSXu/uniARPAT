# 文档导航

本索引只列入口，不枚举每份历史设计和实验文件。历史材料保留在 `docs/design/`、`docs/logs/` 与 `results/`，不默认构成当前执行指令。

| 目的 | 文档 |
|---|---|
| 当前基线、最优指标参考和下一步 | [`status.md`](status.md) |
| 大方向路线图 | [`design/design-structural-refinement.md`](design/design-structural-refinement.md) |
| 当前 ZP 元素初始化、代码职责与使用限制 | [`design/design-element-initialization.md`](design/design-element-initialization.md) |
| 当前周期邻居记录规范与审查边界 | [`design/design-periodic-neighbor-records.md`](design/design-periodic-neighbor-records.md) |
| 协作与实验流程 | [`workflow.md`](workflow.md) |
| 长期决策和评估约定 | [`decisions.md`](decisions.md) |
| 数据、缓存及标签口径 | [`data.md`](data.md)；[`index/z0_REPORT.md`](../index/z0_REPORT.md) |
| 术语与指标 | [`glossary.md`](glossary.md) |

## 当前路线相关证据

- 元素入口实验： [A100/B100](logs/log-2026-09-30-element-identity-control.md)、
  [Z100](logs/log-2026-10-01-eid-zonly-baseline.md)、[ZP100](logs/log-2026-10-02-eid-zonlyproj.md)
- 纯 Z 训练差距诊断：[协议与完成状态](design/design-element-identity-diagnosis.md)、
  [原始报告](../results/eid_diagnosis_s42/20261002T095909Z/report.md)；解释边界见当前元素初始化文档
- 周期多体 encoder 单候选 Q1 valid 判决：[`logs/log-2026-09-29-periodic-manybody-encoder.md`](logs/log-2026-09-29-periodic-manybody-encoder.md)
- eDOS 误差与收益上界：[`logs/log-2026-09-28-model-accuracy-upper-bound.md`](logs/log-2026-09-28-model-accuracy-upper-bound.md)
- 结构条件谱形支持与同组成材料对：[`logs/log-2026-09-26-edos-spectral-support.md`](logs/log-2026-09-26-edos-spectral-support.md)
- eDOS-only 本地候选审计：[`logs/log-2026-09-28-d3a-edos-support-adjudication.md`](logs/log-2026-09-28-d3a-edos-support-adjudication.md)
- gamma 校准与尺度头候选复核：[`logs/log-2026-09-28-h1-gamma-calibration-gate.md`](logs/log-2026-09-28-h1-gamma-calibration-gate.md)、[`logs/log-2026-09-28-h1-next-candidate-review.md`](logs/log-2026-09-28-h1-next-candidate-review.md)

历史实验按主题在 `docs/logs/` 中检索；只有 `status.md` 明确列为当前工作的设计才是活跃执行依据。
