# Design-A7：映射v2

**状态**：L1完成；L2方案已冻结，待执行 ｜ **Backlog**：[待开发总清单Backlog.md](./待开发总清单Backlog.md)

## 决议（2026-09-10讨论锁定）
- L1：`jarvis_mp_mapping_v2.json`（49,942对，与旧json零冲突）。
- L2：同成分分块 + StructureMatcher(ltol0.2/stol0.3/angle5°/原胞化/缩放) + L1藏reference标定阈值再上浮一档（错配毒性>漏配，宁缺毋滥）；磁性材料人工例外（rms=0但物理可不同）。
- L3：无对应（2D/缺陷/分子）诚实标记，只进单源。
- 反向不找（MP-only即可）；上限2.6万条约10万次调用一夜。
- 验收：L1标定F1>0.95；L2抽查100对（结构式+空间群）准确率>90%。
- 跨库id铁律：先canonical归一化再join（census三重bug教训）。
