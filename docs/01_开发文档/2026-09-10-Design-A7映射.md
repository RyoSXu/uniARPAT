# Design-A7：映射v2

**状态**：L1完成，L2待讨论 ｜ **Backlog**：[Backlog待开发总清单.md](./Backlog待开发总清单.md)

## 要点
- L1：`getdata/raw/jarvis_mp_mapping_v2.json`（49,942对，零冲突已验）。
- L2：同成分分块 + StructureMatcher(ltol0.2/stol0.3/angle5°/原胞化/缩放) + L1标定阈值再上浮；磁性例外人工；上限2.6万条，约10万次调用一夜。
- L3：诚实标记只进单源。反向不找。
- 验收：L1标定F1>0.95；L2抽查100对>90%。
