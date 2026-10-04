# 数据约定

本页说明当前训练代码允许使用的数据。为保持简洁，获取过程保留在日志和 Git 历史中；可复现的
缓存边界只在此处维护。

## 当前训练数据池

唯一的生产数据池是 **Q1**：

| 划分 | 样本数 |
|---|---:|
| 训练 | 18,706 |
| 验证 | 2,313 |
| 测试 | 2,287 |

其清单文件为 `data/train4ARPAT/manifest.json`。Q1 排除了 1,682 条电子谱截断或覆盖范围标签
（coverage label）不可用的记录；排除清单为 `data/reference/q1_excluded_records.json`。旧数据池保留在本地
`archive/data/` 下，报告时必须标为 **pre-Q**，不得与 Q1 指标直接比较。

## 可复现边界

- 数据划分固定在 `data/reference/split_v2.yaml`。
- 价电子和截断参考表位于 `data/reference/valence_by_element.json`、`edos_valence_by_material.parquet` 和 `edos_truncation_by_material.parquet`。
- 网格定义为 `data/reference/grids.json`，审计为 `data/reference/c2b_grids_report.json`；生产使用 E0 + P0。非均匀网格需要非卷积
  解码器。
- 训练缓存文件是机器本地文件，且被 Git 忽略。清单与统计信息受版本控制，因此可检测缓存
  是否缺失。
- `stats_v2.json` 的 `n=24988` 是历史处理池统计，不能当作当前 Q1 三划分合计；Q1 合计为 23306。
- `archive/data/grids_c2b/` 只保留历史缓存。156 个链接中原有 30 个失效，部分组合混合
  Q1 结构与 pre-Q 标签行数；不能作为当前训练输入。迁移只保持原目标和失效状态，不恢复这些网格。

## 安全操作

通过训练入口使用 `data/train4ARPAT/`。如确实需要重建 Q1 缓存，先阅读
`tools/data/README.md`，再用 `tools/data/q1_rebuild.py` 并显式指定源和目标路径。
该脚本拒绝覆盖非空缓存，不会移动现有 Q1。它只是主数组重建辅助工具，不会自动恢复当前训练
所需的所有 sidecar（例如 `nvalence`）；完整恢复仍需单独批准并核对清单和全部字段。

原始数据获取不是常规训练的前提条件。保留工具见 `tools/data/README.md`；过去数据活动的
细节可从 Git 历史和带日期日志中恢复。

## eDOS-only 候选支持面裁决

本地集合 `E − (P ∪ C)` 有 121,420 个候选，但它不是获准训练的数据池。D3a 从中冻结抽取
1,000 条，按本地结构归属和 Q1 质量口径审计后，严格可用 507 条；它们没有实质缩短 Q1 valid
最远支持四分位的谱形距离。即使把全部 693 条可读、泛函兼容谱作为保守敏感性上界，595 条 Q4
样本也只有 9 条变近，中位改善仍为零。因此当前候选集合不进入 10–20k 数据 pilot，不修改 Q1
数据契约。完整边界见 [D3a 历史日志](../archive/docs/logs/log-2026-09-28-d3a-edos-support-adjudication.md)。

原 Q1 manifest 的旧排除清单路径通过 `data/quarantine_q1.json` 兼容链接保留；
manifest 字节和全部数组不因整理改写。预测与逐样本明细本地保存，Git 只开放有限元信息/证据。
