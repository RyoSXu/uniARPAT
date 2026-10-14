# 数据约定

本页说明当前训练代码允许使用的数据。为保持简洁，获取过程保留在日志和 Git 历史中；可复现的
缓存边界只在此处维护。

## 当前训练数据池

当前默认输入是 **Q1 完整结构修复版 `crystal-q1-v1`**，位于 `data/crystal_q1_v1/`。
它保存全部冻结 Q1 成员，并在新训练视图中沿用原质量规则隔离两条修复后不合格的记录：

| 划分 | 存储样本数 | 实际训练/评估视图 |
|---|---:|---:|
| 训练 | 18,706 | 18,704 |
| 验证 | 2,313 | 2,313 |
| 测试 | 2,287 | 2,287 |

新清单为 `data/crystal_q1_v1/manifest.json`；原 `data/train4ARPAT/manifest.json` 及全部数组保留，
仅用于历史 Q1/B7/ZP100 复核。原 Q1 排除了 1,682 条电子谱截断或覆盖范围标签
（coverage label）不可用的记录；排除清单为 `data/reference/q1_excluded_records.json`。旧数据池保留在本地
`archive/data/` 下，报告时必须标为 **pre-Q**，不得与 Q1 指标直接比较。

修复版补全 Q1 中 55 条受 80 原子上限影响的结构，保存 275,049 个完整原子，最大 N=192。
54 条组成回退 `N_val` 原来仅统计前 80 个原子，现按完整组成修复。由此 `mp-aaabidtd` 和
`mp-aaabilxy` 的电子谱完整率分别为 0.207431、0.172566，低于既定 0.5 阈值，在新 train
视图隔离；文件仍保留两条完整结构及全部谱数据供追溯。谱数组、覆盖标签、划分与行序不变。
新修复版须单独标注，不能称为历史 Q1 训练的同数据复现。

结构使用 FP32 `cell[M,3,3]`（行基矢、Å）和 `frac_coords[S,3]`；元素用 uint8
`atomic_numbers[S]`，配合 int64 `atom_offsets[M+1]` 和 int32 `n_atoms[M]`。
谱与价电子数均为 FP32，覆盖掩码为 bool。磁盘无晶格哨兵和原子填充，batch 只按真实最大
原子数填充；模型、`3N`、周期邻居均处理完整原子。完整字段与兼容范围见
[结构输入设计](design/design-2026-10-04-crystal-structure-input.md)，证据见
[实施日志](logs/log-2026-10-04-crystal-structure-input.md)。

## 可复现边界

- 数据划分固定在 `data/reference/split_v2.yaml`。
- 历史价电子与截断审计表位于 `data/reference/valence_by_element.json`、`edos_valence_by_material.parquet` 和 `edos_truncation_by_material.parquet`；新版本的 54 条修复值与两条质量隔离由新 manifest 逐条登记。
- 网格定义为 `data/reference/grids.json`，审计为 `data/reference/c2b_grids_report.json`；生产使用 E0 + P0。非均匀网格需要非卷积
  解码器。
- 训练缓存文件是机器本地文件，且被 Git 忽略。清单与统计信息受版本控制，因此可检测缓存
  是否缺失。
- `stats_v2.json` 的 `n=24988` 是历史处理池统计，不能当作当前 Q1 三划分合计；Q1 合计为 23306。
- `archive/data/grids_c2b/` 只保留历史缓存。156 个链接中原有 30 个失效，部分组合混合
  Q1 结构与 pre-Q 标签行数；不能作为当前训练输入。迁移只保持原目标和失效状态，不恢复这些网格。

## 安全操作

训练入口默认使用 `data/crystal_q1_v1/`，按原子对预算拆分计算并保留有效 batch 的样本权重。
新 checkpoint 绑定 manifest 指纹及归一化，不自动复用历史权重。
重建完整结构缓存使用 `tools/data/migrate_crystal_q1.py`，显式提供原 Q1、外部 raw 和全新目标目录；
拒绝覆盖已有目录，不下载外部数据。历史 Q1 如确需重建，先阅读
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
