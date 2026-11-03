# 数据约定

当前训练缓存的工程边界在本页维护；来源、处理细节、字段表和恢复步骤见[中文数据卡](dataset-card-zh.md)／[English](dataset-card-en.md)，许可核查快照见[来源声明](dataset-sources.md)。

## 当前数据与有效视图

默认输入为 `data/crystal_q1_v1/`，schema 为 `crystal-q1-v1`。成员、网格与划分冻结：

| 划分 | 磁盘存储 | 默认训练/评估视图 |
|---|---:|---:|
| train | 18,706 | 18,704 |
| valid | 2,313 | 2,313 |
| test | 2,287 | 2,287 |

完整存储 23,306 条、275,049 个原子，最大单结构 N=192；有效 train 为 220,413 原子，最大 N=160。
两条隔离材料 `mp-aaabidtd`（180 原子）、`mp-aaabilxy`（192 原子）仍保留在磁盘，训练视图按 manifest 过滤。
旧 Q1 曾截到 80 原子；修复版补全 55 条结构，修正 54 条组成回退 N_val。两条修正后的完整率低于既定 0.5 阈值，沿用质量规则隔离。
谱值、覆盖标签、划分和行序未改，不能把修复版训练称为旧 Q1 的同数据复现。[实施证据](logs/log-2026-10-04-crystal-structure-input.md)

当前 manifest SHA-256：`8e1b0adff516835d52d9fb0321dae5e66fb9a2ac5fd964d317e1dd8647de4b80`。
`manifest.json` 逐文件登记 dtype、shape、哈希、结构修复与训练排除；加载前用 `crystal_data_contract` 校验，而不是只检查目录存在。

## 字段与计算边界

- 几何：FP32 `cell[M,3,3]`，晶格行基矢、Å；FP32 `frac_coords[S,3]`，笛卡尔坐标为 `frac @ cell`。
- 原子：uint8 `atomic_numbers[S]`、int64 `atom_offsets[M+1]`、int32 `n_atoms[M]`；磁盘无晶格哨兵或原子 padding。
- 谱与 N_val：FP32；coverage：bool；ID：Unicode。每 split 为 11 个数组，逐行/偏移对齐，文件名与单位见数据卡。
- batch 按最大真实原子数 padding，再按原子对计算预算拆分，保留样本权重和梯度累积；不裁剪完整结构。
- 原生几何消费使用 FP32，周期邻居严格 `d<R`；cell 方向、反射和有效原子 mask 不得丢失。[输入契约](design/design-2026-10-04-crystal-structure-input.md)

生产网格来自 `data/reference/grids.json`：E0 为 [−6,6] eV／128 箱，P0 为 [−280,980] cm⁻¹／64 箱。
磁盘纵轴是处理后的 DOS 密度，不是每箱面积或统一每原子 DOS；重分箱不严格守恒，有限窗口面积不保证等于价电子数或 3N。

## 当前模型如何使用这些数据

- 只从有效 train 拟合每任务谱能量与参考尺度并冻结。当前每原子 blind 总量不使用 N_val；旧 H1 保留旧侧车语义。
- E＋S 使用相同数组：E 监督完整物理谱，S 监督非零谱的 SumNorm 形状；零谱 S=0、E 正常监督。[目标与选点](decisions.md#损失与最佳-checkpoint)
- coverage=True 仅指箱内有有限源采样点，False 不等于已认证的物理零或坏标签。当前 loss 与正式评分保留全窗，不据此删箱或删材料。
- 五条电子谱待核材料、N_val 计数来源差异与结构/DOS 计算几何不一致继续保留风险；未借协议更新重写标签或认证其正确性。[审查](logs/log-2026-10-22-spectral-contract-audit.md)
- 父划分按绝对原子计数组成分组，完整结构核查无材料 ID/绝对组成跨集重复，但存在约分组成跨集共享；历史父池级尖峰阈值未限 train 拟合。不能宣传成严格未见化学式泛化基准。

## 恢复、新数据与历史边界

当前发布候选只包含 33 个数组＋manifest 与 8 个参考文件，共 42 个文件，排除旧元素表、权重和训练结果。
发布准备与恢复记录见[打包日志](logs/log-2026-10-19-mainline-data-repack.md)；数据卡中的发布状态按原核查时点理解，不等于当前已发布附件。
训练数组被 Git 忽略；克隆仓库不会补齐数据或 checkpoint，按[交接清单](handoff.md)另存与验证。

原 Q1 `data/train4ARPAT/` 仅服务历史 B7/ZP100 与迁移对照；pre-Q、历史网格和 `stats_v2.json` 的 24,988 条父池统计不能替代当前数据。
固定参考与旧价电子/截断审计可能覆盖父池，必须按当前 ID 过滤；54 条修复值以当前 manifest/数组为准。
历史 eDOS-only 的 121,420 个候选不是获准数据池，也不代表本机仍有对应结构；[D3a 审查](../archive/docs/logs/log-2026-09-28-d3a-edos-support-adjudication.md)不自动否定所有数据扩展方向。

原始获取不是常规训练的前提。新采集、重建、划分或迁移先明确新数据方案，使用[数据工具](../tools/data/README.md)，拒绝覆盖冻结缓存。
`migrate_crystal_q1.py` 需要旧 Q1 与外部 raw；`q1_rebuild.py` 不能自动恢复全部 sidecar。仅有源码和摘要不等于能逐字节重建历史数据。
