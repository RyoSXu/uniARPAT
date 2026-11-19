# 日志：2026-11-18 — dos_v1 的 PhononDB 改为按原胞缩放

状态：数据集已重建并核对。基线未重训。对应此前在修正前标签上的[100 轮运行](log-2026-11-17-dosv1-h2-e100-run.md)。

## 改动

Phonopy 的声子态密度积分等于 3 乘以原胞原子数。2026-11-15 的构建用单胞原子数做缩放，单胞大于原胞时标签被缩小。结构匹配仍用 PhononDB 单胞。MP 声子和 eDOS 仍用计算胞原子数。没有原胞记录的 PhononDB 会跳过；这次跳过 0 条。

同一来源里“更接近每原子 3 个模式”的选择也改用这个面积。因此不是把旧标签统一乘一个常数。

数字 MP 编号到 8 位字母编号的转换不再依赖已从 emmet-core 移除的 `AlphaID`，规则与原先调用一致：`mp-1` → `mp-aaaaaaab`，`mp-149` → `mp-aaaaaaft`。

## 重建

删除 `views/`、`sealed/`、`tables/` 和旧 `report.json` 后，从 `spectral_db_v1` 重新跑 materials、records、match、assemble。材料标准化 154,377，与上次相同；体积接近但匹配失败仍是 323，低于 500 的停机线。记录 100,920。测试 `tests.pipeline.test_dos_v1` 9 项通过。

| 划分 | 样本 | eDOS | phDOS | 虚频标记 |
|---|---:|---:|---:|---:|
| train | 58,103 | 49,181 | 24,639 | 4,354 |
| valid | 7,474 | 6,342 | 3,080 | 518 |
| test | 7,451 | 6,312 | 3,227 | 537 |

phDOS 来源计数不变：MP pheasy 26,168、MP dfpt 334、PhononDB 4,444。train 少 2 个样本、eDOS 少 3 条，来自面积重排，不是新的剔除规则。主网格 C：eDOS 12.232，phDOS 0.04029（修正前 phDOS C 为 0.03909）。

valid 上每原子声子面积中位：MP 2.994，DFPT 3.000，PhononDB 2.974。PhononDB 低于 2 的比例从修正前的 35.8% 降到 2.5%，与 MP 的 2.1% 同量级。训练目录不含 test。

## 不能沿用的产物

11-17 的 checkpoint 和 `results/runs/dos_v1_h2_e100_throughput_20261116/pb51200/protocol.json` 绑的是修正前的数组和 C。不要用它们续训或当作新数据的协议。新基线需要重新准备协议后再从头训练。用户已确认：学习率用 100 轮余弦，训练在第 50 轮停止。`--epochs 50` 且不传余弦长度会换成另一条学习率曲线，不能这样准备。
