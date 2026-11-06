# 日志：2026-11-06 — LR2＋E＋S 学习曲线的实施与预检

对应[设计](../design/design-2026-11-06-learning-curve.md)。本轮只实现、测试和短时预检，**未启动正式训练**，未读 test。

## 实施

| 位置 | 改动 |
|---|---|
| `utils/train_subset.py` | 按绝对组成整组、固定种子前缀抽样的嵌套子集；子集文件写入/校验（schema、数据 manifest 哈希、比例与重复次数）；契约块 |
| `utils/dataset.py` | `train_subset` 只作用于完整晶体 train 视图：先按 manifest 排除，再取子集并重复 `repeat` 次 |
| `utils/experiment_config.py`、`tools/train.py` | `--train_subset`；只在开启时写入训练记录、cli 与数据集配置；非 E＋S 配方拒绝 |
| `utils/spectral_balance_experiment.py` | `validate_learning_curve`：只允许 seed42 staged LR2＋E＋S；契约新增 `train_subset` |
| `utils/local_global_experiment.py` | 训练记录在未开启时不含 `train_subset`，历史契约逐键不变 |
| `tools/eval/lg_valid_export.py` | 导出时核对子集契约、训练记录、数据集配置与文件哈希 |
| 准备／预检／分析 | `tools/eval/learning_curve_prepare.py`、`tools/checks/check_learning_curve.py`、`tools/eval/learning_curve_analyze.py` |

## 冻结准备

`results/runs/learning_curve_20261106/report.json`（本地产物），tag 前缀 `lcurve20261106`，冻结 83 项输入哈希与源码快照。
每点相对 R42 只改 `train_subset` 与 protocol 路径。

| 点 | 材料／组 | 子集文件哈希 | tag |
|---|---|---|---|
| f25 | 4,676／4,193 | `fb81ec85…` | `_sbal_lcurve20261106_f25_e35` |
| f50 | 9,352／8,426 | `8b99167d…` | `_sbal_lcurve20261106_f50_e35` |

## 验证

- `tests.pipeline.test_learning_curve` 5 项通过；全量 `unittest discover -s tests` 共 510 项，失败的 4 项均因本机缺旧 Q1 `data/train4ARPAT`，与本改动无关。ruff 通过。
- GPU 预检（V100），`results/runs/learning_curve_20261106/preflight/report.json`，两点均通过：
  初值与 R42 相同（`924bd18a…`）；每轮 18,704 行、585 批；整组不跨边界、25% ⊂ 50%；valid 2,313 条不变；训练目标与评估之差 ≤5e−8。

| 点 | 冷／热缓存秒每批 | 估计训练秒每轮 | 20 批峰值分配 |
|---|---:|---:|---:|
| f25 | 0.764／0.474 | 277 | 12.3 GiB |
| f50 | 0.675／0.462 | 270 | 11.6 GiB |

- 成本：R43 每轮中位 421 秒（含 valid 评估）；子集缓存更快变热，估计每点约 4 小时，两点约 8 GPU 小时。
