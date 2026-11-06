# 日志：2026-11-06 — phDOS 输出头深度研究的实施与预检

对应[设计](../design/design-2026-11-06-phdos-head.md)。本轮只实现、测试和短时预检，**未启动正式训练**，未读 test。

## 实施

| 位置 | 改动 |
|---|---|
| `model/transformer.py` | `phdos_head_layers`（默认 None＝历史 6 层）；只允许 legacy LR2 读出取 1 或 2 |
| `utils/experiment_config.py`、`tools/train.py` | `--phdos_head_layers`；只在开启时写入模型参数与训练记录；非 E＋S 配方拒绝 |
| `utils/spectral_balance_experiment.py` | `validate_phdos_head`（只允许 seed42 staged LR2＋E＋S，不叠加其他臂）；`phdos_head_state` 在独立随机流中构造新头；初始化时替换头、复制其余全部张量；契约新增 `phdos_head` |
| `utils/local_residual_experiment.py` | 加法参照参数去掉头深度键 |
| `tools/eval/lg_valid_export.py` | 导出时核对头深度的契约、训练记录与模型参数一致 |
| 准备／预检／分析 | `tools/eval/phdos_head_prepare.py`、`tools/checks/check_phdos_head.py`、`tools/eval/phdos_head_analyze.py` |

## 冻结准备

`results/runs/phdos_head_20261106/report.json`（本地产物），tag 前缀 `phead20261106`，冻结 83 项输入哈希与源码快照；
每臂相对 R42 只改 `phdos_head_layers` 与 protocol 路径。

| 臂 | 头参数 | 全模型参数 | 初值哈希 | tag |
|---|---:|---:|---|---|
| h1_s42 | 1,537 | 45,395,652 | `914a6034…` | `_sbal_phead20261106_h1_s42_e35` |
| h2_s42 | 2,365,441 | 47,759,556 | `264d0a7d…` | `_sbal_phead20261106_h2_s42_e35` |

## 验证

- `tests.pipeline.test_phdos_head` 5 项通过；全量 `unittest discover -s tests -t .` 515 项，失败的 4 项均因本机缺旧 Q1 `data/train4ARPAT`，与本改动无关。ruff 通过。
- 默认配置初值仍为 R42 的 `924bd18a…`；两臂非头张量哈希相同（`72f0bf50…`）且等于 R42 初值去掉头后的哈希。
- GPU 预检（V100），`results/runs/phdos_head_20261106/preflight/report.json`：两臂冷／热缓存约 0.77／0.47、0.72／0.48 秒每批，
  估计训练约 278 秒每轮，峰值分配约 12.6 GiB；lr=0 训练目标与评估之差 ≤5e−8。
- 成本：每臂约 4 小时（含每轮 valid 评估），两臂约 8 GPU 小时。
