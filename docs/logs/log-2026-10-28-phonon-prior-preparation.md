# 日志：2026-10-28 — 物理先验加修正的实现与短检

对应[设计修订](../design/design-2026-10-28-phonon-lattice-head.md#修订物理先验加学到的修正2026-10-28-晚用户同意)。
只做实现与短检，正式训练未启动。

## 改动

| 位置 | 符号 | 作用 |
|---|---|---|
| `model/transformer.py` | `Transformer(phonon_correction=False)` | 开启时 phDOS 仍走原 decoder 与卷积读出，结果加到物理头 logits 上 |
| `utils/spectral_balance_experiment.py` | `_final_phdos_conv`、`initialize_balance_model`、`balance_model_contract` | 原读出最后一层卷积清零；契约 `joint-spectral-balance-phonon-prior-v1` |
| `tools/train.py`、`utils/experiment_config.py` | `--phonon_correction`、`ExperimentConfig.phonon_correction` | 只能与 `--phonon_head` 一起使用 |
| `tools/eval/lg_valid_export.py` | `load_run` | 核对修正开关与契约一致 |
| `tools/checks/check_phonon_head.py` | `--correction` | 修正版真实批次短检 |

## 验证

- 新增流水线测试：初值下修正版与纯物理头的 eDOS、phDOS、总量输出逐元素相同；
  除最后一层卷积外所有张量相同；一次更新后梯度到达被清零的卷积和物理头，训练目标减谱形辅助等于评估误差。
- 契约测试：新契约名、选点规则不变；缺少物理头时拒绝修正开关。
- 真实批次短检 `results/runs/phonon_prior_preflight_20261028/report.json`，前 3 个 train 批次：

| 项 | 物理先验加修正 | 谱形辅助对照 |
|---|---:|---:|
| 每步耗时（秒） | 2.38 / 1.41 / 1.26 | 0.61 / 0.75 / 0.72 |
| PyTorch 峰值分配 | 10,928 MiB | 11,246 MiB |

训练目标减谱形辅助与评估误差相差 1.7×10⁻⁷。纯物理头首轮实测每轮约 720 秒；
修正版每步相近，估计 35 轮约 7.5–8 小时，这仍是推断。

`tools/checks/check-static.sh` 共 564 项，唯一错误仍是既有测试找不到 `data/train4ARPAT/train/train_index.npy`。

## 接续核验（2026-10-28）

原短检绑定的5份源码哈希仍匹配；接续执行28项相关测试均通过。
另用32条真实train材料做一次性更新，覆盖H、He、160原子晶胞、最高原子密度和零phDOS材料；
损失、参数、梯度有限，物理头和修正读出都有梯度。未保存checkpoint或启动正式训练。
模式全部出窗的合成例仍可出现零梯度，短检不能替代全程稳定性或收益验证。
[完整核验与研究取舍](log-2026-10-28-cursor-takeover-review.md)
