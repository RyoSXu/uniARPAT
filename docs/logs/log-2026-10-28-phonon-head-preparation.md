# 日志：2026-10-28 — 声子晶格动力学头的实现与短检

对应[设计](../design/design-2026-10-28-phonon-lattice-head.md)。只做实现、单元测试和真实批次短检，正式 35 轮未启动。

## 改动

| 位置 | 符号 | 作用 |
|---|---|---|
| `model/phonon_head.py` | `PhononSpringHead`、`harmonic_frequencies`、`binned_modes`、`mesh_sizes`、`shifted_mesh`、`atomic_mass_table` | 学习弹簧，组装质量加权动力学矩阵，对角化后按高斯展宽积分到 phDOS 网格，输出归一化谱形的对数 |
| `model/transformer.py` | `Transformer(phonon_head=False)` | 开启时最后构造物理头；phDOS 不再调用共享 decoder 和原输出头，改由物理头读取编码器状态与周期邻居记录 |
| `utils/spectral_balance_experiment.py` | `_backbone_params`、`validate_balance_config`、`balance_model_contract`、`initialize_balance_model` | 只允许在谱形辅助配方上单独开启；契约 `joint-spectral-balance-phonon-head-v1`；A64 张量原样复制，物理头按 seed 单独初始化 |
| `tools/train.py`、`utils/experiment_config.py` | `--phonon_head`、`ExperimentConfig.phonon_head` | 显式开关；关闭时训练记录和配置与原来逐字节相同 |
| `tools/eval/lg_valid_export.py` | `load_run` | 导出时核对开关、契约和模型参数一致 |
| `tools/checks/check_phonon_head.py` | `check` | 真实批次短检，不写 checkpoint，不读 test |

## 数据事实

phDOS 标签在 −280 到 980 cm⁻¹ 窗口内的总量，相对 3N 的中位数为 0.863，第 10 百分位 0.618，第 1 百分位 0.382（train 与 valid 相近）。
因此物理头只给谱形，总量仍由每原子总量头决定。train 标签平均频率 198 cm⁻¹，中位 171 cm⁻¹。

## 验证

- `tests.model.test_phonon_head` 7 项：简单立方晶格布里渊区边界的解析频率；Γ 点三个声学模为零；
  质量加 4 倍频率减半；全频窗内恰好 3N 个模式；负刚度给出虚频且梯度有限；
  旋转加反射、整体平移和填充槽位不改变输出；梯度到达原子状态和展宽参数。
- `tests.pipeline.test_spectral_balance` 新增 2 项：所有 A64 张量逐一相同、物理头初值可复现；
  训练目标减去谱形辅助后等于评估误差，物理头拿到有限梯度，旧 phDOS 输出头没有梯度。
- `tests.pipeline.test_spectral_shape_aux` 补充契约检查：新契约名、选点规则不变、与双 decoder 或缺少谱形辅助的组合被拒绝。
- 真实批次短检 `results/runs/phonon_head_preflight_20261028/report.json`，Tesla V100、前 3 个 train 批次：

| 项 | 物理头候选 | 谱形辅助对照 |
|---|---:|---:|
| 每步耗时（秒） | 3.12 / 1.67 / 1.20 | 0.62 / 0.76 / 0.72 |
| PyTorch 峰值分配 | 9,693 MiB | 11,121 MiB |

训练目标减谱形辅助与评估误差相差 1.4×10⁻⁸。3 次更新后展宽 19.7 cm⁻¹，
预测平均频率 124 cm⁻¹，0 以下权重 0.3%。

首批含预热。按后两批估计每步约 1.7 倍到 2.3 倍，35 轮大约 7–9 小时；这是 3 个批次的推断，不是实测全程。

`tools/checks/check-static.sh` 共 561 项，唯一错误是既有测试找不到 `data/train4ARPAT/train/train_index.npy`，与本改动无关。
