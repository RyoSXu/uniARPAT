# 日志：2026-11-05 — 结构分支泛化正则（G／D）的实施与预检

对应[设计](../design/design-2026-11-05-structure-generalization.md)。用户要求复核后按建议执行；
复核修改了三处：G 的幅度降为应变 ±1%／剪切 0.003；D 改为 DropPath 缩放；筛选参照改为 R42–R44 三 seed 均值。
本轮只实现、测试和短时预检，**未启动正式训练**，未读 test，未提交 Git。

## 实施

| 位置 | 改动 |
|---|---|
| `utils/structure_regularization.py` | G 的 `strain_perturb`（`cell(I+ε)ᵀ`，ε = u·I + 对称剪切；笛卡尔位移 0.02 Å，截断 ±0.06 Å，回卷到晶胞内）；D 的 `drop_local_residual`（每样本一次抽样，保留时乘 1/(1−p)）；两段契约文本 |
| `utils/dataset.py` | `strain_augment` 只在 train split 生效，需要显式晶体几何，与旧的分数坐标 `augment` 互斥 |
| `model/local_global_encoder.py`、`model/transformer.py` | `local_drop_prob`：只允许 staged LR2，最后一次局部调用之后对总残差做 DropPath；`eval()` 时不生效 |
| `utils/experiment_config.py`、`tools/train.py` | `--strain_augment`、`--local_drop_prob`；只在开启时写入训练记录、cli 和数据集或模型配置；G 运行时关闭几何缓存；非 E＋S 配方拒绝这两个选项 |
| `utils/spectral_balance_experiment.py` | `regularization_kind` 与 `validate_regularization`：只允许 staged LR2＋E＋S，G 与 D 不能同时开，p 固定为 0.25；契约新增 `regularization` 块 |
| `utils/local_residual_experiment.py` | 构建加法参照模型时去掉只在训练时生效的 `local_drop_prob` |
| `tools/eval/lg_valid_export.py` | 核对 `regularization` 块与训练记录、模型参数、数据集配置是否一致 |
| `tools/eval/structure_generalization_prepare.py`、`structure_generalization_analyze.py`、`tools/checks/check_structure_generalization.py` | 冻结准备；筛选与最终分析；逐臂 GPU 预检 |

- **默认行为与隔离。**两个选项默认关闭，历史训练记录不变。
  G、D 沿用 LR2 的模型 schema 和 `backbone_reference`，另记 `regularization` 块；训练记录与模型参数不同，因此不会和 LR2 checkpoint 混用。
- **初值不变。**seed42 初值与 R42 逐值相同（`924bd18a…`），因为 D 不新增参数、构造时不消耗随机数。

## 冻结准备

`results/runs/structure_generalization_20261105/report.json`（本地产物），tag 前缀 `sgen20261105`。
每个臂相对 R42 只改动声明过的字段：`strain_augment` 或 `local_drop_prob`、protocol 路径，以及 seed 43/44 臂的 seed 和数据顺序。

| 步骤 | 运行 |
|---|---|
| 第一步（筛选） | `g_s42`、`d_s42` |
| 第二步（只对晋级的臂） | G：`g_s43`、`g_s44`；D：`d_s43`、`d_s44` |

- 报告冻结了 6 条命令与输入哈希，覆盖 R42 契约、参照对照准备、数据和全部 model/utils 源码。
- 准备两次重做：GPU 预检第一版的 D 评估一致性检查写得不对，修正后在新目录重新准备，旧目录已移出 `results/`。
  - 第一版拿 D 和另建的无缓存模型比较，计算路径不同；改为同一模型切换开关。
  - 改后仍不为 0：同一模型在 GPU 上连续评估两次也差约 3e−6。改为与重复评估噪声比较，容差 1e−4。逐位相同由 CPU 单元测试保证。

## 验证

- **单元测试** `tests.pipeline.test_structure_regularization`，覆盖：
  - G 扰动的幅度上界、可复现、回卷到晶胞内、只作用于 train 且与旧扰动互斥；
  - DropPath 的丢弃比例与缩放；
  - D 在 `eval()` 下输出与不带 D 的 LR2 逐位相同、初值哈希相同；全部丢弃时局部参数梯度为 0、全局参数有梯度；
  - 配置的接受与拒绝、契约记录与默认记录不变、CLI 往返。
- 相关 8 个模块共 70 项测试通过。
- **GPU 预检**（V100），`results/runs/structure_generalization_preflight_20261105/report.json`，6 个臂全部通过：

| 臂 | 初值与 R42 相同 | 几何缓存 | 预检检查 | 每批秒数 | 相对热缓存 LR2 | 估计每轮秒数 |
|---|---|---|---|---:|---:|---:|
| g_s42 | 是 | 关 | 体积比 0.973–1.030，标签不变 | 0.584 | 1.15 | 479 |
| d_s42 | 是 | 开 | 开关前后评估差 7.7e−6（重复噪声 9.1e−6） | 0.510 | 1.00 | 418 |
| g_s43／g_s44 | 否（与 R43／R44 初值相同） | 关 | 体积比 0.971–1.030 | 0.54／0.53 | 1.06／1.03 | 443／431 |
| d_s43／d_s44 | 否（与 R43／R44 初值相同） | 开 | 评估差 ≤5.2e−6 | 0.46／0.45 | 0.91／0.88 | 379／369 |

- 热缓存 LR2 基线为 0.509 秒/批（冷缓存 0.812）。训练目标与评估误差之差 ≤1.6e−7。
- 20 批的峰值分配为 11.4–13.5 GiB；整轮峰值以历史 LR2 的约 24.6 GiB 为准。
- **成本估计**：每轮按 417 秒乘以成本比，35 轮时 G 约 4.2–4.7 小时、D 约 3.6–4.1 小时。
  第一步约 8.7 GPU 小时，两臂都晋级时第二步约 16.5 小时。正式训练由执行 agent 完成。
