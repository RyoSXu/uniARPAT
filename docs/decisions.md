# 固定契约与历史边界

更新：2026-11-05。本页维护新开发须遵守的模型、数据与评价约定；结果及待决定事项只在[状态页](status.md)维护。

## 当前研究配方

| 部分 | 固定工作配置 |
|---|---|
| 元素入口 | ZP：原子序号 → Embedding → LayerNorm → Linear；参数继续训练 |
| 几何 | 完整 FP32 晶体，6 Å 严格 `d<R` 有向周期镜像记录；几何由局部消息利用，不另接旧方向注意力打分旁路 |
| 局部更新 | 完整 A 的 LR2 形式，标量/隐藏宽度 64/128；两次调用、每次两个内部块；关闭四个内部加法，保留内部 post-norm、外层 512 维 pre-norm 残差、完整状态读出 |
| 全局与编排 | staged：`L1 L2 G1 G2 G3 G4 G5 G6`；全局 post-norm、8 头、2048 维 GELU FFN；注意力概率 dropout=0，残差/FFN dropout=0.05；Q/K/V/O 恒等初值后继续学习 |
| 输出 | 六层共享 decoder；E0/P0 为 128/64 箱；结构预测非负、无固定上限的每原子窗口总量 |
| 训练预算 | M1、启用 q1_coord、q1_hidden=128、seed42、35 轮、batch32、lr5e-5、FP32；其他参数以冻结准备报告与有效配置为准 |

用户于 2026-10-31 确认采用，参照为 LR2 E＋S ep29；A64 E＋S ep25 是匹配对照。
这些是已检验的工作配置，不宣称每个细节最优或 A 总体优于 B+。
[采用记录](logs/log-2026-10-30-spectral-residual-run.md#采用决定2026-10-31)

## 损失与最佳 checkpoint

任务 t、材料 i 的误差为 `E_it = Δx_t × Σ_b[(ŷ_itb−y_itb)/N_i]² / C_t`；
`C_t = mean_train{Δx_t × Σ_b(y_itb/N_i)²}`，只用有效 train 拟合并冻结。
E 使用完整窗口，包括零箱和零谱；train 全零预测的平均 E 为 1，这不是单材料 R² 的零线。

- **训练 loss**：`0.5 mean(E_e+S_e) + 0.5 mean(E_p+S_p)`。S 为非零目标 SumNorm 形状的 KL＋W1＋Huber，权重各 1、Huber delta=0.02；零谱 S=0，仍计入 batch 分母并接受 E 监督。
- **blind 重建**：形状 softmax × N × 每原子窗口总量 / 箱宽；总量用 softplus 与冻结 train 参考尺度产生。它描述处理后窗口面积，不套价电子数或 3N 上限，不使用逐材料 N_val、目标总量或 coverage。
- **选点**：`balanced_blind_energy`，完整 valid 2,313 条。先分别满足 eDOS/phDOS 失败数≤196/99、中位 R²≥固定旧 A64 ep31−0.02，再最小化 `0.5 mean(E_e)+0.5 mean(E_p)`，并列保留较早轮次。S 不参与选点；保护线不随工作点更新。
- **没有合格轮次**：报告 `no_eligible_best`；不保存/宣称 best，不用 latest 替代。latest 只供按原契约恢复。
- **报告**：分任务给 E 均值/中位/P90、R² 中位/P10、失败数/率；Oracle 和同组成谱差仅作诊断。失败是每材料 R²<0，R²受目标方差与零背景影响，不能直接比较任务难度。

原公式与边界见[全谱协议](design/design-2026-10-23-balanced-blind-objective.md)及[谱形辅助](design/design-2026-10-25-spectral-shape-supervision.md)。
源码分别在 `utils/spectral_balance.py`、`model/losses.py` 与 `utils/checkpoint_selection.py`。
工作命令必须显式启用 `--local_route LR2 --scale_mode per_atom --spectral_shape_aux --selection_metric balanced_blind_energy`、冻结 `--spectral_balance_config` 及获准结构配置；CLI 历史默认不代表当前配方。

## 数据、输出与兼容性

- 输入使用[修复版 Q1](data.md)，固定完整原子、划分、行序和 E0/P0。缺失保持缺失；coverage 只说明有限源点覆盖，不认证物理零或标签可靠性，不用于当前 loss/正式评分删箱。
- 目标为 eDOS/phDOS 联合训练；单任务只作参照对照（`--single_task`，[结果](logs/log-2026-11-03-reference-controls-run.md#4-联合与单任务监督)），不是工作配方。推理输入来自晶体结构；额外数据、预训练及 eDOS-only 监督属于另行设计的研究方案，须明确来源、用途和泄漏防护。
- 最终谱要求 E(3) 不变，原子重排、等价晶胞与超胞一致性分别检验。区分几何验收、完整模型有限数值检查与架构保证。[输出要求](design/design-2026-10-04-e3-output-invariance.md)
- checkpoint 绑定数据、归一化、模型、loss、总量与选点契约；不同契约不得混用。冻结 JSON/YAML、指纹和原始报告不因文档整理改写。
- 用 train/valid 开发与选型；test 不用于调参。历史 test 已有诊断暴露，不能宣称完全未触碰。

## 历史参照与判读

| 历史项 | 保留用途与限制 |
|---|---|
| I1、同数据原 Encoder | 历史性能/架构参照；工作点更新不代表已超越其全部指标 |
| ZP100、B7 `_e9ctl` | 旧 Q1、100/35 轮、best89/33；跨版本与预算仅作历史参照，不作单因素因果比较 |
| H1、`balanced_oracle_mae`、`balanced_blind_r2` | 保留原总量、loss 与选点语义，不追溯改选已完成实验。[历史选点](logs/log-2026-10-15-joint-blind-selection.md) |

新候选执行前明确主目标、配对对照、预算、保护条件和采用判据；不沿用旧中位 R²“平局线”替代当前 E 判据。
**seed 尺度门槛**（用户于 2026-11-05 确认）：候选与工作点比较时，联合 mean E 点差的幅度须超过同配方
LR2＋E＋S 三 seed（42/43/44）极差 **0.001319**；只看单任务时，用该任务的极差（eDOS 0.000651、phDOS 0.003289）。
这是候选自身采用判据之外的必要条件，不能替代材料配对区间；未超过时写为“未分辨”。
两边都有多个 seed 时，比较 seed 均值。三 seed 极差只是粗略估计，不是置信界。
配方变化较大时（训练轮数、数据、loss 或选点不同），须在新配方下重测极差。
[依据](logs/log-2026-11-03-reference-controls-run.md#1-lr2-训练随机性)
材料配对区间条件于已选 checkpoint，不代表训练 seed 方差或消除 valid 反复选型偏差；同组成材料对按组成组处理相关性。
人决定是否采用；`park/closed` 只限定实际候选及实验范围，不自动否定更大研究方向，也不自动启动下一项。
