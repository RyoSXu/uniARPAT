# 全局骨架拆分第二步：仅关闭注意力概率 dropout

创建日期：2026-10-13。用户确认继续，在 N1 上只关闭全局注意力概率 dropout。
实现与短测由 Codex 完成，正式训练继续交由 OpenCode。已有证据见
[N1 结果与独立核验](../logs/log-2026-10-13-global-norm-run.md)，本次准入见
[D1 实施日志](../logs/log-2026-10-13-attention-dropout-preflight.md)。

## 问题与直接对照

N1 相对 C1 的 phDOS R²/MAE 改善可分辨，eDOS oracle 失败率也改善；相对 O1，
eDOS blind R² 与两口径失败率仍较差。剩余差异包括 learned Q/K/V/O、概率 dropout、
内部初值及其交互，不能由 N1 结果自动归为某一个因素。

代码确认旧内容注意力在 softmax 后直接聚合，没有概率 dropout；N1 的
`MultiheadAttention.dropout` 为 0.05。这一项不同于注意力输出残差分支、FFN
中间层、FFN 输出残差分支和 decoder 的 dropout。

| 组 | 全局归一化 | 注意力概率 dropout | 其他 dropout | 未训练初值 |
|---|---|---|---|---|
| N1（复用） | post-norm | 0.05 | 原配方 | 已记录完整 N1 初值 |
| D1（新增） | post-norm | **0** | 原配方，保持 0.05 | **全部张量逐值复制未训练 N1** |

两组保留相同 C1 径向局部更新、learned Q/K/V/O、局部/全局调用顺序、参数量和下游
模块。概率 dropout 没有参数，关闭它不改变权重键、形状或容量。
`D1−N1` 是唯一直接因素对照；`D1−O1` 看剩余骨架差距，`D1−原 baseline` 为整体参考。
后两项仍不能单因素归因。只新增训练 D1，不重跑 N1/O1/原 baseline。

## 配方、身份和随机性

保持 `crystal_q1_v1`，M1/ZP、SumNorm/H1(eta)、E0/P0、Q1_hidden=128、128/64 bin、
六层共享 decoder、FP32。seed42、35 轮、batch32、lr5e-5、原 AdamW/调度/目标，
`balanced_oracle_mae` 选点，train/valid；不读取 test。
局部两次调用共四个内部块，全局六层，固定 `L1 L2 G1..G6`；GPU 几何缓存复用 C1。

`attention-dropout-init-v1` 复用 N1 初始化，完整 `n1_initialization` 必须与已完成
N1 run 的初始化摘要逐项相同。全部状态张量、参数数量和初始状态摘要保持相同；
不读取已训练权重作为初值。初始化保留调用方 CPU/CUDA 随机流，训练开始照原协议重设 seed。

**关闭 dropout 会改变训练期随机数消耗。**相同 seed 与初值不意味着后续其他 dropout
使用相同掩码，或训练轨迹相同。D1−N1 检验在既定训练协议下关闭该项的净效应，
不把差值解释为同一训练轨迹上的纯机制效应。

D1 使用独立 `attention-dropout-model-v1` 契约，记录实际概率为 0、参照概率为 0.05、
其余 dropout 保持。即使权重形状相同，也拒绝 N1/C1 checkpoint 和 optimizer；
D1 同组可正常恢复权重、optimizer 和 RNG。历史 N1 的预检/报告保留原指纹，
新 D1 预检绑定当前源码与所有参考产物，不改写旧证据。

## 判读

按 valid 的 2,313 个材料配对，分别报告 eDOS/phDOS × blind/oracle ×
中位 R²、中位 MAE、R²<0 失败率。材料 bootstrap 2,000 次、seed20261013，
区间条件于各组既定 MAE best checkpoint，不覆盖跨训练种子的波动。

- D1−N1 若 eDOS R² 或失败率改善且区间可分辨，同时 phDOS 没有明显退化，
  支持概率 dropout 在当前 N1 配方下贡献性能差距；不据此认定所有注意力 dropout 都有害。
- 若只有部分模式/指标改善，限定到实际分辨的指标。若区间含零，只说明本次未分辨，
  不证明没有效应或两组等价。恶化则不支持通过关闭该项修复差距。
- D1−O1 用于检查剩余差距；即使缩小，也不证明投影或初值无影响。
  保留原 MAE 选点，同时报告 R²/失败率方向，避免只按最好看的指标改选 checkpoint。

同组成不同结构谱差单列：绝对/约化组成配对、归一化有符号谱差误差、幅度、方向与 skill，
按组成组等权重采样。总体 R² 改善不自动代表结构谱差泛化改善。

## 调用链与验证

`attention_dropout_compare` → `attention_dropout_config` → `tools/train.py` →
`initialize_attention_dropout_model/attention_dropout_contract` → `LocalGlobalEncoder(D1)`。
D1 的局部消息映射为 C1，全局层保持 post-norm，仅设置六层 `attn.dropout=0`。
既有 N1 分析/队列/真实批预检增加参数以供 D1 复用，N1 配置、模型和分析默认行为保持。
真实批预检同时补上漏设的训练模式，经过与正式训练入口的核对，详见实施日志。

验证包括完整未训练状态/容量/RNG 匹配、eval 与 N1 输出一致、train 与手动关闭
N1 概率 dropout 的输出/梯度/RNG 一致、其他 dropout 保持、有限 E(3)/重排/空行、
真实 train 批梯度、恢复与错误 checkpoint 拒绝、统计配对方向、源码/参考变更拒绝、
完成组不重训和失败即停。短测仅更新随后丢弃的模型，不写正式 checkpoint，不构成精度证据。

## 执行与成本

已有 N1 实测完整训练约 88 分钟，自动分析约 1–2 分钟；D1 单组预计约 **1–1.5 小时**，
实际成本由前三轮更新 ETA，不能将短批成本直接当作整轮耗时。
固定配方，不自动改 epoch/batch/lr，不启动投影、其他归一化或跨 seed 实验。

默认只打印单条训练命令；`--run` 才训练，`--summarize` 只读状态；异常即停、不自动重试。
已完成组先核验，不追加训练；已有分析目录不覆盖，完成报告只读核对。
证据输出目录：`results/runs/attention_dropout_attribution_20261013/`，
tag prefix：`drop20261013`。执行日志使用当前日期记录，不执行 Git 提交。
