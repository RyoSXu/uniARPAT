# 全局骨架拆分第一步：仅改变归一化位置

创建日期：2026-10-13。用户确认继续拆分新全局骨架的性能差距；本小单元先补 N1 的
实现、初始化契约与短测，正式训练交由 OpenCode。
已有骨架整体对照与独立复核见[O0/O1 结果](../logs/log-2026-10-13-legacy-backbone-run.md)。

## 问题与代码事实

旧骨架在有、无 C1 局部更新时均优于新骨架，而 C1 在两种骨架内都有总体指标收益。
目前差异包含注意力投影、归一化位置、内部初值和训练机制，未定位到单个因素。

代码对照还确认：原 `TransformerEncoderLayer` 的内容注意力在 softmax 后直接聚合，
**没有注意力概率 dropout**；新 `ScalarGlobalLayer.attn` 的 `MultiheadAttention` 使用
`dropout=0.05`。两者均有残差分支 dropout 和 FFN 中间 dropout。
所以“是否有 Q/K/V/O”也不能直接代替整个注意力路径的比较。

本轮只改变六层全局更新中两次 LayerNorm 的位置：

| 组 | 局部更新 | 全局注意力 | 全局归一化 | 未训练初值 |
|---|---|---|---|---|
| C1（已有） | C1，前段两次调用 | 原新骨架 learned Q/K/V/O，概率 dropout 0.05 | pre-norm | 已记录完整 C1 初值 |
| N1（新增） | 同 C1 | 同 C1 | post-norm | **全部张量逐值复制未训练 C1** |

pre-norm：`h1 = h + Drop(A(LN1(h)))`，`h2 = h1 + Drop(F(LN2(h1)))`。
post-norm：`h1 = LN1(h + Drop(A(h)))`，`h2 = LN2(h1 + Drop(F(h1)))`。
`A`、`F`、dropout 概率、mask 规则、参数数量/键/形状均保持；只改变 LN 的位置。
不增加 Encoder 末端 LN；局部外层 LN、decoder 的归一化与全部下游模块保持。

## 冻结配方与初始化

修复版 `crystal_q1_v1`，M1/ZP、SumNorm/H1(eta)、E0/P0、Q1_hidden=128、128/64 bin、
六层共享 decoder、FP32。seed42、35 轮、batch32、lr5e-5、原 AdamW/调度/dropout/目标，
`balanced_oracle_mae` 选点，train/valid；不读取 test。
局部两次调用共四个内部块，全局六层，固定 `L1 L2 G1..G6`。

`global-norm-init-v1` 复用现有 `initialize_structure_control`，将 N1 的全部状态设置为
**未训练 C1** 的原值。完整 `c1_initialization` 必须复现已完成 C1 run 的摘要，
N1 整模型状态摘要必须等于 C1；包括所有局部/全局/下游权重、bias 与 LayerNorm。
不从已训练 C1 checkpoint 起训，初始化保持调用方 CPU/CUDA RNG；正式循环沿用 seed 重设。

N1 使用独立 `global-norm-model-v1` 契约。同形状的 C1 checkpoint 也必须被拒绝，
不同归一化方案不混用权重或 optimizer 状态；N1 同组正常恢复保留原轨迹。
原 C1/A/B+/G0/O0/O1 默认行为与历史初始化协议保持；历史源码指纹记录保留，
当前源码新增 N1 后由新预检绑定，不能改写旧预检以掩盖变化。

## 分析与判断

只新增训练 N1，复用 C1/O1/同数据 35 轮原 baseline。
全部 valid 2,313 材料按 ID 配对，eDOS/phDOS × blind/oracle ×
中位 R²/中位 MAE/R²<0 失败率；材料配对 bootstrap 2,000 次、seed20261013。
原选点保持，同时报告 MAE 与 R²/失败率方向是否一致。

- **N1−C1 为唯一直接因素对照**：仅改变全局归一化位置，初始参数与容量完全匹配。
  若 R²/失败率改善且区间可分辨，支持该位置在当前注意力/配方下贡献性能差距；
  不据此认定 pre-norm 普遍较差，或排除与注意力投影/概率 dropout 的交互。
- **N1−O1**：看改归一化后仍剩多少骨架差距，仍包含注意力投影、概率 dropout、
  内部初值；不能将残余差距自动称为投影的影响。
- **N1−原 baseline**：同数据同预算参考表现，包含旧几何路径等整包差异。

区间含零表示该协议未分辨差异，不证明等价；仅 MAE 改善不能宣称修复 R²/失败率问题。
固定 seed 的差异包含因模型行为改变造成的后续训练轨迹变化，材料区间不代表跨 seed 稳定。
每组沿用各自 MAE best，结果条件于已选 checkpoint；其他编排/位置未被检验。

同组成不同结构谱差另报：绝对/约化组成配对、归一化有符号谱差误差、幅度与方向，
按组成组重采样；不将总体收益直接当作结构谱差泛化改善。

## 实施、验证与成本

调用链：`global_norm_compare` → `global_norm_config` → `tools/train.py` →
`initialize_global_norm_model/global_norm_contract` → `LocalGlobalEncoder(N1)` →
`ScalarGlobalLayer(norm_first=False)`；结构入口与缓存均复用 C1。

新增 `utils/global_norm_experiment.py`、`tools/eval/global_norm_compare.py`、
`tools/checks/check_global_norm.py`、`tests/pipeline/test_global_norm.py`。
原子/几何/局部消息、decoder/head 不增加新实现；不改数据、损失或历史 checkpoint。

验证包括：与标准 post-norm 层的输出/梯度核对、默认 pre-norm 公式保持、所有初始张量
与 C1 逐值相同、有限 E(3)/重排/空行、真实 train 批梯度、恢复隔离、统计符号及异常停止。
短测只更新随后丢弃的模型，不写正式 checkpoint，也不构成准确率证据。

已有 C1 单轮训练段中位 131.8 秒；单组 35 轮预计约 **1–1.5 小时**，
由实际前三轮更新 ETA，固定配方。GPU 预检成本与验证记录见
[N1 实施日志](../logs/log-2026-10-13-global-norm-preflight.md)。

默认只打印单条命令，`--run` 才训练；完成组先复核，不追加训练，异常即停不自动重试。
完整验证分析输出与参考 run 指纹同目录归档；不自动启动投影/dropout 下一轮或提交 Git。
