# 设计：R2b 固定网格的原子加性 phDOS 读出

> 关联：[状态与计划](../status.md)、[R2a decoder 深度缩减](design-r2a-decoder-depth.md) 和
> [结构表征与读出重构方向](design-structural-refinement.md)。本设计只定义 R2b，不重新开启 G2、
> 连续 query、网格或预训练分支。

## 目标与成功判据

检验固定 P0 网格的总 phDOS 是否适合由 encoder 的原子 token 直接、可加地生成，而非经过频率
query、共享 decoder 和卷积头生成。Q1 没有逐原子 PDOS 真值，因此这是**总谱弱监督下的原子加性
读出**，不是经标签验证的 atomic PDOS 回归。

- **对照：**已完成的 R2a 3 层臂 `_r2a3`：Q1、M1、6 层 encoder、3 层共享 decoder、E0/P0、
  SumNorm KL/W1/Huber、H1 eta/gamma、dropout 0.05、batch 32、seed 42、10 epoch。R2b 关闭时
  必须与它的模型路径逐元素一致，因此该受版本控制的成对结果可复用为控制证据，无须重跑相同臂。
- **实验：**相同配方、相同 seed、唯一架构变量 `--use_atom_additive_phdos`；tag 为 `_apdossum`。
- **win：**phDOS Oracle 测试中位 R² 相对 `_r2a3` 提升至少 `+0.02`，phDOS 失败率不增加
  `1.0` 个百分点；同时 eDOS 中位 R²不低于 `−0.02`、eDOS 失败率不增加 `1.0` 个百分点。满足时
  才讨论等算力确认。
- **park：**无 phDOS 正向跨线、处于平局线，或任一保护条件失败。不得扫描原子头宽度、层数、
  正则化或聚合方式。

## 改动

新增默认关闭的 `use_atom_additive_phdos=False`，仅替换 phDOS 读出：

\[
u_{i,b}=\operatorname{MLP}(h_i)_b,\qquad
a_{i,b}=\operatorname{softplus}(u_{i,b}),\qquad
A_b=\sum_{i\in\mathrm{valid\ atoms}}a_{i,b},\qquad
z_b=\log(A_b+10^{-12}).
\]

- `h_i` 是 6 层 encoder 输出的第 `i` 个原子 token；MLP 是所有原子共享的
  `Linear(512,512) → GELU → Linear(512,64)`，共 295,488 个参数。
- `a` 的形状为 `[B,L,64]`，padding 原子的贡献严格为零；`A` 的形状为 `[B,64]`。
  输出 `z` 仍是现有 SumNorm KL/W1/Huber 所需的 raw logits，故
  `softmax(z)=A/\sum_b A_b`，不改损失或目标尺度。
- R2b 启用时只跳过 phDOS 的 64 个 query、共享 decoder 调用和原 phDOS CNN；eDOS 仍经 R2a 的
  3 层 decoder，H1 eta/gamma 仍由同一 encoder memory 的池化结果产生。
- 关闭时不得创建新的模块或属性，且模型构造、参数名/参数量与 R2a 3 层严格一致。默认 YAML、
  B7 路径、eDOS 路径、数据和损失均不改。

原 phDOS CNN 约 30.68M 参数，连同 phDOS query/tgt 约 0.066M；R2b 预计相对 R2a 净少约
30.45M 参数。这是“原子加性、encoder-only phDOS 读出”这一整体架构因素固有的容量变化，不是
参数匹配的因果分解；若 R2b 无收益，只否定该整体读出方案，不能将原因归咎于任一单独细节。

## 测试关卡

实施后、训练前必须通过：

1. R2b 关闭与显式关闭的 3 层模型参数名、参数量、固定输入的 eDOS/phDOS/eta 都逐元素相等。
2. 启用后 `atom_phdos_contrib` 非负，padding 项严格为零，按原子求和与 `exp(phdos)` 在归一化意义下
   一致；输出为 `[B,64]` 且有限。
3. 联合置换原子元素、坐标、mask 和 encoder memory 时，原子贡献相应置换、总 phDOS 不变；改变
   padding 元素或坐标不影响输出。
4. 真实 Q1 batch 的生产维度 `train_one_step` 在 CPU 上所有损失有限，H1 `loss_eta` 激活，
   原子头与 encoder 获得有限梯度。
5. 在同一 V100、batch 32、真实 Q1 batch 上实测单步显存和耗时；无 OOM 是唯一资源阻断条件，
   成本比只作记录，不设自动阈值。

## Pilot 与记录

完成测试和资源记录后，从零开始运行实验臂：

```bash
setsid nohup python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 \
  --tag _apdossum --decoder_layers 3 --use_atom_additive_phdos \
  > output/apdossum.log 2>&1 &
```

使用 `results/*r2a3*` 为逐元素关闭路径已经证明的控制；新增 R2b 专用 blind 判定脚本，比较
Oracle/Blind 中位 R²、失败率、gap、eta/gamma、Cv、参数量与实测资源。结果、结论和下一关卡分别
写入 `results/`、日期日志和 `status.md`；未经 win 不启动 35 epoch。

## 风险与解释边界

- `a_{i,b}` 在总谱弱监督下不是唯一可辨识的局域物理 PDOS：同一总和可以由多种原子分配构成。
  因此只报告它满足的非负、加性和置换合同，不报告其为物理真值。
- 新读出会显著降低 phDOS 容量；正向结果支持“更强结构归纳偏置优于原读出”的组合，负向结果不等于
  结构信息无用，也不能区分容量、频率耦合和聚合约束的单独作用。
- 该设计不修改盲推理尺度、总量约定、coverage mask 或 CIF 输入边界；不引入标签派生特征或外部数据。
