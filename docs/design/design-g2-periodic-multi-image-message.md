# 设计：G2a 周期多镜像边条件消息

> 关联：[状态与计划](../status.md)、[结构表征方向](design-structural-refinement.md)和[数据约定](../data.md)。
> 本文是 G2 的唯一实施设计；实施、预检或训练前不得以 G1 的最短镜像矩阵替代本文的多重边集合。

## 目标与成功判据

**假设：**当前 B7/G1 把每个原子对压缩为一条最短镜像边，并且只让几何改变 attention score。若将每个
截断半径内的真实周期邻居 `(i, j, T)` 独立保留，并用边距离调制被聚合的 Value（注意力中传递的内容），
则可补充 eDOS/phDOS 所需的局域配位信息。

本轮固定为 **G2a**，只检验“多镜像、径向边条件局域消息残差”这个模块：

- **对照：**新跑的 Q1、M1×10、seed 42、E0/P0、SumNorm KL/W1/Huber、H1 eta/gamma、dropout
  0.05 的 `_g2ctl`。
- **实验臂：**完全同配方的 `_g2edge`，仅开启 `--use_g2`。
- **主指标：**oracle 测试 eDOS/phDOS 的中位 R²和失败率；同时保存 blind 测试、oracle–blind gap
  的 p50/p90/p99、Cv MAE、单轮耗时、峰值显存和边数统计。
- **pilot 判决：**至少一个任务的中位 R²相对 `_g2ctl` 提升 `>= 0.02`；任一任务失败率不得恶化
  `>= 1` 个百分点，另一个任务的中位 R²不得低于对照 `0.02` 以上。满足时才运行 Q1 M1×35 等算力
  确认；否则 G2a park，默认关闭，不扫描 cutoff、层数、归一化、RBF 宽度或融合位置。
- **35 epoch 确认：**除同一 oracle 判据外，必须报告 blind 指标与 gap 分位数；blind 出现有害退化时
  （中位 R²低于对照 0.02 以上或失败率恶化 `>= 1` 个百分点）不合入默认方案。

本轮不比较 G2a 与已 park 的 G1，也不以 B7 的 35 epoch 结果取代新跑的 10 epoch 对照。

## 范围与非目标

G2a **保留** B7 的六层 dense attention、既有 10 Å 相对位置打分、decoder、H1、数据、标签、损失、
网格和 CIF 输入约定。它不增加 hub token，不改变 attention 可见性，也不复用 G1 的 `max_neighbors` 或
`t_range` 截断。

以下内容明确不属于 G2a：

- 用稀疏局域图替换 B7 attention；
- 方向/键角条件的 Value、等变向量消息或球谐新主干；
- cutoff 扫描、top-k 裁边、按晶胞类型分支或改变 batch size；
- primitive/conventional/supercell 数据增强，或任何 Q1 缓存、划分、标签改动。

方向信息已在 B7 的 attention score 路径中存在；G2a 的 Value 只使用径向距离，以避免把“周期多镜像”与
新的旋转协变假设混成两个实验因素。方向条件若有必要，必须在 G2a 结论后另写设计。

## 周期边的定义

设有效原子为 `i,j`，分数坐标为 `f_i,f_j`，实空间晶胞矩阵为 `A`，固定截断半径
`R = 5.5 Å`。一条有向边的接收端是 `i`，发送端是 `j`：

\[
e=(i\leftarrow j,T),\qquad
r_{ijT}=(f_j-f_i+T)A,\qquad T\in\mathbb Z^3.
\]

当 `||r_ijT|| < R` 时保留该边。`i=j,T=(0,0,0)` 排除：B7 原有残差和 attention 自身已提供恒等路径；
`i=j,T≠0` 必须保留，它们是跨胞自镜像而不是普通自环。边数组同时保存 `batch, dst=i, src=j, T, distance`
供测试与审计；G2a 消息只读取 `dst, src, distance`。

为使有限枚举完整且对等价周期坐标稳定，先将
`delta = f_j - f_i - round(f_j - f_i)` 规范到最邻近的中心胞表示。令 `s_min(A)` 为晶胞矩阵最小奇异值，
枚举

\[
K=\left\lceil R/s_{min}(A)+0.5\right\rceil,
\qquad T_k\in[-K,K]\cap\mathbb Z .
\]

这是保守上界：任意满足 cutoff 的边均有 `|T_k| < R/s_min(A)+0.5`。因此不得用 G1 的固定
`[-2,2]^3` 替代，也不得静默裁边。非有限或 `s_min(A)<=1e-6 Å` 的晶胞必须显式报错；Q1 审计若出现
此类记录，停止实施并报告，不修改数据。

同一物理边的相反方向必须同时存在：`(i←j,T)` 对应 `(j←i,-T)`，距离相等。边只以严格 `< R` 判定；
平滑 cutoff 在边内取值，在边界处为零。

## 模块与融合路径

每个 encoder layer 在原 B7 attention、残差、LayerNorm、FFN 和 LayerNorm 完成后，增加一个独立的
`PeriodicEdgeMessage` 残差。令该层原输出为 `h`，距离 RBF 为 `phi_RBF(d)`（64 个中心均匀覆盖
`[0.01R,0.99R]`），`c(d)` 为 G1 已验证的 quintic smooth cutoff，则

\[
v_j=W_vh_j,
\]
\[
m_{i\leftarrow j,T}=c(d_{ijT})\,[v_j\odot\sigma(W_g\phi_{RBF}(d_{ijT}))],
\]
\[
h_i\leftarrow h_i+
\alpha\,W_o\!\left(\frac{1}{\sqrt{\max(1,d_i)}}
\sum_{(j,T)\in\mathcal E(i)}m_{i\leftarrow j,T}\right),
\]

其中 `d_i` 是接收端的真实入边数，`W_v`、`W_g`、`W_o` 分别为 `512→512`、`64→512`、`512→512`
线性层，`alpha` 是每层一个标量并初始化为 0。`1/sqrt(d_i)` 保持多重边的配位信号，同时限制高度数晶胞的
数值方差；不得改为普通平均，因为那会把重复周期邻居重新抹平。

所有边采用扁平稀疏数组并用 `index_add` 聚合，不构造 `[B,L,L,S]` 的稠密镜像张量。padding 原子既不作
发送端也不作接收端。开启 G2 时新增约 3.36M 参数（六层各约 0.559M，低于 B7 约 71.1M 的 5%）；关闭时
模块不实例化，既有状态字典、参数量和输出路径保持不变。开启后从 B7 加载共享权重时仅允许缺少 G2 新键；
在 `alpha=0` 的 eval 前向中 eDOS/phDOS 必须逐元素 bitwise 等于 B7。

## 实施边界

预计改动限定为：

- `utils/g2_periodic_edges.py`：精确周期多重边构建、RBF/cutoff 辅助函数和纯 CPU 可调用接口；
- `model/transformer.py`：可选 G2 模块、六层消息残差及关闭路径；
- `run_ablation_experiments.py`：仅增加默认关闭的 `--use_g2`，将固定 `g2_r_cut=5.5` 写入
  `config_used.yaml`；不暴露用于扫描的邻居数或平移范围参数；
- `tests/test_g2_periodic_edges.py`：下述确定性合同；
- `tools/eval/g2_edge_audit.py`：不读取标签、遍历既有 Q1 三个划分并输出边数审计 CSV。

不得修改默认 YAML、Q1 缓存、G1 文件、训练损失或既有结果文件。G2 审计输出命名为
`results/g2_edge_audit_q1.csv`；每次训练继续使用唯一 tag 的标准 `history/test/samples` 文件。

## 测试与预检关卡

以下关卡全部通过才允许启动 `_g2ctl` / `_g2edge`。任一失败均停止在实现阶段，不启动训练。

1. **枚举完整性。**在随机斜晶胞、小晶胞及高长宽比晶胞上，将 G2 边集与 `K+3` 的暴力枚举逐条比较；
   `dst/src/T/distance` 完整一致，且没有 `i=j,T=0`。
2. **Si 两原子小胞计数。**使用 diamond Si primitive fixture：晶格长度
   `5.431/sqrt(2) Å`、三角均为 `60°`、分数坐标 `(0,0,0)` 与 `(0.25,0.25,0.25)`。在 `R=5.5 Å` 下，
   每个接收原子恰有 34 条入边，其中 18 条为 `i=j,T≠0` 自镜像、4 条为 2.351692 Å 的第一壳层
   异原子边；全图合计 68 条有向边、36 条自镜像和 8 条第一壳层异原子边。完整距离多重集由 fixture
   固定。
3. **物理合同。**检查反向边配对、严格 cutoff、quintic 值域和端点、padding 排除、所有输出有限；
   任意 `frac + integer` 表示同一边距离/消息集合。
4. **等价表示合同。**整体刚体平移、联合原子置换、交换晶格基矢及一般整数幺模基变换后，边的物理距离
   多重集和按置换对应的模型输出保持一致（浮点比较容差 `1e-5`）。
5. **模型合同。**`--use_g2` 默认关闭且不新增参数；B7 权重迁移到 G2 后 `alpha=0` 的 eval 输出
   bitwise 等价；`alpha` 首步取得有限非零梯度，更新一次后 `W_v/W_g/W_o` 取得有限非零梯度；消息对
   原子置换等变。
6. **真实数据冒烟。**Q1 CPU 单步和 V100 batch 32 单步均产生有限的所有损失键（含 H1），无 NaN/Inf；
   CPU 只验证数值合同，不用作成本结论。
7. **边数与资源审计。**先运行 `g2_edge_audit.py`，记录 train/valid/test 的每结构边数、每原子入度、
   `K` 的 p50/p95/p99/max 和最大样本标识。随后在同一 V100、batch 32、同一输入批次下测量 B7 与 G2
   的完整训练单步峰值显存和耗时。固定 batch 32 下发生 OOM 才阻断训练；无 OOM 时，成本增量作为正式
   报告项而非准确率候选的自动淘汰线。任何新的成本硬阈值都必须经用户明确批准后，才可用于停损。

## 训练、记录与停损

预检通过后，使用下面的固定预算：

```bash
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _g2ctl
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _g2edge --use_g2
```

两臂均保持 Q1、batch 32、seed 42 和默认生产配方。训练结束后，在同一最佳 epoch 分别评估 oracle 与
blind；正式日志必须写明完整命令、配置、参数量、边审计摘要、资源实测和指标格式
`e med/fail + p med/fail + (test, epN, oracle|blind, Q1)`。

- **win：**pilot 满足本文主判据，才运行 `_g2longctl` / `_g2longedge` 的 M1×35 等算力确认；确认胜出才
  更新默认方案与 `decisions.md`。
- **park：**pilot 平局、任何任务有害退化、固定 batch 32 OOM 或数值不稳定。默认关闭；不进行参数/半径/层数扫描。
- **不会由本轮推出的结论：**G2a 的结果不能单独归因于“多重边”或“Value 调制”其中之一；它只检验这个
  预注册模块。若 G2a win，机制拆分是后续独立设计；若 G2a park，不否定未来的等变局域主干。

## 风险与后续边界

- 小晶胞可能产生大量合法周期像；这是真实物理邻居而非数据异常。若固定 batch 32 发生 OOM，应重新讨论
  研究问题，不得静默删边。耗时增量本身须报告，但不构成未经用户批准的自动停损线。
- 多重边可能对谱任务没有额外可用信息，或与 B7 dense attention 高度冗余；平局是可接受的科学结论。
- 该模块只使用 CIF 结构，未触及 Q1 的 coverage mask、Z0 `N_valence`、标签或 blind 尺度定义。
- G2a 结束后才回到待办顺序中的 decoder 缩减和 fixed-grid encoder-only atomic PDOS；它们不与 G2a
  并行合并。
