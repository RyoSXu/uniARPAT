# 增强 B 的高阶角度统计：A / B / B+ CPU 对照

日期：2026-10-03。用户批准增强 B，并要求以直观量化结果与 A 对照。
**本单元完成：B+ 能区分既有的受控高阶角度反例，保留已检查的局部对称性，增加较小的 CPU 成本。
正常截断附近的弱响应仍存在；未选择生产 Encoder，未训练或验收完整谱输出。**

## 范围、依据与实现

依据是[上一轮预检](log-2026-10-03-periodic-message-precheck.md)：原版 B 的一、二阶方向状态
在特定纯星形环境中不能恢复高阶角度差别，A 的显式角度消息能区分该受控反例。
本轮保留 A 和原版 B，新增独立 B+；原型、几何、生产模型和上一轮结果文件均未覆盖。
输入为 Q1 train/valid 结构，共同未训练 ZP 和局部投影；不加载谱标签、test、checkpoint，
不创建 optimizer、不更新参数、不接入生产模型。blind/oracle 谱模式不适用。
半径保持严格 `d<6.0 Å`，保留全部周期镜像、自身跨胞记录、计数和原有平滑截断。

调用链：`create_models → 原版 B 权重复制 → HighOrderMessageProbe → 两个 HighOrderEquivariantBlock`
在原来 `64×0e+8×1o+4×2e` 状态旁补充高阶角度不变量，最终仍返回 `[B,L,512]` 标量。
三、四阶矩仅用于生成不变量，不作为新的持久方向通道；原有低阶方向状态继续跨层保留。

每层复用 B 径向 MLP 的八个向量通道系数 `g_ea`，设 `w_ea=c_e g_ea`，按中心计算：

\[
M_{i\ell a}=\sum_{e\in N_i} w_{ea}Y_\ell(u_e),\qquad
D_{ia}=\sum_{e\in N_i}w_{ea}^{2},\qquad \ell\in\{3,4\},
\]

\[
T_{i\ell a}=\frac{\|M_{i\ell a}\|^2/(2\ell+1)-D_{ia}}
 {2\sqrt{\max(1,p_i)}}
=\frac{\sum_{e<f}w_{ea}w_{fa}P_\ell(u_e\cdot u_f)}{\sqrt{\max(1,p_i)}}.
\]

`Y` 使用 e3nn 的 component 归一化，`P₃(q)=(5q³−3q)/2`、`P₄(q)=(35q⁴−30q²+3)/8`。
减去对角项排除同一条记录与自身配对；不同周期镜像仍是不同记录。
少于两条记录的中心显式返回零。八个权重、两个阶数共 **16 个新增标量**，追加到原版
标量更新的 78 个输入后：`U_B:94→128→64`。每层只多 `16×128=2,048` 个参数，
两层增加 **4,096**；其余权重及偏置与原版 B 逐值相同。

该式用逐边求和得到加权 Legendre 角度对统计，固定阶数/通道下成本随记录数增长，
不枚举全部邻边对。范数及减去的对角项都是反射偶不变量；标量权重、共享求和及原有
O(3) 更新保留局部对称性。单元检查已对 26 个完整周期镜像和非均匀有符号权重核对上述
等式与显式枚举的一致性，并检查旋转/反射、填充、零邻边对和新增权重/坐标梯度。

这没有使 B+ 与 A 完全等价：B+ 汇总有限阶、八通道的可分离角度统计；A 的角度 MLP
直接联合处理两个邻居特征、两条距离与夹角。两者都不保证识别所有环境。

| 文件 | 新增符号与职责 |
|---|---|
| [增强原型](../../tools/eval/periodic_message_high_order.py) | `high_order_pair_features` 计算三/四阶配对统计；`HighOrderEquivariantBlock` 追加标量输入；`HighOrderMessageProbe` 复制原版权重并提供原接口及关掉新增输入的消融 |
| [三路对照](../../tools/eval/periodic_message_high_order_compare.py) | `create_models` 对齐权重；`fixed_radial_pair` 固定径向字段；`low_moment_pair` 重建旧反例；`response_metrics` / `angle_controls` 测特征响应；`load_sample` / `sheared_geometry` 读取与物理形变；`main_comparison` 调用既有几何/对称性/资源检查；`shape_summary` 汇总；`main` 提供主运行和独立 RSS 模式 |
| [独立复核](../../tools/eval/verify_periodic_message_high_order.py) | `verify` 核对哈希、逐项/CSV、覆盖、摘要和独立 RSS；`main` 导出复核结果，不重跑网络 |
| [对比图](../../tools/eval/plot_periodic_message_high_order.py) | `main` 从复核后的产物导出 PNG/SVG 和图表数据/哈希 |
| [回归](../../tests/test_periodic_message_high_order.py) | `HighOrderMessageContracts` 七项科学契约；包含实际晶胞输入契约下的形变重建回归 |

原有 `count_features`、球谐/张量积、`prepare_geometry`、`inspect_case`、`compare_features`、
`measure_resources`、`BudgetMonitor` 等均复用。原版 A/B 的 Python 文件和参数量不变。

## 固定协议

- PyTorch 2.2.1、e3nn 0.5.6；CPU 的 PyTorch 计算线程为 1，另有 20 ms RSS 监控。
  float64 主权重转换到 float32/float64；eval，无 dropout。
- 主种子为 20261003，A/B/ZP 的哈希与上一轮逐值匹配。B+ 复制全部旧权重，仅追加列新随机初始化。
  角度对照另外固定 42、7、1234、314159，共 **5 个初始化种子**；这是无训练检查，不是多 seed 训练。
- 真实结构沿用上轮按几何极端值及邻边对分位选出的 **16 个结构（train/valid 各 8）**。
  它不是总体无偏抽样，时间中位数不能外推为整个数据集的成本。
- B+ 在全部 16 个真实结构、两 dtype 上执行原有完整局部协议：五种直接正交变换，
  物理平移重建，原子重排、整数镜像、换基；两个最大邻边对结构另做两 dtype 的 2×1×1 超胞。
  A/B 本轮在相同 16 结构上另做 float32 旋转/反射，并对合成例执行两 dtype 检查；
  上轮相同权重的完整有限检查仍有效，源码及旧产物哈希已复核。
- 直接特征 `(atol,rtol)` 为 float32 `(2e-5,2e-5)`、float64 `(2e-10,2e-10)`；
  物理/晶胞表达重建为 `(1e-4,1e-4)`、`(2e-6,2e-6)`，沿用旧协议，没有根据结果放宽。
- 两类纯角度对照严格共享记录 ID、距离、RBF、截断权重及计数。12 邻居反例同时报告
  **正常截断**和 **`c=1` 受控消融**，后者只隔离表达能力，不是采用的权重公式。
  响应门槛预设为 float32 `1e-8`、float64 `1e-10`；主要表达判读用 float64，
  不把 float32 舍入差或任意未训练幅度排名当作预测收益。
- 计时前已有基线、对称性及剪切的未计时前向。随后每个结构做三次前向与无更新反向，
  三路按轮次轮换顺序；先取每结构三次中位数，再取 16 结构中位数。
  几何枚举、ZP、全局层、decoder、optimizer 不计入局部块时间。
- 内存用三个全新进程测同一 train 最大邻边对结构：64 个有效原子、7,704 条记录、460,180 对。
  三路原型均构造，只有指定一路做前向/反向；RSS 包含库与进程开销，不是每个模块的纯分配量。
- 真结构响应另做固定分数坐标下 `xy=0.02` 的保体积剪切，重建全部邻居。
  距离、夹角及部分记录数可同时改变，不能称为纯角度对照。

## 直观量化结果

| 指标 | A | 原版 B | 增强 B+ |
|---|---:|---:|---:|
| 两层局部参数，含共同 512↔64 投影 | 261,568 | 180,676 | **184,772** |
| 局部前向中位数，ms | 121.12 | 9.80 | **11.04** |
| 局部前向+反向中位数，ms | 827.79 | 27.54 | **30.67** |
| 各结构中位前向+反向的最大值，s | 21.584 | 0.233 | **0.248** |
| 独立进程 RSS 峰值，MiB | 974.19 | 576.28 | **582.74** |
| 高阶反例：可分辨种子/5，float64、`c=1` 消融 | 5/5 | 0/5 | **5/5** |
| 同一反例：标量最大变化的种子中位数 | 2.82e-5 | 5.55e-16 | **1.12e-4** |
| 5.9 Å 正常截断角度对照，float32，五种子 | 全为 0 | 全为 0 | **全为 0** |
| 16 结构共同旋转/反射 float32 特征最大误差 | 9.54e-7 | 7.15e-7 | **7.15e-7** |

同一受控反例的新增输入关闭后，B+ 恢复原版 B 的输出/方向状态与盲点，消融结果在容差内一致。
因此这个改善来自新增高阶入口，并非换掉原有权重或径向差异。B+ 的原生 5.85 Å 高阶反例
在 float32 下仍全为零；float64 五种子中位变化为 **2.64e-12**，也低于门槛。
不能把消融下的 5/5 写成正常截断下已解决所有高阶结构分辨问题。

相对 B，B+ 参数增加 **2.27%**、中位前向+反向增加 **11.36%**、RSS 增加 **1.12%**。
相对 A，B+ 参数少 **29.36%**，本次中位前向约快 **10.98 倍**、前向+反向约快 **26.99 倍**，
独立 RSS 少 **40.18%**。这些都是此 CPU 单样本协议的实测值，不是 GPU、batch 或训练速度承诺。

保体积剪切后的原子输出相对 L2 变化中位数：A **0.03815%**，B **0.07571%**，B+ **0.07601%**。
B+ 与 B 的这项真实结构响应基本接近；不能声称本轮已经证明真实材料上大幅改善。
输出幅度由未训练参数和归一化共同决定，响应更大不等于谱预测更准确。

对比图：[PNG](../../results/periodic_message_high_order_q1_r6_20261003_v2/comparison.png)、
[SVG](../../results/periodic_message_high_order_q1_r6_20261003_v2/comparison.svg)。
图中的反例条目明确为 `c=1` 消融，近截断弱响应同时保留。

## 验证、产物与预算

正式 **994 项全部通过**，没有截止键集差异。B+ 直接框架下全部正交变换的最大误差为
float32 **7.15e-7**、float64 **1.55e-15**；新高阶不变量在真实结构共同旋转/反射中的
float32 最大误差为 **2.79e-8**。物理平移/晶胞表达重建的 float64 最大误差为
**3.22e-8**，在既定容差内；该项包含原晶胞还原/序列化的有限精度影响。
所有真实反向检查都有有限梯度、没有完全未使用的参数张量，参数/缓冲区起止哈希一致。
这不代表每个参数元素都有梯度，也不证明可以训练到目标准确率。

| 类别 | 项数 |
|---|---:|
| 固定径向角度响应 / 高阶反例 / 关闭高阶消融 | 60 / 60 / 20 |
| 填充、空邻域等边界 | 36 |
| 直接 E(3) / 物理重建 E(3) | 280 / 62 |
| 重排、镜像与换基 / 超胞 | 152 / 4 |
| 真实结构三路共同旋转/反射 / 新高阶不变量 | 96 / 32 |
| 真结构剪切响应 / 三轮资源测量 | 48 / 144 |

混合填充合成例的 batch 为 2，复用的物理重建 helper 仅处理 batch=1，因此不把其六项
假想物理重建计入；该例的直接 E(3)、填充与有限性检查实际执行。

新增七项回归，加既有 35 项，共 **42 项通过**；测试体 4.52 s，含导入进程耗时 7.83 s。
独立复核核对 **56,658 个 CSV 单元格**、JSON/JSONL、摘要/覆盖、输入/源码/旧产物哈希、
原权重重现、参数起止哈希与三份独立 RSS；通过。复核不重跑网络。

正式主对照 **418.77 s**、RSS **1,066.70 MiB**；主对照加三个独立 RSS 脚本计时
**453.83 s**。首轮脚本在形变重建处因晶胞函数调用签名错误中断，已修正并补回归；
其 **32.63 s、369 项无失败但未完成**的产物与中断时源码快照
[保留](../../results/periodic_message_high_order_q1_r6_20261003/interrupted_source/snapshot.json)，不纳入通过结论。
含这次中断的主/RSS 脚本计时为 **486.46 s**，相关回归/导入/复核/画图另计，仍在 15 分钟
CPU 计算与 2 GiB RSS 的本轮边界内；阅读、实现及文档时间不计作数值计算预算。
预算是监控线程采样并在 Python 边界检查，不是内核级硬中断。

正式产物：[manifest](../../results/periodic_message_high_order_q1_r6_20261003_v2/manifest.json)、
[summary](../../results/periodic_message_high_order_q1_r6_20261003_v2/summary.json)、
[逐项结果](../../results/periodic_message_high_order_q1_r6_20261003_v2/cases.json)、
[复核](../../results/periodic_message_high_order_q1_r6_20261003_v2/verification.json)、
[图表数据](../../results/periodic_message_high_order_q1_r6_20261003_v2/figure.json)。
独立 RSS：[A](../../results/periodic_message_high_order_rss_q1_r6_20261003/A/summary.json)、
[B](../../results/periodic_message_high_order_rss_q1_r6_20261003/B/summary.json)、
[B+](../../results/periodic_message_high_order_rss_q1_r6_20261003/Bplus/summary.json)。

实际命令如下，产物已存在，重跑必须指定新目录：

```bash
python -W ignore::DeprecationWarning tools/eval/periodic_message_high_order_compare.py --out-dir results/periodic_message_high_order_q1_r6_20261003_v2 --budget-seconds 750
python -W ignore::DeprecationWarning tools/eval/periodic_message_high_order_compare.py --mode rss-A --budget-seconds 45 --out-dir results/periodic_message_high_order_rss_q1_r6_20261003/A
python -W ignore::DeprecationWarning tools/eval/periodic_message_high_order_compare.py --mode rss-B --budget-seconds 45 --out-dir results/periodic_message_high_order_rss_q1_r6_20261003/B
python -W ignore::DeprecationWarning tools/eval/periodic_message_high_order_compare.py --mode rss-Bplus --budget-seconds 45 --out-dir results/periodic_message_high_order_rss_q1_r6_20261003/Bplus
python tools/eval/verify_periodic_message_high_order.py --out-dir results/periodic_message_high_order_q1_r6_20261003_v2 --rss-dir results/periodic_message_high_order_rss_q1_r6_20261003
python tools/eval/plot_periodic_message_high_order.py --out-dir results/periodic_message_high_order_q1_r6_20261003_v2
```

## 判断与剩余范围

B+ 是值得继续评估的轻量接入候选：在此受控高阶反例中补上 B 的盲点，成本仍明显低于 A。
该判断不证明 B+ 与 A 表达能力等价，也不证明 DOS 收益；完整模型和 GPU 仍未执行。
下一单元需要共同决定近截止角度权重是否另做对照、选定方案如何接入全局混合与 decoder/head，
处理旧 `rp_proj` 几何旁路，并单独验收两类完整谱输出的 E(3)。新训练的初始化、预算与收益判据
尚未确定；本轮未训练、未提交。
