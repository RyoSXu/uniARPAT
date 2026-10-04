# 6.0 Å 两套周期消息原型：CPU 预检

日期：2026-10-03。用户批准两套独立原型、CPU 累计 15 分钟及 RSS 2 GiB 上限。
**本单元完成：局部对称性、消息入口和既定一致性检查通过；同时确认表达与数值幅度限制。
尚未选择生产 Encoder，也未验收完整 eDOS/phDOS 模型。**

## 范围与调用链

固定 [候选公式](../design/design-periodic-message-candidates.md)，使用 Q1 train/valid 的结构，
不读取谱标签、test 或 checkpoint，不创建 optimizer，不更新参数，不调用生产 DOS 模型。
blind/oracle 谱推理在本诊断中不适用。原有模型半径和训练配置未改变。

`load_structures → geometry_from_pos → build_g2_edges + periodic_edge_vectors → prepare_geometry`
准备严格 `d<6.0 Å` 的全部镜像、16 RBF、平滑权重及计数。
共同的未训练 `ProbeZP`（Embedding → LayerNorm → Linear）生成 512 个标量通道；
`PeriodicMessageProbe` 投影到 64 通道、更新两次、再返回 512 个标量。
原型 ZP 使用新随机权重并固定数值，不是 ZP100 checkpoint，也不改变生产 ZP 的训练约定。

| 文件 | 主要符号及职责 |
|---|---|
| [原型模块](../../tools/eval/periodic_message_prototypes.py) | `ProbeGeometry`、`prepare_geometry`、`geometry_from_pos` 准备与检查几何；`InvariantMessageBlock` 实现 A；`EquivariantMessageBlock` 实现 B；`PeriodicMessageProbe` 提供共同标量接口；`ProbeZP` 提供共同元素入口 |
| [正式预检](../../tools/eval/periodic_message_precheck.py) | `selection` 只按结构选样本；`changed_geometry` / `supercell_geometry` 重建物理框架；`compare_features` 区分不变标量与等变状态；`measure_resources` 反向但不更新；`BudgetMonitor` 检查计算时间与进程 RSS |
| [补充对照](../../tools/eval/periodic_message_controls.py) | `recreate_probes` 重现并校验原权重哈希；`angle_control` 固定径向字段；`resource_control` 在独立进程测同一案例 |
| [产物复核](../../tools/eval/verify_periodic_message_precheck.py) | `verify` 重新核对输入、源码、产物、摘要和覆盖；不重运行神经网络 |
| [原型测试](../../tests/test_periodic_message_prototypes.py)、[判据测试](../../tests/test_periodic_message_precheck.py) | 12 项原型契约与 3 项证据判据检查；含错误向量帧、NaN、方向/距离不一致和梯度对照 |

A 对每条边生成标量径向消息，对每个中心完整枚举无序邻边对，交换两边后平均角度消息。
按 4,096 对分块，反向使用非重入 checkpoint 重计算；与不重计算、不同块大小的前向和
参数/输入梯度已对照。B 使用 `64×0e+8×1o+4×2e`，保留方向状态，并把逐通道范数平方
直接送入标量更新。两者都显式读取真实镜像计数与平滑有效计数，填充归零，有效空邻域保留。

实施中发现 B 最后一次方向门控发生在标量读出之后；其权重 768 个、偏置 12 个全无有效
梯度。删除此门控保持标量输出公式，B 由初稿 181,456 降至 **180,676** 参数；A 为
**261,568**。不含共同 ZP、全局层或 decoder/head。加强后的合成检查确认两块各参数张量
均有有限非零梯度；正式 16 个真实结构的反向检查也没有完全不活跃的参数张量。
这不等于每个参数元素都有非零梯度，或已经验证训练可拟合性。

## 固定协议与证据

环境：PyTorch 2.2.1、e3nn 0.5.6、CPU 数值计算单线程；另有 20 ms RSS 采样线程。
seed 为 20261003；float64 创建主权重及表示矩阵，再转换为各 dtype。两条路线共享输入/输出
投影，ZP 和径向特征一致。局部块无 dropout，eval 固定权重。两套原型均未接入生产模型。

每个 split 固定 8 个样本：最大邻边对数、最大入度、最小晶胞奇异值、最少/最多原子，
再按邻边对数的 10%/50%/90% 分位补足去重后的 8 个。这个集合有意包含极端值，
统计中位数不能当作全数据集的无偏成本估计。完整 ID 与选择理由见
[samples.json](../../results/periodic_message_precheck_q1_r6_20261003/samples.json)。

正式产物：[manifest](../../results/periodic_message_precheck_q1_r6_20261003/manifest.json)、
[summary](../../results/periodic_message_precheck_q1_r6_20261003/summary.json)、
[逐项结果](../../results/periodic_message_precheck_q1_r6_20261003/cases.json)、
[独立复核](../../results/periodic_message_precheck_q1_r6_20261003/verification.json)。
源码、结构输入与上轮半径表均登记哈希；各 dtype 的参数及缓冲区起止哈希完全一致。

实际入口命令如下；重新执行须选择新目录。运行时 OMP/MKL/OpenBLAS 线程均为 1，
`CUDA_VISIBLE_DEVICES` 为空，命令中的产物目录已有结果，脚本会拒绝覆盖。

```bash
python tools/eval/periodic_message_precheck.py --out-dir results/periodic_message_precheck_q1_r6_20261003
python tools/eval/periodic_message_controls.py --precheck-dir results/periodic_message_precheck_q1_r6_20261003 --mode angles --budget-seconds 60 --out results/periodic_message_controls_q1_r6_20261003/angles.json
python tools/eval/periodic_message_controls.py --precheck-dir results/periodic_message_precheck_q1_r6_20261003 --mode invariant --budget-seconds 60 --out results/periodic_message_controls_q1_r6_20261003/invariant_resource.json
python tools/eval/periodic_message_controls.py --precheck-dir results/periodic_message_precheck_q1_r6_20261003 --mode equivariant --budget-seconds 60 --out results/periodic_message_controls_q1_r6_20261003/equivariant_resource.json
python tools/eval/verify_periodic_message_precheck.py --precheck-dir results/periodic_message_precheck_q1_r6_20261003 --controls-dir results/periodic_message_controls_q1_r6_20261003
```

在运行前固定无量纲特征判据 `|Δ|≤atol+rtol×|reference|`：

| dtype | 直接笛卡尔变换 atol / rtol | 晶胞重新编码后的 atol / rtol |
|---|---:|---:|
| float32 | 2e-5 / 2e-5 | 1e-4 / 1e-4 |
| float64 | 2e-10 / 2e-10 | 2e-6 / 2e-6 |

重编码判据计入已知晶格重建 stabilizer 和坐标舍入，不能用 Å 距离容差代替神经特征容差。
全部物理记录仍按原 [几何判据](log-2026-10-03-periodic-geometry-acceptance.md)核对，
截止键集差异单列，不扩大半径。直接变换使用 `r'=rQ`，B 状态以与行向量规则一致的
`D(Qᵀ)` 比较；另有负例确认“原向量数组不变”会被判为失败。

## 正式检查结果

**840 项全部通过，无截止键集差异。**这是按下述固定判据得到的局部证据。

| 检查 | 项数 | 实际范围 |
|---|---:|---|
| 任意物理帧中的旋转/反射/反演 | 400 | 16 个真实结构及 4 个合成例、两 dtype、两路线、5 个固定正交变换；比较返回标量及各层状态 |
| 重建坐标的物理 E(3) 对照 | 84 | 所有真实样本的整体平移；合成例另含物理旋转、反射；重新编码后独立映射回物理框架 |
| 原子重排、整数镜像、换基 | 272 | 16 个真实结构及斜胞合成例，四种表达；与 E(3) 分开报告 |
| 2×1×1 超胞 | 8 | train/valid 最大邻边对数两个结构、两 dtype、两路线；按两个中心副本对应状态，记录总数为原来的两倍 |
| 前向/反向资源与梯度 | 32 | 16 个真实结构、float32、两路线；检查梯度有限、两块活跃、权重未更新 |
| 填充、自身镜像、空邻域及 L=0 | 24 | 六个合成边界例、两 dtype、两路线 |
| 距离、角度、计数及几何 VJP | 20 | 同元素星形邻域；固定图 ID 下检查距离和方向入口的非零梯度 |

返回标量及受检查中间状态的最大绝对分量差如下；不能换算为谱误差。

| 检查 | A float32 / float64 | B float32 / float64 |
|---|---:|---:|
| 直接 E(3) 变换 | 9.54e-7 / 1.33e-15 | 7.15e-7 / 1.78e-15 |
| 物理坐标重编码 | 2.12e-6 / 1.99e-8 | 3.17e-6 / 3.22e-8 |
| 表达一致性 | 2.38e-6 / 2.25e-8 | 2.77e-6 / 4.39e-8 |
| 超胞对应状态 | 1.55e-6 / 2.91e-15 | 2.12e-6 / 3.33e-15 |

## 资源：A 完整角度枚举成本明显更高

以下是选定 16 个结构、float32、合成例预热后的单次测量，尚无重复测量区间。

| 成本 | A | B |
|---|---:|---:|
| 前向时间中位数 / 最大值（秒） | 0.118 / 3.159 | 0.00941 / 0.0777 |
| 含梯度前向加反向中位数 / 最大值（秒） | 0.830 / 21.951 | 0.0271 / 0.230 |
| 16 个结构的前向加反向合计（秒） | 87.383 | 1.171 |

同进程累计 RSS 不能归属某条路线。补充在全新进程测同一 train 最大案例：
`mp-aaabofsw`、64 原子、7,704 条记录、460,180 个无序邻边对；包括共同库、初始化和一条
路线的反向，A 峰值 **973.45 MiB**，B **565.13 MiB**。
原始结果见 [A](../../results/periodic_message_controls_q1_r6_20261003/invariant_resource.json)、
[B](../../results/periodic_message_controls_q1_r6_20261003/equivariant_resource.json)。
该组只有一次预热，保留原始时间但以它核对 RSS；正式表用于描述主要时间证据。
CPU 原型优势不能直接外推到 GPU、完整模型或训练速度；低入度案例 A 的固定开销也可能更小。

正式运行 562.67 秒，三个补充对照 36.22 秒，脚本计时合计 **598.89 秒**；回归与小型检查
另计，均在获准计算预算内。整轮最大进程 RSS **1,183.87 MiB**，未触及 2 GiB。
15 分钟指 CPU 核查计算预算，代码编辑和文档整理不计入。无训练、无 GPU。

## 必须保留的限制与新反例

**截止附近的角度内容很弱。**两组六邻居的距离、计数相同、向量和近零，但二阶方向矩不同。
补充对照把距离、RBF、权重和计数固定为逐位相同，只改变方向；恒等对照误差为零。
5.0 Å 时两路线均有可见角度响应；5.9 Å 时 float32 返回标量变化均为 **0**，float64 的 A/B
仅为 **1.66e-12 / 1.94e-11**，低于预先使用的 1e-10 响应阈值。B 的方向状态仍改变约
1.44e-5，但传到标量的内容非常弱。原始 float32 差异混有约 1.43e-6 Å 径向舍入，
不能作为截止附近角度已稳定可用的证据。

这与 A 的 `c_e c_f` 及 B 的方向范数平方引入的衰减相符，但尚未逐因素证明其全部原因。
正式近截止项为幅度诊断，未要求有可见响应，因此“840 项通过”不消除此限制。

**当前 B 存在低阶角度盲点。**构造 12 个同元素邻居，距离为 5.85 Å；方向集合由
`(0,±1,±a)` 及其两次循环置换归一化组成，分别取 `a=1.5` 和黄金比。
两组入度均为 `[12,1,…,1]`，邻居间无额外边；向量和与二阶无迹矩均近零，但无序对
`Σcos⁴θ` 为 **8.4406708 / 8.4**。用共享径向字段、`c=1` 的受控消融隔离衰减影响，
A 中心标量输出差为 **2.56e-5**，B 仅 **5.55e-16**。

公式解释是：同距离、同元素下边系数相同，中心 `l=1,2` 聚合消去；叶节点只有同一条径向
方向，第二层的标量收缩也不能恢复两个环境的高阶角关系。该解释限于当前状态、两层消息和
这个星形结构，不否定所有等变架构。A 的有限基与通道同样不保证所有环境唯一可分。
对照 [angles.json](../../results/periodic_message_controls_q1_r6_20261003/angles.json)包含十项：
其中 B 盲点项的 `pass` 表示成功复现反例，不能解释为 B 已通过充分表达能力验收。

## 结论、验证与下一步

在完整记录、相对几何、标量初始化等前提下，A 的不变量/交换对称求和和 B 的 O(3) 运算/
范数读出分别支持局部对称性的架构论证；有限数值检查与此一致。源码/输入/产物哈希、
JSON/JSONL/CSV、840 项摘要与覆盖、参数起止哈希和补充对照均由独立入口复核一致。
15 项新增回归加 20 项既有几何/半径回归共 **35 项通过**，最终回归过程含加载为 6.44 秒。

关键取舍现已具体：A 能保留本轮高阶角度反例的差别，完整邻边对枚举代价较高；B 资源较小，
当前低阶状态存在已确认的盲点。建议下一单元先评估给 B 补充高阶角度不变量的局部方案，
同时保留 A 作为表达对照；这一建议尚未实施或获选。不因 CPU 优势认定 B 的 DOS 更准确。

两条路线都尚未通过完整模型的输出不变性、预测收益、GPU/mixed precision、多 seed 或
任意晶胞/任意变换验收；5.9 Å 响应弱的训练影响未测。完整模型还需处理旧 `rp_proj` 几何
旁路、全局混合、任务 head、尺度与后处理，分别给出全链论证和谱输出数值证据。
