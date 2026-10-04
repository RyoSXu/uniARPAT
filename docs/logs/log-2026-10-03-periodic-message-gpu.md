# 6.0 Å 独立 A/B+ 的 GPU 验收与 CPU 成本对照

日期：2026-10-03。结论：**独立局部模块在本次 GPU 范围内通过；没有启动训练。**
用户批准测试 GPU，沿用已提取的[模块边界](../design/design-periodic-message-module-boundaries.md)、
严格 `d<6.0 Å` 和原固定 16 个 Q1 train/valid 结构。A/B+ 都通过，未选定最终生产 Encoder。
这项验收不代表完整 eDOS/phDOS 模型已经满足 E(3)，也不比较预测准确率。

## 授权、环境与实际范围

- 主检查预算：720 秒、CPU RSS 2 GiB、PyTorch CUDA 分配器 8 GiB；实际 259.64 秒，
  RSS 峰值 1,708.73 MiB。随后短测预算 90 秒，同内存上限，实际 22.94 秒、RSS 1,112.24 MiB。
- GPU 为 Tesla V100-SXM2-32GB；PyTorch 2.2.1 / CUDA 12.1 / e3nn 0.5.6。
  CPU 数值与计时固定为 Torch 单线程；TF32 关闭。检查 FP32 与 FP64，没有使用 AMP。
- 固定无训练权重从原型严格迁移到独立核心，CPU/GPU 每个 dtype 的状态哈希一致，
  运行前后哈希不变；参数量仍为 A 261,568、B+ 184,772。
- 只加载晶体结构、元素与编号，不加载谱标签、test、checkpoint；没有 optimizer 或参数更新。
  blind/oracle 谱模式不适用。旧原型、既有 CPU 证据、物理记录核心、ZP 和生产 Transformer 均未修改。

主检查在 16 个真实结构和六组合成例上调用新核心。合成例覆盖自身跨胞镜像、倾斜晶胞、
混合填充、有效空邻域、全填充和零原子槽。真实结构保留此前按镜像/角度对成本及结构极值
选出的 train/valid 各八个样本，完整清单与哈希见 manifest，没有按本轮耗时重新选样。

## 文件、符号与调用链

| 新增文件 | 职责与主要符号 |
|---|---|
| [`periodic_message_gpu_check.py`](../../tools/eval/periodic_message_gpu_check.py) | `GPUBudget` 控制资源；`numeric_checks` 检查几何/特征/状态，`compare_local` 检查标量、等变状态、角度统计与整数计数；`local_gradients` / `compare_gradients` 检查连续输入和参数梯度；`pipeline_checks` 检查最小注册链；`benchmark_samples` / `benchmark` / `measured_forward` / `summarize_benchmark` 固定批次并记录同步计时 |
| [`confirm_periodic_message_gpu_timing.py`](../../tools/eval/confirm_periodic_message_gpu_timing.py) | `main` 使用相同批次、FP32 和权重补做充分预热后的 GPU 前向加反向，不替换首轮原始耗时 |
| [`verify_periodic_message_gpu_check.py`](../../tools/eval/verify_periodic_message_gpu_check.py) | `verify` 独立核对覆盖、各项判据、计时中位数、CSV/JSON 与哈希；不重运行神经网络 |
| [`test_periodic_message_gpu_check.py`](../../tests/test_periodic_message_gpu_check.py) | 四项 CPU 回归，检查梯度错误、B+ 角度统计遗漏、临界键差异与批次/中位数统计 |

检查调用 `build_periodic_records → PeriodicNeighborRecords → PeriodicLocalMessage`。
CPU/GPU 使用同样的标量输入及权重；另在 GPU 原生构建物理记录，与 CPU 的逐条记录对齐。
最小 GPU 链是同公式 ZP 测试组件 → 独立 A/B+ → 标量线性测试读出；并非生产 Encoder 或 DOS head。
反向只用于验收，不进行参数更新。

## 数值与几何结果

主检查 **966/966 通过**，无失败、跳过或临界键集差异：

| 类别 | 项数 | 实际检查 |
|---|---:|---|
| GPU 原生几何 | 44 | 两 dtype × 22 案例；记录身份与数量一致，距离及位移在容差内 |
| CPU/GPU 特征、GPU 重建特征 | 88 + 88 | A/B+ 的输出、层状态、角度统计、degree/pair_count |
| GPU 正交变换 | 440 | 每案例/路线五个正交变换，含旋转、反射、反演；标量不变、B+ 状态按表示等变 |
| GPU 填充 | 80 | 向填充特征写入 NaN 后，有效输出有限、填充输出为零；全填充/零槽另含在前述案例中 |
| GPU 物理平移 | 64 | 16 真实结构 × 两 dtype × 两路线；坐标实际平移后在 GPU 重建记录 |
| 表达一致性 | 48 | 倾斜合成例与两真实中位成本结构；重排、整数坐标镜像、换基交换/剪切 |
| CPU/GPU 梯度 | 12 | 上述三个结构 × 两 dtype × 两路线；输入、独立连续位移/距离与全部局部参数 |
| 最小 ZP→局部→下游链 | 2 | FP32 两路线，参数注册无遗漏、各参数张量梯度有限且非零、权重不变 |
| 批次准备、资源计时 | 4 + 96 | 四组固定批次；两路线 × 两设备 × 前向/前向反向 × 三次 |

正交检查把每条物理位移作 `r→rQ`，保留镜像身份和真实距离，再执行 GPU 局部消息。
它不把仅有晶胞长度/角度的规范框架重建当成物理旋转。表达一致性单列：此处的变换与
物理框架重建沿用 CPU 几何诊断，再传入 GPU 消息；**未新增 GPU 超胞或任意换基穷举**。

实际误差是输出与层状态的最大绝对差，B+ 的等变状态按相应表示对齐；不是谱误差：

| 量 | FP32 最大误差 | FP64 最大误差 |
|---|---:|---:|
| 原生 GPU 几何距离差（Å） | 3.815e−6 | 3.553e−15 |
| 原生 GPU 几何向量差（Å） | 3.815e−6 | 3.109e−15 |
| CPU/GPU 输出与层状态 | 2.638e−6 | 4.996e−15 |
| GPU 正交变换后输出与对齐状态 | 1.669e−6 | 2.665e−15 |
| GPU 原生重建后输出与状态 | 3.695e−6 | 3.969e−15 |
| GPU 实际平移后输出与状态 | 3.248e−6 | 3.657e−15 |
| 表达一致性后输出与状态 | 2.339e−6 | 4.388e−8 |
| CPU/GPU 梯度绝对差 | 2.515e−8 | 5.811e−17 |
| 全部梯度合并的相对 L2 差 | 8.310e−7 | 1.211e−15 |

沿用既定特征容差：直接比较 FP32 `atol=rtol=2e−5`、FP64 `2e−10`；几何重建与表达
一致性 FP32 `1e−4`、FP64 `2e−6`。距离/位移容差 FP32 `1e−4 Å`、FP64 `1e−6 Å`。
记录身份、degree 和 pair_count 要求整数精确一致，容差不扩大 6.0 Å 邻域。

梯度容差在执行前固定：FP32 `atol=2e−6, rtol=2e−3, relative_L2≤3e−3`，
FP64 `atol=2e−11, rtol=1e−8, relative_L2≤1e−8`。各张量检查逐元素差；当参考梯度范数
高于 `atol·sqrt(numel)` 时，另要求实际非零及张量相对 L2 达标；全部梯度合并也要求相对 L2 达标。
FP32 的位移张量，以及 B+ 第一块的 `direction_gate.weight/bias`，在三个梯度案例中低于
绝对尺度门槛，仅使用逐元素差判断，明细保留这项限制；FP64 没有低于该门槛的梯度张量。
这不承诺所有参数通道在所有晶体上都有非零梯度，也未检查通过离散邻居集合变化求导。

## CPU/GPU 计算成本

下列批次由原固定样本预先组成，名称 `typical` 只是脚本标签，不表示随机训练批次的分布。
较密单结构是原 train 最大角度对案例，不是更大的 batch：

| 批次 | 有效原子 | 周期记录 | 潜在无序角度对 | CPU 几何准备 ms | 传 GPU 并校验 ms |
|---|---:|---:|---:|---:|---:|
| batch 1 | 5 | 276 | 7,509 | 5.92 | 1.30 |
| batch 4 | 47 | 2,912 | 112,304 | 17.66 | 1.61 |
| batch 8 | 57 | 3,240 | 121,148 | 27.49 | 1.73 |
| 较密单结构 | 64 | 7,704 | 460,180 | 20.00 | 1.54 |

FP32，CPU 单线程；每组预热一次、测量三次，以同步墙钟中位数比较。
GPU 在计时前后同步；前向加反向计算 `output.square().mean()` 并取得局部参数/输入梯度。
计时**包含**局部 forward 内派生 RBF/球谐、分组/角度对、接口校验及反向重算，
**不包含**结构加载、镜像构建、传输、输入克隆、ZP、正式 Encoder/head、优化器或梯度检查。
几何准备与传输另列，不能把本表当整轮训练吞吐。

| 批次 | A CPU ms | A GPU ms | A 加速 | B+ CPU ms | B+ GPU ms | B+ 加速 |
|---|---:|---:|---:|---:|---:|---:|
| batch 1 | 424.40 | 38.30 | 11.08× | 25.63 | 23.71 | 1.08× |
| batch 4 | 5,497.60 | 394.39 | 13.94× | 94.98 | 24.33 | 3.90× |
| batch 8 | 6,093.33 | 466.07 | 13.07× | 125.24 | 24.48 | 5.12× |
| 较密单结构 | 21,710.41 | 1,094.79 | 19.83× | 222.62 | 24.82 | 8.97× |

GPU 纯前向中位数，依次为 batch 1/4/8/较密单结构：
A 为 10.25/84.75/100.36/182.08 ms，B+ 为 9.41/9.88/9.97/9.84 ms。
A 显式枚举角度对的成本随角度对数量上升；本组 batch 8 的 GPU 前向加反向 B+ 约为 A 的 1/19.04。
这是当前实现和有限批次的实测，不能作为所有硬件/批次的普遍速度比，CPU 多线程未比较。

首轮 batch 4 的 B+ GPU 反向原始三次为 **583.19/24.33/24.32 ms**。
未定位该峰值原因，原始耗时保留。因这项不确定性，补做三次预热、五次测量的 GPU 短测，
保持结构和权重相同：

| 批次 | A 中位 ms（最小–最大） | B+ 中位 ms（最小–最大） |
|---|---:|---:|
| batch 1 | 38.58（38.36–38.78） | 23.23（22.78–24.33） |
| batch 4 | 394.85（389.56–477.47） | 24.07（23.37–24.59） |
| batch 8 | 467.20（460.87–545.22） | 23.67（23.43–24.16） |
| 较密单结构 | 1,094.70（1,087.54–1,180.11） | 23.02（22.78–23.50） |

补测支持此前 B+ 的约 24 ms 中位成本；A 仍有可见的较高耗时波动，原因未定位。
主表使用首轮匹配的 CPU/GPU 中位数，没有把补测 GPU 和首轮 CPU 偷换拼接。

整轮 PyTorch CUDA 分配峰值 **109.43 MiB**、保留峰值 **128.00 MiB**。
这是诊断进程的总值，包括同时驻留的模型/数据与临时张量，未隔离单一路线的净增量；
不包括驱动/CUDA 上下文及库占用，未包含正式模型、optimizer 状态、训练数据管线。
因此只能说明本次检查未碰到内存上限，不能用来确定实际训练 batch 上限。

## 回归、独立复核与复现

新增四项 CPU 回归全部通过：

```bash
PYTHONWARNINGS=ignore::DeprecationWarning python -m unittest discover -s tests -p test_periodic_message_gpu_check.py -v
```

独立产物复核通过：966 条 JSON/JSONL 一致，CSV 46,368 个字段一致；覆盖、逐项数值/几何
判据、梯度门槛、四组批次、三次计时中位数/速度比和八组短测均已重算。
输入、当前核心与新工具、旧原型和原 CPU 产物哈希一致。复核没有重运行网络。
核心本轮未修改，未重复此前已通过的 56 项 CPU 回归。

主产物：[manifest](../../results/periodic_message_gpu_q1_r6_20261003/manifest.json)、
[summary](../../results/periodic_message_gpu_q1_r6_20261003/summary.json)、
[明细](../../results/periodic_message_gpu_q1_r6_20261003/cases.csv)、
[独立复核](../../results/periodic_message_gpu_q1_r6_20261003/verification.json)；
[预热补测](../../results/periodic_message_gpu_timing_confirmation_q1_r6_20261003/summary.json)。
入口拒绝覆盖已有目录；复现应选择新的结果目录：

```bash
PYTHONWARNINGS=ignore::DeprecationWarning python tools/eval/periodic_message_gpu_check.py \
  --out-dir results/periodic_message_gpu_recheck \
  --budget-seconds 720 --memory-mib 2048 --gpu-memory-gib 8
PYTHONWARNINGS=ignore::DeprecationWarning python tools/eval/confirm_periodic_message_gpu_timing.py \
  --acceptance-dir results/periodic_message_gpu_recheck \
  --out-dir results/periodic_message_gpu_timing_recheck
PYTHONWARNINGS=ignore::DeprecationWarning python tools/eval/verify_periodic_message_gpu_check.py \
  --out-dir results/periodic_message_gpu_recheck \
  --confirmation-dir results/periodic_message_gpu_timing_recheck
```

## 判断与仍未验证的范围

**事实：**两路线在本轮 GPU 数值、局部对称性、计数/填充和反向检查范围内通过。
对所测较大或较密批次，GPU 比单线程 CPU 更快；B+ 的 GPU 局部成本显著小于 A。
极小单结构上的 B+ CPU/GPU 前向加反向相近，不能把大批次速度比直接套用到它。

**推断：**GPU 可作为后续实际训练的设备候选，现有资源证据支持继续制定完整模型接入方案。
这尚不选择生产路线；A/B+ 的未训练特征与计算成本不能判断 DOS 收益。

**未验证：**完整 Transformer 的 E(3) 架构与 eDOS/phDOS 输出；现有方向旁路 `rp_proj`
仍需处理。没有测试 AMP、其他 GPU/库版本、任意变换/晶胞表达、GPU 超胞、完整模型内存与
端到端吞吐、训练后的对称性或准确率。此前正常截断下的近截止弱角度响应仍未修正。
完整接入、初始化协议、实际训练批次、预算与判据需另立方案；本轮没有启动训练或提交。
