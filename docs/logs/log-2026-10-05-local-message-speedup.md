# 日志：2026-10-05 — A 局部消息的等价计算提速

## 范围

用户授权优化和测试训练前的计算成本。本轮采用不改变科学公式的实现优化，
正式训练预算、路线/编排选择与准确率门槛仍待决定。
FP32、6.0 Å 严格 `d<R` 全镜像、全部同中心无序邻边对、内部两块、
ZP/M1/SumNorm/H1、K=2 检查配置与既有共同初值协议保持。
未启动正式 DOS 训练；成本/等价性/E(3) 探针没有创建 optimizer、读取谱标签或 test、
加载或写入 checkpoint。仓库回归测试继续使用各自既有的合成数据与测试流程。

### 调用链与变更

完整模型仍经 `Transformer → LocalGlobalEncoder → PeriodicLocalMessage` 调用局部消息。
公开局部接口、参数名、参数形状和默认构造选项保持。

| 文件 | 符号与职责 |
|---|---|
| [`model/periodic_messages.py`](../../model/periodic_messages.py) | `_MessageGeometry` 存稳定排序与可选全量邻边对；`groups` 改为诊断时派生；`_prepare_message_geometry` 在 A 的一次局部调用内准备并复用两块的邻边对索引 |
| 同上 | 新增 `_unordered_pairs`，向量化生成原上三角顺序的全部同中心邻边对；`InvariantMessageBlock._angle_chunk` 替换逐中心 `_angle_sum`，跨中心按原 `pair_chunk=4096` 分块聚合 |
| 同上 | `InvariantMessageBlock.forward` 复用角度 MLP 首层的原子/径向线性投影；`PeriodicLocalMessage.forward` 仅 A 请求邻边对，B+ 不枚举角度对 |
| [`tests/model/test_periodic_messages.py`](../../tests/model/test_periodic_messages.py) | 新增全量邻边对身份/顺序对照，以及不同分块、重计算开关下输出、全部参数和输入几何梯度对照 |
| [`tools/checks/check_local_message_speed.py`](../../tools/checks/check_local_message_speed.py) | `reference_module` 读取并记录 Git 原实现身份；`structural_batches` 选合格 train 结构；`forward_profile/check` 做无更新成本短测；`verify` 对照完整模型输出与全部注册参数梯度；`main` 提供可复用入口 |

记录构建、外层残差、全局注意力、decoder/head、loss 与训练入口均未修改。
没有跨参数更新缓存原子状态或可学习激活；本次邻边对索引只在同一次局部调用的两块内复用，
尚未实现跨局部调用或跨 epoch 的几何缓存。

### 等价性依据与限制

原角度 MLP 首层对拼接特征作仿射变换。按原列切分其权重，有：

\[
W_c h_c+W_1 h_1+W_2 h_2+W_{r1}r_1+W_{r2}r_2+W_a a+b.
\]

新实现提前计算可复用的原子项和径向项，仍在**完整仿射和之后**应用原 SiLU 与第二层 Linear。
两种邻边交换次序、平均、截断乘积、计数归一化和残差更新保持。
邻边对身份及中心内上三角顺序与原枚举一致，没有采样、删边或缩小半径。
这支持实数运算下的等价性；GPU 求和和仿射分解的浮点舍入有所变化，
以既定数值容差验收，不承诺逐位相同或长期训练轨迹完全相同。

## 证据

### 固定条件与原始证据

硬件为 Tesla V100-SXM2-32GB，PyTorch 2.2.1；单 CPU 线程，关闭 AMP/TF32。
完整模型为 512 维、8 heads、2048 维 FFN、6 层共享 decoder、Q1 查询坐标、dropout 0.05，
A/B+ × staged/alternating、K=2、seed 42、`local-global-init-v1`。
A/B+ 完整参数数分别为 63,467,716 / 63,314,124；两种编排的更新计数与初值仍匹配。

原实现来自提交 `553c64d` 的 `model/periodic_messages.py`，诊断工具保存完整提交号与源码 SHA256。
对照通过严格 state_dict 复制与完整状态哈希确认同权重；上述 GPU 探针前后权重/缓冲区哈希保持。
此 Git 读取只用于诊断原实现，不成为生产模型的依赖。

证据目录为 `results/runs/local_message_speed_20261005/`：

- `reference.json` / `optimized.json`：原实现与优化实现的同批次短测。
- `equivalence_fp32.json` / `equivalence_fp64.json`：两种 dtype 的四组完整模型输出与全部参数梯度。
- `batch_sweep.json` / `no_recompute.json`：大 batch 与关闭角度重计算的资源探针。
- `e3_fp32.json` / `e3_fp64.json`：优化后完整模型的 E(3) 检查，结果见下节。
- `reference_checker.py`：初次成本测量时的检查器源码快照，匹配 `reference/optimized.json` 中记录的检查器哈希；随后只增加完整模型等价性验证入口，成本测量函数未改。

原始结果和运行日志按仓库规则留在被忽略的 `results/runs/`；本日志记录结论与复现方式。
既有验收报告、冻结原型和 checkpoint 均保留。

### A 的完整模型成本

合格 train 视图为 18,704 条。固定随机排列 seed=0，排除 manifest 中两条质量隔离记录。
普通批次 32 个结构、352 个原子、最大 40 原子，16,054 条镜像记录、426,571 个无序邻边对，
实际计算批次 `[32]`。尾部批次含合格视图最大 160 原子结构，共 510 原子、27,182 条记录、
812,157 个邻边对，按现有训练资源规则拆为 `[31,1]`，反向按结构数加权。

以下是**实测中位数**，每项预热两次、计时五次；FP32，默认角度重计算开启：

| 编排/批次 | 原 eval 前向 ms | 新 eval 前向 ms | 原 train 前向+反向 ms | 新 train 前向+反向 ms | 前向+反向提速 | 原/新峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| staged / 普通 32 | 1147.39 | 381.91 | 5101.41 | 1771.68 | 2.88 倍 | 3197.86 / 3252.04 |
| alternating / 普通 32 | 1181.13 | 388.87 | 5176.11 | 1745.60 | 2.97 倍 | 3200.20 / 3255.77 |
| staged / 含 160 原子 | 1690.84 | 654.89 | 7498.94 | 3205.78 | 2.34 倍 | 3163.30 / 3220.11 |
| alternating / 含 160 原子 | 1716.52 | 665.26 | 7472.65 | 3189.68 | 2.34 倍 | 3170.27 / 3219.53 |

B+ 的原/新前向+反向为普通批次 296.76/294.46 ms（staged）、297.69/291.41 ms（alternating），
尾部批次 411.31/403.92 ms、409.58/408.11 ms，成本基本保持。
首次 B+/staged 的旧 eval 前向 195.56 ms、新 138.47 ms 与其他 B+ 前向存在短测波动，
不单独解释为此优化的稳定收益。

前向+反向计时复用现有 `check_crystal_input.profile`，包含记录构建、完整模型、
构造的谱/尺度平方损失和逐参数有限梯度检查；不含 optimizer 更新、加载谱标签、数据加载器与整轮验证。
峰值为 PyTorch CUDA allocated，不含 AdamW 状态和 CUDA 上下文，因此不是正式训练显存上限。
旧的“两点外推约 7 天”估算不能直接按本表写成新的整轮预算，尚未实测整轮吞吐。

### batch=64/128 的成本探针

随机排列的嵌套前 64/128 个结构分别为 732/1515 原子、36,690/78,560 条记录、
1,017,366/2,283,282 个邻边对。现有资源控制拆为 `[46,18]` / `[92,36]`；
不同大小含不同结构和邻边对数，以下吞吐不能单独归因于 batch 大小。

| 路线/编排 | batch 32：结构/s，峰值 GiB | batch 64：结构/s，峰值 GiB | batch 128：结构/s，峰值 GiB |
|---|---:|---:|---:|
| A / staged | 18.1，3.18 | 15.4，4.98 | 13.2，9.38 |
| A / alternating | 18.3，3.18 | 15.5，4.98 | 13.1，9.38 |
| B+ / staged | 108.7，3.30 | 105.1，5.17 | 122.4，9.82 |
| B+ / alternating | 109.8，3.29 | 103.7，5.17 | 122.2，9.82 |

所测 A 批次没有吞吐收益；B+ 的 128 批次比 32 批次约快 11%–13%，但显存更高。
训练 batch 增大还改变每 epoch 的 optimizer 更新次数与梯度噪声，未做准确率对照。
本轮没有采用更大训练 batch，也没有据此分别改变四组共同训练条件。

### 关闭角度重计算的成本探针

角度重计算指反向时重新求中间角度消息以节约激活显存，原默认为开启。
检查工具在同权重模型上临时关闭 A 的既有开关，没有修改生产默认值：

| A 编排/批次 | 关闭后的前向+反向 ms | 相对开启耗时减少 | 关闭后峰值 GiB |
|---|---:|---:|---:|
| staged / 普通 32 | 1243.56 | 29.81% | 6.83 |
| alternating / 普通 32 | 1249.39 | 28.43% | 6.83 |
| staged / 含 160 原子 | 2278.07 | 28.94% | 6.80 |
| alternating / 含 160 原子 | 2277.28 | 28.60% | 6.79 |

这说明存在用显存换时间的进一步选项；已新增 CPU 输出/梯度回归覆盖两种开关。
但正式训练包含 optimizer 状态，数据尾部也不只这一批，默认仍开启重计算。
关闭开关的 batch=128 未实测，不能把两个独立探针合并当作已验收配置。

### 数值、对称性与回归

完整模型等价性采用普通/尾部两个真实 train 批次，两 dtype、四组、共 16 次原/新配对。
每组 235 或 239 个参数张量全部有梯度，无新增断连；模型处于 eval 以排除 dropout 差异，
构造谱/尺度平方损失，仅用于比较输出与梯度。

| dtype | eDOS 输出最大绝对差 | phDOS 输出最大绝对差 | eta 最大绝对差 | 全部参数梯度最大绝对差 | 最大整体相对 L2 |
|---|---:|---:|---:|---:|---:|
| FP32 | 2.414e-6 | 6.985e-8 | 5.364e-7 | 5.960e-7 | 7.938e-7 |
| FP64 | 3.775e-15 | 1.041e-16 | 8.882e-16 | 1.332e-15 | 8.348e-16 |

输出容差 FP32 `atol=rtol=2e-5`、FP64 `2e-10`；全部参数梯度沿用现有 GPU
`compare_gradients`，FP32 `atol=2e-6, rtol=2e-3`，整体及超出绝对噪声底的逐张量相对 L2≤3e-3；
FP64 `atol=2e-11, rtol=1e-8`、相对 L2≤1e-8。所有检查通过。
新增的冻结原型 CPU 回归同时比较原子输入、位移、距离与全部参数梯度，
覆盖 chunk=2/7/4096、重计算开/关、两 dtype、非连续中心分组和空/单邻边。

完整 GPU E(3) 复查通过：两 dtype、四组、六种操作（平移、两次旋转、反射、反演、
含平移的非正旋转），共 48 个模型/变换组合；每组使用相同三个真实结构，检查记录拓扑、
两类 logits、eta、softmax 谱形与 H1 blind 重建。
所有输出最大绝对差 FP32 为 1.133e-6、FP64 为 4.219e-15；对应容差为 2e-5 / 2e-8。
FP32 另检查 192 原子个例的前向/反向，仅用于结构压力测试，不纳入合格 train 成本批次。
这仍是有限输入/变换检查，架构论证沿用[接入设计](../design/design-2026-10-04-local-global-encoder.md)。

`tools/checks/check-static.sh` 通过：Ruff、内存语法编译、**257 项回归**（97.253 s），
仓库政策检查通过（697 个 prospective 文件）。新增两项回归纳入原检查入口，
原有局部 14 项和完整 Encoder 8 项检查保持通过；这些数量不与 257 项相加。
`git diff --check` 通过，五份改动文档的 106 个本地引用均可解析；八份 GPU 报告的
源码身份与最终实现/保存的原实现、原检查器快照一致，成本测量函数与原快照的语法树相同。

### 复现入口

在新的输出路径运行，工具拒绝覆盖已有报告：

```bash
python tools/checks/check_local_message_speed.py --mode reference --out /tmp/lg-reference.json
python tools/checks/check_local_message_speed.py --mode current --out /tmp/lg-current.json
python tools/checks/check_local_message_speed.py --mode verify --precision fp32 --out /tmp/lg-equivalence-fp32.json
python tools/checks/check_local_message_speed.py --mode verify --precision fp64 --out /tmp/lg-equivalence-fp64.json
python tools/checks/check_local_message_speed.py --mode current --batch-sizes 64 128 --out /tmp/lg-batch-sweep.json
python tools/checks/check_local_message_speed.py --mode current --no-recompute-angles --out /tmp/lg-no-recompute.json
```

均为无更新结构检查。正式 E(3) 复查复用 `tools/checks/check_local_global.py`，
仓库完整检查复用 `tools/checks/check-static.sh`，不另造训练入口。

## 结论与交接

状态：**closed**，成本、原实现数值对照、完整 GPU E(3) 和仓库检查通过。
A 的等价计算优化已实施，默认 batch、重计算选项、科学配方和实验契约保持。
未比较 valid eDOS/phDOS 的 oracle/blind 指标，无训练准确率结论。
后续共同确定正式训练预算；大 batch、关闭重计算或其他配置若采用，应明确资源与训练条件。
