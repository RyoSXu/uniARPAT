# A/B+ 局部消息提取解耦与 CPU 接口验收

日期：2026-10-03。状态：**本小单元通过；独立核心已实现，尚未接入完整模型或训练。**
用户要求先控制内聚性，独立于 ZP 与后续全局 Encoder；随后明确批准提取解耦。
本轮只提取已接受公式、明确输入输出和运行 CPU 迁移检查，不选择生产 Encoder 或启动训练。
本日志记录当时的 CPU 小单元；随后完成的 GPU 检查见
[GPU 验收](log-2026-10-03-periodic-message-gpu.md)，最新进度见[当前状态](../status.md)。
边界见[模块设计](../design/design-periodic-message-module-boundaries.md)，历史公式与限制见
[候选](../design/design-periodic-message-candidates.md)及[B+ 对照](log-2026-10-03-periodic-message-high-order.md)。

## 文件、符号与职责

| 模块 → 文件 | 新增或修改符号 | 职责 |
|---|---|---|
| 物理几何 → [`utils/periodic_geometry.py`](../../utils/periodic_geometry.py) | `_validate_radius`、`validate_periodic_inputs`、`periodic_edge_vectors`、`PeriodicNeighborRecords`、`build_periodic_records` | 六字段与显式半径/掩码元数据；输入拒绝、复用完整镜像枚举、中心到邻居位移与距离校验；`__post_init__`/`from_edges`/`to` 管理记录契约和迁移 |
| 局部消息 → [`model/periodic_messages.py`](../../model/periodic_messages.py) | `_MessageGeometry`、`_prepare_message_geometry`、`mlp`、`count_features` | 局部拥有 RBF、权重、入度/有效计数、无序对数及分组；物理记录不绑定这些派生编码 |
| 同文件 → A | `InvariantMessageBlock.__init__`、`_angle_sum`、`forward` | 复用全部无序邻边对、交换对称 MLP、计数缩放与重算；邻边对索引创建在输入 device 上 |
| 同文件 → B+ | `high_order_pair_features`、`HighOrderEquivariantBlock.__init__`、`invariants`、`forward` | 三/四阶统计、0e/1o/2e 等变消息、偶标量更新；直接从本组件配置构造，不复制 B 对象 |
| 同文件 → 公共接口 | `PeriodicLocalMessage.__init__`、`forward` | `route="A"/"B+"`，共同 `h, records, padding_mask → h_local`；两局部块与当前 512↔64 投影，方向状态仅留在内部/诊断输出 |
| 包导入 → [`model/__init__.py`](../../model/__init__.py) | `__all__`、`__getattr__`、`__dir__` | 旧公开 `basemodel`/`Transformer` 按需解析；导入局部核心不加载旧完整模型、旧 Encoder 或训练工具 |
| 诊断 → [`periodic_message_module_check.py`](../../tools/eval/periodic_message_module_check.py) | `sha256`、`write_json`、`aligned_local_models`、`frozen_hashes`、`run_regressions`、`acceptance`、`main` | 实验层严格复制冻结无训练权重、核对输入/旧证据、运行真实结构迁移与回归并保存报告；预算与拒绝覆盖 |
| 测试 → [`test_periodic_message_modules.py`](../../tests/test_periodic_message_modules.py) | `records_from_probe`、`LocalModuleContracts` 及 14 个用例 | 导入隔离、公开类兼容、参数归属、数值与梯度迁移、几何/掩码/空邻域/异常输入、局部 O(3) 与表达一致性 |

内部常量继续采用 `SCALAR_CHANNELS=64`、`RADIAL_CHANNELS=16`、`STATE_IRREPS=64x0e+8x1o+4x2e`、
`EDGE_IRREPS` 到二阶、`HIGH_ORDERS=(3,4)`、`MOMENT_CHANNELS=8`、`EXTRA_SCALARS=16`。
没有删除旧符号；ZP 的模块名、公式、参数键以及 `Transformer.forward` 均未修改。

## 调用链与行为契约

```text
pos、padding_mask、显式 R
    → validate_periodic_inputs → build_g2_edges → periodic_edge_vectors
    → PeriodicNeighborRecords（物理六字段 + R/掩码元数据）
                                                    ↓
外部 ZP / 当前原子标量 h → PeriodicLocalMessage → h_local → 外部下游组件
                              ↓
                  内部径向/计数编码 → A 或 B+ 两块 → 不变标量读出
```

当前外部输入输出为 `[B,L,512]`，掩码 `True` 表示 padding；有效空邻域仍保留自身更新，
填充输出严格为零。跨胞自身、多镜像与反向记录保持原语义，严格 `d<6.0 Å`，不裁剪或按
原子编号合并记录。`from_edges` 支持外部正确笛卡尔框架；由调用者保证其完整性、去重与框架正确。
记录/模块通过显式 `.to(device, dtype)` 对齐，半径只来自记录；没有隐式 4.5/5.5 Å 替代值。

独立不是冻结：公共接口不 `detach` 原子特征。注册的最小 `ZP → 局部 → 标量下游` 测试容器
确认三个子模块的参数均被父模型收集，各参数张量存在有限非零梯度，没有 Parameter 别名共享。
没有创建 optimizer 或执行更新；这不能替代未来训练 runner 的参数/优化器验收。

核心仅依赖 Torch/e3nn 与几何工具。全新子进程确认：物理几何层不加载模型/诊断，局部核心
不加载 `model.model`、`model.transformer`、`model.periodic_manybody`、`tools.eval`、训练 runner
或 `utils.builder`。旧公开类解析后仍为原类对象，未知名称正确报错。

旧 CPU 原型、旧高阶增强、旧 Encoder 位移函数及旧结果保留冻结；它们只承担历史复核与迁移
数值参照。以后新增消息实验调用新核心，避免两个活跃实现。此次没有覆盖原证据或放宽旧指纹。
新 B+ 默认初始化不重演“先构造 B，再复制并新增列”的旧实验随机数协议；
`aligned_local_models` 的复制仅用于迁移验收，完整训练初始化仍须另定。

## 数据、验收与结果

只读取先前固定的 16 个 Q1 train/valid 结构（各 8 个），输入哈希与初轮完全相同；不加载
谱标签、test、checkpoint，不运行生产 DOS 模型或 GPU。blind/oracle 不适用于无训练特征检查。
运行环境为本地 PyTorch 2.2.1，Torch CPU 单线程；使用 FP32 与 FP64。

| 检查 | 数量与结果 |
|---|---|
| 真实结构物理字段迁移 | 32 项全部通过；记录数一致，六字段与旧构建逐元素相等，位移最大差 0 Å |
| 真实结构局部数值迁移 | 64 项全部通过；同权重输出/状态最大差 0，计数完全相等，B+ 额外角度统计相等 |
| 真实小结构局部旋转/反射 | 40 项全部通过；检查标量不变及 B+ 按表示矩阵变换的中间状态，不把方向数组原样不动当作通过 |
| 既有与新增回归 | 56 项全部通过（既有 42 + 新增 14），无跳过 |
| 参数量与权重 | A 261,568；B+ 184,772，保持原数量；正式检查前后权重哈希一致，无更新 |
| 正式 CPU 成本 | 91.65 秒；峰值 RSS 525.54 MiB；每次验收限 180 秒/2 GiB |

局部正交检查最大误差 FP32 为 `5.960464477539062e-7`、FP64 为 `1.1102230246251565e-15`，均在既定直接帧
特征容差（FP32 `2e-5`、FP64 `2e-10` 的 atol/rtol）内。数值迁移最大差 0 是同初值、
固定输入和当前环境的有限证据，不宣称所有环境逐位相等。
新增回归在合成斜胞、两 dtype 下比较输入、位移、距离和全部局部参数的梯度；
没有修改原计算式，也没有通过放宽容差修补迁移。

首次独立测试曾发现两类测试环境问题：子进程继承的 MKL 导入顺序冲突，以及 e3nn 表示矩阵
工厂读取默认 dtype 导致 FP64 对照不足精度。分别在测试子进程先初始化 NumPy、在物理帧
对照显式设置 FP64，并保持原容差。随后首次合并验收的 136 项数值检查全部通过，但测试容器
受全局 FP64 默认影响，与其 FP32 几何不匹配；将容器显式 `.to(pos.dtype)` 后完整复跑通过。
没有因此修改核心或旧证据。失败结果及当时五个源码快照保留于
[`periodic_message_modules_q1_r6_20261003`](../../results/periodic_message_modules_q1_r6_20261003/summary.json)；
首个合并检查用时 92.02 秒，两次合并检查共 183.67 秒，独立测试/导入另计。

正式产物：

- [manifest](../../results/periodic_message_modules_q1_r6_20261003_v2/manifest.json)：源码、冻结证据、输入和产物哈希，零更新与执行范围。
- [summary](../../results/periodic_message_modules_q1_r6_20261003_v2/summary.json)：136 项与 56 项回归摘要、资源。
- [cases](../../results/periodic_message_modules_q1_r6_20261003_v2/cases.json)：逐结构/精度/路线/变换明细。
- [regressions](../../results/periodic_message_modules_q1_r6_20261003_v2/regressions.txt)：实际执行的用例与结果。
- [verification](../../results/periodic_message_modules_q1_r6_20261003_v2/verification.json)：另行读取产物核对 136 项覆盖、56 项回归摘要、5 个新源码、14 个冻结源码、11 个旧产物、7 个输入文件及首轮 5 个源码快照的哈希；未重跑网络。

新增/改动源码语法与行尾检查、`git diff --check` 通过；相关六份文档的本地引用已核对。

复跑时需使用新目录，入口拒绝覆盖：

```bash
PYTHONWARNINGS=ignore::DeprecationWarning python tools/eval/periodic_message_module_check.py \
  --out-dir results/periodic_message_modules_q1_r6_NEW \
  --budget-seconds 180 --memory-mib 2048
```

## CPU/GPU 与未验收范围

CPU 是此前诊断的选择，不是实际训练的强制设备。新核心使用输入 device 上的张量、RBF 和
邻边对索引，支持常规 `.to`；本机只读 `nvidia-smi` 显示 Tesla V100-SXM2-32GB。
此轮没有 GPU 前向/反向、CUDA 时间或显存测量，因此“支持迁移”与“GPU 已验收”分别报告。

预计实际批量训练更适合 GPU，但这是待基准确认的工程判断。小图/小批量、频繁 Python
循环、核启动、CPU/GPU 同步和传输可能限制收益；PyTorch 的
[性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)说明这些开销，
也指出小模型或内存受限模型可以适合 CPU。A 的额外消息量随全部邻边对增长，B+ 的固定通道
矩统计随记录数增长；GPU 加速不会消除二者的算法规模差异，也不能用旧 CPU 约 27 倍差距
推算 GPU 差距。后续基准应固定结构、batch、权重和 FP32，分别报告预处理/传输与驻留设备
前向反向，CUDA 计时同步，并测峰值显存；是否开启 AMP 另验收。

仍未验证 GPU/AMP/实际 batch 成本，完整 Encoder 编排、旧方向打分旁路、两类最终谱与
尺度后处理的不变性，任意结构/变换的数值证明及 train/valid 谱收益。近截止弱响应未改变。
完整模型的架构保证、有限输出测试及新训练初始化/对照/预算仍是后续独立工作。
枚举器保留原 `no_grad` 边选择与距离返回，不增加邻居集合变化的可导保证；旧公共枚举器
其他调用方的输入防护没有自动修复。此次没有提交或推送。
