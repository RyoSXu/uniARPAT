# 工作日志 2026-10-04：完整原子与 FP32 晶体输入

## 目标与授权

用户要求建立科学、可解释的数据逻辑，并授权相关存储、读取、几何、训练计数和资源控制
同步适配；讨论来源为 [结构输入设计](../design/design-2026-10-04-crystal-structure-input.md)。
用户确认不再截断到 80 原子，在听取资源影响后明确要求开始。

本轮不改变 Encoder 架构、E0/P0 谱值或归一化公式，不启动正式训练，不覆盖旧 Q1、不操作
Git 提交/推送、不删除 checkpoint。工作区起始仅有用户未跟踪的该设计文件；在原内容后补充
实施契约并更新状态，没有删除讨论内容。

## 事实与来源

- A4 和 Topup 日志及 Git `e24d2a4^:tools/getdata/a4_process.py` 显示生产结构统一采用
  MP、`symprec=0.1` 标准原胞；元素和分数坐标在旧生成步骤截断到 80。
- Q1 重建保留了旧格式；18,706 train / 2,313 valid / 2,287 test 共 23,306 条中仍有
  55 条原子截断。原处理表的 `nsites_prim` 为完整数量。
- 55 条源结构均位于本地 MP 增量 JSONL。重新标准化后，原胞数量、元素前缀与晶格参数
  全部匹配；分数坐标前缀模整数平移后最大差约 2.22e-16，部分 0/1 边界表达不同。
  转换保存既有 FP32 前缀，追加完整尾部，不用另一个来源重新选择原胞。
- 旧结构坐标/谱输入已为 FP32；历史磁盘 nvalence 是 FP64，训练原本就读取为 FP32。
  新磁盘浮点数组统一为 FP32，FP64 留作可选几何数值参考。

## 数据结果及必要的关联修复

新目录 `data/crystal_q1_v1/` 保存原始 Q1 全部 ID、划分和行序，使用 packed 原子数组、
显式行晶格矩阵和 offsets；无晶格哨兵、磁盘原子填充或 N 上限。

| split | 存储结构数 | 完整原子数 | 最大 N | 结构修复 | 实际视图结构数 |
|---|---:|---:|---:|---:|---:|
| train | 18,706 | 220,785 | 192 | 45 | 18,704 |
| valid | 2,313 | 27,533 | 136 | 6 | 2,313 |
| test | 2,287 | 26,731 | 144 | 4 | 2,287 |

合计 275,049 个原子，补回 1,462 个；仅 cell+coords 的 FP32 载荷约 3.948 MiB。
完整字段、dtype、来源 SHA-256、文件 SHA-256 和逐条修复均在新 manifest 登记。
测试谱数组的读取仅用于复制/值一致性验证，没有用于指标评估、阈值选择或模型选型。

组成核查发现 54 条回退价电子数仅统计了前 80 个原子；对应 `method` 为已有回退分支，
且旧值严格等于截断组成的元素参考表之和。新值按完整组成修复，材料自身谱审计值保留。
这不是单纯保存 dtype：H1 的 gamma 尺度目标必须对应完整晶胞的 N_val，phDOS 的 eta
目标也必须使用完整 N。

沿用原 0.5 电子谱完整率隔离规则，两条 train 记录新比值低于阈值：

| ID | N | N_val 旧→新 | 修正完整率 |
|---|---:|---|---:|
| mp-aaabidtd | 180 | 360→960 | 0.207431 |
| mp-aaabilxy | 192 | 304→896 | 0.172566 |

新训练视图按 manifest 过滤两条，仍保存它们的完整结构和谱供追溯。历史参考表及旧 Q1 未改；
新训练视图最大 N 为 160，192 原子记录仍作为结构完整性与容量验收例，无训练标签参与。
修复版必须单独报告，不自动构成历史 ZP100/B7 的同数据对照。

第一次建立的完整缓存保留在被忽略的 `data/crystal_q1_v1_initial_20261004/`；它是本轮加入
质量视图元数据前的生成版本，不是默认训练输入。最终默认缓存重新从原 Q1 生成。

## 变化地图与公共接口

| 调用链 | 文件/主要符号 | 职责 |
|---|---|---|
| 来源→完整缓存 | `tools/data/migrate_crystal_q1.py`：`migrate_q1`、`restore_structures`、`restore_atom_fields` | 验证 Q1、恢复原子、修复组成回退、登记质量视图；拒绝已有输出目录 |
| 结构与 CIF 边界 | `utils/crystal.py`：`CrystalGeometry`、`validate_crystal_geometry`、`structure_to_crystal_inputs`、`atom_padding_mask` | cell/coords 显式对象、同 dtype/device/掩码校验、完整 CIF 原胞转换、旧哨兵边界 |
| 缓存→数据样本 | `utils/crystal_cache.py`：`PackedCrystalStore`、`crystal_data_contract`；`utils/dataset.py`：`Dos_Dataset`、`_select_rows` | shape/dtype/offset/ID 验证，train 质量过滤，所有标签/统计/掩码同步选行 |
| 样本→batch | `utils/bucket_batch.py`：`crystal_collate`、`split_crystal_batch`、`automatic_pair_budget`、`iter_computation_batches`；`utils/builder.py`：`get_dataloader` | 变长填充、保留顺序的计算拆分、完整单结构预算、评估资源控制 |
| 几何→模型 | `utils/relative_features.py`、`utils/periodic_neighbors.py`、`utils/periodic_geometry.py`；`Transformer.forward`、`basemodel.train_one_step` | 直接消费矩阵、完整原子 ZP/邻居/3N、样本加权梯度累积、一次 optimizer 更新 |
| 训练与恢复边界 | `tools/train.py`、`utils/experiment_config.py`、`tools/training-defaults.yaml`、`utils/ablation_checkpoint.py` | 默认新缓存；FP32 SumNorm/H1；manifest/归一化契约绑定；拒绝跨版本恢复/初始化 |
| 验收 | `tests/data/test_crystal_input.py`、`tools/checks/check_crystal_input.py` | 完整结构、梯度、尺度计数、CIF 原胞/帧、存储一致性、周期记录、GPU 前向/反向成本 |

`CrystalGeometry` 的 cell/coords 保留原生矩阵与任意合法反射；原生几何不会转换回六个参数。
历史 cache→matrix 与默认 CIF 的规范朝向只在导入边界转换一次。`primitive=False` 则保留
原始 CIF/Structure 的矩阵和晶胞。FP32、FP64 数值参考均可使用独立物理记录。
旧 Tensor 公共入口和 B7 CIF 入口保留原语义；旧 Encoder 的单镜像近似与方向打分没有改成
新的局部架构，不能由数据改造推断全模型 E(3) 不变。

样本 batch 组成保持原样。默认原子对预算 `max(B*6400,max_N²)`，只拆计算，不切原子；
每段损失乘 `B_sub/B_total` 累积，最后只裁剪梯度/更新一次。Dropout 与浮点归约仍可能改变
随机轨迹，不能称为逐步更新完全等价。明确过小的手工预算报错，不偷偷截断。

## 验证与实际资源

运行新格式专门测试：存储 23,306 条的 ID、offset/count、dtype 和原谱值一致性；全部 55 条
修复结构在原生 6.0 Å 记录中保留所有原子，无空节点且严格 `d<R`；192 原子进入 ZP 与梯度；
完整 batch 与按样本数加权拆分后的梯度一致；H1 使用完整 N；矩阵反射保持记录与向量变换；
CIF 无 80 上限；默认标准原胞/规范朝向与新缓存一致；checkpoint 拒绝跨数据版本。

GPU 命令（前向/反向、无参数更新）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 tools/checks/check_crystal_input.py \
  --out results/reports/crystal_input_q1_20261004/summary.json
```

完整规模 ZP：70,887,748 参数、FP32、V100-SXM2-32GB、每例预热 2 次后短测 3 次。
同一模型分别接收旧格式和新格式，参数量不变，结束检查 state_dict 权重与 buffer 均未改变。
结果为 [`summary.json`](../../results/reports/crystal_input_q1_20261004/summary.json)：

| 结构 batch | 计算子批 | 前向+反向中位 ms | 分配峰值 MiB |
|---|---|---:|---:|
| 历史固定填充 80，B=32 | 32 | 333.61 | 6553.91 |
| 31 个 8 原子结构 + 1 个完整 192 原子结构 | 31+1 | 247.18 | 2852.86 |
| 压力例：32 个完整 192 原子结构 | 5+5+5+5+5+5+2 | 1312.59 | 5398.77 |

这些是固定例的前向/反向检查，损失用于有限性与资源测试，不是正式 DOS 训练；峰值不包含
AdamW 状态或 CUDA 上下文，不能直接当成完整训练显存，也不能换算成整轮吞吐提升。
少量大结构可显著增加单批计算时间，资源拆分约束了所测峰值，没有验证任意 N 均适合硬件。

仓库验证命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -t .
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s archive/code/tests -t .
OMP_NUM_THREADS=1 bash tools/checks/check-static.sh
```

原始 GPU 摘要保留在被忽略的 `summary_initial.json`；增加填充坐标防护后复测，正式结果
使用上表和 `summary.json`，源码指纹与当前 Transformer/checker 一致。

验收收尾：当前分类全集 120 项、历史分类全集 251 项均通过；静态检查包含 240 项合同测试、
Ruff、内存语法编译和工作区文件边界，全部通过。新格式/恢复/组批相关收尾 32 项也通过；
这些测试存在重叠，不能将数量相加作为独立测试总数。文档相对引用均存在，`git diff --check`
通过。新 manifest 登记的 25 个原 Q1 来源文件的 SHA-256 与现文件一致；未覆盖历史数组。
实际通过默认 template/CLI 建立 loader：18,704 个 train 样本，首批元素 `[32,40]`、cell
`[32,3,3]`，使用 `CrystalGeometry`、完整原子和正确掩码，没有晶格哨兵。

初次检查暴露的旧源码正则断言、默认
数据路径断言和历史 pair 配方 fixture 已按实际新契约修正；没有放宽运行时科学条件。
初始 fixture 仅用所选谱输出损失时观察到零几何梯度，不能由此判断原子未进入模型；
后续以包含 H1 的完整损失检查第 81 个及之后原子状态确实收到梯度，没有修改模型架构。

## 未验收边界

正式训练、优化器完整显存、整轮吞吐、DOS 精度收益、完整 Encoder 的 E(3) 不变性和独立
A/B+ 的完整接入没有执行。新格式的 CIF 转换已实现，但没有获准的已训练新版本 checkpoint；
旧 B7 CIF 推理保持冻结协议。历史半径/模型报告仍属于旧 Q1，不能自动当成完整结构修复版证据。
