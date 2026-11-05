# 设计：训练随机性、无结构参照与单任务对照

创建：2026-11-03。状态：**已确认（三项均按建议），7 组训练于 2026-11-04 完成并经独立核验**；对应[当前问题](../status.md)。
[实施与预检记录](../logs/log-2026-11-03-reference-controls-preparation.md)、[运行、核验与判读](../logs/log-2026-11-03-reference-controls-run.md)。以下事先协议保留原意。

## 问题与判据

### 为什么先做这一组

现有结论几乎都来自 seed42 单次训练，配对 bootstrap 只覆盖材料抽样。架构效应多在联合 mean E
差约 0.006 的量级（LR2 采用依据为 −0.00604，95% 区间上界 −0.00103），但训练随机性的幅度从未测量。
[已有模型的事后平均](../logs/log-2026-10-28-error-structure-ensemble.md)两份即降 0.0098，
说明单模型误差中可能有相当一部分是方差。另外两项基础问题也没有在当前配方下回答过：

- 结构到底贡献了多少？旧配方下只看组成的 O0 eDOS blind 中位 R² 0.4691，原 baseline 0.4729，
  差异未分辨（[旧骨架归因](../logs/log-2026-10-13-legacy-backbone-run.md)）；G0/O0 都不是当前 E＋S 与修正骨架。
- 联合训练是否优于单任务？“联合”是项目前提，但在新配方下单任务研究一直搁置。

这三项不追加任何架构候选，目的是给后续所有比较提供尺度，并为对外比较准备内部参照。

### 组别

共同配方即当前工作点：修复版 `crystal_q1_v1` 有效 train/valid 18,704/2,313，ZP、完整 A64 LR2 形式、
staged L2/G6 修正骨架、共享六层 decoder、每原子总量头、E＋S、35 轮、batch32、lr5e−5、FP32、
冻结 `results/runs/spectral_residual_20261030/protocol.json`（本地产物）的 train 常数，不评估 test。

| 组 | 唯一差异 | seed | 状态 |
|---|---|---|---|
| R42 | 无（已采用工作点 ep29） | 42 | 复用，不重训 |
| R43、R44 | 仅训练随机性 | 43、44 | 新增 |
| S0（3 个） | 不做局部更新、不读几何；六层修正全局骨架与其余模块不变 | 42、43、44 | 新增 |
| Te | 只监督 eDOS（phDOS 项权重为 0） | 42 | 新增 |
| Tp | 只监督 phDOS（eDOS 项权重为 0） | 42 | 新增 |

S0 是“只看组成与原子数”的参照：原子标量从 ZP 出发，只经过与 LR2 相同的六层全局注意力/FFN，
不构建周期记录。它的成本低，因此同样跑 3 个 seed，使“结构贡献”的比较两边都有随机性估计。
删除局部分支同时减少 525,184 个参数（实测 75,552,068 对 76,077,252），
这属于该分支净效应的一部分，不宣称容量匹配。

单任务组保留完整双任务架构及全部参数，只把另一任务的 E 与 S 系数设为 0，以避免把容量变化混入；
未监督任务的输出不报告。被监督任务的系数见“需要决定的事项”第 3 条。

### 选点

- R43/R44：与工作点完全相同，`balanced_blind_energy` 加固定 A64 ep31 双任务保护。
- S0：保护线是采用门槛而非参照模型的属性，S0 很可能整轮都不满足 196/99 失败数上限。
  因此 S0 以联合 mean E 最小选点、**不施加保护**，并在报告中列出它是否满足保护线。
- Te/Tp：以本任务 mean E 最小选点，只检查本任务的保护线并报告，不以其过滤。
- 所有组另从 history 报告“不加保护的联合/单任务 mean E 最小轮”，使联合模型与单任务模型
  在“按各自目标选点”的口径下也可比较；这一项只用 history 数值，不需要额外 checkpoint。

### 主分析与判读（事先固定）

指标：完整 valid、blind，两任务 mean/median/P90 E、中位/P10 R²、失败数，及联合 mean E。
区间为材料配对 bootstrap 2,000 次、seed20261103，所有组使用同一抽样索引矩阵。

1. **训练随机性。**报告 R42/R43/R44 联合 mean E 的均值、极差与样本标准差，以及逐任务同样统计。
   - 极差 ≥ 0.006（LR2−A64 采用差的幅度）：LR2 相对 A64 的改进视为**未在跨 seed 意义下确立**，
     提出 A64 E＋S 的 43/44 补跑（约 9 GPU 小时）作为第二阶段，由用户决定；不自动执行。
   - 极差 < 0.006：保留采用决定，并在后续候选的判据中加入“点差须超过同配方 seed 极差”的要求，
     须用户确认后写入 [decisions.md](../decisions.md)。
2. **同配方集成。**对三份 R 的 blind 物理谱逐点平均后计算 E 与 R²；报告集成相对三份单模型均值、
   相对最好单模型的差值与区间。它回答集成收益有多少来自同配方方差，区别于 10-28 的跨配方平均。
3. **结构贡献。**比较 S0 与 R 两组：逐材料先对各组三个 seed 的 E 求平均，再做配对 bootstrap；
   同时列出 3×3 两两差值。某任务上“结构有可分辨贡献”须同时满足：配对区间不含 0，
   且三个 S0 seed 都比三个 R seed 差。只满足一项则写为方向性证据。
   同组成谱差单列：S0 对同组成材料预测谱形相同，其谱差误差即严格零响应参照。
4. **联合 vs 单任务。**Te 与三份 R 比较 eDOS，Tp 与三份 R 比较 phDOS；与 R 的 seed 均值做配对区间，
   并对照 R 的 seed 极差。单任务只有一个 seed，差值落在 R 的 seed 极差内时写为“未分辨”，
   不论区间是否含 0；超出极差且区间不含 0 才写为联合有利或不利。
   两种选点口径（各自目标、加保护）都报告，结论以“各自目标”口径为主。

所有结论限修复版 Q1、35 轮及本配方；不读 test，不因结果自动追加 seed、预算或候选。

### 需要决定的事项

1. **replicate 是否同时改变数据顺序。**现有训练入口固定 `DistributedSampler(seed=0)`，`--seed` 只改变初值
   和 dropout。建议新增显式 `data_order_seed`，R43/R44 及其他新组设为各自 seed，以覆盖完整训练随机性；
   seed42 组保持 0 以与 R42 一致。若只想测初值随机性，则全部保持 0。
2. **S0 的定义。**建议为零局部调用（上表）；另一选项是保留两次无几何逐原子局部分支（C0 型），
   参数量更接近但不再是“纯组成”模型。历史 C0−G0 的 12 项指标区间全部含零。
3. **单任务 loss 系数。**建议单任务用 `mean(E_t+S_t)`（等价于联合配方中该任务项的 2 倍），
   与联合配方中两项之和的总尺度相同；另一选项是保持 `0.5×mean(E_t+S_t)` 与联合配方逐项一致，
   但总 loss 约减半。AdamW 对整体缩放近似不敏感，差异主要来自梯度裁剪与权重衰减的相对强度
   （当前 `grad_clip` 未启用），预计影响小，但需事先固定。

## 实现与兼容

调用链：`tools/train.py` → `ExperimentConfig` → `validate_balance_config` → `initialize_balance_model`
→ `basemodel.train_one_step` → 选点 → `lg_valid_export.export`。只读核查（2026-11-03）结果：

- 用 seed42 重建未训练初值得到 `924bd18a…`，与 R42 `config_used.yaml` 记录一致；seed43 也能完整构造。
  现有阻断只来自配置校验：`legacy_balance_config` 进入的历史验证链要求 `seed=42`
  （`utils/local_structure_experiment.py::FROZEN`）。

已实施（2026-11-03，用户确认三项均按建议），全部默认关闭；采用的 R42 契约重建后与已保存版本逐键相同：

| 位置 | 改动 |
|---|---|
| `utils/experiment_config.py`、`tools/train.py` | 新增 `data_order_seed`（默认 None＝0）与 `single_task`（None/`edos`/`phdos`）；只在设置时写入训练记录与 cli |
| `utils/builder.py::get_sampler/get_dataloader` | 训练 sampler 使用 `sampler_seed`；None 时保持 0，非训练 split 拒绝 |
| `utils/spectral_balance_experiment.py` | `reference_control_kind`、`expected_selection_metric`、`validate_reference_control`；历史链按 seed42 校验结构配方，实际 seed 进入初值；契约新增 `reference_control`，S0 用 `joint-spectral-balance-structure-blind-v1` |
| `model/local_global_encoder.py`、`model/transformer.py` | 新路线 `S0`：零局部调用、不构建记录或几何缓存，六层沿用 post-norm、注意力概率 dropout 0 |
| `initialize_structure_blind_model` | 从同 seed 未训练 LR2 E＋S 复制全部非局部张量，核对恒等 QKVO |
| `model/model.py::train_one_step` | 单任务 loss 为 `mean(E_t+S_t)`，另一任务不参与 |
| `utils/spectral_balance.py`、`utils/checkpoint_selection.py` | 选点 `unguarded_blind_energy`（S0）、`edos_blind_energy`/`phdos_blind_energy`（单任务）：保护线逐轮记录、不过滤 |
| `tools/eval/lg_valid_export.py` | 核验 S0 路线与对照契约；非保护选点按其自身分数复核 |
| `tools/eval/reference_controls_prepare.py`、`reference_controls_analyze.py`、`tools/checks/check_reference_controls.py` | 冻结准备与命令、四项分析、逐组 GPU 预检 |

每组使用独立 tag（前缀 `refctl20261103`）与独立契约，拒绝与 LR2/A64 checkpoint 混用；
protocol 逐字复制工作点版本，train 常数不重拟合。

## 验证、成本与执行

- 测试：seed42 初值逐值复现 R42 记录；seed≠42 时初值改变而共享张量协议一致；`data_order_seed=None`
  的 epoch0/1 样本顺序与历史相同；S0 不构建记录、E(3)/原子重排输出不变；单任务另一任务梯度为零
  且 loss 等于设定系数下的评估误差；新旧选点契约互相拒绝；恢复后 RNG 与 LR 轨迹一致。
- 短时预检：每类组两批真实 train 更新（含最大 160 原子批）、一批 valid 完整评估、成本与显存。
- 估时（依据已完成运行）：R 组每组约 4.5 h（417 s/轮）；S0 每组约 1.3 h（参照 G0 115 s/轮）；
  Te/Tp 每组约 4.5 h。合计约 **22 GPU 小时**，单卡串行；顺序建议 S0×3 → R43 → R44 → Te → Tp。
  第二阶段 A64 补跑约 9 GPU 小时，仅在判读第 1 条触发且用户批准后进行。
- 异常：OOM、非有限值、契约不符即停，不改参数重跑；中断按原命令从 latest 恢复并记录。
- 长时任务：准备与预检完成后，由用户把执行提示词交给新 agent，按[流程](../workflow.md)每约 5 分钟监测，
  每组完成后汇报，负责到导出、独立复核与分析结束。
