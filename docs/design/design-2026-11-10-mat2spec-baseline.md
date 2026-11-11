# 设计：Mat2Spec 外部基线（在本项目数据划分上训练与评估）

创建：2026-11-10。状态：已完成（用户 2026-11-10 确认选项 a，主四臂已运行；其余选项未做）；对应[当前问题](../status.md)。
结果见[运行日志](../logs/log-2026-11-10-mat2spec-baseline-run.md)，以下事先协议保留原意。移植来源与逐项改动见 [baselines/mat2spec/README.md](../../baselines/mat2spec/README.md)。

## 问题与判据

- **问题**：项目在本数据划分上没有任何外部模型的数字。Mat2Spec（Kong et al., Nat. Commun. 2022；MIT 许可）是已调研候选中唯一许可清楚、
  eDOS 与 phDOS 都覆盖的模型，也是 ARPAT 论文的对比对象。本方案把它移植到当前环境，在冻结的 train/valid 上训练，给出与 h2 可比的数字。
  它是参照，不是候选：不涉及采用决定，也不改变工作点。
- **Mat2Spec 是什么（核实自上游代码）**：
  - 输入：每个原子取 CGCNN 的 92 维元素嵌入；每个原子连 8 Å 内最近的 12 个周期近邻，边特征是距离的 41 维高斯展开；另有 103 维元素比例向量只用于读出时的注意力加权。
  - 编码器：3 层 4 头图注意力（GATGNN 风格，128 维），按原子加权后取均值得到 128 维晶体表示。因为是均值读出，模型看不到原子数。
  - 输出头：把晶体表示和“标签的每个箱”各自编码成高斯混合潜变量，二者共用解码器；训练时除预测损失外还有两路潜变量的 KL 对齐（权重 1.1）和对比损失（权重 0.1）。推理只走结构这一路，不需要标签（有测试）。
  - 一个模型只预测一种谱；eDOS、phDOS 各自单独训练。参数量 284 万（h2 为 4,776 万）。
- **它预测什么、损失是什么**（上游三种设置，由 `--label_scaling` 选择）：

  | 设置 | 训练目标 | 损失 | 能给出什么 | 上游用途 |
  |---|---|---|---|---|
  | SumNorm-KL（或 WD） | 每条谱除以箱和的形状 | 输出 softmax 后的 KL（或 Wasserstein） | 只有形状，没有总量 | eDOS 最好结果（论文表 2）；phDOS 也报告 |
  | Standard-MAE | 逐箱 z-score（减均值除标准差）后的绝对谱 | MAE | 绝对谱（形状＋总量），负值截为 0 | eDOS 主设置之一；并为 SumNorm 预测提供总量 |
  | MaxNorm-MSE | 每条谱除以最大值 | MSE | 只有形状 | phDOS 与 E3NN 对比的设置 |

  上游得到绝对量纲 eDOS 的做法（README 与论文）：SumNorm-KL 的形状 × Standard-MAE 预测的箱和。本方案称之为 **hybrid**（混合重建）。
- **对照与臂**：参照为 h2（`results/runs/phdos_head_20261106/analysis/exports/h2_s42_*`，完整 valid 2,313，blind）。主配方四个臂：
  `edos_sumkl`、`edos_stdmae`、`phdos_sumkl`、`phdos_stdmae`（每种谱一个形状模型、一个绝对谱模型）。
- **口径对齐**（评分直接调用项目的 `normalized_spectral_error` 与 `per_sample_spectral_metrics`，冻结常数取自 `results/runs/phdos_head_20261106/protocol.json`）：

  | 口径 | 定义 | 哪些臂有 | 与 h2 是否可比 |
  |---|---|---|---|
  | 谱形 | 预测与标签各自 SumNorm 后的 KL、W1（累积分布差的箱均值）、TV（总变差）；只在标签非零的材料上算（phDOS valid 有 1 条全零） | 全部臂 | 可比（h2 用其导出的 `pred_shape`） |
  | oracle E／R² | 预测形状 × 标签箱和，再算 E 与 R²（即项目已有的 oracle 诊断） | 全部臂 | 可比 |
  | blind E／R² | 完全由结构得到的谱 | `stdmae`：每原子预测 × N；hybrid：`sumkl` 形状 × `stdmae` 总量 | 可比，这是与工作点主指标对应的数 |
  | 联合 blind mean E | 0.5·eDOS＋0.5·phDOS | 两种谱的 blind 都有时 | 可比（注意下面的不对等） |

  SumNorm 臂单独没有 blind：它不预测总量，上游也不单独给。不另外拟合“每原子平均总量”一类的常数来凑 blind，那不是 Mat2Spec 的做法。
- **不可比或不对等之处（汇报时须一并说明）**：
  1. Mat2Spec 是单任务，每种谱各有模型（blind 的 hybrid 每种谱用两个），h2 是一个联合模型；参数量 284 万×臂数 对 4,776 万。
  2. 预算不同：Mat2Spec 按上游 200 轮，h2 为 35 轮。两者都在同一 valid 上选 checkpoint，valid 数字对双方都偏乐观；Mat2Spec 按任务各自选最好轮次、且不受双任务保护约束，略占便宜。
  3. Mat2Spec 的图只取 8 Å 内最近 12 个近邻；h2 用 6 Å 内全部近邻。这是模型自身的设计，不改。
  4. 单 seed（上游 seed 1）。Mat2Spec 自身的 seed 波动未知，h2 的三 seed 极差 0.0013 不能直接套用；预期差距若远大于 0.004 则不影响结论，否则须如实说明分不开。
  5. 与 Mat2Spec 论文的数字（eDOS R² 0.53–0.57、phDOS R² 0.57–0.63）和 ARPAT 论文中的 Mat2Spec 数字都不可比：数据、网格、划分、量纲均不同。论文数字只用来判断移植结果是否离谱。
  6. 标签单位取每原子谱是本移植的适配（上游用整胞谱），偏向有利于基线；Standard 的统计量只用 train 拟合（上游含 valid/test）。
- **判读（事先说明）**：
  - 每种谱报告 blind E（`stdmae` 与 hybrid 两个数都列，另以 valid 上较好者作为“Mat2Spec blind”并注明是在 valid 上挑的）、oracle E、中位 R²、失败率、谱形指标，以及相对 h2 的材料配对差与 95% bootstrap 区间。
  - h2 在两种谱上都更好且配对区间不含 0 → 记为“在本数据划分上优于 Mat2Spec”，可写入对外比较。
  - Mat2Spec 在任一种谱上持平或更好 → 是重要信号（小得多的单任务模型不输），先复核移植与口径，再与用户讨论含义；不自动触发任何实验。
  - 合理性检查：训练损失应下降；valid E 应明显低于 1（全零预测在 train 上的 E 为 1）；R² 量级应与论文同一档。明显离谱时先怀疑移植，不下结论。
- **数据与契约**：`crystal-q1-v1`（manifest `8e1b0adf…`），有效 train 18,704／完整 valid 2,313；加载代码只接受 `train`、`valid`，不读 test。
  选点只用 valid。checkpoint 与配置、数据、协议、源码指纹绑定，不与项目模型的 checkpoint 混用。

## 实现与兼容

- **移植**：在现有 `torch` 环境（PyTorch 2.2.1、PyG 2.6.1）直接运行，未新建环境。对上游模型文件共 7 处改动，其中 2 处是为恢复 PyG 1.5 语义（softmax 的参数位置、聚合维度），其余是删除未用导入；完整差异在 `baselines/mat2spec/port/mat2spec_model.diff`。上游训练脚本与数据加载未直接复用，由 `adapter/` 按其逻辑重写。
- **调用链**：`adapter/run_queue.py`（排队）→ `adapter/train.py`（单臂训练，逐轮完整 valid 评分）→ `adapter/data.py`（建图与批处理）＋`port/mat2spec_model.py`（模型与 `compute_loss`）；
  训练后 `adapter/evaluate.py export`（导出原谱、逐材料表、摘要）→ `evaluate.py report`（与 h2 配对比较）。评分在 `adapter/project_metrics.py`。
- **必要适配（不属于调参）**：输出维度 128／64；Standard 统计量只用 train；全零谱的归一化加 `1e-8`（否则 NaN）；标签单位取每原子谱。逐条理由见移植说明。
- **训练配方**：

  | 项 | 上游公开脚本 | 本方案主配方 | 说明 |
  |---|---|---|---|
  | 优化器／学习率／权重衰减 | AdamW／1e-3／1e-2 | 相同 | |
  | 学习率调度 | 每 1/4 总轮数减半（按迭代步进） | 相同 | |
  | 轮数 | 200 | 相同 | |
  | batch | eDOS 256（训练脚本默认）；phDOS 8 | 两种谱都 256 | 上游 phDOS 训练集只有 1,220 条，batch 8 对应约 3 万次更新；本数据 18,704 条，规模接近上游 eDOS（论文预印本称约 3 万条材料按 80/10/10 划分；该数字与论文中的 batch 128 来自网页摘要，未逐字核对，配方以公开脚本为准） |
  | 损失权重、dropout、K、网络宽度 | 1／1.1／0.1、0.5、10、3 层 4 头 128 维 | 相同 | 以公开代码为准 |
  | seed | 1 | 相同 | |
  | 选点 | valid 原量纲 MAE 最小 | 同时保存上游判据与项目判据（valid E 最小）两种 best；主报告用项目判据 | 用项目判据选点对基线有利 |

  是否需要为本数据调整：除 phDOS 的 batch 外，没有证据表明必须调整。可选的敏感性与调参见“需用户决定”。
- **开关与默认**：`train.py --task {edos,phdos} --label_scaling {normalized_sum,standardized,normalized_max} --loss {KL,WD,MAE,MSE}`；
  `--target_unit per_atom`（默认）／`cell`；`--limit_train/--limit_valid` 仅供预检，带这些参数的运行不能正式导出。
- **产物位置**：`results/runs/mat2spec_baseline_20261110/`，其下 `graphs/`（近邻缓存，带数据指纹）、`<臂>/`（`fingerprint.json`、`history.csv`、`latest.pth`、`best_project.pth`、`best_upstream.pth`、`execution/{status.json,train.log}`、`exports/`）、
  `execution/queue_status.json`、`analysis/report_best_project.{json,md}`、`preflight/`（本次预检）。已有目录拒绝覆盖，恢复须同一指纹。
- **回滚**：全部是新增文件（`baselines/`、`tests/baselines/`、本文档、`results/runs/mat2spec_baseline_20261110/`），未改现有训练代码、数据或文档；删除这些路径即回到之前状态。

## 验证、成本与执行

- **测试**：`python -m unittest tests.baselines.test_mat2spec_port`，12 项通过（2026-11-10）。覆盖：移植差异与记录一致、上游文件哈希；
  图注意力层与显式公式一致；推理输出不依赖标签；拒绝 test；批处理与逐晶体拼接一致；近邻与上游原语句一致（含全部触发填充的稀疏晶体）；
  有效视图 18,704／2,313；全零谱归一化有限；E 的定义；**重算 h2 导出与其存档摘要一致**；短训练的恢复与契约拒绝。Ruff 与仓库文件边界检查通过。
- **评估自检**：`evaluate.py selfcheck` 对 h2 重算，联合 blind E 0.269768（存档 0.269768），两任务 blind 与 oracle 的 E 均值／中位／P90、中位 R²、P10、MAE 均值差异均 ≤ 3e-7，失败率完全一致。
- **预检**（V100，2026-11-10；原始记录 `results/runs/mat2spec_baseline_20261110/preflight/report.json`）：
  - 六种设置各 8 轮（2,048 条 train、完整 valid）全部跑通，损失有限；导出与报告流程在这些短跑上端到端跑通。预检数字不是基线结果。
  - 全量 train、batch 256、各 4 轮：eDOS 5.2 秒/轮、峰值显存 4.4 GB；phDOS 4.0 秒/轮、3.0 GB。
  - phDOS batch 8：15.7 毫秒/步、显存 1.4 GB，全量每轮 2,338 步约 37 秒。
  - 建图：train＋valid 约 1 分钟（CPU），缓存 48 MB。
  - 中途 `kill -9` 后从 `latest.pth` 恢复、历史连续；排队脚本的顺序执行、跳过已完成、配方不符即停已验证。
- **预算**（200 轮）：

  | 内容 | GPU 时间 |
  |---|---:|
  | 主配方四臂（eDOS 两臂各约 17 分钟，phDOS 两臂各约 13 分钟） | 约 1.0 小时 |
  | 可选 A：phDOS 两臂按上游 batch 8 | 约 4.1 小时 |
  | 可选 B：phDOS MaxNorm-MSE、eDOS SumNorm-WD 各一臂 | 约 0.5 小时 |
  | 可选 C：学习率三档（5e-4／1e-3／2e-3）× 四臂 | 另加约 2.0 小时 |
  | 导出与报告 | 数分钟 |

- **中断、恢复与异常停止**：每轮写 `latest.pth`（含优化器、调度器、随机数状态与历史）；被外部中断后用同一命令重启排队脚本即从 latest 恢复，指纹不符则拒绝。
  非有限损失或预测（退出码 3）、显存不足（4）、契约不符（5）时该臂停止，排队脚本随即停止，不重试、不改参数。
- **长时任务**：准备完成后由用户把执行提示词交给新 agent，按[流程](../workflow.md)持续监测到导出、核验与汇报。

## 需用户决定

1. **配方范围**：(a) 只跑主配方四臂；(b) 主配方＋可选 A（phDOS 的上游 batch 8，回应“改了它的 batch”这一质疑）；(c) 再加可选 C 的小范围调参。
   调参越多，基线越强、对外比较越站得住，但 valid 被用得越多、报告越复杂。
2. **标签单位**：每原子谱（默认，偏向有利于基线）或再补一组上游字面的整胞谱 Standard 臂（另加约 0.5 小时）。
3. **seed**：单 seed（默认，与项目单 seed 流程一致）或三 seed（时间 ×3）。
4. **提交**：`baselines/` 含 MIT 许可的上游代码副本和被 `.gitignore` 忽略的 `atom_init.json`；是否提交、如何处理忽略规则由用户定。本次未提交。

两种谱的训练先后不影响成本或结论（四臂合计约 1 小时、互不依赖），默认顺序为 eDOS 两臂在前。
