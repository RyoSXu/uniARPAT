# 日志：2026-11-10 — Mat2Spec 外部基线运行

对应设计：[design-2026-11-10-mat2spec-baseline.md](../design/design-2026-11-10-mat2spec-baseline.md)。记录日期：2026-11-10；状态：已完成（选项 a，主四臂）。

## 范围与身份

- **问题**：在本项目冻结的 train/valid 上给出外部模型 Mat2Spec 的数字，与工作点 h2 比较。Mat2Spec 是参照，不是候选，不涉及采用决定。
- **臂**：`edos_sumkl`、`edos_stdmae`、`phdos_sumkl`、`phdos_stdmae`（每种谱一个只预测形状的 SumNorm-KL 模型、一个预测绝对谱的 Standard-MAE 模型）。
  用户 2026-11-10 确认只跑选项 (a)；未跑上游 phDOS batch 8、学习率调参、整胞标签和更多 seed。
- **数据与口径**：`crystal-q1-v1`（manifest `8e1b0adff516…`），有效 train 18,704，完整 valid 2,313；未读取 test。
  评分常数取自 `results/runs/phdos_head_20261106/protocol.json`（`0f2f667609ce…`）。blind 为完全由结构得到的谱；oracle 为预测形状 × 标签箱和，只作谱形诊断。
- **配方**：seed 1、200 轮、batch 256、lr 1e-3（每 50 轮减半）、AdamW weight decay 1e-2、标签单位每原子谱，其余为上游默认。参数量每个模型 284 万。
- **选点**：每臂按任务各自在 valid 上选点，保存两种 best；主表用项目判据（valid E 最小：Standard 臂用 blind E，SumNorm 臂用 oracle E），上游判据（valid 原量纲 MAE 最小）作对照。不受项目的双任务保护约束。
- **命令**（在项目根目录，`PY=/opt/tefscloud/miniconda3/envs/torch/bin/python`，`R=results/runs/mat2spec_baseline_20261110`）：

  ```bash
  setsid nohup $PY baselines/mat2spec/adapter/run_queue.py --root $R --arms edos_sumkl,edos_stdmae,phdos_sumkl,phdos_stdmae --epochs 200 --batch_size 256 > $R/execution/queue.out 2>&1 &
  $PY baselines/mat2spec/adapter/evaluate.py export --run_dir $R/<臂> --checkpoint {best_project,best_upstream}
  $PY baselines/mat2spec/adapter/evaluate.py report --root $R --checkpoint {best_project,best_upstream}
  ```

- **指纹**：源码 SHA-256 前缀 `mat2spec_model.py e198d4603427`、`pytorch_stats_loss.py 5d3a2f761d6f`、`data.py c4e9eedf2c01`、`train.py 7a03a756dd7e`、`project_metrics.py 48d976bd5d81`；
  PyTorch 2.2.1、PyG 2.6.1、pymatgen 2025.5.2。完整指纹在各臂 `fingerprint.json`。

## 执行与证据

- **启动前检查**：GPU 空闲、无残留队列进程、四臂目录不存在；`tests.baselines.test_mat2spec_port` 12 项通过；`evaluate.py selfcheck` 31 项一致（重算 h2 联合 blind E 0.269768）。
- **训练**：队列 15:37 启动、16:45 全部完成，无中断、无恢复、无非有限值、无显存不足。每臂 200 轮完整，训练损失全程下降。

  | 臂 | 耗时 | 每轮 | 峰值显存 | best 轮次（项目／上游判据） | 训练损失 首→末 |
  |---|---:|---:|---:|---|---|
  | edos_sumkl | 17.6 分钟 | 5.2 秒 | 4.6 GB | 86／186 | 1.51 → 0.78 |
  | edos_stdmae | 17.6 分钟 | 5.2 秒 | 4.6 GB | 197／199 | 1.55 → 1.02 |
  | phdos_sumkl | 13.6 分钟 | 4.0 秒 | 3.2 GB | 54／102 | 1.93 → 0.70 |
  | phdos_stdmae | 13.6 分钟 | 4.0 秒 | 3.2 GB | 131／154 | 1.30 → 0.82 |

  两个 SumNorm-KL 臂的 valid E 在早期（第 86／54 轮）最低，之后回升（末轮 0.392／0.234），而训练损失继续下降；两个 Standard-MAE 臂的 valid E 到第 150 轮后基本走平。
- **产物**（本地，不随 Git）：`$R/<臂>/{history.csv,latest.pth,best_project.pth,best_upstream.pth,fingerprint.json,execution/,exports/}`；
  `$R/analysis/report_best_{project,upstream}.{json,md}`；`$R/execution/{queue_status.json,queue.out}`；预检在 `$R/preflight/`。
- **独立核验**（不使用 adapter 的评分代码，纯 numpy，标签与原子数直接读冻结的 valid 数组）：
  - 八个导出（四臂 × 两种 best）的 E 均值、中位 R²、失败数与各自摘要一致（差 ≤ 1e-7，失败数完全相同）；每个导出为 2,313 条且材料顺序与 valid 索引及 h2 导出一致；导出中的标签与数据数组一致。
  - 每个 checkpoint 的轮次等于 `history.csv` 中对应判据的最小值所在轮，历史为连续 200 轮。
  - 重算 h2 blind E（0.332315／0.207222）、配对均值差与 hybrid 的 E，与报告一致；另用不同随机种子重做联合差的 bootstrap，区间 [+0.0101, +0.0224]。
  - 标签泄漏检查：对训练好的 `edos_stdmae`、`phdos_stdmae`、`phdos_sumkl`，把送入标签分支的标签全部置零后重新预测，结果与导出只差 GPU 浮点噪声（相对 ≤ 4e-7）。
- **主表**（完整 valid 2,313，项目判据选点；Mat2Spec 为单 seed）：

  eDOS

  | 模型 | 口径 | E 均值 | E 中位 | E P90 | 中位 R² | R² P10 | 失败数（率） | 相对 h2 的 E 配对差 [95% 区间] |
  |---|---|---:|---:|---:|---:|---:|---:|---|
  | h2 | blind | 0.3323 | 0.1618 | 0.6376 | 0.484 | 0.092 | 174（7.52%） | |
  | Mat2Spec Standard-MAE | blind | 0.3555 | 0.1605 | 0.7056 | 0.479 | 0.072 | 167（7.22%） | +0.0232 [+0.0122, +0.0347] |
  | Mat2Spec hybrid | blind | 0.3729 | 0.1737 | 0.6923 | 0.445 | 0.085 | 147（6.36%） | +0.0406 [+0.0240, +0.0599] |
  | h2 | oracle | 0.3186 | 0.1528 | 0.6064 | 0.508 | 0.156 | 128（5.53%） | |
  | Mat2Spec SumNorm-KL | oracle | 0.3594 | 0.1663 | 0.6797 | 0.474 | 0.109 | 127（5.49%） | +0.0409 [+0.0242, +0.0591] |
  | Mat2Spec Standard-MAE | oracle | 0.3477 | 0.1582 | 0.7209 | 0.498 | 0.058 | 175（7.57%） | +0.0291 [+0.0169, +0.0418] |

  phDOS

  | 模型 | 口径 | E 均值 | E 中位 | E P90 | 中位 R² | R² P10 | 失败数（率） | 相对 h2 的 E 配对差 [95% 区间] |
  |---|---|---:|---:|---:|---:|---:|---:|---|
  | h2 | blind | 0.2072 | 0.1304 | 0.4334 | 0.746 | 0.345 | 61（2.64%） | |
  | Mat2Spec Standard-MAE | blind | 0.2165 | 0.1348 | 0.4315 | 0.730 | 0.313 | 57（2.46%） | +0.0092 [+0.0037, +0.0147] |
  | Mat2Spec hybrid | blind | 0.2180 | 0.1395 | 0.4137 | 0.721 | 0.343 | 40（1.73%） | +0.0108 [+0.0045, +0.0171] |
  | h2 | oracle | 0.2012 | 0.1263 | 0.4206 | 0.754 | 0.377 | 43（1.86%） | |
  | Mat2Spec SumNorm-KL | oracle | 0.2053 | 0.1291 | 0.3969 | 0.738 | 0.375 | 40（1.73%） | +0.0041 [−0.0023, +0.0101] |
  | Mat2Spec Standard-MAE | oracle | 0.2181 | 0.1351 | 0.4535 | 0.736 | 0.265 | 93（4.02%） | +0.0170 [+0.0110, +0.0229] |

  谱形指标（SumNorm 形状，标签非零的材料；KL／W1／TV 均值）：eDOS h2 0.282／0.0266／0.247，Mat2Spec SumNorm-KL 0.308／0.0300／0.258；
  phDOS h2 0.296／0.0197／0.241，Mat2Spec SumNorm-KL 0.291／0.0188／0.244。Standard-MAE 臂因负值截零使形状含精确零，KL 偏大（0.588／0.708），不宜用 KL 评价该臂。

  联合 blind mean E（0.5 eDOS＋0.5 phDOS）：h2 0.2698；Mat2Spec Standard-MAE 0.2860，差 +0.0162 [+0.0100, +0.0226]；hybrid 0.2955，差 +0.0257 [+0.0167, +0.0358]。
  两种谱上 blind 较好的都是 Standard-MAE 直接预测（在 valid 上比较得出），上游的 hybrid 做法在本数据上更差。
- **上游判据选点**（对照）：eDOS Standard-MAE blind 0.3576、phDOS 0.2176、联合 0.2876（差 +0.0179 [+0.0115, +0.0243]）；SumNorm-KL oracle 为 0.3800／0.2107。结论方向不变。
- **配对差的范围**：抽样单位是 valid 材料（percentile bootstrap，1 万次），条件于已选 checkpoint；不含训练 seed 波动，也不消除在 valid 上选点的偏差。

## 结论与交接

- **事实**：
  1. blind 口径下 h2 在两种谱上都好于 Mat2Spec，配对区间都不含 0：eDOS 差 0.023（相对约 7%），phDOS 差 0.009（相对约 4.5%），联合差 0.016。按设计的事先判读，记为“在本数据划分上 blind 口径优于 Mat2Spec”。
  2. phDOS 的谱形（oracle）上两者分不开：差 +0.0041，区间含 0；Mat2Spec 的失败数、KL、W1、E P90 还略好，中位 R² 低 0.016。按事先判读属于“持平”，是需要讨论的信号。
  3. 中位 R² 两者接近（eDOS 0.479 对 0.484，phDOS 0.730 对 0.746）；差距主要体现在 E 均值和尾部（eDOS 的 E P90 0.706 对 0.638）。
  4. 结果不离谱：训练损失下降，valid E 远低于 1，R² 量级与论文同一档（论文的数据与网格不同，不可直接比）；已做泄漏与独立复算检查。
- **不对等之处**：Mat2Spec 是单任务、每种谱单独训练和选点，每个模型 284 万参数，训练 200 轮；h2 是 4,776 万参数的联合模型，35 轮，选点受双任务保护约束。
  项目自己的单任务参照（6 层头 LR2，[11-03 日志](log-2026-11-03-reference-controls-run.md)）为 eDOS 0.3381、phDOS 0.2044，与 Mat2Spec 的差是 0.017／0.012。
  标签取每原子谱、用项目的 E 选点，都偏向有利于 Mat2Spec。图只取 8 Å 内最近 12 个近邻是 Mat2Spec 自身设计。
- **推断**：差距的量级（联合 0.016）约为 MACE 冻结特征收益（0.008）的两倍，远小于数据量翻倍的收益（0.042）。
  一个 2022 年的小型单任务图网络与工作点只差这么多，与状态页“模型细节的收益已接近噪声、数据与预训练才是主要杠杆”的判断一致。
- **未证明**：Mat2Spec 只有一个 seed，其自身的 seed 波动未知；phDOS blind 的 0.009 约为 h2 该任务三 seed 极差（0.0033）的 3 倍，eDOS 的 0.023 远大于极差（0.0007），但 Mat2Spec 一侧的波动没有测。
  未检验上游 phDOS 的 batch 8、学习率调参和整胞标签；Mat2Spec 的最好结果可能还略高于本次数字。test 未评估。
- **同步**：本会话按约束未修改 `docs/status.md`、`docs/index.md`；建议由维护它们的会话补入结果和入口。未提交 Git，未删除任何 checkpoint。
- **用户决定（2026-11-10，经项目审核会话转达）**：暂不补 seed，也不做上游 phDOS batch 8 和学习率调参；本轮到此结束。
  理由是新数据集冻结后基线需要重跑，那时再测 seed 波动更有意义。现有代码、checkpoint 和产物保留，不提交 Git、不清理。
  因此上面“未证明”的各项保持未测状态；引用本次数字时须注明 Mat2Spec 为单 seed。
- **仍待决定**：是否把这组数字用于对外比较及其表述。
