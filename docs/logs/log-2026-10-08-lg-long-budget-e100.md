# 日志：2026-10-08 — 长预算对照（100 轮：原六层 baseline 与 B+ 分段 G=6）

创建日期：2026-10-08。状态：**pending**（两组 100 轮训练执行中，完成核验、逐样本导出与配对比较待收尾）。
本日志检验**较长训练预算能否缩小新 Encoder 与原六层 baseline 的性能差距**；
35 轮结果只作短预算参考；旧产物全部保留；未读取 test；未做 Git 提交。

## 范围

- 任务与假设：在 seed=42、100 轮预算下，测量 B+ 分段（局部 2 次＋全局 6 层）与原六层 Encoder＋ZP
  的差距是否比 35 轮时更小。前置事实（见[全局深度对照日志](log-2026-10-08-lg-global-depth-q1v1.md)）：
  35 轮预算下 G=6 只补回 eDOS R² 中位（分段 −0.0062/−0.0046，区间含 0），eDOS 失败率反而更高
  （+2.94*/+2.12* pp），phDOS R² 仍低于 baseline；且两组 best epoch 均为末轮 35、选点指标仍在下降，
  **35 轮预算可能截断 G=6 轨迹**。本轮直接检验这一截断假设。
- 两组（单卡顺序执行、每组独立进程）：

  | 组别 | 配置 | tag |
  |---|---|---|
  | 原六层 Encoder＋ZP | M1，`lg_baseline` 模式 | `_lg_baseline_q1v1_e100_s42` |
  | B+ 分段，局部 2 次＋全局 6 层 | M1，`local_route=B+`、`local_layout=staged`、`local_depth=2`、`global_depth=6` | `_lgd_s42g6e10020261008_k2_g6_bp_staged_s42` |

- 改动的文件或配置：

  | 文件 | 符号与职责 |
  |---|---|
  | [`train.py`](../../tools/train.py) | 每轮记录新增 `lr` 列（该轮实际生效学习率）；纯增量列，不改公式/契约/选点 |
  | [`check_long_budget.py`](../../tools/checks/check_long_budget.py) | 新：启动前核查（冻结配方、100 轮 LR 轨迹、初始化、契约、断点恢复、真实批次前向/反向）与 `--verify-completed` 完成核验 |

  科学配置零改动；模型结构、损失、optimizer、调度、数据契约、选点规则逐项沿用各自已完成 35 轮的
  冻结配方（修复版 crystal_q1_v1、ZP、SumNorm/H1、Q1 谱查询 `q1_coord`、原损失、decoder/head、
  既定提速策略）。几何仍只通过局部聚合进入，E(3) 不变性保持。

## 证据

- 命令（`results/runs/lg_long_budget_20261008/queue.sh`，`set -e` 前组异常即停）：

  ```bash
  PYTHONUTF8=1 python tools/train.py --epochs 100 --batch_size 32 --lr 5e-5 --seed 42 \
    --tag _lg_baseline_q1v1_e100_s42 --atom_feat z_only_proj --skip_test_eval \
    --selection_metric balanced_oracle_mae --lg_baseline --q1_coord
  PYTHONUTF8=1 python tools/train.py --epochs 100 --batch_size 32 --lr 5e-5 --seed 42 \
    --tag _lgd_s42g6e10020261008_k2_g6_bp_staged_s42 --atom_feat z_only_proj --skip_test_eval \
    --local_route B+ --local_layout staged --local_depth 2 --global_depth 6 \
    --selection_metric balanced_oracle_mae --q1_coord
  ```

  seed=42、100 轮、batch=32、lr=5e-5、FP32、`balanced_oracle_mae` 选点、仅 train/valid、
  无 AMP、不加载任何已训练权重（35 轮或历史权重均未加载）。

- 学习率：**100 轮余弦调度、warmup=5、min_lr=1e-6**（`build_warmup_cosine_scheduler` 默认值，已核对）。
  实测轨迹 lr[0]=1e-5 → lr[5]=5e-5 → lr[35]=3.89e-5 → lr[100]=1e-6。
  **35 轮与 100 轮轨迹不同**（lr[35] 在 35 轮轨迹上已到 1e-6），因此
  **100 轮实验的前 35 轮不作为旧实验的复现**，两轮结果分开报告。

- 启动前核查（`results/runs/lg_long_budget_20261008/preflight.json`，有限批次、非完整资源扫描）：

  | 核查项 | baseline | deep G=6 |
  |---|---|---|
  | 冻结配方（CLI 逐字段） | 仅 `epochs 35→100` 不同 | 仅 `epochs 35→100` 不同 |
  | 装配配置块 vs 35 轮 | **逐字节相同** | **逐字节相同** |
  | 16 个 transformer 配方键 | 全部相同 | 全部相同 |
  | 初值哈希 | 全部逐值复现 | 全部逐值复现 |
  | 契约 schema / `training.epochs` | `lg-baseline-model-v1` / 100 | `local-global-depth-model-v1` / 100 |
  | 与 35 轮契约差异 | 仅 `epochs` | 仅 `epochs` |
  | 35 轮契约被拒绝 | 是 | 是 |
  | 断点恢复权重/optimizer/RNG | 逐值相等 | 逐值相等 |
  | LR 快进 0/5/21/35/100 与轨迹 | 一致 | 一致 |
  | 真实批次前向/反向 loss | 2.89 / 2.52 | 2.97 / 2.57 |

  初值哈希（两组均与 35 轮记录一致，未加载已训练权重）：

  | 组 | 哈希 |
  |---|---|
  | baseline | `common 79c99638…`、`encoder 927adeaf…`、`state 7bb47f4f…`、`reference_common 8eea1e0c…` |
  | deep G=6 | `shared 9746b582…`、`G3–G6 944ad93d…`、`common 0e3f9c7b…`、`reference_common 8eea1e0c…`、`state dbdbbdf5…` |

  另核对两条命令行解析结果与核查所用配置逐字段相同。

- **执行异常与定位（编码）**：`tools/train.py` 读 `tools/training-defaults.yaml` 抛
  `UnicodeDecodeError('ascii')`，即[四组对照日志](log-2026-10-07-local-global-four-arm-s42.md)记录的那次。
  本轮定位到：进程默认文本编码会在 GPU 训练开始后的**不确定时点**由 UTF-8 翻转为 `ANSI_X3.4-1968`
  （实测在第 1~2 次 `train_one_step` 前后，重复实验时点不同）。已排除 DataLoader worker
  （`num_workers=0` 同样复现）与 `pin_memory`（开关均复现）；两次连续 `train_and_eval` 在无中间训练时均正常。
  这解释了四组对照"前三组成功、第四组失败"。**缓解**：`PYTHONUTF8=1`（PEP 540）使文本 I/O 固定为 UTF-8、
  不依赖 locale；实测 6 步训练后第二次读取仍正常。本轮两组训练与核查均以此启动，核查并强制要求 UTF-8 模式。
  未修改共享 runner 的读取代码（`encoding='utf-8'` 一行永久修复待用户决定）。原始现场见
  `results/runs/lg_long_budget_20261008/preflight.log`。

- 结果（valid 2,313 条逐样本配对，各自 best checkpoint 独立重算；中位数，fail 为 R²<0 占比 %；
  `balanced_oracle_mae` 选点；**范围限定 seed=42、100 轮**）：

  | 组 | best ep | sel | eDOS blind R²/MAE/fail | eDOS oracle R²/MAE/fail | phDOS blind R²/MAE/fail | phDOS oracle R²/MAE/fail |
  |---|---|---|---|---|---|---|
  | baseline（35 轮，短预算参考） | 34 | 0.6629 | 0.4729 / 1.3752 / 8.78 | 0.5099 / 1.3189 / 5.71 | 0.7272 / 0.00701 / 4.24 | 0.7352 / 0.00694 / 3.59 |
  | **baseline（100 轮）** | 92 | 0.6330 | 0.4775 / 1.3318 / 13.58 | 0.5207 / 1.2610 / 10.16 | 0.6998 / 0.00686 / 6.79 | 0.7088 / 0.00683 / 5.58 |
  | deep G=6（35 轮，短预算参考） | 35 | 0.6554 | 0.4667 / 1.3797 / 11.72 | 0.5053 / 1.3039 / 7.83 | 0.7144 / 0.00708 / 4.71 | 0.7258 / 0.00698 / 3.72 |
  | **deep G=6（100 轮）** | 83 | 0.6287 | 0.4488 / 1.3500 / 15.35 | 0.5076 / 1.2562 / 11.37 | 0.7071 / 0.00690 / 5.58 | 0.7159 / 0.00683 / 4.89 |

  数据口径：crystal_q1_v1 修复版（18,704 / 2,313 / 2,287），仅 train/valid，blind/oracle 拆分，
  未读取 test。deep G=6 的 best epoch 为 **83（非末轮）**，100 轮预算未再截断该臂；
  baseline 的 best epoch 为 92。

- 配对 bootstrap（2,313 材料逐 ID 配对，2,000 次重采样，seed 20261008；差值 = deep G=6 − baseline，
  95% 分位区间；`*` 表示区间不含 0）：

  | 差值 | 35 轮（短预算参考） | **100 轮（本轮）** | 差距走向 |
  |---|---|---|---|
  | eDOS blind R² 中位 | −0.0062 | **−0.0287*** | **扩大** |
  | eDOS oracle R² 中位 | −0.0046 | **−0.0131*** | **扩大** |
  | eDOS blind fail (pp) | +2.94* | +1.77* | 缩小但仍显著 |
  | eDOS oracle fail (pp) | +2.12* | +1.21* | 缩小但仍显著 |
  | phDOS blind R² 中位 | −0.0128* | +0.0073 | **闭合** |
  | phDOS oracle R² 中位 | −0.0094* | +0.0071 | **闭合** |
  | phDOS blind fail (pp) | +0.48 | **−1.21*** | 反超（deep 更好） |
  | phDOS oracle fail (pp) | +0.13 | −0.69 | 闭合 |
  | 全部 MAE 中位 | 区间含 0 | 区间含 0 | 均不显著 |

- 预算效应（同臂 100 轮 − 35 轮，配对 bootstrap）：

  | 差值 | deep G=6 | baseline |
  |---|---|---|
  | eDOS blind R² 中位 | **−0.0179*** | +0.0046 |
  | eDOS oracle R² 中位 | +0.0024 | +0.0108 |
  | eDOS blind/oracle fail (pp) | **+3.63* / +3.55*** | **+4.80* / +4.45*** |
  | phDOS blind/oracle R² 中位 | −0.0073 / −0.0100* | **−0.0274* / −0.0264*** |
  | phDOS blind/oracle fail (pp) | +0.86* / +1.17* | +2.55* / +1.99* |
  | eDOS blind MAE 中位 | −0.0296* | −0.0434* |

  **两臂的失败率与 phDOS R² 在长预算下都显著变差，MAE 都显著变好。**

- 逐轮轨迹（`trajectory_analysis.json`，观察，非采用判断）：**四条轨迹全部呈现
  "训练损失持续下降、验证指标停滞或回退"**，且 100 轮比 35 轮更严重：
  - deep G=6 100 轮：blind eDOS R² 第 32 轮见顶 0.4768，末轮 0.4448（−0.0320）；
    eDOS 失败率第 10 轮见底 7.91%，末轮 15.48%（+7.57 pp）；而 train loss 从 0.1294 降到 0.1211。
  - baseline 100 轮：blind eDOS R² 第 58 轮见顶 0.4839，末轮 0.4767；phDOS R² 第 30–35 轮见顶后降 0.030；
    eDOS 失败率第 17 轮见底 7.74%，末轮 13.75%（+6.01 pp）；train loss 从 0.1878 降到 0.1852。
  - **选点规则放大了这一背离**：`balanced_oracle_mae` 选出的 best epoch（baseline 92、deep 83）
    远离 R² 顶点（32–63）与失败率谷底（10–17）。

- 耗时与显存（CSV `epoch_time_s` 只计训练段；显存为 `peak_vram_mb` 分配口径）：

  | 组 | 训练段中位 | 训练段合计 | 墙钟 | 显存中位/最大 |
  |---|---|---|---|---|
  | baseline 100 轮 | 135.2 s | 13,524 s | 4 h06 min | 7,136 / 7,141 MB |
  | deep G=6 100 轮 | 171.3 s | 17,267 s | 5 h13 min | 8,520 / 8,740 MB |

  deep G=6 每轮为 baseline 的 1.27 倍、显存 1.19 倍；单卡合计约 9 h19 min（含验证与 checkpoint）。

## 结论

- 状态：**pending**（两组 100 轮完成、比较完成；采用判据与选点规则选型仍待用户决定）。**不自动采用新方案。**
- **相对差距是否缩小：主指标没有，反而显著扩大。**eDOS R² 差距从 35 轮的 −0.0062/−0.0046（区间含 0、
  不可辨）扩大到 100 轮的 **−0.0287*/−0.0131***（区间不含 0）。失败率差距缩小（+2.94*→+1.77*）
  但仍显著更差；phDOS 差距闭合（R² 转为不可辨、blind 失败率 −1.21* 反超 baseline）。
  按 `decisions.md` 的平局线（|ΔR²|<0.02 且 |Δ失败率|<1pp）：eDOS blind R² 差距 0.0287 **超出平局线**。
- **失败率是否改善：绝对值没有改善，全部恶化。**deep G=6 的 eDOS blind 失败率 11.72%→15.35%，
  baseline 8.78%→13.58%；仅相对差距缩小。phDOS 失败率同样恶化。
- **后期是否出现训练改善而验证停滞：是，且为本轮最强信号。**四条轨迹全部如此；
  选点指标（MAE 型）持续改善而 R²/失败率在早期见顶后持续回退。
- **对原始问题的回答：较长训练预算没有缩小新 Encoder 与 baseline 的 eDOS 性能差距，反而使其扩大。**
  长预算的真实作用是**同时压低两臂的 MAE 并推高两臂的失败率**，而不是修复 Encoder 差距。
  这与 35 轮结论一致并更强：差距更像**泛化/校准问题**（失败率、尾部），而非预算或容量不足。
  此为推断，单 seed=42 未验证跨初始化稳定。
- **采用判断（eDOS R²/失败率 + phDOS 保护三者同时）：不采用。**
  eDOS R² 显著更差、eDOS 失败率显著更差；phDOS 已保护（R² 不可辨、blind 失败率更好）。
  与 35 轮相比，phDOS 从"未保护"变为"已保护"，但 eDOS 两项判据从"部分达标"退为"均不达标"。
- **选点规则是本轮暴露的独立问题**：MAE 型 `balanced_oracle_mae` 在 100 轮下选出的 checkpoint
  （92/83）远离 R² 顶点（32–63）与失败率谷底（10–17），且 sel 与 R²/失败率方向第四次背离
  （sel 0.6330/0.6287 优于 35 轮的 0.6629/0.6554，而失败率差 4–5 pp）。若把 R²/失败率纳入判据，
  需要同时更换或加权选点规则，否则 best checkpoint 仍会偏向低 MAE 高失败率的晚期轮次。
- **预算边界**：结论限定 **seed=42、100 轮、crystal_q1_v1、`balanced_oracle_mae` 选点**；
  35 轮与 100 轮的 LR 轨迹不同，100 轮的前 35 轮不作为旧实验复现；不外推到其他预算、数据、
  硬件或初始化。单 seed 不宣称跨初始化稳定。

## 交接

- 下一项关卡工作：采用判据与**选点规则**是否一起改（把 R²/失败率纳入）；是否需要多 seed 复核
  "长预算扩大 eDOS 差距"的方向；`PYTHONUTF8=1` 缓解之外的永久编码修复（`tools/train.py` 读
  `tools/training-defaults.yaml` 显式 `encoding='utf-8'`，一行，待批准）；路线/深度/编排选型。
- 对 status、backlog 和 decisions 的更新：`status.md` 增长预算对照结果、"长预算扩大 eDOS 差距并
  恶化失败率"与选点规则张力；`index.md` 增本日志入口。
- 产物：`results/runs/lg_long_budget_20261008/`（`preflight.json`、`completion.json`、
  `monitor_snapshot.sh`、`queue.sh`、`export_and_compare.sh`、`trajectory_analysis.py`/`.json`、
  `exports/` 下两臂逐样本 CSV 与 summary 及 4 份配对比较）；训练产物
  `results/runs/ablation_m1_lg_baseline_q1v1_e100_s42/` 与
  `results/runs/ablation_m1_lgd_s42g6e10020261008_k2_g6_bp_staged_s42/`，
  历史 `results/reports/history_m1_*.csv`。旧 35 轮产物全部保留未改写。均未做 Git 提交。
