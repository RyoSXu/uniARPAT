# 日志：2026-10-08 — 全局深度解耦对照（B+ 局部 2 次 × 全局 6 层，seed=42）

创建日期：2026-10-08。状态：**pending**（两组已完成训练，逐样本导出与配对比较待收尾）。
本日志记录全局层数与局部调用数解耦的实现、严格初值对齐、训练前验证、无训练诊断与两组对照执行；
已有 B+ 两组与原六层 baseline **不重训、直接复用**；未读取 test；未做 Git 提交。

## 范围

- 任务与假设：检验**固定局部更新时，增加全局层数能否补回与原六层 baseline 的差距**，并检查编排影响。
  前提事实（见[baseline 对照日志](log-2026-10-08-lg-baseline-q1v1.md)）：已有四组局部—全局臂训练损失更低、
  验证 R² 更差，**因此差距不能直接归因于全局层数不足**。本对照只解耦"局部调用数"与"全局层数"这一个因素。
- 对照组（直接复用，不重训）：B+/staged 与 B+/alternating 的 K=2、全局 2 层结果
  （[四组对照日志](log-2026-10-07-local-global-four-arm-s42.md)）与原六层 `lg_baseline`
  （[baseline 对照日志](log-2026-10-08-lg-baseline-q1v1.md)）。
- 新增两组（固定 B+、2 次局部调用＝4 个内部局部块，各含 6 次全局注意力及 FFN）：
  - 分段：`L1 → L2 → G1 → G2 → G3 → G4 → G5 → G6`
  - 交替后补全局：`L1 → G1 → L2 → G2 → G3 → G4 → G5 → G6`
- 改动的文件或配置：

  | 文件 | 符号与职责 |
  |---|---|
  | [`experiment_config.py`](../../utils/experiment_config.py) | `ExperimentConfig.global_depth`：可选，缺省 `None` 保持旧语义 |
  | [`local_global_encoder.py`](../../model/local_global_encoder.py) | `LocalGlobalEncoder(..., global_depth=None)`：`execution_order` 泛化为分段/交替（等深时逐位等于旧顺序），`update_counts` 分列局部/全局计数 |
  | [`transformer.py`](../../model/transformer.py) | `Transformer(..., global_depth=None)`：透传；`global_depth` 需 A/B+ 路线 |
  | [`local_global_experiment.py`](../../utils/local_global_experiment.py) | `is_deep_depths`、`deep_reference_state`、`initialize_deep_model`、`training_record`、`depth_comparison_configs`；`DEEP_INIT_PROTOCOL`/`DEEP_CONTRACT_SCHEMA` |
  | [`train.py`](../../tools/train.py) | `--global_depth`；仅显式设置时写入配置；`evaluate_local_global_split` 增默认关闭的 `return_spectra` |
  | [`local_global_compare.py`](../../tools/eval/local_global_compare.py) | `summarize` 的 identity 增 `global_depth`（缺省 `None` 与旧配置一致） |
  | [`lg_paired_compare.py`](../../tools/eval/lg_paired_compare.py) | 可选 `--scope`（缺省保持原对比范围文本） |
  | [`local_global_depth_compare.py`](../../tools/eval/local_global_depth_compare.py) | 新：深度对照的计划/执行/汇总入口（复用 `training_command`/`summarize`） |
  | [`check_global_depth.py`](../../tools/checks/check_global_depth.py) | 新：训练前有限验证 |
  | [`lg_train_subset_diagnosis.py`](../../tools/eval/lg_train_subset_diagnosis.py) | 新：固定 train 子集无训练诊断 |
  | [`test_global_depth.py`](../../tests/pipeline/test_global_depth.py) | 新：顺序、同初值、契约隔离、E(3)、恢复，7 项 |
  | [`check-static.sh`](../../tools/checks/check-static.sh) | 注册 `tests.pipeline.test_global_depth` |

## 证据

- 训练命令、标签、随机种子和 epoch 预算（两组顺序执行、各自独立日志）：

  ```bash
  python tools/train.py --epochs 35 --batch_size 32 --lr 5e-05 --seed 42 \
    --tag _lgd_s42g6sel20261008_k2_g6_bp_staged_s42 --atom_feat z_only_proj \
    --skip_test_eval --local_route B+ --local_layout staged --local_depth 2 \
    --global_depth 6 --selection_metric balanced_oracle_mae --q1_coord
  # alternating 同，--local_layout alternating、tag ..._bp_alternating_s42
  ```

  M1、ZP（`z_only_proj`）、`q1_coord`、SumNorm/H1、FP32、batch=32、lr=5e-5、35 轮、
  `balanced_oracle_mae` 选点、仅 train/valid、无 AMP、不加载历史权重；其余取自四组
  `config_used.yaml` 的同一冻结配方（`check_global_depth.py` 逐项核对 16 个键）。
  执行方式与四组一致并自第 1 轮生效：`shared_geometry=true`、GPU 结构缓存
  `cell-frac-content-v1`（lazy、上限 `min(16GiB,half_device)`）。

- 初始化对齐（`local-global-depth-init-v1`，seed=42、B+、K=2）：
  逐值复制已有 `local-global-init-v1` 未训练 B+ 参考的**全部共同张量**（263 个，含两次局部更新、
  前两层全局及下游模块），G3–G6 共 48 个张量独立初始化；训练未导入任何已训练权重。

  | 哈希 | 值 | 含义 |
  |---|---|---|
  | `shared_state_sha256` | `9746b5825aab96ba…` | **等于已有 B+ K=2 参考模型整体状态** |
  | `extended_global_state_sha256` | `944ad93d990eca8b…` | G3–G6 独立初值 |
  | `state_sha256` | `dbdbbdf50a076083…` | 整模型（两编排相同） |
  | `reference_common_state_sha256` | `8eea1e0c95063424…` | **等于四组 `common_state_sha256` 记录值** |
  | `common_state_sha256` | `0e3f9c7bc57bb4a5…` | 本模型非局部消息部分（含 G3–G6，不与四组直接比较） |

  两个新编排 `state_sha256` 逐值相同；更新计数 `local_calls=2 / internal_local_blocks=4 /
  global_attention=6 / global_ffn=6`。契约为独立 schema `local-global-depth-model-v1`。

- 训练前验证（`results/runs/lg_global_depth_20261008/preflight.json`，有限批次、非完整资源扫描）：
  - **旧调用顺序与初始化保持**：A 与 B+ 两路线 × staged/alternating 四种组合，`common_state_sha256`、
    `state_sha256`、更新计数与执行顺序**全部复现四组记录值**（A `659844be…`、B+ `9746b582…`）；
    显式 `global_depth=2` 亦复现历史顺序。
  - **新编排同初值**：两编排五项哈希逐值相同；`ConfigBuilder` 重建可复现；G3–G6 与共享部分不同源。
  - **真实批次前向/反向**：2 个真实 train 批（含 `[23,9]` 尾批拆分）loss 2.95/2.58、梯度范数 3.35/3.64，
    全部参数梯度存在且有限。
  - **两类谱 E(3)**：真实 valid 批 × 6 种物理变换（平移/旋转×2/反射/反演/含平移非真旋转），
    eDOS、phDOS、H1 scale 及 shape/blind 重建谱；FP32 最大相对差 ≤2.4e-6，FP64 ≤4e-15，
    双精度为真实 double 计算（模型与浮点输入同转），参数对象同一性核对未被转换替换。
  - **契约隔离**：浅层契约／缺失契约／换编排契约均被拒绝。
  - **断点恢复**：权重、optimizer 状态、RNG 重放逐值相等；LR 在 0/5/21/35 轮快进后与连续训练一致。
  - **短测耗时与显存**（B+ 路线、真实批、含 optimizer 状态与几何缓存）：warm 0.44 s、尾批 0.57 s、
    冷缓存 0.45 s/步；`max_memory_allocated` 4.3–4.5 GiB、`max_memory_reserved` 11.4 GiB。
    **统计口径**：PyTorch CUDA 分配器单步计数，含模型/optimizer AdamW 矩/惰性几何缓存，
    不含 CUDA 上下文；warm 为同批预热后测，cold 为清空缓存后测，两者均不构成整轮吞吐结论。
- `tools/checks/check-static.sh` 全套 **280 项测试**与仓库检查通过
  （`results/runs/lg_global_depth_20261008/check_static.log`）。

- 无训练诊断（`results/runs/lg_train_subset_diagnosis_20261008/`，五组各自 best checkpoint、
  eval 模式、固定 train 子集 2,048 条，`default_rng(20261008)` 抽样、行序排序、
  material ID 与摘要哈希落盘 `report.json`；形状谱落盘 `spectra_*.npy`）：

  | 组 | train blind eDOS R² | valid blind eDOS R² | train→valid 落差 | train/valid blind eDOS fail % |
  |---|---|---|---|---|
  | baseline 原六层 | 0.5668 | **0.4729** | **0.094** | 2.69 / **8.78** |
  | A/staged | 0.6218 | 0.4616 | 0.160 | 1.61 / 11.28 |
  | A/alternating | 0.6197 | 0.4542 | 0.166 | 1.56 / 11.11 |
  | B+/staged | 0.6160 | 0.4547 | 0.161 | 1.42 / 10.94 |
  | B+/alternating | 0.6290 | 0.4564 | 0.173 | 1.32 / 11.63 |

  - 局部残差强度（`RMS(out−in)/RMS(in)`，有效原子槽累计）：局部块 0.37–0.49，**全局块 0.55–0.66**；
    baseline 逐层 0.39–0.48。
  - Encoder 输出尺度：baseline 1.05→1.01（基本不变）；四组局部—全局 **1.00→1.50（+50%）**，
    增量主要来自全局块。
  - 同组成不同结构（绝对组成 27 对／约化组成 74 对）：目标形状 TV 中位 eDOS 0.29／phDOS 0.31；
    预测形状 TV baseline 0.004/0.007，四组 0.015–0.037/0.041–0.084（约化口径），即结构响应大 4–8 倍；
    预测 TV 与目标 TV 的 Spearman 从 0.23–0.36 升到 0.53–0.71。但预测差异**只覆盖目标差异的
    1–13%（eDOS）/ 2–28%（phDOS）**。
  - **事实与推断分开**：事实为上表与统计量。推断（未验证）：差距更像泛化差距而非容量/优化不足——
    臂能更好拟合并区分结构，却更差泛化。**结构响应增大不自动代表预测改善**，两者已分开记录。
    证据边界：单 seed、2,048 子集、配对 27/74 对偏薄。

- 结果（valid 2,313 条逐样本配对，各自 best checkpoint 独立重算；中位数，fail 为 R²<0 占比 %；
  `balanced_oracle_mae` 选点）：

  | 组 | best ep | sel | eDOS blind R²/MAE/fail | eDOS oracle R²/MAE/fail | phDOS blind R²/MAE/fail | phDOS oracle R²/MAE/fail |
  |---|---|---|---|---|---|---|
  | **baseline（原六层）** | 34 | 0.6629 | **0.4729** / 1.3752 / **8.78** | **0.5099** / 1.3189 / **5.71** | **0.7272** / 0.00701 / **4.24** | **0.7352** / 0.00694 / **3.59** |
  | B+/staged G=2 | 25 | 0.6652 | 0.4547 / 1.3802 / 10.94 | 0.4977 / 1.3234 / 7.39 | 0.7092 / 0.00715 / 4.76 | 0.7204 / 0.00704 / 4.02 |
  | B+/alternating G=2 | 31 | 0.6581 | 0.4564 / 1.3745 / 11.63 | 0.5018 / 1.3093 / 7.39 | 0.7139 / 0.00710 / 5.14 | 0.7241 / 0.00697 / 4.15 |
  | **B+/staged G=6** | 35 | 0.6554 | 0.4667 / 1.3797 / 11.72 | 0.5053 / 1.3039 / 7.83 | 0.7144 / 0.00708 / 4.71 | 0.7258 / 0.00698 / 3.72 |
  | **B+/alternating G=6** | 35 | 0.6560 | 0.4599 / 1.3710 / 12.24 | 0.5053 / 1.3050 / 8.08 | 0.7099 / 0.00713 / 4.84 | 0.7190 / 0.00703 / 4.19 |

  数据口径：crystal_q1_v1 修复版（18,704 / 2,313 / 2,287），仅 train/valid，blind/oracle 拆分，
  未读取 test。选点规则沿用 `balanced_oracle_mae`；两组 G=6 的 best epoch 均为 **35（预算末轮）**，
  其选点指标到末轮仍在下降，**35 轮预算可能截断 G=6 轨迹**，这是本轮结论的硬边界。

- 配对 bootstrap（2,313 材料逐 ID 配对，2,000 次重采样，seed 20261008；差值 = 组 − 对照，
  95% 分位区间；`*` 表示区间不含 0）：

  **G=6 − G=2（同编排，唯一差异是全局栈深度）**

  | 差值 | B+/staged | B+/alternating |
  |---|---|---|
  | eDOS blind R² 中位 | **+0.0119*** | +0.0035 |
  | eDOS oracle R² 中位 | +0.0076 | +0.0035 |
  | eDOS blind/oracle fail (pp) | +0.78 / +0.43 | +0.61 / +0.69 |
  | phDOS blind/oracle R² 中位 | +0.0052 / +0.0055 | −0.0040 / −0.0051 |
  | phDOS blind/oracle fail (pp) | −0.04 / −0.30 | −0.30 / +0.04 |
  | 全部 MAE 中位 | 区间全部含 0 | 区间全部含 0 |

  **G=6 − 原六层 baseline**

  | 差值 | B+/staged G=6 | B+/alternating G=6 | （参照）B+/staged G=2 |
  |---|---|---|---|
  | eDOS blind R² 中位 | **−0.0062** | −0.0130* | −0.0182* |
  | eDOS oracle R² 中位 | **−0.0046** | **−0.0046** | −0.0122* |
  | eDOS blind fail (pp) | **+2.94*** | **+3.46*** | +2.16* |
  | eDOS oracle fail (pp) | **+2.12*** | **+2.38*** | +1.69* |
  | phDOS blind R² 中位 | −0.0128* | −0.0173* | −0.0180* |
  | phDOS oracle R² 中位 | −0.0094* | −0.0162* | −0.0149* |
  | phDOS blind/oracle fail (pp) | +0.48 / +0.13 | +0.61 / +0.61 | +0.52 / +0.43 |
  | 全部 MAE 中位 | 区间全部含 0 | 区间全部含 0 | 区间全部含 0 |

- 耗时与显存（CSV `epoch_time_s` 只计训练段；显存为 `peak_vram_mb` 分配口径，与四组/baseline 同口径）：

  | 组 | 训练段中位 | 训练段合计 | 显存中位/最大 |
  |---|---|---|---|
  | baseline | **135.4 s** | 4,742 s | **7,136 / 7,140 MB** |
  | B+/staged G=2 | 161.0 s | 5,774 s | 7,986 / 8,137 MB |
  | B+/alternating G=2 | 160.3 s | 5,739 s | 7,968 / 8,124 MB |
  | **B+/staged G=6** | 172.5 s | 6,169 s | 8,511 / 8,679 MB |
  | **B+/alternating G=6** | 172.7 s | 6,178 s | 8,513 / 8,682 MB |

  G=6 相对 G=2 每轮训练段 +7.1%/+7.7%、显存中位 +6.6%/+6.8%；相对 baseline 每轮 1.27 倍、显存 1.19 倍。

## 结论

- 状态：**pending**（两组完成、比较完成；采用判据与路线/深度选型仍待用户决定）。**不自动采用新方案。**
- **能补回的部分（事实，配对区间支持）**：固定 2 次局部调用时，把全局栈从 2 层加到 6 层，
  在**分段编排**下把 eDOS R² 与原六层 baseline 的差距从 −0.0182*（blind）/−0.0122*（oracle）
  收窄到 **−0.0062 / −0.0046，两个区间均含 0**，即 eDOS R² 中位已与 baseline 不可辨；
  相对同编排 G=2 的 blind eDOS R² 提升 +0.0119*（区间不含 0）。
- **补不回的部分（事实）**：
  - **eDOS 失败率不降反升**：+2.1~+3.5 pp（八组区间全部不含 0），高于 G=2 的 +1.7~+2.9 pp。
    R² 中位追平但尾部更差，说明全局深度改善的是主体分布而非失败样本。
  - **phDOS 未回到 baseline**：R² 差距 −0.009~−0.017（多组区间不含 0），与 G=2 同量级；
    phDOS 失败率差值全部含 0（相对 G=2 也无显著变化，**相对 G=2 的 phDOS 水平是保住的**）。
  - **编排影响可见但不稳健**：G=6−G=2 的 eDOS blind R² 提升仅分段编排显著（+0.0119*），
    交替编排 +0.0035（含 0）；两组 G=6 的 oracle R² 同为 0.5053。单 seed 不能区分编排效应与噪声。
- **采用判断（按"eDOS R²/失败率 + phDOS 保护"三者同时看）：不采用。**
  eDOS R² 达标（分段编排与 baseline 不可辨），但 eDOS 失败率显著更差、phDOS R² 仍显著低于 baseline，
  两项判据未满足。`balanced_oracle_mae` 给出 0.6554/0.6560（优于 baseline 0.6629 与 G=2），
  与 R²/失败率方向**再次不一致**，继续支持"MAE 型选点掩盖 R²/失败率差异"的判断。
- **对原始问题的回答**：现有差距**不能**主要归因于全局层数不足。加大全局深度只补回了 eDOS R² 中位
  这一项（且仅分段编排显著），失败率反而变差；结合无训练诊断（四组 train 拟合优于 baseline、
  valid 更差、落差 1.7–1.8 倍），更可能的解释是**泛化差距**，加容量并未解决它。
  这是推断而非结论：单 seed=42、35 轮，未验证跨初始化稳定。
- **预算边界（必须限定）**：两组 G=6 的 best epoch 均为 35（末轮），选点指标仍在下降；
  35 轮预算可能截断 G=6 轨迹，其表现可能低估。所有结论限定为 **seed=42、35 轮、
  crystal_q1_v1、`balanced_oracle_mae` 选点**，不外推到其他预算、数据、硬件或初始化。
- 历史 ZP100（100 轮）与 B7（35 轮）为原 Q1 未修复数据、无 blind/oracle 拆分、旧执行实现，
  不与本表直接比较。

## 交接

- 下一项关卡工作：eDOS/phDOS 采用判据决定（是否把 R²/失败率纳入）；是否需要多 seed 复核
  G=6 的方向；若要检验预算截断假设，需要单独批准延长预算（本轮未申请、未执行）；
  路线/深度/编排选型。
- 对 status、backlog 和 decisions 的更新：`status.md` 增全局深度对照结果、"R² 中位可补回但
  失败率不可补回"的判据张力与预算截断限定；`index.md` 增本日志入口。
- 产物：`results/runs/lg_global_depth_20261008/`（`preflight.json`、`check_static.log`、
  `train_*.log`、`monitor_snapshot.sh`、`export_and_compare.sh`、`exports/` 下两臂逐样本 CSV
  与 summary、`paired_vs_baseline.json`、`paired_g6_vs_g2_{staged,alternating}.json`）；
  无训练诊断 `results/runs/lg_train_subset_diagnosis_20261008/`（含 `spectra_*.npy` 与
  `subset_pairs.json`）；训练产物 `results/runs/ablation_m1_lgd_s42g6sel20261008_k2_g6_bp_{staged,alternating}_s42/`
  与 `results/reports/history_m1_lgd_*.csv`。均未做 Git 提交。
