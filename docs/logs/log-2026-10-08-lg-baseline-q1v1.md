# 日志：2026-10-08 — 原六层 Encoder＋ZP baseline 对照（seed=42）

创建日期：2026-10-08。状态：**closed**（本轮对照完成；采用判据与路线选型仍待决定）。
新增独立 baseline 对照模式并完成 35 轮训练、逐样本 valid 导出与配对 bootstrap 比较。
四组局部—全局实验未重训、产物未改写；历史 ZP100 指纹协议保留未动；未执行 Git 提交。

## 范围

- 任务与假设：在 seed=42、35 轮预算下，测量**替换整套 Encoder**（原六层 → A/B+ 局部—全局家族）
  的效果；对照组为"原六层 Encoder＋ZP"，共同参数与四组 `local-global-init-v1` 参考模型对齐。
- 改动的文件或配置（无科学配置变更）：

  | 文件 | 符号与职责 |
  |---|---|
  | [`experiment_config.py`](../../utils/experiment_config.py) | `ExperimentConfig.lg_baseline`：独立 baseline 对照开关，不是路线值 |
  | [`transformer.py`](../../model/transformer.py) | `Transformer(..., lg_baseline=False)`：记录模式并强制 `token_num>=119`；旧 Encoder 与 `rp_proj` 几何打分保持不变 |
  | [`local_global_experiment.py`](../../utils/local_global_experiment.py) | 抽出 `untrained_reference_state`（四组参考模型构造，行为不变）；`validate_local_global_config` 允许 baseline 在无路线时使用选点指标 |
  | [`lg_baseline_experiment.py`](../../utils/lg_baseline_experiment.py) | `validate_baseline_config`、`baseline_reference_params`、`initialize_baseline_model`、`baseline_contract`；`BASELINE_INIT_PROTOCOL/CONTRACT_SCHEMA` |
  | [`train.py`](../../tools/train.py) | 接入 baseline 验证/初始化/契约/RNG；`evaluate_split` 让 baseline 复用 blind/oracle 评估；`evaluate_local_global_split` 增可选 `sample_ids` |
  | [`check-static.sh`](../../tools/checks/check-static.sh) | 注册 `tests.pipeline.test_lg_baseline` |
  | [`check_lg_baseline.py`](../../tools/checks/check_lg_baseline.py) | 训练前有限验证：配方、初始化、前向/反向、评估、断点恢复、代表性 GPU 批次 |
  | [`lg_valid_export.py`](../../tools/eval/lg_valid_export.py) | 从 best checkpoint 导出 valid 逐样本结果（含材料 ID），复核选点规则 |
  | [`lg_paired_compare.py`](../../tools/eval/lg_paired_compare.py) | 按材料 ID 配对的 bootstrap 比较 |
  | [`test_lg_baseline.py`](../../tests/pipeline/test_lg_baseline.py) | 初始化对齐、契约隔离、评估复用、验证拒绝，5 项 |

## 证据

- 命令、标签、随机种子和 epoch 预算：

  ```bash
  python tools/train.py --epochs 35 --batch_size 32 --lr 5e-05 --seed 42 \
    --tag _lg_baseline_q1v1_e35_s42 --atom_feat z_only_proj --skip_test_eval \
    --selection_metric balanced_oracle_mae --lg_baseline --q1_coord
  ```

  M1、ZP（`z_only_proj`）、`q1_coord`、SumNorm/H1（`norm=sumnorm`、`scale_mode=eta`）、
  FP32、batch=32、lr=5e-5、35 轮、`balanced_oracle_mae` 选点、仅 train/valid、无 AMP、
  不加载历史权重。模型/损失/优化器/调度/dropout/谱网格/数据顺序取自四组 `config_used.yaml`
  的同一冻结配方（`check_lg_baseline.py` 逐项核对 16 个键）。
- 初始化对齐（`lg-baseline-init-v1`）：重建四组 `local-global-init-v1` 的未训练
  A/staged/K=2/seed=42 参考模型，将 `encoder.*` 之外的共同参数与缓冲区逐值复制（147 个张量），
  旧 Encoder 独立初始化（62 个张量）。`reference_common_state_sha256 = 8eea1e0c95063424…`
  **与四组 `runtime.local_global_init.common_state_sha256` 完全一致**。
  共同部分 `79c996382bf2c3a5…`、旧 Encoder `927adeaf6ca39f25…`（本次运行）、
  整模型 `7bb47f4f4eb828d9…` 分别记录于 `config_used.yaml.runtime.lg_baseline_init`。
- 训练前验证（`results/runs/lg_baseline_q1v1_20261008/preflight.json`，有限批次、非完整资源扫描）：
  冻结配方 16 键一致、data contract 同 manifest；`TransformerEncoder` 6 层 + 共享 `RPEncoding`
  + 逐层 `rp_proj`（原几何注意力打分保留）；2 个真实 train 批（含 `[23,9]` 尾批拆分）前向/反向
  loss 2.87/2.52、梯度范数 1.24/1.15、全参数有限；64 个真实 valid 样本 blind/oracle 有限且
  generic 列与 oracle 列逐值相同；断点恢复权重/optimizer 逐值相等、RNG 重放一致、契约不匹配即拒绝、
  LR 调度在 0/5/21/35 轮快进后与连续训练一致；代表性 GPU 批次 0.20 s/4.8 GiB（352 原子）、
  0.37 s/6.7 GiB（380 原子）。
- 结果文件：`results/runs/ablation_m1_lg_baseline_q1v1_e35_s42/`（config/best/latest）、
  `results/reports/history_m1_lg_baseline_q1v1_e35_s42.csv`（35 行，loss 全有限）、
  `results/runs/lg_baseline_q1v1_20261008/`（`train_run.log`、`preflight.json`、`timing_vram.json`、
  `exports/` 下五组逐样本 CSV 与 summary JSON、`paired_compare.json`）。
- 指标（valid 2,313 条，逐样本配对；各自 best checkpoint 重算；中位数，fail 为 R²<0 占比 %）：

  | 组 | best ep | sel | eDOS blind R²/MAE/fail | eDOS oracle R²/MAE/fail | phDOS blind R²/MAE/fail | phDOS oracle R²/MAE/fail |
  |---|---|---|---|---|---|---|
  | **baseline（原六层）** | 34 | 0.6629 | 0.4729 / 1.3752 / 8.78 | 0.5099 / 1.3189 / 5.71 | 0.7272 / 0.00701 / 4.24 | 0.7352 / 0.00694 / 3.59 |
  | A/staged | 32 | 0.6620 | 0.4616 / 1.3665 / 11.28 | 0.5012 / 1.3169 / 8.26 | 0.7142 / 0.00708 / 4.84 | 0.7277 / 0.00703 / 4.32 |
  | A/alternating | 34 | 0.6607 | 0.4542 / 1.3615 / 11.11 | 0.5007 / 1.3143 / 8.13 | 0.7108 / 0.00719 / 4.89 | 0.7226 / 0.00697 / 3.89 |
  | B+/staged | 25 | 0.6652 | 0.4547 / 1.3802 / 10.94 | 0.4977 / 1.3234 / 7.39 | 0.7092 / 0.00715 / 4.76 | 0.7204 / 0.00704 / 4.02 |
  | B+/alternating | 31 | 0.6581 | 0.4564 / 1.3745 / 11.63 | 0.5018 / 1.3093 / 7.39 | 0.7139 / 0.00710 / 5.14 | 0.7241 / 0.00697 / 4.15 |

  baseline 重算 `balanced_oracle_mae` 0.6629263793 与 history 记录 0.6629263798 差 5e-10（独立重算）。
- 配对 bootstrap（2,313 材料逐 ID 配对，2,000 次重采样，seed 20261008；差值 = 组 − baseline，
  95% 分位区间；`*` 表示区间不含 0）：

  | 差值 | A/staged | A/alternating | B+/staged | B+/alternating |
  |---|---|---|---|---|
  | eDOS blind R² 中位 | −0.0113* | −0.0187* | −0.0182* | −0.0165* |
  | eDOS oracle R² 中位 | −0.0087 | −0.0092* | −0.0122* | −0.0081 |
  | phDOS blind R² 中位 | −0.0130 | −0.0164* | −0.0180* | −0.0133* |
  | phDOS oracle R² 中位 | −0.0075 | −0.0126* | −0.0149* | −0.0111* |
  | eDOS blind fail (pp) | +2.51* | +2.33* | +2.16* | +2.85* |
  | eDOS oracle fail (pp) | +2.55* | +2.42* | +1.69* | +1.69* |
  | eDOS blind MAE 中位 | −0.0088 | −0.0137 | +0.0050 | −0.0007 |
  | phDOS blind MAE 中位 | +0.00007 | +0.00018 | +0.00014 | +0.00009 |
  | eDOS/phDOS oracle MAE 中位 | 全部区间含 0 | | | |

  phDOS 失败率差值区间多数含 0（仅 B+/alternating blind +0.91*）。MAE 差值绝对值
  eDOS ≤ 0.014、phDOS ≤ 0.0002，区间全部含 0。
- 耗时与显存（CSV `epoch_time_s` 只计训练段；整轮墙钟另计；显存为 `peak_vram_mb` 分配口径）：

  | 组 | 训练段中位 | 训练段合计 | 整轮墙钟 | 显存中位/最大 |
  |---|---|---|---|---|
  | baseline | **135.4 s** | 4,742 s | ~150 s | **7,136 / 7,140 MB** |
  | A/staged 1–21 轮（旧执行） | 1,420.9 s | 29,868 s | — | 4,288 MB |
  | A/staged 22–35 轮（新执行） | 415.1 s | 5,805 s | ~455 s | 24,237 MB |
  | A/alternating | 412.1 s | 14,422 s | ~455 s | 24,110 / 24,835 MB |
  | B+/staged | 161.0 s | 5,774 s | ~177 s | 7,986 / 8,137 MB |
  | B+/alternating | 160.3 s | 5,739 s | ~177 s | 7,968 / 8,124 MB |

  baseline 每轮为 A 的 0.33 倍、B+ 的 0.84 倍，显存约为 A 的 0.30 倍、B+ 的 0.89 倍；
  旧 Encoder 无几何缓存，每轮等价于热缓存轮。A/staged 两段不可混算。

## 结论

- 状态：closed（对照完成）。**限定为 seed=42、35 轮下"替换整套 Encoder"的效果，不单独归因于
  局部聚合，也不自动替换现有 baseline。**
- 主要观察（事实，配对区间支持）：四组局部—全局臂的 eDOS/phDOS 中位 R² **全部低于**原六层 baseline
  （blind eDOS −0.011～−0.019，四组区间均不含 0；oracle 多数不含 0），eDOS 失败率**全部显著更高**
  （+1.7～+2.9 pp，八组区间均不含 0）；MAE 差异全部不显著。方向在两条路线和两种编排下一致。
- **选点指标与判据的张力**：`balanced_oracle_mae`（MAE 型）给出 0.6581–0.6652 vs baseline 0.6629，
  组间不可辨；而逐样本配对的 R² 与失败率一致偏向 baseline。MAE 型选点可能掩盖 R²/失败率差异，
  采用判据需要单独决定。
- 单 seed=42、35 轮，不能宣称跨初始化稳定；R² 差值幅度（0.008–0.019）与失败率差值（1.7–2.9 pp）
  是同一预算下的配对结果，不外推到其他预算、数据或硬件。
- 历史 ZP100（100 轮）与 B7（35 轮）为**原 Q1 未修复数据**、无 blind/oracle 拆分、旧执行实现，
  不与本表直接比较；本表全部为 crystal_q1_v1 修复版、同预算（35 轮）、同配对 valid 集。

## 交接

- 下一项关卡工作：eDOS/phDOS 采用判据决定（是否把 R²/失败率纳入）；路线/编排选型；
  是否需要多 seed 复核本对照方向。
- 对 status、backlog 和 decisions 的更新：`status.md` 增 baseline 对照结果与判据张力；
  `index.md` 增本日志入口。四组与 baseline 产物均保留在 `results/runs/`，原始日志在
  `results/runs/lg_baseline_q1v1_20261008/`。
