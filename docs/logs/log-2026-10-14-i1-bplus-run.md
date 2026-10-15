# 2026-10-14 I1→I2 完整 B+ Encoder 对照运行

对应设计：[I1 完整 B+ Encoder 对照设计](../design/design-2026-10-14-i1-bplus-encoder.md)；准备记录见[准备日志](log-2026-10-14-i1-bplus-preparation.md)。

## 运行范围

- 复用已完成的 I1 参考目录，只训练 I2（完整 B+）；未从 I1 best 续训，未加载历史权重初始化 I2。
- 固定设置：seed42、35 轮、batch32、lr5e-5、FP32；选点规则 `balanced_oracle_mae`。
- 数据：修复版 `crystal_q1_v1`，train/valid 为 18,704 / 2,313；未使用 test 做选型。
- 输出目录：`results/runs/i1_bplus_encoder_20261014/`；I2 运行目录 `results/runs/ablation_m1_ibp_enc20261014_i2_k2_g6_staged_s42_e35/`。

## 运行状态（事实）

- 入口 `tools/eval/i1_bplus_compare.py --run`，launcher 退出码 0（`monitor/exit_code`）。
- I2 训练 35 轮，`best_epoch=35`，`checkpoint_epoch=35`，队列 `queue.json` 状态 `complete`。
- 训练日志扫描 nan / traceback / out of memory / error 均为 0；未出现 OOM 或进程失败。
- 分析报告 `analysis/report.json`：`analysis_complete=true`。
- 指纹与 checkpoint：I2 best checkpoint sha256 `e555306d…01d3e2`，配置 sha256 `098d372c…06f806`。

## 成本（事实）

| 项目 | I2 | I1（参考） |
|---|---:|---:|
| 每轮耗时中位数 | 172.8 s | 131.4 s |
| 训练轮时合计 | 6,178 s（约 1.72 h） | 4,718 s（约 1.31 h） |
| 峰值分配显存中位数 / 最大 | 8,512 / 8,678 MiB | 6,692 / 6,753 MiB |

I2 训练启动于 2026-10-14 16:51，训练结束于 18:44，分析导出结束于 18:51；入口全流程约 2.01 h，轮时合计不含全部 checkpoint 写出与后处理。

## 总体指标（完整 valid，所选 checkpoint）

中位数；差值为 I2 − I1，括号为 bootstrap 95% CI（按材料配对，2000 次）。

| 指标 | I2 | I1 | 差值 [CI] |
|---|---:|---:|---|
| blind eDOS R² | 0.4831 | 0.4801 | +0.0031 [−0.0062, 0.0100] |
| oracle eDOS R² | 0.5192 | 0.5193 | −0.00005 [−0.0089, 0.0077] |
| blind eDOS MAE | 1.3526 | 1.3617 | −0.0091 [−0.0388, 0.0193] |
| oracle eDOS MAE | 1.2930 | 1.2971 | −0.0041 [−0.0322, 0.0165] |
| blind eDOS 失败率 | 9.68% | 8.34% | +1.34 pp [0.52, 2.20] |
| oracle eDOS 失败率 | 6.44% | 5.75% | +0.69 pp [−0.04, 1.43] |
| blind phDOS R² | 0.7370 | 0.7400 | −0.0030 [−0.0123, 0.0047] |
| oracle phDOS R² | 0.7479 | 0.7515 | −0.0036 [−0.0116, 0.0041] |
| blind phDOS MAE | 0.00686 | 0.00686 | +1.3e-6 [−0.00015, 0.00017] |
| oracle phDOS MAE | 0.00677 | 0.00677 | +9.1e-7 [−0.00013, 0.00023] |
| blind phDOS 失败率 | 4.93% | 4.41% | +0.52 pp [−0.13, 1.17] |
| oracle phDOS 失败率 | 4.32% | 3.59% | +0.73 pp [0.09, 1.43] |

## phDOS 保护判断（按设计阈值）

- 中位 R² 相对 I1 下降不超过 0.02：blind −0.0030，oracle −0.0036，**满足**。
- 失败率增幅不超过 1 个百分点：blind +0.52 pp，oracle +0.73 pp，**点估计满足**；oracle 的 CI 下界为 0.09，即增幅统计上不为零，需留意。

## 同组成谱差（重点）

指标为组成组等权的有符号谱差误差差值（`mean_group_error_delta`，I2 − I1）。已从 `pair_errors` / `paired_group_interval` 及独立重算确认：误差为 `.5*sum(abs(Δpred−Δtarget))`，负值表示 I2 误差更小，正值表示更大。

**train（同组成）**

| 谱 | 组数 | I2−I1 误差差 [CI] | I2 误差比（相对零差响应） | I1 误差比 |
|---|---:|---|---:|---:|
| absolute eDOS | 1330 | −0.0069 [−0.0080, −0.0058] | 0.937 | 0.962 |
| reduced eDOS | 1790 | −0.0081 [−0.0093, −0.0070] | 0.940 | 0.970 |
| absolute phDOS | 1331 | −0.0181 [−0.0205, −0.0159] | 0.794 | 0.853 |
| reduced phDOS | 1791 | −0.0202 [−0.0222, −0.0180] | 0.798 | 0.864 |

**valid（同组成）**

| 谱 | 组数 | I2−I1 误差差 [CI] | I2 误差比 | I1 误差比 |
|---|---:|---|---:|---:|
| absolute eDOS | 170 | +0.0048 [0.0003, 0.0104] | 1.021 | 1.003 |
| reduced eDOS | 180 | +0.0046 [0.0004, 0.0105] | 1.020 | 1.004 |
| absolute phDOS | 170 | +0.0016 [−0.0031, 0.0065] | 1.0008 | 0.9957 |
| reduced phDOS | 180 | −0.0005 [−0.0053, 0.0046] | 0.9971 | 0.9986 |

零差响应基线：valid absolute eDOS 组均误差 0.2748，I2 为 0.2805，I1 为 0.2757；两者误差比点估计略高于 1。I2−零差响应为 +0.005678，95% CI [−0.000853, +0.013871]；I1 的区间也含零，均未检出优于零差响应的收益，不解释为与零差响应等价或确定更差。

方向一致性与 skill（valid，absolute eDOS）：I2 skill 中位数 −0.0011，I1 −0.0003；positive skill 比例 0.428 对 0.459。I2 预测差异的 TV 中位数 0.031，I1 为 0.023，目标为 0.270。

## 解释（推断，附不确定性）

- **事实**：I2 在 train 的同组成 eDOS/phDOS 误差上明显低于 I1，CI 全部位于零的同一侧；在 valid 的同组成 eDOS 上略差于 I1，CI 不含零，幅度约 0.005。
- **推断**：完整 B+ 方案提升了训练组的谱差拟合，未带来本次 valid 收益。局部表示和参数容量同时改变（新增 29,448 参数），不能把训练收益单独归于容量；train/valid 的组成与环境不同，也不能把分裂全部归因于过拟合。
- **不确定性**：单一训练 seed；CI 以所选 checkpoint 为条件，不反映训练种子方差；valid 的同组成组数仅 170/180。I2 的 best 位于预算末轮 35，本次未检验更长预算下的排序或是否收敛。
- 总体 eDOS 的 R²/MAE 未检出变化；blind eDOS 失败率增加 1.34 pp（CI 下界 0.52），由 193 / 2,313 增为 224 / 2,313，净增 31 个失败材料。phDOS oracle 失败率也检出上升，点估计仍在设计容忍线内，不能把 phDOS 描述为完全没有变化。

## Codex 独立核验

35 轮、best/latest、选点规则、初始化与起始 RNG 契约、45 份代码指纹及 I1 参考指纹均通过。
两模型完整 train/valid 的材料 ID 与目标对齐；从原始谱独立重算 16 张配对表、24 个组成组区间，
误差分别不超过 1.12e−16 / 1.39e−17。另重算 12 项 valid 总体对照，CPU FP32 物理指标中位偏差
不超过 1.20e−7，168,136 个失败标签全部一致；输入产物指纹不变。未重跑模型推理、训练或 test。
证据：[verification_codex.json](../../results/runs/i1_bplus_encoder_20261014/analysis/verification_codex.json)。

## 未解决事项

- valid 同组成 eDOS 的轻微变差与 blind eDOS 失败率增加是否需要进一步检验，由用户决定。
- 本运行未包含 test 评估；未做任何 test 选型。
- 工作区仍有大量未提交改动，本次未提交、未推送、未切换分支。
