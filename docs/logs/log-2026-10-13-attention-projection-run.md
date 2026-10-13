# 2026-10-13 — I1/I0 投影初始化与学习能力对照：训练执行、监测与配对归因

问题、协议与判读框见[设计](../design/design-2026-10-13-attention-projection-attribution.md)，训练前准入见[预检日志](log-2026-10-13-attention-projection-preflight.md)。
本轮只执行已确认的 I1 与 I0（seed 42、35 轮、batch 32、lr 5e-5、FP32、`balanced_oracle_mae` 选点，仅 train/valid），复用 D1、O1 与原 baseline 做配对分析。未读取 test，未改代码、配方或参考产物，未提交 Git，未启动下一轮。

术语（首次出现）：**恒等初值** 指 Q/K/V/O 投影以单位阵初始化、偏置为零；**有效可训练容量** 指 I0 因固定 Q/K/V/O 而减少的可训练参数；**配对 bootstrap** 指以材料为单位有放回重抽样 2,000 次，对同一批材料两组结果的差值求 95% 区间。

## 执行与身份

- 队列：tmux 独立会话 `uniarpat_proj_20261013`，驱动 PID 18006；按 I1 → I0 顺序执行。未使用既有会话 `uniarpat_lsc_diag_20261010`。
- I1：训练结束于 19:15:43 CST，35/35 轮，日志无 Traceback、OOM 或非有限值。
- I0：训练结束于 20:42:35 CST，35/35 轮，训练子进程 PID 31232，日志无 Traceback、OOM 或非有限值。
- 自动分析完成后队列标记 `complete`，`driver_tmux.exit` 为 0；端到端约 2.9 小时（队列启动约 17:48，结束约 20:45）。
- 两组测试评估均按要求跳过（`Test loader and automatic test evaluation skipped by request`）。
- 产物目录：`results/runs/attention_projection_attribution_20261013/`；两组运行目录 `results/runs/ablation_m1_ap_proj20261013_i{1,0}_k2_g6_staged_s42_e35/`，历史 `results/reports/history_m1_ap_proj20261013_i{1,0}_k2_g6_staged_s42_e35.csv`。
- 两个空目录 `results/runs/ablation_m1_ap_unit_projection_i{0,1}_k2_g6_staged_s42_e35` 为预检残留，未作为完成结果使用。

## 监测

- 只读脚本 `monitor.sh` 每 8 分钟写入 `monitor_raw.log`，记录进程、GPU、有限性、耗时与 ETA。
- epoch 训练耗时中位：I1 131.4 s、I0 130.9 s；I0 于 20:42:35 结束。
- 监测中出现的 PID 1030 为 I0 训练进程（31232）的子进程（同进程组，数据加载），不是重复运行；GPU 仅 31232 占用显存。
- 排查过程中一次对上级目录的递归 `grep` 超时，已作废，不影响结果。末尾的 "Exited with code 1" 来自检查命令中 `grep -c` 无匹配返回 1，并非训练退出；`i0_train.log` 与 `i1_train.log` 中无 `Exited`。
- 峰值显存：I1 中位 6,692 MiB（最大 6,753 MiB），I0 中位 6,582 MiB（最大 6,647 MiB）。

## 核验

- 两组历史各 35 行，epoch 1–35 连续；`queue.json`、`driver_tmux.exit` 与日志一致。
- 选点（`balanced_oracle_mae` 最小）：I1 为第 31 轮（0.65196），I0 为第 30 轮（0.65457）。两者均非末轮。
- 分析产物齐全：`analysis/report.json` 含 84 项总体差值、28 项同组成组差值；I0/I1/D1/O1/原 baseline 的 summary、valid 逐材料结果、谱和同组成配对表共 36 个分析文件。
- 末轮（第 35 轮）valid 指标：I1 为 eDOS R²(med) 0.520（MAE 1.303）、phDOS R²(med) 0.747（MAE 0.0068）、Balanced Score 0.6549；I0 为 eDOS R²(med) 0.517（MAE 1.306）、phDOS R²(med) 0.750（MAE 0.0068）、Balanced Score 0.6565。

## 结果

以下为 **blind / oracle** 两口径中位值与配对 95% 区间（材料 bootstrap）；`*` 表示区间不含零。
比较使用修复版 `crystal_q1_v1` 的 2,313 个 valid 材料；这里的原 baseline 是同数据、seed42、35 轮重训的旧 Encoder，不是旧数据上 100 轮的 ZP100。

所选 checkpoint 的 blind 总体结果如下，失败率为 R²<0 的材料比例。

| 组（best epoch） | eDOS R² | eDOS MAE | eDOS 失败率 | phDOS R² | phDOS MAE | phDOS 失败率 |
|---|---:|---:|---:|---:|---:|---:|
| 原 baseline（34） | 0.4729 | 1.3752 | 8.78% | 0.7272 | 0.00701 | 4.24% |
| O1（32） | 0.4793 | 1.3592 | 8.60% | 0.7350 | 0.00681 | 4.19% |
| D1（29） | 0.4707 | 1.3504 | 9.68% | 0.7278 | 0.00704 | 3.85% |
| I1（31） | 0.4801 | 1.3617 | 8.34% | 0.7400 | 0.00686 | 4.41% |
| I0（30） | 0.4749 | 1.3566 | 8.65% | 0.7372 | 0.00691 | 4.24% |

### 1. I1 − D1（恒等初值，可训练架构不变）

| 指标 | 差值 | 95% 区间 |
|---|---|---|
| eDOS blind R² | +0.009* | [+0.002, +0.019] |
| eDOS blind 失败率 | −1.34 pp* | [−2.29, −0.48] |
| eDOS blind MAE | +0.011 | [−0.027, +0.038] |
| eDOS oracle R² | +0.008 | [−0.00004, +0.01780]，含零 |
| eDOS oracle 失败率 | −0.95 pp* | [−1.69, −0.17] |
| phDOS blind R² | +0.012* | [+0.005, +0.020] |
| phDOS blind 失败率 | +0.56 pp | [−0.13, +1.25] |
| phDOS oracle R² | +0.016* | [+0.006, +0.022] |
| phDOS oracle 失败率 | +0.35 pp | [−0.22, +0.91] |

判读：恒等投影初值是当前设置下可分辨的正向因素，改善 eDOS blind R²、两口径失败率及 phDOS R²；eDOS MAE 未见可分辨改善，phDOS 失败率点估计上升、区间含零，不能称为全面改善。

附：I0 − D1（两组各训练 35 轮后的补充比较，同时改变初值和投影可训练性）：eDOS blind 失败率 −1.04 pp*（区间 [−1.99, −0.13]）、eDOS oracle 失败率 −1.17 pp*（[−1.90, −0.39]）、phDOS blind R² +0.009*（[+0.003, +0.019]）、phDOS oracle R² +0.013*（[+0.004, +0.020]）；eDOS R² 与其余失败率未分辨。方向与 I1−D1 一致，但不是第二次独立重复，也不能单独归因于初值。

### 2. I0 − I1（固定 Q/K/V/O，含可训练容量变化）

所有总体区间均含零（eDOS blind R² −0.005，区间 [−0.013, +0.001]；eDOS blind 失败率 +0.30 pp，区间 [−0.35, +0.91]）。同组成谱差亦均含零。判读：在本设置下，固定投影相对于恒等初值的可学习投影，无可分辨的净效应，不证明等价或投影学习无影响。该差值同时包含可训练参数由 75.9M 减至 69.6M 的容量变化，不能归于"投影学习"本身。

### 3. 相对 O1 与原 baseline 的剩余差距

- **I1 − O1**：eDOS blind R² +0.001、eDOS blind 失败率 −0.26 pp，均不可分辨；phDOS oracle R² +0.009* [+0.001, +0.017]。与此前 D1−O1 的 eDOS 失败率差距（+1.08 pp*）相比，I1−O1 的 eDOS 失败率差距已不可分辨。
- **I0 − O1**：eDOS blind 失败率 +0.04 pp，区间含零。
- **I1 − 原 baseline**：eDOS blind R² +0.007*（区间下界 +0.0001，边界）、eDOS oracle R² +0.009*；phDOS blind R² +0.013*、oracle R² +0.016*；eDOS 失败率未分辨。
- **I0 − 原 baseline**：phDOS blind R² +0.010*、oracle R² +0.014*；eDOS 指标未分辨。

### 4. 同组成谱差（单列，不与上面总体区间混合）

- 所有 I1−D1 与 I0−I1 的同组成组平均误差差值区间均含零（I1−D1 phDOS absolute −0.004，区间 [−0.009, +0.001]；I1−O1 与 I0−I1 均接近 0）。
- 同组成正技能比例（`positive_skill_fraction`）多数在 0.38–0.46 之间，`skill_median` 多数为负（约 −0.000 至 −0.009）；未见组内稳定的正向成谱差技能。该结论与历史 D1 相同。

## 判读边界

- 所有结论仅限单 seed（42）与 35 轮；材料 bootstrap 不代表跨训练种子的稳定性。
- I0 − I1 同时包含可训练容量变化，不是纯投影学习效应。
- 区间条件于所选 checkpoint（按 `balanced_oracle_mae` 选点），不含训练随机性。
- 各指标使用未作多重比较校正的 95% 区间；I1−原 baseline 的 eDOS blind R² 下界约 +0.00014，接近零，不能单凭这一项宣称稳定突破。
- 新旧骨架的 Encoder 内部参数化、容量与其余初值差异仍存在，结果不直接归因于归一化或 Q/K/V。
- I1/I0 的 best 非末轮不证明 35 轮已经充分；本轮没有检验延长预算后的排序。

## 结论与下一步

- I1−D1 是投影初值的直接证据：只改变未训练起点后，两组各训练 35 轮，eDOS blind R²、两口径失败率与 phDOS R² 得到可分辨改善；相对 O1 的 eDOS 失败率差距已不再可分辨，不等于差距为零。
- I0−I1 无可分辨净效应，说明在此设置下，固定投影与可学习投影在该结果上的差异未被检出。
- I1 相对原 baseline 的四项总体 R² 提高，eDOS blind 的区间接近零；phDOS 失败率相对 D1 的点估计上升尚未分辨，不能把它写成已经确认的剩余性能差距。同组成结构谱差仍未显示改善。
- 这轮支持将此前整体退化的一部分归于全局骨架设置，不能据此证明修正后的 I1 内局部更新仍有净收益。后者需要与相同修正骨架、关闭局部更新的模型直接比较；本轮尚未执行。
- 未启动下一轮。下一因素需用户确认后再决定。

## Codex 独立核验（2026-10-13）

用户报告完成后，独立核验通过。完整数值和运行文件哈希保存于
`results/runs/attention_projection_attribution_20261013/independent_audit.json`；临时独立统计脚本为
`/tmp/ap_independent_audit_20261013.py`。没有启动正式训练、执行 optimizer 更新、读取 test 或修改历史 checkpoint。

- **准入与产物**：重新执行 `verify_preflight` 与两组 `verify_completed`，40 份源码指纹、7 组参考身份和历史证据指纹一致；I1/I0 全部初值与训练起始 RNG 配对协议保持。五组 history 均为连续 epoch 1–35、所有数值有限；逐一从历史重新选点，与 best checkpoint 及报告一致；latest 均为 35。队列 complete、驱动退出码 0、两次训练子进程启动顺序 I1→I0，与日志一致；训练日志均有跳过 test 的记录，未检出 Traceback、OOM、NaN/Inf。
- **固定投影确实固定**：I0 的 best/latest 均严格满足六层 Q/K/V/O 恒等矩阵、零偏置，投影摘要等于未训练记录；仅这 24 个参数冻结、optimizer 没有其状态，共有 247 份可训练参数状态。I1 的 best/latest 投影摘要均已偏离恒等初值，optimizer 有 271 份参数状态。两个组的模型均保持 post-norm、概率 dropout=0 和其他 dropout=0.05。
- **独立重算统计**：从五组 2,313 材料逐样本表，独立重算 84 项总体差值、2,000 次配对区间与零判断，数值最大绝对差为 0。依据真实 valid 结构重新构造绝对/约化组成，复算 20 张谱差配对表，最大绝对差为 `4.44e-16`；170/180 组等权 bootstrap 的 28 项区间复算差为 0，全部含零。没有把同组成区间解释成总体性能结论。
- **重新推理**：以各组所选 checkpoint、原配置在 V100 FP32 上独立重跑 I1/I0 的完整 valid，各 2,313 条，目标数组逐值一致。与原导出比较如下；这是数值复核，不是逐位相同证明。

| 组 | eDOS 归一化谱形最大绝对差 | phDOS 归一化谱形最大绝对差 | 四项 R² 中位值最大差 | 四项 MAE 中位值最大差 |
|---|---:|---:|---:|---:|
| I1 | 2.61e-7 | 1.94e-6 | 6.56e-7 | 3.58e-7 |
| I0 | 3.43e-7 | 1.91e-6 | 3.58e-7 | 7.15e-7 |

两组各四项失败标签逐材料不变：I1 的 eDOS blind/oracle 为 193/133、phDOS 为 102/83；I0 分别为 200/128、98/81。极端常数 phDOS 目标 `mp-aaabnqya` 的方差为 0，blind R² 为约 −180.8 万/−204.6 万，重推理绝对差为 0.25/0.75；没有将所有逐材料 R² 差值写成微小误差，相关失败标签保持。五组配置、best/latest 共 15 个文件的重推理前后 SHA256 一致。

本轮只新增独立核验证据，更新此日志、`docs/status.md` 和 `docs/index.md`，收紧“差距消失”、I0−D1 的初始化归因及 phDOS 剩余差距表述。没有改模型、配方、选点或参考结果；源码指纹未变，未重复上一轮已执行的 391 项检查。三份文档的 148 处本地引用、`git diff --check` 和仓库产物规则（779 个候选文件）检查通过，未提交 Git。
