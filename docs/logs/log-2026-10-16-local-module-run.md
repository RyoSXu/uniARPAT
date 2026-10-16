# 2026-10-16 局部 A/B+ 选型正式运行与结果

对应[选型方案](../design/design-2026-10-16-local-module-selection.md)与[准备记录](log-2026-10-16-local-module-preparation.md)。本轮只训练完整 A（I3），seed42、35 轮、batch32、FP32；I1/I2（B+）复用已核验结果，未重跑参考组。

## 运行记录

- 入口：`tools/eval/local_module_compare.py --tag-prefix locsel20261016 --preflight results/runs/local_module_selection_20261016/preflight_final.json --out results/runs/local_module_selection_20261016 --run`
- 启动前：无同组训练进程，GPU 空闲；正式前已用干跑核对仅输出一条 I3 命令。
- 起止：包装启动 2026-10-16 13:42:55 CST；训练 13:43:26 开始，18:16:22 第 35 轮结束；分析完成后包装记录 END 18:22:03 CST，**退出码 0**。
- 耗时：总墙钟约 4 h 39 min（含分析）；训练 35 轮逐轮耗时中位 423.1 s，合计 14,807 s（约 4.1 h）。
- 峰值显存：逐轮 `peak_vram_mb` 中位 24,439 MiB，最大 25,146 MiB。
- 异常：日志中 error / traceback / nan 计数均为 0；未自动重试或续跑。
- 测试集：训练日志确认 `Test loader and automatic test evaluation skipped by request`，未做 test 评估。

## 核验

- 历史 CSV `results/reports/history_m1_lms_locsel20261016_i3_k2_g6_staged_s42_e35.csv` 含 35 轮数据行（另有表头）。
- checkpoint 选点：按 `balanced_blind_r2` 在 valid 上取最大，第 30 轮（0.6122）。日志 `New best ... Epoch 30` 与报告 `best_epoch: 30` 一致；`checkpoint_best.pth` SHA256 前缀 `22c2aa616b055df6`，与报告记录一致。
- 分析报告 `results/runs/local_module_selection_20261016/analysis/report.json`：`analysis_complete: true`；预检 SHA256 与准备记录一致（`098127032d46b77b8cffebb8e2d5d0de9db46856c1d60fb4cd416c8875193572`）。
- 主表 bootstrap：2000 次，seed 20261016，按材料配对（2,313 个材料）。谱差诊断沿用历史组成组 bootstrap，2000 次，seed 42。
- Codex 独立核验通过：57份源码及数据/初值/RNG/契约匹配，三模型的35轮历史与best/latest一致；I1/I2的12份复用预测/摘要导出指纹未变。
- 从原始NPZ重算物理谱指标及252,204个失败标记，标记完全一致，中位数最大误差1.19e−7；从数据集原子计数与目标重建24张谱差表、40组区间，复算36项材料对比和3组联合区间，联合区间误差为0。未增加模型推理、参数更新或test评估，原始报告及导出指纹未变。
- 本地证据：[核验结果](../../results/runs/local_module_selection_20261016/analysis/verification_codex.json)、[复算脚本](../../results/runs/local_module_selection_20261016/analysis/verify_codex.py)。包装状态、启动脚本与输出已保存在[monitor](../../results/runs/local_module_selection_20261016/monitor/)；状态记录确认退出码0。

## 结果（valid blind 中位 R²，选点后）

| 模型 | eDOS R² | phDOS R² | 均衡分数 | eDOS 失败率 | phDOS 失败率 |
|---|---:|---:|---:|---:|---:|
| I1（简单径向，参照） | 0.4829 | 0.7379 | 0.6104 | 8.91% | 4.54% |
| I2（完整 B+，ep29） | 0.4811 | 0.7436 | 0.6124 | 9.94% | 4.02% |
| I3（完整 A，ep30） | 0.4848 | 0.7396 | 0.6122 | 7.87% | 4.37% |

对比差值（均衡分数 `balanced_blind_r2`，95% 区间）：

- **A − B+（I3 − I2）**：−0.00020，区间 [−0.00669, +0.00535]，区间包含 0。
  - eDOS：R² +0.0036，失败率 −2.08 pp [−2.94, −1.25]，失败数182对230，净少48例；phDOS：R² −0.0040，失败率 +0.35 pp [−0.30, +1.04]。A相对B+的两任务保护条件均通过。
  - 路线条件（联合增益区间下界 > 0）**未满足**。
  - 反向B+−A的eDOS失败率增幅超过1 pp保护线，反向路线条件也未满足；不能仅因联合区间含0就按性能等价处理。
- **A − I1（I3 − I1）**：+0.00180，区间 [−0.00340, +0.00710]，区间包含 0。
  - eDOS：R² +0.0019，失败率 −1.04 pp [−1.77, −0.39]，失败数182对206；phDOS：R² +0.0017，失败率 −0.17 pp。两任务保护条件均通过。
  - 升级条件（同一联合规则）**未满足**，`eligible_for_adoption: false`，原因为 `joint_blind_improvement_not_detected`。

Oracle 与同组成谱差（仅诊断，不参与选点或替换）：

- Oracle R²：I3 eDOS 0.5200、phDOS 0.7455；I2 为 0.5191、0.7536。
- **train、绝对组成、eDOS**：I3相对零差响应的误差差为−0.0112 [−0.0131, −0.0094]；I2为−0.0160，I1为−0.0106。此前此句缺少split/任务说明，不能读成valid收益。
- **valid、绝对组成**：I3相对零差响应的误差差为eDOS +0.00074 [−0.00265, +0.00403]、phDOS −0.00108 [−0.01033, +0.00830]，均未检出收益；误差比分别为1.00270/0.99646。相对I1/B+的两任务valid谱差区间也均含0，约化组成判读相同。

成本：I3单轮423.1 s、峰值25,146 MiB（24.56 GiB）；I2为172.5 s、8,678 MiB（8.47 GiB）；I1为131.7 s、6,753 MiB（6.60 GiB）。
A约为B+的2.45倍耗时、2.90倍峰值显存；单次记录不代表严格成本区间，不把MiB数值直接写成GB。

## 结论边界

- **统计路线判据未分胜负；用户已暂定A。** A−B+的联合增益区间含0，原报告保留 `route_reference: undetermined`。用户随后依据A的eDOS失败率更低，接受较高成本，将A作为后续局部研究参考；这项阶段取舍不表示A总体更优或A/B+等价，B+实现与证据保留。
- **A的eDOS失败率下降得到本轮材料配对支持**，相对I1/B+均检出；phDOS失败率差异未检出。这项收益与A的额外成本须共同报告，不据此改变预设联合主判据。
- **是否满足 I1 升级条件：否。** A−I1 区间包含 0，不自动替换 I1（`automatic_baseline_replacement: false`）。
- 结论限 seed42、35 轮、当前宽度与骨架；区间是在已选 checkpoint 上的不确定性，不是训练 seed 不确定性。
- 未运行额外种子、轮数、宽度或新公式；下一步先明确A的有限容量方案与成本，保留当前公式、残差、块数和全局编排。

## 产物

- 运行目录：`results/runs/local_module_selection_20261016/`（`i3_train.log`、`queue.json`、`analysis/report.json` 等）
- 候选目录：`results/runs/ablation_m1_lms_locsel20261016_i3_k2_g6_staged_s42_e35/`（`checkpoint_best.pth`、`checkpoint_latest.pth`、`config_used.yaml`）
- 历史：`results/reports/history_m1_lms_locsel20261016_i3_k2_g6_staged_s42_e35.csv`
