# 日志：2026-10-22 — 完整 A 内部残差加法对照 LR2 正式运行与核验

对应[设计](../design/design-2026-10-21-local-internal-residual.md)和[准备记录](log-2026-10-21-local-internal-residual-preparation.md)。本轮于 2026-10-21 晚启动、2026-10-22 完成。

## 范围与执行

- 只关闭完整 A 四个内部块的显式加法：`LN64(s+F(s,R))` → `LN64(F(s,R))`。完整消息及状态输入、内部 post-norm、外层 pre-norm、完整状态读出和外层原子残差均保留。
- 只新增 LR2 一组，seed42、35轮；A64复用已认证 checkpoint 和 train/valid 导出，不重训。配置 `init_ckpt` 为空；LR2 全部未训练张量与准备记录中的 A64 初值一致，state SHA256 `2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。
- 配方为修复版 `crystal_q1_v1` train/valid（18,704/2,313）、batch32、lr `5e-5`、FP32、staged L2/G6。候选标签：`_lres_resid20261021_noadd_k2_g6_staged_s42_e35`。

执行命令：

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_residual_compare.py \
  --tag-prefix resid20261021 \
  --preflight results/runs/local_internal_residual_20261021/preflight_final.json \
  --out results/runs/local_internal_residual_20261021 --run
```

## 运行与核验

- 预检 `passed=true`，SHA256 `2a187dd0abee0b731063ba311018bea0142d435884d079ecd7cd5ab931495811`。
- 本地时间：启动 `2026-10-21 21:40:19 +08:00`，结束 `2026-10-22 02:13:48 +08:00`；wall time **4:33:28.423**；退出码 **0**。supervisor PID 3505、比较 CLI PID 3506、训练器 PID 3914。queue 状态 `complete`，仅 LR2 一组完成；未重试、未改配置、未追加预算。
- 保存PID/起止时间/退出码：`results/runs/local_internal_residual_20261021/open-code-run-control.json`；训练日志 `lr2_train.log`、总控日志 `open-code-run.log`、51次周期监测快照 `open-code-run-monitor.jsonl`。
- 正式产物：候选 `results/runs/ablation_m1_lres_resid20261021_noadd_k2_g6_staged_s42_e35/`；history `results/reports/history_m1_lres_resid20261021_noadd_k2_g6_staged_s42_e35.csv`；分析 `results/runs/local_internal_residual_20261021/analysis/report.json`。
- history 恰有35行，epoch连续为1–35。按 `balanced_blind_r2` 最大值（并列取较早轮）独立复算 best 为 **ep29，0.623923**；`checkpoint_best.pth` 为ep29，`checkpoint_latest.pth` 为ep35。报告标明分析完整。
- 保存配置和实际执行记录均为 LR2：四个 `internal_residual_add=false`、四个内部 post-norm、两个外层 pre-norm、两次完整 state readout；`skip_test_eval=true`。训练日志明确记载 `Test loader and automatic test evaluation skipped by request`，没有 test evaluation 记录；本轮未读取 test。
- A64 `config_used.yaml`、best、latest 的 SHA256 均与预检绑定值一致，确认复用参考且未重训。准备/保存配置确认未用历史训练 checkpoint 初始化。
- 收尾独立只读核验了 history 与 best/latest checkpoint epoch、保存配置及执行开关、queue/control、分析报告和 A64 参考哈希；未运行额外训练或测试评估。

## Blind valid 指标与联合判据

失败率定义为材料 blind R²<0 的比例。区间为材料配对 bootstrap 2,000次、seed20261016；任务与模型使用共同抽样索引。

| 任务 | LR2 blind 中位 R² | A64 blind 中位 R² | LR2−A64 R²差 [95% CI] | LR2失败率 | A64失败率 | 失败率差 [95% CI]，pp |
|---|---:|---:|---:|---:|---:|---:|
| eDOS | 0.492047 | 0.481721 | +0.010326 [+0.001250, +0.018202] | 9.166% | 8.474% | +0.692 [−0.086, +1.471] |
| phDOS | 0.755799 | 0.740719 | +0.015080 [+0.007705, +0.023371] | 3.632% | 4.280% | −0.649 [−1.211, −0.086] |

联合 `balanced_blind_r2`：LR2 **0.623923**，A64 **0.611220**，LR2−A64 **+0.012703 [+0.006638, +0.018374]**，95%区间不含零。eDOS/phDOS 的 R²点差分别 +0.010326/+0.015080，失败率点差分别 +0.692/−0.649 pp；两任务设计保护线均通过。eDOS失败率差区间上界为+1.471 pp，但设计中的失败率保护按点估计判定，点差+0.692 pp低于+1 pp上限。

报告判定 `no_internal_addition_conditions_met=true`。上述区间条件于各自选出的 checkpoint，不包含训练 seed 不确定性；单 seed 结果不外推跨 seed 稳定性。

## 完整 train/valid 同组成谱差诊断

下表为组成组平均 signed-difference error 的差（LR2−A64）；负数表示 LR2 误差更低。区间按组成组 bootstrap 2,000次、seed42；诊断用途，不作多重比较校正。

| split | 组成口径 | 任务 | 组成组数 | LR2−A64误差差 [95% CI] |
|---|---|---|---:|---:|
| train | absolute | eDOS | 1,330 | −0.005098 [−0.006135, −0.004129] |
| train | absolute | phDOS | 1,331 | −0.013209 [−0.015309, −0.011418] |
| train | reduced | eDOS | 1,790 | −0.005054 [−0.005907, −0.004240] |
| train | reduced | phDOS | 1,791 | −0.013176 [−0.014848, −0.011637] |
| valid | absolute | eDOS | 170 | +0.001494 [−0.002130, +0.006293] |
| valid | absolute | phDOS | 170 | −0.007157 [−0.011427, −0.003018] |
| valid | reduced | eDOS | 180 | +0.001504 [−0.001964, +0.006056] |
| valid | reduced | phDOS | 180 | −0.007263 [−0.011644, −0.003145] |

train 的四项误差差区间均低于零；valid 中 phDOS 两种组成口径均低于零，eDOS 两项区间均包含零。谱差为诊断指标，不参与 checkpoint 选点或采用判据。

## 耗时、显存与异常

- 35轮 `epoch_time_s` 合计 **14,390.06秒（3:59:50）**，中位数 **411.01秒**。每轮 history 峰值分配显存最大 **25,145.96 MiB（约24.56 GiB）**。
- 51次周期 `nvidia-smi` 采样最高 **31,444/32,768 MiB（95.96%，余1,324 MiB）**；这是周期采样的设备占用，可能漏过采样间瞬时峰值，与 PyTorch `max_memory_allocated` 口径不同。
- 训练/总控日志未见 OOM、NaN/Inf、异常退出或 traceback。日志有 PyTorch scheduler 顺序/epoch 参数 `UserWarning`，与既往受控运行同类；本轮未改调度器或配方。

## 设计判断与边界

LR2 达到本设计的数值采用条件：联合 blind 增益95%区间下界大于零，且两任务点估计保护线通过。结果只支持在本固定配方、单 seed 下关闭四个内部显式加法；**不自动替换 I1，也不启动后续实验**。I1 仍为开发基线，是否把 LR2 纳入完整 A 方案由用户决定。
