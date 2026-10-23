# 日志：2026-10-21 — 完整 A 增量读出 LR1 正式运行

## 范围

- 按[设计](../design/design-2026-10-21-local-increment-readout.md)及[准备记录](log-2026-10-21-local-increment-preparation.md)，只训练 LR1 一组 35 轮、seed42；A64 使用已认证结果，不重训。
- 唯一模型变化为外层读出 `W_out(s2-s0)+b_out`。完整消息、输出偏置、内外残差、内部 post-norm、外层 pre-norm 均保持约定；`s0`不 detach。LR1 的初值张量与 A64 相同，不加载历史训练权重。
- 任务使用 `crystal_q1_v1` 完整 train/valid（18,704/2,313）；checkpoint 按 `balanced_blind_r2` 选点。test loader 与 test evaluation 均跳过。

## 命令、运行与产物

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_increment_compare.py \
  --tag-prefix res20261021 \
  --preflight results/runs/local_increment_20261021/preflight_final.json \
  --out results/runs/local_increment_20261021 --run
```

- 标签：`_lri_res20261021_delta_k2_g6_staged_s42_e35`；LR1、staged、局部深度2、全局深度6、batch32、lr `5e-5`、FP32。
- 本地时间：启动 `2026-10-21 15:56:21 +08:00`，结束 `2026-10-21 20:30:27 +08:00`；总 wall time **4:34:06**。35 个 epoch 的 history `epoch_time_s` 合计 14,404.17 秒（4:00:04），中位数 411.54 秒。
- PID：launcher 15185，比较 CLI 15187，训练器 15658。`queue.json` 为 `complete`，`open-code-run-exit.txt` 记录退出码 **0**；未重试、未改配置、未追加预算。
- 预检 `results/runs/local_increment_20261021/preflight_final.json` 通过，SHA256 `e5522ef07b27034895c4c2fd90ed9504074d5f0793d54a6c1b12d7a8f045c65a`。训练器日志明确记载 `Test loader and automatic test evaluation skipped by request`。

主要产物：

- LR1 checkpoint/history：`results/runs/ablation_m1_lri_res20261021_delta_k2_g6_staged_s42_e35/`
- history：`results/reports/history_m1_lri_res20261021_delta_k2_g6_staged_s42_e35.csv`
- 全量分析与组成配对导出：`results/runs/local_increment_20261021/analysis/`
- 汇总报告：`results/runs/local_increment_20261021/analysis/report.json`
- PID、起止时间与退出码：`results/runs/local_increment_20261021/open-code-run-control.json`、`open-code-run-exit.txt`

## 完成核验

- history 恰有 35 行，epoch 连续为 1–35；`checkpoint_latest.pth` 的 epoch 为35。
- history 中 `balanced_blind_r2` 最大值为 **0.614266**，对应 epoch34；`checkpoint_best.pth` 也为 epoch34。并列时按较早 epoch 选点。
- `config_used.yaml` 记录 LR1、seed42、35轮、`balanced_blind_r2`、`skip_test_eval=true`；运行记录为4个内部 post-norm 块、2个外层 pre-norm、2次增量读出。A64 参考目录为 `results/runs/ablation_m1_lawfresh_migr20261019_a64_k2_g6_staged_s42_e35/`，只复用已认证 checkpoint 与 train/valid 导出。
- 35轮训练及分析期间没有 OOM、NaN/Inf、traceback 或非零退出。日志中有 PyTorch scheduler `UserWarning`（step 顺序/epoch 参数提示）；本轮未更改调度器或配置，结果按已通过预检的训练与选点契约报告。
- Codex追加独立核验：`analysis/verification_codex.py`与`verification_codex.json`，`passed=true`。
  认证69份当前源码、63份历史源码及69份绑定源码快照，复核best/latest、共同初值元信息和实际归一化/读出记录；29份输入产物保持不变。
  从保存谱数组复算物理指标及168,136个失败标志，核对16份配对表、24个组成组区间、12个材料区间和联合区间。
  联合区间误差为0、物理指标中位数最大误差`3.98e−7`、配对量最大误差`1.43e−14`，均通过。
  本轮核验未执行模型前向、参数更新或test加载，也未执行Git提交。

## Blind valid 指标与采用判据

中位 R²按材料计算；失败率为 blind R²<0 的材料比例。LR1 与 A64 均使用各自按 `balanced_blind_r2` 选出的 checkpoint（LR1 ep34，A64 ep31）。

| 任务 | LR1 blind 中位 R² | A64 blind 中位 R² | LR1−A64 R²差 [95% CI] | LR1失败率 | A64失败率 | 失败率差 [95% CI]，pp |
|---|---:|---:|---:|---:|---:|---:|
| eDOS | 0.4811 | 0.4817 | −0.000614 [−0.007339, +0.007665] | 9.468% | 8.474% | +0.994 [+0.216, +1.773] |
| phDOS | 0.7474 | 0.7407 | +0.006706 [−0.000544, +0.014272] | 4.237% | 4.280% | −0.043 [−0.562, +0.519] |

联合分数为两任务 blind 中位 R²等权平均：LR1 **0.614266**、A64 **0.611220**，LR1−A64 **+0.003046 [−0.001525, +0.008896]**。材料配对 bootstrap 2,000次、seed20261016，两任务/模型共用抽样索引；区间仅描述选定 checkpoint，不包含训练 seed 不确定性。

按设计的点估计保护线，两任务 R²/失败率保护均通过；eDOS 失败率点差 +0.994 pp 接近 +1 pp 上限，其材料 bootstrap 区间上界为 +1.773 pp。联合增益区间下界未大于零，故 `increment_conditions_met=false`，**未达采用条件**。

## 完整 train/valid 同组成谱差诊断

表中为组成组平均的 signed-difference error 差（LR1−A64）；负值表示 LR1 误差较低，正值表示较高。区间以组成组为单位 bootstrap 2,000次、seed42；单训练 seed，诊断用途，不作多重比较校正。

| split | 组成口径 | 任务 | 组成组数 | LR1−A64 误差差 [95% CI] |
|---|---|---|---:|---:|
| train | absolute | eDOS | 1,330 | −0.000099 [−0.000632, +0.000444] |
| train | absolute | phDOS | 1,331 | −0.001311 [−0.002559, −0.000059] |
| train | reduced | eDOS | 1,790 | −0.000246 [−0.000679, +0.000175] |
| train | reduced | phDOS | 1,791 | −0.000783 [−0.001731, +0.000189] |
| valid | absolute | eDOS | 170 | +0.002690 [−0.000047, +0.005782] |
| valid | absolute | phDOS | 170 | −0.002579 [−0.006546, +0.001372] |
| valid | reduced | eDOS | 180 | +0.002080 [−0.000536, +0.005271] |
| valid | reduced | phDOS | 180 | −0.002410 [−0.006401, +0.001135] |

这是固定 checkpoint 的结构诊断，不参与选点或采用判定。train absolute phDOS 的组均误差差区间低于零；valid 各项区间均包含零。

## 耗时、显存与结论

- GPU：Tesla V100-SXM2 32GB。history 的 epoch 峰值分配显存最大 **25,146 MiB（24.56 GiB）**；每5分钟 `nvidia-smi` 监测的进程总占用最高 **31,442/32,768 MiB（95.95%，余1,326 MiB）**。两种数值分别是 PyTorch `max_memory_allocated` 与 `nvidia-smi` 进程占用，口径不同；运行完成，无 OOM。
- 状态：**park**。联合区间包含零，未达到正向采用条件；双任务点估计保护通过。保留 A64 为本项参考、继续保留 I1；不自动替换、不追加候选/预算、不启动后续实验。单 seed 结果不作跨 seed 泛化结论。
- 下一步：由用户另行决定是否采用或设计下一项实验；本轮不代替结构选型结论。
