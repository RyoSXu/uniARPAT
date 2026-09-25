# 日志：2026-09-23 — eDOS slope-loss Q1 valid-only pilot

## 范围

- **任务与假设：**经马尚酱批准，按已确认设计比较 Q1 M1 seed 42、batch 32、10 epoch 的 slope-loss control/candidate。唯一差异为 `edos_slope_ratio=0` 对 `0.10`。
- **改动的文件或配置：**运行产物使用 `output/ablation_m1_edoslpctl/`、`output/ablation_m1_edoslp/` 与对应 `results/history_*.csv`；只运行 `--skip_test_eval`，不创建 test loader、不做 test inference。

## 证据

- **启动前检查：**工作区既有更改已保留；两个目标 output/history tag 均不存在；Q1 manifest、train eDOS 标签和 valid index 可读；Tesla V100 可用。
- **固定命令：**
  - Control：`python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _edoslpctl --seed 42 --batch_size 32 --lr 5e-5 --dropout 0.05 --norm sumnorm --edos_slope_ratio 0 --skip_test_eval`
  - Candidate：`python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _edoslp --seed 42 --batch_size 32 --lr 5e-5 --dropout 0.05 --norm sumnorm --edos_slope_ratio 0.10 --skip_test_eval`
- **无更新冒烟：**candidate 首个 train batch 完成前向、校准与反向传播；`lambda=1205.8605`、base/slope 梯度范数分别为 `0.499442` / `4.14179e-05`，203 个可训练梯度张量均有限，参数未更新。
- **训练完成：**两臂各 10 epoch。第 10 epoch 训练日志中的 valid median R²：control eDOS/phDOS `0.462/0.720`，candidate `0.462/0.719`；两次均输出 `Test loader and automatic test evaluation skipped by request`。正式结论使用第 10 epoch `checkpoint_latest.pth`，不使用 best checkpoint。
- **有效集审计命令：**`python3 tools/eval/edos_slope_pilot_verdict.py --control-checkpoint output/ablation_m1_edoslpctl/checkpoint_latest.pth --control-config output/ablation_m1_edoslpctl/config_used.yaml --candidate-checkpoint output/ablation_m1_edoslp/checkpoint_latest.pth --candidate-config output/ablation_m1_edoslp/config_used.yaml --output-prefix results/edos_slope_pilot_q1_valid --device cuda`
- **Q1 valid、epoch 10 全体指标**（n=2,313；格式为 median R² / fail%）：control eDOS oracle `0.4620/4.63%`、eDOS blind `0.4266/7.87%`、phDOS oracle `0.7199/2.03%`、phDOS blind `0.7106/2.94%`；candidate 分别为 `0.4617/4.63%`、`0.4260/7.57%`、`0.7188/2.03%`、`0.7103/2.85%`。配对 Δmedian 依次为 `−0.00032/−0.00054/−0.00103/−0.00023`；失败率变化依次为 `0/−0.303/0/−0.086` 个百分点。整体保护项通过。
- **高粗糙度主指标：**固定阈值为 train eDOS roughness p90 `0.3516`，valid 高粗糙度组 n=205。eDOS oracle median R² 从 `0.30045` 到 `0.29878`，配对 Δ=`−0.00167`（95% bootstrap CI `−0.01788…0.01379`），未达到要求的 `+0.02`；失败率从 `6.34%` 到 `5.85%`，Δ=`−0.488` 个百分点。高梯度 slope-error MAE 从 `0.040417` 到 `0.040580`，Δ=`+0.000162`（95% CI `−0.000611…0.000881`），没有支持预期机制。
- **结果文件：**`results/edos_slope_pilot_q1_valid.json`、`results/edos_slope_pilot_q1_valid_paired_metrics.csv`、两臂 sample/strata CSV，以及 `results/history_m1_edoslpctl.csv`、`results/history_m1_edoslp.csv`。
- **数据边界：**两臂都启用 `--skip_test_eval`；verdict 工具只评估 Q1 valid，并读取 train roughness 标签计算固定 p90 分组阈值；没有创建 test loader、执行 test inference 或读取 test 结果。

## 结论

- **状态：park。**`primary_met=false`、`guardrails_met=true`、`mechanism_supported=false`。
- **原因：**高粗糙度组没有达到预设的 `+0.02` eDOS oracle median R² 改善，斜率误差机制也未获支持；整体保护指标无明显退化。该单 seed、valid-only pilot 不支持进入 M1×35 确认或超参扫描。

## 交接

- **下一项关卡工作：**保持 slope-loss 默认关闭，不做 M1×35 确认、不扫描权重；与马尚酱讨论下一条独立的 eDOS 错误机制假设后，再单独设计 pilot。
- **对 status、backlog 和 decisions 的更新：**同步更新 `docs/status.md` 与 `docs/index.md`。不修改 `docs/decisions.md`：该 pilot 是单 seed、valid-only 的局部结论，不应固化为通用基准或可复用决策。
