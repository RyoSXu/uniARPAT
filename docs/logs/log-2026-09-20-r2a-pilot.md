# 工作日志 2026-09-20：R2a 共享 Decoder 深度缩减成对 Pilot

## 范围

- **假设：**将 B7 M1 的共享 decoder 从 6 层缩减为 3 层，能够在不越过双任务负向平局线的前提下
  降低成本。
- **唯一变量：**`num_decoder_layers=6`（控制 `_r2a6ctl`）对 `=3`（实验 `_r2a3`）。两臂均为
  Q1、M1、E0/P0、SumNorm KL/W1/Huber、H1 eta/gamma、dropout 0.05、batch 32、seed 42、10 epoch。
- **数据口径：**Q1 train/valid/test 为 18,706 / 2,313 / 2,287；测试 CSV 均含 2,287 样本。

## 证据

- 执行命令：
  ```bash
  python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 --tag _r2a6ctl --decoder_layers 6
  python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 --tag _r2a3 --decoder_layers 3
  python3 tools/eval/r2a_pilot_verdict.py --ctl_tag _r2a6ctl --exp_tag _r2a3
  ```
- 控制 → 实验的 Oracle 指标：eDOS 中位 R²/失败率 `0.4751/4.50% → 0.4587/4.98%`
  （Δ `−0.0164/+0.48pt`）；phDOS `0.7220/1.22% → 0.7209/1.57%`
  （Δ `−0.0011/+0.35pt`）。两个任务均未越过 `−0.02/+1pt` 的负向边界。
- Blind：eDOS Δ `−0.0151/+0.35pt`；phDOS Δ `+0.0011/+0.31pt`。$C_v$ MAE 从
  `0.2906` 至 `0.2924`（Δ `+0.0019`）。
- 成本：参数量从 71,152,964 降至 58,540,868（−17.7%）；实际训练平均每轮
  `190.0s → 156.0s`（0.821x，−17.9%），峰值显存 `7,145 → 5,803 MB`（0.812x，−18.8%）。
- 机器可读证据：两臂 `results/history_m1_r2a*.csv`、Oracle/Blind summary 与 sample CSV，
  以及 `results/r2a_pilot_verdict.json`。V100 单 batch 预检另见
  `results/r2a_decoder_resource_v100.{json,csv}`。

## 结论

- **状态：ACCEPT_LOW_COST_CARRIER。**R2a 没有 accuracy win，故 B7 6 层仍是唯一的准确率参考，
  默认 YAML 也保持 6 层；不得将 3 层的平局表述为精度提升。
- R2a 同时满足预注册的双任务非劣边界，且时间和显存均严格下降，因此获准作为下一项“原子加性
  phDOS 读出”实验的低成本 decoder 载体。
- 不启动 R2a 的 35-epoch 确认，也不扫描 decoder 深度；本设计只检验 6→3 这一项缩减。

## 后续

下一项是为固定 P0 网格的原子加性 phDOS 读出单独撰写单因素设计。Q1 不含逐原子 PDOS 真值，
因此该方向只能表述为总谱弱监督下的原子加性约束，不能宣称预测了经标签验证的 atomic PDOS。
