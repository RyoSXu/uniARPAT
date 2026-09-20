# 工作日志 2026-09-20：R2a 共享 Decoder 深度缩减预检

## 范围

- **假设：**B7 M1 的共享 Transformer decoder 从 6 层缩减至 3 层，可能保留足够的双谱读出能力，
  同时降低计算和显存成本。
- **单一因素：**新增 `--decoder_layers`，默认 6；R2a 实验值为 3。它只覆写
  `num_decoder_layers`，不改变 encoder、输出头、数据、损失、H1、优化器或其他实验开关。
- **本日志不含训练结论：**这里只完成实现、合同测试和成本记录；Q1 M1×10 成对 pilot 尚未启动。

## 改动与测试

- 新增 `ExperimentConfig.decoder_layers` 与 `run_ablation_experiments.py --decoder_layers`；默认值 6 保持
  B7 行为不变。
- 新增 `tests/test_r2a_decoder_depth.py`：默认与显式 6 层逐元素等价、3 层输出形状与梯度、配置字段，
  以及生产维度、真实 Q1 batch 的 SumNorm + H1 `train_one_step` 冒烟。
- 新增 `tools/eval/r2a_decoder_resource.py`：同一 Q1 train batch（batch 32）上的 V100 稳态训练步成本记录；
  工具只报告实测值，不内置自动停损阈值。
- 验证：R2a 专项 4/4 通过；全量 `python3 -m unittest discover tests` 为 72/72 通过；
  `python3 run_ablation_experiments.py --help` 已显示 `--decoder_layers`。

## V100 成本记录

- 设备：Tesla V100-SXM2-32GB；Q1 train 固定第一个 batch，32 个晶体、81 个有效原子；先执行两个
  不计时预热步，再记录 5 个同批次训练步。
- 控制（6 层）：71,152,964 个可训练参数，328.89 ms/step，峰值已分配显存 7,141.12 MB。
- R2a（3 层）：58,540,868 个可训练参数（−12,612,096，−17.7%），272.03 ms/step（0.827x，−17.3%），
  峰值已分配显存 5,802.27 MB（0.813x，−18.7%）；两臂均完成且各损失有限，无 OOM。
- 机器可读证据：`results/r2a_decoder_resource_v100.json` 与
  `results/r2a_decoder_resource_v100.csv`。

## 下一关卡

R2a 技术预检通过，准入 Q1 M1×10 成对 pilot。执行时从零开始，使用 seed 42 与独立 tag：
`_r2a6ctl`（6 层控制）和 `_r2a3`（3 层实验）。按
`design/design-r2a-decoder-depth.md` 的双任务非劣与成本判据决定后续原子加性 phDOS 读出的载体；
不因本次成本记录直接启动 35-epoch 确认。
