# 工作日志 2026-09-21：R2b 原子加性 phDOS 读出 Pilot

## 范围

- **假设：**固定 P0 的总 phDOS 由 encoder 原子 token 的非负贡献求和，能够改善 phDOS；这是总谱
  弱监督下的原子加性读出，而非逐原子 PDOS 真值回归。
- **唯一变量：**R2a 3 层载体开启 `--use_atom_additive_phdos`。它跳过 phDOS query、decoder 调用和
  CNN，以共享 `512→512→64` 原子 MLP、Softplus 和有效原子求和替代；eDOS、encoder、Q1、损失、
  H1、batch、seed 和 epoch 不变。
- **对照：**关闭路径已逐元素验证，使用受版本控制的 R2a `_r2a3`；实验臂 `_apdossum` 从零开始，
  Q1 M1×10、seed 42。

## 证据

- 命令：
  ```bash
  python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 \
    --tag _apdossum --decoder_layers 3 --use_atom_additive_phdos
  python3 tools/eval/r2b_pilot_verdict.py --ctl_tag _r2a3 --exp_tag _apdossum
  ```
- 数据口径：Q1 test 2,287 样本，最佳 checkpoint 都为 epoch 10。

| 指标 | R2a `_r2a3` | R2b `_apdossum` | 增量 |
|---|---:|---:|---:|
| eDOS Oracle 中位 R² / 失败率 | 0.4587 / 4.98% | 0.4507 / 4.63% | −0.0081 / −0.35pt |
| phDOS Oracle 中位 R² / 失败率 | 0.7209 / 1.57% | 0.6881 / 1.57% | −0.0328 / +0.00pt |
| eDOS Blind 中位 R² / 失败率 | 0.4189 / 8.70% | 0.4134 / 8.26% | −0.0054 / −0.44pt |
| phDOS Blind 中位 R² / 失败率 | 0.7126 / 2.45% | 0.6794 / 2.49% | −0.0331 / +0.04pt |
| Cv MAE | 0.2924 | 0.3321 | +0.0396 |
| 参数量 | 58.541M | 28.088M | −52.0% |
| 平均每轮耗时 | 156.0s | 120.8s | 0.774x（−22.6%） |
| 峰值显存 | 5,803MB | 4,779MB | 0.824x（−17.6%） |

- 机器可读证据：`results/history_m1_apdossum.csv`、Oracle/Blind summary 与 sample CSV、
  `results/r2b_pilot_verdict.json`，以及 V100 预检结果
  `results/r2b_atom_additive_resource_v100.{json,csv}`。

## 结论

- **状态：PARK。**R2b 的 phDOS Oracle 中位 R²下降 0.0328，低于预注册的 −0.02 保护下界；即使
  eDOS 基本持平且成本显著下降，也不满足 phDOS win 条件。
- 不启动 35-epoch 确认，不扫描原子头宽度、深度、聚合或正则化；代码默认关闭。
- 此结果仅否定当前“低容量固定网格、总谱弱监督的原子加性 phDOS 读出”组合。它不表示结构信息无用，
  也不能把退化唯一归因于原子加性、容量下降或失去频率 decoder 的任一细节。

## 后续

当前 decoder/原子加性结构方向没有 accuracy win。下一项需回到状态页的独立候选选择：先评估
C2.1b（归一化/损失归因）的可执行设计，或推进 C4/E5/E6/E7 等不声称准确率提升的工程队列；
未经新假设，不重复 R2b 或展开其超参扫描。
