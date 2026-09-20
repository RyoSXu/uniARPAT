# 设计：R2a 共享 decoder 深度缩减

> 关联：[状态与计划](../status.md)、[结构表征与读出重构方向](design-structural-refinement.md)。
> 本文只定义 R2a；原子加性 phDOS 读出须在 R2a 结论落盘后另立单因素设计。

## 目标与成功判据

检验 B7 M1 的共享 Transformer decoder 是否存在可安全删除的冗余层。当前 decoder 为 6 层，且每次
分别处理 eDOS 的 128 个 query 与 phDOS 的 64 个 query；本实验仅将其深度改为 3 层。

- **对照：**B7 生产配方：Q1、M1、6 层 encoder、6 层共享 decoder、E0/P0、SumNorm KL/W1/Huber、
  H1 eta/gamma、dropout 0.05、batch 32、seed 42、10 epoch。
- **实验：**完全相同的配方，唯一差异为 `num_decoder_layers=3`。
- **主要判据：**在 Test Oracle 上，eDOS 与 phDOS 的中位 R²相对对照都不得低于 0.02，任一失败率
  不得上升 1.0 个百分点；即两任务均不越过负向平局线。
- **准入下一项的条件：**满足主要判据，且单轮平均耗时或峰值显存至少有一项严格降低，R2a 才可作为
  后续 encoder-only 原子加性 phDOS 读出的低成本载体。否则维持 B7 的 6 层 decoder。
- **确认训练：**R2a 本身不以“统计平局”进入 35 epoch；它只是一项工程性筛选。除非出现任一任务
  Oracle 中位 R²提升至少 0.02，才另行记录理由并申请等算力确认。

## 改动

- 在 `ExperimentConfig` 和 `run_ablation_experiments.py` 新增唯一命令行参数
  `--decoder_layers`，默认值为 6；它只覆写
  `model.params.sub_model.transformer.num_decoder_layers`。
- `Transformer` 的既有 `num_decoder_layers` 构造参数直接生效；不改 encoder、query、输出头、
  损失、数据、优化器、H1 或推理路径。
- 默认值 6 必须保留 B7 的状态字典与关闭路径行为；不得借 R2a 顺带调整隐藏维度、FFN 宽度、dropout、
  学习率、warmup 或选择指标。

## 测试关卡

实施后、训练前必须通过：

1. 默认配置与显式 `decoder_layers=6` 的参数名、参数量和固定输入输出逐元素相等。
2. `decoder_layers=3` 的 eDOS/phDOS 输出形状仍为 `[B,128]` / `[B,64]`，H1 `eta` 仍为 `[B,2]`；
   padding 不产生 NaN/Inf。
3. 以 Q1 真实 batch 执行一次 `train_one_step`：所有损失有限，`loss_eta` 激活，3 层 decoder 参数
   获得有限梯度。
4. 在相同设备、batch 32、真实 Q1 batch 上记录 B7 与 3 层模型的单步峰值显存和耗时；该测量只做
   成本记录，不设未经批准的硬性资源阈值。

## Pilot 执行与记录

两个臂从零开始、顺序执行并使用不同 tag，避免共享检查点：

```bash
setsid nohup python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 --tag _r2a6ctl > output/r2a6ctl.log 2>&1 &
setsid nohup python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 --tag _r2a3 --decoder_layers 3 > output/r2a3.log 2>&1 &
```

训练结束后应生成一个可复用的成对判定脚本，比较两臂的 Oracle/Blind 中位 R²、失败率、gap、eta/gamma
误差、Cv MAE、参数量和资源测量。正式证据写入 `results/`，结论写入单独的日期日志；只有结论改变
后续载体选择时才更新 `status.md` 与 `decisions.md`。

## 成本与风险

- 3 层 decoder 预计降低 decoder 的计算和参数量，但 encoder 及相对几何注意力仍保持不变；总耗时
  的降幅不能由层数比例推断，必须实测。
- 共享 decoder 同时服务两条谱，缩减可能优先伤害 eDOS 或 phDOS；故不能只以总 loss 或单任务分数
  选型。
- Q1 缓存中只提供每晶体的总 phDOS 与 coverage mask，不提供逐原子 PDOS 标签。后续候选应命名为
  “原子加性 phDOS 读出”：它是对总谱的弱监督结构约束，并非可由本数据直接验证的原子 PDOS 真值。
- 该设计不授权启动 atomic 分支、P0 预训练或 35-epoch 训练。
