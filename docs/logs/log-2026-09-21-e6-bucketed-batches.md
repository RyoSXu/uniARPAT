# 工作日志 2026-09-21：E6 长度分桶与动态裁剪 batch

## 范围

- **目标：**减少固定 82-token 缓存中全 padding 原子槽的计算；不改模型、损失、数据、batch size 或准确率目标。
- **唯一变量：**默认关闭的 `--use_bucket_batch`。每个 rank 的原 `DistributedSampler` epoch 索引被固定 20-batch 窗口稳定排序，collate 仅裁掉 `elements`／`positions` 的末端全 padding 行。
- **语义：**每个 rank 的样本索引多重集不变；但 batch 组成和更新顺序改变，不能称为逐步更新等价，也不报告 accuracy 比较。

## 证据

- Q1 train 有效原子数：中位数 8、p90 26、最大 80。固定宽度 batch 32 为 2,560 原子槽；随机顺序的理论动态裁剪为固定槽的 0.589，20-batch 窗口分桶加裁剪为 0.178。
- 合同：分桶索引覆盖、无重复和确定性；Q1 裁剪 batch 的 tiny Transformer logits 与固定宽度输入最大差小于 `1e-5`（实测 `4.2e-7`；矩阵尺寸改变会改变浮点归约顺序，故不要求 bitwise）；真实 Q1 SumNorm/H1 `train_one_step` 全部有限且 eta 激活。
- V100 门禁命令：

  ```bash
  python3 tools/eval/e6_bucket_resource_gate.py
  python3 -m unittest discover tests
  ```

  B7 `_e9ctl` epoch 33、Q1 train、batch 32；每臂 warm-up 3 步后测量 20 步，不读 test split，不跑 epoch。

| 指标 | 固定宽度 | E6 分桶＋裁剪 | 比值 |
|---|---:|---:|---:|
| 平均原子槽 | 2,560.0 | 438.4 | 0.171× |
| 平均训练步耗时 | 329.44 ms | 177.48 ms | 0.539× |
| 峰值显存 | 7,141 MB | 7,144 MB | 1.000× |

- 全套 **87/87** 测试通过。机器可读资源结果：`results/e6_bucket_resource_v100.{json,csv}`。

## 结论

- **状态：技术通过。**E6 可作为后续训练的默认关闭低成本载体；耗时降低 46.1%，但峰值显存无实质改善，因此不授权提高 batch size。
- 分桶改变 batch 组成；后续如选用，必须在实验配置与日志中记录 `--use_bucket_batch`，不得将其训练指标与非分桶 B7 直接解释为模型或科学因素的因果变化。

## 后续

- 下一项独立工程候选为 E7 lint/CI；它只保护回归，不运行训练。
- 已更新 `status.md`、`decisions.md` 与索引。
