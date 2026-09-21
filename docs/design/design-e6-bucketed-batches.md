# 设计：E6 长度分桶与动态裁剪 batch

> 对应 `docs/status.md` 当前关卡。E6 仅降低 padding 计算成本，不是准确率方案；默认关闭。

## 目标与成功判据

- **事实审计：**Q1 train 的有效原子数中位数为 8、p90 为 26、最大为 80。现有缓存把每条样本固定填充为 80 个原子槽；batch 32 的随机顺序即使动态裁剪仍消耗固定槽的 58.9%。固定 20-batch 窗口内按长度排序后，估计消耗降至 17.8%。
- **唯一因素：**开启一个默认关闭的 `--use_bucket_batch`。它在每个 rank 已由 `DistributedSampler` 确定的 epoch 索引上，以固定 20 个 batch 窗口稳定排序；collate 只裁去每 batch 最后完全 padding 的原子槽。
- **语义边界：**每个 rank 的样本索引多重集与 `DistributedSampler` 相同、每条样本仍仅出现一次（单卡）；但 batch 组成和更新顺序改变，**不**声称逐步优化更新与 B7 等价，也不把资源结果解释为 accuracy 改进。
- **准入门禁：**关闭路径不变；启用后索引覆盖和 epoch 确定性成立，裁剪前后同一个 batch 的单样本 forward logits 最大绝对差不超过 `1e-5`（矩阵尺寸改变会改变浮点归约顺序，不能要求 bitwise），Q1 真 batch 的损失／梯度有限。V100 同配置实测至少 20 个训练步，无 OOM 且报告时间、显存与 padding 槽；只有资源收益才允许作为可选训练载体。

## 改动

- 新增 `LengthBucketBatchSampler`：包装现有 `DistributedSampler`，不改变其 rank、seed、epoch 或补齐规则；窗口大小固定为 20 batch，不暴露扫描旋钮。
- 新增 collate：只裁 `elements` 与 `positions` 的尾端全 padding 行，其余标签／统计量／网格字段保持原状；默认 DataLoader 路径完全不变。
- runner/config 只增加 `use_bucket_batch=False`，并记录有效配置。AMP 可独立叠加，但 E6 资源门禁先用 FP32，避免混淆因素。

## 测试关卡

- 合成索引：覆盖、无重复、固定 epoch 确定、不同 epoch 保持覆盖；窗口外不排序。
- collate：padding 裁剪长度正确，标签和可选字段逐元素保留；裁剪 batch 与原 batch 对同一模型输出一致。
- Q1 CPU smoke：启用 sampler/collate 的 `train_one_step` 有限且 H1 eta 激活。
- V100：B7 checkpoint、Q1 train、batch 32、warm-up 后 20 步，报告资源与平均原子槽；不跑 epoch、测试集或准确率比较。

## 成本与风险

- 模型当前对输入长度的所有相对特征实现必须尊重 padding mask；任何裁剪不等价即 park。
- 因为 batch 组成变化，未来若以 E6 承载科学 pilot，必须在日志中记录该因素，并不得与 FP32／非分桶历史训练直接宣称 accuracy 因果。
