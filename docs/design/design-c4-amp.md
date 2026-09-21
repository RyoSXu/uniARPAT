# 设计：C4 CUDA 自动混合精度（AMP）

> 对应 `docs/status.md` 当前关卡。C4 是降低实验成本的工程改动，不是准确率候选；B7 的 FP32 配方和默认路径保持不变。

## 目标与成功判据

- **目标：**在支持 CUDA FP16 的设备上，让 M1 的前向、损失和反向传播使用自动混合精度（automatic mixed precision，AMP），以降低显存或缩短后续 Q1 训练的每步耗时。
- **唯一因素：**新增默认关闭的 `--use_amp` 开关。开启时使用 CUDA FP16 `autocast` 与动态 `GradScaler`；不改变模型、数据、batch、损失、学习率、随机种子或评价口径。
- **技术门禁：**
  1. 关闭时输出与既有 FP32 路径逐元素一致；CPU 上请求 AMP 必须显式拒绝，不得静默降级。
  2. 在 V100、B7 512 维生产 batch 32 的同一真实 Q1 batch 上，AMP 前向概率相对 FP32 的平均绝对差不超过 `1e-4`，最大绝对差不超过 `2e-3`；AMP 单步损失、参数梯度和更新后参数均有限。
  3. `GradScaler` 必须实际参与反向与 step，且梯度裁剪开启时先 unscale；恢复 checkpoint 时应恢复 scaler 状态。
  4. 以 warm-up 后至少 20 次同 batch 训练步报告时间与峰值显存；无 OOM 和数值门禁通过才准入后续训练。若成本无下降，AMP 仍可保留为默认关闭的兼容选项，但不作为后续实验载体。

## 改动

- `ExperimentConfig` 与 runner 增加 `use_amp=False`；有效配置记录该值。
- `basemodel` 仅在 CUDA + 开启时进入 AMP 上下文，使用现有 scaler 完成 `scale → backward → unscale（若裁剪）→ step → update`；关闭路径维持原有 `backward → step`。
- 评估前向同样在启用时使用 AMP，避免训练和验证精度设置不一致；所有物理反归一化与指标累计维持原有 FP32 语义。
- runner checkpoint 仅在 AMP 启用时保存／恢复 scaler；旧 FP32 checkpoint 继续可读。

## 测试关卡

- CPU 合同：默认关闭不变；`use_amp=True` 在 CPU 上报清晰错误。
- CUDA 合同：生产形状 Q1 batch 的 AMP 前向概率误差、单步有限性、scaler 更新和梯度裁剪顺序。
- 资源脚本：B7 同一 batch 32，FP32 与 AMP 各 warm-up 后运行至少 20 步，输出版本控制的 JSON/CSV；不训练 epoch、不读取测试集。

## 成本与风险

- FP16 attention、softmax、KL/W1 与几何特征可能放大舍入误差；概率门限和真实 Q1 batch 门禁优先于吞吐收益。
- AMP checkpoint 不可与 FP32 resume 混用：恢复时必须以 checkpoint 中的 `use_amp` 与当前配置一致，否则报错。
- C4 不改变 B7 默认，也不产生 oracle/blind 指标或 accuracy 结论。
