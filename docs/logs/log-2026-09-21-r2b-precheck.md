# 工作日志 2026-09-21：R2b 原子加性 phDOS 读出预检

## 范围

- **假设：**固定 P0 的 phDOS 可以由 encoder 原子 token 的非负贡献直接求和生成；Q1 只有总 phDOS，
  故这是弱监督原子加性读出，不是有逐原子真值的 PDOS 回归。
- **唯一变量：**`use_atom_additive_phdos`。关闭时为 R2a 3 层载体；开启时跳过 phDOS query、共享 decoder
  的 phDOS 调用和 phDOS CNN，以 `512→512→64` 的共享原子 MLP、Softplus 和有效原子求和取代。
- **未启动训练：**本日志只涵盖实现、合同和 V100 成本预检；Q1 M1×10 的 `_apdossum` 尚未启动。

## 证据

- `tests/test_r2b_atom_additive_phdos.py` 覆盖关闭路径逐元素等价、非负贡献、严格 padding 零贡献、
  原子置换、梯度、配置字段以及真实 Q1 batch 的生产维度 SumNorm + H1 `train_one_step`。
- R2b 专项 4/4 通过；在加入 R2b 后待全套回归完成再启动训练。
- V100（Tesla V100-SXM2-32GB）同一 Q1 train batch、batch 32、两次不计时预热加五次计时训练步：
  R2a 为 58,540,868 参数、271.82 ms/step、5,799.73 MB；R2b 为 28,087,683 参数、
  204.84 ms/step（0.754x，−24.6%）、4,760.10 MB（0.821x，−17.9%）。两臂损失有限，无 OOM。
- 机器可读证据：`results/r2b_atom_additive_resource_v100.json` 与
  `results/r2b_atom_additive_resource_v100.csv`；测量工具不设自动成本阈值。

## 下一关卡

R2b 通过技术与资源预检。完成全套回归后，以已有 `_r2a3` 为关闭路径控制，从零开始运行唯一实验臂：

```bash
setsid nohup python3 run_ablation_experiments.py --model M1 --epochs 10 --seed 42 \
  --tag _apdossum --decoder_layers 3 --use_atom_additive_phdos \
  > output/apdossum.log 2>&1 &
```

结果只按 `design/design-r2b-atom-additive-phdos.md` 的 phDOS win 和 eDOS 保护判据裁定；无 win 不启动
35 epoch 或头部超参扫描。
