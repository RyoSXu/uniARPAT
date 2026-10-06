# 日志：2026-10-06 — 正式 A 路径关闭角度重计算

## 范围与现状

用户明确授权先关闭角度重计算，并告知实验已停止。本轮开始时工作区干净、训练进程退出、
V100 32 GiB 空闲。四组 seed=42 对照曾以 K=2、35 epochs、batch=32、lr=5e-5、
`balanced_oracle_mae`、tag_prefix=`s42sel20261005` 启动；A/staged 的完整 checkpoint/history
到第 21 轮，其余组状态见本次只读汇总快照。第 22 轮未完成的步数不作为可恢复 epoch。

本轮只修改正式模型的执行方式和运行记录，不重启训练、不更新参数、不改写原实验产物。
FP32、batch、全量 6.0 Å 邻边对、内部两块、外层编排、loss、初始化和 checkpoint 科学契约保持。
角度分块仍为 4096；没有采用大分块、几何缓存或 AMP。
以上为本小单元的执行设置；后续已按用户决定采用大分块与结构缓存，见
[几何缓存日志](log-2026-10-06-local-geometry-cache.md)。本页保留原 4096 配对测量口径。

## 变更地图

调用链仍为 `train_and_eval → Transformer → LocalGlobalEncoder → LocalResidualLayer → PeriodicLocalMessage`。

| 文件 | 符号与职责 |
|---|---|
| [`model/local_global_encoder.py`](../../model/local_global_encoder.py) | `LocalResidualLayer.__init__` 显式传入既有 `recompute_angles=False`，正式 A 的四个内部块保存角度激活；B+ 没有此重计算分支 |
| [`tools/train.py`](../../tools/train.py) | 新增 `write_local_global_execution`，读取实际内部块设置并记录从哪一 epoch 生效；`train_and_eval` 在恢复成功后、恢复训练 RNG 前调用，仅对尚有 epoch 待训练的局部—全局模型执行 |
| [`tests/model/test_local_global_encoder.py`](../../tests/model/test_local_global_encoder.py) | 新增两 dtype、两编排、dropout=0.05 的开启/关闭完整模型对照；核对实际绕过 checkpoint、全部梯度、随机流和不变权重 |
| [`tests/pipeline/test_local_global_comparison.py`](../../tests/pipeline/test_local_global_comparison.py) | 新增执行记录兼容旧配置、保留科学契约/其他 runtime 字段/随机流、重复调用不重写的检查；既有真实 runner 合成恢复测试补执行记录断言 |

没有新增模型参数、state_dict 键、CLI 选项或 checkpoint schema。
独立 `PeriodicLocalMessage` 的默认构造选项保持；正式模型组装明确选择关闭。
独立成本工具仍允许显式切换重计算，以便复核两种执行方式。

恢复时 `config_used.yaml.runtime.local_global_execution` 记录实际设置，例如：

```yaml
- from_epoch: 22
  route: A
  recompute_angles: [false, false, false, false]
  pair_chunk: [4096, 4096, 4096, 4096]
```

旧配置没有此字段仍可恢复；记录从下一 epoch 开始，不推断未记录的历史执行设置。
资源设置不进入 `model_contract`；归一化、数据、路线、编排、预算和 RNG 的既有恢复校验继续执行。
成本报告应按执行设置的生效区间分段，不能把前 21 轮与后续轮次耗时混为单一配置。

## 验证与证据

正式通过目录为 `results/runs/local_global_no_recompute_20261006_v3/`，按仓库规则保留在忽略目录：

- `checkpoint_probe.py` / `checkpoint_restore_fp32.json`：只读恢复真实第 21 轮状态，固定普通/尾部批次的梯度、随机流、成本与文件身份检查。
- `random_b32_deterministic.json` / `tail_b32_deterministic.json`：确定性配对和同设置重复运行对照。
- `e3_fp32.json` / `e3_fp64.json`：四组完整模型、两 dtype 的有限 E(3) 检查。
- `static.log` / `focused.log` / `gpu.log` / `summary.json`：仓库检查、相关回归、GPU 原始日志与暂停队列状态。

第一次与第二次探针因常规 FP32 逐点梯度阈值失败而停止，源码与日志分别保留在
`local_global_no_recompute_20261006/` 和 `local_global_no_recompute_20261006_v2/`。
没有修改原有数值容差；失败原因和最终通过范围如下。

### 真实 checkpoint 恢复

采用现有 `restore_ablation_checkpoint` 恢复 A/staged 第 21 轮权重、完整 AdamW 状态和 RNG。
现有 `model_contract` 与当前配置严格一致，manifest 身份一致；共同初值协议及全部初始化哈希不变。
恢复模型完整 state_dict 与 checkpoint 逐值哈希一致；本次真实 A 模型的
**239 个参数张量**全部有有限梯度。

检查前后：模型/缓冲区、optimizer 的每个状态张量与参数组不变；原 `config_used.yaml`、
best/latest checkpoint、history CSV 的 SHA256 保持。未创建训练队列或进行 optimizer step。
仅结构输入参与探针，平方均值代理损失用于核查梯度，并非正式 SumNorm 标签监督或指标评估。

### 浮点差异的核查

在开启 dropout、恢复相同随机流的常规 GPU 配对中，普通/尾部批次输出最大绝对差分别为
9.537e-6 / 1.335e-5，均通过输出容差。全部梯度整体相对 L2 为 8.101e-7 / 1.563e-5；
但部分接近零的梯度元素未通过既有 `atol=2e-6, rtol=2e-3` 逐点检查，因此不能把常规
FP32 逐元素梯度对照表述为全部通过。

同一设置重复运行也出现误差：重计算开启时整体相对 L2 为 2.884e-7 / 1.721e-6，
关闭时为 2.843e-7 / 1.450e-5；部分同设置重复对照同样未过逐点容差。
推断：这些差异与普通 GPU 的 FP32 舍入/计算顺序波动相符，不能仅归因于改变重计算开关。

诊断进程设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并仅在配对验证时启用
`torch.use_deterministic_algorithms(True)`：两类真实批次、两种重计算设置的输出和
全部 239 个参数梯度最大绝对差均为 **0**；随机流相同。没有放宽容差，也没有将确定性
开关添加到生产训练入口。计时前恢复普通 GPU 执行。

CPU 新增对照也通过，覆盖 FP32/FP64、两编排、dropout 随机流与权重保持。
这些证据支持保存激活与重新计算的等价性，不承诺普通 GPU 的逐位重现或最终 R²/MAE 相同。

### 含真实 optimizer 状态的固定批次成本

V100-SXM2-32GB、PyTorch 2.2.1、FP32、单 CPU 线程，完整模型处于 train；
每项预热两次、重复三次，复用 `check_crystal_input.profile`。两种设置使用同一第 21 轮权重。
普通批次为 32 个合格 train 结构、352 原子，实际计算 `[32]`；尾部批次含 160 原子结构，
共 510 原子，保持原资源规则拆为 `[31,1]`，反向按结构数加权。

| 批次 | 开启前向+反向 ms | 关闭前向+反向 ms | 耗时减少 | 开启/关闭峰值 GiB |
|---|---:|---:|---:|---:|
| 普通 32 | 2746.03 | 1729.63 | 37.01% | 3.71 / 7.35 |
| 含 160 原子 | 5204.98 | 3240.97 | 37.73% | 3.68 / 7.33 |

峰值为 CUDA allocated，包含已恢复的真实 AdamW 状态；不含 CUDA 上下文，未执行 optimizer step。
计时包含记录构建、完整前向/反向与有限梯度检查，不是整轮训练吞吐。
上次未训练权重/不含 optimizer 的约 29% 探针与本次范围不同，不解释为固定收益保证。
整轮峰值与剩余训练的实际耗时仍由恢复后的日志验收。

### 全链与仓库检查

四组完整模型、FP32/FP64、六种 E(3) 操作的 logits、eta、谱形和 blind 重建通过；
最大绝对差 FP32 为 1.610e-6、FP64 为 3.997e-15。FP32 另查 192 原子结构压力例。
本轮 E(3) 检查用于功能验收，其短测成本不用于上表。

相关 31 项检查通过；完整 `check-static.sh` 的 Ruff、语法编译与 **259 项回归**通过
（102.423 s），最终政策检查通过（698 个 prospective 文件）。新增两项已包含于 259 项，
不与其他阶段数量相加。三份最终 GPU 报告与当前源码哈希一致；四份改动文档的
88 个本地引用均可解析，`git diff --check` 通过。

## 结论与恢复说明

状态：**closed**。正式 A 路径已关闭重计算；实际训练权重的确定性输出/梯度、恢复兼容、
随机流、有限 E(3) 和仓库回归通过。常规 FP32 的逐点差异保留说明，不宣称其全部通过。
原实验文件没有修改，训练保持停止；未提交本轮改动。

恢复时使用原队列参数，无需新增参数或更换 tag：

```bash
python tools/eval/local_global_compare.py \
  --local_depth 2 --epochs 35 --batch_size 32 --lr 5e-5 \
  --selection_metric balanced_oracle_mae \
  --seeds 42 --tag_prefix s42sel20261005 --run
```

A/staged 从最近完整第 21 轮恢复，下一轮为 22；其他未启动组仍从各自共同初值开始。
本轮未执行此命令，checkpoint 科学契约、谱指标和训练预算不因本实现优化重新选型。
