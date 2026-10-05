# 日志：2026-10-05 — 局部—全局代码审核收尾

## 范围

用户提供 MIMO、GLM、Kimi 三份外部审查结果；经逐项核查后授权完成收尾并验证。
本轮补测试与检查器覆盖、改善无法核验的产物报错、明确允许的共同配置和异常后恢复方式。
保留[初次接入](log-2026-10-05-local-global-encoder.md)的模型计算、局部公式、实验契约与主链，
没有正式训练、test 选型、checkpoint 删除或 Git 提交。完整 GPU 复查不更新权重；
仓库测试包含既有的合成数据训练回归，不属于正式对照实验。

## 文件、符号与反馈处理

| 文件与符号 | 变更与依据 |
|---|---|
| [`test_local_global_encoder.py`](../../tests/model/test_local_global_encoder.py)：`TestLocalGlobalEncoder.create`、原梯度/原子边界用例 | helper 可开启 Q1；原梯度前缀补 `atom_norm`；四组均检查 Z=118 可用以及有效原子槽中的 -1/0/119 拒绝 |
| 同文件：`test_q1_parameter_gradients_are_present_and_respect_zero_initialization` | 四组直接检查 `atom_norm/edos_trunk/phdos_trunk` 参数可训练、梯度存在且有限、模块有非零梯度；Q1 输出投影非零梯度，初始 fc1 零梯度符合既定零初始化 |
| 同文件：`test_mixed_empty_rows_match_separate_inference_and_keep_backward_finite` | 四组检查非空/空行混合 batch：encoder 空行严格零；与非空单样本及零宽空单样本推理一致；blind 空谱为零，反向链有限且活动 |
| [`test_local_global_comparison.py`](../../tests/pipeline/test_local_global_comparison.py)：原命令/拒绝用例、`test_existing_history_without_configuration_is_explicitly_unverifiable` | 共同选项仍可解析且记录契约；history 有而配置缺失时明确错误、文件不改写，不冒充 pending |
| [`local_global_compare.py`](../../tools/eval/local_global_compare.py)：`summarize` | 缺配置报 `ValueError`，含 tag、配置/历史路径与恢复说明；保留严格身份核验与异常即停止行为 |
| [`check_local_global.py`](../../tools/checks/check_local_global.py)：`check` | GPU 梯度前缀补 `atom_norm` 和两类 Q1 trunk；每组逐参数检查可训练/存在/有限，允许个别零梯度但模块必须有非零梯度；报告增加 `gradient_checks` |
| [接入设计](../design/design-2026-10-04-local-global-encoder.md) | 列明 Q1/Q2、谱轴编码、增广和覆盖掩码的默认与边界；说明配置缺失和队列停止后的恢复，不自动继续异常组 |
| 当前状态、索引及原日志 | 连接本次收尾证据，原 252 项/693 文件/GPU 计时保留为当次快照 |

新增固定用例三项，原局部—全局 12 项扩为 15 项。Z=0 作为 padding 仍合法；拒绝测试仅针对
被明确标为有效原子的零序号，不改变输入契约。Q1 fc1 的首步零梯度由零输出投影造成，
不是断开梯度；检查不能要求每个参数首步梯度都非零。

分数坐标增广与旧几何打分不是一回事：相同分数扰动在 `A→AQ` 下按
`δf AQ=(δf A)Q` 变换，不能把笛卡尔各向异性当作不等变。默认增广关闭，
具体是否启用是运行前应明确的共同条件；完整配置仍被模型契约记录。
未把共同选项全部加入拒绝清单，未加入“训练失败后自动继续”策略。

当前 SumNorm 将 min/max 槽设为 `(0,sum)`；现有非负谱口径中 oracle 的旧 clamp 是恒等操作。
本轮不为维护备注修改谱重建、argparse 或 CPU 参考初始化，也不把未来默认 CUDA device 的假设当作现有问题。

## 验证与证据

针对性检查命令（单线程 CPU，仅屏蔽检查进程的 DeprecationWarning）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore::DeprecationWarning \
  python -m unittest tests.model.test_local_global_encoder tests.pipeline.test_local_global_comparison -v
```

已执行 15 项，27.497 s 全部通过；涉及的四个代码/测试文件 ruff 通过。
原始输出见 [专项测试日志](../../results/runs/local_global_review_followup_20261005/unit-tests.log)。

完整仓库检查命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore::DeprecationWarning \
  bash tools/checks/check-static.sh
```

ruff、255 项测试（96.984 s）及 695 个候选仓库文件的文件策略检查全部通过。
原始输出见 [仓库检查日志](../../results/runs/local_global_review_followup_20261005/static-check.log)。
新增三项固定回归解释了原 252 项到当前 255 项的变化；两次检查不是两组独立用例。

GPU 检查沿用相同 V100、seed 42、K=2、M1/Q1、512 维、六层共享 decoder、E0/P0，
仅加载 train 结构字段，FP32 包含 192 原子个例，FP64 只测小批次。
新检查器读取完整可训练参数并记录梯度统计，不更新模型；原始报告不覆盖。

等价的独立重跑命令：

```bash
python tools/checks/check_local_global.py \
  --out results/runs/local_global_review_followup_20261005/fp32.json \
  --depth 2 --repeats 3 --precision fp32 --include-largest
python tools/checks/check_local_global.py \
  --out results/runs/local_global_review_followup_20261005/fp64.json \
  --depth 2 --repeats 3 --precision fp64
```

| 最终报告 | 四组完整模型 | E(3) 检查的输出最大绝对差 | 增强梯度检查 |
|---|---|---|---|
| [FP32](../../results/runs/local_global_review_followup_20261005/fp32.json) | 全部通过，含 192 原子个例 | 1.0728836e-6 | 每组 11 个模块前缀均通过 |
| [FP64](../../results/runs/local_global_review_followup_20261005/fp64.json) | 全部通过，小批次 | 3.9968029e-15 | 每组 11 个模块前缀均通过 |

E(3) 变换使用小批次；192 原子个例检查完整前向/反向、梯度与固定批次成本。
每组每种精度检查六个 E(3) 变换，共 48 次；物理记录拓扑保持，
两类谱、尺度与 blind 重建输出有限并满足既定容差。每个被选参数可训练、梯度存在且有限，
每个模块梯度总绝对值大于零。两类 Q1 trunk 各有两个首步零梯度张量（fc1 权重和偏置），
符合零初始化；输出投影接受非零梯度。
见 [GPU 运行日志](../../results/runs/local_global_review_followup_20261005/gpu-check.log)。

两份新报告的七个源码 SHA-256 均与最终工作区一致。相对原报告，仅检查器源码哈希改变，
其余六个模型与辅助源码哈希不变；四组各自的初值记录与对应原报告完全一致。
新旧报告分别保留，完整检查前后权重哈希不变、optimizer 更新次数为零，未读取谱标签。
最终五份文档的 108 个本地引用有效；已跟踪文件的 `git diff --check` 及十个新增文件的
行尾空白检查通过。文档归档不改变模型源码，因此无需重复执行已通过的模型检查。

## 结论与交接

本次处理的是已确认的覆盖与说明缺口，主体计算和科学选型保持。
已执行的回归、GPU 检查及最终改动复核均通过，本轮确认的收尾范围内未发现遗留问题。
正式训练、AMP、其他硬件、完整模型晶胞/超胞、整轮吞吐及路线准确率
仍按[当前状态](../status.md)另行确定和验收；上述通过结果不替代这些尚未执行的验证。
