# 日志：2026-10-05 — 局部—全局完整接入与对照实验代码

本页保留初次接入的检查快照。外部审核后的覆盖补齐、配置说明与最新工作区验证见
[审核收尾日志](log-2026-10-05-local-global-review-followup.md)；原报告与测试计数不覆盖。

## 范围与决定

用户授权先实施已确认内容，再实施 A/B+ 与编排对照所需代码。
本次接通 ZP → 6.0 Å 物理记录/局部聚合 → 全局标量混合 → 两类谱及 eta/gamma，
没有运行正式训练、读取 test 预测、迁移历史 checkpoint 或更改局部核心公式。
实际训练深度、预算、重复种子、valid 选点方式与采用门槛仍由用户决定。

已确认原则与连接公式见[接入设计](../design/design-2026-10-04-local-global-encoder.md)。
新路径几何只进入局部消息；旧方向打分与历史初始化指纹保留在历史分支，退出新模型。
跨模块为 512 维反射偶标量，局部/注意力/FFN 使用独立残差与外层更新前 LN；
每轮参数独立，B+ 的方向状态不跨局部调用传递。

## 模块、文件与符号

| 模块/调用链 | 文件与主要符号 | 职责与行为 |
|---|---|---|
| 局部/全局 Encoder | [`model/local_global_encoder.py`](../../model/local_global_encoder.py)：`LocalResidualLayer`、`ScalarGlobalLayer`、`LocalGlobalEncoder`、`LOCAL_RADIUS` | 外层完整状态残差；标准可学习 Q/K/V MHA 与 FFN；两种编排和更新计数；空行不进入全屏蔽 softmax |
| 完整组装 | [`model/transformer.py`](../../model/transformer.py)：`Transformer.__init__`、`Transformer.forward` | 新 route/layout/depth 配置；记录单次构建与复用；新全局无 RP；decoder 不收结构；119 行元素表；空结构有限边界 |
| 共同初值/实验身份 | [`utils/local_global_experiment.py`](../../utils/local_global_experiment.py)：`validate_local_global_config`、`comparison_configs`、`initialize_comparison_model`、`comparison_contract`、`capture_training_rng`、`restore_training_rng` | 四组隔离配置、独立新初值协议、共同部件同初值哈希；完整配置身份；Python/NumPy/Torch CPU/CUDA 随机数保存恢复 |
| 训练及 valid | [`tools/train.py`](../../tools/train.py)：`train_and_eval`、`validate_zproj_config`、`evaluate_local_global_split`、`evaluate_split`、`build_arg_parser` | 新分支接入、显式选点、两种模式分别报告；历史指纹分支隔离，历史 RNG 函数使用独立别名 |
| 谱重建 | [`utils/spectral_inference.py`](../../utils/spectral_inference.py)：`reconstruct_h1_spectra` | 批量 softmax 谱形；blind 仅用结构原子/价电子计数、预测 eta/gamma 和 bin 宽度；B7 冻结推理函数未修改 |
| 保存/恢复 | [`utils/ablation_checkpoint.py`](../../utils/ablation_checkpoint.py)：`build_ablation_checkpoint`、`restore_ablation_checkpoint` | 新 checkpoint 必须带模型契约和 RNG；拒绝不同模型/预算/数据契约，optimizer 不兼容时新对照停止；历史 schema 与兼容行为保持 |
| 配置 | [`utils/experiment_config.py`](../../utils/experiment_config.py)：`ExperimentConfig` | 新增 `local_route`、`local_layout`、`local_depth`、`selection_metric` 四字段；默认关闭新路径 |
| 四组命令/结果 | [`tools/eval/local_global_compare.py`](../../tools/eval/local_global_compare.py)：`training_command`、`summarize`、`main` | 默认打印四组命令；`--run` 才执行；汇总核对保存配置与选点列，pending/partial/complete 分开 |
| 无更新 GPU 检查 | [`tools/checks/check_local_global.py`](../../tools/checks/check_local_global.py)：`check`、`main`、`_WithSpectralGrids` | 仅读 train 结构字段；复用原 `check_crystal_input.profile`，验证完整输出、梯度、参数未更新和固定批次成本 |
| 回归入口 | [`test_local_global_encoder.py`](../../tests/model/test_local_global_encoder.py)、[`test_local_global_comparison.py`](../../tests/pipeline/test_local_global_comparison.py)、[`check-static.sh`](../../tools/checks/check-static.sh) | 12 项新增用例接入现有仓库检查；复用隔离 runner fixture，模拟恢复，无新数据训练 |

新增公共模型选项必须显式给出 A/B+、staged/alternating 和正整数 K。
每组含 K 次局部调用、2K 个内部局部块、K 次注意力与 K 次 FFN；K=1 两种编排完全相同，
四组生成器要求 K≥2。本次 K=2 仅用于无更新验收，没有选择正式训练层数。

## 对照入口与运行行为

以下命令只验证命令生成器，实际执行结果为四行训练命令，没有启动这些训练：

```bash
python tools/eval/local_global_compare.py \
  --local_depth 2 --epochs 2 --batch_size 2 --lr 5e-5 \
  --selection_metric balanced_oracle_mae --seeds 42 --tag_prefix codecheck
```

这组数字不是已批准的研究预算。正式执行前替换为选定配置，再显式添加 `--run`。
每个 seed 自动生成 `A/staged`、`A/alternating`、`B+/staged`、`B+/alternating`，
共同配置默认 M1/ZP/Q1/SumNorm/H1、仅 train/valid。tag 格式为
`_lg_{prefix}_k{K}_{a|bp}_{layout}_s{seed}`，不覆盖已有历史实验标签。
`--summarize` 不训练；汇总各组选定 valid epoch 的 blind/oracle MAE、R²、失败率及耗时/显存。
两种可选 balanced MAE 是两任务 MAE 中位数的等权平均，eDOS 优先及保护 phDOS 的最终采用门槛另定。

同一路线的两种编排完整 state 初值一致，A/B+ 的 ZP、外层 LN、全局、decoder/head 与尺度头初值一致。
复制参考参数后立即释放参考模型，不共享 Parameter；新 ZP 使用标准构造器初值，不套用历史单位投影/指纹。
新训练前设定同 seed 随机数流，恢复时使用保存的流。配置与 checkpoint 同时记录模型和数据契约。

## 无参数更新证据

新增 12 项 CPU 用例覆盖四组两 dtype 的完整输出 E(3)、原子重排、填充坐标 NaN 隔离、
空结构、Z=118、逐层梯度/独立 Parameter、共同初值和 K=1 编排同一性；
另覆盖命令解析、非法配置提前拒绝、契约混用、RNG 重放、blind/oracle 计算、汇总状态及实际 runner 模拟恢复。
完整模型 E(3) 论证见[对称性设计](../design/design-2026-10-04-e3-output-invariance.md#已实施新路径的全链论证)。

GPU 使用 Tesla V100-SXM2-32GB、Torch 2.2.1、单线程 CPU、无 AMP/TF32、模型 eval、seed 42。
外部宽度 512、8 头、FFN 2048、共享 decoder 六层、128/64 bin、Q1 谱坐标入口、K=2。
结构样本为缓存 train 索引 1/3/5，原子数 2/6/2，合并 batch=3、宽度 6；
仅加载晶格/坐标/元素，价电子数由冻结元素表得到，不读任何谱标签。
原记录 306 条，距截止最近的内部记录为 0.0387149 Å。FP32 另测索引 9452、192 原子，
该质量隔离个例只使用结构字段，不进入训练/准确率判断。

每组每 dtype 测整体平移、两个旋转、平面反射、中心反演和不正旋转加平移，共 48 次完整变换检查。
晶格与分数坐标按真实物理框架变换；逐项核对记录索引与镜像拓扑不变。
比较 logits、eta/gamma、两类 softmax 谱形与 H1 blind 物理谱，每组均通过，全部模型参数/缓冲区未变。
最大绝对误差：FP32 全输出 1.252e-6、FP64 3.775e-15；使用容差分别为 atol=rtol=2e-5 与 2e-8。
这属于有限样本、固定权重的证据，不证明所有输入或截止边界情况。

| 路线/编排 | 参数量 | FP32 batch=3 前向+反向中位 ms | FP32 单个 192 原子中位 ms | 两批次峰值 CUDA 分配 MiB |
|---|---:|---:|---:|---:|
| A / staged | 63,467,716 | 237.43 | 2998.87 | 1469.33 |
| A / alternating | 63,467,716 | 238.27 | 2995.91 | 1463.37 |
| B+ / staged | 63,314,124 | 136.58 | 126.11 | 1466.24 |
| B+ / alternating | 63,314,124 | 142.95 | 127.20 | 1466.24 |

成本包括每次重新构建物理记录与完整前向/反向，不含初始化、optimizer 状态/更新、整轮数据加载与 valid。
复用原 profiler 的两次预热、三次计时；B+ 首次正式计时约 1.29 s 的峰值原样保留，
中位数不能代替已确认的整轮稳定吞吐。CUDA 分配峰值不含驱动/上下文；这些批次不确定训练 batch 上限。
FP64 只测 batch=3，没有测 192 原子。路线的成本不同，不宣称参数量相同或准确率谁更好。

本地原始证据（运行目录按仓库约定不入 Git）：
[FP32 最终报告](../../results/runs/local_global_preflight_20261005/fp32_final.json)、
[FP64 最终报告](../../results/runs/local_global_preflight_20261005/fp64_final.json)。
早期 FP32 报告仍保留为 `fp32.json`，最终报告增加了 decoder 不接收结构的明确边界。
报告包含源文件哈希、样本、变换误差、共同初值哈希、原始计时与内存。

检查命令为 `tools/checks/check_local_global.py --out <新报告路径> --depth 2 --precision fp32 --include-largest`，
以及 `--precision fp64` 的小批次检查；检查器拒绝覆盖已有报告。

执行仓库既定检查：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore::DeprecationWarning \
  bash tools/checks/check-static.sh
```

结果：ruff、源码内存编译、252 项单元/合同测试全部通过（86.923 s），
仓库文件策略检查通过（693 个 prospective 文件）。252 项包含本轮新增 12 项；不能与此前阶段累计回归数相加。
DeprecationWarning 仅在检查命令中屏蔽，未修改运行时警告规则。`git diff --check` 无空白错误，
最终文档本地引用已核对；模型接入之外的既有改动保留，无提交或推送。
原始[仓库检查输出](../../results/runs/local_global_preflight_20261005/static-check.log)、
[四组生成命令](../../results/runs/local_global_preflight_20261005/plan-commands.log)和
[GPU 检查输出](../../results/runs/local_global_preflight_20261005/gpu-check.log)保存在同一本地目录；
初次接入完成时，两份最终 GPU 报告的源文件哈希与当时工作区逐项核对一致。
随后仅增强检查器的梯度断言与报告；模型计算源码未改，检查器新版本的证据见审核收尾日志。

## 结论与下一步

状态：**接入与对照代码已实施，正式实验 pending。**
完整输出和梯度的有限检查支持当前计算链可用；无新谱准确率、结构响应改善或路线胜出的结论。
下一步确定 K、训练预算与种子、valid 选点方式和两任务采用判据，再实际运行四组。
AMP、其他硬件、完整模型等价晶胞/超胞、正常截断弱角度响应与整轮成本仍未解决。
历史 ZP100/B7 使用原 Q1；与修复版输入比较时需注明数据口径，不能直接作严格同数据收益对照。
