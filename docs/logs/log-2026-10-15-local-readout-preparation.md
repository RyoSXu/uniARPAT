# 日志：2026-10-15 — 局部状态直连读出实现与短检

后续：三组正式训练已完成并于 2026-10-16 独立核验，见[运行结果](log-2026-10-16-local-readout-run.md)；本页保留执行前的准备事实。

## 范围与实现

用户确认[接口设计](../design/design-2026-10-15-local-readout-interface.md)：联合训练、完整 B+、
原损失与配方固定，仅增加 `memory = hG + g×(hL−h₀)`，g 初值为零、全模型共享。
I1 / I2 / I4 全部从头训练，seed42、35轮、batch32、lr5e−5、FP32，
共同使用 `balanced_blind_r2`。本轮仅实现和短检，未启动正式三组训练或读取 test。

| 调用环节 → 文件 | 关键符号与职责 |
|---|---|
| Encoder → `model/local_global_encoder.py` | `LocalGlobalEncoder.__init__/forward`：I4 使用完整 B+、staged L2/G6；新增 `local_readout_gate`，返回融合后的原子状态。 |
| 模型入口 → `model/transformer.py` | `Transformer.__init__` 接受 I4；原 `forward` 将同一 memory 交给两任务 Decoder 和池化 H1 头，无新增几何入口。 |
| 配方/初始化 → `utils/local_readout_experiment.py` | `local_readout_configs`、`validate_i4_model`、`initialize_local_readout_model`、`local_readout_contract`：冻结三组配方，I4 复制完整未训练 I2 初值，仅新增一个零门值；独立恢复契约。 |
| 公共分派/训练 → `utils/local_global_experiment.py`、`tools/train.py` | 校验、初始化与契约分派；`train_and_eval` 接入 I4 起始 RNG 记录及每轮门值。既有选点/恢复逻辑复用。 |
| 实验入口 → `tools/eval/local_readout_compare.py` | `verify_preflight/execute/analyze/gate_diagnostics`：复用现有顺序队列、完成核验、完整 train/valid 导出及配对统计；保存所选轮门值与轨迹，异常即停。 |
| 短检/回归 → `tools/checks/check_local_readout.py`、`tests/pipeline/test_local_readout.py` | `check/gate_zero_equivalence` 与 14 项回归，覆盖连接、梯度、RNG、对称性、恢复、选点和失败停止。 |

I4 共 75,923,661 参数，I2 为 75,923,660，I1 为 75,894,212。
历史路线的默认参数键与旧契约保留；I4 只增加 `encoder.local_readout_gate`。
g 的大小以及同组成谱差只作诊断，不代替联合 blind 主指标。

## 真实数据短检

最终 V100 预检通过，183.97 s，52 个源码指纹；数据为修复版 `crystal_q1_v1`，
有效 train/valid 18,704 / 2,313。三组分别对相同两批 train 做临时更新、对一批 valid 评估，
另做重复计时；未保存正式训练 checkpoint。采样器 seed0，训练 RNG seed42，
完整未训练 I1→I2→I4 初值及共同起始随机数流一致。

- 零门值检查包含原联合损失和 H1 监督，比较模型输出、原子输入及全部共同参数梯度。
  FP64 相对梯度差 1.89e−15；原 I2 自身重复为 1.86e−15。
  FP32 分别为 3.59e−5 / 3.36e−5，全部随机数流一致，门值梯度有限且非零。
  FP64 对照将模型与前后向状态转为双精度，损失沿用训练入口既有的 FP32 输出处理。
- 首次 GPU 检查在逐元素 atol=2e−6 处因 2.27e−6 的差异停止。
  独立重复检查发现原 I2 自身也有约 7.70e−6 的最大绝对差，随后新增自身重复参照与 FP64 检查，
  调整本次 FP32 门槛后完整重跑；未改模型公式或历史证据。
  最终 FP32 逐元素 atol=2e−5/rtol=1e−4，梯度整体相对差≤2e−4，且候选差≤3×自身重复差+1e−5；
  FP64 atol=1e−10/rtol=1e−9、整体相对差≤1e−9。不宣称 GPU FP32 逐位等价。
- I4 对称性短检显式设置 g=0.25，避免只检查关闭直连的情况。完整输出 E(3) 与原子重排
  最大相对差 FP32 2.03e−6、FP64 7.30e−12，均低于 1e−4 / 1e−9；NaN padding 检查通过。
- 三组全部可训练参数的真实梯度有限；权重、AdamW 与 RNG 恢复通过，不同接口/选点契约拒绝。
  I4 两批更新后 g≈−1.996e−5，证明该临时运行有学习信号，不证明泛化收益。

| 同一批更新 | I2 耗时中位数 | I4 耗时中位数 | I4−I2 峰值分配显存 |
|---|---:|---:|---:|
| 32结构、352原子，单微批 | 0.2351 s | 0.2420 s（+2.9%） | +4.94 MiB |
| 32结构、380原子，微批23+9 | 0.3559 s | 0.3588 s（+0.8%） | +1.56 MiB |

这是两个代表批次的短测，不能代替整轮计时或完整缓存峰值；三组仍规划约 5–6 h 单卡。

## 证据与交接

本地证据：[preflight_final.json](../../results/runs/local_readout_interface_20261015/preflight_final.json)、
[admission_codex.json](../../results/runs/local_readout_interface_20261015/admission_codex.json)。
预检 SHA256 `64d3bdef336fa4bc44c02406860b3091949bef4d7342d6a76064c7a9795aa148`；
目录的 `gradient_numerics/` 保留初次失败日志及独立精度探针。最终准入只读重验通过，
三组均为 pending、运行目录无正式产物，已生成隔离命令。

全仓 `tools/checks/check-static.sh` 通过：Ruff、内存语法检查、438 项测试（333.75 s）
及 805 个候选仓库文件边界检查。8 份相关文档的 85 个本地引用与差异检查通过。
正式执行使用 `tools/eval/local_readout_compare.py --tag-prefix ifc20261015 --preflight
results/runs/local_readout_interface_20261015/preflight_final.json --out
results/runs/local_readout_interface_20261015 --run`，须使用 UTF-8 模式和既定 Torch 环境。
入口依次完成三组及分析，仅分析全部成功后队列才记为 complete。
候选对 I2、I1 都需通过联合 blind 区间与两任务保护；未过线保留 I1。
同组成谱差与 Oracle MAE 即使改善也不覆盖主表，门值/直连收益不证明信息丢失的唯一根因。
