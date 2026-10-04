# 设计：H1 gamma 两参数部署校准门

状态：**已执行并 park。**train 最优映射近似恒等，Q1 valid blind eDOS 中位 R²变化
`−0.00050`，主门与机制门均未通过；不扫描映射或 loss。结果见
`../logs/log-2026-09-28-h1-gamma-calibration-gate.md`，上界证据见
`../logs/log-2026-09-28-model-accuracy-upper-bound.md`。

## 项目决策

判断冻结 B7 的 H1 gamma 是否主要存在可由全局单调校准纠正的收缩偏差，并且该纠正能否在不改变
oracle 谱形、phDOS 或模型权重的前提下，让 Q1 valid blind eDOS 中位 R²提高至少 `0.02`。

这是应用优先的最小门：通过即可形成可部署的尺度改进候选；失败只 park 这一个两参数校准，不把
gamma 上界、H1 memory 或其他头结构一并判死，也不自动启动新的头训练。

## 单一因素

冻结 B7 `_e9ctl` epoch 33 的全部权重与 eDOS shape。原 H1 输出 `gamma_pred` 为 control；candidate
只应用一个从 Q1 train 拟合后冻结的单调映射：

`gamma_cal = sigmoid(a * logit(clamp(gamma_pred, eps, 1-eps)) + b)`，`eps=1e-6`。

只拟合两个标量 `a,b`；初始化 `a=1,b=0`。目标为现有 Q1 train `gamma_true`，固定使用 soft-target
binary cross entropy；不扫描目标、正则、分组权重、映射类型或初始化。拟合只消费冻结 B7 的
`gamma_pred` 与 train 标签，不更新 checkpoint、encoder、decoder、H1 或谱形头。

选择 soft-target BCE 的原因是它在 logit 上直接给出 `prediction-target` 梯度，避免当前
sigmoid 后普通 MSE 在接近边界时额外乘 `p(1-p)`；这是假设，不是已证收益。两参数形式只检验全局
收缩／偏置，不承担学习新的结构信息。

## 数据与隔离

- 只构造 Q1 train 18,706 与 valid 2,313；显式拒绝 test split。
- 用冻结 B7、`eval()` 和 `inference_mode()` 一次生成 train/valid 的 eDOS shape 与 gamma；核对 checkpoint
  epoch、seed、SHA-256、样本顺序、有限值和 `[0,1]` 范围。
- `a,b` 只在 train 上拟合一次，随后冻结；valid 标签只用于最终裁决，不用于早停、选目标、选映射或
  改阈值。
- control/candidate 共用逐样本 shape、`N_val`、目标和顺序。candidate 仅替换 gamma；oracle eDOS、
  phDOS oracle/blind 应逐值不变，作为硬合同。

## 产物与实现边界

获批后先实现一个 `tools/eval/` 下的独立 valid-only 工具和必要的纯函数合同测试：

1. 冻结 B7 train/valid 推理与身份核对；
2. 两参数确定性拟合；
3. 输出 train 校准统计、valid control/candidate 逐样本表和裁决 JSON；
4. 拒绝 test、非 B7 checkpoint、非 Q1、非有限值和重复输出覆盖。

本门不修改 `EtaHead`、训练 runner、checkpoint schema 或 CIF 推理入口。只有 valid win 后才另写生产
接入设计，说明两个标量的保存、版本和 CIF 推理兼容性；本门本身不改变默认模型。

## 验收与行动

主要门槛：

1. valid blind eDOS candidate−control 中位 R² `>=+0.02`；
2. valid blind eDOS 失败率增量 `<+1pt`；
3. valid gamma 绝对对数比误差中位数下降，且 bootstrap 95% 区间上界 `<0`；
4. oracle eDOS、phDOS oracle/blind、eDOS shape 和模型参数逐值不变；不读取 test。

行动：

- 全部通过：记为 calibration win，只提出生产接入与受污染 test 历史下的最终确认安排；不自动改默认。
- 主指标平局或机制门失败：park 两参数校准，不扫描映射或 loss，不自动升级到新 H1 结构。
- 合同、数据隔离或数值失败：只修同一设计的技术问题，重新申请执行；不作科学结论。

## 明确不处理

- 不改 encoder、decoder、谱形损失、eta/phDOS、N_val 数据契约或 Q1 split。
- 不训练 M1，不做 gamma 分层采样、权重扫描、isotonic/spline/神经网络校准或 valid 调参。
- 不把 blind 改善解释为 oracle 谱形改善，也不声称解决 support Q4／高熵谱形瓶颈。
