# 日志：2026-09-23 — eDOS slope-loss pilot 设计草案

## 范围

- **任务与假设：**基于已完成的冻结 B7 Q1 valid 形状诊断，拟定是否值得验证高粗糙度 eDOS 一阶差分损失；本轮只写设计，不实现损失或启动训练。
- **改动的文件或配置：**新增 `docs/design/design-edos-slope-loss-pilot.md`；更新 `docs/index.md` 与 `docs/status.md`；无代码、配置、数据、标签或 checkpoint 改动。

## 证据

- **设计依据：**valid 高粗糙度组预测粗糙度偏差中位数 `−0.318`（95% CI `−0.328…−0.312`），高梯度边差分误差占比差 `+0.142`（`+0.121…+0.166`），与过度平滑相容；峰位位移不稳。
- **重复实验排查：**C1.3 的旧方案组合了 TV、通用梯度、峰/尾加权；T1/T2 有害，LR 对齐重审后仍落后对照约 `0.06`。旧 `gradient_loss` 同时覆盖 eDOS/phDOS 并对 raw model outputs 计算，而 SumNorm 基础目标把 logits 先经 `log_softmax` 解释为分布。新草案限制在归一化 eDOS 分布的一阶差分，未重开旧权重扫描。
- **评估与保护：**预设 Q1 valid、高粗糙度 train-p90 子组、10 epoch 同 seed 成对比较、梯度比一次性校准、双任务 oracle/blind 负向保护线；pilot 不读 Q1 test。当前训练入口默认构造 test loader 并在结束时自动评估 test，因此草案明确要求先增设默认行为不变的 skip-test 模式及 valid 样本级 verdict 工具。
- **结果与测试：**设计文档为 `docs/design/design-edos-slope-loss-pilot.md`。本轮仅有文档改动，未运行训练或代码测试；`git diff --check` 通过。

## 结论

- **状态：pending（待用户评审）。**这只是基于观察性诊断提出的单因素候选，C1.3 既往负结果提高了 park 风险；设计不声称新损失有效，也不授权实现或训练。
- **必须确认项：**是否接受新的默认关闭 eDOS-only slope-loss 参数、一次性梯度比 `0.10` 校准，以及为避免 pilot 读 test 而增加的 skip-test 入口。任一项未获确认，不进入代码实现。

## 交接

- **下一项关卡工作：**评审设计；确认后再另行开始实现和测试，训练前重新确认 pilot。
- **对 status、backlog 和 decisions 的更新：**更新 `docs/status.md` 与 `docs/index.md`；不改 `docs/decisions.md`，候选尚未实验验证。
