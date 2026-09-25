# 设计：高粗糙度 eDOS 一阶差分损失成对 Pilot

> 对应 `docs/status.md` 的当前关卡。设计已确认，代码与合同测试已完成；本文件不代表获准启动训练。

## 目标与成功判据

- **假设：**冻结 B7 epoch 33 的 Q1 valid 形状诊断显示，高粗糙度谱的预测粗糙度偏低，误差集中在目标高梯度边，且 full-coverage 子集方向一致。给 eDOS 的归一化谱形增加一阶差分匹配项，可能改善高粗糙度谱的形状与 oracle R²。
- **历史边界：**C1.3 已试过 TV、通用梯度、峰/尾加权的组合；T1/T2 表现有害，LR 对齐重审后仍落后对照约 0.06。现行通用 `grad_w` 同时约束 eDOS/phDOS，直接作用在模型输出 logits 上；SumNorm 主损失则在内部用 `log_softmax` 形成分布。故新假设不是重开 C1.3 或扫描 `grad_w`，而是只比较 SumNorm 后的 eDOS 概率谱一阶差分，并保持 phDOS、TV、峰/尾权重和优化器不变。
- **唯一干预：**令 `z_e` 为 eDOS logits，`p=softmax(z_e)`，`q` 为 SumNorm eDOS 目标；对 128 个 bin 定义 `D(x)_k=x_{k+1}-x_k`，`k=0…126`。候选目标为 `L_total=L_B7+λ_slope·mean[(D(p)-D(q))²]`，均值覆盖 batch 和全部相邻 bin。使用与形状诊断相同的每-bin 一阶差分，不再除以固定 E0 bin 宽；不对样本按高粗糙度筛选或加权，不加 phDOS 项，不改标签或 coverage mask。
- **权重校准：**不做权重扫描。新增 opt-in 参数 `edos_slope_ratio`，默认 `0` 以确保关闭路径不变；本次候选固定 `ρ=0.10`。在 seed 42 初始化完成、首次 optimizer update 前，取训练流首个固定 batch，在 eval mode 下分别计算 eDOS 基础损失与未加权 slope 项对全体可训练 M1 参数的梯度范数，并固定 `λ_slope=ρ·||∇L_edos_base||₂/(||∇L_slope||₂+ε)` 至训练结束。λ 和梯度比写入配置/日志；范数为零或非有限则停止，不启动训练。校准只读训练标签、无 optimizer step，不动态改 LR 或后续权重。
- **基线与算力：**新鲜成对训练，Q1、M1、E0/P0、SumNorm KL/W1/Huber、H1 eta/gamma、dropout `0.05`、batch `32`、seed `42`、各 `10` epochs；控制与实验除 `edos_slope_ratio=0` 对 `0.10` 外完全相同。建议唯一 tags 为 `_edoslpctl` 与 `_edoslp`，启动前再次确认未占用。
- **主指标：**只在 Q1 valid 做 pilot 判定。高粗糙度组按 Q1 train 目标粗糙度 p90 固定分层（当前阈值 `0.351600`；valid 高组 205 条），比较第 10 epoch checkpoint 的 oracle eDOS 中位 R²与失败率；同一材料 ID 上做 2,000 次 paired bootstrap，报告差值 CI。pilot win 要求高组 `Δmedian R²≥+0.02`、`Δfail≤+1pp`，并且总体 eDOS oracle med 不低于 `−0.02`、失败率不增加超过 `+1pp`。
- **机制与副作用门槛：**同时报告高粗糙度组 `roughness_bias` 和高梯度边 slope-error MAE。只有高组 oracle 主指标过线，且高梯度边 slope-error MAE 的 paired-bootstrap 95% CI 全部低于零，才称为“pilot 支持 slope 假设”。eDOS blind、phDOS oracle/blind 的 med 均不得差于 `−0.02`，失败率不得增加超过 `+1pp`；任一越线则拒绝，不进入确认训练。主指标未达 win 线且未越保护线则 park；单 seed pilot 只用于筛选，不宣称泛化收益。

## 改动

- 实现保持单一目标项：新增 eDOS-only SumNorm slope loss 和默认关闭的 `edos_slope_ratio`；不复用或改变既有 `grad_w` 语义，不改其他损失、模型结构、训练超参或数据。
- `run_ablation_experiments.py --skip_test_eval` 默认关闭，启用时不构造 test loader，也不执行 test inference。`tools/eval/edos_slope_pilot_verdict.py` 固定评估 epoch 10 的 control/candidate checkpoint，仅对 Q1 valid 做逐样本 oracle/blind、粗糙度分层与 paired-bootstrap verdict。
- 合同测试覆盖 slope 公式、梯度有限性、初始梯度范数比校准、默认关闭配置、phDOS 专用输出头梯度隔离、skip-test loader 隔离、checkpoint 校准元数据和 valid 样本 ID 对齐。

## 测试关卡

- 代码与 CPU 合同测试已完成；真实训练 batch 冒烟、运行时开销与无 NaN/Inf 检查尚未执行，需在 pilot 获准后、正式训练前完成。默认 `edos_slope_ratio=0` 时不添加新损失项；开启路径的公式与校准由合成输入合同测试覆盖。
- 两臂训练后只报告 Q1 valid 第 10 epoch 结果和 paired CI，不看 Q1 test。若通过上述全部 win/副作用门槛，再将干预与权重校准协议冻结，另行提交 Q1 M1×35 等算力确认设计；只有确认方案获准后，才在最终确认中一次性使用 test。

## 成本与风险

- 成本为实现/测试、两臂 M1×10 以及每臂一次无 optimizer step 的权重校准 backward；一阶差分只增加 O(batch×127) 张量运算。实际 epoch 时间、峰值显存和无 NaN/Inf 仍须记录，不预设未测的性能收益。
- 既往 C1.3 负结果提高了本项的 park 概率；此方案仅因新诊断定位到 SumNorm 归一化形状的局部差分错误而值得小规模验证。valid 同时参与 checkpoint 选择与 pilot 判定，存在验证集乐观偏差；不能据单 seed/valid 结果宣称泛化提升。
- pilot 启动仍需单独授权；不得退而复用语义不同的 `grad_w`，也不得直接运行会自动读取 test 的训练入口。
