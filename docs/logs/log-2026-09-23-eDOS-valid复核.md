# 日志：2026-09-23 — eDOS Q1 valid 分层复核

## 范围

- **任务与假设：**复核 Q1 test 上观察到的 eDOS 粗糙度、熵、peak-share 与预测误差关联是否也出现在 valid；同时检查这些谱形分层是否伴随盲尺度 gamma 误差变化。
- **改动的文件或配置：**审计脚本增加 `valid/test` 切分支持和 C2.1b valid 样本序号核验；未改模型、数据、标签、评分定义或训练配置。

## 证据

- **命令、标签、随机种子和 epoch 预算：**`python3 tools/eval/edos_error_attribution.py --split valid`；只读复用 B7 epoch 33 检查点，无训练。分层阈值沿用训练集 p90；bootstrap 2,000 次。
- **基线核验：**Q1 valid 2,313 条，按 `sample_index` 对齐 C2.1b eDOS 记录后，重算 oracle R² 的最大绝对差为 `3.81e-6`，低于 `2e-5` 容差。
- **总体指标：**Q1 valid B7 eDOS oracle 中位 R²/失败率为 `0.5150/6.27%`，blind 为 `0.4731/9.12%`；这是 valid 诊断值，不替代 test 基线。
- **粗糙度：**训练集 p90=`0.351600`，valid 高组 205 条。oracle 中位 R²差（高组−其他）为 `−0.1812`（95% CI `−0.2407…−0.1256`），失败率差为 `+5.97pt`（`+1.48…+10.72pt`）。仅保留全 coverage 样本时，中位差仍为 `−0.1373`（`−0.2034…−0.0782`）。
- **熵：**训练集 p90=`0.9586`，valid 高组 201 条。oracle 中位差为 `−0.1669`（`−0.1938…−0.1180`），失败率差为 `+9.48pt`（`+4.48…+14.74pt`）；全 coverage 中位差为 `−0.1694`（`−0.1961…−0.1211`）。
- **Peak-share：**训练集 p90=`0.091973`，valid 高组 239 条。oracle 中位差为 `−0.0648`（`−0.1285…−0.0123`），但失败率差 `+2.81pt` 的区间 `−0.81…+6.69pt` 跨零；因此 valid 对其中位差有信号，对失败尾部则未复现清晰差异。
- **Blind gamma：**gamma 绝对误差与 oracle–blind gap 的 Spearman `ρ=0.620`（test `ρ=0.635`）。按训练集 p90 分层时，高组相对其他组的 gamma 绝对误差中位差：粗糙度 `−0.0016`（95% CI `−0.0125…+0.0112`）、熵 `+0.0114`（`−0.0004…+0.0225`）、peak-share `+0.0042`（`−0.0074…+0.0130`）；均未显示稳健分层。高熵组 blind gap 中位差为 `+0.0259`，但不能据此归因于 gamma 校准。
- **结果文件与测试：**`results/edos_error_attribution_q1_valid_samples.csv`、`results/edos_error_attribution_q1_valid_strata.csv`；局部审计单测 `python3 -m unittest tests.test_edos_error_attribution`（6/6），`ruff check tools/eval/edos_error_attribution.py tests/test_edos_error_attribution.py` 通过。全库测试见本次结束验证结果。

## 结论

- **状态：closed（valid 复核）；根因仍未证实。**test 与 valid 均支持高粗糙度/高熵谱形更难预测；peak-share 的失败率关联在 valid 上不稳。gamma 误差整体伴随 blind gap，但没有证据表明复杂谱形分层主要由 gamma 误差造成。
- **边界：**这是同一 Q1 切分体系下对同一检查点的 test/valid 诊断，不是独立外部泛化验证。分层关联不代表因果；bootstrap 区间未作多重比较校正，不据此直接选择模型模块或损失改动。

## 交接

- **下一项关卡工作：**先提出并评审一个可证伪的单因素机制设计，针对高粗糙度/高熵谱形；明确成功指标和失败保护线后，才讨论 10 epoch 成对 pilot。当前不启动训练。
- **对 status、backlog 和 decisions 的更新：**更新 `docs/status.md` 的 eDOS 当前关卡与下一候选；暂不改 `docs/decisions.md`，因为尚无可复用的因果或方案胜出结论。
