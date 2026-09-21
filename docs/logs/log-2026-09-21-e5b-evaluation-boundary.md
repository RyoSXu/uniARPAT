# 工作日志 2026-09-21：E5b 评估边界审计

## 范围

- **目标：**判断生产 `evaluate_split` 与历史 `basemodel.test_one_step` 是否是可无行为合并的重复评估代码。
- **方法：**只读审计二者的调用面、归一化／反归一化、指标和产物；不修改模型、评估或训练。

## 证据

- 生产 runner 只调用 `evaluate_split`。它在 SumNorm 下将 logits softmax 为分布、以 oracle 总量恢复物理谱、生成逐样本 CSV，并在需要时计算 Cv。
- `test_one_step` 仅由 `tools/legacy/train.py`、`tools/legacy/test.py`、`tools/legacy/test_cif.py` 调用；它保留 raw-logit NormMAE、旧 M5 分支和 `dosdata/` 导出。
- 两条路径唯一共用的逐样本 MAE/MSE/R²计算已由 `utils.metrics.per_sample_spectral_metrics` 提供；其余语义不同。
- 调用面由 `rg` 审计；E5a 后全套 83/83 测试通过。

## 结论

- **状态：E5 完成。**不合并生产与历史评估路径：强行提取会把 legacy raw-logit 指标或生产 SumNorm 物理恢复语义带入对方，违反本项“无行为变化”约束。
- 生产评估边界为 `run_ablation_experiments.py:evaluate_split`；历史兼容边界为 `tools/legacy/` 加 `basemodel.test_one_step/test`。共同的基础指标函数保持在 `utils.metrics`。

## 后续

- 下一独立工程项为 E6 分桶 batch；先设计 padding 比率、等价性和资源门禁，AMP 保持可选而非必选。
- 状态页和决策页已更新；本阶段没有代码行为改动。
