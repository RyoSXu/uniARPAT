# 工作日志 2026-09-21：E5a checkpoint 代码边界

## 范围

- **目标：**将 ablation runner 的 checkpoint payload、AMP 一致性检查、恢复和原子写入从训练控制流中提取为唯一实现。
- **不变项：**模型前向、损失、优化器策略、scheduler、评估、数据、结果格式和训练预算均未改变；未启动训练或读取 test split。
- **改动：**新增 `utils/ablation_checkpoint.py`；runner 改为调用其 `build`、`restore` 与原子保存函数，不再内嵌 checkpoint schema 细节。

## 证据

- FP32 payload 的键保持为 `epoch/model_name/seed/use_amp/model/optimizer/best_val_score`；AMP 仅额外保存 `amp_scaler`。
- 旧 checkpoint 缺失 `use_amp` 时按 FP32 兼容；AMP 设置不一致仍拒绝恢复；optimizer 恢复失败仍按旧行为宽容回退。
- 新增 checkpoint 边界合同覆盖：FP32 payload 与旧格式兼容、AMP 不匹配拒绝、原子临时文件替换、AMP scaler 状态 round-trip。
- `python3 -m unittest discover tests`：**83/83 通过**；`git diff --check` 通过。

## 结论

- **状态：E5a 完成。**runner 的 checkpoint 责任已独立，C4 AMP 的恢复规则不再散落在训练循环中；无数值、数据或训练行为变化。
- E5 尚未整体结束：评估边界和历史兼容层将作为后续独立阶段另立设计，避免一次重构跨越多个高风险运行路径。

## 后续

- 下一关卡为 E5b 评估边界设计：先列出 `evaluate_split` 与 `basemodel.test_one_step` 的重叠／独有职责，再决定是否能无行为合并。
- 已更新 `status.md` 和索引；无需更新科学决策。
