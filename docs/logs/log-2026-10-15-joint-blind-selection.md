# 日志：2026-10-15 — 联合 blind 选点与双任务保护

用户确认长期平等考虑 eDOS/phDOS，以完整 valid 的 blind 中位 R² 为主指标；
Oracle MAE 与同组成谱差仅作诊断。当前开发基线保留 I1，本轮不启动训练或更换局部公式。

## 已实现的规则

- 新选项 `--selection_metric balanced_blind_r2`：
  `S = 0.5×blind_r2_edos_median + 0.5×blind_r2_phdos_median`，**取最大**。
  使用两任务各自中位数的平均，不先对每个材料平均；不截断负 R²。
- 初始最佳值为负无穷，分数严格更高才保存 best；并列保留较早轮次。
  恢复、汇总、导出和预算核验共用方向及历史公式检查。
- 日志以 blind R² 为主，另报两任务 blind 失败率；完整汇总保留各任务、模式的 R²/MAE/失败率。
  `balanced_score` / `best_val_score` 保存原始 S，不取负号；契约明确记录 `mode=max`。
- 新对照报告按材料配对重采样，两个任务与两个模型共用同一抽样索引，给出联合分数差的 95% 区间。
  分别约束两任务：中位 R² 下降≤0.02、失败率增加≤1 pp，保护线使用点估计。
  联合区间下界>0、双任务保护均满足且新选点契约一致，才标为满足数值采用条件；不会自动替换基线。
  区间只反映所选 checkpoint 的材料抽样不确定性，不代表跨训练种子稳定性。
- 新旧选点混用时仍可看诊断数字，但不标为可采用；需要用同一选点、数据和预算完成新对照。
  旧 I1/I2/I3 的权重、history、契约及历史结论保持原样，不能重新标成 blind 最优 checkpoint。

## 实现与调用链

- `utils/checkpoint_selection.py`：`BLIND_R2_METRIC` / `SELECTION_METRICS` 定义选项；
  `blind_joint_score` / `validation_score` 计算分数；`selection_mode` / `initial_best_score` /
  `is_better_score` 处理方向、首轮和并列；`selected_history_row` 校验并选择历史轮次；
  `selection_contract` 记录新协议 `joint-blind-r2-selection-v1`。
- `utils/local_global_experiment.py` / `utils/lg_baseline_experiment.py`：配置验证与契约接入。
  新选择器复用既有模型/配方约束；历史专用实验工厂保持旧协议。新旧契约拒绝互相恢复。
- `tools/train.py`：新分数写入 valid 汇总、history、best/latest，恢复沿用相同方向；
  `evaluate_local_global_split` 的历史 oracle 别名保留，实际选点只读显式 blind 列。
- `tools/eval/local_global_compare.py` / `lg_valid_export.py` / `tools/checks/check_long_budget.py`：
  `summarize` / `select_best` / 完成核验统一使用 `selected_history_row`。
- `tools/eval/lg_paired_compare.py` 的 `joint_blind_comparison` / `compare` 增加联合比较与对称保护；
  `global_norm_compare.comparison_report` 传递新主指标，历史 MAE 对照输出口径保留。
- `tools/checks/check_local_structure_controls.py` 将选择器加入源码指纹；
  `tests/pipeline/test_checkpoint_selection.py` 的 10 项回归接入现有静态检查入口。

调用链：显式选择器 → 原模型/联合损失 → 完整 valid blind 指标 → S 最大的 checkpoint →
恢复/导出一致性检查 → 材料配对联合区间与双任务保护 → 人决定是否替换基线。
模型公式、损失权重和联合训练配置不因平等选点而改变。

## 验证与边界

- 相关 24 项测试通过（2.5 s）：包括真实训练入口的合成 35 轮控制流及恢复，
  证明 oracle 更好或单任务更高不能覆盖联合最优；没有实际数据训练。
- 新旧契约隔离、负分数首轮保存、并列、非有限值、历史公式漂移、导出方向、
  两任务共用抽样、联合收益掩盖单任务 R²/失败率恶化和混合选点阻断均有回归覆盖。
- `tools/checks/check-static.sh` 通过：Ruff、语法、424 项测试（281.9 s）及 798 个候选仓库文件边界检查。
- 实际 I1/I2/I3 的 config/best SHA256 与已核验报告一致，旧契约重建相同，
  原选点仍为 ep31 / ep35 / ep35；未修改历史文件。
  新选择器的 I1 实际配置捕获与原配置逐项一致，确认模型、损失、优化器及其权重未变。
- 8 份本轮相关文档的 76 个本地引用与差异检查通过；状态页保留 47 行。

后续新实验须显式使用新选项和隔离 tag；旧历史工厂、checkpoint 与预检不自动升级。
源码指纹已变化，旧预检不能授权新源码训练。完整新规则的真实数据训练效果尚未检验。
