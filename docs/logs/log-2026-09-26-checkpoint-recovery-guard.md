# 日志：2026-09-26 — 训练恢复失败保护

## 范围

- 经马尚酱确认，先修复近期审阅发现的恢复失败后仍继续训练、提前覆盖配置的问题。
- 修改 `run_ablation_experiments.py` 的 `train_and_eval`，扩展现有
  `tests/test_e5_checkpoint_boundary.py`，同步 E5 设计与状态记录。
- 保留开始时的未提交文档与诊断 CSV；生产模型、损失、checkpoint 格式和学习率策略未修改。

## 变更与证据

- **恢复调用链：**`train_and_eval` 仍调用唯一的 `restore_ablation_checkpoint`；checkpoint／history
  恢复异常现在记录原因并重新抛出，终止本次运行，不再转为同目录新训练。
- **产物顺序：**配置写入在恢复成功之后；已有兼容恢复配置原样保留，包括 slope 校准记录。
  新实验及配置缺失的兼容恢复仍可生成配置。
- **测试符号：**新增 `TestRunnerRecovery`，使用临时目录、微型 Linear＋AdamW、模拟 batch 与
  训练步，调用真实 runner 控制流；复用既有 `_parts` 和 `_Scaler`。
- **失败验收：**AMP 不匹配、slope 不匹配、缺失 slope 校准、checkpoint 损坏、权重形状不兼容、
  history 缺少必要列六种情况，均确认训练／评估零调用，config／history／best／latest 字节不变。
- **成功验收：**旧 FP32、AMP、slope 三种恢复均只执行下一 epoch，恢复模型／optimizer 状态，
  分别保留 scaler 或 slope 校准；新实验正常写出配置、history 和 checkpoint。
- **执行结果：**
  - `python3 -m unittest tests.test_e5_checkpoint_boundary tests.test_edos_slope_loss`：23/23 通过。
  - `python3 -m unittest discover tests`：133/133 通过，33.3 秒。
  - `python3 -m ruff check run_ablation_experiments.py tests/test_e5_checkpoint_boundary.py`：通过。
  - `git diff --check`：通过。

## 结论

- **状态：closed（本项修复完成）。**已有恢复校验能在 runner 层阻止失败后继续训练及覆盖产物。
- 验证覆盖上述失败类型与成功路径；不代表任意配置差异都已获得检测。既有 optimizer 状态恢复的
  宽容回退按 E5 契约保留，本轮未扩大为完整实验配置认证。
- 测试使用隔离的微型模型和文件；没有启动 Q1 训练或写入生产 checkpoint。

## 交接

- 下一项：执行用户已确认的冻结 G2 结构信息通路诊断，先记录诊断设计和证据分支。
- 更新 `docs/status.md`，解除本项恢复保护关注点。科学决策保持原状。
