# 日志：2026-09-23 — eDOS slope-loss 实现与合同测试

## 范围

- **任务与假设：**按已确认设计实现默认关闭的 eDOS 一阶差分损失、一次性梯度比校准、跳过 test 的训练入口和 Q1 valid 成对判定工具；本轮不启动训练。
- **改动的文件或配置：**`model/losses.py`、`model/model.py`、`utils/experiment_config.py`、`utils/ablation_checkpoint.py`、`run_ablation_experiments.py`、`tools/eval/edos_error_attribution.py`、`tools/eval/edos_slope_pilot_verdict.py`，以及对应 checkpoint、loss、评估器测试；更新本设计、`docs/status.md` 与 `docs/index.md`。

## 证据

- **损失与校准：**eDOS logits 经 softmax 形成概率谱，只对相邻 bin 的一阶差分做 MSE；候选在固定首 batch、eval mode、无 optimizer step 下按 `ρ=0.10` 校准 λ，并将比率、λ 与梯度范数写入运行配置和 checkpoint。默认比率 `0` 不添加新损失项。
- **评估隔离：**`--skip_test_eval` 在创建 test loader 前返回空值，训练收尾分支直接跳过 test inference。成对工具固定 `split="valid"`、epoch 10，以同序材料 ID 做 paired bootstrap；脚本不提供 test 评估入口。
- **执行验证：**`py_compile` 覆盖本轮 Python 文件；`python3 -m unittest discover -s tests -p 'test_edos_slope*.py' -v` 与 `python3 -m unittest discover -s tests -p 'test_e5_checkpoint_boundary.py' -v` 通过；改动 Python 文件的 Ruff 检查、两个评估脚本 `--help` 和 `git diff --check` 通过。
- **未执行项：**未读取训练、valid 或 test 数据；未执行真实 batch 冒烟、模型训练、pilot verdict 或 test inference。因此不报告性能或准确率结论。

## 结论

- **状态：pending（代码就绪，pilot 待单独批准）。**合成输入和 checkpoint/loader 合同测试通过；真实训练 batch 与两臂结果尚未验证。单测通过不代表 accuracy 假设成立。
- **边界：**不改 `docs/decisions.md`，因为新损失尚无实验结论；本轮没有创建 pilot 结果 CSV 或 checkpoint。

## 交接

- **下一项关卡工作：**若获单独授权，先检查 `_edoslpctl`、`_edoslp` 标签及输出目录未占用；执行无 optimizer step 的校准 batch 冒烟后，再启动 Q1 M1×10 成对训练。训练后只用 Q1 valid epoch 10 checkpoint 运行 `tools/eval/edos_slope_pilot_verdict.py`。
- **对 status、backlog 和 decisions 的更新：**`status.md` 已改为“代码就绪、pilot 待单独批准”；索引和设计页已同步；不写入已验证精度结论。
