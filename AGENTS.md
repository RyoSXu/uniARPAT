# AGENTS.md — 项目导航

## 首先阅读

每次会话先阅读 `docs/status.md`，再阅读 `docs/index.md`；涉及术语或指标时，继续阅读
`docs/glossary.md`。模型或实验工作还要阅读 `docs/status.md` 中对应条目；数据工作还要
阅读 `docs/data.md` 和 `index/z0_REPORT.md`。

`docs/workflow.md` 是必须遵守的会话流程，其中规定了计划、决策、日志、结果和设计文档
的存放位置。

## 项目概览

uniARPAT 从晶体结构预测电子态密度和声子态密度。当前参考实验是 B7 `_e9ctl`
（Q1、M1×35、最佳 epoch 33）：eDOS 中位 R² 为 0.518、失败率 5.73%；phDOS 为
0.741、失败率 3.50%。

默认方案为 M1、总和归一化的 KL/W1/Huber 损失、H1 eta/gamma 盲推理头、0.05 dropout，
以及 Q1 清洁数据池。

## 常用命令

```bash
python3 -m unittest discover tests
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _pilot
python3 run_ablation_experiments.py --model M1 --epochs 35 --tag _e9ctl
```

未经明确批准，禁止运行 `--model all --epochs 100`。禁止在不同归一化方案之间复用检查点。
长训练使用 `setsid + nohup`；正式摘要存入 `results/`，检查点保留在 `output/`。

## 约束

- 每个实验只改变一个因素，并只给出一个结论。先做 pilot，再做等算力对照。
- 使用中位 R² 和失败率；报告数据口径以及 oracle/blind 模式。
- 平局线为 `|Δmed| < 0.02` 且 `|Δfail| < 1 个百分点`。
- 工作树有未提交改动时保留无关改动。未经逐项明确批准，不得删除检查点。
- 启动下一项实验前，确保上一项结论已写入日志、`status.md`，并在会影响后续工作时
同步写入 `status.md`。
