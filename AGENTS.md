# AGENTS.md — 给 agent 的导航页（稳定约定，不记流水）

> 活状态看 `docs/STATUS.md`（3分钟读完），再看本页。流水记工作日志，结论进 Backlog/Decisions。

## 1. 这是什么
uniARPAT：由晶体结构端到端预测 eDOS + phDOS。当前默认：M1 共享骨干 + sumnorm-KL/W1/Huber + H1 η/γ 盲推头 + dropout 0.05 + Q1 干净池。唯一合法对照：B7 `_e9ctl`（Q1 口径，e med 0.518/fail 5.73%，p med 0.741/fail 3.50%）。

## 2. 先读什么
| 改哪里 | 先读 |
|---|---|
| 任何任务 | `docs/STATUS.md` → `docs/INDEX.md` → `docs/GLOSSARY.md` |
| 跑实验/改模型 | `README.md` Quickstart + `docs/01_开发文档/待开发总清单Backlog.md` Phase 表 |
| 数据口径疑问 | `docs/01_开发文档/待开发总清单Backlog.md` Q1 条 + `index/z0_REPORT.md` D1–D4 |
| 术语（med/fail/gap/park/pre-Q） | `docs/GLOSSARY.md` |

## 3. 常用命令
```bash
python3 -m unittest discover tests        # 全套单测（截至09-18为34项，约1分钟）
python3 run_ablation_experiments.py --model M1 --epochs 35 --tag _e9ctl   # B7对照配方
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _xxx      # pilot初筛
```
禁：裸跑 `--model all --epochs 100`（会污染成绩）；跨归一化复用 checkpoint（见 Backlog 跨归一化对口径）。

## 4. 目录地图
- `model/` 骨干+头、`datasets/` 数据集、`utils/` 特征/调度、`tools/getdata/` 抓数加工、`tools/eval/` 评估脚本
- `configs/config.yaml` 被 runner 改写，唯一可复现的是 `output/ablation_*/config_used.yaml` + `results/history_*.csv`
- `data/train4ARPAT/` Q1干净缓存（18706/2313/2287），旧缓存 `data/archive/*_preQ1/`，`output/` 不入库
- `docs/` 见 `docs/INDEX.md`；中文历史目录名冻结保留，只加英文索引不改名

## 5. 实验纪律
- 单因子、一臂一 verdict；10轮 pilot 胜者才进长跑；等算力对照；盲推报 gap 分布 p50/p90/p99。
- 数字后缀模板：`e med/fail + p med/fail + (test|valid, epN, oracle|blind, Q1|pre-Q)`。
- 五态：合并(win)/park(打平留代码默认off)/挂起(待长跑)/死刑(永不重做)/关闭(可重开)。详见 `docs/GLOSSARY.md`。

## 6. Session 协议
- 开工：读 `docs/STATUS.md`，认领下一棒，检查上一 verdict 是否落盘（Gate铁律）。
- 收尾必做：工作日志 `docs/03_工作日志/日志-YYYY-MM-DD-<主题>.md`（改了什么/实测/verdict/下一步）+ 更新 `docs/STATUS.md` + 结论同步 Backlog。
- 长跑用 `setsid+nohup`，产物 `results/history_*.csv + test_*_summary.csv + samples_*.csv`，verdict 脚本入库 `tools/eval/` 不放 `/tmp`。
