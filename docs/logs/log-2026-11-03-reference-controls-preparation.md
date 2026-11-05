# 日志：2026-11-03 — 训练随机性、无结构与单任务对照的实施与预检

对应[设计](../design/design-2026-11-03-variance-and-reference-controls.md)。用户于同日确认三项待决定事项均按建议：
replicate 同时改变数据顺序；S0 为零局部调用；单任务 loss 为 `mean(E_t+S_t)`。
本轮只实现、测试和短时预检，**未启动正式训练**，未读 test，未提交 Git。

## 实施

改动见设计“实现与兼容”表。默认行为不变：`data_order_seed`、`single_task` 只在设置时写入训练记录；
用当前代码重建 R42（staged LR2＋E＋S ep29）的模型契约，与已保存版本逐键相同，seed42 初值哈希仍为 `924bd18a…`。
S0 相对 LR2 少 525,184 个参数（75,552,068 对 76,077,252）。
既有 `test_spectral_residual` 把“seed 43 必须被拒绝”列为断言；现在 staged LR2＋E＋S 的 seed replicate
是声明过的对照，断言改为“midpoint 下换 seed 仍拒绝、staged 换 seed 接受”。

## 冻结准备

`tools/eval/reference_controls_prepare.py` 产出 `results/runs/reference_controls_20261103/`（本地产物）：
`report.json`（7 组命令、逐组变更字段、源码与输入哈希）、`protocol.json`（与工作点逐字节相同）与源码快照。
准备时重新核验了 R42 的契约、35 轮 history 与 ep29 导出。tag 前缀 `refctl20261103`，执行顺序
`s0_s42 → s0_s43 → s0_s44 → lr2_s43 → lr2_s44 → edos_s42 → phdos_s42`。每组的训练记录相对 R42
只变动设计声明的字段（seed、data_order_seed、local_route/local_depth、single_task、selection_metric、protocol 路径）。

## 验证

- 新增 `tests.pipeline.test_reference_controls` 7 项，覆盖：配置的接受与拒绝；契约隔离，以及默认记录不变；
  S0 逐张量复制同 seed 的 LR2，几何变化时输出逐位不变；replicate 初值确定且与 seed42 不同；
  单任务 loss 等于评估误差加 S，且另一任务读出无梯度；参照选点忽略保护线但核验记录；sampler 默认顺序不变。
- 相关 6 个测试模块 49 项通过。全仓 `check-static.sh`：Ruff 通过，594 项测试中唯一错误是已知缺失的旧 Q1
  数组 `data/train4ARPAT/train/train_index.npy`，与本次改动无关。
- GPU 预检 `results/runs/reference_controls_preflight_20261103/report.json`（V100，本地产物）7 组全部通过：
  seed42 单任务初值等于 R42；replicate 初值互不相同；数据顺序按声明改变（seed 43/44 与历史顺序不同，
  其余相同）；两次一次性更新有限；lr=0 时训练目标与评估误差差 ≤1.3e−7；单任务未监督读出的梯度恰为 0。
  预检用的临时 tag 目录已移除。

| 组 | 两批一次性更新（s） | 峰值分配（MiB） |
|---|---|---:|
| S0（3 组） | 0.27–1.39 / 0.19–0.31 | 3,643–4,015 |
| LR2 seed43/44 | 0.84 / 0.67–0.69 | 8,471–9,452 |
| eDOS/phDOS 单任务 | 0.64–0.65 / 0.71 | 7,112–7,515 |

首批含冷缓存与初始化开销，不代表整轮吞吐，也不代表整轮峰值（历史 LR2 整轮峰值约 24.6 GiB）。
按历史 LR2 每轮约 417 s、S0 单步约为 LR2 的 0.3 倍估计：S0 每组约 1.4 h，其余每组约 4.6 h，
合计约 23 GPU 小时。正式训练按设计交由执行 agent 完成。

分析入口 `tools/eval/reference_controls_analyze.py` 已用 R42 导出自检：逐材料 blind E 的重算与导出的相对差 ≤2.6e−7；
联合 E 0.271915、失败数 174/72，均与正式结果一致。
