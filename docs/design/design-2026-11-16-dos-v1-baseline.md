# 设计：dos_v1 的 h2 基线

创建：2026-11-16。状态：已被用户批准的[100轮新方案](design-2026-11-16-dos-v1-h2-e100.md)取代，35轮执行提示词不再执行；以下保留历史准备与短检记录。
对应[当前状态](../status.md)；[准备结果](../logs/log-2026-11-16-dos-v1-baseline-preparation.md)。

## 目标与固定配方

建立最新数据口径上的性能起点，不把跨版本分数差解释成数据扩充的因果收益。
使用 `data/datasets/dos_v1/views/dos_v1/` 的 train/valid，不读 sealed test。
两任务为 eDOS 与 phDOS，blind 模式（谱形与总量均由结构预测）。

模型为 h2：M1、ZP、staged LR2、局部 2 次／全局 6 层、共享 decoder 6 层、
phDOS 头 2 层、q1_coord、hidden 128、E＋S、每原子总量。
seed42，35 轮，batch32，lr5e-5，FP32；不加 MACE、扰动、额外候选或其他 seed。
从头初始化，旧 checkpoint 不复用。其余固定项见有效配置与训练默认模板。

训练集各任务的有效标签分别拟合归一化常数，验证集各任务仅在其有标签的样本上求平均，再等权相加。
新数据尚无保护参照，所有轮次均可参与选点，联合 E 最低者为 best，并列取较早轮次。
不套用旧 Q1 的保护线。完成后报告各任务 E 均值／中位／P90、R² 中位／P10、失败率，
附来源、虚频与争议标签分组；从导出的完整 valid 原谱独立重算 E，并核对 history 选点。
本次仅建立基线，不据单 seed 结果宣称优于旧数据模型。

## 准备与执行边界

- 准备工具：`tools/eval/dos_baseline_prepare.py`；监督工具：`tools/eval/dos_baseline_execute.py`。
- 准备目录：`results/runs/dos_v1_baseline_20261116/`，含 `report.json`、`protocol.json`、
  `command.txt`、`source_snapshot/`、`preflight.json`。原始命令以 `command.txt` 为准。
- 唯一标签：`_sbal_dosv1base20261116_h2_s42_e35`。
- 计划训练产物：`results/runs/ablation_m1_sbal_dosv1base20261116_h2_s42_e35/`。
- 验证导出复用 `tools/eval/lg_valid_export.py` 的 `export` 函数，指定 `spectra_out` 保存原谱。
- 正式训练前必须通过预检；当前 `preflight.json` 为失败，监督工具将拒绝启动。
- 经用户决定解决显存问题后，生成全新准备记录和标签，不覆盖本次失败证据；完成短检再交执行 agent。
- 监督工具采用独占锁、进程组、独立日志、10 秒脚本心跳与异常检查；agent 按事件唤醒，约 30 分钟保底检查。
  OOM、非有限值、契约不符停下报告，不自动重跑。外部中断恢复须检查进程已停、latest/history/优化器/RNG 与原契约，
  显式使用 `--resume`。脱离会话运行，不能依赖 agent 存活。

初次准备的成本预检未完成，不能据此估时。

## 同日追加：获准的较小计算分块

用户选择保持有效 batch32、缩小计算分块。新显式 `pair_budget=51200`，替代默认 batch32 时的 204800；
该预算是每块补齐后的原子对数量上限（B×L²），不是固定材料数。160 原子晶体一块最多 2 个。
沿用现有梯度累积，每个有效 batch 只更新一次；部分标签按整批的各任务标签数作分母。
dropout 随机数消费可能不同，不声明与旧分块轨迹逐值相同。

新准备目录 `results/runs/dos_v1_baseline_pb51200_r2_20261116/`，
唯一标签 `_sbal_dosv1basepb51200r2_20261116_h2_s42_e35`；原始命令、报告、协议、源码快照与 `preflight.json` 均在此目录。
对应模型目录 `results/runs/ablation_m1_sbal_dosv1basepb51200r2_20261116_h2_s42_e35/`。
旧准备全部保留，不覆盖。

v2 协议允许显式预算；其实际值仍绑定训练配置和数据/checkpoint 契约。
旧版 backbone_reference 仅描述原模型，保留历史默认计算预算，不伪称该历史参照使用新预算。
v1 的冻结限制不变。验证导出传递保存配置中的同一预算。

预检通过：普通 20 批×2 次临时更新，最大 32 晶体组合分为 16 块（每块 2 个），
峰值 15668.7 MiB；训练与验证联合 E 差 2.22e-7。
普通批次冷／热约 0.441／0.401 秒，1816 批／轮，纯训练外推 35 轮约 7.07 小时。
此估计不含验证、存盘与全量冷缓存成本；短检不保证全程显存与吞吐。

长时执行入口为 `tools/eval/dos_baseline_execute.py --preparation-dir results/runs/dos_v1_baseline_pb51200_r2_20261116`，
由执行 agent 在独立 tmux 会话中启动。执行 agent 负责训练、valid 原谱导出、独立复算和最终汇报；
不增加 seed 或候选、不读 test、不修改本次科学配方。
