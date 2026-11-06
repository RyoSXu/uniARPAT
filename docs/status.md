# 当前项目状态

更新：2026-11-06。只记录大事件与当前判断；细节见链接。[接手步骤](handoff.md) · [固定契约](decisions.md) · [证据导航](index.md)

## 目标与工作点

从晶体结构联合预测 eDOS 与 phDOS，两任务等权，目标是提升模型性能、设计与方法（含数据）。
工作点：**LR2＋E＋S，staged L2/G6，best ep29**（2026-10-31 确认），配置见 [decisions.md](decisions.md)。

| 模型（修复版 Q1，完整 valid，blind） | 联合 mean E↓ | eDOS 中位 R² | phDOS 中位 R² |
|---|---:|---:|---:|
| A64 E＋S（对照） | 0.2780 | 0.489 | 0.736 |
| LR2 E＋S（工作点） | 0.2719 | 0.491 | 0.743 |

改善 0.006，配对区间不含 0；同组成谱差未见改善。[结果与采用](logs/log-2026-10-30-spectral-residual-run.md)

## 大事件

| 日期 | 事件与结论 |
|---|---|
| 10-23～10-31 | 新目标 E＋S 与 blind 每原子总量；LR2 采用为工作点。[目标](design/design-2026-10-23-balanced-blind-objective.md) |
| 10-28～11-02 | 双 decoder、声子物理头、midpoint 编排均未采用；E(3)/晶胞一致性与截断边界检查结束。[证据导航](index.md) |
| 11-03～11-04 | 同配方三 seed 联合 E 极差 0.0013；三 seed 集成 0.257，大于任何架构改动的收益；去掉结构（S0）后变差，结构贡献在 phDOS 上更明显。[结果](logs/log-2026-11-03-reference-controls-run.md) |
| 11-05 | 结构利用诊断：瓶颈是泛化（train/valid 误差约差 2 倍），不是结构算子缺信息；标签噪声只占小部分。[结果](logs/log-2026-11-05-structure-use-audit-run.md) |
| 11-05～11-06 | 泛化正则单 seed 筛选：G（几何扰动）改善 0.0021，属“效应太小”，列为最终配方备选；D（局部丢弃）变差 0.0049，结束。[方案](design/design-2026-11-05-structure-generalization.md)，原始结果 `results/runs/structure_generalization_20261105/analysis/screen.json`（运行日志未写） |
| 11-06 | 改为单 seed 流程，多 seed 只在定稿时做一次（[规则](decisions.md#历史参照与判读)）；长时实验改为脚本监控、agent 按事件唤醒（[AGENTS.md](../AGENTS.md)）。 |
| 11-06 | 学习曲线：**数据量远未饱和**。联合 E 25%/50%/100% = 0.3527/0.3138/0.2719，50%→100% 下降 0.042（门槛 0.004）；eDOS 降 0.037、phDOS 降 0.047，均远超各自门槛；log-log 斜率约 0.2（phDOS 0.29 > eDOS 0.15）。[方案与结果](design/design-2026-11-06-learning-curve.md) |

## 当前判断与下一步

- 模型细节的改动收益已接近噪声；学习曲线确认数据量是主要瓶颈（数据翻倍的收益远大于任何架构改动和三 seed 集成）。
- **下一步**：投入数据或预训练，具体路线待定。
- g_s43（G 的第二个 seed）于 2026-11-06 在第 25 轮手动停止（用户决定，按单 seed 流程不再确认），未导出；latest 为第 24 轮，checkpoint 保留。其 execution 状态文件停在 running，不代表仍有任务。

## 长期风险

- 数据：输入结构与 DOS 计算几何不一致（来源泛函不同）、5 条电子谱待核、N_val 来源差异；正式评分保留全部材料与完整窗口。[数据页](data.md)
- 历史 test 曾用于诊断，不能称未触碰的留出集；AMP 未验收。

新训练须另定方案；历史设计、日志和旧提示词不自动成为执行队列。
