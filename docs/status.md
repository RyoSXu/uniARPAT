# 当前项目状态

更新：2026-11-03。[接手步骤](handoff.md) · [固定契约](decisions.md) · [完整证据导航](index.md)

## 目标与工作点

固定 eDOS/phDOS 联合训练、两任务平等考虑，研究结构差异能否转化为可泛化的谱预测收益。
当前采用 **完整 A64 的 LR2 形式＋E＋S，staged L2/G6，best ep29**（用户于 2026-10-31 确认）。
模型结构、loss、选点及数据契约集中在 [decisions.md](decisions.md)，具体产物位置见 [handoff.md](handoff.md)。
这是研究工作点；I1 与旧模型的历史性能参照、原 checkpoint 契约继续保留。

## 当前主结果

修复版 Q1、完整 valid 2,313 条、blind；两组都用同一冻结协议、35 轮预算与选点规则，seed42。
E 为按 train 谱能量归一化的每原子全谱平方误差，越低越好；失败定义为每材料 R²<0。

| 模型／选中轮 | 联合 mean E↓ | eDOS 中位 R² | phDOS 中位 R² | eDOS／phDOS 失败数 |
|---|---:|---:|---:|---:|
| A64 E＋S／25（匹配对照） | 0.27795645 | 0.489122 | 0.735590 | 171／74 |
| LR2 E＋S／29（工作点） | 0.27191453 | 0.491335 | 0.743409 | 174／72 |

联合 E 差（LR2−对照）为 −0.00604192，材料配对 95% 区间 [−0.01163879, −0.00102599]；
通过预设改善判据及固定双任务保护。eDOS 失败多 3 条、phDOS 少 2 条，R² P10 均下降，尾部没有一致改善。
同组成 absolute/reduced × eDOS/phDOS 四项谱差误差区间均含 0，**尚不能宣称结构谱差泛化改善**。
已独立复跑完整 valid 并从原谱复算。证据限本次已选 checkpoint 和单 seed，不能表示跨 seed 稳定性。
[正式结果、核验与采用决定](logs/log-2026-10-30-spectral-residual-run.md)

## 已完成的阶段判断

| 问题 | 已知结果与边界 |
|---|---|
| 局部更新与骨架退化 | 既有对照支持局部结构信息有价值；全局归一化、注意力 dropout、投影初始化三项归因已结束。保留 I1 历史性能参照。[证据入口](index.md#当前问题的证据) |
| A/B+ 与局部细节 | 暂定完整 A 研究路线，未证明总体优于 B+；扩宽、内外 norm 和增量读出均未达到各自采用条件。[局部对照](index.md#模型设计) |
| 新目标与 LR2 | 修正 H1 的科学范围，采用无固定上限每原子总量、等权 E＋S；LR2 迁移后通过本次总体改善判据。[目标](design/design-2026-10-23-balanced-blind-objective.md)、[谱形辅助](logs/log-2026-10-26-spectral-shape-aux-run.md) |
| 中段编排 | `L1 G1 G2 G3 L2 G4 G5 G6` best ep24 对 staged ep29 的联合 E 与四项谱差区间均含 0；不升级工作点、不自动追加扫描。[结果](logs/log-2026-11-01-local-global-layout-run.md) |
| 双 decoder、声子物理先验加读出修正 | 两项均未达到采用条件，候选已结束；六层共享 decoder 保留。[拆分](logs/log-2026-10-28-split-decoder-run.md)、[物理头](logs/log-2026-10-29-phonon-prior-run.md) |
| 谱差归因 | 冻结读出与组比较未定位单层瓶颈；不能推断表示没有结构信息或 train/valid 分布相同。[读出](logs/log-2026-10-27-shape-aux-stage-readout.md)、[比较](logs/log-2026-10-27-shape-gap-comparability.md) |
| 物理与晶胞一致性 | 16 条真实 train 结构的两精度有限检查通过；精确截断合成例有 FP32/FP64 硬计数反例，未证明影响真实性能。[验收与反例](logs/log-2026-11-02-crystal-model-invariance.md) |
| 真实截断边界 | 全量有效 train 中距 6 Å ≤1e-6 Å 有 3 条、≤1e-4 Å 有 104 条；后者两个固定旋转未见成员或逐原子计数变化。调查结束，暂无依据优先追加平滑计数训练。[统计](logs/log-2026-11-02-cutoff-boundary-census.md) |

## 尚未解决与下一步

- 核心问题仍是同组成谱差泛化；下一项假设、对照和预算尚未选定。头数、FFN 等工作配置不等于最优，但不以穷举为推进前提。[方向地图](design/design-2026-09-19-structural-refinement.md)
- 数据风险仍在：5 条电子谱待核材料、N_val 来源差异、coverage 含义，以及输入结构与 DOS 计算几何未逐条一致；正式评分保留全部材料与完整窗口。[语义审查](logs/log-2026-10-22-spectral-contract-audit.md)、[几何审查](logs/log-2026-10-14-i1-pair-audit.md)
- 现有两模型平均只支持误差互补，未分离架构与训练随机性；外部结构池的 2026-10-28 本机盘点不支持直接启动预训练。[误差诊断](logs/log-2026-10-28-error-structure-ensemble.md)、[结构盘点](logs/log-2026-10-28-structure-pool-inventory.md)
- E(3)、原子重排及等价晶胞目标不变；有限数值验收不等于全面保证，AMP 尚未验收。历史 test 曾用于诊断，不能称未触碰的留出集。

新训练须另定方案。长时间训练与监控先生成提示词，由用户指定执行 agent；历史设计、日志及旧提示词不自动成为执行队列。
