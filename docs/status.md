# 当前项目状态

更新日期：2026-11-02。本页只保留当前目标、工作参考、阶段结论与下一步；实验过程见[文档索引](index.md)。

## 当前目标

固定eDOS/phDOS联合训练、两任务平等考虑，研究晶体结构差异能否转化为可泛化的谱预测收益。
局部更新已有总体价值；全局骨架退化归因与本轮loss候选比较已完成。基础Encoder工作点保留，
声子物理先验加普通读出修正未达到采用条件，用户已确认结束本候选。
LR2向E＋S迁移35轮已完成，best ep29通过预设改善判据并独立核验；用户于2026-10-31确认采用为当前研究工作点。
完整模型仍须满足[E(3)输出要求](design/design-2026-10-04-e3-output-invariance.md)。

## 当前参考与口径

- **工作参考**：完整A64的LR2形式＋非负、无固定上限的每原子窗口总量＋独立谱形监督，参考checkpoint为LR2 E＋S ep29。训练为
  `L=0.5 mean(E_e+S_e)+0.5 mean(E_p+S_p)`；E是train谱能量归一化的每原子blind全谱平方误差，
  S是固定权重1的KL/W1/Huber，零谱S=0且仍接受E监督。[配方](design/design-2026-10-25-spectral-shape-supervision.md)
- **骨架与预算工作点**：ZP；完整A 64/128、内部post-norm、完整状态读出、关闭四个内部加法，保留外层pre-norm残差；
  staged L2/G6，全局post-norm、注意力概率dropout=0、Q/K/V/O恒等初值后继续学习；
  8头、2048维GELU FFN、6层共享decoder。seed42、35轮、batch32、lr5e-5、FP32。
  这是已运行研究配置，不宣称各项最优或跨seed稳定。[采用记录](logs/log-2026-10-30-spectral-residual-run.md#采用决定2026-10-31)
- **选点**：完整valid先要求eDOS/phDOS失败数≤196/99、两任务中位R²分别不低于固定旧A64−0.02，
  再最小化等权mean E；S不参与选点。无合格轮次报告`no_eligible_best`，latest仅供恢复。
  分别报告E均值/中位/P90、R²中位/P10及失败率；Oracle与同组成谱差只作诊断。
- **数据**：修复版`data/crystal_q1_v1/`，有效train/valid/test=18,704/2,313/2,287；
  网格E0/P0、完整原子、coverage不参与当前loss，新blind重建不使用N_val侧车。[数据约定](data.md)
- **历史参照保留**：I1及同数据重训原Encoder用于性能/架构对照，原A64 E＋S ep25保留为本轮匹配对照，旧H1 A64/LR2保留原选点与契约。
  ZP100/B7是旧Q1和不同预算参考，不作同数据因果对照。工作配方更新不等于性能基线替换。
- **代码入口**：工作配方须显式指定`--local_route LR2`、`per_atom`、冻结protocol、`--spectral_shape_aux`和`balanced_blind_energy`；
  裸训练入口仍保留历史默认，不能当作获准研究配方。不同目标/总量/选点契约不混用checkpoint。

## 已知结论

固定完整valid、blind；纯E与E＋S使用相同归一化及选点，旧A64按历史blind中位R²选点；LR2 E＋S与A64 E＋S ep25配对：

| 模型／选中轮 | 联合mean E↓ | eDOS中位R² | phDOS中位R² | eDOS／phDOS失败率 |
|---|---:|---:|---:|---:|
| 旧A64／31 | 0.27848 | 0.4817 | 0.7407 | 8.47%／4.28% |
| 纯E／32 | 0.28490 | 0.4621 | 0.7218 | 7.91%／2.72% |
| E＋谱形辅助／25 | 0.27796 | 0.4891 | 0.7356 | 7.39%／3.20% |
| LR2 E＋S／29 | 0.27191 | 0.4913 | 0.7434 | 7.52%／3.11% |

谱形辅助相对纯E联合误差差为−0.00695，材料配对95%区间[−0.01130, −0.00220]，支持本次改善；
phDOS失败率则增加0.476 pp。相对旧A64，误差点差−0.00052、区间含零，只称恢复到参考附近，
微小点差约97%来自5条待核材料，正式评分仍全部保留；不宣称突破、统计等效或跨seed稳定。
第9轮后中断并按原契约恢复，原因未知，35轮及结果已独立核验。
实现与本次结果已提交`66f2011`。[正式结果与边界](logs/log-2026-10-26-spectral-shape-aux-run.md)

LR2 E＋S ep29相对A64 E＋S ep25的联合mean E差为−0.00604，材料配对95%区间[−0.01164,−0.00103]；双任务固定保护均通过，符合本次改善判据。完整valid独立复跑与原谱复算已核验。eDOS失败数较对照多3条、phDOS少2条，两任务R² P10点估计均下降；同组成谱差四项区间均含0。用户已采用为研究工作点，证据仍限本次单seed总体改善。[正式运行与核验](logs/log-2026-10-30-spectral-residual-run.md)

LR2＋E＋S中段编排（L1 G1 G2 G3 L2 G4 G5 G6）对staged ep29的seed42、35轮对照已完成；midpoint best ep24通过双任务固定保护。联合mean E差为−0.00152，材料配对95%区间[−0.00532,+0.00210]含0；同组成absolute/reduced eDOS/phDOS四项误差差区间均含0。2026-11-02已独立复跑完整valid并从原谱、晶体组成复算区间，结果吻合。按预设判据不升级工作点、不追加扫描；staged保持。[正式运行、分析与独立核验](logs/log-2026-11-01-local-global-layout-run.md)

| 研究项 | 当前结论 |
|---|---|
| 局部A/B+ | 暂定完整A研究路线；没有证据证明A总体优于B+，实现均保留。[选型](logs/log-2026-10-16-local-module-run.md) |
| 宽度、内外归一化、增量读出 | W1/LN1/LN2/LR1均未达各自升级条件；64/128、内部post-norm、外层pre-norm和完整状态读出保留。[索引](index.md#模型设计) |
| 内部残差LR2 | E＋S迁移ep29通过本次联合误差判据及双任务保护，用户于2026-10-31确认采用；不据此宣称同组成谱差改善。[旧结果](logs/log-2026-10-22-local-internal-residual-run.md)、[设计](design/design-2026-10-30-spectral-residual.md)、[正式结果](logs/log-2026-10-30-spectral-residual-run.md) |
| 全局模块与编排 | 三项骨架修正已有证据；头数、FFN及其他编排尚未选优，工作点不等于最优。[路线图](design/design-2026-09-19-structural-refinement.md) |
| LR2＋E＋S中段编排 | midpoint ep24相对staged ep29未达到总体升级判据，同组成谱差四项区间均含0；staged工作点保留，不追加预算。[设计](design/design-2026-11-01-local-global-layout.md)、[正式结果](logs/log-2026-11-01-local-global-layout-run.md) |
| 结构谱差 | 总体恢复没有解决同组成eDOS谱差。冻结逐阶段读出看到三层表示都分开，valid上线性探针和模型输出未检出优于零响应，不能指定单层瓶颈。有限组比较未检出验证组相对全部训练组的结构差明显更小，未定位原因；本轮诊断结束。[读出](logs/log-2026-10-27-shape-aux-stage-readout.md)、[比较](logs/log-2026-10-27-shape-gap-comparability.md) |

## 尚未解决与下一步

1. 双 decoder 已跑完且不采用。best 为第 24 轮，联合 mean E 点差 −0.00130，配对区间含零。
   共享 6 层 decoder 仍是工作点；不自动改成部分拆分或更深的共享 decoder。
   [结果](logs/log-2026-10-28-split-decoder-run.md)
2. 数据风险继续单列：5条电子谱待核材料、N_val来源差异、coverage物理含义及输入／DOS计算几何不一致；
   未删除标签、改mask或重分箱。[语义审查](logs/log-2026-10-22-spectral-contract-audit.md)、[几何与来源](logs/log-2026-10-14-i1-pair-audit.md)
3. 完整模型等价晶胞／超胞一致性、全面对称性验收与AMP仍未完成。
4. 结构预训练还不能开始：本机只有 Q1 的 23,306 条几何，没有额外结构。
   去掉 valid/test 的 ID 和绝对同组成后，剩下的仍是现有 train。
   [盘点](logs/log-2026-10-28-structure-pool-inventory.md)
5. 已有两份模型的预测平均，联合mean E降0.0098，区间不含零；双decoder单独的本次点差为−0.0013。
   这支持不同模型的误差互补，不能据此估计种子方差或认定随机性主导；架构与训练随机性仍未分离。
   [诊断与复核](logs/log-2026-10-28-error-structure-ensemble.md)
6. 纯物理头首轮第 8 轮的数值缺陷已修复；「物理先验加学习修正」按预检命令完成35轮，best为第26轮。
   相对谱形辅助ep25，phDOS mean E差+0.00598（配对95%区间[+0.00187,+0.01022]），联合差+0.00327
  （[−0.00191,+0.00812]，含0），不称改善；用户已确认结束本候选。eDOS、失败率及去均值logits后的修正谱形TV见
   [正式运行与核验记录](logs/log-2026-10-29-phonon-prior-run.md)。模型选择只用valid，未评估test。
   独立复算亦未检出同组成eDOS谱差改善，phDOS两种组成口径的谱差误差均增加；仅作诊断，不改变选点。
   设计、修复、短检和接续核验见[设计](design/design-2026-10-28-phonon-lattice-head.md)、
   [中断与修复](logs/log-2026-10-28-phonon-head-run-failure.md)、
   [修正版短检](logs/log-2026-10-28-phonon-prior-preparation.md)、
   [接续记录](logs/log-2026-10-28-cursor-takeover-review.md)。

模型选择只用train/valid；历史test曾用于诊断，不称完全未触碰的留出集。
旧实验结论限各自数据、目标、选点和预算；历史设计或提示词不自动成为当前执行指令。
