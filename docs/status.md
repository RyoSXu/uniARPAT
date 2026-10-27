# 当前项目状态

更新日期：2026-10-26。本页只保留当前目标、参考模型、关键结论和下一步；历史过程与验收细节见[文档索引](index.md)。

## 当前目标

固定 eDOS/phDOS **联合训练**，两类谱平等考虑。用户于2026-10-23确认新协议：训练与完整valid选点同用**按train谱能量归一化的每原子blind全谱平方误差**，两任务各50%；选点先保护两任务失败率，保留原R²下降≤0.02保护线，再最小化联合误差。完整R²/P10/失败率仍报告，Oracle与同组成谱差仅作诊断。新协议已有首组正式训练结果；历史选点不追溯改写。[公式、兼容与验收](design/design-2026-10-23-balanced-blind-objective.md)、[正式运行与核验](logs/log-2026-10-23-balanced-blind-run.md)
最终两类谱须满足 E(3) 不变性，原子重排与等价晶胞另作一致性检查。要求见[对称性设计](design/design-2026-10-04-e3-output-invariance.md)。

2026-10-25固定谱形辅助候选已完成：相对纯 E ep32 的联合 blind mean E 改善`−0.00695 [−0.01130, −0.00220]`；相对旧 A64 ep31 点差`−0.00052`但区间含零。单 seed 与材料抽样区间不代表跨 seed；辅助改善不等于 Encoder 定型，不自动替换 I1/A64/LR2。[正式结果](logs/log-2026-10-26-spectral-shape-aux-run.md)

**上一阶段的全局骨架归因已结束。**保留局部更新，沿用已有三项全局修正证据。用户于 2026-10-16 确认转向基础 Encoder 定型：先审查局部 A/B+ 选型、内部宽度、更新公式和残差连接，再确定全局模块与整体编排；I4 等增强暂不扩展。
用户已暂定完整A作为局部研究路线，接受其较高成本；I1继续作为性能参照，局部更新的已有价值无需重新证明。[匹配选型](design/design-2026-10-16-local-module-selection.md)已完成，不宣称A总体优于B+，也不替换I1。

## 当前参考与口径

- **数据**：修复版 `data/crystal_q1_v1/`，实际 train/valid/test 视图为 18,704 / 2,313 / 2,287；字段和历史数据边界见[数据约定](data.md)。
- **元素入口**：固定 ZP（Embedding → LayerNorm → Linear），参数继续训练。
- **当前开发基线 I1**：简单径向局部更新 C1，两次局部调用后接六层全局；全局为 post-norm，注意力概率 dropout=0，Q/K/V/O 恒等初值且继续学习。其他 dropout 保持 0.05。
- **已有配方与选点**：M1、SumNorm KL/W1/Huber、H1 eta/gamma、E0/P0；seed42、35轮、batch32、lr5e-5、FP32。早期按 `balanced_oracle_mae`，后续按 `balanced_blind_r2`；各自契约保留。本次完整A64通过独立开关启用新loss/总量/选点协议，不改变已有模型的评估含义。[原选点边界](logs/log-2026-10-15-joint-blind-selection.md)、[新协议准备](logs/log-2026-10-23-balanced-blind-preparation.md)
- **同数据性能对照**：修复版数据上重训的原六层 Encoder＋ZP，seed42、35轮。ZP100 与 B7 是旧 Q1 的历史参考，数据与预算差异必须注明，不能作为同数据因果对照。

历史按 Oracle MAE 选出的 checkpoint 在完整 valid 上的 **blind 中位 R²**（不是按 blind R² 重新选点）：

| 模型 | eDOS | phDOS |
|---|---:|---:|
| 重训原 baseline | 0.4729 | 0.7272 |
| I1（保留局部更新、修正全局设置） | 0.4801 | 0.7400 |
| I2（完整 B+） | 0.4831 | 0.7370 |
| I3（完整 A） | 0.4851 | 0.7373 |

模型选择只使用 train/valid；历史 test 曾用于诊断，不称为从未触碰的独立留出集。长期规则见[已确定约定](decisions.md)。

## 已知结论

- **局部更新有总体价值**：C0/C1/C2 与 G0、O0/O1 对照支持已测试几何局部更新的总体收益；整体退化不能单独归于局部模块。[局部对照](logs/log-2026-10-09-local-structure-controls.md)、[旧/新骨架对照](logs/log-2026-10-13-legacy-backbone-run.md)
- **全局设置贡献了此前的退化**：归一化位置、注意力概率 dropout、投影初值的直接对照分别改善了部分指标；修正后三项并保留局部与顺序，I1 的总体 R² 恢复。[归一化](logs/log-2026-10-13-global-norm-run.md)、[dropout](logs/log-2026-10-13-attention-dropout-run.md)、[投影初值](logs/log-2026-10-13-attention-projection-run.md)
- **固定投影未显示额外收益**：I0−I1 区间含零；冻结同时减少可训练容量，不能据此认定可学习投影有害或无用。
- **结构谱差收益尚未一般化**：I1 在完整 train 上有有限谱差收益，完整 valid 未检出相应收益；训练支持较近的 phDOS 子组出现探索性改善，eDOS 在已知条件筛选与支持较近的分组中仍未检出收益。[收益拆解](logs/log-2026-10-14-i1-benefit-diagnosis.md)、[材料对审查](logs/log-2026-10-14-i1-pair-audit.md)
- **本次局部直连未达采用条件**：新 blind 规则下 I4 相对 I1/I2 的联合收益区间均含零；train 谱差改善，valid 未检出对应收益。相对 I1，eDOS 失败率增加 0.994 pp，区间 [0.173, 1.816] pp。[接口运行与核验](logs/log-2026-10-16-local-readout-run.md)

结论限单 seed 与既定预算，各实验选点规则须区分。历史 I1 相对 baseline 的 eDOS blind R² 区间下界接近零，不宣称跨种子稳定突破；新规则下尚未重训原 baseline。不据谱差诊断替换 I1，也不由此否定未检验的局部研究方向。

## 尚未解决与下一步

1. **I1/I2/I4 新选点对照已核验完成，继续保留 I1**。blind R²（eDOS/phDOS）分别为 0.4829/0.7379、0.4811/0.7436、0.4830/0.7456。I4−I1 联合分数 +0.00394 [−0.00167, 0.00990]，I4−I2 +0.00194 [−0.00300, 0.00658]；本次不自动追加接口变体或预算。[结果与边界](logs/log-2026-10-16-local-readout-run.md)
2. **完整A以64/128、内部post-norm为共同参考**：[扩宽](logs/log-2026-10-20-local-width-fresh-run.md)未达升级条件；[内部pre-norm对照](logs/log-2026-10-21-local-internal-norm-run.md)联合增益+0.00089 [−0.00387, +0.00619]，未达采用条件。外层512维LN2已完成35轮并独立核验：LN2−A64联合blind +0.00305 [−0.00102, +0.00724]，区间含零；两任务R²/失败率保护线通过，但未达采用条件。valid绝对组成eDOS谱差误差增加+0.00195 [+0.00029, +0.00397]，其余组成诊断见[运行日志](logs/log-2026-10-21-local-outer-norm-run.md)。LR2已满足内部残差加法对照的设计数值采用条件；I1仍为开发基线，是否纳入完整A方案待用户决定，不自动替换或追加训练。全局头数/FFN仍未选型；不宣称容量或归一化最优，也未确定退化机制。

   LR1 35轮、seed42 已完成并独立核验，best ep34/latest35。LR1−A64 联合 blind +0.00305 [−0.00152, +0.00890]，区间含零，未达采用条件；双任务点估计保护线通过，eDOS失败率增加0.994 pp、接近上限。valid同组成谱差各区间均含零。仅用 train/valid、未读 test；继续保留 A64 与 I1，不自动替换或追加实验。[正式运行与核验](logs/log-2026-10-21-local-increment-run.md)

   [内部残差加法LR2对照](design/design-2026-10-21-local-internal-residual.md)已完成一组35轮、seed42：四个内部块使用`LN64(F(s,R))`，保留完整消息、原归一化、完整状态读出和外层残差；best ep29/latest35。LR2−A64联合blind +0.012703 [+0.006638, +0.018374]，两任务点估计保护线通过，达到设计数值采用条件。未读test；不自动替换I1或启动后续实验。[准备与交接](logs/log-2026-10-21-local-internal-residual-preparation.md)、[正式运行与核验](logs/log-2026-10-22-local-internal-residual-run.md)。

3. **保留数据对应风险**：审查的 394 个材料中，125 个输入与 DOS 计算结构未通过严格匹配；冻结模型替换几何未检出改善，训练阶段的影响仍未知。磁态与细致计算条件仍缺元信息。

4. **新版尺度与目标修正包已完成一组正式联训**：A64、seed42、35轮，eligible ep26/29/32/33，best ep32/latest35。相对固定原选中A64 ep31，valid等权blind mean E增加+0.00642 [+0.00127, +0.01147]；判为本次完整包比较恶化，不自动替换I1/A64/LR2，也不能归因单一组件。仅单seed和材料抽样区间；5条待核材料仍保留，未评估test。[准备与短检](logs/log-2026-10-23-balanced-blind-preparation.md)、[正式结果与边界](logs/log-2026-10-23-balanced-blind-run.md)。

完整模型的等价晶胞/超胞一致性与 AMP 尚未验收。历史设计中的待办只记录当时建议，不作为当前执行指令。
