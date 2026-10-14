# 当前项目状态

更新日期：2026-10-14。本页只保留当前目标、参考模型、关键结论和下一步；历史过程与验收细节见[文档索引](index.md)。

## 当前目标

从晶体结构预测 eDOS 与 phDOS，重点提高**同组成、不同结构材料的谱形差异预测**；优先改善 eDOS，同时保护 phDOS。
最终两类谱须满足 E(3) 不变性，原子重排与等价晶胞另作一致性检查。要求见[对称性设计](design/design-2026-10-04-e3-output-invariance.md)。

**本轮全局骨架归因已结束。**保留局部更新，暂不追加修正骨架的有/无局部对照，也不继续拆分其他骨架因素。
下一步先分析现有模型的总体收益来源与结构谱差泛化；尚未启动这项分析或新训练。

## 当前参考与口径

- **数据**：修复版 `data/crystal_q1_v1/`，实际 train/valid/test 视图为 18,704 / 2,313 / 2,287；字段和历史数据边界见[数据约定](data.md)。
- **元素入口**：固定 ZP（Embedding → LayerNorm → Linear），参数继续训练。
- **后续诊断参考 I1**：简单径向局部更新 C1，两次局部调用后接六层全局；全局为 post-norm，注意力概率 dropout=0，Q/K/V/O 恒等初值且继续学习。其他 dropout 保持 0.05。
- **本阶段配方**：M1、SumNorm KL/W1/Huber、H1 eta/gamma、E0/P0；seed42、35轮、batch32、lr5e-5、FP32，按 `balanced_oracle_mae` 选点。完整协议见[投影对照设计](design/design-2026-10-13-attention-projection-attribution.md)。
- **同数据性能对照**：修复版数据上重训的原六层 Encoder＋ZP，seed42、35轮。ZP100 与 B7 是旧 Q1 的历史参考，数据与预算差异必须注明，不能作为同数据因果对照。

所选 checkpoint 在完整 valid 上的 **blind 中位 R²**：

| 模型 | eDOS | phDOS |
|---|---:|---:|
| 重训原 baseline | 0.4729 | 0.7272 |
| I1（保留局部更新、修正全局设置） | 0.4801 | 0.7400 |

模型选择只使用 train/valid；历史 test 曾用于诊断，不称为从未触碰的独立留出集。长期规则见[已确定约定](decisions.md)。

## 已知结论

- **局部更新有总体价值**：C0/C1/C2 与 G0、O0/O1 对照支持已测试几何局部更新的总体收益；整体退化不能单独归于局部模块。[局部对照](logs/log-2026-10-09-local-structure-controls.md)、[旧/新骨架对照](logs/log-2026-10-13-legacy-backbone-run.md)
- **全局设置贡献了此前的退化**：归一化位置、注意力概率 dropout、投影初值的直接对照分别改善了部分指标；修正后三项并保留局部与顺序，I1 的总体 R² 恢复。[归一化](logs/log-2026-10-13-global-norm-run.md)、[dropout](logs/log-2026-10-13-attention-dropout-run.md)、[投影初值](logs/log-2026-10-13-attention-projection-run.md)
- **固定投影未显示额外收益**：I0−I1 区间含零；冻结同时减少可训练容量，不能据此认定可学习投影有害或无用。
- **结构谱差问题仍在**：总体恢复没有带来同组成谱差的可分辨改善。旧模型诊断发现响应幅度小、方向难以迁移；小样本直接监督只支持部分训练对可学，尚未证明泛化。[泛化诊断](logs/log-2026-10-10-local-structure-attribution-diagnosis.md)、[小样本拟合](logs/log-2026-10-12-local-structure-small-fit.md)

结论限单 seed、既定预算和选点；I1 相对 baseline 的 eDOS blind R² 区间下界接近零，不宣称跨种子稳定突破。完整 A/B+ 在修正骨架上的效果、最终路线与编排选型仍未确定。

## 尚未解决与下一步

1. **先分清总体收益来自哪里**：用 I1 已有权重分析组成平均谱、组内谱形偏差及 blind 尺度预测，并比较 train/valid。
2. **再定位结构谱差为何不能泛化**：检查响应幅度与方向、训练覆盖，以及可得的标签来源和计算条件。覆盖、表示/优化、监督和标签条件都仍是待验证解释。
3. 根据分析结果再设计有区分力的干预。此前线性读出与损失隔离没有定位唯一根因，不自动重复旧诊断或启动训练。[逐阶段读出](logs/log-2026-10-11-stage-readout-diagnosis.md)、[困难对诊断](logs/log-2026-10-12-hard-pair-attribution.md)、[损失隔离](logs/log-2026-10-12-local-structure-loss-isolation.md)

完整模型的等价晶胞/超胞一致性与 AMP 尚未验收。历史设计中的待办只记录当时建议，不作为当前执行指令。
