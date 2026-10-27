# 设计：固定总量与选点，仅补回独立谱形监督

用户于2026-10-25同意只补一个候选，用于确定后续Encoder研究的训练底座。
不展开辅助权重搜索，不更换局部模块、全局层、编排或联合训练约束。

进度同步（2026-10-27）：35轮已完成并独立核验，best ep25；相对纯E联合误差改善，
相对旧A64的差异区间含零，phDOS失败率较纯E增加但通过固定A64保护。
本组合作为后续研究工作参考，未替换I1历史性能参照，也未解决同组成eDOS谱差。
本轮候选比较结束，不继续搜索权重；[完整结果与限制](../logs/log-2026-10-26-spectral-shape-aux-run.md)、
[当前状态与下一步](../status.md)。以下保留执行前公式和判断标准。

## 问题与唯一候选

[已完成修正包](../logs/log-2026-10-23-balanced-blind-run.md)的联合blind mean E恶化，
两任务在总量取标签最优值时仍留下更高误差，支持谱形参与退化，但不能定位其训练原因。
本次只检验：在相同总量表达和主目标上增加独立谱形监督，能否改善结果。

沿用[归一化blind E](design-2026-10-23-balanced-blind-objective.md)，固定：

`L = 0.5 × mean(E_eDOS + S_eDOS) + 0.5 × mean(E_phDOS + S_phDOS)`。

对物理目标y的离散和大于0的材料，令q=y/sum(y)，
S复用现有SumNorm `KL + W1 + Huber`，w_w1=1、w_huber=1、Huber delta=0.02。
不改箱宽、mask或窗口。零目标没有合法概率谱形，S=0，E仍正常监督；
辅助项按完整批次材料数平均，不按非零材料数重新加权。

辅助系数固定为1，沿用历史谱形项自身系数，不用valid校准或搜索。
这是一个预先固定的候选，不声称1最优，或两任务梯度、难度及最终精度相同。
不复活eta/gamma监督，也不使用N_val侧车。新头和blind推理公式保持不变；
训练改变共享权重后，谱形和总量预测都可能变化，不能称为只改变输出谱形。

## 对照、冻结条件与成本

- 主对照：已完成纯E模型的best ep32，按同一blind E和保护规则选出；不重训。
- 工程参考：旧A64原选中ep31，选点规则不同，只描述与既有性能的差距。
- 候选：完整A64、64/128、I3/staged L2/G6，seed42、35轮、batch32、lr5e-5、FP32。
  元素入口、内外残差、归一化、decoder、优化器、调度、数据及初值协议均保持。
- 复用同一train常数及A64保护线；绑定主对照配置、完整history、best与valid导出，
  protocol内容完全相同。初值指纹相同，从头训练，不续训旧权重。
- 仅有效train 18,704条和完整valid 2,313条；test不参与训练、评价或选型。
- 只新增一组35轮。纯E正式运行4.43小时；计划约4–5小时单卡，首轮实测后更新ETA。
  两批短检不能估计全程峰值；原完整运行已分配约24.56 GiB、设备占用接近32 GiB。

## 选点与决策

两任务失败数分别不超过196/99，中位R²分别不低于0.4817208648−0.02及0.7407193780−0.02。
合格轮次中最小化等权mean E，辅助S不参与选点。无合格best报告no_eligible_best，latest仅供恢复。

报告两任务blind E均值/中位数/P90、中位/P10 R²及失败率；主比较用材料配对bootstrap
2,000次、seed20261025，同一次抽样用于两任务和两模型。旧A64并列报告，不改旧选点。
区间仅条件于选中checkpoint，不代表跨seed稳定性或排除反复使用valid的偏差。

- 相对纯E联合误差差的95%区间上界<0：支持该固定组合改善本次主目标。
- 改善纯E但误差仍高于旧A64：只称部分恢复，不宣称达到旧参考水平。
- 满足相对纯E改善条件、通过冻结保护且误差点估计不高于旧A64：可交用户决定是否作为训练底座，不自动替换I1/LR2。
- 区间含0、恶化或无合格best：保留结果，结束本次候选，不自动调权、加预算或换Encoder。

标签最优总量下的谱形误差、同组成signed谱差和Oracle只作诊断；五条待核材料保留并单列贡献。
诊断改善也不证明局部模块已定型或valid结构谱差普遍改善。

## 实现边界

训练入口 → `ExperimentConfig.spectral_shape_aux` → 独立loss契约 →
`basemodel.train_one_step`中E与S反传 → 原blind E评估/选点 → 原valid导出。

- `model/losses.py::nonzero_sumnorm_klw_loss`复用原谱形核，处理零谱；旧核行为不变。
- `utils/spectral_balance_experiment.py::shape_supervision_contract`冻结辅助公式；
  `balance_model_contract`仅开启候选时使用独立schema。默认false不进入旧training record。
- `tools/eval/spectral_balance_prepare.py::completed_energy_control`绑定主对照；`prepare`只生成单组命令。
- `tools/checks/check_spectral_balance.py::check`核对初值及loss减辅助后与评估一致；
  `tools/eval/lg_valid_export.py::load_run`拒绝开关或公式与checkpoint契约不符。

执行与验证见[准备日志](../logs/log-2026-10-25-spectral-shape-preparation.md)。
