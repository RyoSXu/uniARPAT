# 设计：两类谱等权的 blind 全谱目标

2026-10-23 用户确认：新实验采用**归一化 blind 全谱误差＋双任务失败率保护**。
本关卡替代继续增加 Encoder 变体；完整 A64 骨架、联合训练、seed42 保持。
旧实验保留其原始 loss、选点与结论，不追溯重新选 checkpoint，不自动替换 I1。

## 为什么改

[语义审查](../logs/log-2026-10-22-spectral-contract-audit.md)确认三项实际问题：
eDOS 窗口离散面积/N_val 不具普遍的 1 上限；训练侧车与未来结构输入的计数来源尚不统一；
中位 R² 选点不会控制失败率，且不同零背景会改变 R² 的均值参照。
这些证据不等于 eDOS loss 数值太小，也不证明共享梯度失衡是退化根因。

[scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)说明
R² 依赖目标方差，不宜据不同数据的绝对 R² 判定难度。这里的任务权重是研究取舍，
不能通过改指标令两任务的 R² 看起来相等。[GradNorm 原论文](https://arxiv.org/abs/1711.02257)
针对的是梯度动态平衡；本方案不启用该算法，也不把 loss 系数相同解释为梯度相同。

## 一个训练目标，也是一套选点误差

记任务 t 的现有物理标签为 y，预测为 ŷ，完整输入晶胞的原子数为 N，均匀箱宽为 Δ。
仅用有效 train、包括其零谱，冻结每任务常数：

`C_t = mean_train [ Δ_t × Σ_b (y_t,b / N)² ]`。

每材料误差与联合 loss：

`E_i,t = Δ_t × Σ_b ((ŷ_i,t,b − y_i,t,b) / N_i)² / C_t`

`L = 0.5 × mean_batch(E_eDOS) + 0.5 × mean_batch(E_phDOS)`。

新 loss **替代**原 KL/W1/Huber 与 eta/gamma 辅助监督，不额外叠加这些旧项。
完整窗口与 coverage 使用方式不变；不裁切失败样本、未知谱段或待核标签。
监督 y 使用标签是正常训练；blind 预测 ŷ 的计算只使用结构与冻结常数，不使用标签统计量。

这套目标的可解释性与边界：

- 两任务各占 50%，先按材料平均；晶胞原子数不同不会仅因谱的广延尺度而增加权重。
- train 上全零预测的平均 E 恰为 1，两任务有同一种数值参照；valid 的全零预测均值未必为 1。
- 同时换算坐标与 DOS 单位，或同时倍增 N、标签与预测，E 不变。
- train/待评分谱补入预测正确的零箱，C 和 E 不变；补入区预测非零仍受罚。这不需要改 mask。
- 单材料零谱、常数谱和极弱谱都有有限误差，不除以该材料的方差或近零谱能量。
- 这是平均全谱平方误差目标，不等于典型材料最优，也不保证每任务共享梯度、难度或 R² 相等。
  强谱/尖峰的剩余影响仍须报告；不宣称已证明优于 KL/W1/Huber。

冻结数据上的 C_e=5.542719504890618、C_p=0.03628462802488803，均来自有效 train 18,704 条。
它们有各自物理单位，只有 E 是无量纲；不能直接比较两个 C 的数值。

## 总量输出与推理

复用 `ScaleHead` 的 MLP，输出 e/p 两个原始实数 u。用 train 窗口离散面积/原子的中位数 r 作数值参照：

`q = softmax(shape_logits)`；`ŷ_t = q_t × N × r_t × softplus(u_t) / Δ_t`。

r_e=4.53302986761555，r_p=2.590471916322713。两任务均非负、无固定上限，无目标截断。
这预测的是**现有处理标签的窗口离散面积**；不将它命名为占据电子覆盖率，
也不据允许 phDOS 超 3N 就宣称物理模式数可以超 3N。现有分箱并不严格守恒，标签认证仍待完成。

新推理只需元素/几何、N 和 checkpoint 中冻结的 train 常数；不消费 N_valence 侧车。
这避免替换元素价电子表时引入新的逐样本计数歧义。旧侧车仍保留供历史 H1 与审查使用。
5 条 valid 负能量待核项及其他计数差异不因此被认证或删除。

初始化复制未训练 A64 的所有骨架张量；eta MLP 同形张量映射到新 mass MLP。
头的输出语义不同，不宣称两种头的物理初始预测相同。不同目标/总量语义禁止混用 checkpoint。

## 最佳 checkpoint 与采用边界

显式使用 `balanced_blind_energy`：完整 valid 两任务 mean(E) 的等权平均，**越低越好**。
先按固定 A64 原规则选中点（ep31）过滤，每任务分别要求：

- 失败数不高于 A64：eDOS ≤196/2313（8.47384%），phDOS ≤99/2313（4.28016%）。
- 保留原 R² 保护线：每任务中位 R² 不低于 A64−0.02。

合格轮次中取联合误差最小，完全并列保留更早轮次。整次没有合格轮次，报告
`no_eligible_best`，仅保留 latest 供恢复；不得把它改称 best 或临时放松门槛。
这是当前 valid 点估计保护，不是对新材料失败率不增加的统计保证。

仍报两任务 blind/oracle 中位 R²、P10、失败率，以及 E 的均值/中位数/P90。
Oracle 和同组成谱差是诊断；不参与选点。不用 test 选型。

第一组检验的是尺度、loss、选点的完整修正包，相对现存 A64 选中模型进行工程非退化比较。
旧 A64 并未按新目标重新训练/选点，因此该比较不能单独归因到 loss 或总量头。
只跑 seed42，35轮；先实测短检，长训练交给 OpenCode。单 seed/选中 checkpoint 的配对区间
不能冒充跨训练种子的稳定性证明，也不纠正反复使用valid选型的偏差；是否替换参考仍由人决定。

## 实现与验收

调用链：训练入口 → 冻结协议/契约 → mass 头与 blind 重建 → 同一 E 用于反传和评估 → 过滤选点 → 导出。

| 文件 | 本轮生产变更符号与职责 |
|---|---|
| `utils/spectral_balance.py`：定义与校准 | `ENERGY_METRIC`、`BALANCE_SCHEMA` 命名并隔离协议；`fit_task_normalizer` 拟合 train 的谱能量及总量数值参照；`prepare_balance_protocol` 生成冻结协议并绑定 A64 参考；`validate_balance_protocol` 检查定义；`load_balance_protocol` 验证载荷并重算 train 常数 |
| `utils/spectral_balance.py`：预测与选点 | `reconstruct_balanced_spectra` 无标签重建；`normalized_spectral_error` 返回每材料误差；`balanced_energy_score` 两任务等权；`balanced_checkpoint_eligible` 分别保护两任务；`energy_selection_contract` 记录规则；`selected_energy_history_row` 校验完整历史、过滤、排序 |
| `utils/spectral_balance_experiment.py` | `legacy_balance_config` 复用原骨架校验；`validate_balance_config` 检查新实验组合；`balance_model_contract` 隔离新目标与权重；`initialize_balance_model` 复用未训练 A64 张量 |
| `model/transformer.py` | `Transformer.__init__` 新增 `mass_head`；`Transformer.forward` 输出 `mass_logits`，保留两类谱形 logits |
| `model/model.py` | `basemodel.__init__` 检查目标/头/协议匹配；`basemodel.train_one_step` 调用统一重建与误差，等权反传并保持微批按材料加权 |
| `utils/experiment_config.py`、`utils/local_global_experiment.py` | `ExperimentConfig.spectral_balance_config` 显式启用；`validate_local_global_config` 分流校验；`initialize_comparison_model` 分流初值；`training_record` 保持旧协议默认键不变；`comparison_contract` 构造新身份 |
| `utils/checkpoint_selection.py` | `SELECTION_METRICS` 注册新指标；`selection_contract`、`validation_score`、`selected_history_row` 分流新规则；既有方向/比较函数对该指标沿用最小化语义 |
| `tools/train.py` | `build_arg_parser` 接收新协议；`train_and_eval` 接线、按材料记录 loss、保护/保存/恢复 best/latest；`evaluate_local_global_split` 使用统一重建和误差，报告两任务与尾部指标 |
| `tools/eval/lg_valid_export.py` | `load_run`、`select_best` 校验新 loss/总量契约并重算选点，不把 latest 冒充 best |
| `tools/eval/lg_valid_export.py`、`tools/eval/local_global_compare.py` | `export` 复核数据与重算分数/保护线后输出逐材料结果；`summarize` 支持新选点与无合格 best 状态 |
| `tools/eval/spectral_balance_prepare.py` | `prepare` 生成冻结 JSON、旧选中点诊断、源码快照及命令；`main` 为命令行入口，不启动训练 |
| `tools/checks/check_spectral_balance.py` | `check` 做两次一次性实批更新、loss/评估一致性和共享梯度诊断；`main` 为命令行入口，不写实验权重 |

测试入口按职责归组：`SpectralBalanceTests.test_*` 共11项数学/规则检查，
`BalancedPipelineTests.test_*` 共8项调用链检查；`protocol`、`row`、`grid_sample` 及测试类夹具
只构造测试输入。`tools/checks/check-static.sh` 注册两组测试，不改变训练行为。

解析测试覆盖单位、正确零背景、晶胞倍增、零谱、梯度、双任务保护及并列；调用链测试覆盖
训练/评估一致性、微批累积、A64 初值、局部 E(3)/原子重排夹具、恢复、导出和无合格 best。
正式完整模型的全部对称性验收与训练精度仍不能由这些短检替代。

执行和证据见[准备日志](../logs/log-2026-10-23-balanced-blind-preparation.md)。
