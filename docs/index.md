# 文档导航

本索引只列入口，不枚举每份历史设计和实验文件。历史材料统一保留在 `archive/` 的 `docs/code/data/results` 分类下，不默认构成当前执行指令。

当前先读 `status.md`，再按记录规范、候选公式、模块边界定位接口；执行证据按阶段查下方日志。
用户已决定结束本轮整体 Encoder 骨架归因，暂不追加修正骨架有/无局部更新对照。
保留局部更新，以修正三项全局设置后的 I1 为诊断参考，当前重点是：
**为什么总体指标改善，同组成、不同结构材料的谱差预测却仍未改善？**
优先分析总体收益的来源与组内结构谱差的训练到验证落差；尚未启动新的诊断或训练。
以下为前一阶段证据：G0 同骨架六层全局、关闭局部更新的对照已完成，结果为
「几何局部更新有净增益但综合表现未追平原 baseline、C0 未显示可分辨净效应」。
O0/O1 旧全局骨架归因对照已完成训练与配对归因，结论与边界见下方证据与状态页。
N1 归一化位置对照已完成训练与配对归因（只把 C1 全局 pre-norm 改为 post-norm）：
phDOS 的 R²/MAE 与 eDOS oracle 失败率的收益可分辨，eDOS 中位 R² 未分辨；剩余骨架差距仍主要在 eDOS 失败率。
D1 注意力概率 dropout 对照已完成训练、配对归因与 Codex 独立核验（在 N1 上只关闭六层全局注意力概率 dropout）：eDOS blind 失败率与 blind MAE 有可分辨改善（−0.95 pp、−0.031），eDOS R² 未分辨，phDOS 无可分辨退化；N1−O1 的 eDOS 失败率差距由 +2.03 pp 缩至 +1.08 pp，仍可分辨。随后投影初始化对照见下。
I1/I0 投影初始化与学习能力对照已完成训练、配对归因与 Codex 独立核验（seed 42、35 轮，均在未训练 D1 上只改 Q/K/V/O 投影）：I1−D1 的 eDOS blind R²（+0.009）与失败率（−1.34 pp）、phDOS R² 有可分辨改善，eDOS MAE 与 phDOS 失败率未分辨；I0−I1 无可分辨净效应，但含可训练容量变化、不证明等价；I1−O1 的 eDOS 失败率差距已不可分辨。I1 相对原 baseline 的四项总体 R² 为正、区间不含零，但 eDOS blind 下界接近零，同组成结构谱差尚无改善。结论限单 seed；用户决定暂不追加局部开关对照，保留已有局部更新价值证据，转向同组成结构谱差研究。
此前困难材料对和线性读出诊断属于历史证据，
不自动触发后续实验。
局部与完整模型的有限 CPU/GPU 检查已通过；四组首轮对照（seed=42）、原六层 baseline 对照与全局深度
解耦对照（B+ K=2、G=6 两组）已完成，路线/编排/深度选型未定，配置、完成状态与执行设置见状态页。

| 目的 | 文档 |
|---|---|
| 当前基线、最优指标参考和下一步 | [`status.md`](status.md) |
| 目录职责、路径迁移、设计日期和 Git 忽略规则 | [`repository.md`](repository.md) |
| 设计与日志模板 | [`design/template.md`](design/template.md)、[`logs/template.md`](logs/template.md) |
| 大方向路线图 | [`design/design-2026-09-19-structural-refinement.md`](design/design-2026-09-19-structural-refinement.md) |
| 当前 ZP 元素初始化、代码职责与使用限制 | [`design/design-2026-10-02-element-initialization.md`](design/design-2026-10-02-element-initialization.md) |
| 当前周期邻居记录规范与审查边界 | [`design/design-2026-10-03-periodic-neighbor-records.md`](design/design-2026-10-03-periodic-neighbor-records.md) |
| 最终谱输出的 E(3) 目标、两条 Encoder 路线与证据边界 | [`design/design-2026-10-04-e3-output-invariance.md`](design/design-2026-10-04-e3-output-invariance.md) |
| 6.0 Å A/B 与增强 B+ 的公式、实际原型接口与取舍 | [`design/design-2026-10-04-periodic-message-candidates.md`](design/design-2026-10-04-periodic-message-candidates.md) |
| 已提取的物理记录/A/B+ 接口、职责边界与后续验收 | [`design/design-2026-10-04-periodic-message-module-boundaries.md`](design/design-2026-10-04-periodic-message-module-boundaries.md) |
| 已实施的局部—全局 Encoder、共同初值与路线×编排对照入口 | [`design/design-2026-10-04-local-global-encoder.md`](design/design-2026-10-04-local-global-encoder.md) |
| 当前整体 Encoder 归因：六层全局、无局部更新 G0 | [`无局部更新对照设计`](design/design-2026-10-12-global-only-attribution.md) |
| 旧/新骨架 × 局部更新 O0/O1 | [`旧骨架归因设计`](design/design-2026-10-12-legacy-backbone-attribution.md) |
| 全局骨架拆分：归一化位置 N1 | [`N1 单因素设计`](design/design-2026-10-13-global-norm-attribution.md)；[执行日志](logs/log-2026-10-13-global-norm-run.md) |
| 全局骨架拆分：注意力概率 dropout D1 | [`D1 单因素设计`](design/design-2026-10-13-attention-dropout-attribution.md)；[实施与准入](logs/log-2026-10-13-attention-dropout-preflight.md)；[执行日志](logs/log-2026-10-13-attention-dropout-run.md) |
| 全局骨架拆分：恒等投影初值/学习 I1/I0 | [`投影对照设计`](design/design-2026-10-13-attention-projection-attribution.md)；[实施与准入](logs/log-2026-10-13-attention-projection-preflight.md)；[执行与核验](logs/log-2026-10-13-attention-projection-run.md) |
| 协作与实验流程 | [`workflow.md`](workflow.md) |
| 长期决策和评估约定 | [`decisions.md`](decisions.md) |
| 数据、缓存及标签口径 | [`data.md`](data.md)；[`data/reference/edos-label-audit.md`](../data/reference/edos-label-audit.md) |
| 完整原子、FP32 存储与矩阵输入契约 | [`结构输入设计`](design/design-2026-10-04-crystal-structure-input.md)、[`实施日志`](logs/log-2026-10-04-crystal-structure-input.md) |
| 术语与指标 | [`glossary.md`](glossary.md) |

## 当前路线相关证据

- I1/I0 投影初始化与学习能力对照已完成：共同未训练 D1、恒等映射，其中 I0 只固定 Q/K/V/O；初始化收益可分辨、固定投影净效应未分辨，整体恢复不等于同组成结构谱差改善。设计、验证和边界见[投影对照设计](design/design-2026-10-13-attention-projection-attribution.md)、[实施日志](logs/log-2026-10-13-attention-projection-preflight.md)、[执行日志](logs/log-2026-10-13-attention-projection-run.md)。
- D1 直接对照保持 N1 的 post-norm 和全部初值，只关闭六层注意力概率 dropout；其余 dropout 保持。训练已完成（seed 42、35 轮，best 第 29 轮），D1−N1 的 eDOS blind 失败率 −0.95 pp 与 blind MAE −0.031 可分辨，eDOS R² 与 oracle 失败率未分辨，同组成谱差未见改善：[设计](design/design-2026-10-13-attention-dropout-attribution.md)、[实施日志](logs/log-2026-10-13-attention-dropout-preflight.md)、[2026-10-13 执行日志](logs/log-2026-10-13-attention-dropout-run.md)。
- N1 单因素入口与训练前检查：保持完整 C1 未训练初值与 learned Q/K/V/O、概率 dropout，只改变六层全局归一化位置：[设计](design/design-2026-10-13-global-norm-attribution.md)、[实施日志](logs/log-2026-10-13-global-norm-preflight.md)。
- N1 全局归一化位置对照已完成（seed 42、35 轮，复用 C1/O1/原 baseline 做 2,313 材料配对）：phDOS 的 R²/MAE 与 eDOS oracle 失败率的收益可分辨，eDOS 中位 R² 与 blind 失败率未分辨；N1−O1 的 eDOS 残差仍显著；同组成谱差未见改善：[2026-10-13 执行日志](logs/log-2026-10-13-global-norm-run.md)。
- O0/O1 旧全局骨架归因对照已完成（seed=42、35 轮，复用 G0/C1/原 baseline 做 2,313 材料配对）：旧几何路径仅 phDOS 等小幅可分辨贡献，C1 收益两种骨架方向一致、骨架差距在有无局部时均存在、交互未分辨，同组成谱差未见可分辨改善：[2026-10-13 执行日志](logs/log-2026-10-13-legacy-backbone-run.md)。
- 用户已确认 O0/O1 旧全局骨架归因：关闭旧几何打分，分别无局部 / 加当前 C1，复用 G0/C1/原 baseline；设计、实施与短测：[旧骨架归因设计](design/design-2026-10-12-legacy-backbone-attribution.md)、[预检日志](logs/log-2026-10-12-legacy-backbone-preflight.md)。
- G0 无局部更新对照已完成（seed=42、35 轮，复用 C0/C1/C2/原 baseline 配对）：C0 未显示可分辨净效应、C1/C2 几何局部更新有净增益、与原 baseline 的差距在关闭局部后仍存在且更大；剩余差距仍未唯一定位：[2026-10-12 G0 执行日志](logs/log-2026-10-12-global-only-run.md)；训练前准入与预检证据：[2026-10-12 G0 预检日志](logs/log-2026-10-12-global-only-preflight.md)。
- C1 联合损失与仅谱差损失的同材料/初值/预算短对照（train-only、seed 42、各 600 步）已完成：两对仍失败、整体充分拟合未过，单材料两类谱明显变差；当前预算下去掉联合监督不能修复困难对，根因未定：[2026-10-12 损失隔离日志](logs/log-2026-10-12-local-structure-loss-isolation.md)
- C1 小样本拟合困难材料对诊断（train-only、seed 42、隔离复现 600 步＋参照对）：输入和完整表示可区分；阶段距离放大 sqrt(512) 的错误已更正，不能定位 decoder 瓶颈；梯度冲突是部分状态观察，control 的同对无冲突仍失败，根因未定；复现漂移来源未定位：[2026-10-12 困难对归因日志](logs/log-2026-10-12-hard-pair-attribution.md)
- C1 小样本结构谱差拟合（train-only、seed 42、共同已训练起点、两组各 600 步）：显式监督降低配对误差，但单材料 eDOS 拟合变差；两组未过预设充分拟合门槛，未检验泛化：[2026-10-12 小样本拟合日志](logs/log-2026-10-12-local-structure-small-fit.md)
- 逐阶段结构谱差可读性诊断（冻结 C1/C2、完整 train 配对、无截距读出、decoder 保留能量位置，seed 42、train/valid）：valid 无阶段优于严格零响应，decoder 读出 train→valid 反号；结论限于所测试的线性读出，不能据此排除其他读出或定位整体性能差距：[2026-10-11 逐阶段读出诊断日志](logs/log-2026-10-11-stage-readout-diagnosis.md)
- 结构响应泛化诊断（C0/C1/C2 best checkpoint、只读、seed 42、train/valid）：幅度不足成立但幅度校准在 valid 有害、方向错误支持最强、覆盖部分成立、pooled 探针不支持“后续利用不足”；[2026-10-10 泛化诊断日志](logs/log-2026-10-10-local-structure-attribution-diagnosis.md)
- 局部结构贡献诊断（已有四模型、train/valid、无更新）：训练谱差改善而验证配对收益未分辨，角度路径影响预测；[诊断日志](logs/log-2026-10-09-local-structure-diagnosis.md)与[最小对照协议](design/design-2026-10-09-local-structure-attribution.md)。同骨架 C0/C1 seed=42 各 35 轮及三组配对分析已完成：简单径向有总体收益，B+ 的 eDOS 额外收益未分辨，同组成结构响应仍未显示 valid 收益；结果/成本/复核见[对照实施日志](logs/log-2026-10-09-local-structure-controls.md)。
- 长预算对照（baseline／B+ 分段 G=6 各 100 轮，seed=42）：差距是否缩小、失败率、训练改善而验证停滞与选点规则张力：[2026-10-08 长预算对照](logs/log-2026-10-08-lg-long-budget-e100.md)
- 全局深度解耦对照（B+ K=2、G=6 两组，seed=42、35 轮）、严格初值对齐、训练前验证与固定 train 子集无训练诊断：[2026-10-08 全局深度对照](logs/log-2026-10-08-lg-global-depth-q1v1.md)
- 原六层 Encoder＋ZP baseline 对照（seed=42、35 轮）与逐样本配对 bootstrap 比较：[2026-10-08 baseline 对照](logs/log-2026-10-08-lg-baseline-q1v1.md)
- A/B+ × staged/alternating 四组首轮对照（seed=42 各 35 轮）结果、口径与执行异常：[2026-10-07 四组对照](logs/log-2026-10-07-local-global-four-arm-s42.md)
- 正式采用 65536 角度分块与有预算的 GPU 结构缓存、层间几何复用、失效与 checkpoint 恢复检查：[2026-10-06 几何缓存](logs/log-2026-10-06-local-geometry-cache.md)
- 正式 A 关闭角度重计算、真实第 21 轮状态只读恢复、数值波动诊断与含 optimizer 的成本：[2026-10-06 关闭重计算](logs/log-2026-10-06-local-global-no-recompute.md)
- A 的等价实现提速、同权重输出/梯度与全链 E(3) 复查、大 batch/重计算资源探针：[2026-10-05 提速日志](logs/log-2026-10-05-local-message-speedup.md)
- 外部审核反馈核查、回归覆盖补齐、共同配置和最终复查：[2026-10-05 审核收尾](logs/log-2026-10-05-local-global-review-followup.md)
- 完整 Encoder 与四组对照代码、全链 E(3)、CPU/GPU 无更新检查与固定批次成本：[2026-10-05 接入日志](logs/log-2026-10-05-local-global-encoder.md)
- 周期消息工作按阶段归档、正式/中断目录与提交边界：[2026-10-04 归档记录](logs/log-2026-10-04-periodic-message-archive.md)
- A/B+ 的 GPU 数值、局部对称性、梯度与固定 batch CPU/GPU 成本：[2026-10-03 GPU 验收](logs/log-2026-10-03-periodic-message-gpu.md)
- A/B+ 核心提取、共同接口与梯度/依赖验收：[2026-10-03 解耦验收](logs/log-2026-10-03-periodic-message-modules.md)
- 增强 B+ 与 A/B 的参数、CPU 成本与受控分辨能力：[2026-10-03 高阶角度对照](logs/log-2026-10-03-periodic-message-high-order.md)
- 元素入口实验： [A100/B100](../archive/docs/logs/log-2026-09-30-element-identity-control.md)、
  [Z100](../archive/docs/logs/log-2026-10-01-eid-zonly-baseline.md)、[ZP100](logs/log-2026-10-02-eid-zonlyproj.md)
- 纯 Z 训练差距诊断：[协议与完成状态](../archive/docs/design/design-2026-10-02-element-identity-diagnosis.md)、
  [诊断报告与迁移导航](../results/reports/eid_diagnosis_s42/20261002T095909Z/README.md)；解释边界见当前元素初始化文档
- 周期多体 encoder 单候选 Q1 valid 判决：[`logs/log-2026-09-29-periodic-manybody-encoder.md`](../archive/docs/logs/log-2026-09-29-periodic-manybody-encoder.md)
- 当前 6.0 Å 几何验收与边界：[`logs/log-2026-10-03-periodic-geometry-acceptance.md`](logs/log-2026-10-03-periodic-geometry-acceptance.md)
- 两套局部消息的 CPU 预检、资源与表达限制：[`logs/log-2026-10-03-periodic-message-precheck.md`](logs/log-2026-10-03-periodic-message-precheck.md)
- eDOS 误差与收益上界：[`logs/log-2026-09-28-model-accuracy-upper-bound.md`](../archive/docs/logs/log-2026-09-28-model-accuracy-upper-bound.md)
- 结构条件谱形支持与同组成材料对：[`logs/log-2026-09-26-edos-spectral-support.md`](../archive/docs/logs/log-2026-09-26-edos-spectral-support.md)
- eDOS-only 本地候选审计：[`logs/log-2026-09-28-d3a-edos-support-adjudication.md`](../archive/docs/logs/log-2026-09-28-d3a-edos-support-adjudication.md)
- gamma 校准与尺度头候选复核：[`logs/log-2026-09-28-h1-gamma-calibration-gate.md`](../archive/docs/logs/log-2026-09-28-h1-gamma-calibration-gate.md)、[`logs/log-2026-09-28-h1-next-candidate-review.md`](../archive/docs/logs/log-2026-09-28-h1-next-candidate-review.md)

历史实验按主题在 `archive/docs/logs/` 中检索；只有 `status.md` 明确列为当前工作的设计才是活跃执行依据。

本次仓库整理与验证见[整理日志](logs/log-2026-10-04-repository-cleanup.md)，历史材料入口见[archive](../archive/README.md)。

提交与推送前核验见[提交准备日志](logs/log-2026-10-04-repository-publish.md)。
