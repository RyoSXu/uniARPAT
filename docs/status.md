# 当前项目状态

更新日期：2026-10-04。此页只记录当前基线、研究方向和待共同决定事项；执行细节见对应日志，
代码与结果的分批归档见[归档记录](logs/log-2026-10-04-periodic-message-archive.md)。
2026-10-04 的目录整理、路径变更与兼容性验证见[整理日志](logs/log-2026-10-04-repository-cleanup.md)；科学状态与训练预算沿用本页。

## 基线

用户已在[路线图](design/design-2026-09-19-structural-refinement.md)中固定 **ZP 元素初始化**：
原子序号 → Embedding → LayerNorm → Linear → Encoder。固定模块结构，参数继续训练。
当前已训练研究基线为 Q1/M1 的 ZP100 `_eidzproj100_s42`：seed 42、100 轮、best epoch 89。
用户指定 **B7 `_e9ctl` 为目前最优指标参考**：Q1/M1、seed 42、35 轮、best epoch 33。
后续实验相对 ZP100 报告改进，相对 B7 报告与最优参考的差距；各任务指标分别报告。
A100 保留为历史性能对照，B100/Z100 用于解释元素入口实验。

选择 ZP 是结构清晰与已知性能代价之间的研究取舍。其四项 valid 中位 R²相对 A100 下降
0.00357–0.01407，eDOS oracle 失败率增加 1.04pp，仍未通过历史实验的全部采用门槛。
单 seed 结果不能证明等价或跨初始化稳定。当前实现与证据见
[`design-2026-10-02-element-initialization.md`](design/design-2026-10-02-element-initialization.md)。

默认目标仍为 SumNorm KL/W1/Huber，blind 使用 H1 eta/gamma，网格为 E0/P0。
模型选择使用 train/valid；历史 test 已被用于若干诊断，不应称为从未触碰的独立留出集。

**数据输入已更新，历史训练结果仍使用原 Q1。**当前默认目录 `data/crystal_q1_v1/` 保存
23,306 条冻结 Q1 成员的完整 FP32 晶格矩阵、变长原子列表、谱与元数据；55 条结构截断已补全，
54 条组成回退价电子数已修复。按已有电子谱完整率阈值，新 train 视图再隔离两条，实际
18,704 / 2,313 / 2,287。旧 Q1 数组和 checkpoint 未覆盖，旧结果不自动代表新修复版。
见 [数据约定](data.md)、[结构输入设计](design/design-2026-10-04-crystal-structure-input.md)
与 [实施日志](logs/log-2026-10-04-crystal-structure-input.md)。

新数据已接入原有模型的输入/计数/训练资源控制；V100 的完整规模 ZP 前向和反向已通过，
没有参数更新或正式训练。这不是独立 A/B+ 的完整 Encoder 架构接入，也没有新增两类谱的
完整 E(3) 保证。训练成本只做固定 batch 短测，整轮吞吐、准确率及新训练预算仍未验收。

## 研究方向

目标是取得**结构条件谱形突破**：增强模型利用晶体结构预测谱形的能力，优先关注 eDOS，同时保护 phDOS。
用户已明确最终两类谱须满足 **E(3) 不变性**，包括整体平移、旋转和反射；中间可使用等变
特征，最终经保持对称性的运算生成不变输出。原子重排与等价晶胞表达另作一致性检查。
几何、完整模型数值证据和架构保证分别验收，要求与两条 Encoder 路线比较见
[`design-2026-10-04-e3-output-invariance.md`](design/design-2026-10-04-e3-output-invariance.md)。
元素入口暂固定 ZP，优先研究 Encoder 的结构消息与原子状态更新。周期多体局部—全局 Encoder
历史单候选已完成并 park，当前没有已获准继续训练的新架构。大方向见
[`design/design-2026-09-19-structural-refinement.md`](design/design-2026-09-19-structural-refinement.md)：

- encoder 如何捕获并转换周期结构信息；
- 共享 encoder 配合任务专属 decoder/head 是否适合两类谱；
- eDOS-only 标签如何与架构、监督训练或预训练结合。

## 当前理解

- [元素身份对照](../archive/docs/logs/log-2026-09-30-element-identity-control.md)：A100/B100 的四项 valid 中位 R²
  差值区间均含零；当前三项性质的输入未显示可区分收益，尚不能证明所有元素性质无用。
- [ZP100 判读](logs/log-2026-10-02-eid-zonlyproj.md)：加投影未明显改善 Z100 的结构响应。
  固定 320 对同组成材料，目标谱差中位 TV 为 0.26975；ZP/Z/A/B 预测为
  0.02732/0.02724/0.03820/0.03840。TV 增大本身不代表预测更准确。
- [已完成诊断](../archive/docs/design/design-2026-10-02-element-identity-diagnosis.md)：Z/ZP 在固定 train 子集上也落后于 B；
  未发现所检查路径中的漏参数、冻结或断梯度问题。入口参数化与共享初值仍是混合解释，根因未定位。
- 代码显示基础 Encoder 逐层更新原子状态，几何进入注意力权重。由公式可推导：在 eval 模式下，
  相同有效 value 的归一化加权平均不随几何打分变化；几何进入消息内容是下一步研究假设。
- [周期多体历史候选](../archive/docs/logs/log-2026-09-29-periodic-manybody-encoder.md)未过原门槛；
  联合边内容、谱差辅助、G2、D3a 与 gamma 校准的既有结论限于各自测试的实现。

## 当前阶段

**物理记录和独立 A/B+ 已实现，CPU/GPU 局部验收已通过；完整模型尚未接入，未启动新训练。**
用户阶段性确定 **6.0 Å**，保留严格 `d<R` 和半径内全部周期镜像，包含自身跨胞镜像。
这个半径来自结构统计，不是已经证明的 DOS 最优作用范围；生产模型默认半径与训练配置尚未修改。

| 小单元 | 已完成的范围与证据 |
|---|---|
| [记录规范](design/design-2026-10-03-periodic-neighbor-records.md) | 六个物理字段、位移方向、多镜像/反向记录、填充与空邻域契约 |
| [半径结构审查](logs/log-2026-10-03-periodic-neighbor-radius-audit.md) | 全量 Q1 train/valid，3–8 Å 共 11 个候选；6.0 Å 为所查网格最小无空邻域、第一组全覆盖候选，前三组完整覆盖约 97.9% |
| [几何验收](logs/log-2026-10-03-periodic-geometry-acceptance.md) | 全量记录契约；144 个结构的独立镜像与物理变换，另查表达一致性和 10 个结构的 2×1×1 超胞；3,352 项、20 项回归通过 |
| [A/B 预检](logs/log-2026-10-03-periodic-message-precheck.md) | 固定 16 个 train/valid 结构与合成例，两 dtype、无更新；840 项与累计 35 项回归通过 |
| [B+ 高阶对照](logs/log-2026-10-03-periodic-message-high-order.md) | 补充三/四阶角度不变量；994 项与累计 42 项回归通过；受控 `c=1` 反例五个种子的 A/B/B+ 可分辨数为 5/0/5 |
| [模块解耦](logs/log-2026-10-03-periodic-message-modules.md) | 物理记录与 A/B+ 公共接口独立于 ZP、旧 Encoder、诊断工具；136 项与累计 56 项回归通过，同权重迁移输出/状态最大差 0 |
| [GPU 局部验收](logs/log-2026-10-03-periodic-message-gpu.md) | V100，FP32/FP64，共 966 项、另四项 CPU 回归通过；局部数值/对称性/梯度与固定批次成本，未运行正式 DOS 模型 |

35/42/56 项回归是逐阶段累计，GPU 阶段另新增四项，不能把表中数量相加为独立用例总数。
三次中断或失败结果保留追溯，
正式结论使用各日志指定的通过目录；目录对应见[归档记录](logs/log-2026-10-04-periodic-message-archive.md)。

当前可复用接口是 [`build_periodic_records`](../utils/periodic_geometry.py) 与
[`PeriodicLocalMessage`](../model/periodic_messages.py)，按[模块边界](design/design-2026-10-04-periodic-message-module-boundaries.md)
接收物理记录与外部原子标量；原型保留冻结，仅用于历史复核和迁移参照。
ZP、完整 Encoder/读出及其调用链没有改变。原型的 B 与独立核心的 B+ 要区别；生产路线尚未选定。

GPU 固定 batch 8 的 FP32 前向加反向中位耗时 A/B+ 为 **466.07/24.48 ms**，相对单线程 CPU
约快 **13.07/5.12 倍**。充分预热短测支持该中位成本，原始延迟峰值保留。
整轮 PyTorch CUDA 分配/保留峰值 109.43/128 MiB，不含 CUDA 上下文或完整训练状态。
这些数据只说明所测局部实现与批次的成本，不证明 B+ 的 DOS 准确率更高，也不确定训练 batch 上限。

## 尚未解决与下一步

- 正常截断下的近截止弱角度响应仍存在；不能把 `c=1` 消融的受控分辨能力当作正常截断改善。
- 新物理记录入口补充了输入防护；旧公共枚举器本身及其其他调用方的缺口没有自动修复。
  `from_edges` 校验字段，但不独立证明外部记录完整性/去重；离散邻居集合变化的可导性未承诺。
- 局部验收不等于完整模型 E(3) 保证。AMP、其他硬件/环境、任意变换/晶胞表达、正式模型内存与
  端到端吞吐、两类谱的输出不变性及准确率仍未验收。

下一小单元是共同确定**完整 Encoder 接入方案**：处理现有 `rp_proj` 方向打分旁路，
追踪全局混合、pooling、两类 decoder/head、尺度与后处理。全链架构论证和有限输出测试分别验收；
原子重排、等价晶胞与超胞按各自契约另报。几何入口与消息使用需配合设计，不预先永久固定编码方案。

接入方案之后另定初始化协议、角度权重对照、训练预算/批次和指标。现有训练入口仍绑定历史
ZP100 配方与指纹，不能自动用于新 Encoder；模型选择只用 train/valid，按 ZP100/B7 参考报告。
目前未授权新训练，入口连续实验队列暂不优先安排。
