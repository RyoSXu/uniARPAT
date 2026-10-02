# 当前项目状态

更新日期：2026-10-02。此页只记录当前基线、研究方向和待共同决定事项；历史结果见对应日志。

## 基线

用户已在[路线图](design/design-structural-refinement.md)中固定 **ZP 元素初始化**：
原子序号 → Embedding → LayerNorm → Linear → Encoder。固定模块结构，参数继续训练。
当前已训练研究基线为 Q1/M1 的 ZP100 `_eidzproj100_s42`：seed 42、100 轮、best epoch 89。
用户指定 **B7 `_e9ctl` 为目前最优指标参考**：Q1/M1、seed 42、35 轮、best epoch 33。
后续实验相对 ZP100 报告改进，相对 B7 报告与最优参考的差距；各任务指标分别报告。
A100 保留为历史性能对照，B100/Z100 用于解释元素入口实验。

选择 ZP 是结构清晰与已知性能代价之间的研究取舍。其四项 valid 中位 R²相对 A100 下降
0.00357–0.01407，eDOS oracle 失败率增加 1.04pp，仍未通过历史实验的全部采用门槛。
单 seed 结果不能证明等价或跨初始化稳定。当前实现与证据见
[`design-element-initialization.md`](design/design-element-initialization.md)。

默认目标仍为 SumNorm KL/W1/Huber，blind 使用 H1 eta/gamma，网格为 E0/P0。
模型选择使用 train/valid；历史 test 已被用于若干诊断，不应称为从未触碰的独立留出集。

## 研究方向

目标是取得**结构条件谱形突破**：增强模型利用晶体结构预测谱形的能力，优先关注 eDOS，同时保护 phDOS。
元素入口暂固定 ZP，优先研究 Encoder 的结构消息与原子状态更新。周期多体局部—全局 Encoder
历史单候选已完成并 park，当前没有已获准继续训练的新架构。大方向见
[`design/design-structural-refinement.md`](design/design-structural-refinement.md)：

- encoder 如何捕获并转换周期结构信息；
- 共享 encoder 配合任务专属 decoder/head 是否适合两类谱；
- eDOS-only 标签如何与架构、监督训练或预训练结合。

## 当前理解

- [元素身份对照](logs/log-2026-09-30-element-identity-control.md)：A100/B100 的四项 valid 中位 R²
  差值区间均含零；当前三项性质的输入未显示可区分收益，尚不能证明所有元素性质无用。
- [ZP100 判读](logs/log-2026-10-02-eid-zonlyproj.md)：加投影未明显改善 Z100 的结构响应。
  固定 320 对同组成材料，目标谱差中位 TV 为 0.26975；ZP/Z/A/B 预测为
  0.02732/0.02724/0.03820/0.03840。TV 增大本身不代表预测更准确。
- [已完成诊断](design/design-element-identity-diagnosis.md)：Z/ZP 在固定 train 子集上也落后于 B；
  未发现所检查路径中的漏参数、冻结或断梯度问题。入口参数化与共享初值仍是混合解释，根因未定位。
- 代码显示基础 Encoder 逐层更新原子状态，几何进入注意力权重。由公式可推导：在 eval 模式下，
  相同有效 value 的归一化加权平均不随几何打分变化；几何进入消息内容是下一步研究假设。
- [周期多体历史候选](logs/log-2026-09-29-periodic-manybody-encoder.md)未过原门槛；
  联合边内容、谱差辅助、G2、D3a 与 gamma 校准的既有结论限于各自测试的实现。

## 下一步

以 ZP 为入口，设计几何如何参与消息内容及原子状态更新，并考虑周期镜像和方向对称性。
新实验实施前另定初始化协议、对照、预算及判据。现有训练入口的 ZP 流程仍绑定历史 ZP100
配置与指纹，需要在新方案中处理；目前未授权新训练，入口连续实验队列暂不优先安排。
