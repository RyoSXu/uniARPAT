# H1 下一候选的多模型审查

日期：2026-09-28

状态：**审查完成；当时形成一个待评审设计，后因主线调整撤回；未改代码、未训练、未读取 test。**

## 任务背景

H1 gamma 全局两参数校准已 park，但冻结 B7 的真 gamma 替换仍有 `+0.04184` 的 valid blind eDOS
理想上界。用户要求停止低杠杆的连续诊断，转向能产生实际模型升级的候选；同时要求在调研和思考阶段
使用不同 OpenCode 模型扩大思路，再由协调者按仓库事实复核。

## OpenCode 分工与交付状态

所有任务均要求只读、禁止训练、禁止读取 Q1 test、不得写仓库文件。

| 模型 | 角色 | 状态 | 可用结论 |
|---|---|---|---|
| DeepSeek V4.1 Flash | 反方审查“独立 gamma 头 + decoder 表征” | 完整交付 | 原方案同时改变头共享、输入来源和梯度路径；SumNorm decoder 没有绝对尺度监督，不能把 decoder 表征直接视为 gamma 信息源；裁决为 revise |
| Qwen 3.8 Flash | 基于冻结事实快速提出唯一候选 | 完整交付 | 只把 H1 前固定平均池化改为单查询 softmax 加权池化，保留共享头、损失和接口 |
| Muse Spark 1.3 Contributor | 失败模式审查 | 失败 | 读完指定材料后连续 `ECONNRESET`，无结论 |
| MiMo V2.6 Flash | 最多三个可训练候选 | 超时停止 | 读完材料和模型代码后未形成报告，不计入决策 |
| GLM 5.3 Flash | 接替失败模式审查及快速裁判 | 超时停止 | 完成取证但未形成报告，不计入决策 |
| MiMo V2.6 Pro | 三候选最终裁判 | 超时停止 | 未形成报告，不计入决策 |

完整报告只存在于本轮终端临时输出；本日志仅保留影响决定的结论，不把工具调用轨迹或半成品写入项目。

## 先撤回原提议

上一轮口头提出“独立 gamma 头同时读取 encoder pooled memory 与 eDOS decoder 表征”。代码与反方审查
确认它不满足单因素：

1. `EtaHead` 的共享 trunk 会被拆分；
2. gamma 输入从 encoder 平均池化改为 decoder 表征；
3. gamma 梯度会新增一条回流 decoder 的路径；
4. 新头容量与初始化也同时改变。

此外，`model/losses.py::sumnorm_klw_loss` 对 eDOS logits 做归一化分布监督，decoder 没有绝对总量
标签；现有证据不能支持“decoder hidden state 携带 gamma 需要的绝对尺度”。因此该复合方案撤回，不进入
实现。

## 协调者代码复核

事实：

- `model/transformer.py::Transformer.forward` 在 `scale_mode="eta"` 时只调用
  `global_masked_pool(memory, mask_atom)`，再把 `[B,512]` 送入原 `EtaHead`。
- `memory` 只含原子 token；两个哨兵已在构造 `atom_src` 时剥离。`mask_atom` 与 memory 长度严格对齐，
  `True` 表示 padding。
- `model/heads.py::global_masked_pool` 对有效原子做无参数平均；`EtaHead` 是
  `512→128→64→2→sigmoid`，B7 权重已存在。
- Q1 H1 没有比较过平均池化与可学习池化。G2a、联合边内容和谱差辅助改变的是结构消息或谱形监督，
  没有直接检验 H1 聚合器。
- 历史代码审计指出平均池化符合 eta/gamma 作为比率的语义；这反驳“必须改成求和”，但不证明均匀
  原子权重是最优充分统计。

推断：gamma 的排序相关 `0.870` 与全局校准近恒等，说明剩余误差是逐材料误差；固定平均可能压低少数
关键局部环境的贡献，但这不是已证根因。真 gamma 替换的 `+0.04184` 只是候选的效应量上界；达到主门
仍需回收约 48%，不能写成预期收益。

## 候选取舍

1. **专用 gamma 残差 MLP：不选。**它只增加同一平均池化表征上的容量；现有 `EtaHead` 已有约 74k
   参数，没有容量瓶颈证据，且冻结主干版本不符合直接改进完整模型的优先级。
2. **gamma MSE 改 soft-target BCE：不选。**它是单一因素，但全局 BCE 校准没有显示可用效应，且
   会改变 encoder 的辅助梯度；目前没有比聚合器更直接的样本级信息机制。
3. **H1 可学习原子池化：进入设计。**它只改变 `memory→EtaHead` 的聚合算子；旧头、监督、主谱形、
   decoder 和部署公式保持。零初始化打分向量使 step 0 数学上退化为原平均池化，同时保留可学习梯度。

## 防止再做低杠杆实验的规则

模型候选进入实现前必须同时满足：

1. 有至少 `+0.04` 的已测主指标理想上界，且明确上界不是预期收益；
2. 干预直接作用于该误差通道，而非只增加描述性指标；
3. 能写成一个配置开关和一个因素，旧 checkpoint 有受控初始化；
4. 一次 M1×10 valid-only 就能决定保留或停止，不依赖下一项诊断；
5. 通过后有明确的生产调用链，失败后不自动扫描容量、池化形式或损失。

本轮只有 H1 可学习原子池化同时满足这些条件。后续撤回记录见
[`design-h1-attentive-pooling-pilot.md`](../design/design-h1-attentive-pooling-pilot.md)。
