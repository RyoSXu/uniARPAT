# 设计：冻结 G2 的结构信息通路核验

## 目标与范围

- 用户已确认在恢复保护修复后执行本诊断。问题是：G2 几何残差是否改变 encoder 表征，这种变化
  是否传到 decoder 与谱输出，以及输出变化是否对应真实谱差。
- 固定 `_g2ctl`、`_g2edge` 的 `checkpoint_latest.pth`：M1、epoch 10、seed 42、Q1、FP32、
  E0/P0、SumNorm 与 H1。核对完整有效配置仅 `use_g2` 不同。B7 epoch 33 的既有结果仅作背景。
- 仅读取 Q1 train 18,706、valid 2,313；三个推理状态为 control、edge、同一 edge 权重的六层
  `PeriodicEdgeMessage.alpha` 暂时置零。关闭残差只发生于内存，退出时恢复原值，不训练或保存权重。

## 方法与量

1. **复用生产边界。**通过 `ConfigBuilder`、`Dos_Dataset`、`basemodel.data_preprocess` 读取并
   重建物理谱，用 `per_sample_spectral_metrics` 计算全 bin 的 eDOS/phDOS oracle/blind R²。
   材料 ID 与缓存逐行对齐。valid 的原始两臂 oracle 汇总须复现各自 epoch 10 history，容差 `2e-5`。
2. **同组成材料对。**按每种元素的绝对计数分组，排除缓存结构签名完全相同的对；复用
   `structure_signal_diagnostic.structure_signature`。valid 必须复现既有 CSV 的 320 对与 ID，
   复用其中结构匹配标签，分别报告单元素、多元素、结构不匹配的多元素对。train 使用相同规则。
3. **内部表征。**只用 forward hook 读取最终 encoder 原子 token 和 eDOS decoder 输出。
   encoder 距离在同种元素内部做最小平方距离的一对一匹配，再计算均方根差（RMS）／两端平均
   RMS；decoder 按固定能量 query 对齐，用同样的相对 RMS。该量是同一模型内部的响应大小，
   不能将不同模型的坐标轴直接相减，也不能将跨层距离比称为信息保存率。
4. **谱差是否正确。**记录目标／预测概率谱 TV；同时记录带符号谱差误差
   `0.5 × sum(abs((p_a-p_b)-(q_a-q_b)))` 和谱差余弦。只增加预测 TV 不能证明信息更有用。
5. **残差实际作用。**记录各层 alpha 和真实原子上 `RMS(h_after-h_before)/RMS(h_before)`；
   对 edge 与关闭残差的同材料，记录谱 TV、表征相对 RMS与主指标变化。关闭残差的结果只说明
   已训练模型的依赖，不能替代重新训练的消融因果效应。
6. **数值对照。**每臂首个 batch 重复前向并同步重排原子编号／坐标，检查概率谱变化 ≤`1e-5`。
   高于 `1e-4` 的 TV／相对 RMS 标为超过数值级别的响应；这不是准确率收益阈值。

## 汇总与判断

- 全体 train/valid 分别报告 eDOS/phDOS oracle/blind 中位 R²和失败率；valid 的两臂差和
  edge−关闭残差差使用既有成对 bootstrap 工具（2,000 次、固定 seed）报告区间。
- 同组成对共享材料，成对差异的区间按组成组重采样；同时报告对加权和组成组等权描述量。
- 分支：
  - 残差在 encoder 层的实际影响及关闭干预均处于数值级别：优先检查注入是否生效。
  - encoder 响应明确、decoder／谱响应接近数值级别：支持进一步调查读出利用率。
  - 谱响应明确但带符号谱差误差及全体指标未改善：不能认定“读出没收到信息”，应收窄到信号
    是否有用／是否对应标签；不直接设计扩大几何响应或增加损失权重的 pilot。
  - 没有可区分证据：记录未决点并结束本诊断，不强行选择架构或训练。
- 新训练仍须以全体 eDOS 中位 R²和失败率为主验收、blind/phDOS 为保护项；本轮只在证据充分时
  提出一个单因素 pilot 设计。

## 交付与验证

- 可重跑入口放在 `tools/eval/g2_structure_path_audit.py`；同一前缀下保存逐样本、成对、层残差、
  干预、汇总 CSV 与带输入哈希的 JSON。默认拒绝覆盖已有文件。
- 合成 CPU 测试覆盖同元素匹配的排列不变性、带符号谱差误差、内存 alpha 恢复与 hook 的输出
  等价性、仅 train/valid 读取约束；真实数据运行核对样本数、ID、history、有限值及 checkpoint
  运行前后的哈希。按同组成组顺序推理以限制内部表征缓存，结果恢复为原始材料顺序。

## 完成结果（2026-09-26）

- 全部推理及核验已完成。G2 开关影响 encoder、decoder 与谱输出；冻结 edge 依赖该分支，但
  相对独立训练 control 未得到准确率收益，同组成材料对的谱差误差也没有明确改善。
- 按上述证据不足时的结束分支关闭诊断，G2 保持 park，不提出新训练或架构改动。
- 完整证据见 `../logs/log-2026-09-26-g2-structure-path-audit.md` 和
  `../../results/g2_structure_path_q1.json` 及同前缀 CSV。
