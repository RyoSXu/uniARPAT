# 模型精度瓶颈与收益上界审计

日期：2026-09-28

## 范围

- 目标：针对近期候选连续 pass／park、没有进入默认模型的问题，增加“整体指标收益上界”前置门，
  重新排序模型精度方向。
- 口径：冻结 B7 `_e9ctl` epoch 33，只读 Q1 valid 2,313 条既有逐样本结果；不读取 test，不重新前向、
  不训练、不修改模型、数据、checkpoint 或评分定义。
- 输入：`edos_error_attribution_q1_valid_samples.csv`、`edos_spectral_support_q1_train_valid_samples.csv`
  和 `edos_spectral_support_q1_valid_pairs.csv`。按唯一 `mpid` 一对一连接，使用 B7 本身的
  oracle/blind R²，不混入后续 warm-start pilot control。
- 输出：`results/model_accuracy_upper_bound_q1_valid.csv`。它是反事实上界，不是候选的预期收益。

## 方法

对每个已知困难群体计算两种反事实：

1. **完美群体修复：**把该群体的 blind eDOS R²设为 1，其余样本不变。这回答“只修这类样本时，
   全体 valid 指标最多能动多少”。由目标或结果定义的群体不能用于部署路由，只用于淘汰低覆盖方向。
2. **oracle 尺度替换：**只把该群体的 blind R²替换为同一样本已有 oracle R²，谱形不变。这隔离
   H1 gamma 尺度恢复的最大空间，不表示真实 gamma 头能够达到 oracle。

B7 valid 基线为 blind `0.47314/9.12%`、oracle `0.51498/6.27%`，因此全体 oracle−blind 中位差为
`+0.04184`，失败率差为 `−2.85pt`。若从最差样本开始完美修复，至少要修复 76/2,313（3.29%）
才能让全体中位 R²提高 `0.02`；要让失败率下降 1pt，至少要纠正 24 个失败样本。

## 结果

| 群体 | n／占比 | 完美修复的全体 Δmedian | oracle尺度替换的全体 Δmedian／Δfail | 决策 |
|---|---:|---:|---:|---|
| 单元素 | 37／1.60% | +0.00539 | +0.00094／0.00pt | 上界不足，不能作全体精度主案 |
| coverage<1 | 58／2.51% | +0.01255 | +0.00031／−0.48pt | 上界不足；评分口径问题不等于主模型突破 |
| 高粗糙度（train p90） | 205／8.86% | +0.04352 | +0.00115／−0.48pt | 形状上界足够，但 slope-loss 已直接失败 |
| 同绝对组成 pair 覆盖材料 | 394／17.03% | +0.05942 | +0.00788／−0.61pt | 上界足够，但联合内容和谱差辅助均失败 |
| train-support Q4 | 595／25.72% | +0.14277 | +0.00424／−0.65pt | 形状上界大；D3a 当前数据路线未覆盖该缺口 |
| 高熵 valid Q4 | 579／25.03% | +0.12969 | +0.01937／−1.86pt | 形状上界大，但尚未定位可干预模块 |
| gamma 绝对误差 Q4 | 579／25.03% | +0.13047 | +0.03030／−2.59pt | 尺度上界超过主门，保留 |
| gamma 对数比误差 Q4 | 579／25.03% | +0.13321 | +0.03275／−2.77pt | 尺度上界超过主门，保留 |
| 全体 gamma oracle 替换 | 2,313／100% | — | +0.04184／−2.85pt | 当前唯一直接连接全体 blind 主指标的未闭合方向 |

群体有大量重叠，不能相加。例如 support Q4 与高粗糙度重叠 166 条，与 gamma 对数误差 Q4 重叠
160 条；同组成 pair 覆盖材料与 support Q4 只重叠 73 条。高粗糙度／高熵主要是 oracle 谱形问题，
gamma 误差是另一个尺度通道，二者不能混成同一根因。

## 调用链复核

- `model/transformer.py::Transformer.forward`：encoder memory 经
  `model/heads.py::global_masked_pool` 做有效原子平均，再由 `EtaHead` 的共享 MLP+sigmoid 同时输出
  `eta_ph` 与 `gamma_e`。
- `model/model.py::basemodel.train_one_step`：`eta_true` 与 `gamma_true` 拼接后使用统一普通 MSE，
  `eta_sup_w=1.0`；B7 epoch 33 的 `train_loss_eta=0.00570`，约为同轮两项主谱形损失之和的 1.6%。
- `utils/b7_cif_inference.py::predict_b7_blind`：部署时 eDOS 总量严格使用
  `N_val * gamma_pred / delta_edos`；因此 gamma 误差直接改变用户得到的物理谱，而 oracle 形状不受影响。
- Q1 valid 的 `gamma_true/pred` Spearman 为 `0.870`，说明头没有整体失效；但预测向中间收缩：真实
  gamma 的 99%／100% 分位为 `1/1`，预测对应为约 `0.967/0.997`，且 `gamma_true≈1` 的样本预测中位
  约 `0.911`。gamma 绝对误差与 oracle−blind gap 的 Spearman 为 `0.620`。

H1 在 pre-Q 上曾以“有界 eta/gamma 头”相对旧固定尺度通过并成为默认；Q1 重建后只复核了 blind gap。
C2.1b 检查的是 H1 辅助梯度是否压制主谱形，没有比较 gamma 头结构、训练目标或部署校准。因此不能把
“H1 已采用”解释为“Q1 gamma 已优化到没有可行动空间”。

## 方向裁决

1. **降级 encoder 几何消息。**G2a、联合边内容和谱差辅助都已用独立 valid 结果否定当前具体干预；
   单元素完美修复上界又不足。没有新的全体收益上界证据前，不继续更换消息函数或添加角机制。
2. **降级局部形状损失。**高粗糙度上界虽过线，但 slope loss 未改变总体或该组；同组成辅助也只让
   机制误差改善 0.465%。不再从表型直接派生另一种局部损失。
3. **保留谱形支持为长期瓶颈。**support Q4／高熵的理想上界很大，但 D3a、读出和现有辅助目标均未
   给出可应用机制，当前不实施。
4. **唯一进入下一设计：H1 gamma 校准。**它不解释 oracle 谱形误差，却直接决定部署用 blind eDOS，
   上界超过项目主门，且现有预测排序相关较高、存在系统性向中间收缩，适合先检验低参数校准能否
   收回足够差距。

最强反证是：`+0.04184` 只是使用真 gamma 的 oracle 上界，真实两参数校准必须收回约 48% 才能达到
`+0.02`；当前 Spearman 0.870 也说明剩余误差可能主要是样本级不可约残差，简单校准仍可能平局。
因此下一步只准入一次 train-fit／valid-only 的冻结校准门，不直接重构 H1 或启动 M1×10。

## OpenCode 执行记录

按用户偏好，曾并行派发 DeepSeek V4.1 Flash 的上界复算和 MiMo-V2.6-Flash 的调用链审计。前者在
写文件阶段连续 `ECONNRESET`，后者在限定时间内持续扩读历史日志而未形成报告；两者均被停止，未留下
仓库文件。上表由协调者用确定性本地复算生成并独立核对，不把未完成的模型输出作为证据。
