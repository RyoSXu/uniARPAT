# 设计：G2 encoder 与读出的联合适配实验

## 授权与待判定问题

- 马尚酱在冻结读出结果后确认“好，继续”，本轮只增加一个因素：现有 encoder 是否参与更新。
  上轮 matched 是冻结 encoder 对照；本轮 joint 从同一 G2 ep10 初值训练 encoder 和原读出。
- 待判定：相同结构谱差任务下，开放 encoder 更新能否改善冻结读出的留出组成误差？成功只支持
  此初始化、目标、预算下联合适配的作用；失败不证明 encoder 没有结构信息，也不定位具体坏层。
- 沿用上轮纯谱差目标，暂不增加完整谱形约束。它是机制干预，不能直接替换生产模型。

## 固定对照与数据

- 源：`output/ablation_m1_g2edge/checkpoint_latest.pth`，Q1 M1、ep10、seed42、FP32。
- 对照复用 `results/g2_frozen_readout_q1_*`：original、matched、shuffled，以及零谱差参考。
  新臂必须从源 checkpoint 初始化，不能接着上轮 matched_final 训练。
- 配对清单逐项复核后与上轮完全一致：train 2591对／3115材料／1198约化组成；valid 320对／
  394材料；主valid为原Q1 train从未出现过的117约化组成、163对、256材料。
- 只读取train／valid；验证标签不用来选步数、参数或checkpoint。末步统一评估，完整保留连续
  结果；本轮一次训练不估计训练seed不确定性。

## 调用链与唯一改动

`冻结元素嵌入 → atom_src + 固定几何 → 可训练现有encoder → 可训练原eDOS读出`

- 通过源模型 encoder 的前置观察钩子缓存实际 `atom_src`、原子间距离／方向和G2边；只去掉
  padding，重组batch时恢复mask并重编号边。输入不含标签派生特征，不复制另一套生产前向公式。
- encoder包含现有6层的RP投影、前馈、归一化和6个G2消息模块；元素嵌入、融合投影保持冻结。
  使用原 `TransformerEncoder.forward`，不换架构、不添加通路、不改生产文件。
- 读出复用 `FrozenEdosReadout`：decoder、eDOS query、target token、CNN head，原权重初始化。
- 两块各用独立AdamW（lr5e-5、betas0.9/0.99、weight_decay0.01）和各自梯度范数裁剪1.0。
  这样读出保留上轮的优化规则，不因多出encoder梯度而改变读出裁剪系数。
- 两块都保持eval模式、关闭dropout而开启梯度；FP32、16对/batch、seed20260926顺序、每轮
  无放回且保留尾batch，10轮／1620步。组权重和带符号谱差MSE复用上轮实现。
- H1头参数冻结，但blind评估必须使用**新encoder输出重新计算gamma**。不能继续沿用旧表征的
  缓存blind尺度冒充当前盲推理。训练目标不含H1；oracle只用于另报缩放正确时的谱形准确率。

## 开跑检查与验收

- 源配置、checkpoint、Q1输入与上轮登记哈希匹配；可安全加载的旧特征缓存与兼容性记录匹配。
  开始前保存本轮设计／脚本副本和输入哈希，结束再次校验，拒绝覆盖已有目录或结果。
- 全部配对材料：新缓存前向复现源预测最大TV≤1e-5、blind尺度相对误差≤1e-5。
  首个训练batch：关闭encoder梯度时，与旧缓存读出前向的损失相对误差≤1e-5、读出梯度相对
  RMS≤1e-3。这些检查通过才进入训练；不根据valid表现调整实验。
- 合成测试覆盖不规则padding／重复材料的边重编号、初始前向和尺度等价、encoder／读出有更新
  而源参数／H1不变、读出裁剪独立，以及结果门槛，防止实现差异冒充实验因素。
- 主指标仍为带符号谱差TV误差，先组内平均再约化组成等权；按同一组成做2000次bootstrap。
  **联合适配获支持：**主valid相对冻结matched、original、shuffled和零谱差每项均至少改善5%，
  且改善95%区间下限都大于0。冻结matched是主因果对照，其余防止只胜过退化参照。
- train相对这四项每项改善至少10%只记训练内学习。留出不满足时，具体的联合适配方案未获
  泛化支持；同时报告相对冻结matched的连续效应，不能把“没过门槛”写成“完全没作用”。
- 报告谱差MSE、预测／目标TV、谱差方向余弦、配对材料eDOS oracle/blind中位R²和失败率。
  phDOS未按目标训练，不声称保护通过；10轮指机制任务的1620次更新，不等于完整Q1训练10轮。

## 产物与结束规则

- 新入口 `tools/eval/g2_encoder_adaptation_probe.py`，复用已有配对、损失、指标和bootstrap。
- 标签／目录 `output/g2_encoder_adaptation_q1/`，长运行使用setsid + nohup；保存输入缓存、
  设计与脚本副本、最终encoder／readout／H1状态、预测和history，保留源产物。
- 正式结果前缀 `results/g2_encoder_adaptation_q1`；单独日志记录结果并更新status和index。
  不自动延长训练、不追加超参扫描；下一项根据本轮可排除的解释提出，不自动启动。
