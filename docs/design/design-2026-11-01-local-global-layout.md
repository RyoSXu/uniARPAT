# 设计：在LR2＋E＋S上把第二次局部更新移到全局中段

用户于2026-11-01同意准备这一组对照。当前工作点仍为已采用的staged LR2＋E＋S ep29；
本轮先实现与短检，正式35轮训练交给OpenCode，不重训对照。

## 问题与唯一干预

当前顺序为`L1 L2 G1 G2 G3 G4 G5 G6`。假设：第二次完整局部更新处理已经汇集全局上下文的
原子状态，可能更有利于谱预测与同组成结构谱差。候选明确为`L1 G1 G2 G3 L2 G4 G5 G6`。

这不是已有`alternating`开关的含义；L2/G6下原开关为`L1 G1 L2 G2 G3 G4 G5 G6`，保持原义。
早期旧骨架、旧loss、旧选点的编排试验不能替代当前工作配方的验证。
局部＋全局组合有[GraphGPS等方法先例](https://arxiv.org/abs/2205.12454)，但不证明这一具体顺序更好。
已完成的阶段读出也没有把瓶颈定位到全局层，本次结果只约束所测编排。

## 对照、共同条件与预算

- 对照：`ablation_m1_sbal_lr2es20261030_lr2_s42_e35`的原选中best ep29，完整35轮，直接复用。
  [结果及采用记录](../logs/log-2026-10-30-spectral-residual-run.md)
- 候选：`local_route=LR2`、`local_layout=midpoint`；仍是两次完整A调用／四个内部块、六次全局注意力与FFN。
- 64/128局部宽度，`LN64(F(s,R))`、内部post-norm、外层pre-norm残差、完整状态读出保持。
- ZP、Q1几何入口、6 Å严格截断、完整原子与共享静态几何保持；全局post-norm、概率dropout=0、
  Q/K/V/O恒等初值后学习、其他dropout=0.05、8头、2048维GELU FFN、6层共享decoder保持。
  这些是本轮控制条件，不宣称最优。
- 模块顺序仅改变调用关系；独立模块对象、所有参数键／形状／量、未训练初值及初始化RNG行为相同。
  从头训练，不载入已训练的ep29或旧H1权重。
- 每原子无固定上限总量、`0.5 mean(E_e+S_e)+0.5 mean(E_p+S_p)`联合loss不变。
  逐字复制旧protocol，不重新拟合train常数或设置保护线。
- 只运行seed42、35轮、batch32、lr5e-5、FP32这一组。仅有效train 18,704条与完整valid 2,313条；
  不评估test，不改数据、归一化、coverage、网格、门控或其他读出候选。
- 按已完成staged组成本预估约4–5小时单卡，短检后确认可运行性，第一完整轮后更新ETA。
  两批短检不能覆盖整轮耗时或全数据峰值显存。

## 选点和判读

仍按完整valid的`balanced_blind_energy`：eDOS／phDOS失败数分别≤196／99，
两任务中位R²分别≥0.4817208648−0.02／0.7407193780−0.02，再最小化等权mean E。
S、Oracle与同组成谱差不参与选点；平局保留较早轮次，无合格best不得用latest替代。

与staged ep29在同一材料上配对，差值统一为midpoint−staged。
材料配对bootstrap 2,000次、seed20261101，同一抽样矩阵用于两任务和联合指标。
分别报告E均值／中位／P90、R²中位／P10、失败数与失败率，以及成本。
通过固定保护且联合E差95%区间上界<0，支持本次总体改善，采用权交用户；否则工作点保持。

同组成signed谱形差按绝对／约化组成分别生成配对，以组成组为重采样单位，2,000次、seed42。
分别报告两任务候选相对staged的组均误差差及区间、相对零响应的技能、预测／目标谱差幅度。
相应任务和组成口径的误差差区间上界<0，才支持该项谱差改善；幅度变大本身不构成改善。
不能用总体E改善替代谱差判断，也不能将某一项结果写成两任务普遍改善。
五条待核材料仍纳入正式评分，单列贡献。Oracle和总量／谱形分解只作诊断。

两类结果分开记录：总体改善而谱差不明，只支持总体收益；谱差某项改善而主目标不明，
只保留对应诊断，不自动替换工作点；两者都不明或保护失败，结束本候选，不自动加预算或扫描更多顺序。
单seed材料／组成组重采样区间条件于已选checkpoint，不代表训练种子稳定性或消除valid反复选型偏差。

## 实现及验收

`tools/train.py --local_layout midpoint` → 配置与独立契约 → 共同初值 →
`LocalGlobalEncoder.execution_order`执行固定序列 → 原E＋S训练／选点 → 原完整valid导出。

- `model/local_global_encoder.py`仅增加受限的`MIDPOINT_ORDER`；`model/transformer.py`限制为LR2 per_atom路径。
- `utils/spectral_layout_experiment.py`定义干预契约，验证实际顺序并复用原LR2的模块检查，不修改旧验证规则。
- `utils/spectral_balance_experiment.py`只在新编排下使用独立schema；参考初始化仍来自canonical staged A64。
- `tools/eval/spectral_layout_prepare.py`复用原准备器，绑定已完成staged LR2对照，打印唯一命令，不启动训练。
- `tools/checks/check_spectral_balance.py`复用真实train批次短检；导出器核对新schema、布局及完整干预定义。

测试覆盖实际调用顺序和L2输入、手写顺序的前向／梯度、共同初值及RNG、旧契约保持、
同形状checkpoint隔离与恢复、配置／导出漂移拒绝，以及小型FP32／FP64几何的E(3)和原子重排。
这些有限案例不等于完整训练模型的全面晶胞／超胞一致性验收。

实际准备与短检见[日志](../logs/log-2026-11-01-local-global-layout-preparation.md)；正式训练、完整valid配对与结果判读见[运行日志](../logs/log-2026-11-01-local-global-layout-run.md)。
