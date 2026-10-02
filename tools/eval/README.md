# 评估工具

可复用的分析与结论判定脚本放在这里。脚本必须读取受版本控制的结果 CSV，说明输入数据口径
（Q1 或 pre-Q）和 oracle/blind 模式，并写出可复现的摘要。不得把一次性结论逻辑放在 `/tmp`。

请使用描述性名称，例如 `c5_pilot_verdict.py`。图和临时输出应保留在此目录外，除非它们本身是
有意纳入项目的产物。

## 元素身份与纯原子序号实验

`element_identity_preflight.py` / `element_identity_valid_verdict.py`：A100/B100 的实施核查、
manifest 与 Q1 valid 判读；`eid_zonly_*` 对应 Z100，`eid_zproj_*` 对应 ZP100。
其中 `*_monitor.py` 读取进程、history 与产物状态，`*_preflight.py` 会执行数值检查并写入
实验记录；历史 preflight 命令不能直接用来覆盖已有 manifest。

`element_identity_diagnosis.py`：对已有 B100/Z100/ZP100 做 train/valid 拟合、参数与优化器、
当前梯度、表示及注意力诊断，设计见 `docs/design/design-element-identity-diagnosis.md`。
正式结果目录为 `results/eid_diagnosis_s42/20261002T095909Z/`，包含追溯快照；中断目录
`20261002T093859Z/` 按用户决定保留本地并忽略。原报告的解释边界见
`docs/design/design-element-initialization.md`，当前标准为 ZP100 baseline 与 B7 最优指标参考。

## 既有结构与读出诊断

`g2_structure_path_audit.py`：冻结 G2 epoch 10 两臂，仅评估 Q1 train/valid，并在内存中关闭
G2 残差，追踪 encoder／decoder／谱输出响应与误差；设计见
`docs/design/design-g2-structure-path-audit.md`。默认拒绝覆盖已有诊断结果。

`g2_frozen_readout_probe.py`：固定 G2 encoder，复用原 eDOS 读出，对比正确对应与组内随机对应的
谱差训练；在训练中未出现的约化组成上验收。设计与固定预算见
`docs/design/design-g2-frozen-readout-probe.md`；拒绝覆盖已有运行目录及结果。

`g2_encoder_adaptation_probe.py`：沿用冻结读出配对任务，只开放现有encoder更新；相同初值、
读出优化规则及1620步预算，对比已保存的冻结matched结果。blind尺度从更新后的表征重算；
设计见 `docs/design/design-g2-encoder-adaptation-probe.md`；源文件、数据与旧结果均校验哈希。

`g2_small_fit_probe.py`：标签无关抽取16个train组成对，先核对输入区分性，再比较冻结／联合
实际网络各2000步的小样本拟合；保存首次全部达到逐对门槛的权重与末步结果，只检验训练可拟合性。
设计见 `docs/design/design-g2-small-fit-probe.md`；只加载train，拒绝覆盖已有产物。
`plot_g2_small_fit.py`从完成后的正式history绘制各检查点的最差pair残留和通过数量，不改变判据。

`g2_value_fit_probe.py`：复用同一16对缓存和两臂对照，只更新6个G2消息模块与原读出；
训练前检查跨冻结层梯度，结束后逐值检查冻结边界并重载最终／首次拟合权重。
设计见 `docs/design/design-g2-value-fit-probe.md`；旧执行快照与当前helper分别登记哈希。
若2000步已完成而结果导出中断，可用`--finalize-existing`从现有最终权重汇总；先核对旧执行
快照与保护输入，不执行训练。缺失的中间逐对轨迹会明确标记，不能由末步预测补造。

`g2_non_value_fit_probe.py`：冻结G2消息参数，只训练其余encoder与读出的16对互补臂；
复用同一训练、梯度边界和重载实现，保留冻结模块的反向传播。比较包含前三个臂，并保留
已有缺失明细标记；设计见`docs/design/design-g2-non-value-fit-probe.md`。

`g2_last_layer_fit_probe.py`：固定G2与前5个普通encoder层，只训练最后一层及读出；沿用同一
16对和2000步。首次／最终权重重载时核对冻结参数及缓冲区；设计见
`docs/design/design-g2-last-layer-fit-probe.md`。
