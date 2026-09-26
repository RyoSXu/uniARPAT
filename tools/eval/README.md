# 评估工具

可复用的分析与结论判定脚本放在这里。脚本必须读取受版本控制的结果 CSV，说明输入数据口径
（Q1 或 pre-Q）和 oracle/blind 模式，并写出可复现的摘要。不得把一次性结论逻辑放在 `/tmp`。

请使用描述性名称，例如 `c5_pilot_verdict.py`。图和临时输出应保留在此目录外，除非它们本身是
有意纳入项目的产物。

`g2_structure_path_audit.py`：冻结 G2 epoch 10 两臂，仅评估 Q1 train/valid，并在内存中关闭
G2 残差，追踪 encoder／decoder／谱输出响应与误差；设计见
`docs/design/design-g2-structure-path-audit.md`。默认拒绝覆盖已有诊断结果。

`g2_frozen_readout_probe.py`：固定 G2 encoder，复用原 eDOS 读出，对比正确对应与组内随机对应的
谱差训练；在训练中未出现的约化组成上验收。设计与固定预算见
`docs/design/design-g2-frozen-readout-probe.md`；拒绝覆盖已有运行目录及结果。

`g2_encoder_adaptation_probe.py`：沿用冻结读出配对任务，只开放现有encoder更新；相同初值、
读出优化规则及1620步预算，对比已保存的冻结matched结果。blind尺度从更新后的表征重算；
设计见 `docs/design/design-g2-encoder-adaptation-probe.md`；源文件、数据与旧结果均校验哈希。
