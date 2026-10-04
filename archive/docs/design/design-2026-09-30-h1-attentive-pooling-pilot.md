# H1 可学习原子池化 pilot：撤回记录

状态：**已退出当前主线；未实现、未训练。**本候选只改变 H1 从 encoder 原子表示恢复 eta/gamma 的
聚合方式，主要针对逐材料尺度误差；当前主线已转为结构条件谱形突破，因此不继续实施或排入近期计划。

这项撤回只表示它不符合当前优先级，不证明所有尺度预测方案无效，也不否认 gamma oracle 上界。只有用户
重新选择尺度预测路线时，才重新评审该候选。

原候选分析、调用链核验与停止条件保留在
[`log-2026-09-28-h1-next-candidate-review.md`](../logs/log-2026-09-28-h1-next-candidate-review.md)；
全局两参数校准的实际结果见
[`log-2026-09-28-h1-gamma-calibration-gate.md`](../logs/log-2026-09-28-h1-gamma-calibration-gate.md)。
