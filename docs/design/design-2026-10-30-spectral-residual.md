# 设计：在当前E＋S配方验证LR2

用户于2026-10-30同意推进。只新增一组训练，不加载旧LR2训练权重；正式训练交由OpenCode执行和监测。
这是旧候选向当前联合任务的受控迁移，不是继续搜索局部公式。

进度同步：35轮已完成，best ep29；独立完整valid复跑和原谱复算支持本次联合误差改善，
通过固定保护，同组成谱差四项区间仍含0。用户于2026-10-31确认采用LR2＋E＋S为研究工作点，
原A64 E＋S ep25保留匹配对照，旧A64 ep31仍为选点保护参照；没有重选历史checkpoint。
[正式结果与独立核验](../logs/log-2026-10-30-spectral-residual-run.md)。以下保留执行前设计及判据。

## 问题与唯一改变

旧LR2在旧H1总量头、旧谱形loss和blind中位R²选点下取得收益，但其所选轮次eDOS失败212条，
超过当前固定保护线196条。旧结果不能证明它适用于当前配方，也不能回溯选择未保存的轮次。

主对照为已完成的完整A64 E＋S best ep25，直接复用，不重训：
`results/runs/spectral_shape_aux_20261025/report.json`。
唯一候选保留完整A消息、64/128宽度、内部post-norm、完整状态读出和外层pre-norm残差，
将四个内部块从`LN64(s+F(s,R))`改为`LN64(F(s,R))`。
状态s仍进入F的消息及更新计算；这不等于删除所有原子信息通路，也不保证结构敏感性增强。

## 冻结条件

- 当前每原子非负、无固定上限总量头；不使用N_val侧车重建。
- 联合loss仍为`0.5 mean(E_e+S_e)+0.5 mean(E_p+S_p)`；S系数1，零谱S=0且按全批次平均。
- 逐字复制原protocol，保留train拟合常数、固定旧A64保护与完整valid选点，不重新拟合。
- 相同未训练初值、参数键和参数量；不得加载旧H1/LR2或已训练E＋S权重作为初始化。
- seed42、35轮、batch32、lr5e-5、FP32；ZP、staged L2/G6、共享6层decoder及其他骨架设置不变。
- 仅当前有效train 18,704条与完整valid 2,313条；不评价test，不改coverage、标签、网格或归一化。
- 不叠加双decoder、门控、物理声子头、AMP或其他候选。

## 选点与判断

仍要求两任务失败数分别≤196/99，中位R²分别≥0.4817208648−0.02、0.7407193780−0.02。
合格轮次中最小化等权blind mean E，S与Oracle不参与选点；平局取较早轮次。
无合格轮次报告`no_eligible_best`，不得用latest或旧规则补选。

与E＋S ep25在同一批材料上配对比较；差值统一为LR2−A64。
报告联合及各任务E均值、中位数、P90，R²中位数、P10和失败数/率。
材料配对bootstrap 2,000次、seed20261030，同一抽样用于两任务与联合指标。

- 通过固定保护且联合mean E差的95%区间上界<0：支持LR2在本次当前配方中改善主目标，交用户决定是否采用。
- 区间含0、恶化或无合格best：保留A64工作参考，结束本候选，不自动调参或追加预算。
- 分别解释两任务变化；联合改善不代表每一任务、尾部或每个材料都改善。

Oracle、总量/谱形误差、同组成signed谱差只作诊断。沿用已有绝对/约化组成分组与组配对bootstrap，
不将相关材料对当独立样本；5条待核电子谱仍纳入正式评分，另列贡献。
区间条件于所选checkpoint，不表示跨seed稳定性，也不能消除反复使用valid的选择偏差。

## 实现与执行

训练调用链：已有`local_route=LR2` → `validate_balance_config`接受固定E＋S配置 →
`initialize_balance_model`复制A64原始初值并验证四个加法关闭 → 原loss、评估及选点 → 原valid导出。

- `utils/local_residual_experiment.py::residual_ablation_contract`共用已有干预定义，旧契约内容不变。
- `utils/spectral_balance_experiment.py`为LR2 E＋S使用独立schema，保留原A64与旧LR2协议。
- `tools/eval/spectral_balance_prepare.py::completed_spectral_control`核验完成的主对照；旧纯E入口仍可用。
- `tools/eval/spectral_residual_prepare.py::prepare`绑定主对照、输入与源码，输出一组命令，不启动训练。
- `tools/checks/check_spectral_balance.py::check`复用真实批次短检；按实际初值指纹比较，允许旧记录缺少后来新增的记账字段。
- `tools/eval/lg_valid_export.py::load_run`核验新schema、LR2路线及干预公式一致。

正式训练前核对准备/短检的全部输入与源码哈希。前台工具曾约一小时终止进程，故训练应从启动时
脱离会话，保存PID、完整stdout与退出码；中断只允许原tag、原契约、连续history对应的latest恢复。
无异常不重试，不因早期指标改变配方。旧A64 E＋S全程约4小时，预估本次4–5小时；
两批短检不是全程速度或峰值显存保证，第一完整轮后再更新ETA。

准备和实际验证见[本轮日志](../logs/log-2026-10-30-spectral-residual-preparation.md)。
