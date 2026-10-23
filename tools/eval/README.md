# 当前分析与验收入口

本目录包含周期几何与 A/B+ 验收、局部—全局对照、材料配对诊断、结果复核和绘图入口。
当前科学状态与执行授权以 [docs/status.md](../../docs/status.md) 为准。

历史实验、冻结原型和对照位于 [archive/code/tools/eval](../../archive/code/tools/eval/)。
当前验证器显式引用归档工具；历史 manifest 的旧路径/哈希由路径库核验，不能放宽源码保护。
已有证据复核仅检查保存材料的身份与一致性，不重新运行神经网络或证明 DOS 精度。

在仓库外可用工具的完整路径查看 `--help`；默认输入/输出依仓库根定位，自定义路径依启动目录解释。
新的诊断原始载荷默认进入本地运行目录，正式小摘要需审查后才能添加 Git 精确例外。
实验预算与资源预检按已确认设计执行；保留既有证据，使用新的隔离输出目录。

`local_readout_compare.py` 对应[局部状态直连读出](../../docs/design/design-2026-10-15-local-readout-interface.md)：
默认打印 I1/I2/I4 三条命令；`--run` 校验新预检后依次训练并分析，`--summarize` 只读历史，
`--analyze` 只分析完成的运行。三组都使用联合 blind R² 选点，Oracle 与组成谱差仅作诊断。

迁移后只有源码与数据时，使用 `local_width_fresh_compare.py` 从头建立A64/W1两组容量对照，
不依赖旧权重或预测。先运行 `tools/checks/check_local_width_fresh.py` 完成本机短检；
配方、成本与交接见[迁移后容量对照](../../docs/logs/log-2026-10-19-local-width-fresh-preparation.md)。

`local_norm_compare.py` 比较完整A内部pre/post-norm：仅新增训练LN1，复用迁移后的A64权重与train/valid导出。
先用 `tools/checks/check_local_norm.py` 认证历史Git源码、参考产物及当前默认行为，再完成GPU短检；
默认打印命令，`--run`才训练，`--summarize`只读进度。方案及边界见
[局部内部归一化](../../docs/design/design-2026-10-20-local-internal-norm.md)。

`local_outer_norm_compare.py` 使用同一认证、队列和统计流程，单独检验完整A外层512维post-norm（LN2）；
内部64维post-norm保持不变，复用A64外层pre-norm参考，只新增一组训练。
先运行 `tools/checks/check_local_outer_norm.py`；默认打印命令，`--run`才训练。
见[外层归一化设计](../../docs/design/design-2026-10-21-local-outer-norm.md)与[交接](../../docs/logs/log-2026-10-21-local-outer-norm-preparation.md)。

`local_increment_compare.py` 对照完整A的完整状态与增量读出（LR1），保留输出偏置、内外残差和原归一化位置。
先运行 `tools/checks/check_local_increment.py`；仅新增训练LR1，复用认证的A64结果与导出。
默认打印命令，`--run`才训练；见[设计](../../docs/design/design-2026-10-21-local-increment-readout.md)和
[交接](../../docs/logs/log-2026-10-21-local-increment-preparation.md)。

`local_residual_compare.py` 只检验四个A内部块是否显式加回输入状态（LR2），保留外层残差、原归一化及完整状态读出。
先运行 `tools/checks/check_local_residual.py`，再用新预检训练LR2；复用认证A64与冻结导出。
默认打印命令，`--run`才训练；见[设计](../../docs/design/design-2026-10-21-local-internal-residual.md)和
[交接](../../docs/logs/log-2026-10-21-local-internal-residual-preparation.md)。

`spectral_contract_audit.py` 只读取冻结train/valid标签、已有预测和history，核查eDOS窗口积分、
H1输出范围、覆盖/补零对R²的影响及回顾性选点取舍；不构造模型或修改正式协议。
传入`--analysis-dir`与新的`--out`目录，拒绝覆盖既有证据；需要恢复既有预测产物。
命令、解析测试及解释边界见[科学口径审查](../../docs/logs/log-2026-10-22-spectral-contract-audit.md)。

`spectral_balance_prepare.py` 仅用有效train拟合每任务谱能量常数，绑定A64完整valid失败率参考，
生成独立的每原子blind全谱loss/总量/选点协议与待执行命令。先运行
`tools/checks/check_spectral_balance.py` 完成实批短检，再执行打印命令；不会自动启动训练。
训练后原`lg_valid_export.py`可校验约束选点并导出新误差；无合格best时明确拒绝导出。
见[新目标设计](../../docs/design/design-2026-10-23-balanced-blind-objective.md)及
[准备与交接](../../docs/logs/log-2026-10-23-balanced-blind-preparation.md)。
