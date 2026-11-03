# 分析、对照与验收工具

当前配方为 **staged LR2＋E＋S，best ep29**；科学状态见[状态页](../../docs/status.md)，产物恢复见[交接](../../docs/handoff.md)。
下面的已完成入口服务复核和经批准的复现，不构成待执行队列。旧目标/总量/选点的工具不能只改标签就当作当前协议实验。

## 当前协议相关入口

| 入口 | 职责与依赖 |
|---|---|
| `spectral_balance_prepare.py` | 从有效 train 拟合 E 常数并绑定固定 valid 保护，或以 `--shape-aux --control-preparation` 准备 E＋S；只生成协议/命令，不训练 |
| `spectral_residual_prepare.py` | 基于已完成 A64 E＋S 准备 LR2 单候选；对应[已采用运行](../../docs/logs/log-2026-10-30-spectral-residual-run.md) |
| `spectral_layout_prepare.py` | 基于已采用 staged LR2 ep29 准备 midpoint；[对照已完成，未采用](../../docs/logs/log-2026-11-01-local-global-layout-run.md) |
| `lg_valid_export.py` | 认证配置、history 选点与 best 契约，前向完整 valid 并导出；无合格 best 时拒绝导出。`export(..., spectra_out=...)` 可保存原谱，CLI 只提供逐材料/摘要输出 |
| `lg_paired_compare.py` | 既有局部—全局对照的配对诊断；新 E 协议结果还须按冻结定义从原谱复算，不能直接复用旧主指标 |
| `spectral_contract_audit.py` | 只读冻结标签/预测，审查总量、coverage 与历史指标；不改正式数据或训练协议 |
| `shape_aux_stage_readout.py`、`shape_aux_group_comparability.py` | 已完成的阶段表示与组可比性诊断；依赖各自冻结模型/产物，不代表已定位泛化瓶颈 |
| `crystal_model_invariance.py` | 冻结完整 LR2，检查 E(3)、重排、晶胞与超胞；包含截断边界反例，不能把 overall fail 隐去而只报真实样本通过 |
| `cutoff_boundary_census.py` | 有效 train 的几何全量统计与固定旋转计数探针，不读谱标签、不做模型前向；6.02 Å 只是审计包络，生产仍严格 6 Å |

新实验须另定协议和唯一标签，再用对应 `tools/checks/` 预检绑定输入/源码；准备报告中的原始 command 才是该次执行依据。
当前配方显式启用 LR2、per_atom、冻结 protocol、谱形辅助及 `balanced_blind_energy`，裸 CLI 默认保留历史行为。
长时间训练/监控由准备 agent 提供提示词，用户交给执行 agent，见[工作流程](../../docs/workflow.md)。

## 保留的历史对照

| 主题 | 入口或文件模式 |
|---|---|
| 首轮 A/B+、编排和深度 | `local_global_compare.py`、`local_global_depth_compare.py` |
| 旧/新骨架及三项全局归因 | `global_only_compare.py`、`legacy_backbone_compare.py`、`global_norm_compare.py`、`attention_{dropout,projection}_compare.py` |
| I1/I2/I3/I4 与完整模块选型 | `i1_{a,bplus}_compare.py`、`local_readout_compare.py`、`local_module_compare.py` |
| 容量、归一化、读出、内部残差 | `local_width{,_fresh}_compare.py`、`local_norm_compare.py`、`local_outer_norm_compare.py`、`local_increment_compare.py`、`local_residual_compare.py` |
| 几何、消息与性能验收 | `periodic_*`、`verify_periodic_*`；结果须区分数学测试、短批成本与正式训练 |

各工具原设计和最终结果从[模型证据导航](../../docs/index.md#模型设计)进入；旧 H1 LR2 结果不能替代 E＋S 迁移结果。
部分更早工具位于 [archive/code/tools/eval/](../../archive/code/tools/eval/)，按原契约显式导入。

## 路径与输出

用相应脚本的 `--help` 查参数；默认输入/输出按仓库根定位，自定义相对路径按启动目录解释。
原始预测、逐材料表、源码快照和运行日志放本地 `results/runs/`，新输出拒绝覆盖；不要改写已冻结协议或放宽源码哈希认证。
已有证据验证器只认证其检查的保存材料，不能替代重新前向模型或证明精度。

历史 CIF 入口为 [tools/infer_b7_cif.py](../infer_b7_cif.py)，只支持 B7 `_e9ctl` ep33；拒绝不匹配 checkpoint、无序结构及超过旧 80 原子限制的 CIF，不适用于当前 LR2。
