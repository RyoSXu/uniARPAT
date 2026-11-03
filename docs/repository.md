# 代码地图与仓库边界

本页按“调用链 → 文件 → 关键符号”帮助定位开发入口；当前科学配置见[固定契约](decisions.md)。

## 主调用链

`tools/train.py → ConfigBuilder / Dos_Dataset → basemodel / Transformer → LocalGlobalEncoder → decoder/head → loss 与 valid 选点 → checkpoint / 导出`

| 模块 | 文件 → 主要符号与职责 |
|---|---|
| 配置与训练 | [tools/train.py](../tools/train.py) → `train_and_eval`、`evaluate_local_global_split`；[experiment_config.py](../utils/experiment_config.py) → `ExperimentConfig`；[builder.py](../utils/builder.py) → `ConfigBuilder` |
| 数据读取 | [dataset.py](../utils/dataset.py) → `Dos_Dataset`；[crystal_cache.py](../utils/crystal_cache.py) → `PackedCrystalStore`、`crystal_data_contract`；[crystal.py](../utils/crystal.py) → `CrystalGeometry` |
| 完整原子计算 | [bucket_batch.py](../utils/bucket_batch.py) → `crystal_collate`、`automatic_pair_budget`、`split_crystal_batch`；按计算预算分块，不裁剪原子，训练按子批样本权重累积 |
| 几何与缓存 | [periodic_geometry.py](../utils/periodic_geometry.py) → `build_periodic_records`；[periodic_geometry_cache.py](../model/periodic_geometry_cache.py) → `PeriodicGeometryCache`；缓存独立于可训练状态 |
| 局部与全局 | [local_global_encoder.py](../model/local_global_encoder.py) → `LocalResidualLayer`、`ScalarGlobalLayer`、`LocalGlobalEncoder`；[periodic_messages.py](../model/periodic_messages.py) → `PeriodicLocalMessage`、`InvariantMessageBlock` |
| 模型与读出 | [model.py](../model/model.py) → `basemodel`；[transformer.py](../model/transformer.py) → `Transformer`、`TransformerDecoder`；[heads.py](../model/heads.py) → 总量头 |
| E＋S 与 blind 重建 | [spectral_balance.py](../utils/spectral_balance.py) → `load_balance_protocol`、`reconstruct_balanced_spectra`、`normalized_spectral_error`；[losses.py](../model/losses.py) → `nonzero_sumnorm_klw_loss` |
| 选点与恢复 | [checkpoint_selection.py](../utils/checkpoint_selection.py) → `selected_history_row`；[spectral_balance.py](../utils/spectral_balance.py) → `balanced_checkpoint_eligible`、`selected_energy_history_row`；[ablation_checkpoint.py](../utils/ablation_checkpoint.py) → `restore_ablation_checkpoint` |
| 正式导出 | [lg_valid_export.py](../tools/eval/lg_valid_export.py) → `load_run`、`select_best`、`export`；重核模型与选点契约后导出完整 valid |

旧路线开关保留兼容语义，不代表重新获准研究。ZP 不默认加载归档 G1/宏晶格等分支；旧 H1 重建仍在 `utils/spectral_inference.py`。

## 目录职责

| 位置 | 用途 |
|---|---|
| `model/`、`utils/` | 可导入的模型及支持库；保留 checkpoint 参数与契约边界 |
| `tools/` | 供人显式执行的训练、数据、检查、分析和历史 B7 CIF 入口；默认配置在 `training-defaults.yaml` |
| `data/crystal_q1_v1/`、`data/reference/` | 当前修复版缓存与固定参考；原始数组本地保存，少量清单/参考受版本控制 |
| `results/runs/`、`results/reports/`、`results/figures/` | 运行载荷、历史/摘要及图像；新文件不能仅因在“报告”目录就自动进入 Git |
| `docs/` | 常用说明；`design/`、`logs/` 分别保存带日期协议和证据，使用各自模板 |
| `tests/model/`、`tests/data/`、`tests/pipeline/` | 数学/模型、数据/几何、训练/导出/恢复等契约检查 |
| `archive/` | 历史 code/docs/data/results/metadata；实际复现还需原输入、环境与授权 |
| `.github/` | 手动触发的静态检查工作流；检查依赖见[测试说明](../tests/README.md) |

## 路径与历史身份

默认输入/输出按仓库位置定位，用户显式传入的相对路径按启动目录解释。
`data/train4ARPAT/` 是旧 Q1，不可替代修复版；本机未必保有旧数组。`data/quarantine_q1.json` 仅保留旧清单路径兼容。
历史网格与 pre-Q 不作为当前训练输入。

旧路径、改名与源码身份由 [迁移登记](../archive/metadata/repository-migration-2026-10-04.json)和 [utils/paths.py](../utils/paths.py) 解析：
`repository_path` 负责迁后位置，`historical_path` 核对原字节，`protected_path` 还要求执行源码符合迁后登记。
历史 manifest 的路径与哈希不批量改写；缺失旧源码/载荷时不能宣称历史链完整复核。

设计/日志使用创建日期命名，更新内容不改编号；模板见[设计](design/template.md)和[日志](logs/template.md)。
科学运行记录中的本地产物路径不是仓库附件，迁机保存范围见[交接](handoff.md)。

## Git 边界

[.gitignore](../.gitignore) 排除权重、数组、逐材料表、下载载荷、日志和缓存；获准小型元信息以完整路径开放。
任何可提交文件上限 **2 MiB**；强制加入被忽略载荷也会被检查拒绝。
[check_repository.py](../tools/checks/check_repository.py) 可检查 worktree/index/revision；提交钩子查暂存区，推送钩子查新增对象与最终树，中途加入再删除的大载荷仍被拦截。

新克隆需执行 `git config core.hooksPath tools/checks`；该设置不随 Git 克隆传播，不能仅凭旧文档认定已启用。
历史仍含大对象，未重写历史；向新远端完整迁移须另定方案。提交与推送都按用户明确授权执行。
