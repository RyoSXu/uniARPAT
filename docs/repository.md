# 仓库目录与文件约定

仓库保留 8 个业务一级目录。`.github/` 保存自动检查配置，`.git/` 是 Git 内部状态。
完整改名与兼容关系见[迁移登记](../archive/metadata/repository-migration-2026-10-04.json)，
实际验收见[整理日志](logs/log-2026-10-04-repository-cleanup.md)和[验证记录](../archive/metadata/repository-validation-2026-10-04.json)。

| 目录或文件 | 用途与边界 |
|---|---|
| `model/` | 当前模型、损失、输出头及独立 A/B+ 消息库；保留 checkpoint 参数键 |
| `utils/` | 可导入的支持库：数据集、配置、恢复、指标、几何、热力学和路径；不存数据载荷 |
| `tools/` | 显式运行的入口；`train.py` 训练/评估，`infer_b7_cif.py` CIF 推理，`training-defaults.yaml` 默认配置 |
| `tools/data/` | 获取、划分及重建工具；执行前须确认输入、输出和数据契约 |
| `tools/eval/` | 对照实验、ZP/局部结构诊断、周期几何/A/B+ 验收、证据复核与绘图入口 |
| `tools/checks/` | 静态检查、Git 文件与历史载荷检查，以及提交/推送钩子 |
| `data/reference/` | 固定划分、元素表、价电子/截断表、排除清单、网格与审计 |
| `data/crystal_q1_v1/` | 当前默认的完整结构修复版 Q1；字段及有效划分见 [data.md](data.md) |
| `data/train4ARPAT/` | 原 Q1，仅用于历史模型复核；保留原数组与顺序 |
| `data/quarantine_q1.json` | 指向参考清单的兼容链接；保留 Q1 原 manifest 的旧路径字段 |
| `results/reports/` | 当前正式结论和有限证据；Git 仅允许已审查的完整路径 |
| `results/runs/` | 模型权重、有效配置、数组、逐样本表和运行日志，本地保存 |
| `results/figures/` | 当前正式图像位置；新增图像须审查体积和用途 |
| `docs/` | 状态、导航、流程、数据、术语、决策、目录约定 |
| `docs/design/`、`docs/logs/` | 当前设计和证据；各自保存 `template.md` |
| `tests/model/` | 当前模型、输出头、消息模块的数学与回归测试 |
| `tests/data/` | 周期邻居和物理记录测试 |
| `tests/pipeline/` | 训练恢复、推理、路径、Git 边界及支持库的调用契约 |
| `archive/code/` | 退役方案的 `model/utils/tools/tests`；旧入口与旧热力学批处理，独立复现时显式导入 |
| `archive/docs/` | 历史 `design/logs`；保留结论及其当前证据链接 |
| `archive/data/` | v1/v2 历史缓存与旧网格；机器本地，不可自动替代 Q1 |
| `archive/results/` | 历史 `reports/runs/figures`；权重和原始数据本地，有限摘要受控保留 |
| `archive/metadata/` | 迁移登记、汇总验证、环境快照及必要的原始源码身份 |

根目录只保留 `README.md`、`AGENTS.md`、`.gitignore`、`requirements.txt`、`ruff.toml`。
带本机路径的旧依赖锁表移为历史环境快照，不作为跨机器安装要求。
`utils` 与 `tools` 分别承担“供代码导入”和“供人执行”，避免循环依赖；
`Dos_Dataset` 位于 `utils/dataset.py`，不需要独立的 `datasets/` 目录。

## 输入、输出与历史复现

默认仓库输入/输出按文件位置定位；显式自定义相对路径按调用者的启动目录解释。
当前默认输入为 `data/crystal_q1_v1/`；原 Q1 与历史 checkpoint 按各自数据及模型契约保留。
训练摘要/历史进入 `results/reports/`，预测/目标数组及逐样本表进入对应 `results/runs/ablation_<model><tag>/`。
CIF 推理默认写入 `results/runs/b7_cif_blind/`；`--output` 仍可指定其他目录。
新的实验标签、模型接入或训练预算仍需独立批准。

旧 `output/index/results`、旧脚本/配置/文档名通过 [`utils/paths.py`](../utils/paths.py) 读取迁移登记。
`repository_path` 解析旧位置且支持重复调用，`input_path` 区分仓库默认输入与自定义路径。
历史批处理入口已归档；热力学计算类 `ThermodynamicCalculator` 保留在 `utils/thermodynamics.py`。
旧 G1/宏晶格/pair 辅助/周期多体分支需要时才显式加载归档实现；默认 B7/ZP 不加载这些分支。
共享模型中的旧参数开关仍保留，以兼容历史 checkpoint；其保留不表示旧候选重新成为当前方案。

历史 manifest、有效配置、数据、权重和正式原报告保留原路径字段及 SHA-256。
`historical_path` 验证原始字节；确有外部哈希引用的源码原文保留在 `archive/metadata/sources/`，
其余重复副本以已核验的 Git 提交、原路径和 SHA-256 定位，恢复到被忽略的本地缓存。
`protected_path` 额外要求执行源码等于登记的迁后哈希，不能凭旧快照放行后续任意改动。
Git 对象在浅克隆中可能缺失；必要证据的原文快照不依赖 Git 历史恢复。

旧网格原有 30 个失效链接及混用 Q1/pre-Q 的组合保持历史状态，不能用于当前训练。
另有 10 条历史源码身份在整理前已缺失，见迁移登记中的
`historical_sources_unresolved_before_migration`；对应历史链不能宣称完整复核。
归档保留复现线索，实际复跑仍须具备原始输入、权重、环境和单独授权。

## 文档和文件命名

设计正文使用 `design-YYYY-MM-DD-topic.md`，日志使用 `log-YYYY-MM-DD-topic.md`。
旧设计优先使用可证明的创建日期，否则采用首次 Git 新增提交的上海时区日期；
证据存于迁移登记的 `design_dates`，内容更新和本轮移动不改变创建编号。
模板分别为 [design/template.md](design/template.md) 与 [logs/template.md](logs/template.md)。
Python 模块使用清楚的 `snake_case`；文档主题使用连字符；历史实验标签和证据文件名保留以维持身份。
`valence_by_element.json` 是按元素的价电子参考，不是迁移表；
`edos_valence_by_material.parquet` 与 `edos_truncation_by_material.parquet` 是按材料的标签参考。

## Git 文件边界

[`.gitignore`](../.gitignore) 默认排除权重、原始数组、下载载荷、序列化数据、CSV/TSV/JSON/JSONL、
日志和缓存；仅用完整路径开放 Q1 元信息、必要小参考表及经过审查的有限摘要/证据。
`results/reports/`、`archive/results/` 和 `archive/data/` 使用封闭容器规则，
新文件不能仅因进入“报告”目录就自动进入仓库。其他 JSON 配置如有需要，同样须审查后添加精确例外。

[`tools/checks/check_repository.py`](../tools/checks/check_repository.py) 检查真实 Git blob 体积，
任何可提交文件不得超过 2 MiB；强制加入的忽略文件也会被提交钩子拒绝。
提交前检查暂存区；推送前检查所有最终目录树和实际新增的 Git 对象。
远端已存在的旧对象不重复计入本次上传；最终目录仍不能跟踪被忽略的载荷。
中间提交新加入、随后删除的载荷仍会被拦截；向新远端推送时历史对象同样受检。
本机启用 `core.hooksPath=tools/checks`；新克隆需显式启用相同设置，钩子配置不会随 Git 克隆传播。
自动检查还执行工作区文件边界检查；本地钩子可被手动绕过，服务器保护需另行配置。

忽略规则不会自动移除已跟踪文件。原始逐样本文件仅本地保存；旧 Git 历史存在大原始表，
当前方案通过推送钩子拦截，未重写历史；
向新远端完整推送前需要独立决定如何清理历史。
