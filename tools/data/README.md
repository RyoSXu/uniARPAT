# 数据工具

这些工具用于获准的数据活动，不属于常规训练前置步骤。当前缓存与契约见[数据约定](../../docs/data.md)，来源、处理与恢复见[数据卡](../../docs/dataset-card-zh.md)。

| 脚本 | 用途 |
|---|---|
| `migrate_crystal_q1.py` | 由旧 Q1 与原始结构建立完整 packed FP32 缓存，修复组成回退 N_val 并登记隔离视图 |
| `q1_rebuild.py` | 已批准恢复方案中的 Q1 主数组重建；不自动恢复所有 sidecar |
| `build_v2_cache.py` | 从验证后的处理表建立缓存 |
| `a6_split.py` | 为新版本建立冻结分层划分 |
| `c2b_grids.py` | 生成已批准的网格变体 |
| `export_release_provenance.py` | 校验冻结谱/mask 与父表，导出逐材料来源和划分审计；不重建数据 |
| `download_delta_tables.py`、`fetch_mp_raw.py` | 新采集所需的 MP Delta 表、结构及 DOS；API 凭据用环境变量 `MP_API_KEY` |
| `download_phonondb.py`、`resume_dl.py` | PhononDB 源归档下载与通用断点下载 |

## 执行边界

- 新活动先明确来源、输入/输出、数据版本、验证与凭据处理；常规模型任务不重采集、重新划分或改标签。
- 默认外部源仍为仓库相邻的 `getdata/raw`，可用 `--raw` 指定；自定义路径按启动目录解释。
- `build_v2_cache.py --out`、`c2b_grids.py --out` 使用新的输出目录，拒绝覆盖非空目录；完整结构迁移也须使用新目标。
- `a6_split.py --out` 虽要求新 YAML，仍会回写显式源目录的 parquet；不得对共享源随意执行。
- `c2b_grids.py --structure-cache` 必须与源池划分 ID/顺序及结构行数一致。
- `q1_rebuild.py` 拒绝覆盖现有 Q1，不能代替含 N_val 等全部 sidecar 的完整恢复方案。
- `export_release_provenance.py --cache ... --processed ... --out <新目录>` 只读既有缓存与父表，拒绝已有输出；Release 使用方无需父表。
- `download_phonondb.py --ref ... --dest ...` 可显式指定清单与目标；不提交凭据或原始载荷。

固定参考在 `data/reference/`，旧网格在 `archive/data/grids_c2b/`；新数据不覆盖二者。变更成员、谱值、网格或完整结构契约须另定数据版本。
