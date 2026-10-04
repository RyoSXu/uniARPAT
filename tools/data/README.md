# 数据工具

本目录只保留与已冻结 Q1 数据约定仍有关联的少量脚本，不属于常规训练流程。原始获取内容和旧的
一次性处理脚本已在结果写入 `docs/data.md`、`data/reference/edos-label-audit.md`、日志和受版本控制清单后删除。

## 支持的脚本

| 脚本 | 用途 |
|---|---|
| `download_delta_tables.py` | 为已批准的新数据活动下载 Materials Project Delta 表。 |
| `fetch_mp_raw.py` | 获取 Materials Project 结构和 DOS 记录；需要 `MP_API_KEY`。 |
| `download_phonondb.py` | 为已批准的新数据活动下载 PhononDB 源归档。 |
| `resume_dl.py` | 通用的可恢复文件下载工具。 |
| `build_v2_cache.py` | 从已验证的处理后数据集建立本地训练缓存。 |
| `a6_split.py` | 为新数据版本创建冻结的分层划分。 |
| `c2b_grids.py` | 生成已明确批准的网格变体。 |
| `q1_rebuild.py` | 仅作为已批准恢复计划的一部分重建 Q1。 |

## 规则

- 常规模型工作不得使用这些脚本。
- 每项新数据活动先写设计文档，说明输入、输出、数据结构版本、验证和凭据处理方式。
- 通过环境变量设置 `MP_API_KEY`，绝不提交凭据。
- 重建的缓存只有在其清单、划分、隔离数量和验证报告均与已记录的数据约定一致后才能使用。

## 路径与输出保护

所有脚本已平放于本目录。处理脚本的仓库根从文件位置推导，外部源默认仍是仓库相邻的
`getdata/raw`，可用 `--raw` 显式指定；本轮没有下载、划分或重建数据。

- `build_v2_cache.py --out ...` 和 `c2b_grids.py --out ...` 要求新的输出目录，拒绝覆盖非空目录。
- `a6_split.py --out ...` 要求新的划分 YAML，仍会按原协议回写显式源目录中的 parquet；不得对共享源随意执行。
- `c2b_grids.py --structure-cache ...` 必须显式给出与源池划分 ID/顺序及结构行数完全一致的缓存。
- `q1_rebuild.py` 拒绝覆盖现有 Q1；它不自动恢复 `nvalence` 等全部训练 sidecar，不能代替完整恢复方案。
- `download_phonondb.py --ref ... --dest ...` 可以显式指定下载清单和目标。

正式参考表位于 `data/reference/`，历史网格位于 `archive/data/grids_c2b/`；生成新数据不覆盖二者。
