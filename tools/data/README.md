# 数据工具

本目录只保留与已冻结 Q1 数据约定仍有关联的少量脚本，不属于常规训练流程。原始获取内容和旧的
一次性处理脚本已在结果写入 `docs/data.md`、`index/z0_REPORT.md`、日志和受版本控制清单后删除。

## 支持的脚本

| 脚本 | 用途 |
|---|---|
| `fetch/download_delta_tables.py` | 为已批准的新数据活动下载 Materials Project Delta 表。 |
| `fetch/fetch_mp_raw.py` | 获取 Materials Project 结构和 DOS 记录；需要 `MP_API_KEY`。 |
| `fetch/download_phonondb.py` | 为已批准的新数据活动下载 PhononDB 源归档。 |
| `fetch/resume_dl.py` | 通用的可恢复文件下载工具。 |
| `process/build_v2_cache.py` | 从已验证的处理后数据集建立本地训练缓存。 |
| `process/a6_split.py` | 为新数据版本创建冻结的分层划分。 |
| `process/c2b_grids.py` | 生成已明确批准的网格变体。 |
| `process/q1_rebuild.py` | 仅作为已批准恢复计划的一部分重建 Q1。 |

## 规则

- 常规模型工作不得使用这些脚本。
- 每项新数据活动先写设计文档，说明输入、输出、数据结构版本、验证和凭据处理方式。
- 通过环境变量设置 `MP_API_KEY`，绝不提交凭据。
- 重建的缓存只有在其清单、划分、隔离数量和验证报告均与已记录的数据约定一致后才能使用。
