# 仓库整理记录

- [repository-migration-2026-10-04.json](repository-migration-2026-10-04.json)：被路径库读取的迁移登记。
  `path_moves/prefix_moves` 兼容旧位置，`design_dates` 保存文档日期证据，
  `source_versions/source_git_versions` 核验原始源码，`source_before_sha256/source_after_sha256` 限制可执行适配，
  `source_aliases` 汇合改名后的源码身份；另记录原有缺失和失效链接。
- [repository-validation-2026-10-04.json](repository-validation-2026-10-04.json)：两轮整理的验证结果与局限，
  汇总测试、文件保全、输出比较和 Git 边界。可再生清单和原始检查日志不重复保存在仓库。
- `sources/`：确有外部哈希引用或不能由固定 Git 身份替代的原文，按 SHA-256 命名。
  可由 Git 原提交恢复且已逐字核验的重复源码副本已移除；恢复缓存放在被忽略的本地运行目录。
- `environment-snapshot-requirements.txt`：旧机器的环境快照，含本机依赖路径；不作为跨机器安装锁表。

迁移登记是运行兼容的一部分，不能把它当成普通清单删除。旧汇总 JSON 的结果合并进验证记录，
逐文件盘点可以重新生成；重复 CSV、重复树清单和通过日志已清理。
