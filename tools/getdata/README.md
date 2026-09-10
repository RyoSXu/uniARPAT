# 数据抓取工具（tools/getdata）

> v2数据管线的可执行部分。raw数据（GB级）永不进库，只在工作机 `/root/home/newstudy/getdata/raw/`。

| 脚本 | 用途 | 状态 |
|---|---|---|
| `download_delta_tables.py` | MP Delta整表断点续传下载 | ✅已完成（4.4GB） |
| `census.py` / `census_rebuild.py` | eDOS全集+声子映射census（旧/新两版，rebuild为正统） | 🔄跑中 |
| `census_repair*.py` / `census_final.py` | 中间修复稿，保留备查，勿直接跑 | 🔕 |
| `fetch_mp_raw.py` / `fetch_raw_pilot.py` | 单批/试点抓取 | ✅pilot完成 |
| `process.py` | 加工骨架（待填CAP/GRID） | ⚪待A4 |

规范见 `../../docs/01_开发文档/数据加工规范DataSpec.md`。
