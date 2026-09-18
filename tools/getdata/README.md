# 数据抓取工具（tools/getdata）

> v2数据管线的可执行部分。raw数据（GB级）永不进库，只在工作机 `/root/home/newstudy/getdata/raw/`。

| 脚本 | 用途 | 状态 |
|---|---|---|
| `download_delta_tables.py` | MP Delta整表断点续传下载 | ✅已完成（4.4GB） |
| `census.py` / `census_rebuild.py` | eDOS全集+声子映射census（旧/新两版，rebuild为正统） | 🔄跑中 |
| `census_repair*.py` / `census_final.py` | 中间修复稿，保留备查，勿直接跑 | 🔕 |
| `fetch_mp_raw.py` / `fetch_raw_pilot.py` | 单批/试点抓取 | ✅pilot完成 |
| `process.py` | 加工骨架旧稿（已被 `a4_process.py`/`a4_pass2.py` 替代，勿用） | 🔕存档 |
| `c2b_grids.py` | C2b网格臂定义（E0–E3/P0–P2，见 `data/grids_c2b/grids.json`） | ✅2026-09-14（生产冻结E0+P0） |
| `q1_rebuild.py` | Q1隔离物化（A5同构，干净池18706/2313/2287） | ✅2026-09-16 |
| `recompute_phonondb.py` | A3b PhononDB复算入库（10,034零错误） | ✅2026-09-11 |
| `download_phonondb.py` | PhononDB MDR包下载 | ✅已完成 |
| `extract_dual.py` / `extract_phdos.py` | 双谱/声子抽取旧稿（被A4管线替代，勿用） | 🔕存档 |
| `fetch_structures.py` | 结构批量抓取 | ✅Top-up已用 |
| `dump_atom_features.py` | 原子特征导出（一次性） | 🔕存档 |
| `a3c_phonondb_coverage.py` | A3c：MDR serial→legacy→canonical + 交集统计（断点续跑） | ✅2026-09-11（9938映射） |
| `a3c_validate.py` | A3c分布验证（元素/maxfreq/晶系，全本地） | ✅2026-09-11（通过） |
| `a7b_jarvis_coverage.py` | A7b：JVASP→canonical + 交集（A3c免费复用+API补） | ✅2026-09-11（45154映射） |
| `a7c_fetch_jarvis.py` | A7c：JVASP谱拉取（webpages，一页双谱，断点续跑） | ✅2026-09-12（32453条） |
| `a2b_gap.py` | A2b三源分歧矩阵（公共网格+配准+结构delta） | ✅2026-09-12 |
| `top1_mp_increment.py` | Top-up#1：增量集summary+eDOS task映射 | ✅2026-09-12（7724全回） |
| `delta_edos_index.py` + `dosmap_final.py` | Delta归属索引 + matcher终判 | ✅2026-09-12 |
| `a4_process.py` / `a4_pass2.py` | A4两遍式加工（重切+winsorize+验收） | ✅2026-09-12（24988行） |
| `a6_split.py` | A6分层切分（8:1:1+硬隔离+探针） | ✅2026-09-12 |
| `build_v2_cache.py` | A5 npy缓存+manifest（--check经训练loader） | ✅2026-09-12 |
| `a8_cifs.py` | A8逐样本canonical CIF生成+打包 | ✅2026-09-12（24988零失败） |
| `resume_dl.py` | 断点续传通用下载器（Range+重试） | ✅figshare实测 |

> 注意：`jarvis.db.webpages` 会在**当前目录**掉落 `tmp????????` XML 缓存（4千+个，316MB），
> 跑批量抓取后记得清理；已在根 `.gitignore` 忽略（勿提交）。

规范见 `../../docs/01_开发文档/数据加工规范DataSpec.md`。
