# 日志：2026-11-15 — 谱数据库 spectral_db_v1

用户决定：数据库只保留原始数据和来源说明；清洗、筛选、去重、切分属于数据集。不要求把下载原文件放进数据库，但每条记录要能说来源。旧数据集的谱不入库。结构比对不入库。

## 做了什么

新建 `data/spectral_db_v1/`（parquet 不进 Git）。不修改 `data/spectral_store_v1/`。

| 表 | 行数 | 来源 |
|---|---:|---|
| `materials.parquet` | 154,377 | MP API summary，0.88.1，2026-11-10；库版本 2026.09.28；许可均为 BY-C |
| `spectra_edos.parquet` | 691,978 | MP Delta total-dos，下载于 2026-11-05 |
| `spectra_phdos_mp.parquet` | 27,914 | MP Delta phonon dos，下载于 2026-11-05 |
| `spectra_phdos_phonondb.parquet` | 10,034 | PhononDB 压缩包，phonopy 重算（沿用已有结果，每行有压缩包 SHA-256 和 phonopy 版本） |
| `spectra_phdos_jarvis.parquet` | 7,480 | 已下载的 JARVIS 声子页，结构来自 `jdft_3d-12-12-2022` |

构建脚本 `tools/data/spectral_store/build_db_v1.py`。说明和哈希在该目录的 `README.md`、`manifest.json`。

## 入库时做的换算（都记在 README）

- eDOS：能量减去该任务的 `efermi`；自旋通道相加。点的顺序不改。分开的自旋通道、位点上的力、速度、电荷、氧化态没有留。
- MP 声子：THz × 33.35641 换成 cm⁻¹，密度除以同一常数。没有泛函列，因为来源没给出可核实的泛函。
- JARVIS：下载的是笛卡尔坐标，分数坐标由 `coords @ inv(lattice)` 换算。强度不按 3N 重标定。
- PhononDB 的 `mp_reference` 是数据卡上的旧整数编号，不是现在的 MP 编号。

没有做的：原胞标准化、质量门禁、选哪条谱当标签、结构匹配、划分。

## 核对（2026-11-15 13:10 左右，北京时间）

`build_db_v1.py verify` 与 `spectral_store_v1` 的 staging 逐条对谱，差异为 0。eDOS 的 staging 只有挂到材料的 626,340 条，这 626,340 条全部在新表里且数组一致；新表多出的 65,638 条是 Delta 里有、但 2026-11-10 的 `task_ids` 已不列的任务。eDOS 任务号没有重复。材料编号没有重复。

另外数到的来源事实：

- summary 里 `deprecated` 全是 false。另有 111 个材料被 electronic_structure 点了规范 DOS 任务，但不在 summary 里。
- summary 里 62,972 个材料有规范 DOS 任务，这些任务都能在 Delta 里找到。其中 59,983 条的 core 归属就是点名它的材料；2,989 条的 core 归属是别的材料，谱留着，但不标成该材料的规范任务。
- eDOS 有自旋向下通道的 478,463 条。读不出来的 0 条。全部材料、eDOS、MP 声子都没有无序占位。
- MP 声子挂到材料的 25,468 条，没有归属的 2,446 条。dfpt 1,513，pheasy 26,401。
- JARVIS 7,480 条全是笛卡尔坐标；有谱 4,067，页面没有 DOS 3,413。zip 里另外约 6.8 万条 dft_3d 没有下载声子页，不在表里。
- PhononDB 10,034 个压缩包都在，哈希已写入。

## 还没做

数据集。下一步才是从这套表做清洗、筛选、去重和切分，并写过程说明。现有的 v2 / v3 视图仍留在旧目录里。
