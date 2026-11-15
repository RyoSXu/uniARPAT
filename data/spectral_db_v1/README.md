# 谱数据库 spectral_db_v1

构建时间：2026-11-15T12:54:09+08:00（北京时间）。代码提交：`b29a9071b9ce2ad431c510da574dada5cb1c569f`。
本目录是数据库本身。清洗、筛选、去重、切分和训练视图都不在这里，那些是数据集的事。

## 这个库记什么

每条记录只保留来源给出的事实，以及把谱读出来所必需的换算。下面这些**不在库里**：

- 旧数据集 `crystal_q1_v1` 的谱和标签。旧材料如果现在还在 MP 里，就只是普通的 MP 材料。
- 结构匹配、体积比、原型是否一致、标签该不该用。这些是数据集构建时算的。
- 质量判断：积分是否够、谱是否异常、有没有虚频、原子数上限。库里不打这些分。
- 任何 train / valid / test 划分。

`data/spectral_store_v1/` 是之前的工作库，里面混有视图、切分和旧标签，本库不改它。

## 表

| 文件 | 行数 | 一行是什么 |
|---|---:|---|
| `materials.parquet` | 154,377 | MP summary 里的一个材料，结构按下载的原样 |
| `spectra_edos.parquet` | 691,978 | Delta 里的一条电子总态密度 |
| `spectra_phdos_mp.parquet` | 27,914 | Delta 里的一条 MP 声子态密度 |
| `spectra_phdos_phonondb.parquet` | 10,034 | 一份 PhononDB 压缩包重算出的声子态密度 |
| `spectra_phdos_jarvis.parquet` | 7,480 | 一条已下载的 JARVIS 声子页 |

每张表的 SHA-256 在 `manifest.json`。

### materials.parquet

来源：MP API `materials/summary`，接口 0.88.1，下载于 2026-11-10。`canonical_dos_task` 来自同一次下载的 `materials/electronic_structure`（MP 指定的 DOS 任务号，去掉 `mp-` 前缀），不是我们挑的。

- 材料 154,377 个，其中 MP 标记 deprecated 的 0 个。
- 有结构的 154,377 个，没有结构的 0 个。
- 许可计数：{"BY-C": 154377}。
- 数据库版本计数：{"2026.09.28": 154377}。
- MP 给出规范 DOS 任务、且该材料在 summary 里的：62,972。规范任务指向 summary 之外材料的：111。
- 结构字段是下载的晶格矩阵、每个位点的元素和占位（`元素:占位`，多个物种用逗号接）、分数坐标、磁矩。没有做原胞标准化。位点上的力、速度、电荷、氧化态没有保留。无序位点（不止一个物种）共 0 个。

### spectra_edos.parquet

来源：MP Delta 表 `s3://materialsproject-parsed/core/electronic-structure/total-dos/`，本地下载于 2026-11-05（`tools/data/raw/delta_mp`，清单 `manifest.json`）。材料归属来自 MP API `materials/core` 的 `task_ids`（2026-11-10）。`calc_type` 来自同一次 `calc_types`。`run_type` 是 Delta 的分区值（GGA、GGA+U、HSE06、PBE、PBEsol、r2SCAN、unknown），不是另行核实过的泛函。

- 共 691,978 条。能挂到当前材料的 626,340 条，Delta 里有、但 2026-11-10 的 `task_ids` 已经不列的 65,638 条（`material_id` 为空，谱仍保留）。
- 归属材料自己把这条任务指定为规范 DOS 的：59,983。材料表里点名的规范任务有 62,972 条能在 Delta 里找到；其中 2,989 条的 core 归属不是点名它的那个材料，所以没有标成规范。Delta 里找不到的规范任务号：0。
- 有自旋向下通道、密度已按通道相加的：478,463。只有一个通道的，密度就是该通道。分开的自旋通道没有保留。
- 读不出来的：0（`parse_error` 记录原因，行不删）。
- 无序位点：0。
- 横轴是能量减去该任务自己的 `efermi`（eV），`efermi` 另存，绝对能量可以加回去。纵轴是自旋相加后的态密度。点的顺序和 Delta 里一样，没有为了入库而删点、排序或合并重复能量；积分用的清洗不在本库。

### spectra_phdos_mp.parquet

来源：Delta 表 `phonon/electronic-structure/dos/`，下载于 2026-11-05。`method` 是 Delta 的 `phonon_method`（pheasy 或 dfpt）。`asr`、`cnsr` 是 MP API `materials/phonon` 给出的求和规则偏离；接口没有给出的留空。Delta 和接口都没有给出可核实的泛函，所以这张表没有泛函列。

- 共 27,914 条。挂到材料的 25,468，没有材料归属的 2,446。
- 有求和规则数值的 27,747。能读出的谱：dfpt 1,513，pheasy 26,401（这里只表示谱本身可读，材料归属见上一条）。
- 横轴是频率乘以 33.35641，从 THz 换成 cm⁻¹；纵轴是密度除以 33.35641。原始 THz 数组没有另存，用这个常数可以换回去。
- 无序位点：0。没有计算结构的：0。

### spectra_phdos_phonondb.parquet

来源：NIMS MDR PhononDB，索引来自 `atztogo/phonondb` 的 MDR 说明，压缩包在 `data/spectral_store_v1/raw_phonondb/zips/`。每行有 `archive_sha256`。谱不是压缩包里的成谱，是用 phonopy 从力常数重算的（`MP = 20 20 20`，`TETRAHEDRON = .TRUE.`，默认频率采样）。`phonopy_version` 和 `mesh` 在每行上。本表直接沿用这次重算，没有再跑一遍 phonopy。

- 共 10,034 条，压缩包哈希已写入 10,034 条，缺包 0。
- 压缩包里没有泛函字段。数据卡写的是 PBEsol，记在 `functional_on_card`，`functional_in_archive` 为 false。
- `mp_reference` 是数据卡上的旧整数 MP 编号（形如 `mp-25`），不是现在的 `mp-` 加字母编号，没有和当前 MP 逐条核对。
- 同时保留了 cm⁻¹ 和原始 THz 两套数组，以及原胞、惯胞、超胞矩阵、Born 电荷和介电张量（phonopy 输入里的 JSON）。

### spectra_phdos_jarvis.parquet

来源：JARVIS-DFT 网页上的声子字段（频率、强度、声子模、最小有限差分模），加上 `jdft_3d-12-12-2022.json.zip` 里的结构、化学式、泛函、参考编号和空间群。只收录实际下载过声子页的条目，不是 zip 里全部 75,993 条 dft_3d。

- 共 7,480 条。有谱 4,067，页面没有 DOS 3,413，长度对不上 0。
- 坐标：下载文件里是笛卡尔坐标的 7,480 条，分数坐标的 0 条。库里的 `calc_frac` 在笛卡尔时由 `coords @ inv(lattice)` 换算，晶格和元素保持原值。
- `functional` 只用数据卡上的 `func`。卡上没有的就空着，不再填默认值。`reference` 保留数据卡原文，不判断它是不是 MP 编号。
- 强度是页面上的原始数值，没有按 3N 重标定。

JARVIS 和 PhononDB 的许可原文不在下载文件里，本库不补写。MP 材料的许可是接口给出的，写在材料表的 `license` 列。

## 重建

```bash
python tools/data/spectral_store/build_db_v1.py all --workers 8
python tools/data/spectral_store/build_db_v1.py verify
```

已有输出时会拒绝覆盖，需要同时加 `--force`。`verify` 把本库和 `spectral_store_v1` 的 staging 表逐条对谱。eDOS 和 MP 声子是建库时从 Delta 重读的，再和 staging 对；PhononDB 和 staging 里已经重算好的表对；JARVIS 是按网页和 zip 重读后再对。
