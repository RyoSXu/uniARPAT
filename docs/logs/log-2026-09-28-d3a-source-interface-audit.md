# 日志：2026-09-28 — D3a 数据源接口审计（eDOS-only 候选支持面，只读）

## 范围

- **任务：**为后续 500–1000 条 eDOS-only 候选支持面裁决，查清现有本地文件能否直接完成
  候选集合构造、确定性抽样、结构／组成关联、eDOS 原始谱读取与 E0／SumNorm／coverage／
  trunc／Z0 质量审计。以实际读取到的本地 schema 与少量记录为证据。
- **执行模型：**`opencode-go/deepseek-v4.1-flash`。
- **约束（马尚酱）：**只读调查；仅新增本文件，不修改任何其他文件；不下载、不联网、不调用
  MP API、不读环境凭据、不重建缓存、不训练、不运行 GPU；不写 raw/v2_release/data/results/index；
  不执行 `git commit/checkout/reset/clean`；不更新 `status.md`／`index.md`／`decisions.md`。
- **本次新增文件：**`docs/logs/log-2026-09-28-d3a-source-interface-audit.md`（仅此一个）。
- 结论一律不用“通过阈值”表述；未实测的产出量标为未知。

## 证据

### 0. 工作区与只读手段

- 起始 `git status --short` 为空；`HEAD=37c5860`（`main`）。
- 只读命令类型：`git status/log/show/ls-tree`；`ls/find/du/wc`；`head/sed/grep`；
  `python3`（`json`、`pyarrow`、`collections`，仅解析 schema／计数／集合运算，未对大文件整表载入）。
- 未读取环境变量、未访问网络、未调用任何写接口。

### 1. 集合精确复算（事实）

以本地文件为准做集合运算，全部落在 `getdata/raw` 与 `v2_release/`：

| 集合 | 定义 | 来源文件 | 字段 | 实测大小 |
|---|---|---|---:|---|
| E | 有效 eDOS | `/root/home/newstudy/getdata/raw/census_effective_ids.json` | JSON 字符串数组（裸 `mpid`） | 154,373 |
| S | 有 summary 的 census | `.../census_summary_ids.json` | 同上 | 154,377 |
| P | 有 MP 声子 | `.../dual_spectra_ids.json` | 裸 `mpid` 数组 | 26,609 |
| P₂ | MP 声子映射 | `.../census_phonon_map_canonical.json` | dict：`mpid → {"pheasy": [task_id, …]}` | 154,373 键 |
| A | 有声子无 eDOS | `.../edos_absent.json` | 裸 `mpid` 数组 | 7,965 |
| C | 现有 24,988 成品 | `/root/home/newstudy/getdata/v2_release/v2_processed.parquet` | 24 列，含 `mpid` | 24,988 |

实测关系（`python3` 集合运算）：

- `A ⊂ P ⊂ E`；`P − A = 18,644`（MP 原生双谱）；`C ⊄ P`，`C − P = 6,344`（外源声子）。
- `C` 的 `src_ph` 计数：`mp 18,644 / phonondb 4,049 / jarvis 2,295`；`src_edos` 全部 `mp`。
- **候选 `E − (P ∪ C) = 121,420`（精确复现）**；等价地 `E − (P ∪ C_ext)`，其中 `C_ext = C − P = 6,344`。
  即 `154,373 − 26,609 − 6,344 = 121,420`，与旧日志一致。
- 候选集合本身**没有独立清单文件**，只能由上述三个集合做差得到。

补充事实：`data/train4ARPAT/manifest.json` 的 `stats_v2.n=24,988`；`quarantine.version=Q1-20260916`、
`n_removed=1,682`（train/valid/test=1,334/164/184）。

### 2. 本地存放方式与候选可追溯性（事实）

#### 2.1 结构／组成：候选已全部本地具备

| 文件 | 行数 | 集合 | 关键字段 |
|---|---:|---|---|
| `raw/pretrain_structures/structures_*.jsonl`（31 个分片） | 154,377 | `S` | `material_id, structure, nsites, nelements, composition_reduced, symmetry, band_gap, formation_energy_per_atom` |
| `raw/mp_structures.jsonl` | 26,609 | `P` | `mpid, summary{band_gap,efermi,structure,…}` |

实测：`pretrain_structures` 唯一 `material_id` = 154,377，`E` 的 154,373 个全被覆盖，
**121,420 个候选全部命中（121,420/121,420）**。候选结构、约化组成、晶系／空间群、原子数
本地直接可用，无需下载。

#### 2.2 eDOS 原始谱：本地为主体，但候选的“逐材料”谱对不上

| 文件 | 行数 | 命中集合 | 是否含谱本体 |
|---|---:|---|---|
| `raw/mp_edos_raw.jsonl` | 18,644 | `P − A` | 是：`edos_raw[0]{identifier,structure,spin_up_densities,spin_down_densities,energies,efermi}` |
| `raw/mp_increment_raw.jsonl` | 7,724 | 6,344 ∈ `C`，**1,380 ∈ 候选** | 否：仅 `mpid, mp_summary, dos_task_id, provenance, summary_unresolved, es_unresolved` |
| `raw/delta_mp/core/electronic-structure/total-dos/*` | ~4.4 GB | task 级 | 是：schema `identifier, structure, spin_up_densities, spin_down_densities, energies, efermi`（run_type 分区：GGA/GGA+U/HSE06/PBE/PBEsol/r2SCAN/unknown） |
| `raw/delta_edos_index.parquet` | 691,978 | task 级 | 否：仅 `identifier, nsites, volume, species, run_type`（**无 material_id、无能量轴**） |

实测交叉：

- `mp_edos_raw.jsonl ∩ 候选 = 0`；`mp_structures.jsonl = P`；`mp_phdos_raw.jsonl = P`（26,609，含声子谱）。
- 候选在 `mp_increment_raw.jsonl` 有 **1,380** 条记录，其中 `dos_task_id` 非空 **820** 条
  （即这 820 个候选本地已有 `mpid → Delta task` 指针，谱本体需按 `identifier` 到 Delta 表取）。
- `mp_increment_dosmap.json` 共 2,529 键，`mode` 分布 `delta_vol 1,471 / matcher_ok 606 / unresolved 366 / unique 86`；
  其中命中候选 **560**。`census_phonon_map_canonical.json` 是 `P` 的 task 映射，**不覆盖** `E − P`。
- `delta_edos_index.parquet` 的 `run_type`：GGA 498,455 / GGA+U 178,372 / HSE06 10,836 / unknown 3,069 /
  PBE 957 / PBEsol 146 / r2SCAN 143。该表**无 material_id**，`mpid ↔ task` 归属需结构匹配。
- 以 `(nsites, 物种集合)` 为粗块做本地可读性抽查：**候选 112,726/121,420** 的块在 Delta 索引中存在
  （每块有多个 task）；抽查 5 个候选（`mp-aaaaaaac` 等）均命中块且块内有 9–22 个 task。
  块存在是“可能有本地谱”的必要条件，**不等于**已归属，也不是最终可提取率。

#### 2.3 抽查：material id → 结构／eDOS／Delta task/run_type

- 结构：`mp-aaaaaaac`（Pd，nsites=1）等候选均能从 `pretrain_structures` 直接取到结构、组成、`band_gap`。
- eDOS 逐材料：候选在 `mp_edos_raw.jsonl` 为 0；仅 820 个候选有本地 `dos_task_id` 指针。
- Delta task/run_type：`delta_edos_index.parquet` 可提供 task 级 `nsites/volume/species/run_type`，
  但候选 `mpid` 未与之直接关联，需结构匹配（见第 3 节脚本）。

#### 2.4 质量审计字段：候选侧现状

| 审计项 | 定义位置／依据 | 现有覆盖 | 候选是否需要重算 |
|---|---|---|---|
| E0 窗口 | `data/grids_c2b/grids.json`：`E0 = linspace(-6,6,129)`（128 bin）；训练取 `E0+P0` | 定义本地齐全 | 谱对齐后重算覆盖 |
| P0 窗口 | 同文件：`P0 = linspace(-280,980,65)`（64 bin） | — | 与本轮 eDOS-only 无关 |
| 能量对齐 | 退役 `a4_process.edos_of`／`c2b_grids.edos_raw_of`：`energies − efermi` | 逻辑在 Git 历史 | 是 |
| coverage | `box_average` 产 `edos_mask`（bin 有数据=1） | 池：`v2_processed.edos_mask`；候选无 | 是 |
| trunc | `index/z0_trunc.parquet`（`mpid, trunc`），判据 `C_full/N_val` | 24,988 行 = `C` | 是 |
| Z0 `N_val` | `index/z0_nvalence.parquet`（`mpid,N_val,method,T_own,ratio,confmin`）+ 元素参考 `index/z0_zval.json` | 24,988 行 = `C` | 是（需候选原始谱 + 元素表） |
| SumNorm | 训练侧 `datasets/dataset.py`：`dos_sumnorm`（总和归一化，替代 min-max） | 消费端逻辑 | 不依赖候选额外存储 |

即：E0 窗口与加工逻辑本地可复现，但 **Z0/trunc/coverage 现有文件只覆盖 24,988 成品（`C`），
对 121,420 候选为零覆盖**，必须凭候选原始谱重算，且重算依赖 eDOS 逐材料谱（第 5 节阻塞项）。

### 3. Git 历史中已退役的可复用符号／提交（只读定位）

退役脚本仍在 Git 历史，可 `git show` 查看但不恢复、不修改：

| `git show` 定位 | 可复用符号／职责 |
|---|---|
| `git show e24d2a4^:tools/getdata/delta_edos_index.py` | `build_index()`：单遍扫 Delta 抽 `(identifier,nsites,volume,species,run_type)`；`match_targets()`：按 `(nsites,species)` 分块 + `volume` 相对差 <1% 归属，逐级 `unique/multi/none` |
| `git show e24d2a4^:tools/getdata/dosmap_final.py` | `main()`：vol<0.2% 直收 + `StructureMatcher(ltol=0.2,stol=0.3,angle_tol=5)` 复核，终判 `delta_vol/matcher_ok/…` |
| `git show e24d2a4^:tools/getdata/a4_process.py` | `E_EDGES/P_EDGES/CAP_ATOMS/WINSOR_PCTL/SYMPREC`、`box_average()`（覆盖率掩膜）、`load_jsonl_map()`、`main()` 组 eDOS/phDOS 记录 |
| `git show e24d2a4^:tools/getdata/fetch_structures.py` | `FIELDS` + `main()`：批量取结构 summary（网络入口） |
| `git show e24d2a4^:tools/getdata/top1_mp_increment.py` | `rest_batch()`：REST 批量取数（网络入口） |
| `git show e24d2a4^:tools/getdata/extract_dual.py` / `extract_phdos.py` | 迈 Delta 抽 eDOS／phDOS 记录 |
| `git show 92d6501 --stat` | “data v2 track complete”，上述脚本的成组提交 |
| `git show eb89325` | 退役 M4 `cif2dos.py`（`D cif2dos.py`），B7 盲推理现由 `b7_cif_infer.py` 承担 |

提交级定位：删除发生在 `e24d2a4`（2026-09-18 “Comprehensive project structure overhaul”），
其父 `e24d2a4^` 仍含全部 `tools/getdata/*`；`b88269c` 是最早把拿数工具纳入仓库的提交。
**安全备注（事实）：**`fetch_structures.py`／`top1_mp_increment.py` 在历史中含硬编码 MP API key
（本日志不复制该值）；现役入口改为 `tools/data/fetch/fetch_mp_raw.py` 的 `MP_API_KEY` 环境变量约定。

现役（受控，勿在常规训练中跑）入口：`tools/data/fetch/{download_delta_tables,fetch_mp_raw,
download_phonondb,resume_dl}.py`、`tools/data/process/{build_v2_cache,a6_split,c2b_grids,q1_rebuild}.py`。

### 4. 不使用 valid/test 标签的确定性 500–1000 抽样所需字段（事实 + 可行性）

- 候选与 `C` 按集合构造天然不相交（`候选 ⊂ E−(P∪C)`），而 valid/test 只存在于 `C`／`index/split_v2.yaml`，
  因此**抽样无需读取任何 valid/test 标签即可零泄漏**。
- 所需字段及本地可用性：

| 字段 | 来源 | 候选可用 |
|---|---|---|
| `mpid`（唯一键） | `pretrain_structures` | 121,420/121,420 |
| `composition_reduced` | 同上 | 是 |
| `symmetry`（含 `crystal_system`／空间群号） | 同上 | 是 |
| `nsites, nelements` | 同上 | 是 |
| `structure`（原胞，用于后续谱对齐） | 同上 | 是 |
| 可选分层：`band_gap, formation_energy_per_atom` | 同上 | 是 |

- 确定性做法（提议，未实施）：候选 `mpid` 排序后按 `sha256(seed || mpid)` 升序取前 N；
  或先按 `composition_reduced` 的元素 Z 计数分组、组内取种子哈希，实现“组成隔离版”抽样，
  与 `index/split_v2.yaml` 的组成硬隔离口径可比。**本地字段齐全，抽样本身可在只读范围内完成。**

### 5. 实施阻塞项与最小安全实现范围

**阻塞项（事实／推断／未知分清）：**

- **B1（阻塞，事实）：**候选缺逐材料 eDOS 原始谱。`mp_edos_raw.jsonl` 与候选交集为 0；
  仅 1,380 个候选在 `mp_increment_raw.jsonl` 有记录、其中 820 个有 `dos_task_id` 指针，且该文件**不含谱本体**。
- **B2（阻塞，事实+推断）：**候选谱只能从本地 Delta 表按 `identifier` 取（谱本体在本地），
  但 Delta **无 material_id**，`mpid ↔ task` 归属需结构匹配（复用 `delta_edos_index.match_targets` /
  `dosmap_final` 的 vol 规则 + `StructureMatcher`）。这是本地计算，**无需联网**，但尚未执行。
- **B3（阻塞，事实）：**Z0 `N_val`、`trunc`、`coverage` 现有文件只覆盖 24,988，候选零覆盖，
  需在归属到谱后重算（`z0_zval.json` 元素表可直接复用，`box_average` 逻辑在历史）。
- **B4（缺口，事实）：**121,420 中有 **8,694** 个候选的 `(nsites, 物种)` 块在 `delta_edos_index`
  中不存在；这些材料是否完全无本地谱、或需放宽匹配／下载，**未知**。
- **B5（未知）：**真实结构匹配归属率未知。块级存在 112,726/121,420 只是必要条件的粗上界，
  不能当作可提取率；HSE/泛函血缘配对规则也需在归属后重新执行。
- **B6（范围，事实）：**JARVIS eDOS 为 optB88vdW，与 MP 口径不同（旧日志已判混用即改质量标准），
  本候选支持面若坚持 MP 单源，则 JARVIS 谱不可用于扩充。

**最小安全实现范围（提议，待批准，不含任何训练/采集）：**

1. **纯只读段（当前即可做）：**由 `census_effective_ids.json`、`dual_spectra_ids.json`、
   `v2_processed.parquet` 生成 121,420 候选名单；按第 4 节确定性抽 500–1000；joined
   `pretrain_structures` 结构／组成。全部本地、可复现、可回滚（仅生成结果清单）。
2. **本地归属段（需批准，不联网）：**对抽样集复用退役 `delta_edos_index`／`dosmap_final` 逻辑，
   从本地 `delta_mp` 匹配 `mpid → task/run_type`，产出归属与未归属名单；实测匹配产出后方可谈量。
3. **本地质量审计段（需批准，不联网）：**对已归属样本，按 `E0` 窗口重算 eDOS 网格、`edos_mask`、
   `trunc` 与 Z0 `N_val`，产质量审计表。仅在这一段之后才可能给出“可用量”结论。
4. **明确排除：**任何 MP REST／S3 下载、`MP_API_KEY` 读取、Delta 表重建、缓存重建、训练与 GPU；
   8,694 个无本地块的候选一律记“待归属/可能需下载”，不在本轮解决。

**不得凭空设定的量：**本轮只报实测计数（121,420、18,644、6,344、112,726、820、560、8,694 等），
不给出“通过质量审计后的可训练数量”或任何通过阈值。

## 结论

- **状态：closed（只读接口审计完成）。**
- **可达部分：**候选集合构造（121,420，精确复现）、确定性抽样（零 valid/test 泄漏）、
  结构／组成关联（121,420 全覆盖）**本地已具备**，可在纯只读范围内直接完成。
- **受阻部分：**候选的**逐材料 eDOS 原始谱**与 **Z0／trunc／coverage 质量标签**本地未就绪；
  谱本体在本地 Delta 但需 `mpid↔task` 结构匹配归属（本地可做、未做），Z0 类标签需在归属后重算。
- 因此“现有本地文件能否直接完成候选支持面”分两层：**名单+结构+抽样 能**；
  **原始谱+质量审计 不能直接完成**，须先做一次本地归属（不联网）。
- 本日志仅为接口可行性证据，不构成任何扩充训练、数据契约变更或采集授权。

## 交接

- **下一项关卡工作：**由马尚酱决定是否批准“本地归属段 + 本地质量审计段”（第 5 节第 2、3 步），
  并明确抽样规模与是否采用组成隔离口径；在批准前不做任何写入或采集。
- **对 status、index 和 decisions 的更新：**按任务约束**不更新**这三处；
  本日志作为 D3a 只读接口审计的候选证据存档。
