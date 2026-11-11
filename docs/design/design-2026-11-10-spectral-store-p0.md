# 设计：2026-11-10 — 谱数据库 spectral-store-v1（P0）

上级方案与用户决定见 [部分标签数据扩充](design-2026-11-10-partial-label-data.md)。本页只定数据库与训练视图的设计，未建库、未训练。
用户确认本页后进入 P1（建库，CPU）。

**实施结果（2026-11-10）**：已建库并核验，见[日志](../logs/log-2026-11-10-spectral-store-build.md)。偏差：新划分按用户决定推迟（旧 test 作最终双标签 test）；本页“划分顺序”中新双标签 test 不再执行；声子泛函无法核实；无规范任务时的补充规则经验证不可用。

## 目标

- **灵活**：一个材料可以有任意子集的谱。新增一种谱等于新增一张表，不需要重建。网格、来源和泛函的选择放在训练视图配置里，不写死在数据里。
- **可核验**：从新库生成的“兼容视图”在旧材料上能复现 `crystal_q1_v1`；每个数字可追溯到来源任务。
- **防泄漏**：新 test 在任何训练之前封存；新增训练材料与所有评估集按组成隔离。

## 已核实的接口事实（2026-11-10）

- MP API 可用，数据库版本 2026.09.28，summary 共 154,377 个材料；材料 ID 格式与现有数据集一致（`mp-aaaa…`）。
- `materials/electronic_structure` 给出每个材料的规范 DOS 任务号（如 `aaadryyt`），格式与本地 Delta `total-dos` 表的 `identifier` 相同。
  因此 eDOS 可以“按 MP 规范任务取本地谱”，不必自选任务。
- `materials/phonon` 共 27,750 个文档，`identifier` 即材料 ID 去掉 `mp-` 前缀，带 `phonon_method`（pheasy/dfpt）和 `sum_rules_breaking`。
  本地 Delta phonon 表 27,914 行，一个材料可能同时有两种方法。
- **泛函未直接给出**：Delta 中声子的 `run_type` 为空，API 中 dfpt 文档的 `calc_meta` 为空，pheasy 文档只有工作流目录。
  P1 须从 MP 文档或对应论文逐来源核实，核实前在库中记为 `unverified`，不采用推测值。eDOS 的 `run_type` 在 Delta 中有（GGA / GGA+U 等）。

## 存储布局

目录 `data/spectral_store_v1/`，与其他数据一样不入 Git；只追加，不覆盖已有版本。

| 文件 | 每行 | 主要字段 |
|---|---|---|
| `materials.parquet` | 一个材料 | `material_id`、来源库与版本、`license`、`deprecated`、`formula`、绝对组成键、约化组成键、`n_atoms`、输入结构（`cell`、`frac`、`Z`） |
| `geometries.parquet` | 一个材料的一种几何 | `material_id`、`role`（input / edos_calc / phdos_calc）、结构、与输入结构的体积比和结构匹配结果 |
| `spectra_edos.parquet` | 一条谱 | 见下方通用字段；另含 `efermi`、自旋通道数 |
| `spectra_phdos.parquet` | 一条谱 | 见下方通用字段；另含单位换算记录、`sum_rules_breaking` |
| `spectrum_types.json` | 一种谱 | 横轴名、单位、零点约定、原生单位到规范单位的换算 |
| `splits/` | — | 各划分的 ID 清单、SHA-256、切分规则与随机种子 |
| `sealed/` | — | 新 test 的谱，与公开部分物理分开 |
| `build_manifest.json` | — | 输入文件哈希、API 版本与时间、代码提交、各步计数 |

**每条谱的通用字段**：`spectrum_id`、`material_id`、`source`（mp / phonondb / jarvis / …）、`source_task_id`、`method`、`functional`（含 `unverified`）、
`representation`、坐标与数值、质量门禁结果、`license`。

`representation` 有两种取值：
- `native`：原生坐标和数值，换算到规范单位，不预先分箱。
- `legacy_binned`：只用于旧数据中本地已无原始谱的 PhononDB/JARVIS 声子，直接存冻结的 P0 分箱值和 mask，并注明不能换网格。

以后加 X 射线吸收谱或光学谱，就是新增 `spectra_<type>.parquet` 并登记到 `spectrum_types.json`。

## 构建规则（P1）

1. **材料与结构。**
   - 从 MP summary 取全部材料，排除 deprecated 和非 BY-C 许可。
   - 输入结构沿用旧规则：pymatgen 标准原胞，`symprec=0.1`。
   - eDOS/phDOS 的计算结构从 Delta 表一并存入 `geometries`。
2. **eDOS。**
   - 按 `electronic_structure` 的规范任务号从本地 Delta 取谱，GGA/GGA+U 由 MP 的选择决定。
   - 处理沿用旧规则：自旋通道相加，用该任务自己的 `efermi` 作零点。
   - 规范任务不在本地 Delta 快照中的，记为缺失，不改用其他任务。
3. **phDOS。**
   - 第一轮只用 MP：pheasy 优先于 dfpt，与旧规则一致；THz→cm⁻¹ 时坐标乘 33.35641、密度除以 33.35641。
   - MDR 留到之后单独一级再加。
4. **质量门禁**（沿用 Q1）：
   - 谱可读且有限；
   - eDOS 的 `C_full/N_val ≥ 0.5`，新材料的 `N_val` 用元素价电子表计算（D3a 已验证与逐材料审计的一致率为 99.97%）；
   - 训练视图按原子数上限 160 过滤（与现有默认 train 相同），库里保留全部材料。
5. **去重。** MP 材料 ID 已按结构去重；同组成的不同多晶型是不同的材料，都保留。
6. **划分**（顺序固定，ID 清单冻结并记哈希）：
   1. 旧划分原样冻结：`train_q1`、`valid_q1`（开发集，用于选点）、`test_q1`（次要结果，注明曾用于诊断）。
   2. **评估排除集**：与 `valid_q1`、`test_q1` 绝对组成或约化组成相同的新材料，全部不进入新划分。这比旧划分只排除绝对组成更严格。
   3. **新 test**：
      - 双标签 test 从过门禁后、两种谱都有的新材料中按约化组成分组随机取约 1,000 条；池子不足 5 千时取约 800 条。
      - eDOS test 从其余只有 eDOS 的新材料中按约化组成分组取约 2,000 条。
      - 两者的约化组成彼此不重叠。谱写入 `sealed/`。
   4. **eDOS 开发集**：从剩余的只有 eDOS 的新材料中按约化组成分组取约 2,000 条。只用于报告，不用于选点。
   5. 新 test 与 eDOS 开发集的约化组成也加入排除集。其余新材料进入 `train_new`；与排除集同约化组成的材料一律不进 train。
   6. 随机种子固定写入 `splits/`，并报告各集合的原子数、元素和晶系分布。
7. **尖峰处理。**
   - 旧规则用父池（含 valid/test）计算每箱 p99.9 阈值，存在轻微泄漏。
   - 新库改为只用 train 部分计算阈值，作为视图配置的一部分冻结；库里存未截断的值。
   - 旧材料在兼容视图中直接使用冻结的旧标签。

## 训练视图

- 由一个 YAML 配置生成，内容包括：用哪些谱、网格、来源和泛函的白名单、哪些划分、原子数上限、尖峰阈值来源。
- 输出沿用现有的 11 个数组格式，另加：
  - `has_<type>`：逐材料的任务存在标记；
  - `source_<type>`：来源编号，供默认关闭的来源标记开关使用。
- 现有加载代码只需读取这些新增字段。
- 视图目录写入配置哈希；不同视图的归一化常数各自冻结，不复用旧 checkpoint。
- **封存**：视图生成器默认拒绝输出 `sealed/` 中的谱；只有显式加上 `--unseal-final` 并写入日志时才能输出。
  加载代码遇到标记为 sealed 的视图同样拒绝读取。

| 视图 | 内容 | 用途 |
|---|---|---|
| `q1_compat` | 旧 23,306 个材料，旧标签 | 核验：须与 `crystal_q1_v1` 逐值一致 |
| `u1` | `train_q1` ＋ 新的 MP phDOS 材料（只有 phDOS 或双标签） | 实验 U1 |
| `u2_20k` / `u2_50k` / `u2_all` | `train_q1` ＋ 分级加入的新 eDOS 材料（嵌套抽样，种子固定） | 实验 U2，得到真实学习曲线 |
| `u3` | `u1` ∪ `u2_all` | 实验 U3 |

所有视图的选点集都是 `valid_q1`；另输出 `edos_dev`。

## P1 核验（建库后、训练前）

1. **兼容视图逐值一致**：数组、ID 顺序、mask、N_val 都要一致。
2. **新旧处理一致**：对旧材料用新流程从 Delta 原始谱重新处理 eDOS 和 MP phDOS，与旧标签比较。
   - 除旧尖峰截断箱外应一致；11-05 门禁中 eDOS 有 20,410 条完全一致、550 条截断。
   - 报告不一致的数量和原因。
3. **泄漏检查**：新 train 与所有评估集的材料 ID、绝对组成、约化组成重叠均为 0。
4. **分布比较**：`train_new`、`edos_dev`、新 test 与 `valid_q1` 的原子数、元素、晶系、eDOS 带隙分布。
   - 回应 D3a 的负面信号：其严格集只改善了 5/595 个 valid 难样本。
5. **几何检查**（不训练）：
   - 用 h2 把 `valid_q1` 的输入换成计算结构推理一次，报告误差变化；
   - 逐条核查体积偏差大于 30% 的 74 条（eDOS 14、phDOS 60）。
6. **计数**：各步的入、出、排除原因计数写入 `build_manifest.json`。

## 产出与成本

- 代码：`tools/data/spectral_store/`（建库、划分、视图三个入口）。测试覆盖：
  - 分箱与旧代码一致；
  - 划分不重叠、封存拒读；
  - 视图配置哈希；
  - 兼容视图一致。
- 数据下载：API 只取元数据，即 summary 约 155 次分页请求、electronic_structure 约 155 次、phonon 约 28 次；谱都来自本地 Delta 表。
- 计算：CPU 数小时。主要耗时在读取 4 GB 的 eDOS 表和结构标准化。
- 磁盘：预计 10–20 GB。

## 本页不包括

- 模型代码改动：部分标签损失、批采样、EMA、混合精度验收，见上级方案 P2。
- MDR 与其他来源：之后单独加入。
- Mat2Spec：由独立会话负责。
- 任何训练。
