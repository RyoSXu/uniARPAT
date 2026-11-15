# 日志：2026-11-12 — 第 4–5 步：切分 v2 构建、视图导出与训练侧 T1/T2/T3/T5

对应设计：[切分 v2](../design/design-2026-11-11-splits-v2.md)（D1 C3、D2、D3 10/10/80、D4、D7、2026-11-12 记录身份规则；训练侧 T1/T2/T3/T5 与 W1，均已批准）。记录日期：2026-11-12；状态：已完成（未训练，未提交 Git）。

## 范围与身份

- 数据：spectral-store-v1（库表 v1.1，`records_v11/`），切分盐 `spectral-store-v1/splits-v2/2026-11-12`，以约分化学式组为单位；旧 valid/test 成员不变（C3）。
- 产物：
  - 公开清单 `data/spectral_store_v1/splits/`（train、valid_q1、valid_supp、dev、dev_indep、holdout）；
  - 封存清单 `sealed/splits/`（test_q1、test_supp、test、test_indep）；
  - 视图 `views/s1_v2`（eDOS 128／phDOS 64）与 `views/s1_v2_x2`（256／128），封存部分在 `sealed/views/`。
- 脚本：`tools/data/spectral_store/build_splits_v2.py`（构建）、`splits_v2.py`（归属规则）、`verify_s1_v2.py`（只读核验）。

## 执行与证据

### 构建中的两个问题

1. 晶胞格式：MP 来源存为 9 元列表，PhononDB/JARVIS 为 3×3，合并时报形状不齐。改为逐条 reshape 成 3×3 后重跑。
2. **泄漏检查发现 452 个化学式组同时在 train 和评估集**：全部是旧 train 的化学式组，其中的新材料被哈希分到了新 dev/test。
   - 原因：`assign` 没有实现设计 §3.2 的"旧 train 组留在 train"。
   - 修复：`assign` 增加参数 `legacy_train_formulas`，旧 train 组一律归 train；构建脚本在写出前检查"train 与任一评估集共享化学式组"，有则报错。
   - 带泄漏的那次产物已删除后重建，日志留作 `build_splits_v2.leaky.log`。

### 最终集合（材料或条目数）

| 集合 | 清单 | 视图行 | eDOS 标签 | phDOS 标签 |
|---|---:|---:|---:|---:|
| train | 55,038 | 54,995 | 47,868 | 24,951（双标签 17,824） |
| valid_q1（旧 valid，选型） | 2,313 | 2,302 | 2,302 | 2,302 |
| valid_supp（落在旧 valid 组的新材料） | 680 | 680 | 429 | 284 |
| dev（新） | 4,280 | 4,280 | 3,567 | 713 |
| dev_indep（新的独立条目） | 42 | 42 | 0 | 42 |
| holdout（C3 移出的旧 train） | 1,021 | — | — | — |
| test_q1／test_supp／test／test_indep（封存） | 2,287／936／4,174／52 | 2,168（test_q1） | — | — |

- 清单与视图行数的差：旧集合中没有 v2 双标签的行（与 `q1_v2` 一致），以及 W>3 的 eDOS 异常谱（15 个）。
- train 新标签来源：eDOS 为 MP 规范任务 30,226 个；phDOS 来源为 MP 声子（Pheasy）5,362、PhononDB 1,683、JARVIS 260、MP DFPT 4。
- train 中的标记：eDOS 有争议 395、体积差 >10% 509；phDOS 有争议 11、体积差 >10% 162、自带结构条目（link own）757、改挂（rehung）91。
- train 拟合的 C：`s1_v2` 为 eDOS 11.509、phDOS 0.04136；`s1_v2_x2` 为 12.249、0.04548。

### 核验（`verify_s1_v2.py`，报告 `splits/verify_report.json`）

- 公开与封存视图的全部负载文件 sha256 与清单一致（各 110 与 88 个）。
- 每个公开集合都能由训练用的 `Dos_Dataset` 读取，并带部分标签标志；无标签的任务目标全为 0，覆盖掩码全为 False。
- 旧材料行与 `q1_v2` 逐行对照：几何、原子数、价电子数、掩码完全相同；谱只在等于两侧异常截断阈值的箱上不同。
- 泄漏：train 与评估集共享的化学式组为 0；旧 valid/test 清单不变；公开视图中没有 holdout 材料。
- 继承自 `q1_v2` 的现象：train 中有 11 个旧材料带 phDOS 标签，但没有任何完全覆盖的箱。平衡损失用全窗口，不读掩码，因此不影响训练。核验改为只对新数据报错，旧材料的这类行计数报告。

### 训练侧改动（T1/T2/T3/T5，W1；T4 推迟）

- **T1 v2 协议**：`prepare_balance_protocol_v2` 只在带该任务标签的有效 train 行上拟合 C，网格取自视图清单，选型集必须是公开的双标签集合（`valid_q1`），写入 `label_contract: v2`。
  - 参照模式为"无"（每轮都有资格），等第一次 v2 下的 h2 运行导出后再固定参照。
  - 命令行工具：`tools/eval/spectral_balance_prepare_v2.py`。
- **T2 契约守门**：视图、协议、检查点三方的 `label_contract` 必须一致，任一不一致即拒绝。
  - v1 协议不得带 `label_contract` 字段，旧数据上的 v1 行为不变。
  - 数据契约只在视图清单带该字段时才写入，因此 v1 契约哈希不变。
- **T3 部分标签（W1）**：
  - 数据集读入 `has_edos/has_phdos`，放在 batch 第 18 字段；
  - 训练损失按任务只在有标签的行上求均值，再 0.5/0.5 相加；分块累积时用整个 batch 的标签数作分母，与不分块的结果相同；
  - 评估对无标签的任务记 NaN，摘要只在有标签的行上统计；某个任务完全没有标签时，联合分为 NaN；
  - 非 LR2 评估器和 `test_one_step` 遇到部分标签直接报错。
- **T5 分来源评估**：`utils/source_report.py` 按来源、链接类型、有争议、体积标记、独立条目和"干净"子集，分任务输出 n、E 均值、E 中位、R² 中位和失败率。
  - 旧材料的结构标记从该标签所来自记录的结构身份判定中取得。
  - `tools/eval/lg_valid_export.py` 对 v2 运行自动写出 `<out>_groups.json`；新增 `--view-split`，可对其他公开集合（如 dev）评分，不读 test。

### 测试与短检查

- 新单元测试 `tests/data/test_partial_labels.py`，11 项通过。覆盖：
  - 只在有标签行上拟合 C；协议存盘后重载，以及数据改动后被拒；
  - 契约双向拒绝；选型集必须双标签；
  - W1 损失与分块累积等价；无标签行梯度为 0；
  - batch 标志类型检查；分组报告；
  - 真实 `dev_indep` 读取与 collate。
- 全部现有测试 `python -m unittest discover -s tests -t .`：566 项，第一次跑出 7 个错误。
  - 4 个是已知前置条件：本机缺 `data/train4ARPAT` 原数组，`tests/README.md` 已记录。
  - 3 个由本次改动引起：`train_one_step` 的 `_accumulation` 改成四元组后，旧调用方仍传三元组。已改为兼容三元组，这 3 项复跑通过。
  - 归档测试 251 项，9 个错误，全部同为缺 `data/train4ARPAT`；静态检查 `check-static.sh` 通过。
- 协议：`results/runs/s1_v2_protocol_20261112/`（`valid_q1` 选型，C 与上表一致）。
- 短检查 `tools/checks/check_s1_v2_smoke.py`，报告在 `results/runs/s1_v2_protocol_20261112/smoke/report.json`。内容如下：
  - h2 配置经运行器组装后，valid 角色读 `valid_q1`；
  - 在真实部分标签 batch 上做 4 次一次性更新，损失与梯度有限；
  - lr=0 时训练目标与评估联合 E 之差为 1.2e-7；
  - 未训练模型完整评分 valid_q1（2,302）与 dev（eDOS 3,567／phDOS 713），分组报告正常生成；
  - 未写 checkpoint，未读 test。
- 旧 valid 的结构核对：
  - phDOS 结构不同 34 个，无法核对 118 个；
  - eDOS 可核对的 2,034 个全部匹配，其余 268 个在结构身份表中没有对应的 MP eDOS 记录。

## 2026-11-12 追加：完全重新切分（v3，用户决定）

用户决定不保留旧 valid/test 成员，把所有材料重新切成一个 train / valid / test（[设计中的记录](../design/design-2026-11-11-splits-v2.md)）。上面的 v2 产物保留作对照，不再用于训练。

- 规则：`splits_v3.py`，只看约分化学式；含稀有元素的组进 train，其余按哈希 10/10/80。新盐 `spectral-store-v1/splits-v3/2026-11-12`。
- 构建：`build_splits_v3.py`。标签、链接、标记、异常谱规则沿用 v2，记录表复用 `records_v11/` 并核对一致。
- 产物：清单在 `splits_v3/` 和 `sealed/splits_v3/`；视图为 `views/s1_v3`（128/64）和 `views/s1_v3_x2`（256/128），test 在 `sealed/views/`。

| 集合 | 清单 | 视图行 | eDOS | phDOS | 双标签 | 其中旧材料（原 train／valid／test） |
|---|---:|---:|---:|---:|---:|---|
| train | 56,334 | 56,193 | 48,809 | 25,980 | 18,596 | 14,938／1,818／1,796 |
| valid | 7,242 | 7,226 | 6,268 | 3,289 | 2,331 | 1,846／243／226 |
| test（封存） | 7,247 | 7,229 | 6,279 | 3,403 | 2,453 | 1,922／252／265 |

- 独立条目：train 806、valid 129、test 104，用 `is_entry` 标出。
- 清单与视图行数的差：旧材料中没有 v2 标签行的那部分。
- train 拟合的 C：`s1_v3` 为 eDOS 11.580、phDOS 0.04151；`s1_v3_x2` 为 12.376、0.04572。
- 核验 `verify_s1_v3.py`（报告 `splits_v3/verify_report.json`），全部通过：
  - 公开与封存视图的哈希一致；train 和 valid 都能由 `Dos_Dataset` 读取；无标签的任务目标为 0。
  - 每个旧材料行与它在 `q1_v2` 原集合中的那一行一致：几何相同，谱只在异常截断的箱上不同。
  - 没有化学式组跨集合；每个单元的归属都与规则重算一致；所有旧材料都在清单中。
  - train 中有 14 个旧材料带 phDOS 标签但没有完全覆盖的箱（继承自 `q1_v2`，不影响损失）。
- 协议 `results/runs/s1_v3_protocol_20261112/`：选型集为 valid（7,226 行，含部分标签），每个任务在自己有标签的行上求均值，再 0.5/0.5 相加。`_valid_cohort` 不再要求每行都有两种标签，只要求每个任务至少有标签行。
- 短检查（`check_s1_v2_smoke.py`，报告 `results/runs/s1_v3_protocol_20261112/smoke/report.json`）：
  - h2 配置在 `s1_v3` 上做 4 次一次性更新；lr=0 时训练目标与评估联合 E 之差为 9.0e-8；
  - 未训练模型完整评分 valid（eDOS 6,268／phDOS 3,289），分组报告生成；
  - 第一次运行时分组报告失败：`source_report` 写死了旧材料只出现在旧 train/valid。改为在三份旧原始导出中查找标签来源（只读编号列）后重跑通过。未写 checkpoint，未读 test。

## 2026-11-12 追加：独立条目去重后重建 v3（用户确认"照做，C"）

- `resolve_entries.py`：
  - 原先"匹配多个"的 62 条：严格口径下唯一匹配的 5 条改挂到 MP 材料，其余 57 条保留为独立条目。
  - 1,034 条独立记录按"同化学式、两两默认匹配且晶系一致"合并为 938 个单元：单条记录 870 个，两条 52 个，三条 12 个，四条以上 4 个；其中 41 个单元含不同来源的记录。
- 去重前的 v3 产物移到 `data/spectral_store_v1/superseded/v3_pre_entry_dedup_20261112/`，对应协议移到 `results/runs/superseded/`，未删除。
- 视图的 `link_codes` 新增 `entry_match: 5`，表示该标签来自同一独立条目的另一条记录。
- 核验增加一项：清单中的独立条目必须都是簇的代表记录。
- **重建结果（取代上一节的 v3 数字）**：

| 集合 | 清单 | 视图行 | eDOS | phDOS | 双标签 | 独立条目 |
|---|---:|---:|---:|---:|---:|---:|
| train | 56,261 | 56,120 | 48,809 | 25,907 | 18,596 | 733 |
| valid | 7,225 | 7,209 | 6,268 | 3,273 | 2,332 | 112 |
| test（封存） | 7,236 | 7,218 | 6,279 | 3,393 | 2,454 | 93 |

  - C：`s1_v3` 为 eDOS 11.580、phDOS 0.04154；`s1_v3_x2` 为 12.376、0.04576。
  - `verify_s1_v3.py` 全部通过：没有化学式组跨集合；所有单元的归属与规则一致；旧材料全在清单中；清单中的独立条目都是簇的代表记录。
  - 协议 `results/runs/s1_v3_protocol_20261112/`（valid 7,209 行）。短检查：lr=0 时训练目标与评估联合 E 之差 2.4e-8，未训练模型完整评分 valid 并生成分组报告；未写 checkpoint，未读 test。

## 结论与交接

- 事实：切分与视图按批准的规则构建完成，泄漏为 0；训练器已能在 v2 部分标签视图上运行 h2 配方。上述结论仅限所列测试和短检查，未做任何正式训练。
- 未做：
  - 在线备份（位置待用户定）；
  - T4 网格去写死（只在使用 256/128 视图时需要）；
  - 第 5 步新基线与多任务对照臂。
- 多任务对照臂需要放开 `validate_phdos_head` 对 `--train_subset` 等开关的限制，并给出具体实验方案，由用户确认后再实施。
