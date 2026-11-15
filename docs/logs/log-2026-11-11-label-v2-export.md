# 第 2 步：旧材料的 v2 标签导出（2026-11-11）

按[标签契约 v2](../design/design-2026-11-11-label-contract-v2.md)和用户已批准的[异常谱规则](log-2026-11-11-spectrum-anomaly-rule.md)（经评审会话转达，原话"全部按建议"），把旧 train/valid/test 材料的标签从第 1 步认定的原谱重新生成，只换分箱。
不训练；冻结数据集、旧协议和 checkpoint 都不改；旧 test 只写入 `sealed/`，只看形状、有限值和计数。

## 做法

- 脚本 `tools/data/spectral_store/make_view_v2.py`。命令（在 `tools/data/spectral_store/` 下）：

      python make_view_v2.py --name q1_v2 --sealed --raw-points
      python make_view_v2.py --name q1_v2_x2 --bins edos=256,phdos=128 --sealed

- **原谱**：
  - eDOS：11-05 审计匹配的任务；归属其他材料的取 `staging/edos_legacy`；test 取 `sealed/edos_legacy_test`。
  - phDOS：审计匹配的 MP 任务。test 及没有匹配任务的 train/valid 标签，取本材料能通过旧规则门禁的 MP 声子谱。否则取第 1 步链接的 PhononDB / JARVIS 记录；JARVIS 按旧规则乘 3N ÷ 全谱面积。
- **分箱**：`common.exact_bin`，两种分辨率：
  - 主基准 `views/q1_v2`：eDOS 128、phDOS 64；
  - 2 倍分辨率 `views/q1_v2_x2`：eDOS 256、phDOS 128。
  - 计算晶胞原子数不同的，按原子数比缩放到输入晶胞。
  - mask 记录箱是否完全落在采样范围内（工作点不用）。
- **异常谱**：eDOS 的 W > 3 时剔除。phDOS 不筛，每行记 `phdos_modes_per_atom`（全谱面积 ÷ 输入原子数）。
- **尖峰截断**：旧规则（逐箱 p99.9、只截孤立尖峰）；阈值在同分辨率的 v2 train 标签上重新拟合，对三个划分统一施加。
- **只保留双标签的行**：训练代码还不支持部分标签。缺任一标签的材料不进视图，原因记在清单的 `reports` 里。
- **C**：用 `fit_task_normalizer` 在 v2 train 标签上拟合，写入清单 `normalizer_train`。
- **原始采样点导出**：每个导出标签背后的原始点，已换算到输入晶胞，含 JARVIS 缩放与来源编号。
  - train/valid 在 `exports/q1_v2_raw/`（train 37,322 行、valid 4,604 行）；test 在 `sealed/exports/q1_v2_raw/`。
- **修正**：`exact_bin` 对非负输入改为把累积求和的舍入误差（约 −4e-16）截到 0，加了单元测试。只影响 1e-16 量级，此前的测量结论不变。

## 结果

### 条数（两种分辨率相同）

| 划分 | 旧标签 | 导出（双标签） | eDOS 找不回 | eDOS 异常（W>3） | phDOS 找不回 |
|---|---:|---:|---:|---:|---:|
| train | 18,706 | 18,661 | 30 | 7 | 8 |
| valid | 2,313 | 2,302 | 8 | 0 | 3 |
| test（封存，只计数） | 2,287 | 2,168 | 6 | 0 | 113 |

- 每个材料只按第一个命中的原因计数，判断顺序是：eDOS 找不回 → eDOS 异常 → phDOS 找不回。
- **phDOS 来源**：
  - train：MP 14,940、PhononDB 1,864、JARVIS 1,857；
  - valid：MP 1,863、PhononDB 235、JARVIS 204；
  - test：MP 1,722、PhononDB 237、JARVIS 209。
- **phDOS 找不回的原因**：
  - train/valid 中有 11 条：材料自己有 MP 声子谱，但没有一条能复现冻结标签，审计也没记录匹配任务；
  - 另有 train 2 条：同化学式的 PhononDB 条目对不上。
- **test 的 113 条**：同化学式的 MP 声子谱 0 条（核对只输出计数）。之前"来源是其他编号的 MP 任务"的推断不成立，原因未明。
  - 细分口径：门禁日志的 no_candidate 46 / no_gate_pass 67，指同化学式的已入库候选（PhononDB 或 JARVIS）有没有；
  - 本会话报的 75 / 38，指 PhononDB 索引里有没有同化学式条目。
  - 两种细分的总数相同。
- **缩放到输入晶胞**：train 中 eDOS 5 条、phDOS 25 条；valid 中 phDOS 5 条；test 中 phDOS 1 条。

### C 与尖峰阈值

| 视图 | C eDOS | C phDOS | 被截断的箱（train / valid / test） |
|---|---:|---:|---|
| q1_v2（128/64） | 10.421 | 0.044927 | eDOS 366 / 51 / 46；phDOS 246 / 31 / 35 |
| q1_v2_x2（256/128） | 10.840 | 0.049575 | eDOS 650 / 81 / 76；phDOS 375 / 35 / 44 |

- v1 的 C 是 eDOS 5.5427、phDOS 0.036285。
- 阈值见各视图清单的 `spike_thresholds`。按中位数比，128 格 eDOS 阈值是在冻结 v1 train 上按同规则拟合值的 1.21 倍，phDOS 是 1.09 倍（v2 保留的面积更多）。

### valid 上的新旧标签对照（2,302 个材料，材料顺序与 v1 一致）

| | v1 ÷ v2 面积（中位，P10–P90） | 新旧标签间 E（用 v2 的 C；均值 / 中位 / P90） | 谱形相关（中位，P10） |
|---|---|---|---|
| eDOS | 0.711（0.560–0.825） | 0.121 / 0.068 / 0.277 | 0.970，0.891 |
| phDOS | 0.872（0.679–0.950） | 0.027 / 0.017 / 0.057 | 0.997，0.960 |

- 面积比与契约文档的估计一致：旧规则保留 eDOS 约 68–71%、phDOS 约 87%。谱形高度一致，说明 v1→v2 基本只是分箱的变化。
- 把 2 倍分辨率视图每两箱合并后，与主视图逐值一致的行占 eDOS 96.2%、phDOS 98.1%。其余行推断来自两种分辨率的尖峰截断不同（未逐行核实）。
- 两个视图都能被 `utils/crystal_cache` 读取，哈希校验通过。

## 备份与磁盘

- 增量备份到 `/root/spectral_store_backup/v1-20261111/`：6,689 个文件，5.22 GB。
- 系统盘 `/` 现在只剩 13 GB（87%）。PhononDB 全部下载完后，库里的原始压缩包会继续增大，下次备份前需要先看空间。
- **后续**：用户决定不需要本地备份（原话"本地备份没什么必要，可以清理"，经评审会话转达）。评审会话已删除 `/root/spectral_store_backup`，原库没动。不再运行 `backup_store.py`，在线备份放在哪里由用户稍后决定。

## 局限与待办

- 训练代码还不能读 v2 视图的契约标记，也不能读 256/128 格：
  - 它不会因 `label_contract: v2` 拒绝 v1 协议和检查点；
  - 网格是否写死为 128/64 也还未查。
  - 这些都属于第 5 步（新基线）开跑前的代码改动。
- 新材料的 v2 导出（规范标签与 Static 来源码）依赖第 4 步的切分，本次没有生成。
