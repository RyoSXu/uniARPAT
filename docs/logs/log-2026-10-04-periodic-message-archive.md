# 周期几何与局部消息工作的文档整理和 Git 归档

日期：2026-10-04。用户批准先整理文档，再将当前工作分批提交。
本次只整理、复核并归档此前的 6.0 Å 工作，没有新增训练、GPU 数值作业或模型接入。
当前科学状态见[状态页](../status.md)，模块职责见[边界设计](../design/design-periodic-message-module-boundaries.md)。

## 文档职责与当前结论

- `status.md` 汇总各阶段的已完成范围和下一步；精确执行条件、明细与历史成本留在各日志。
- 记录规范补上独立物理接口，候选公式区别冻结原型与当前 A/B+ 核心；E(3) 目标页补上 CPU/GPU
  局部验收进度。日期更新为整理日，原实验日志保留 2026-10-03 的执行日期。
- 模块边界和导航指向当前可复用接口，CPU 解耦日志补充后续 GPU 证据入口；不改写当时的执行范围。
- 明确保留：完整 Transformer 尚未接入，生产路线与训练预算未定；局部通过不等于完整谱输出
  不变性，计算成本低不等于预测准确率高，近截止弱角度响应仍未解决。
- 各阶段累计回归数量不能相加为独立用例总数；初轮 B 原型与独立 B+ 核心也不能混称。

## 正式结果目录

这些目录承担当前通过结论，各自 manifest/verification 记录实际输入、源码、产物哈希和执行范围：

| 阶段 | 正式目录 | 依据 |
|---|---|---|
| 3–8 Å 半径审查 | [`periodic_neighbor_radius_q1_20261003`](../../results/periodic_neighbor_radius_q1_20261003/) | 全量 Q1 train/valid 结构，11 个候选；[审查日志](log-2026-10-03-periodic-neighbor-radius-audit.md) |
| 6.0 Å 几何 | [`periodic_geometry_q1_r6_20261003_v2`](../../results/periodic_geometry_q1_r6_20261003_v2/) | 3,352 项通过；[几何日志](log-2026-10-03-periodic-geometry-acceptance.md) |
| A/B 预检 | [`periodic_message_precheck_q1_r6_20261003`](../../results/periodic_message_precheck_q1_r6_20261003/) | 840 项通过；[预检日志](log-2026-10-03-periodic-message-precheck.md) |
| A/B 补充对照 | [`periodic_message_controls_q1_r6_20261003`](../../results/periodic_message_controls_q1_r6_20261003/) | 固定径向角度与独立 RSS；与预检证据配套 |
| B+ 高阶增强 | [`periodic_message_high_order_q1_r6_20261003_v2`](../../results/periodic_message_high_order_q1_r6_20261003_v2/) | 994 项通过；[高阶日志](log-2026-10-03-periodic-message-high-order.md) |
| A/B/B+ 独立 RSS | [`periodic_message_high_order_rss_q1_r6_20261003`](../../results/periodic_message_high_order_rss_q1_r6_20261003/) | 三个子进程的资源证据，与高阶对照配套 |
| 解耦与 CPU 迁移 | [`periodic_message_modules_q1_r6_20261003_v2`](../../results/periodic_message_modules_q1_r6_20261003_v2/) | 136 项及累计 56 项回归通过；[解耦日志](log-2026-10-03-periodic-message-modules.md) |
| GPU 局部验收 | [`periodic_message_gpu_q1_r6_20261003`](../../results/periodic_message_gpu_q1_r6_20261003/) | 966 项通过；[GPU 日志](log-2026-10-03-periodic-message-gpu.md) |
| GPU 预热补测 | [`periodic_message_gpu_timing_confirmation_q1_r6_20261003`](../../results/periodic_message_gpu_timing_confirmation_q1_r6_20261003/) | 三次预热、五次测量，八组路线/批次；保留首轮耗时峰值 |

运行记录里的 `code_version` 是当时 HEAD，运行时还存在未提交源码；实际执行身份以源码
SHA256 和快照追溯。归档不把历史版本号改写成新提交，也不修改已登记哈希的源码或结果。

## 中断和失败的追溯目录

以下三个目录保留，但不计入正式通过结论：

| 目录 | 实际状态与追溯 |
|---|---|
| [`periodic_geometry_q1_r6_20261003`](../../results/periodic_geometry_q1_r6_20261003/) | manifest 留在 `running`，另有 `interruption.json`；空索引列表的 dtype 问题导致中断。保留完整记录明细和当时源码，正式重跑在 `_v2` |
| [`periodic_message_high_order_q1_r6_20261003`](../../results/periodic_message_high_order_q1_r6_20261003/) | `incomplete / needs_changes`；误用晶胞重建函数参数，369 项记录后中断。保留 `interrupted_source`，正式重跑在 `_v2` |
| [`periodic_message_modules_q1_r6_20261003`](../../results/periodic_message_modules_q1_r6_20261003/) | `complete / fail`；136 项数值检查通过，但回归容器 dtype 不匹配，整轮判决失败。保留 `source_at_run` 五个快照，正式重跑在 `_v2` |

没有重命名、删除或覆盖这些结果。历史 manifest 的 `running` 是中断时状态，不能单独当作
当前仍在运行或已完成的依据；应连同中断标记与对应日志阅读。
高阶对照目录中的 `figure_initial` 也保留图表初稿及绘图源码快照；当前对比图使用该目录根部的
`comparison.png/svg`。两版图表均与各自 `figure.json` 的哈希一致。

## 分批提交的边界

依赖顺序为：半径/几何 → A/B 原型 → B+ 对照 → 独立核心 → GPU 验收 → 当前文档。
每批包含对应工具、测试、执行日志和结果；最后一批汇总当前规范、设计、导航与工具说明。
这组提交用于归档已有工作，不额外授权完整模型改动或训练。

| 批次 | 内容 | 文件数 | 提交 | 提交信息 |
|---|---|---:|---|---|
| 1 | 半径审查、几何验收与正式/中断结果 | 26 | `6376065` | `feat(geometry): 归档6埃邻域审查与几何验收` |
| 2 | 冻结 A/B 原型、无训练预检、补充对照 | 17 | `1e1453c` | `feat(eval): 归档周期消息AB无训练预检` |
| 3 | B+ 高阶增强、A/B/B+ 对照、图与 RSS | 32 | `973b284` | `feat(eval): 归档B+高阶角度增强与对照` |
| 4 | 物理记录/A/B+ 独立核心、按需导入和迁移验收 | 20 | `36777bc` | `refactor(model): 解耦周期几何与局部消息模块` |
| 5 | GPU 数值/梯度、固定批次计时、预热补测与复核 | 12 | `60acc74` | `test(cuda): 归档局部消息GPU验收与成本对照` |
| 6 | 当前状态、规范/设计、导航和本归档记录 | 10 | 本记录所在提交 | `docs(research): 整理周期消息进度与后续边界` |

共 117 个文件，最后一批的哈希从 Git 历史查询，避免在本文件中循环引用自身提交。

按明确文件清单逐批暂存，检查差异和内容一致性后提交，保持原分支与既有历史。
仓库统一忽略 `*.jsonl`，本次仅将五个被 manifest 哈希覆盖的周期验收 `cases.jsonl`
显式纳入对应结果批次；忽略规则不变。数据缓存、checkpoint、运行缓存不进入这些提交。

整理期间另出现晶格与原子位置拆分讨论稿 `docs/design/design-2026-10-04-crystal-structure-input.md`。
它不在起始清单，保留原样、未暂存；这 117 个文件的归档不包含该新增讨论稿。

## 归档前的验证

本次验证文档本地引用、Python 语法、行尾与差异；复核正式结果的输入/源码/产物哈希，
并调用现有 A/B、B+、GPU 产物核对函数重新核对明细、覆盖与摘要。
原始 CSV 使用标准 CRLF，差异检查用临时 `cr-at-eol` 设置识别，不改变文件或持久 Git 配置。
两份绘图库生成 SVG 保留路径末尾空格，单独做 XML 解析及图表哈希检查；手写源码/文档仍严格检查行尾。
六个正式 manifest 共 147 次哈希比较、60 个唯一文件通过；三次中断/失败的七个执行源码快照
与对应记录一致。A/B、B+、GPU 既有产物核对函数再次通过，文档本地引用、语法与差异检查通过。
没有重跑神经网络或重复全量几何枚举；各阶段的数值与回归结果来自原日志。
提交前逐批核对暂存文件，最后检查所有目标已入库、工作区状态及历史连续性。

## 后续 CI 修复

用户随后提供 GPU 批次的 CI 静态检查失败日志；定位到归档前已存在的诊断钩子变量清理。
局部修复与本地完整检查（182 项契约测试通过）见
[静态检查修复记录](log-2026-10-04-static-ci-hook-cleanup.md)。该修复另行追加，不重写上述六批提交或历史验收结果。
