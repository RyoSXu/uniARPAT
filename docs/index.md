# 文档导航

按用途而非日期阅读文档。以下活跃页面保持简洁；已替代材料可从 Git 历史恢复，但不得覆盖
当前规则。

| 需要了解什么 | 阅读位置 | 权威性 |
|---|---|---|
| 当前基线、已完成、当前关卡、待办与阻塞项 | `status.md` | 唯一整体计划 |
| 如何跨会话、跨 agent 工作 | `workflow.md` | 必须遵守的流程 |
| 术语、指标与实验状态 | `glossary.md` | 统一词汇 |
| 已确定的选择与基准事实 | `decisions.md` | 固化结论 |
| 训练数据与缓存边界 | `data.md` | 当前数据约定 |
| 结构重构方向的事实、候选与保留项 | `design/design-structural-refinement.md` | 讨论底稿 |
| R2a 共享 decoder 深度缩减设计 | `design/design-r2a-decoder-depth.md` | 已完成设计 |
| R2b 原子加性 phDOS 读出设计 | `design/design-r2b-atom-additive-phdos.md` | 已完成设计 |
| C2.1b 验证集损失归因审计设计 | `design/design-c2-1b-loss-attribution.md` | 已完成诊断 |
| C4 CUDA AMP 工程设计 | `design/design-c4-amp.md` | 已完成工程设计 |
| E5 训练／评估 checkpoint 代码边界设计 | `design/design-e5-code-boundaries.md` | 已完成工程项 |
| E6 长度分桶与动态裁剪 batch 设计 | `design/design-e6-bucketed-batches.md` | 已完成工程设计 |
| E7 最小 lint/CI 门禁设计 | `design/design-e7-lint-ci.md` | 已完成工程设计 |
| B7 M1 CIF 盲推理入口设计 | `design/design-b7-cif-blind-inference.md` | 已完成工程设计 |
| 当前 G2 周期多镜像边条件消息设计 | `design/design-g2-periodic-multi-image-message.md` | 当前实施设计 |
| 当前 E10 宏观晶格状态实验 | `design/design-e10-macro-lattice.md` | 当前实施设计 |
| 重大改动提案 | `design/_template.md` | 实施前复制并填写 |
| 已完成任务的证据 | `logs/` 与 `../results/` | 历史记录 |

## 命名规则

- 长期维护的说明页使用小写 kebab-case，例如 `workflow.md`。
- 带日期的记录使用 `YYYY-MM-DD-主题.md`；工作日志额外使用 `log-` 前缀。
- 新文档一律使用中文。命令、路径、代码标识、项目代号和已广泛使用的技术术语可保留英文；
  首次使用少见缩写时必须给出中文解释。
- 每个新增的活跃文档都必须加入此表。不得新增第二份索引、状态页、任务清单、归档目录或
  无边界的探索目录。
