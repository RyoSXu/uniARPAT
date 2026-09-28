# AGENTS.md — 项目协作约定

uniARPAT 从晶体结构预测 eDOS 与 phDOS。当前科学目标与证据见 `docs/status.md`；总体路线见
`docs/design/design-structural-refinement.md`。

## 开始前

- 阅读 `docs/status.md`、`docs/index.md`；按任务需要阅读 `docs/workflow.md`、`docs/data.md`、`docs/glossary.md` 和相关证据。
- 修改前检查 `git status --short`，保留已有改动，不覆盖无关工作。

## 研究与实施

- 人负责目标、约束和取舍；Agent 应独立探索、核查证据、提出替代解释，也可以质疑当前路线图。路线图是方向，不是候选白名单。
- 区分代码/日志支持的事实、推断和待验证假设。历史实验只约束其实际检验的方案，不自动否定更大的研究方向。
- 在修改模型、改变数据契约或启动训练前，先交代目标、影响范围、判断标准和成本；需用户决定的事项得到确认后再执行。
- 实验设计可以是单因素对照，也可以是相互依赖的完整架构方案；按问题选择合适预算，不预设固定 pilot 长度。
- 模型选择使用 train/valid；不得用 test 调参或选型。结果报告中说明数据口径、任务、指标与 blind/oracle 模式。

## 保留边界

- 不同归一化方案不得复用 checkpoint。
- 未经明确批准，不运行 `--model all --epochs 100`，不删除 checkpoint，不执行破坏性 Git 操作或提交。
- 完成后简要说明改动、依据、验证及未解决事项；文档改动检查引用和差异，代码改动执行相关检查。
