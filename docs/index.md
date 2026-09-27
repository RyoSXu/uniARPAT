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
| 模型调研与升级的 agent 分工、模型配置和提示词 | `design/design-model-research-agent-orchestration.md` | 执行分工；候选1已完成 pilot 并 park |
| 联合边内容与谱监督的升级候选取舍 | `design/design-model-upgrade-candidates.md` | 候选1已完成 pilot 并 park；候选2未自动启动 |
| 联合边内容实现与CPU验收 | `logs/log-2026-09-27-joint-content-implementation.md` | 256维实现完成；后续阶段A资源已通过 |
| 联合边内容基线兼容性审查 | `logs/log-2026-09-27-joint-content-flash-review.md` | Flash初审及协调者纠正；不是独立最终验收 |
| 联合边内容资源核查与valid-only pilot设计 | `design/design-joint-content-pilot.md` | 已执行；阶段B正式平局并 park |
| 联合边内容pilot设计核查日志 | `logs/log-2026-09-27-joint-content-pilot-design.md` | 只读核查完成；未运行GPU、真实数据前向或训练 |
| 联合边内容RNG前置与阶段A资源核查 | `logs/log-2026-09-27-joint-content-rng-resource-tools.md` | V100三臂冷启动单步通过；阶段B随后获批并完成 |
| 联合边内容valid-only判决工具 | `logs/log-2026-09-27-joint-content-verdict-tool.md` | epoch10 latest、三臂同序与预注册阈值已锁定并正式执行 |
| 联合边内容Q1 valid-only三臂pilot | `logs/log-2026-09-28-joint-content-pilot.md` | 正式平局并park；未读取test，不进入M1×35 |
| R2a 共享 decoder 深度缩减设计 | `design/design-r2a-decoder-depth.md` | 已完成设计 |
| R2b 原子加性 phDOS 读出设计 | `design/design-r2b-atom-additive-phdos.md` | 已完成设计 |
| C2.1b 验证集损失归因审计设计 | `design/design-c2-1b-loss-attribution.md` | 已完成诊断 |
| C4 CUDA AMP 工程设计 | `design/design-c4-amp.md` | 已完成工程设计 |
| E5 训练／评估 checkpoint 代码边界设计 | `design/design-e5-code-boundaries.md` | 已完成工程项 |
| E6 长度分桶与动态裁剪 batch 设计 | `design/design-e6-bucketed-batches.md` | 已完成工程设计 |
| E7 最小 lint/CI 门禁设计 | `design/design-e7-lint-ci.md` | 已完成工程设计 |
| B7 M1 CIF 盲推理入口设计 | `design/design-b7-cif-blind-inference.md` | 已完成工程设计 |
| D4 phDOS 尖峰与负频坐标质量审计 | `design/design-d4-phdos-spike-imaginary-audit.md` | 已完成诊断设计 |
| D4b 负频坐标来源与稳定性审计 | `design/design-d4b-negative-coordinate-provenance.md` | 已完成只读审计设计 |
| D4c JARVIS `min_fd_phonon_mode` 语义审计 | `design/design-d4c-jarvis-min-fd-semantics.md` | 已完成并关闭的只读审计设计 |
| eDOS Q1 valid 预测形状误差诊断 | `design/design-edos-shape-error-diagnostic.md` | 已完成的只读诊断设计 |
| 高粗糙度 eDOS 一阶差分损失 pilot | `design/design-edos-slope-loss-pilot.md` | Q1 valid 成对 pilot 已完成并 park；未进入 35 epoch 确认 |
| 当前 G2 周期多镜像边条件消息设计 | `design/design-g2-periodic-multi-image-message.md` | 当前实施设计 |
| 冻结 G2 结构信息通路核验 | `design/design-g2-structure-path-audit.md` | 已完成的只读诊断；G2 保持 park |
| 冻结 G2 表征的结构谱差读出实验 | `design/design-g2-frozen-readout-probe.md` | 已完成；固定读出干预未获支持 |
| G2 encoder 与读出联合适配实验 | `design/design-g2-encoder-adaptation-probe.md` | 已完成；开放 encoder 更新未获留出收益支持 |
| G2 固定小样本的实际网络可拟合性检验 | `design/design-g2-small-fit-probe.md` | 已完成；联合第700步通过16／16，冻结末步8／16 |
| G2 消息分支的固定小样本拟合 | `design/design-g2-value-fit-probe.md` | 已完成；检查点最多10／16，末步4／16；逐对中间轨迹缺口见日志 |
| 冻结G2消息的固定小样本互补拟合 | `design/design-g2-non-value-fit-probe.md` | 已完成；第700步首次16／16，后续预定检查点及末步均保持 |
| 最后一个普通encoder层的固定小样本拟合 | `design/design-g2-last-layer-fit-probe.md` | 已完成；第2000步首次16／16并重载复现，仅一个成功检查点 |
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
