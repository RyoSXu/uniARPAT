# 日志：2026-10-04 — 仓库第二轮整理

## 范围与结果

用户批准统一历史归档、收紧 Git 大载荷边界、分类测试、移动入口/模板及清理冗余材料。
本轮移动 1662 个文件/链接；保留 8 个业务一级目录。根部只保留 5 个必要项目文件。
历史材料统一进入 `archive/code/docs/data/results/metadata` 分类；当前目录见[目录约定](../repository.md)。
36 份历史设计、108 份历史日志退出当前文档分类；当前设计/日志仍按证据日期编号。

90 份原始或中间载荷退出 Git 索引，本地 68.70 MiB 保留。
有限摘要、参考表、原始执行源码证据用精确例外保留，单文件上限 2 MiB。
本地提交/推送钩子已启用；CI 检查静态契约和每个受控 blob。旧历史中的原始表仍在，
实际推送钩子的无网络模拟已拒绝向新远端传输，不把忽略规则宣传为历史已经清理。

清理 21 份重复盘点/通过日志和 269 份重复源码快照；269 份都已逐一从固定 Git 身份恢复并校验 SHA-256。
80 份必要原文保留。重复源码恢复缓存和 3 个可再生缓存文件检查后清除。
失败原因及最终通过状态合并保存，不删除唯一科学证据。

## 变化地图与调用关系

- 训练/推理：`tools/train.py::train_and_eval`、`tools/infer_b7_cif.py::main` 保留入口参数，默认配置为
  `tools/training-defaults.yaml`。权重/有效配置仍在当前 B7/ZP 运行目录；预测/目标数组与逐样本表写入对应运行目录，
  CIF 默认写 `results/runs/b7_cif_blind`，自定义输出参数保持原含义。
- 输入/身份：`utils.paths::repository_path/input_path` 解析两轮旧名、分段路径和报告前缀；
  `historical_path/_git_source` 核验原始字节，`protected_path/_source_name` 限制实际执行源码只能接受登记适配。
  Q1 manifest 保持原文，`data/quarantine_q1.json` 兼容链接指向明确命名的参考清单。
- 当前支持库：`utils.periodic_neighbors` 保留原邻居接口；`utils.thermodynamics::ThermodynamicCalculator`
  的类 AST 与原库一致。旧热力学五组批处理归档，当前训练导入纯计算库。
- 历史兼容：`model.transformer::Transformer` 对旧 G1/宏晶格/周期多体实现按开关加载；
  `tools.train::maybe_get_pair_aux_loader` 对 pair 辅助按需加载。当前默认入口和默认模型实测不导入归档代码。
- Git 边界：`tools.checks.check_repository::check_entries/check_index/check_worktree/check_revisions/push_revisions`
  共用真实忽略规则并检查 blob 原体积；钩子调用暂存检查及待发送历史检查。
- 测试/文档：当前测试分为 `model/data/pipeline`，历史测试位于 `archive/code/tests`；
  两个 `template.md` 分别回到设计与日志目录。迁移 JSON 是运行兼容登记，已移入 `archive/metadata`，不能当清单删除。

## 实际验证

| 检查 | 结果与边界 |
|---|---|
| 当前 / 历史兼容测试 | 102 / 251 项通过，合计 353 |
| Ruff、内存语法编译、CI 契约与工作区载荷检查 | 222 项通过 |
| 不带本地数组/权重的可提交仓库副本 | 同 222 项检查，跳过 1 个明确依赖本地 Q1 的测试；复制模拟，并非远程克隆 |
| 仓库外启动 CLI | 61 个当前/历史维护入口帮助通过；B7 实际 CIF 推理的谱与整理前完全相同 |
| 当前 B7/ZP 固定输入 | 114 个张量与 431 个其他值完全一致，CPU fp32 eval；B7 CIF blind、ZP 固定 train/valid 样本 |
| 文件保全 | 813 个载荷 SHA-256 不变，Q1 全部 26 个文件不变；165 个权重后缀文件 inode/体积/修改时间不变，4 个关键 checkpoint 再核 SHA-256 |
| 模型/热力学 | 7 个模型模块去除仅导入节点后 AST 一致；热力学类 AST 一致 |
| 既有预检/高阶/GPU 证据 | 840 / 994 / 966 条案例身份和一致性核对通过；没有重新执行神经网络验收 |
| Git 边界 | 强制加入的载荷、超限 blob、历史原始表均被拒绝；隔离索引中的最终目录树通过，实际索引仅移除批准载荷 |

全部细节见[验证记录](../../archive/metadata/repository-validation-2026-10-04.json)与
[迁移登记](../../archive/metadata/repository-migration-2026-10-04.json)。本轮初始检查发现的导入、分段路径、
报告前缀和临时夹具问题均修正后复核，未把失败尝试记录为通过。

## 保留的问题与 Git 状态

共享 Transformer 的历史参数开关为兼容权重保留，尚未整体拆分。旧网格 30 个失效链接及
10 条整理前缺失的源码身份保留说明；旧热力学批处理默认输入仍缺失，不作为当前入口。
原始诊断 `report.md` 的旧设计链接保留原文，旁边 README 提供现路径。

本轮不训练、不提交、不推送、不切分支、不重写历史、不删除 checkpoint。
HEAD 保持原值；实际索引只有 90 份批准载荷的删除，其余布局与源码改动保留待审阅状态。
用户已有的晶体结构设计稿保持未跟踪，正文保留。当前实际配置有 `github` 远端，
此前只读方案记录“无远端”；本轮未改远端，也未访问远端确认其历史。
