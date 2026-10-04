# 日志：2026-10-04 — 仓库目录整理

## 范围与保留边界

按用户批准的 7 个业务一级目录方案迁移目录、导入、默认路径、CI、文档导航和设计日期。
方案与创建时间证据见[目录约定](../../../docs/repository.md)及[逐文件迁移记录](../../metadata/repository-migration-2026-10-04.json)。
没有重建数据、下载、启动正式训练、更新 checkpoint、改变归一化或提交 Git。

开始时 HEAD 为 `a6a3a0e`，工作区仅有未跟踪的晶体结构输入讨论稿。该稿仍未跟踪，正文保留，
仅修正 4 个因目录迁移失效的链接。权重在同一文件系统重命名目录，没有复制或删除。

## 变更地图

- 数据调用链：`ConfigBuilder.get_dataset → utils.dataset.Dos_Dataset`；生产网格与元素特征
  从 `data/reference/` 读取，Q1 样本格式和内容保持原样。
- 训练/推理：`tools.train.train_and_eval` 默认配置、报告和运行目录按仓库根定位；
  B7 加载仍严格核对 M1/seed42/epoch33 与所有参数键。
- 路径与证据：`repository_path / input_path / historical_path / protected_path` 分别处理
  路径映射、自定义输入、历史原文身份与实际执行输入保护；原 manifest 中的路径和哈希保留。
- 数据命令平放于 `tools/data/`；显式源/目标参数和非空目标保护避免误写 Q1 或旧网格。
  `c2b_grids` 额外核对源池 ID 顺序及结构行数；没有执行生成流程。
- 文档：42 份无日期设计正文加编号，两个模板移至 docs 根部；README 明确 ZP100 研究基线和 B7 指标参考。
- CI：改为 `tools/checks/check-static.sh`，扫描实际存在的活动/退役源码，历史证据源码单独做语法检查，
  不依赖迁移期间尚未更新的 Git 索引。
- 忽略规则：修正 Q1 清单例外，明确 runs/archive 与五份已归档 JSONL 的边界。

## 验收

整理前：`python -m unittest discover -s tests -v`，338 项通过。
整理前还保存 B7 CIF blind 和 ZP100 固定 Q1 train/valid 样本的 CPU 输出、参数键与输入字段。
整理后检查记录在 `results/reports/repository_migration_20261004/`；最终结果与实际命令补记于下。

| 操作 | 实际结果与范围 |
|---|---|
| 完整 `python -m unittest discover -s tests -v` | 345 项通过；包括原 338 项和新增 7 项路径/身份检查 |
| `bash tools/checks/check-static.sh` | Ruff、活动/历史 Python 语法通过，既有 CI 契约加路径测试共 189 项通过 |
| 最后 4 个启动导入顺序修正 | 对 4 个文件 Ruff/compileall；相关 `archive.code.tests.test_joint_content_pilot` 43 项通过 |
| 57 个活动命令 `python <absolute-entry> --help` | 从 `/tmp` 启动全部成功；冻结和无 argparse 的退役入口不计入 |
| B7 实际 CIF 前端 | CPU、2 原子 Si、epoch33，使用旧绝对 checkpoint 路径；自定义临时目录中生成 NPZ/JSON/CSV，谱与整理前逐位一致 |
| B7/ZP100 CPU 固定输入回归 | 114 个张量和 431 个值全部一致，最大输出绝对差 0；ZP 使用固定 Q1 train/valid 各 3 个样本，严格加载相同参数键 |
| 模型源码 AST 对照 | 7 个 `model/*.py` 模块的 Python AST 全部相同，模型计算没有改变 |
| 历史证据函数复核 | A/B 840 项、B+ 994 项、GPU 966 项记录通过；CPU 冻结源码和实际执行保护通过；未重新运行神经/GPU 实验 |
| 文件完整性 | 981 份输入/报告/图像/运行文件保持原 inode、大小、mtime；有基准哈希的文件 SHA-256 一致；164 份权重原样保留，4 份 B7/ZP100 best/latest 另核 SHA-256 |
| 冻结源码与原文 | 14 份冻结/退役代码校验一致，348 份内容寻址原文校验通过 |
| 目录与引用 | 7 个业务根、43 份带日期正文；408 个当前 Markdown 文件引用通过，126 个有效链接与 30 个既有失效链接的状态保持 |
| Git 边界 | 26 个忽略规则样例通过，迁后的 656 个原受控文件均不被新规则屏蔽；HEAD/暂存内容没有改变；`git diff --check` 通过 |

第一次整理后测试有 3 项失败：测试仍以工作目录推导产物，而入口已按仓库根定位。
修正临时仓库注入后相关 13 项通过，再运行完整 345 项通过；仅这次测试新增的 3 个合成运行目录
已隔离到仓库外，没有触碰既有实验。启动检查另修正 4 个工具的导入先后顺序，随后复核通过。
训练有效配置继续保存原 CLI 的路径表示，路径只在实际文件读取时解析，避免改变历史配方比较口径。

逐目录与逐文件清单分别见 `directories_after.md / directories_after.csv`、`files_after.csv`；
检查汇总为 `validation.json`。验证产生的 Python/Ruff 缓存已清理，后续运行可再次生成。

## 已知历史限制

10 条历史源码身份整理前即不能从现有文件、快照或 Git 定位；不更改 manifest 哈希掩盖缺口。
具体身份与所属证据列在迁移 JSON。冻结原型/对照和 legacy 源码保持原文，完整历史复跑未验收。
原诊断报告中 1 个旧相对链接也保持原文，旁边新增 README 导航到当前设计和报告，当前文档导航已改为此入口。

旧网格 30 个失效链接与部分组合的行数混用保持并注明；恢复需要另定数据口径。
`stats_v2.json` 的历史池 `n=24988` 不能代替 Q1 总数 23306。
`q1_rebuild.py` 不是完整 sidecar 恢复工具，需独立核对 `nvalence` 等训练字段。
