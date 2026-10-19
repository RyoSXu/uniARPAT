# 日志：2026-10-19 — 数据发布的中英文说明与来源追溯

后续范围更正：本页记录最初包含 45 个文件的候选。用户随后确认仅发布当前主线数据，
新候选移除 3 个历史元素特征文件，保留 42 个文件；旧表审计不再作为发布附件。
当前范围、路径及验证以[重新打包记录](log-2026-10-19-mainline-data-repack.md)为准。

## 范围

用户要求补齐用于迁移/开源的数据说明，提供中英文来源与处理逻辑，并参考公开数据集的说明方式。
本轮核查既有文件、历史处理代码、上游说明与许可证据，补数据卡和逐材料来源附件。
不修改模型、训练配方、数据成员、谱值、划分或现有 manifest；未训练、推理、上传或操作远端 Release。
开始时 Git 工作区干净；本轮未提交。

## 核查结果

依据：当前 33 个数组、冻结 `v2_processed.parquet`、`q1_excluded_records.json`、完整结构 manifest；
历史处理代码 `8cf13336df6f56297ecc305cbc56ad4cace9caa6`（`e24d2a4^`）的
`tools/getdata/a4_process.py`、`a4_pass2.py`、`recompute_phonondb.py`；当前 Q1/结构迁移及 loader。

- 当前存储 23,306 条，全部结构与 eDOS 来自 MP。phDOS 为 MP 18,644、PhononDB 2,390、JARVIS 2,272。
  默认 train 过滤两条 PhononDB 记录，train/valid/test 为 18,704/2,313/2,287。
- 全部两类谱和覆盖 mask 按 ID 与父表逐行比较，数值相同；完整原子数与父表 `nsites_prim` 相同。
- 标签是密度值：自身任务 Fermi 对齐、spin sum、THz/cm⁻¹ 的轴与密度同时换算、JARVIS 按 3N 缩放、
  原采样点盒平均和既有尖峰处理。SumNorm 在 loader 执行，离散和不直接等于物理积分。
- 原盒平均未插值箱边界，不能称严格积分守恒；尖峰阈值为历史父池级，而非 train-only 估计。
- 当前完整存储结构的 ID 和绝对原子计数组成无跨集交集；约分组成 train–valid / train–test / valid–test
  交集为 254/253/44 个组成键。历史划分按最多 80 原子的绝对组成分组，不能称化学式完全隔离。
- 父表的 JARVIS `ph_weight=0.5` 仅作为历史来源元信息导出；当前 loader 未消费，不宣称现有训练已降权。
- MP 声子引用只保留 `pheasy/dfpt` 方法类别，缺唯一 task ID；链接按来源标识构造，不认证实时可达性。
- 恢复中的归一化检查发现 phDOS 全零窗口谱 train 7 / valid 1 / test 0，均与父表相同，eDOS 无全零谱。
  来源 CSV 增加 `edos_zero_window` / `phdos_zero_window`；原因未逐条归因，默认视图不变。

## 变化地图

| 文件 | 职责 |
|---|---|
| [中文数据卡](../dataset-card-zh.md)、[English](../dataset-card-en.md) | 等价描述来源、流程、单位/字段、实际划分边界、恢复与限制 |
| [来源说明](../dataset-sources.md) | 中英双语引用、分来源许可证据和明确缺口 |
| `tools/data/export_release_provenance.py` | `sha256` 记录指纹；`export_provenance` 先校验冻结谱/mask、ID、split、完整 N，再输出 CSV/JSON；`main` 提供只读输入的命令入口 |
| `README.md`、`docs/data.md`、`docs/index.md`、`tools/data/README.md` | 只加文档及工具入口，状态页不扩写 |

本地候选仍在 `results/runs/github_data_release_20261019/`，不进入 Git 的数据资产包括：
原 13.81 MiB 压缩包、新中英文 README、来源说明、来源 CSV、审计 JSON、发布清单、恢复验证和 SHA256SUMS。
来源导出完整覆盖 23,306 行，其中 23,304 行默认可用；源父表和 CSV 的完整指纹在审计 JSON 中。
CSV 为独立附件，不改变现有训练缓存或模型调用链。

## 许可依据与未解决项

参考 Matbench 的字段/来源元数据、MPtrj 的快照/引用规则、Hugging Face Dataset Cards 的来源/限制组织，
未使用这些参考项目的样本。PhononDB DOI 注册元数据明确 CC BY 4.0；NIST 官方区分普通数据、SRD 和第三方作品；
mendeleev 1.3.0 包许可证为 MIT，数值上游参考仍需保留。

MP Terms of Use 返回 403，未把父 README 的 CC BY 4.0 断言冒充本轮完整核查结果。
旧 `periodic_table_v2.csv` 最早追到 `19c21ed`，缺最初来源/许可；PubChem 风格相似不足以归属。
以上为首次准备时的证据边界，后续已补核 MP 逐记录许可并追到旧表原文件，见
[许可专项核查](log-2026-10-19-data-license-review.md)；旧表整体授权仍未闭合。
整包尚不宣称统一开源许可；未代用户为自有贡献选择新许可证。

## 验证

导出工具在真实父表和当前缓存上运行通过：全部 23,306 行谱/mask、split 和完整原子数一致。
压缩包及全部 45 个文件重新 SHA-256 核对通过，包哈希保持
`fa63a69150c2344b728a4f94de2c699271492e3aeffe8d72a66fe7e85bbce6b0`。
文档新增前的干净代码恢复已验证 train/valid loader；本轮只改说明及来源附件，不重复运行模型测试。
test 仅参与来源、组成、数组一致性与文件哈希检查，没有模型评估或选型。
初次 SumNorm 验证错误地要求全体谱归一后和为 1，因既有全零 phDOS 未通过；
最终按实际契约验证非零谱和为 1、全零谱保持 0，不把这项失败表述为数据恢复故障或静默删样本。
最终收尾检查通过：中英文文档及附件内部链接、SHA256SUMS 全部 8 项、已有输出目录拒绝覆盖、
人为不一致谱值拒绝导出且不创建输出目录。以 `git archive HEAD` 建干净代码目录，先校验已有 manifest/参考表，
再恢复压缩包并核对全部 45 文件；train/valid 的 SumNorm loader、样本数、有限值、非零谱和及全零谱保持验证通过。
未运行模型测试或训练；这些检查不等于已解决公开再分发许可与全零谱成因。
