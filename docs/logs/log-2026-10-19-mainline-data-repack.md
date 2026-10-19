# 日志：2026-10-19 — 当前主线数据重新打包

## 范围与依据

用户确认只发布当前主线训练所需的冻结数据、必要参考、来源记录和中英文说明，并授权重新打包。
工作区已有数据发布文档和来源导出工具改动，全部保留。本轮只调整本地发布候选与配套文档，未提交或上传。

此前把整个 `data/reference/` 纳入候选，误将历史元素性质表当作主线必需项。
依据[元素入口设计](../design/design-2026-10-02-element-initialization.md)和
[当时验证](log-2026-10-02-eid-zonlyproj.md)，ZP 仅使用原子序数，不读取质量、半径、电负性。
本轮又在不含这些表的干净代码副本中验证加载与前向，不修改模型或数据契约。

## 当前发布内容

新候选目录：`results/runs/github_data_release_20261019_mainline/`。
旧 `results/runs/github_data_release_20261019/` 保留本地历史，不作为当前发布附件目录。

- 数据包内共 **42 个文件**：当前缓存 33 个数组及 manifest，加 8 个参考文件。
- 8 个参考为：`split_v2.yaml`、`grids.json`、`c2b_grids_report.json`、`q1_excluded_records.json`、
  `valence_by_element.json`、`edos_valence_by_material.parquet`、`edos_truncation_by_material.parquet`、`edos-label-audit.md`。
- 从新候选排除 `periodic_table_v2.csv`、`atom_features_mendeleev.csv`、`atom_features_mendeleev.meta.json`；
  原仓库与旧候选中的文件均保留。`element-table-audit.json` 仅留在旧候选，不作为新附件。
- 保留中英数据卡、来源/许可说明、逐材料来源 CSV、来源审计、MP 逐条许可记录与审计、发布清单、验证记录和 SHA256SUMS。
  来源与 MP 许可的 4 个附件逐字节复用，未重新采集上游数据或改写许可结论。
- 沿用 `crystal-q1-v1` 和建议标签 `data-crystal-q1-v1`。本轮只是缩小包装范围，材料、谱值、mask、价电子数、划分及原 manifest 均不变。

压缩包 `uniARPAT-crystal-q1-v1.tar.gz` 为 **13,979,000 bytes（13.33 MiB）**，SHA-256：

```text
166c626950ad4a180bbddb17c919142ab1887cb7f5bf62a0ae48b02bf7dae2bb
```

原数据 manifest SHA-256 保持：

```text
8e1b0adff516835d52d9fb0321dae5e66fb9a2ac5fd964d317e1dd8647de4b80
```

## 文件与数据流

两份 `docs/dataset-card-*.md` 更新 42 个文件的实际范围、8 个参考文件和恢复说明；
`docs/dataset-sources.md` 仅描述保留内容的来源、引用与许可，历史元素表调查继续保留在原日志。
`docs/data.md` 和两份前期日志增加当前候选入口，避免把旧候选范围当成最终范围。
发布副本重写内部链接以适配附件名称；发布清单重新记录压缩包与说明附件哈希。
无模型、loader、训练函数或公共接口变更，`export_release_provenance.py` 保留原实现。

## 验证与边界

- 从 Git `e4cc7abdef7590632925c81ec5b2dd3853455ffb` 导出干净代码，在临时副本中省略 3 个历史特征文件后恢复新数据包。
  包成员严格为 42 个普通文件，全部文件与原冻结载荷及本地数据的 SHA-256 一致；原 manifest 不变。
- `crystal_data_contract` 通过；真实 SumNorm/坐标开启的 train/valid loader 得到 **18,704 / 2,313** 条，
  网格为 128/64；非零谱离散和为 1，全零 phDOS 保持 train 7、valid 1。
- 当前 ZP + 完整 A/B+ 分别通过 CPU 构建与一个训练样本的前向，`d_model=16`、单层 decoder；输出有限、谱维度正确。
  临时令 `pandas.read_csv` 读取即报错，加载与前向中调用次数为 0。
  这是恢复及依赖检查，不是完整训练、性能或对称性验收。
- 全三划分 23,306 个存储 ID、来源 CSV 行序/原子数、23,304 个默认可用标记核对通过；MP 许可记录 ID 集合一致且均为 BY-C。
  test 仅检查存储数组、来源与元信息，没有模型推理或指标计算。父谱逐行相等证据沿用未变的来源审计及未变的数组哈希。
- 文档与附件内部链接、差异空白检查和最终 10 个附件的 SHA256SUMS 校验通过。

本轮未改变任何来源许可，也未替用户为项目贡献指定新许可证；后续正式发布仍保留分来源署名、条款和处理说明。
