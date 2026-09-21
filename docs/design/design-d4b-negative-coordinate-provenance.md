# 设计：D4b 负频坐标来源与稳定性审计

> 对应 `docs/status.md` 的当前 D4b 关卡。本项只读，不重建 Q1，不改 P0、标签、划分或训练配置。

## 目标与成功判据

- **问题：**D4 的 230 条高负频坐标质量样本（阈值仍是仅由 Q1 train 固定的 p90）能否逐条追溯到
  `v2_processed.parquet` 的 phDOS 来源？各来源原始记录是否包含可直接使用的动力学稳定性、虚频或
  计算收敛字段？
- **已知边界：**P0 负坐标质量是谱形代理；即使原始频率网格含负数，也不能替代声子本征模的稳定性
  判定。本项把“原始 DOS 在负坐标有质量”和“有独立的虚频／稳定性字段”分别报告。
- **完成判据：**
  1. Q1 test 的 2,287 条及 D4 high 的 230 条都必须一对一回连 `mpid`、`src_ph`、`ph_ref`；
  2. 对 MP、PhononDB、JARVIS 三类来源，明确列出可用字段、对应的原始归档文件、以及是否有逐材料
     的直接稳定性或收敛标识；
  3. 若 archive 可读，逐条提取与所选 phDOS 来源一致的原始频率范围和负坐标 DOS 质量；JARVIS 另行
     提取其已有的 `min_fd_phonon_mode`，但只记为来源提供的模式最小值，不把它擅自转成标签；
  4. 报告 high/other 的来源构成及来源内 B7 phDOS 失败率，分清来源混杂和 D4 原有的标签形状分层。

## 改动

- 新增 `tools/eval/d4b_negative_coordinate_provenance_audit.py`：读取已有 D4 逐样本表、Q1
  `test_index.npy` 和 Git 外的原始归档（路径由 `--raw-root` 显式指定）；只写 `results/` 的派生
  CSV/JSON。原始归档缺失时必须明确失败，绝不以缓存标签冒充原始元数据。
- 选择 MP 原始记录时按照 C2b 的既有优先规则并核对 `ph_ref`；PhononDB 记录仅审计其
  `freq_THz`/`dos`，JARVIS 只读取 Q1 test 所需的行和其公开保留的字段。不得扫描或更改 Q1 缓存。
- 输出逐样本的来源／字段可用性表、按来源与 high/other 分层的摘要、以及机器可读结论。
- 新增合成单测，覆盖高组布尔值解析、来源／引用一对一合同、MP 选择与负坐标质量计算；不将 14 GB
  JARVIS 归档纳入 CI。

## 测试关卡

- 合成合同测试与 E7 静态门禁。
- 本地完整只读审计：

  ```bash
  python3 tools/eval/d4b_negative_coordinate_provenance_audit.py \
    --raw-root /root/home/newstudy/getdata/raw
  ```

- 结果中必须同时报告原始归档覆盖率和稳定性字段覆盖率；“没有字段”是有效的审计结论，不能用推测值
  填充。

## 成本与风险

- 原始 archive 在 Git 外且 JARVIS 文件较大；审计须以流式读取和目标 `mpid` 集合为界，避免加载全表。
- `min_fd_phonon_mode`、频率网格的负端或 DOS 负坐标质量均不足以单独证明计算收敛、实验可实现性或
  应删除样本。只有来源中本来就存在且完整映射的独立稳定性／收敛标识，才允许另起数据处理设计。
- 来源内的 B7 失败率是描述性分层，不是新的测试集选择或重训依据。
