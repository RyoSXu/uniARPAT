# 设计：D4 phDOS 尖峰与负频坐标质量审计

> 这里的 D4 是项目的数据工作标签，不是 `index/z0_REPORT.md` 中电子谱截断规则的 D4。
> 本项是只读诊断，不训练、不修改 Q1 标签或数据划分。

## 目标与成功判据

- **问题：**B7 的 phDOS 失败是否集中在两类可由 P0 标签量化的谱：负频坐标质量高的谱，或单 bin
  尖峰集中的谱？若是，是否强到足以支持后续独立的数据语义／质量假设？
- **术语边界：**P0 的负频是频率坐标，DOS 值仍非负。缓存没有单独的动力学稳定性标签，所以本项只报告
  “负频坐标质量”代理，绝不把它直接称为已证实的虚频或标签错误。
- **预注册描述量：**对原始非负 phDOS 标签 `y`，令 `S=sum(y)`：
  - `negative_mass_fraction=sum(y[center<0])/S`；P0 中共有 14 个负中心 bin。
  - `peak_share=max(y)/S`，是离散 bin 集中度，不假定尖峰本身有误。
  - `uncovered_mass_fraction=sum(y[~coverage_mask])/S`，仅作 coverage 完整性检查。
- **阈值与检验：**仅以 Q1 train 的负频坐标质量和尖峰集中度 p90 固定阈值；将 valid/test 映射为
  high/other。coverage 外质量仅检查是否为零或退化，不强行构造空的比较组。对 B7 test 的 phDOS，
  报告各组样本数、中位 R²、`R²<0` 失败率及 high−other 失败率差的固定 seed 2,000 次分层 bootstrap
  95% 区间。
- **可行动门槛：**只有当某一描述量的 high 组样本数不少于 100、失败率高出 other 组至少 3 个百分点、
  且 bootstrap 区间下界大于 0 时，才允许针对该代理另立数据语义／质量假设。它仍不自动授权标签删除、
  重建或训练。

## 改动

- 新增 `tools/eval/d4_phdos_spike_imaginary_audit.py`，读取 Q1 三个 split 的 phDOS、mask、P0 网格，
  以及 B7 `samples_m1_e9ctl_test.csv`；对齐 test 行数并读取 `test_index.npy` 作可追溯索引。
- 写出机器可读的全 split 描述量摘要、B7 分层摘要和 2,287 行 test 明细到 `results/`。不读 checkpoint、
  不计算预测、不写缓存。
- 新增合成单测，锁定 14 个负中心、训练阈值、零总量安全处理和 bootstrap 判决。

## 测试关卡

- 合成数组合同测试与 E7 静态门禁。
- 正式只读命令：

  ```bash
  python3 tools/eval/d4_phdos_spike_imaginary_audit.py
  ```

- 结果必须确认各 split 样本数为 18,706 / 2,313 / 2,287、无非有限描述量、B7 行数精确对齐。

## 成本与风险

- 描述量来自离散 P0 和现有标签，只能说明 B7 错误的分层关联，不能证明因果、物理不稳定性或应怎样修复。
- train p90 阈值避免由 test 性能反向选择阈值；test 仅用于 B7 泛化失败分层。
- 任一发现先进入设计／审计结论；Q1 的删除、标签变换、D3 数据工作或损失改动需要独立批准。
