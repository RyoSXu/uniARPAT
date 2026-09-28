# 日志：2026-09-28 — D3a eDOS-only 候选支持面裁决

## 范围

- **项目决策：**判断本地 121,420 个 eDOS-only 候选是否值得进入 10–20k 定向扩数 pilot。
- **改动范围：**新增可重跑的本地归属、质量审计和支持面工具及 16 项 CPU 合同测试；冻结 1,000 条
  样本并产出逐样本、Q4 配对和汇总结果；同步数据、状态、索引与决策文档。
- **明确排除：**未联网、未读取凭据、未下载或改写原始数据、未重建 Q1、未训练、未运行 GPU、
  未读取 test 标签，也未修改模型或训练默认值。

## 方法与冻结边界

- 候选集合为 `E − (P ∪ C) = 121,420`；500 条概率臂按
  `sha256("d3a-edos-support-v1\0" + mpid)` 固定，500 条定向臂只使用候选和 Q1 train 的结构元数据。
- 冻结样本清单 SHA-256 为
  `9538a1628f9dffc958735c7c189e5162d9b22dc44d7476bfb6cc4f3c6ffad874`；1,000 条唯一，和现有
  24,988 条成品零重叠。
- 本地 Delta task 先按 `(nsites, species)` 与体积粗筛，再用冻结 `StructureMatcher` 参数复核；只接受
  GGA/GGA+U。谱按 E0 128 bin 和冻结 winsor 阈值处理。
- 严格可用要求近似 `C_full/N_val >= 0.5`。候选 task-cell `N_val` 由 Z0 元素表估算，因此另做已有
  6,344 条增量材料的边界复核，并增加不依赖截断筛选的“全部可读谱”支持面敏感性。

## 结果

### 归属与严格质量

| 状态 | 数量 |
|---|---:|
| 严格可用 | 507 |
| 近似截断 | 186 |
| 无本地候选 task | 280 |
| 有 task 但结构不匹配 | 21 |
| run type 不兼容 | 6 |

概率臂 500 条中严格可用 265 条，点估计 53.0%，Wilson 95% 区间为
`[48.62%, 57.34%]`；下限未达到预注册的 50%。507 条严格可用谱全部为 GGA/GGA+U。

### 截断近似复核

现有 6,344 条增量材料中，冻结 Z0 有 6,328 条有限截断标签。元素表近似相对冻结 D1 判定有
2 条假排除、0 条假放行，有限标签判定一致率为 `6326/6328 = 99.968%`。其余 16 条冻结值缺失：
近似方法把 Q1 已隔离的 14 条也判为低比值，另外 2 条缺少可比结果。近似判定足够保守，但不冒充
逐材料完整 Z0；正式结论同时要求下述全部可读敏感性不改变方向。

### valid Q4 支持面

| 候选谱集合 | 数量 | Q4 严格改善 | 改善比例 | 距离中位相对下降 | 中位下降 bootstrap 95% |
|---|---:|---:|---:|---:|---:|
| 严格可用 | 507 | 5/595 | 0.840% | 0 | `[0, 0]` |
| 全部可读敏感性 | 693 | 9/595 | 1.513% | 0 | `[0, 0]` |

严格集到原 Q1 train 目标谱的最近 TV 中位数为 0.4782，96.65% 高于冻结 train Q3 阈值
0.253176，说明它们自身大多新颖；但这种新颖性没有覆盖当前 Q1 valid Q4 的缺口。这里的“支持面改善”
只描述目标谱最近邻距离，不表示模型准确率收益。

## 裁决

- **状态：closed。**概率臂 Wilson 下限、Q4 中位相对下降、bootstrap 下限和 Q4 改善覆盖比例四项
  主门槛全部失败；样本唯一性和 run type 兼容性保护项通过。
- 按预注册停止：不做 10–20k pilot，不修改数据契约，不采集，不训练。该结论只关闭当前本地候选集合、
  元数据定向抽样和本地 Delta 归属组合；不能排除新的外部来源或未来按可靠谱代理定向检索。

## 代码与产物

- `tools/eval/d3a_edos_support_adjudication.py`
  - 集合与抽样：`candidate_ids`、`load_metadata`、`select_sample`。
  - 归属链：`load_direct_task_map` → `index_candidates` → `load_delta_rows` →
    `match_sample_tasks`。
  - 谱处理：`process_spectrum` 调用 `valence_count`、`box_average`、
    `winsorize_isolated`、`normalize_spectrum`。
  - 裁决：`support_metrics`、`wilson_interval`、`bootstrap_median_ci`；`main` 原子写出正式结果。
- `tests/test_d3a_edos_support_adjudication.py`：16 项抽样、分层、谱处理、统计和写入合同测试。
- `tools/ci/check-static.sh`：纳入上述合同测试。
- 正式结果：`results/d3a_edos_support_sample.csv`、
  `results/d3a_edos_support_q1_valid_q4.csv`、
  `results/d3a_edos_support_adjudication.json`。

## 验证

- `python3 -m unittest tests.test_d3a_edos_support_adjudication -v`：16/16 通过。
- `python3 -m unittest discover tests`：253/253 通过。
- `bash tools/ci/check-static.sh`：Ruff、Python 编译和 179 项缓存无关合同测试通过；
  `git diff --check` 通过。
- 结果复算：1,000 条唯一、与成品零重叠、严格可用 507、全部可读 693、Q4 为 595 条；严格／
  敏感性改善分别为 5/9 条，汇总状态与门槛一致。

## 交接

- 下一关回到模型机制，但不直接训练。推荐先对冻结 B7 做逐层响应收缩定位：在既有 Q1 valid
  同组成材料对上比较 encoder memory、decoder hidden state 与最终谱输出的配对差异保留率，判断差异
  首先在哪一段显著收缩。它只用于选择下一项独立 valid 干预的模块，不据此宣称因果或精度收益。
- 若 encoder 已收缩，下一候选须针对结构条件表征且区别于已平局的径向／联合边内容；若 decoder
  才收缩，设计条件化读出干预；若只在输出／目标处收缩，才考虑谱差或不确定性目标。若没有稳定层间
  转折，则停止架构定位，重新审查训练目标假设。
