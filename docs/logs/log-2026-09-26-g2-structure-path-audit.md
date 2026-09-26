# 日志：2026-09-26 — 冻结 G2 结构信息通路核验

## 范围

- 经马尚酱确认，在训练恢复保护修复后，检查 G2 分支是否影响 encoder、decoder 和谱输出，以及
  这种影响是否改善真实结构之间的谱差。设计见 `../design/design-g2-structure-path-audit.md`。
- 固定 Q1 train 18,706、valid 2,313，M1 epoch 10、seed 42、FP32、E0/P0、SumNorm、H1；
  `_g2ctl` 与 `_g2edge` 完整有效配置仅 `use_g2` 不同。
- 三个状态：独立训练的 control、原始 edge、同一 edge 权重在内存中将六层 G2 alpha 置零的
  edge_zero。没有训练、读取 test 或改写生产 checkpoint；B7 epoch 33 不参与本轮对照。

## 文件、符号与调用链

- `tools/eval/g2_structure_path_audit.py`：新增可重跑诊断入口。
  - `main` → `run_audit`：固定输入、检查配置与 checkpoint、组织推理、核对哈希并独占写出结果。
  - `load_split` → 既有 `ConfigBuilder`／`Dos_Dataset`，只允许 train/valid；
    `evaluate_model` → 既有 `basemodel.data_preprocess`／`Transformer.forward`。
  - `FeatureProbe` 读取最终 encoder 原子特征、eDOS decoder 特征及六层实际残差；
    `zero_g2_residuals` 临时关闭并在退出时恢复 alpha，`numerical_control` 检查重复与原子重排。
  - `make_pair_plan` 复用结构签名与既有 valid 材料对；`matched_atom_rms` 在同元素内部匹配原子，
    `relative_rms` 给出归一化响应，`contrast_metrics` 计算带符号谱差误差。
  - `spectral_batch_metrics` 复用 `per_sample_spectral_metrics`；`validate_history` 核对历史记录；
    `summarize`、`cluster_median_interval` 汇总结果及按组成组重采样的区间。
- `tests/test_g2_structure_path_audit.py`：`TestStructurePathMetrics` 与 `TestStructurePathProbe`
  共 7 项合成 CPU 测试；`tools/ci/check-static.sh` 纳入该模块。
- `tools/eval/README.md`、设计文档、`docs/index.md` 与 `docs/status.md` 同步入口和已完成状态。
  生产模型、训练损失和数据格式没有变化。

## 命令与结果文件

```bash
python3 -u tools/eval/g2_structure_path_audit.py --device cuda --batch-size 32 --output-prefix results/g2_structure_path_q1
```

- 实际完成时间约 395.5 秒；三个状态共 63,057 条逐样本结果，21,019 条同样本关闭干预。
- 同元素绝对计数、缓存结构不同的材料对：train 2,591 对／1,331 组，valid 320 对／170 组；
  三状态共 8,733 行。valid 材料 ID、配对及目标谱 TV 均复现既有谱形支持 CSV。
- 正式记录为 `results/g2_structure_path_q1` 前缀的 8 份 CSV 和 1 份 JSON：
  `samples`、`pairs`、`interventions`、`summary`、`comparisons`、`pair_summary`、
  `pair_comparisons`、`residuals`；JSON 保存输入与脚本 SHA256、alpha、数值对照和运行信息。
- 以下区间均为 95% bootstrap 区间、2,000 次重采样、固定 seed 20260926；全体指标按相同样本
  成对重采样，材料对按组成组重采样。区间没有覆盖不同训练 seed 带来的不确定性。

## 证据

### 1. G2 分支有实际作用，变化能传到谱输出

在同一 edge 模型内，开启与关闭 G2 的同样本响应中位数为：

| 划分 | encoder 相对 RMS | eDOS decoder 相对 RMS | eDOS 概率谱 TV | phDOS 概率谱 TV |
|---|---:|---:|---:|---:|
| train | 0.05313 | 0.06284 | 0.01401 | 0.01841 |
| valid | 0.05368 | 0.06374 | 0.01469 | 0.01826 |

RMS 是均方根差除以两端特征的平均尺度；TV 是归一化概率谱的总变差距离。两种距离只描述各自
层的响应，不能据此计算跨层信息保存率。六层残差／输入 RMS 的 valid 中位数为 0.01281–0.01827，
alpha 为 −0.03106 至 +0.03017；关闭后各层实测残差均为零。全部 valid 样本的 encoder、decoder
响应和 eDOS TV 均大于预设数值响应线 `1e-4`。

这排除了“该分支完全未生效／其影响在读出前完全消失”的解释。关闭整个分支同时去掉其元素与
几何相关贡献，不能把这些变化全部归因于正确的结构信息，更不能认定任何局部信息都被保留。

### 2. 已训练 edge 依赖该分支，独立训练对照仍无准确率收益

下表均为中位 R²／失败率，失败指 R² < 0，单位为 %。

| Q1 划分、ep10 状态 | eDOS oracle | phDOS oracle | eDOS blind | phDOS blind |
|---|---:|---:|---:|---:|
| train control | 0.49502／2.764 | 0.74765／0.909 | 0.45788／5.180 | 0.74026／1.668 |
| train edge | 0.49339／2.823 | 0.74792／0.887 | 0.45730／5.635 | 0.74088／1.721 |
| train edge_zero | 0.48375／3.341 | 0.73529／1.176 | 0.44367／6.495 | 0.72673／1.962 |
| valid control | 0.46203／4.626 | 0.71985／2.032 | 0.42658／7.869 | 0.71056／2.940 |
| valid edge | 0.46132／4.885 | 0.72281／2.335 | 0.42896／8.258 | 0.71307／2.853 |
| valid edge_zero | 0.45302／5.620 | 0.71294／2.421 | 0.40687／9.166 | 0.70132／3.113 |

- **edge−control，valid eDOS oracle：**Δmedian `−0.000709`，CI `−0.007604…0.005365`；
  Δfail `+0.259` 个百分点，CI `−0.259…0.735`。四项 oracle/blind 中位数差的区间均包含零；
  中位数和失败率点估计均处于既有平局线内。仍没有晋级依据。
- **edge−edge_zero，valid eDOS oracle：**Δmedian `+0.008300`，CI `0.004149…0.014336`；
  Δfail `−0.735` 个百分点，CI `−1.124…−0.389`。blind Δmedian 为 `+0.022090`，
  CI `0.015831…0.028596`。说明当前权重依赖该分支；不能当作重新训练后的 G2 收益，或将
  edge_zero 当作独立训练的 control。

### 3. 同组成材料的结构条件谱差仍未得到明确改善

valid eDOS 配对结果如下；所有数值均为对应材料对的中位数。

| 材料对范围 | 对数／组数 | 目标 TV | control 预测 TV | edge 预测 TV | edge_zero 预测 TV |
|---|---:|---:|---:|---:|---:|
| 全部 | 320／170 | 0.26975 | 0.006361 | 0.007188 | 0.007106 |
| 单元素 | 21／10 | 0.27177 | 0 | 0.000764 | 0 |
| 多元素 | 299／160 | 0.26883 | 0.007188 | 0.007537 | 0.007611 |
| 多元素且结构匹配判为不同 | 258／128 | 0.28229 | 0.006773 | 0.007530 | 0.007570 |

带符号谱差误差定义为 `0.5 × sum(abs((pred_a-pred_b)-(target_a-target_b)))`，越低越好。

- 全部 valid 对的误差中位数为 control `0.268933`、edge `0.268267`、edge_zero `0.267934`。
  对组成组先取中位数再等权汇总时，三者为 `0.269739/0.272501/0.272046`，没有稳定优势。
- **逐对差的中位数** edge−control 为 `−0.00001135`，按组成组重采样 CI
  `−0.00022318…0.00001002`；edge−edge_zero 为 `−0.00000843`，CI
  `−0.00006397…0.00000242`。这是成对误差差值的统计量，不能与上面的“两臂中位数之差”混用。
- train 也未显示明显改善：全部对的目标 TV 中位数 `0.26583`，control／edge 的预测 TV 为
  `0.005689/0.005846`，带符号谱差误差为 `0.266239/0.266230`。
- 单元素盲点被打破到超过数值误差的程度，但响应仍小；不能将“能区分”解释为“已预测对”。

### 4. 实际执行的核验

- 原始两臂 valid oracle 的中位 R²和失败率均在 `2e-5` 容差内复现各自 epoch 10 history。
- 重复前向／同步原子重排的最大谱 TV 为 `4.87e-7`，低于设计上限 `1e-5`。
- 原 alpha 在干预结束后逐位恢复；23 份已登记输入的 SHA256 在运行前后相同，包含两份 checkpoint
  和两份配置；样本 ID、计数、配对和核心数值有限性检查通过。
- `python3 -m unittest tests.test_g2_structure_path_audit`：7/7 通过。
- 新入口与测试的 Ruff 检查通过；`bash tools/ci/check-static.sh`：静态检查、编译和 66/66 项
  CPU 测试通过。前一恢复修复已运行完整测试 133/133；新增本模块后没有另跑完整套件。
- 汇总中 train 的置信区间留空，control 的 G2 残差列留空；这些是未定义项，不是非有限预测。

## 结论

- **状态：closed（诊断完成）；G2 保持 park。**唯一机制结论：G2 分支有可测的内部与输出作用，
  但本轮没有证明它改善同组成结构的真实谱差，或超过独立训练的对照模型。
- 本轮证据不支持以“信息完全未注入／完全未传到读出”为由放大 G2 或更换读出；也不能反推
  结构无用、标签有错或优化是唯一根因。隐藏距离与冻结关闭干预不足以区分这些解释。
- 按已确认设计的结束分支，不强行提出新的架构或单因素训练 pilot。保留 B7 默认，不进入 G2
  35-epoch 确认、不扫描超参。本次没有形成需要写入 `decisions.md` 的新模型选择。

## 交接

- 已确认的恢复保护与冻结通路诊断均已完成，当前无待运行训练。
- 下一次选题必须先提出能区分剩余解释的具体干预及相反结果预期；主验收仍为全体 eDOS 中位
  R²／失败率，结构配对误差作为机制证据，blind／phDOS 作为保护项。当前证据不足以选定唯一因素。
- `docs/status.md` 标记完成、明确未决问题与暂停新 pilot 的依据；`docs/index.md` 标记设计已完成。
