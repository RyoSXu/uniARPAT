# 日志：2026-10-09 — 局部结构信息的贡献诊断

## 范围

用户同意先诊断已有模型，再整理最小对照协议。本单元完成诊断与协议，未启动新的正式训练，
未修改模型公式、checkpoint、归一化或数据。只读取 train/valid。
推理时临时屏蔽分支，仅存在于诊断进程，结束后恢复；不作为生产设置或重训消融。

代码与调用链：

- [lg_structure_diagnosis.py](../../tools/eval/lg_structure_diagnosis.py)：`diagnose` 重建已保存模型，
  复用 `load_run/select_best → ConfigBuilder → evaluate_split`，按材料 ID 对齐。
- `pair_errors/summarize_pairs/paired_group_interval`：有符号谱差误差、零响应参照及组成组重采样。
- `geometry_table/attach_support/strata`：当前完整晶胞的径向支持代理与 train 定义阈值的分组。
- `InformationProbe/feature_pairs`：置换不变的中间表示汇总；`intervention`：异常退出也恢复的推理 hook。
- `structure_matches`：相近结构与目标谱差，不自动判定坏标签。
- [check_lg_structure_diagnosis.py](../../tools/checks/check_lg_structure_diagnosis.py)：`verify`
  独立复核历史导出、失败标记与谱差公式，并收紧结构匹配容差。
- [test_lg_structure_diagnosis.py](../../tests/pipeline/test_lg_structure_diagnosis.py)：五项科学/状态恢复检查；
  [check-static.sh](../../tools/checks/check-static.sh) 注册该测试。既有其他改动均保留。
- [最小对照协议](../design/design-2026-10-09-local-structure-attribution.md)：C0/C1/C2 的输入、
  初值、复用条件与结果解释。尚未实现或启动 C0/C1。

## 数据、运行与校验

正式产物：[lg_structure_diagnosis_20261009](../../results/runs/lg_structure_diagnosis_20261009/)。
主要文件为 `report.json`、`manifest.json`、`verification_reproducible.json`、逐材料 CSV、谱数组 NPZ、
逐配对 CSV 与 train/valid 几何表；大数组保留本地，不加入 Git。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/tefscloud/miniconda3/envs/torch/bin/python -B tools/eval/lg_structure_diagnosis.py \
  --out-dir results/runs/lg_structure_diagnosis_20261009

PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/tefscloud/miniconda3/envs/torch/bin/python -B tools/checks/check_lg_structure_diagnosis.py \
  --diagnosis-dir results/runs/lg_structure_diagnosis_20261009 \
  --out results/runs/lg_structure_diagnosis_20261009/verification_reproducible.json
```

复跑需换一个全新输出目录；先加 `--smoke` 可做 64 条/划分的小规模调用链检查。
正式运行 V100、FP32、eval，无参数更新，耗时 **214.99 秒**，PyTorch 峰值分配 **1554.80 MiB**。
几何参考为有效 train **18,704** 条及 valid **2,313** 条；模型训练集前向沿用原诊断的
固定 **2,048** 条材料，按 ID 重新定位，不能当作全训练集指标。

检查了 manifest 与 **22 个 train/valid 数据文件**的哈希；未打开 test 数据数组。
各模型仍使用原 `balanced_oracle_mae` 选点，训练 seed 都为 42：

| 简称 | 运行标签 | checkpoint epoch |
|---|---|---:|
| baseline35 | `_lg_baseline_q1v1_e35_s42` | 34 |
| B+35 | `_lgd_s42g6sel20261008_k2_g6_bp_staged_s42` | 35 |
| baseline100 | `_lg_baseline_q1v1_e100_s42` | 92 |
| B+100 | `_lgd_s42g6e10020261008_k2_g6_bp_staged_s42` | 83 |

四个模型的内存状态与磁盘 checkpoint 起止哈希一致。推理屏蔽后原输出恢复，
逐项满足 `rtol=2e-5, atol=2e-6`；同设置重复前向作为数值噪声参照。
B+35/B+100 恢复最大绝对差为 0.0001907/0.0001869，同设置重复最大差均 0.0002136，
这些最大值来自物理幅度谱，不是单位和为 1 的形状谱。

首次 smoke 使用单一绝对误差阈值 2e-5，因物理谱最大漂移 2.2888e-5 停止；
该阈值没有区分谱尺度。增加同设置重复对照并使用上述相对/绝对容差后，
`lg_structure_diagnosis_20261009_smoke_r1` 在 27.98 秒完成；首次失败目录保留。

独立复核：四组全量 valid 的 ID 与历史导出一致，逐材料指标满足
`rtol=2e-4, atol=2e-5`；从 `(预测-目标)_i-(预测-目标)_j` 独立重算配对误差，与表中
结果相差小于 1e-12。极端负 phDOS R² 对浮点误差敏感，逐点最大绝对差可达 0.5，
所以不能仅用最大绝对差判断不同单位指标是否一致；四组逐材料失败标记也完全一致。
详细逐指标差异保存在 `verification_reproducible.json`。

仓库检查：`PYTHON=/opt/tefscloud/miniconda3/envs/torch/bin/python bash tools/checks/check-static.sh`
通过 **285 项测试**、静态检查及当时 718 个候选文件的仓库检查；补齐文档和独立复核入口后
再检查 720 个候选文件通过。此检查进程显式隐藏 GPU，
真实 GPU 数值与恢复证据来自上述诊断，不能把 CPU 回归表述为 GPU 测试。

## 事实：训练拟合增强，验证收益没有随之出现

以下均为 repaired Q1、单 seed、各自保存的 best。eDOS blind 指标：

| 模型 | train 子集 R² 中位 | valid R² 中位 | train/valid 失败率 % |
|---|---:|---:|---:|
| baseline35 | 0.56680 | 0.47292 | 2.69 / 8.78 |
| B+35 | 0.64601 | 0.46668 | 1.37 / 11.72 |
| baseline100 | 0.71589 | 0.47753 | 1.27 / 13.58 |
| B+100 | 0.81076 | 0.44879 | 0.59 / 15.35 |

valid 的其他任务/模式：

| 模型 | eDOS oracle R² / fail % | phDOS blind R² / fail % | phDOS oracle R² / fail % |
|---|---|---|---|
| baseline35 | 0.50991 / 5.71 | 0.72723 / 4.24 | 0.73521 / 3.59 |
| B+35 | 0.50529 / 7.83 | 0.71442 / 4.71 | 0.72585 / 3.72 |
| baseline100 | 0.52071 / 10.16 | 0.69985 / 6.79 | 0.70878 / 5.58 |
| B+100 | 0.50765 / 11.37 | 0.70713 / 5.58 | 0.71587 / 4.89 |

两套预算的学习率轨迹不同，以上是预算方案的比较，不是同一条训练轨迹的早晚 checkpoint。

## 事实：谱差响应变大，但 valid 配对预测未显示可靠改善

绝对组成相同：train 27 对/27 组，valid **320 对/170 组**；约化组成相同：train
74 对/57 组，valid **360 对/180 组**。两种口径分开统计，不能把 360 对写成历史 320 对。
表中是 eDOS 单位和谱形、约化组成口径：

| 模型 | train 预测 TV / 谱差误差中位 | valid 预测 TV / 谱差误差中位 |
|---|---|---|
| baseline35 | 0.00425 / 0.28596 | 0.00777 / 0.26819 |
| B+35 | 0.03163 / 0.27712 | 0.02648 / 0.26657 |
| baseline100 | 0.01860 / 0.24315 | 0.02500 / 0.27156 |
| B+100 | 0.10263 / 0.20369 | 0.05661 / 0.26424 |

目标 TV 中位 train/valid 为 0.29059/0.26846。B+100 的谱差向量夹角余弦中位，
train **0.524**，valid **0.115**；valid 的典型谱差方向仍预测较弱。

按组成组等权的 `B+ - baseline` 有符号谱差误差差值（负数更好，2,000 次组成组重采样）：

| 预算 | train 差值 [95% 区间] | valid 差值 [95% 区间] |
|---|---|---|
| 35 | −0.01580 [−0.02650, −0.00727] | +0.00147 [−0.00341, +0.00676] |
| 100 | −0.03105 [−0.04782, −0.01843] | −0.00241 [−0.00865, +0.00403] |

绝对组成口径的 valid 区间同样包含零。phDOS 的 valid 配对误差比较两种预算也都未分辨出
优势。区间含零不是等价证明；这些是材料/组成组的变动范围，不是跨训练 seed 稳定性。

## 事实：局部分支影响预测，没有整体抹平结构差异的迹象

B+ 的同组成 valid 特征汇总相对距离（token 均值/逐通道 RMS 的有损汇总）：

| 阶段 | B+35 | B+100 |
|---|---:|---:|
| 元素入口 | 3.4e-8 | 3.1e-8 |
| 第一局部残差后 | 0.0471 | 0.0712 |
| 第二局部残差后 | 0.0736 | 0.1202 |
| 第六全局层后 | 0.1233 | 0.2085 |
| eDOS decoder 后 | 0.1108 | 0.1857 |

可以看到结构差异进入了局部状态，并在后续汇总中仍然可见；不能据此证明所有信息保真，
也不能由不同阶段距离大小直接判断哪个模块更有用。

固定标签无关的 **256 条 valid 子集**，eDOS oracle R²：

| 推理状态 | B+35 | B+100 |
|---|---:|---:|
| 原模型 | 0.46734 | 0.49169 |
| 局部残差乘 0.9 | 0.46157 | 0.47868 |
| 仅 RBF 输入置零 | 0.46548 | 0.48813 |
| 关闭高阶角度输入 | 0.46113 | 0.48532 |
| 屏蔽全部角度路径 | 0.42400 | 0.35685 |
| 关闭完整局部残差 | 0.08454 | 0.05199 |

此结果降低了「局部/角度完全没有进入预测」的解释优先级。屏蔽会改变训练后的内部状态分布，
整块局部残差还含非几何变换，不能把分数下降当作几何的净收益，更不能拿它代替 C0/C1 重训。

## 事实：结构支持、截断与标签仍有未解决的问题

- valid 仅 **345/2,313** 条在 train 中有同约化组成材料。没有同组成参考不等于没有见过
  相似局部环境，也不等于新元素或新化学体系。
- 有同组成参考的材料中，按 train 径向支持距离四分位划分，valid 近组 66 条、远组 100 条。
  B+100 的 blind eDOS R² 为 **0.71555 / 0.29566**，失败率 **4.55% / 30.00%**；
  baseline100 为 0.69340/0.30915、3.03%/24.00%。这是相关性和小分组描述，未证明覆盖差是因果根因。
- 5–6 Å 壳层在 valid 典型结构中占 **43.20%** 邻边记录，但只占 **2.01%** cutoff 权重；
  涉及外壳的邻边对占总 pair-cutoff 权重 **4.17%**。这些是未乘学习到的消息/门控的几何权重，
  不是实际消息贡献或 DOS 物理重要性，尚不足以决定增大半径。
- 宽松结构匹配得到 train 6 对、valid 45 对。独立收紧到 `ltol=.01, stol=.05,
  angle_tol=.5, scale=False` 后仍有 **train 1 对、valid 12 对**。例如 train AgBiS2 一对，
  归一化结构匹配 RMS=0.000274、单原子体积比 1.00058，eDOS 目标 TV=0.19963；
  valid CaSO4 一对 RMS=0.000420、体积比 0.99863，eDOS TV=0.15572。
  严格匹配也允许实际几何差异，缓存缺少完整计算条件；未证实标签错误，更未估计数据精度上限。
  ID、全部匹配与容差见 `verification_reproducible.json`。

## 结论与交接

状态：**第一阶段完成；局部净贡献归因 pending**。

1. **证据最强：泛化不足。**新模型能更好拟合训练谱与训练材料间谱差，valid 没有可靠的
   配对改善。100 轮扩大训练/验证差距，优先级高于继续增大宽度、深度或预算。
2. **解释优先级降低：局部没有响应、角度完全没被使用、全局层把结构差异整体抹平。**
   表示与推理敏感性均不支持这些笼统说法，但不排除特定信息压缩或读出瓶颈。
3. **尚需区分：复杂局部方案的归纳偏置/容量、训练结构覆盖、监督与读出，以及部分标签条件。**
   径向覆盖与相近结构证据提供线索，没有单独证明根因。

下一步仍是按 [协议](../design/design-2026-10-09-local-structure-attribution.md)补新骨架 C0
无几何与 C1 简单径向对照，严格满足条件时复用 C2。暂不把扩大半径、增加角度阶数、增加深度
或训练轮次排在前面；也不因推理屏蔽结果就恢复原几何打分旁路或改变 E(3) 要求。
