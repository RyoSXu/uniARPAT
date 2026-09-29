# H1 gamma 两参数部署校准门

日期：2026-09-28

结论：**park。**全局两参数仿射 logit 校准没有把冻结 B7 的 gamma 上界转化为 Q1 valid 精度收益，
不接入生产推理，不扫描映射、目标、正则或初始化，也不自动升级到新 H1 结构。

## 决策与隔离

本门只检验一个因素：冻结 B7 `_e9ctl` epoch 33 的模型、eDOS shape 与所有权重，只把 H1 的
`gamma_pred` 替换为：

`sigmoid(a * logit(clamp(gamma_pred, 1e-6, 1-1e-6)) + b)`。

`a,b` 从 Q1 train 18,706 条样本一次性拟合，固定 soft-target BCE、初始化 `(1,0)`；随后只在
Q1 valid 2,313 条样本裁决。工具没有 split 选择器，没有加载 Q1 test，也没有 M1 训练、checkpoint
写入或模型参数更新。checkpoint 身份为 M1、epoch 33、seed 42，SHA-256 为
`cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40`。

## 实现与运行

- 工具：`tools/eval/h1_gamma_calibration_gate.py`
- 合同测试：`tests/test_h1_gamma_calibration_gate.py`
- 命令：`python3 tools/eval/h1_gamma_calibration_gate.py --device cuda`
- 汇总：`results/h1_gamma_calibration_q1.json`
- valid 逐样本证据：`results/h1_gamma_calibration_q1_valid_samples.csv`

工具固定检查 B7 路径、checkpoint SHA、配置关键字段和 train/valid 样本数；拒绝覆盖既有结果。首轮
调用在读取模型或数据前因脚本入口没有加入仓库根路径而失败，没有生成结果。只修复 `sys.path`
启动合同后，原命令完成唯一一次科学运行。

新增符号及职责：

- `apply_gamma_calibration`：实现预注册的单调两参数映射；
- `fit_gamma_calibration`：用一次确定性 L-BFGS-B 拟合 `a,b`；
- `paired_bootstrap_median_interval`：计算逐样本配对中位差区间；
- `adjudicate_gate`：按预注册主门、失败率门、机制门和不变量门裁决；
- `run_gate`：验证身份，依次收集 train、拟合、收集 valid 并形成结果口径；
- `_collect_split`：复用项目统一的 SumNorm 物理重建和 `per_sample_spectral_metrics` R²。

## 结果

train 拟合得到：

- `a = 1.01135761`
- `b = −0.00009453`
- soft BCE：`0.60950536 → 0.60949813`
- gamma 绝对对数误差中位数：`0.07765259 → 0.07758133`

该映射近似恒等，只让 valid gamma 最大绝对变化达到 `0.002515`。Q1 valid 结果为：

| 指标 | control | candidate | candidate−control |
|---|---:|---:|---:|
| blind eDOS 中位 R² | 0.47313863 | 0.47264230 | **−0.00049633** |
| blind eDOS 失败率 | 9.12235% | 8.99265% | −0.12970pt |
| gamma 绝对对数误差中位数 | 0.08975458 | 0.08922359 | −0.00053099 |

- blind eDOS 中位 R²配对 bootstrap 95% 区间：`[−0.002093, +0.001378]`；未达到 `+0.02`。
- gamma 绝对对数误差中位差 95% 区间：`[−0.001108, +0.000966]`；上界不小于零。
- 失败率保护门通过，但主门和机制门失败。
- oracle eDOS 中位 R²/失败率为 `0.51498282/6.26891%`，control blind 为
  `0.47313863/9.12235%`，逐位复现收益上界审计的冻结 B7 口径。
- 模型 state SHA-256 前后均为
  `766636c906663ad9b256d18d59f19c4c5fc6b7a4668f11f919f1b8756aaac6b8`；eDOS shape 共用，phDOS
  未进入 candidate 变换，不变量门通过。

## 结论边界

该结果排除的是“冻结 B7 gamma 的全局收缩／偏置可由一个 train 拟合的单调仿射 logit 映射转化为
全体 valid `+0.02` 收益”。它不否定 gamma oracle 上界，也不证明 H1 memory、监督目标或结构条件
尺度头没有改进空间。按预注册停止条件，不以本结果为理由追加 isotonic、spline、分组映射、loss
扫描或新头训练；任何新 H1 方案必须重新提出独立机制、单一因素和 valid 检验。

## 验证

- `python3 -m unittest tests.test_h1_gamma_calibration_gate`：5 项通过；
- `ruff check tools/eval/h1_gamma_calibration_gate.py tests/test_h1_gamma_calibration_gate.py`：通过；
- `python3 -m py_compile ...` 与 `git diff --check`：通过；
- 结果表 2,313 行、mpid 唯一、数值列全为有限值；control 指标复现既有冻结 B7 valid 结果。
- `python3 -m unittest discover tests`：275 项通过，耗时 108.425 秒。
