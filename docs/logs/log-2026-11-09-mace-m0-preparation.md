# 日志：2026-11-09 — MACE 归因 M0 的实施与预检

对应[设计](../design/design-2026-11-09-mace-m0.md)。本轮只生成特征、写工具、测试和短时预检，**未启动正式训练**，未读 test。

## 特征

- `tools/data/build_element_mean_features.py`（训练环境，仅 numpy）从 M1 特征生成 `data/external/mace_mp0b3_medium_elemmean_v1/`（本地），用时数秒。
- train 85 种元素；未见元素原子数 train/valid/test = 0/0/1。无近常数通道，train 标准差 0.019–1.22。
- 元素均值对 MACE 原始特征的解释方差约 75%（各通道合计）。

## 实施

| 位置 | 内容 |
|---|---|
| `tools/data/build_element_mean_features.py` | 每元素 train 均值查表，manifest 记录来源特征指纹与元素计数 |
| `tools/eval/mace_m0_prepare.py` | 冻结 h2 与 M1 的配置、checkpoint、导出、train 推理与分析报告；只改特征目录与 protocol 路径 |
| `tools/eval/mace_m0_analyze.py` | 事先固定的主判读（d0 = M0 − h2）与 eDOS 配对读数，另报相对 M1、同组成技能、train/valid 比 |
| `tests/pipeline/test_element_mean_features.py` | 查表、未见元素回退、指纹与判读阈值 |

模型与训练代码无改动；预检复用 `tools/checks/check_mace_features.py`。

## 冻结准备

`results/runs/mace_m0_20261109/report.json`（本地），tag `_sbal_macem020261109_m0_s42_e35`。

## 验证

- 新增 3 项与原 6 项特征测试通过；ruff 通过。
- 分析脚本空跑：以 M1 导出代替 M0，得到“元素主导”、相对 M1 差为 0、占 M1 收益比例 1.0，符合预期（产物在临时目录，未保留）。
- GPU 预检 `results/runs/mace_m0_20261109/preflight/report.json`：复现 h2 初值；除投影外张量与 h2 相同；初始输出与 h2 最大差 3.3e−6，h2 重复前向噪声 3.6e−6；
  投影梯度非零；约 277 秒每轮训练，峰值 12.6 GiB；lr=0 目标与评估之差 5.6e−8。
- 成本：约 4.5 GPU 小时（同 M1）。
