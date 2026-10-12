# 2026-10-12 — C1 小样本结构谱差拟合对照

## 范围与协议

前一轮冻结表示诊断没有发现可迁移的线性谱差读出，但不能证明完整表示缺信息。
本轮检验：已有 C1 网络能否在少量已见材料上拟合结构谱差，直接监督谱差是否帮助训练。
用户授权分钟级诊断直接执行。本轮更新与探针合计 409.10 秒，峰值分配约 3,901 MiB。
只使用修复版 `crystal_q1_v1` 的 train、seed 42；没有 valid/test 评估。
任务指标是 **SumNorm 单位和谱形**误差，不是 blind/oracle 物理尺度指标。
原模型、数据划分、正式契约和原 checkpoint 均未修改，没有 Git 提交。

- 起点：C1 已完成训练的 MAE best ep34，目录
  `results/runs/ablation_m1_lsc_attrib20261009_k2_g6_c1_staged_s42_e35/`。
  两组复制相同权重，**这不是从头训练试验**。
- 从上一轮完整 train 绝对组成配对表取候选，限制两端各不超过 40 原子，不裁剪结构。
  2,568 对／1,322 组符合限制；按 seed 42 均匀抽 16 个组成组，每组随机取一对，
  抽样不用标签或预测。最终 32 材料、420 原子、最大 28 原子。
- 同一完整批次，各更新 600 步，FP32、dropout 0.05，全部参数可训练。
  两组新建 AdamW，betas=(0.9,0.99)、weight decay=0.01、固定 lr=5e-5，
  不沿用旧 optimizer 或调度状态。每 100 步评价，固定比较第 600 步。
- control：原完整训练损失；额外计算零权重谱差损失，使前向次数和 dropout 随机数消耗与 candidate 相同。
- candidate：原损失加既有 `edos_pair_contrast_loss`。在锁定 train 批次的 eval 模式下校准一次，
  使加权谱差梯度范数等于 eDOS 基础损失梯度范数；λ=3.67738，原始范数 0.50876／0.13835。
- 每组 8 步成本探针的权重丢弃，正式诊断重新复制共同权重与空 optimizer。

谱差误差 `E = 0.5*sum(abs((p_i-p_j)-(y_i-y_j)))`，严格零响应参照
`E0 = 0.5*sum(abs(y_i-y_j))`，skill=`1-E/E0`。每组只有一对，组成组等权与配对等权一致。
近零目标差阈值 1e-6，本批 16 对均非近零。预先固定充分拟合判据：平均 E≤平均 E0 的 20%，
且至少 75% 配对误差减半。

## 结果

| train／固定材料指标 | 起点 | 原损失，600 步 | 加谱差监督，600 步 |
| --- | ---: | ---: | ---: |
| 组均谱差误差 E | 0.22939 | 0.08265 | 0.05080 |
| E／零响应误差，E0=0.23239 | 0.98707 | 0.35566 | 0.21862 |
| 配对 skill 中位数 | -0.00011 | 0.68119 | 0.85386 |
| 配对方向余弦均值 | 0.00479 | 0.84363 | 0.92483 |
| 误差减半配对 | 0/16 | 11/16 | 14/16 |
| 误差减少至少 80%配对 | 0/16 | 5/16 | 11/16 |
| 单材料 eDOS 谱形 TV 误差 | 0.19443 | 0.05125 | 0.08758 |
| 单材料 phDOS 谱形 TV 误差 | 0.15693 | 0.03605 | 0.03519 |
| 预设充分拟合判据 | 未过 | 未过 | 未过 |

candidate 的配对误差比 control 低 38.53%，15/16 对更好。
组成组 bootstrap 的 candidate−control 均值为 −0.03185，区间 [-0.05167,-0.01483]；
仅描述这批 train 组，不能作泛化或跨训练种子推断。单材料 eDOS 误差更高，不能据此采用新目标。

两个 candidate 配对的 skill 仍不足 0.03：单原子 Al 的 `mp-aaaaaafe / mp-aaacevps`，
以及 `mp-aaabipws / mp-aaabjbnl`。复用已有 train 几何表的 16 维 pooled 径向代理距离，
分别为 0.86054／0.01165，不能统一解释为输入太相似。有损代理不证明完整 C1 输入相同或标签错误。

## 判读与下一步

事实支持：本批一部分训练结构谱差可拟合，显式谱差监督加强了这种拟合。
不支持“网络完全不能利用结构”的笼统解释。两组仍未过整体预设门槛，不能降低门槛后宣称通过。
尚未证明：全部选中结构差或整个 train 可充分拟合、从头训练同样有效、valid 泛化改善、
新架构超过旧 baseline、监督是唯一原因、表达能力或优化问题已被排除。
起点、样本规模、原子数限制、固定学习率、预算和损失权重都限定结论。

优先分钟级检查两个困难材料对：输入可区分性、局部／全局／decoder 的完整表示差异与梯度，
以及可得的标签来源和计算条件；缺元数据时登记未知项，不直接判错。
若随后决定正式检验谱差监督的泛化，保持原 train/valid 做同预算 control/candidate；
两组 35 轮粗估 4–6 GPU 小时，依据历史 C1 成本和新增前向量，需重新预检成本后交给 OpenCode。

## 实现与复核

- `tools/eval/local_structure_small_fit.py`：`select_pairs` 锁定选样；`new_model` 构造临时副本；
  `evaluate`／`fit_metrics` 复用已有谱差指标；`run` 管理成本、训练、RNG、指纹；
  `check_initial_metrics` 区分浮点容差与精确一致的离散项。
- 调用链：诊断入口 → `ConfigBuilder`／`crystal_collate` → `basemodel.train_one_step`／
  既有 `_calibrate_edos_pair_weight` → 原 C1 与既有谱差损失。
- 新增 `tests/pipeline/test_local_structure_small_fit.py` 的 6 项测试并接入 `check-static.sh`。
- 两组起始与结束 RNG 摘要严格一致。原 checkpoint／配置、train 数据和 core code 指纹一致。
  训练完成后起点指标逐值相等检查触发退出：谱差均值差 7.40e-9，余弦差 4.07e-6。
  修正为标量 1e-6、余弦 1e-5 的 FP32 容差，权重／RNG／离散项仍精确一致。
  未重跑训练或改变结果，保留原执行脚本及未复核报告。
- 从原始 train 原子与谱独立核对组成／归一化，重算每个保存点的指标与判据，首次 230 项复核通过。
  完整 `check-static.sh` 通过：318 项测试、ruff、语法及仓库文件检查；CPU 检查使用隐藏 CUDA 的环境。

证据目录 `results/runs/local_structure_small_fit_c1_s42_20261012/`：
`report.json`、选样 CSV、两组 trace、逐保存点预测／配对、独立核查报告与可重放脚本。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/tefscloud/miniconda3/envs/torch/bin/python -B tools/eval/local_structure_small_fit.py \
  --out results/runs/local_structure_small_fit_c1_s42_20261012_replay \
  --steps 600 --eval-every 100 --max-wall-s 900
```

状态：短诊断已完成，充分拟合未全面展示，泛化与根因仍待检验。
