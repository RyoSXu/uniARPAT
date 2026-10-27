# 日志：2026-10-27 — 谱形辅助第25轮的冻结逐阶段读出

对应[事先规则](../design/design-2026-10-27-shape-aux-stage-readout.md)。只读诊断，不训练、不改损失、不改选点、不读 test。

## 范围与身份

- 对象是谱形辅助 best，epoch 25，目录
  `results/runs/ablation_m1_sbal_shape20261025_shape_a64_s42_e35`。
  契约 `joint-spectral-balance-shape-model-v1`。checkpoint SHA-256 前缀 `a27fffbadc2f3569`。
- 参数保持 eval、FP32、冻结。抽取前后状态哈希一致，checkpoint 文件哈希不变。
  钩子在抽取后摘除；若仍留着或状态改变，脚本会中止。
- 绝对组成。配对在完整划分上生成，表示只取参与配对的材料。
  train 2,591 对 / 1,331 组 / 3,115 材料；valid 320 对 / 170 组 / 394 材料。
  配对数和组数与 2026-10-11 的绝对组成表一致。当时 valid 绝对行写的 420 个材料，
  等于约化组成的参与材料数，也是两种口径的并集；绝对组成单独是 394。
  多出的 26 条只在约化口径里成对，本次不使用。
- train 的 eDOS 有 1 对目标 TV 低于 1e-6，已从探针主指标剔除，仍留在距离统计里。
  其余划分的近零对为 0。
- 重算的 394 条 valid 谱形与已归档导出比对：eDOS 预测最大绝对差 1.79e-7，
  phDOS 为 7.45e-7，两侧目标差为 0。
- 不打开 test 数组。manifest 哈希与 checkpoint 记录一致，train/valid 载荷按清单复算。

命令：

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 python3 -B tools/eval/shape_aux_stage_readout.py \
  --out-dir results/runs/shape_aux_stage_readout_20261027
```

退出码 0。墙钟 467.6 秒。Tesla V100 上 PyTorch 峰值分配 1,340.9 MiB。
短测目录 `shape_aux_stage_readout_20261027_smoke` 只检查管线，不参与判读。

## 表示距离

相对 L2 用送进读出的向量。分界 1e-3 按设计在开跑前固定。
下表是 valid、全部绝对组成对、组成组等权。局部和全局两任务共用同一套原子状态，数字相同。

| 层 | 配对中位 | 组均 [95% 区间] | 分类 |
|---|---:|---:|---|
| 局部末层 | 0.0703 | 0.1042 [0.0900, 0.1188] | 分开 |
| 全局末层 | 0.0860 | 0.1347 [0.1143, 0.1571] | 分开 |
| eDOS decoder | 0.1073 | 0.1911 [0.1624, 0.2219] | 分开 |
| phDOS decoder | 0.1337 | 0.2120 [0.1812, 0.2441] | 分开 |

从局部到全局再到 decoder，组均距离变大，没有落到 1e-3 以下。
train 的组均同样远高于分界：局部 0.1030、全局 0.1308、eDOS decoder 0.1807、phDOS decoder 0.2041。

## 线性探针与模型输出

无截距线性读出只在 train 上选正则，再锁定评 valid。「优于零响应」指误差差的区间上界小于 0。
原子层和 phDOS decoder 的选定正则都在网格边缘 1e22，预测谱差 TV 中位约为 1e-20，
与严格零响应重合。eDOS decoder 选定 1e6，不在边缘。

valid 组均谱差误差（零响应 / 该层探针 / 模型自己的谱形）：

| 任务 | 零响应 | 局部 / 全局探针 | decoder 探针相对零响应 | 模型相对零响应 | 预测 TV 中位 / 目标 TV 中位 |
|---|---:|---:|---:|---:|---:|
| eDOS | 0.27483 | 与零响应相同 | +0.000033 [−0.000016, +0.000075] | +0.002091 [−0.001792, +0.006140] | 0.02020 / 0.26975 |
| phDOS | 0.30596 | 与零响应相同 | 与零响应相同 | −0.002548 [−0.010155, +0.005701] | 0.03321 / 0.28185 |

valid 上三层探针和模型输出都没有优于零响应。eDOS 模型的点估计略差于零响应，phDOS 略好，两个区间都含零。
模型预测的谱差幅度大约是目标的 7.5%（eDOS）和 11.8%（phDOS）。

train 上模型自己的输出优于零响应：eDOS −0.007464 [−0.009080, −0.005983]，
phDOS −0.031810 [−0.036304, −0.027592]。
eDOS decoder 探针在 train 上也优于零响应，但只低 0.000031 [−0.000051, −0.000012]，
相对 0.270 的零响应误差大约万分之一；valid 没有保持。
phDOS decoder 探针在 train 上已被收缩成零响应，而模型输出本身明显优于零响应。
因此这条线性探针读不出，不能写成该层没有训练信号。

## 判读

两条任务都走到事先规则的第 5 条：三层距离都分开，valid 上三层线性探针都没有优于零响应。
动作是停住，不把结果指定成局部消息、全局混合或 decoder/读出的瓶颈。

这只覆盖单位和谱形差、绝对组成、这一份 epoch 25 checkpoint，以及这一类无截距线性读出。
原子层是 mean/RMS 池化。单种子的组重采样区间不代表换种子仍如此。
物理谱里的总量差不在本次目标里。

## 证据

`results/runs/shape_aux_stage_readout_20261027/report.json`，以及同目录的配对表、特征和谱形数组。
这些本地产物不进入 Git。

单元测试 `tests.pipeline.test_shape_aux_stage_readout` 覆盖分界、距离对称、六条判读顺序和双 decoder 钩子。
`tools/checks/check-static.sh` 跑了 547 项：1 项错误是既有测试找不到 `data/train4ARPAT/train/train_index.npy`，与这次读出无关。
