# 六层全局、无局部更新的整体 Encoder 归因对照

创建日期：2026-10-12。用户确认先整理仓库，再由 OpenCode 完成正式训练与监测。
当前目标是解释局部—全局方案相对原 Encoder 的整体性能差距；本轮只新增一组 G0，
不继续围绕困难材料对，也不同时搜索插入顺序、损失或训练超参数。

## 问题与已知混杂

原 baseline 已通过 `RPEncoding/rp_proj` 几何打分利用结构。新 Encoder 同时移除该路径、
增加可学习 Q/K/V 投影，并把更新后归一化改为更新前归一化。
已有 baseline 比较检验的是整套 Encoder 替换，不能把差距直接归因于局部更新。
已有 C0 也不是恒等映射：它保留两次局部调用及每次两个非线性内部块。

因此先固定新全局骨架，彻底关闭局部更新，作为此前 C0/C1/C2 的缺失参照。

| 组别 | 局部调用 | 全局层数 | 与 G0 的比较含义 |
|---|---:|---:|---|
| G0（新增） | 0 | 6 | 同一新骨架，无局部分支，也不使用几何 |
| C0（复用） | 2，无几何 | 6 | 加入当前逐原子非线性局部分支的净效应 |
| C1（复用） | 2，简单径向 | 6 | 加入当前径向局部更新的净效应；与 C0 的差异是条件于分支的几何增益 |
| C2（复用） | 2，B+ | 6 | 加入当前 B+ 局部更新的净效应 |
| 原 baseline（复用） | 不适用 | 原六层 | 原 Encoder 与新 G0 的整套差异，包含几何打分和注意力/归一化改动 |

C0/C1/C2 的顺序固定为 `L1 L2 G1 G2 G3 G4 G5 G6`，G0 为 `G1 G2 G3 G4 G5 G6`。
删除局部分支同时减少对应参数，这是该分支净效应的一部分，不宣称容量相同。

## 固定协议与初值

- 修复版 `data/crystal_q1_v1`、M1/ZP、SumNorm/H1、E0/P0、eDOS/phDOS 128/64 bin、
  Q1 坐标输出头隐藏宽度 128、六层共享 decoder。
- seed **42**、**35 轮**、batch **32**、lr **5e-5**、FP32；AdamW、学习率轨迹、dropout、
  数据顺序与 C0/C1/C2 原配方一致；仅 train/valid，`balanced_oracle_mae` 选点。
- `local_route=G0, local_depth=0, global_depth=6, local_layout=staged`；不创建局部模块或参数，
  不构建周期邻居记录，不创建几何缓存。其他路线的深度约束和计算行为保持。
- `global-only-init-v1` 从未训练 B+ L2/G6 参考复制**所有剩余张量**，
  即删除 `encoder.local_layers.*` 后的全局层、元素入口、decoder 和 head；逐张量核对并保存摘要。
  不复制已训练 checkpoint，不把“同种子”当作共同初值已经相同。
- 使用独立 `global-only-model-v1` checkpoint 契约。恢复同一组时保留 optimizer/RNG、原始
  配方与训练起始证据；已经完成 35 轮则直接复用，不重训或追加 epoch。

已有参考 `results/runs/ablation_m1_lgd_s42g6sel20261008_k2_g6_bp_staged_s42`；
C0/C1 路径与选点身份来自 `results/runs/local_structure_attribution_20261009/analysis/report.json`；
原 baseline 使用修复版同预算 `results/runs/ablation_m1_lg_baseline_q1v1_e35_s42`，
历史旧 Q1 的 ZP100/B7 不作为此次严格对照。

## 验证与执行入口

调用链：`global_only_compare` → `global_only_config` → `train_and_eval` →
`initialize_comparison_model` → `initialize_global_only_model` → `LocalGlobalEncoder(G0)`。

| 文件 | 新增/修改职责 |
|---|---|
| `model/local_global_encoder.py` | G0 显式零局部调用；仍复用 `ScalarGlobalLayer`，禁止几何输入 |
| `model/transformer.py` | G0 原子态直达全局骨架，跳过记录构建及几何缓存 |
| `utils/global_only_experiment.py` | 固定配置、参考初值复制、独立契约 |
| `utils/local_global_experiment.py`、`tools/train.py` | G0 校验/初始化/契约分派、命令行及运行记录 |
| `tools/checks/check_global_only.py` | 真实数据、共同初值、有限梯度、E(3)、恢复与固定批次成本预检 |
| `tools/eval/global_only_compare.py` | 默认只打印一条命令；显式训练/汇总/配对分析，复用原四组结果 |
| `tests/pipeline/test_global_only.py` | 真正无局部执行、权重身份、输入与恢复边界、命令组装和报告参照验证 |

真实 GPU 预检仅更新一次性模型，检查两批 train 与一批 valid、FP32/FP64 的六种 E(3)
变换、内存 checkpoint 往返，以及两批完整 optimizer 步骤成本。正式组始终从未训练初值开始。
训练前预检记录源码/数据/参考身份；入口拒绝预检后发生的源码或参考漂移。

## 判读与成本边界

按全量 valid 的材料 ID 配对，分别报告两类谱 × blind/oracle 的中位 R²、中位 MAE、
R²<0 失败率、best epoch、耗时和显存；报告差值方向固定为**各已有组 − G0**。
材料配对 bootstrap 95% 区间条件于本次 seed 和所选 checkpoint，不代替跨种子稳定性。

- 若局部组低于 G0，支持该模块在**当前插入位置与预算**下有净代价；再固定模块测试位置，
  不能立即判定其全部编排无效或核心公式错误。
- 若局部组高于 G0，但仍低于原 baseline，支持几何局部更新有收益，但不足以补回整套
  Encoder 替换的差距；不能把剩余差距单独归因于全局层或旧几何打分。
- 若任务或指标方向不一致，分别报告；不重新更换选点规则来制造胜出，不自动启动下一组。

成本初步参考 C0 每轮 train 中位 120 秒、35 轮 train 总计 1.17 小时；G0 预计同量级。
计入验证、保存及分析，预留 **1–2 小时**。这是预算估计，正式训练开始后用前几轮实测更新，
不自动调整 batch、lr 或轮数。异常停止并保留现场，不自动重试。
