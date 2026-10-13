# 旧几何打分、全局骨架与局部更新的归因对照

创建日期：2026-10-12。用户已确认实施 O0/O1 入口与短测，正式训练由 OpenCode 执行。
本轮解释整体 Encoder 的性能差距；不同时搜索局部公式、位置、深度或训练超参数。

## 问题与范围

[G0 结果](../logs/log-2026-10-12-global-only-run.md)支持当前新骨架内 C1/B+ 几何局部更新的总体收益，
但不能解释与原 baseline 的综合差距。G0 与原 baseline 同时相差几何打分、注意力参数化、
归一化位置及 Encoder 内部初值；不能据此认定局部模块导致退化，或把剩余差距唯一归因于其他改动。

只新增两组，与已有 G0/C1 组成下表，并复用原 baseline 作为几何打分参照：

| 全局更新 | 无局部更新 | 当前径向 C1，前段两次调用 |
|---|---|---|
| 原六层 `q=k=v=h`、post-norm，关闭几何打分 | O0（新增） | O1（新增） |
| 新六层可学习 Q/K/V/O、pre-norm | G0（复用） | C1（复用） |

O1 的顺序为 `L1 L2 G1..G6`；每次局部调用含两个内部块。O0 为 `G1..G6`，无局部参数。
两组复用原 `TransformerEncoderLayer` 内容注意力、FFN 和 post-norm，明确关闭几何打分，
不创建 `rp_proj/RPEncoding`。O1 的局部包装与消息均复用现有 C1，不增加新的局部公式。
全局层只读取反射偶不变标量与原子 mask，结构只经 C1 的距离消息进入，保持 E(3) 要求。
原 baseline 的方向打分只用于历史参照，不重新启用该路径作为本轮候选。

## 初值与固定协议

- 修复版 `crystal_q1_v1`、M1/ZP、SumNorm/H1、E0/P0、Q1、128/64 bin、六层共享 decoder。
- seed **42**、**35 轮**、batch **32**、lr **5e-5**、FP32、既有 AdamW 与学习率轨迹；
  仅 train/valid，`balanced_oracle_mae` 选点，损失、dropout 和数据顺序沿用既有配方。
- `legacy-backbone-init-v1` 重建**未训练**原 baseline：保留其所有剩余张量，
  原 `encoder.layers.i.*` 映射到 `encoder.global_layers.i.layer.*`，删除几何编码与打分张量。
  原 baseline 的构造和初始化摘要必须逐项复现已完成 run 的记录。
- O1 的全部 `encoder.local_layers.*` 从未训练 C1 参考逐值复制，C1 全模型初值摘要必须复现
  已有 C1 记录。O0/O1 的其余张量逐值相同，下游与既有新骨架对照共享初值。
- 初始化保存并恢复调用方 RNG；正式训练入口重新设 seed 的既有行为保持，记录真实开始/恢复 RNG。
  不导入已训练权重作为新组初值。
- 使用独立 `legacy-backbone-model-v1` 契约，拒绝混用 O0/O1、新骨架或原 baseline checkpoint。
  同组中断恢复仍恢复权重、optimizer/RNG 与原学习率轨迹，已完成组不追加训练。

**旧/新骨架内部的参数化、容量和初值不完全匹配。**首轮只归因到骨架整体包，
不能从 O0−G0 或 O1−C1 唯一认定 LayerNorm 或 Q/K/V/O 的作用。固定 seed 的局部加入效应
也包含额外容量和训练随机轨迹变化；配对区间不代替跨种子稳定性。

## 判断方式

全量 valid 2,313 个材料按 ID 配对，各自沿用同一 MAE 选点。对 eDOS/phDOS × blind/oracle
报告中位 R²、中位 MAE、R²<0 失败率，材料配对 bootstrap 2,000 次，seed 20261012。

| 差值 | 所检验的解释 |
|---|---|
| 原 baseline − O0 | 保留原骨架时，移除整条旧几何打分路径的作用 |
| O1 − O0 | C1 加入旧内容骨架的净效应 |
| C1 − G0 | 同一局部模块在新骨架内的既有净效应 |
| O0 − G0；O1 − C1 | 无局部/有局部条件下的旧、新骨架整体差异 |
| `(O1−O0)−(C1−G0)` | C1 收益是否随骨架变化；共享同一次材料重采样 |

同时导出预测与目标的归一化形状谱，沿用绝对/约化组成的有符号谱差误差：先求组成组内平均误差，
再按组成组配对重采样，多个共享材料的配对不当作独立样本。主报告将总体样本指标和
同组成结构响应分别列出；不因总体 R² 上升就认定结构谱差泛化已经改善。

- 若原 baseline−O0 有可分辨差异，支持旧几何路径有条件的贡献；不证明其方向参数化符合 E(3)。
- 若 O0/G0 和 O1/C1 显示骨架差异，后续再拆归一化位置与注意力投影，不能立即认定单一根因。
- 若交互项可分辨，支持局部更新收益依赖所测骨架；不证明现有局部公式最优或插入位置无关。
- 区间含零表示该协议未分辨差异，不证明等价；任务/指标方向不一致时分别报告。

## 实施、成本与交接

调用链：`legacy_backbone_compare` → `legacy_backbone_configs` → 原 `tools/train.py` →
新初始化/契约分派 → `LocalGlobalEncoder(O0/O1)` → `LegacyContentGlobalLayer` →
原 `TransformerEncoderLayer(geometry_scoring=False)` → 既有 decoder/head。

主要新增文件：`utils/legacy_backbone_experiment.py`、`tools/checks/check_legacy_backbone.py`、
`tools/eval/legacy_backbone_compare.py`、`tests/pipeline/test_legacy_backbone.py`。
逐样本导出工具增加可选谱文件出口，既有调用保持原行为。

默认入口只打印两条训练命令；`--run` 才执行 O0/O1，`--summarize` 只读历史，
`--analyze` 只读完整模型。正式执行需成功预检绑定配置、源码、train/valid 数据与参考文件。
异常停止并保留现场，不自动重试；通过队列状态、进程、history 与 checkpoint 联合监测。

预算先预留单卡 **3–4 小时**完成两组，实际成本由真实批次短测和前几轮测量修正，
不自动改变配方。短测只更新随后丢弃的临时模型，不写正式 checkpoint；预检与实测日志另存。
本轮实施与验证见[日志](../logs/log-2026-10-12-legacy-backbone-preflight.md)。
