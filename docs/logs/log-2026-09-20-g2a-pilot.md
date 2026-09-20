# 工作日志 2026-09-20：G2a 周期多镜像边条件消息成对 Pilot

## 范围

- **任务与假设：**固定截断半径 $R=5.5\text{ \AA}$，保守保留所有周期镜像多重边 $(i, j, T)$，并用边径向距离 RBF 调制被聚合的 Value 残差消息，检验是否能补充 eDOS/phDOS 所需的局域配位与环境信息。
- **实验因素：**单一变量 `--use_g2`（6 层独立的 `PeriodicEdgeMessage` 残差，新增 3,351,558 参数，约 +4.71%）。
- **执行配置：**Q1 清洁池（18,706 / 2,313 / 2,287）、M1 架构、E0/P0 网格、SumNorm KL/W1/Huber 损失、H1 eta/gamma 尺度监督头、dropout 0.05、batch size 32、seed 42、10 epochs。
- **预注册判据：**
  - win：至少一个任务的 Oracle 测试中位 $R^2$ 相对对照提升 $\ge 0.02$；任一任务失败率不得恶化 $\ge 1.0$ 个百分点，另一任务中位 $R^2$ 不得低于对照 $0.02$ 以上。满足时方准入 35-epoch 确认。
  - park：处于平局线内（$|\Delta\text{med}| < 0.02$ 且 $|\Delta\text{fail}| < 1.0\text{ pt}$）或无正向跨线结果。

## 证据

- **执行命令：**
  ```bash
  setsid nohup python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _g2ctl > output/g2ctl.log 2>&1 &
  setsid nohup python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _g2edge --use_g2 > output/g2edge.log 2>&1 &
  python3 tools/eval/g2_pilot_verdict.py --ctl_tag _g2ctl --exp_tag _g2edge
  ```
- **关键产物：**
  - 控制臂：`results/history_m1_g2ctl.csv`、`test_m1_g2ctl_summary.csv`、`samples_m1_g2ctl_test.csv`、`test_m1_g2ctl_blind_summary.csv`、`samples_m1_g2ctl_blind_test.csv`
  - 实验臂：`results/history_m1_g2edge.csv`、`test_m1_g2edge_summary.csv`、`samples_m1_g2edge_test.csv`、`test_m1_g2edge_blind_summary.csv`、`samples_m1_g2edge_blind_test.csv`
  - 检查点：`output/ablation_m1_g2ctl/` 与 `output/ablation_m1_g2edge/`
- **成对指标对比（Test 集 2,287 样本，均为 Epoch 10 选型）：**

| 指标 | 控制臂 (`_g2ctl`) | G2a 实验臂 (`_g2edge`) | 增量 / 比值 | 判决阈值 |
|---|---|---|---:|---|
| **可训练参数量** | 71,152,964 | 74,504,522 | +3,351,558 (+4.71%) | 预期容量 |
| **eDOS Oracle 中位 $R^2$ / 失败率** | 0.4751 / 4.50% | 0.4685 / 4.72% | −0.0066 / +0.22pt | 平局线内 |
| **phDOS Oracle 中位 $R^2$ / 失败率** | 0.7220 / 1.22% | 0.7172 / 1.66% | −0.0048 / +0.44pt | 平局线内 |
| **eDOS Blind 中位 $R^2$ / 失败率** | 0.4340 / 8.35% | 0.4316 / 8.61% | −0.0024 / +0.26pt | 平局线内 |
| **phDOS Blind 中位 $R^2$ / 失败率** | 0.7115 / 2.14% | 0.7081 / 2.71% | −0.0033 / +0.57pt | 平局线内 |
| **eDOS Gap (p50 / p90 / p99)** | 0.011 / 0.155 / 0.840 | 0.012 / 0.155 / 0.751 | 持平 | - |
| **phDOS Gap (p50 / p90 / p99)** | 0.001 / 0.027 / 0.454 | 0.001 / 0.028 / 0.437 | 持平 | - |
| **$\eta$ MAE / $\gamma$ MAE** | 0.0433 / 0.0714 | 0.0443 / 0.0713 | +0.0010 / −0.0001 | 尺度头稳定 |
| **$C_v$ MAE** | 0.2906 | 0.3010 | +0.0104 | 略微退化 |
| **单轮平均耗时 (s)** | 189.9s | 241.7s | 1.273x (+27.3%) | 成本增加 |
| **峰值显存 (MB)** | 7,145 MB | 8,322 MB | 1.165x (+16.5%) | 无 OOM |

- **结果格式表达：**
  - 控制臂 Oracle：`e 0.4751/4.50% + p 0.7220/1.22% (test, ep10, oracle, Q1)`
  - 控制臂 Blind：`e 0.4340/8.35% + p 0.7115/2.14% (test, ep10, blind, Q1)`
  - G2a Oracle：`e 0.4685/4.72% + p 0.7172/1.66% (test, ep10, oracle, Q1)`
  - G2a Blind：`e 0.4316/8.61% + p 0.7081/2.71% (test, ep10, blind, Q1)`

## 结论

- **状态：PARK**（默认关闭，不进入 35-epoch 确认）。
- **原因：**
  1. 准确率未能取得正向突破：eDOS 与 phDOS 的 Oracle 中位 $R^2$ 变化分别为 −0.0066 和 −0.0048，失败率分别变化 +0.22pt 和 +0.44pt，双双严格落在平局线（$|\Delta\text{med}| < 0.02$ 且 $|\Delta\text{fail}| < 1.0\text{ pt}$）之内，未满足任一任务 $\ge +0.02$ 的晋级门槛。
  2. 盲推理尺度与 gap 分布保持稳定，但未能修正或增益谱形特征；$C_v$ MAE 亦略微劣质化（+0.0104）。
  3. 成本显著增加：每轮耗时增加 27.3%（与 resource gate 测量的 +29.5% 吻合），显存增加 16.5%，在无精度收益的情况下性价比不足。
  4. 依据停损约定，G2a 代码保持默认关闭；不展开截断半径、层数、RBF 核宽或聚合方式等超参扫描。

## 交接

- **下一项关卡工作：**
  - G2a 正式 park 后，依据 `docs/status.md` 待办顺序，中期候选进入：
    1. Decoder 缩减与 fixed-grid encoder-only atomic PDOS 设计；
    2. 或转向独立工程队列（C4 混合精度、E5/E6/E7、CIF 推理入口标准化）。
- **文档更新：**
  - 更新 `docs/status.md`：记录 G2a pilot 结论（park），移除当前执行阻塞，指向下一项。
  - 更新 `docs/decisions.md`：记录 G2a 成对 pilot 的科学结论。
