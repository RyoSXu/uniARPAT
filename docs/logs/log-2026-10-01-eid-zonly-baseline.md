# 纯原子序号表示（Z100）单种子临时 baseline：实施、运行与判读记录

日期：2026-09-30 启动，2026-10-01 完成训练与判读。对应设计 [`../design/design-element-identity-zonly.md`](../design/design-element-identity-zonly.md)。
结论口径：单次运行（seed 42）；test 未参与任何计算（训练结束日志明确跳过 test 评估）。
**结论：未达到工程采用标准**（eDOS oracle 失败率上升 1.43pp ≥ 1pp 越线），不登记为 baseline；A100 保留为对照，本轮不调整门槛、不启动下一轮。

## 1. 实现差异（最小必要范围）

| 文件 | 变更 |
|---|---|
| `model/transformer.py` | `Transformer.__init__`：`atom_feat_mode == "z_only"` 时只建 `tok_emb` 与 `atom_norm`，`num_emb_encoder/num_norm/fuse_proj` 置 `None` 不实例化；旧三模式（`legacy3`/`legacy3_const`/`mendeleev24`）分支语句与创建顺序逐字不变。`Transformer.forward`：z_only 走 `atom_src = self.atom_norm(atom_emb)`，其余路径不变。 |
| `utils/atom_feature.py` | `AtomFeatureEncoder.__init__` 对 `feat='z_only'` 显式 `ValueError`，防止误入性质表读取路径。 |
| `run_ablation_experiments.py` | `build_arg_parser` 的 `--atom_feat` choices 加入 `z_only`。 |
| `tools/eval/eid_zonly_preflight.py` | 新增：训练前检查（36 项）、manifest 写入、`--verify-run`、`--finalize`。复用 `element_identity_preflight` 的哈希/版本/构造辅助。 |
| `tools/eval/eid_zonly_monitor.py` | 新增：训练监控（等待目标 epoch、健康快照、异常检测）。 |
| `tools/eval/eid_zonly_valid_verdict.py` | 新增：valid 判读。复用 `element_identity_valid_verdict` 的逐样本 oracle/blind、配对 bootstrap、尺度误差与同组成 pair 逻辑；Z100 独立加载核对，A100/B100 逐样本结果核对后复用。 |

z_only 不新增 MLP/投影/门控/元素特征；encoder、decoder、谱头、尺度头、损失、隐藏维度、层数、任务权重与选点规则均未改动。

## 2. 训练前检查与溯源（36/36 通过，manifest：`results/eidzonly100_s42_manifest.json`）

- 结构：z_only 三个模块不存在且不进 `state_dict`；encoder 实测输入逐值等于 `atom_norm(tok_emb(atom_idx))`（forward hook 比对）；monkeypatch 性质表读取为异常后 z_only 构造与前向仍成功；`AtomFeatureEncoder(feat='z_only')` 抛错。
- 参数量：A100 模式 71,152,964 → Z100 **70,625,092**，减少恰为 **527,872**（=2048+1024+524,800），与设计预估一致。
- RNG 事实（实测）：删除模块改变初始化随机数消耗——`_reset_parameters` 的 xavier 扫描按模块创建顺序对所有 dim>1 参数赋值，落点移动，同 seed 下 Z100 与 A100 的共享参数初值（含 `tok_emb`）**均不同**；同模式（z_only）同 seed 重初始化逐值可复现。
- 数值：`edos[4,128]/phdos[4,64]/eta[4,2]` 形状正确，正反向数值有限（197 个梯度张量全有限）。
- 兼容性：`legacy3`/`legacy3_const`/`mendeleev24` 的 seed-42 初始 `state_dict` 哈希与改前快照**逐值一致**（`efca7088…`/`efca7088…`/`55a47764…`），旧路径未被改变；A100(best,ep73)/B100(best,ep87) checkpoint 可 `strict=True` 载入各自模式；z_only 显式拒绝 legacy3 权重（RuntimeError，符合预期）。
- 数据/配方：性质表 `8cfba286…`、split `8880b496…`、数据 manifest 与 A100 manifest 记录一致；Q1 train 18,706 / valid 2,313；网格 128/64、dropout 0.05；计划 CLI 与 A100 冻结配方除 atom_feat/tag 外逐项一致；FP32、`skip_test_eval=true`、`init_ckpt=''`、pair_aux 关闭；train 样本顺序与 legacy3 管线逐 epoch 一致；tag `_eidzonly100_s42` 的目录/history/summary 全部空闲，无自动续跑风险。
- 训练启动后 `--verify-run`（8/8 通过）：真实 `config_used.yaml` 生效模式为 `z_only`，其余同上；指纹未变。训练完成后 `--finalize`：`checkpoint_best.pth`=ep82、`checkpoint_latest.pth`=ep100，SHA-256 已记入 manifest。

## 3. 运行记录

命令（从零训练，未加载任何旧权重）：

```bash
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat z_only \
  --skip_test_eval --tag _eidzonly100_s42
```

- 进程与日志：`setsid` 持久进程 PID 7122（数据加载子进程 7189/8909），日志 `results/train_m1_eidzonly100_s42.log`。
- 产物：`output/ablation_m1_eidzonly100_s42/`（best=ep82、latest=ep100 两个 checkpoint）、`results/history_m1_eidzonly100_s42.csv`。
- 资源实测：**189.2 s/轮**，峰值显存 **7125 MiB**，训练合计 18,923 s ≈ **5.26 GPU 小时**（预估 5.3h 吻合）；含前置检查与 valid 判读另加约 0.5 小时。
- 异常：**无** NaN/Inf、无 OOM、无中断/续跑、无 test 路径访问；每约 5 轮监控一次（1..100 共 20 次检查点），history 与 checkpoint 每轮持续更新。
- 训练损失分量（history 逐轮记录）：`train_loss` 由 eDOS/phDOS/eta 分量构成，如 ep100 = 0.1845（eDOS 0.1322 + phDOS 0.0475 + eta 0.0048）。

学习曲线（runner valid，`balanced_score` / eDOS 中位 R² / phDOS 中位 R²）：

| epoch | Z100 | A100 | B100 |
|---|---|---|---|
| 25 | 0.6862 / 0.5003 / 0.7215 | 0.6795 / 0.5089 / 0.7399 | 0.6696 / 0.5078 / 0.7337 |
| 35 | 0.6651 / 0.5100 / 0.7292 | 0.6572 / 0.5206 / 0.7442 | 0.6550 / 0.5221 / 0.7358 |
| 50 | 0.6471 / 0.5212 / 0.7174 | 0.6429 / 0.5290 / 0.7293 | 0.6376 / 0.5328 / 0.7201 |
| 75 | 0.6444 / 0.5229 / 0.7101 | 0.6315 / 0.5309 / 0.7265 | 0.6267 / 0.5312 / 0.7235 |
| 100 | 0.6415 / 0.5223 / 0.7036 | 0.6257 / 0.5292 / 0.7239 | 0.6290 / 0.5327 / 0.7169 |

eDOS 失败率（runner valid）：Z100 从 ep25 的 6.01% 升至 ep100 的 9.90%（A100 5.58%→9.51%，B100 5.14%→9.04%）；
phDOS 失败率 Z100 3.07%→6.18%（A100 2.59%→5.84%）。三臂均为固定预算下缓慢变化，Z100 全程 balanced 略逊于两臂。

## 4. valid 判读（`results/eid_zonly_s42_valid_verdict.json`）

选点规则保持 `balanced_score = 0.5×MAE_edos_median + 0.5×MAE_phdos_median` 最小（已知受原始 MAE 尺度影响，未改规则）。
**Z100 best=ep82（0.63689）；A100 best=ep73（0.62277）；B100 best=ep87（0.62447）**。

主表（各自 best checkpoint，Q1 valid 逐样本，2,000 次配对 bootstrap 95% 区间；Δ = Z100 − A100）：

| 指标（中位 R²） | Z100 | A100 | B100 | Δ | Δ 95% CI | 失败率 Z/A | Δ失败率 pp (95% CI) |
|---|---|---|---|---|---|---|---|
| eDOS oracle | 0.52546 | 0.53232 | 0.53674 | −0.00686 | [−0.01562, +0.00245] | 9.86% / 8.43% | **+1.43 (+0.56, +2.38)** |
| eDOS blind | 0.47278 | 0.48447 | 0.48390 | −0.01169 | [−0.02169, +0.00074] | 12.97% / 12.06% | +0.91 (−0.09, +1.90) |
| phDOS oracle | 0.70537 | 0.72352 | 0.72114 | −0.01815 | [−0.02843, −0.00791] | 5.97% / 5.88% | +0.09 (−0.78, +0.91) |
| phDOS blind | 0.69664 | 0.71309 | 0.71183 | −0.01645 | [−0.02901, −0.00840] | 6.74% / 6.44% | +0.30 (−0.52, +1.08) |

第 100 轮（`checkpoint_latest`）oracle/blind 中位 R²：Z100 0.52234/0.47142/0.70364/0.69655；A100 0.52916/0.47837/0.72385/0.71945（顺序同上表）；
第 100 轮 Δ（Z−A）：−0.0068/−0.0069/−0.0202/−0.0229，失败率差 +0.39/−0.04/+0.35/+0.65pp。方向与 best 一致，phDOS 差距晚期更大。

辅助对照 Δ = Z100 − B100（best）：eDOS oracle −0.01127 [−0.01953, −0.00008]（失败率 +1.08pp）、eDOS blind −0.01112、phDOS oracle −0.01577、phDOS blind −0.01519（三项区间均不含 0）。Z100 对两个对照臂均偏低。

尺度路径（best，绝对误差中位）：Z100 gamma 0.0513 / eta 0.0256（log-ratio 0.0868/0.0302）；对照 A100 0.0492/0.0247、B100 0.0497/0.0241。Z100 尺度误差略大，但差距远小于谱形差距。

同组成 320 对（结构响应诊断，不单独决定采用）：目标谱差 TV 中位 0.26975；预测谱差 TV 中位 Z100 0.02724、A100 0.03820、B100 0.03840；「预测差−目标差」TV 中位 Z100 0.27147、A100 0.26876、B100 0.26595。Z100 的同组成对比度略低于两臂，且对比误差略大；TV 差异不代表更准确。

复用溯源：A100/B100 逐样本结果取自 `results/eid_s42_valid_samples.csv`，复用前核对其 manifest 模式、`config_used.yaml` 与 checkpoint SHA-256、表/split 指纹、样本 ID 与判读 JSON 中位数交叉核对（最大往返误差 <1e-7，属 CSV float32 十进制往返量级）；pair 复用列与固定 320 对 ID/目标 TV 核对一致。所有新产物独立命名（`results/eid_zonly_s42_valid_{verdict.json,samples.csv,pairs.csv}`），未覆盖 A100/B100 任何文件。

## 5. 采用结论：未达标

工程采用标准（单种子临时 baseline）：相对 A100 四项中位 R² 下降均 <0.02 **且** 四项失败率上升均 <1 个百分点，且实现/配置/数值检查全部通过。

- 四项中位 R² 下降：0.0069 / 0.0117 / 0.0181 / 0.0164，**均 <0.02 ✓**（但 phDOS oracle/blind 区间不含 0，下降是稳定方向；第 100 轮 phDOS 下降已达 0.020–0.023）。
- 失败率上升：eDOS oracle **+1.43pp ≥ 1pp ✗**（区间 +0.56..+2.38 不含 0）；其余三项 +0.91 / +0.09 / +0.30pp ✓。
- 实现、配置与数值检查全部通过 ✓。

**未达标**（败在 eDOS oracle 失败率一项）。按预案：完整报告结果，保留 A100 为对照，不登记 Z100 为 baseline，不自行调整门槛，不启动下一轮训练或补种子。

## 6. 边界与不确定性

1. 单 seed（42）结果；bootstrap 区间只覆盖固定 checkpoint 下的 valid 样本重采样，不含训练 seed 波动、选点不确定性与未知数据分布。
2. 删除数值分支改变初始化随机数消耗，Z100 与 A100 不共享参数初值（连 `tok_emb` 都不同）；差异是**简化架构整体效果**，不能单独归因于三项性质。
3. `balanced_score` 受原始 MAE 尺度影响；本轮保持规则一致，未改用 R² 选点。
4. 本实验未检验未见元素泛化（valid 元素均在 train 出现）；不涉及几何 encoder 路径；单元素材料同元素原子共享初始特征的机制未触及。
5. 未达标不表述为「三项性质必要」；同样不能反向宣称 Z100 在其他预算、架构或数据条件下不可用。

## 7. 交付物索引

- 实现：`model/transformer.py`（`Transformer.__init__`/`forward` z_only 分支）、`utils/atom_feature.py`（`AtomFeatureEncoder` z_only 拒绝）、`run_ablation_experiments.py`（`build_arg_parser` choices）。
- 工具：`tools/eval/eid_zonly_preflight.py`、`tools/eval/eid_zonly_monitor.py`、`tools/eval/eid_zonly_valid_verdict.py`。
- 溯源：`results/eidzonly100_s42_manifest.json`（命令、计划 CLI、指纹、代码版本与训练相关文件哈希、36 项检查、checkpoint SHA-256 与 epoch 身份）。
- 训练：`results/history_m1_eidzonly100_s42.csv`、`results/train_m1_eidzonly100_s42.log`、`output/ablation_m1_eidzonly100_s42/`。
- 判读：`results/eid_zonly_s42_valid_verdict.json`、`results/eid_zonly_s42_valid_samples.csv`、`results/eid_zonly_s42_valid_pairs.csv`、`results/eid_zonly_verdict_run.log`。
- 异常/中断说明：无。
