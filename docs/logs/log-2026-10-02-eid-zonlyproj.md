# 元素表示投影（ZP100）对齐初始化候选：实施、运行与判读记录

日期：2026-10-01 启动，2026-10-02 完成训练与判读。对应设计 [`../design/design-element-identity-zonlyproj.md`](../design/design-element-identity-zonlyproj.md)。
结论口径：单次运行（seed 42）；test 未参与任何计算（训练结束日志明确跳过 test 评估）。
**结论：未达到工程采用标准**（eDOS oracle 失败率上升 1.04pp ≥ 1pp 越线，与 Z100 败在同一项、幅度略小），
不登记为 baseline，A100 保留为对照；对 Z100 的问题（加投影是否改善）答：**未观察到可区分的改善**——
四项中位 R² 的 Δ多为小幅正值但 95% 区间全部跨 0，主指标 eDOS oracle 基本不动。

## 1. 实现差异（最小必要范围）

| 文件 | 变更 |
|---|---|
| `model/transformer.py` | `Transformer.__init__`：`atom_feat_mode in ("z_only","z_only_proj")` 分支；`z_only_proj` 新增 `atom_proj = Linear(512→512, bias=True)`（`z_only` 下为 `None`，不进 `state_dict`），`num_emb_encoder/num_norm/fuse_proj` 同样置 `None`。`Transformer.forward`：`z_only_proj` 走 `atom_src = atom_proj(atom_norm(tok_emb(atom_idx)))`。旧三模式（`legacy3`/`legacy3_const`/`mendeleev24`）分支语句与创建顺序逐字不变。 |
| `utils/atom_feature.py` | `AtomFeatureEncoder.__init__` 对 `feat in ('z_only','z_only_proj')` 显式 `ValueError`（`z_only` 的报错文本逐字不变）。 |
| `run_ablation_experiments.py` | `build_arg_parser` 的 `--atom_feat` choices 加入 `z_only_proj`；新增 `validate_zproj_config`（拒绝 `--init_ckpt`、`--reset_rng_after_init`、`--freeze_backbone`、pair aux、AMP、未开 `--skip_test_eval`）；`train_and_eval` 增加受控对齐三步（重建参考 → 复制+核对 → 检查后恢复随机状态并在进循环前复核），非 `z_only_proj` 路径逐字不变。 |
| `utils/zproj_alignment.py` | 新增：对齐初始化唯一实现（参考重建、共享参数复制、单位投影、训练前数值核对、样本顺序哈希、随机状态保存/恢复/摘要、记录写盘），runner 与 preflight 共用。 |
| `tools/eval/eid_zproj_preflight.py` | 新增：训练前检查（59 项）、manifest 写入、`--verify-run`、`--finalize`。复用 `element_identity_preflight` 与 `eid_zonly_preflight` 的哈希/版本/构造辅助。 |
| `tools/eval/eid_zproj_monitor.py` | 新增：训练监控（等待目标 epoch、健康快照、异常检测），路径常量指向本候选。 |
| `tools/eval/eid_zproj_valid_verdict.py` | 新增：valid 判读。复用 `element_identity_valid_verdict` 的逐样本 oracle/blind、配对 bootstrap、尺度误差与同组成 pair 逻辑；ZP100 独立加载核对，A100/B100/Z100 逐样本结果核对后复用。 |

`z_only_proj` 不使用元素性质或常数分支，不新增门控或额外元素特征；encoder、decoder、谱头、尺度头（H1 eta/gamma）、
损失、隐藏维度、层数、任务权重与选点规则均未改动。新增参数恰为 `512×512+512 = 262,656`。

## 2. 训练前检查与溯源（manifest：`results/eidzproj100_s42_manifest.json`）

**59/59 通过**。要点：

- **参考重建与初始指纹**：Z100 manifest 记录的 `state_hash_z_only_init = 4eda5029…` 与其冻结常数一致；
  按 Z100 原配置（其 `config_used.yaml` 的 `config` 段）与 seed 42 重建的未训练模型指纹**逐值复现**；
  runner 构造路径（`ConfigBuilder.get_model → basemodel → Transformer`）同样复现该指纹；Z100 的
  `config_used.yaml` 与 best/latest checkpoint 哈希未漂移。**未加载任何训练好的权重**。
- **唯一模型改动**：`z_only_proj` 的 `state_dict` 恰好多 `atom_proj.{weight,bias}` 两键；
  encoder 实测输入逐值等于 `atom_proj(atom_norm(tok_emb(atom_idx)))`（forward hook）；
  `z_only` 公式复核不变；构造与前向不读性质表；`AtomFeatureEncoder(feat='z_only_proj')` 抛错；
  可训练参数 **70,887,748** = 70,625,092 + 262,656。
- **对齐初始化**（与训练进程同一实现）：共享参数 199 个键逐值复制自 Z100 初值（独立复核重复比对）；
  `atom_proj` = `I(512)`/零偏置（独立复核）；前向与梯度有限（199 个梯度张量全有限）；
  对齐后前向与 z_only 参考在同一批次上**逐位相同**（edos/phdos/eta 最大绝对差 0）；
  检查结束后随机状态恢复为 Z100 训练开始时状态（摘要 `298bdcd45778a4a7…`），另有一条不经 loader 的
  独立重放路径给出同一摘要——实测证明 DataLoader/数据集构造不消耗随机数。
- **数据顺序**：候选与 z_only 管线 epoch 0/1 样本顺序逐值一致（哈希 `a46f188be91ab62b…` /
  `ecb0c8cfe6f8e19f…`），sampler 覆盖全部 18,706 条；顺序由 `DistributedSampler(seed=0)+set_epoch`
  决定，与随机状态无关。
- **兼容性**：`legacy3`/`legacy3_const`/`mendeleev24` 的 seed-42 初始 `state_dict` 哈希与改前快照
  **逐值一致**（`efca7088…`/`efca7088…`/`55a47764…`）；A100/B100/Z100 checkpoint 各自 `strict=True`
  载入自己的模式；`z_only_proj` 显式拒绝三臂训练权重（RuntimeError，符合"不加载训练好的权重"）。
- **数据/配方**：性质表 `8cfba286…`、split `8880b496…`、数据 manifest 与 A100 manifest 记录一致；
  Q1 train 18,706 / valid 2,313；网格 128/64、dropout 0.05；计划 CLI 与冻结配方除 `atom_feat`/`tag`/`dropout`
  外逐项一致；FP32、`skip_test_eval=true`、`init_ckpt=''`、pair_aux 关闭。
- **tag 空闲核对**：无 checkpoint（runner 不会自动续跑）、无 history/summary、manifest 未占用；
  崩溃的第 1 次启动残留的 `config_used.yaml` 如实记录（见第 3 节），未删除。
- 训练启动后 `--verify-run`（13/13 通过）：真实 `config_used.yaml` 生效模式 `z_only_proj`、其余配方同上；
  `zproj_align.json` 与 manifest 记录一致（指纹、6/6 检查、随机状态摘要、样本顺序、共享参数数）；
  进程 PID 已登记。训练完成后 `--finalize`（3/3）：`checkpoint_best.pth`=ep89、`checkpoint_latest.pth`=ep100，
  SHA-256 已记入 manifest（`2d217097…` / `22ad8120…`），`zproj_align.json` 与配置自 verify-run 后未变。

## 3. 运行记录

命令（从零训练，共享初值来自未训练参考模型，不加载任何 checkpoint）：

```bash
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat z_only_proj \
  --skip_test_eval --tag _eidzproj100_s42
```

- 对齐做法（`zproj_align.json` 与 manifest `alignment` 段记录）：参考模型以 z_only 模式按 runner 同一
  构造路径重建（初值指纹 `4eda5029…`），共享参数 199 键复制、`atom_proj` 置单位阵/零偏置；
  训练前检查（6 项）后恢复参考构造结束时的四路随机状态（python/numpy/torch CPU/torch CUDA，
  摘要 `298bdcd45778a4a7…`），并在进入训练循环前复核摘要未变（否则中止）。数据顺序由
  `DistributedSampler(seed=0)+set_epoch` 决定，已核对与 z_only 管线逐值一致。因此候选与 Z100 的
  差别只有新增投影及其梯度。实测旁证：ep1 `train_loss` ZP100 1.20072 vs Z100 1.20315（A100 1.23913、
  B100 1.24019）——共享初值与随机流下两臂第 1 轮几乎重合，未对齐的两臂差约 0.036。
- 进程与日志：`setsid` 持久进程（监控按命令行匹配登记 PID `[18588, 18589, 18705, 20137]`，
  含启动包装、python 主进程与 DataLoader 子进程），日志 `results/train_m1_eidzproj100_s42.log`。
- 产物：`output/ablation_m1_eidzproj100_s42/`（`checkpoint_best.pth`=ep89、`checkpoint_latest.pth`=ep100、
  `config_used.yaml`、`zproj_align.json`）、`results/history_m1_eidzproj100_s42.csv`。
- 资源实测：**189.3 s/轮**，峰值显存 **7138 MiB**，训练合计 18,928 s ≈ **5.26 GPU 小时**（预估 5.3h 吻合）；
  含前置检查与判读另加约 0.7 小时。
- **异常（1 次，训练前）**：第 1 次启动（2026-10-01 20:56，PID 5575）在第 1 轮训练循环开始前，
  因 `utils/zproj_alignment.py::write_alignment_record` 收到 `str` 路径调用 `Path.parent` 抛
  `AttributeError` 退出：**0 epoch、无 checkpoint、无 history**。证据保留：
  `results/train_m1_eidzproj100_s42_attempt1_crashed.log`、`results/eidzproj100_s42_manifest_attempt1.json`
  与崩溃尝试写下的 `output/ablation_m1_eidzproj100_s42/config_used.yaml`（均未删除）。
  修复为 `write_alignment_record` 接受 `str`/`Path`；未改任何参数或配方。经用户确认后按原命令重启，
  重启前的检查把上述残留如实记入 manifest `prior_attempt` 段。除此之外：**无** NaN/Inf、无 OOM、
  无中断/续跑、无 test 路径访问；每约 5 轮监控一次（1..100 共 20 次检查点），history 与 checkpoint
  每轮持续更新。
- 训练损失分量（history 逐轮）：ep1 = 1.2007（eDOS 0.5680 + phDOS 0.6197 + eta 0.0130）；
  ep89 = 0.1892（0.1342 + 0.0502 + 0.0048）；ep100 = 0.1846（0.1322 + 0.0477 + 0.0047）。

学习曲线（runner valid，`balanced_score` / eDOS 中位 R² / phDOS 中位 R²）：

| epoch | ZP100 | Z100 | A100 | B100 |
|---|---|---|---|---|
| 25 | 0.6809 / 0.5019 / 0.7250 | 0.6862 / 0.5003 / 0.7215 | 0.6795 / 0.5089 / 0.7399 | 0.6696 / 0.5078 / 0.7337 |
| 35 | 0.6635 / 0.5129 / 0.7278 | 0.6651 / 0.5100 / 0.7292 | 0.6572 / 0.5206 / 0.7442 | 0.6550 / 0.5221 / 0.7358 |
| 50 | 0.6521 / 0.5260 / 0.7110 | 0.6471 / 0.5212 / 0.7174 | 0.6429 / 0.5290 / 0.7293 | 0.6376 / 0.5328 / 0.7201 |
| 75 | 0.6405 / 0.5230 / 0.7108 | 0.6444 / 0.5229 / 0.7101 | 0.6315 / 0.5309 / 0.7265 | 0.6267 / 0.5312 / 0.7235 |
| 100 | 0.6368 / 0.5231 / 0.7105 | 0.6415 / 0.5223 / 0.7036 | 0.6257 / 0.5292 / 0.7239 | 0.6290 / 0.5327 / 0.7169 |

ZP100 全程 balanced 与 Z100 互有胜负（25/35/75/100 轮略优、50 轮略逊），差距 ≤0.005；两者一致低于 A100/B100。
eDOS 失败率（runner valid）：ZP100 从 ep25 的 5.53% 升至 ep100 的 9.90%（Z100 6.01%→9.90%，A100 5.58%→9.51%）；
phDOS 失败率 ZP100 3.11%→6.01%（Z100 3.07%→6.18%）。

## 4. valid 判读（`results/eid_zproj_s42_valid_verdict.json`）

选点规则保持 `balanced_score = 0.5×MAE_edos_median + 0.5×MAE_phdos_median` 最小（已知受原始 MAE 尺度影响，未改规则）。
**ZP100 best=ep89（0.63310）；Z100 best=ep82（0.63689）；A100 best=ep73（0.62277）；B100 best=ep87（0.62447）**。

主表（各自 best checkpoint，Q1 valid 逐样本，2,000 次配对 bootstrap 95% 区间）：

| 指标（中位 R²） | ZP100 | Z100 | A100 | B100 | Δ=P−Z100 (95% CI) | Δ=P−A100 (95% CI) | 失败率 P/Z/A | Δ失败率 P−A100 pp (95% CI) |
|---|---|---|---|---|---|---|---|---|
| eDOS oracle | 0.52460 | 0.52546 | 0.53232 | 0.53674 | −0.00087 [−0.00742, +0.00820] | −0.00772 [−0.01563, +0.00276] | 9.47% / 9.86% / 8.43% | **+1.04 (+0.17, +1.95)** |
| eDOS blind | 0.48090 | 0.47278 | 0.48447 | 0.48390 | +0.00812 [−0.00028, +0.01591] | −0.00357 [−0.01393, +0.00750] | 12.62% / 12.97% / 12.06% | +0.56 (−0.43, +1.60) |
| phDOS oracle | 0.70987 | 0.70537 | 0.72352 | 0.72114 | +0.00450 [−0.00764, +0.01422] | −0.01365 [−0.02822, −0.00202] | 6.14% / 5.97% / 5.88% | +0.26 (−0.52, +1.08) |
| phDOS blind | 0.69902 | 0.69664 | 0.71309 | 0.71183 | +0.00238 [−0.00664, +0.01288] | −0.01407 [−0.02762, −0.00396] | 6.66% / 6.74% / 6.44% | +0.22 (−0.52, +0.99) |

Δ=P−Z100 失败率差（best）：−0.39 / −0.35 / +0.17 / −0.09pp，区间均含 0。
辅助对照 Δ = P−B100（best）：eDOS oracle −0.01214 [−0.02029, +0.00017]（失败率 +0.69pp）、eDOS blind −0.00300、
phDOS oracle −0.01127、phDOS blind −0.01281（后者区间不含 0）。

第 100 轮（`checkpoint_latest`）oracle/blind 中位 R²：ZP100 0.52309/0.47941/0.71054/0.69767；
Z100 0.52234/0.47142/0.70364/0.69655；A100 0.52916/0.47837/0.72385/0.71945（顺序同上表）。
第 100 轮 Δ（P−Z100）：+0.0008/+0.0080/+0.0069/+0.0011（四项区间均含 0）；失败率差 −0.00/−0.04/−0.17/−0.43pp。
第 100 轮 Δ（P−A100）：−0.0061/+0.0010/−0.0133/−0.0218；失败率差 +0.39/−0.09/+0.17/+0.22pp。

尺度路径（best，绝对误差中位）：ZP100 gamma 0.0488 / eta 0.0252（log-ratio 0.0857/0.0302）；
对照 Z100 0.0513/0.0256（0.0868/0.0302）、A100 0.0492/0.0247、B100 0.0497/0.0241。
同组成 320 对（结构响应诊断，不单独决定采用）：目标谱差 TV 中位 0.26975；预测谱差 TV 中位
ZP100 0.02732、Z100 0.02724、A100 0.03820、B100 0.03840；「预测差−目标差」TV 中位 ZP100 0.26938、
Z100 0.27147、A100 0.26876、B100 0.26595。投影没有改变同组成对比度坍缩（预测谱差 TV 与 Z100 同量级）。

复用溯源：A100/B100/Z100 逐样本结果取自 `results/eid_s42_valid_samples.csv`（A/B 原始）与
`results/eid_zonly_s42_valid_samples.csv`（Z100 及其 A/B 复制列），复用前核对三臂 manifest 模式、
`config_used.yaml` 与 best/latest checkpoint SHA-256、表/split 指纹、样本 ID 顺序，A/B 列与原始 CSV
逐值一致（最大差 0.0），Z100 列中位数/失败率与其判读 JSON 交叉核对（最大往返误差 <1e-6）；
pair 复用列与固定 320 对 ID/目标 TV 核对一致。所有新产物独立命名
（`results/eid_zproj_s42_valid_{verdict.json,samples.csv,pairs.csv}`），未覆盖 A100/B100/Z100 任何文件。

## 5. 两个问题的结论

**Q1（与 Z100 比：这个初始化方案下，加投影是否改善表现？）——未观察到可区分的改善。**
四项中位 R² 的 Δ（P−Z100）在 best 为 −0.0009/+0.0081/+0.0045/+0.0024、第 100 轮为
+0.0008/+0.0080/+0.0069/+0.0011：方向上 8 个差值中 7 个为小幅正值（eDOS blind 最大约 +0.008），
但 **95% 区间全部跨 0**，主指标 eDOS oracle 基本不动；失败率差区间也都含 0。best 选点分数
（balanced 0.63310 vs 0.63689）改善 0.0038，但两者 best 轮次不同（ep89 vs ep82）且 R² 差异在重采样
噪声内。按项目描述线（|Δ中位 R²| < 0.02）也未见超过描述线的差异。因此只能说：在本次配方、
本次对齐初始化下，**投影没有带来稳定可见的收益，也没有明显损害**；不能声称改善，也不能声称投影无用
（单 seed、区间跨 0）。同组成预测谱差 TV 与 Z100 几乎相同（0.0273 vs 0.0272），投影未改变结构响应坍缩。

**Q2（与 A100 比：是否达到新 baseline 的采用标准？）——未达标。**

- 四项中位 R² 下降（best，P−A100）：0.0077 / 0.0036 / 0.0137 / 0.0141，**均 <0.02 ✓**；
  phDOS oracle/blind 区间不含 0（−0.0282..−0.0020 / −0.0276..−0.0040），下降是稳定方向。
- 失败率上升：eDOS oracle **+1.04pp ≥ 1pp ✗**（区间 +0.17..+1.95 不含 0）；其余 +0.56 / +0.26 / +0.22pp ✓。
- 实现、配置与数值检查全部通过 ✓（preflight 59/59、训练内对齐 6/6、verify-run 13/13、finalize 3/3）。

**未达标**（败在 eDOS oracle 失败率一项，与 Z100 同项、幅度略小）。按预案：完整报告结果，
**保留 A100 为对照，不登记 ZP100 为 baseline**，不自行调整门槛，不启动下一轮训练或补种子。

## 6. 边界与不确定性

1. 单 seed（42）结果；bootstrap 区间只覆盖固定 checkpoint 下的 valid 样本重采样，不含训练 seed 波动、
   选点不确定性与未知数据分布。结论限于本次配方与本次对齐初始化，不能声称跨种子稳定。
2. 对齐范围：共享初值逐值、训练随机状态（dropout 流）与数据顺序一致，初始前向逐位相同；但投影使
   可训练参数 +262,656、优化器状态数量与梯度路径随之不同，ZP100 与 Z100 的差别是**新增投影整体效果**
   （含其优化动力学），不是"纯投影矩阵的作用"这一更细的问题。
3. `balanced_score` 受原始 MAE 尺度影响；本轮保持规则一致，未改用 R² 选点（ZP100 与 Z100 的 best
   轮次不同，比较时应同时看第 100 轮）。
4. 本实验未检验未见元素泛化（valid 元素均在 train 出现）；不涉及几何 encoder 路径；单元素材料
   同元素原子共享初始特征的机制未触及；同组成对比度坍缩未被改善（预测谱差 TV 0.0273 vs 目标 0.270）。
5. 第 1 次启动崩溃属实现缺陷（记账代码 Path/str），发生在第 1 轮前、0 epoch；按用户确认后按原命令重启。
   两次启动共用同一 tag，重启前已核对无 checkpoint、无 history，不构成自动续跑或结果覆盖。
6. 未达标不表述为「投影无用」或「Z100 已等价于 A100」；同样不能反向宣称 ZP100 在其他预算、
   架构或数据条件下不可用。

## 7. 交付物索引

- 实现：`model/transformer.py`（`z_only_proj` 分支）、`utils/atom_feature.py`（拒绝 `z_only_proj`）、
  `run_ablation_experiments.py`（choices、`validate_zproj_config`、对齐三步）、`utils/zproj_alignment.py`（对齐实现）。
- 工具：`tools/eval/eid_zproj_preflight.py`、`tools/eval/eid_zproj_monitor.py`、`tools/eval/eid_zproj_valid_verdict.py`。
- 溯源：`results/eidzproj100_s42_manifest.json`（命令、计划 CLI、指纹、代码版本与训练相关文件哈希、
  59 项检查、对齐细节、`prior_attempt` 崩溃记录、checkpoint SHA-256 与 epoch 身份、进程 PID）；
  `output/ablation_m1_eidzproj100_s42/zproj_align.json`（训练内对齐记录）；
  `results/eidzproj100_s42_manifest_attempt1.json`、`results/train_m1_eidzproj100_s42_attempt1_crashed.log`（第 1 次启动）。
- 训练：`results/history_m1_eidzproj100_s42.csv`、`results/train_m1_eidzproj100_s42.log`、
  `output/ablation_m1_eidzproj100_s42/`（best=ep89、latest=ep100）。
- 判读：`results/eid_zproj_s42_valid_verdict.json`、`results/eid_zproj_s42_valid_samples.csv`、
  `results/eid_zproj_s42_valid_pairs.csv`、`results/eid_zproj_verdict_run.log`。
- 异常/中断说明：第 1 次启动训练前崩溃 1 次（见第 3 节），修复后按原命令重启；正式训练无中断、无续跑。
