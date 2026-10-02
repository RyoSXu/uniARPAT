# 元素身份对照实验（seed 42，A100/B100 各 100 轮）：实施、运行与判读记录

日期：2026-09-30。对应设计 [`../design/design-element-identity-control.md`](../design/design-element-identity-control.md)。
结论口径：单次运行（seed 42）的初步证据；test 未参与任何计算。基线是否升级由用户另行决定，本页不改 [`../status.md`](../status.md)。

## 1. 实现差异（文档最小范围，3 处）

| 文件 | 变更 |
|---|---|
| `utils/atom_feature.py` | `AtomFeatureEncoder` 新增 `legacy3_const` 分支：沿 `legacy3` 路径取得 `F`，复制后将 `F[1:]` 全部替换为常数 `c = F[1:119].mean(dim=0)`，保留 `F[0]`；不引入随机抽样，可训练层创建顺序不变，`legacy3` 路径逐值不变。新增 `legacy3_const_vector()` 供 manifest 与判读核对常数。 |
| `run_ablation_experiments.py` | `build_arg_parser` 的 `--atom_feat` choices 加入 `legacy3_const`。`model/transformer.py` 经核对无需改动（非 `mendeleev24` 模式均为 3 维性质层、仅 `mendeleev24` 零初始化 `tok_emb`）；pair-aux 冻结校验仍要求 `atom_feat == legacy3`，继续拒绝新模式。 |
| `tools/eval/element_identity_preflight.py` | 新增：实施后训练前检查（26 项）与实验 manifest 写入、`--verify-run` 真实配置核对、`--finalize` checkpoint SHA-256 补记。 |
| `tools/eval/element_identity_valid_verdict.py` | 新增：Q1 valid 判读工具。复用 `periodic_manybody_valid_verdict.py` 的逐样本 oracle/blind 与配对 bootstrap，移除固定 B7 路径、epoch 33、`use_periodic_manybody` 与「A 等于历史 B7」断言；判读前逐臂核对 manifest、配方、表指纹、常数与 checkpoint 身份。 |

判读/前置工具在训练后做过一处文本 IO `encoding="utf-8"` 修正（脚本方式运行时该环境默认编码为 ASCII）；模型与 runner 代码自训练开始未再改动。

## 2. 前置检查与溯源

- 检查 26/26 通过：两臂 `feature_map` 均 `[119,3]`，A 与现行表逐值一致，B 第 0 行为零、第 1..118 行恒为 `c`；同 seed 初始 `state_dict` 键、形状、参数值逐值一致（可训练参数 71,152,964）；同臂重初始化可复现；正反向数值有限，M1 双谱输出 `edos[4,128]/phdos[4,64]` 与 H1 `eta[4,2]` 形状正确；`mendeleev24` 零初始化未触发；两臂 train 样本 ID 顺序一致（train 18,706，epochs 0-1）；目标 tag 空闲；表/split 哈希与设计文档一致。
- 常数向量：`c = F[1:119].mean(dim=0)` 实测 `float32` = `[0.49465635418891907, 0.39238882064819336, 0.4033089280128479]`，与文档预注册值一致。
- 指纹：`utils/periodic_table_v2.csv` SHA-256 `8cfba286b85f175c73431011a5c24bb8b110223ed60e81bae0357a02af9318fc`；`index/split_v2.yaml` SHA-256 `8880b49652f72d5b5c5690afcc06ccaf643e3072c4789f58920256df42e43c41`；数据 manifest `data/train4ARPAT/manifest.json` 哈希已记录。
- manifest：`results/eidprop100_s42_manifest.json`、`results/eidconst100_s42_manifest.json`（含命令、计划 CLI 与哈希、代码版本 `3352732`+dirty、表指纹、常数、配置哈希、checkpoint SHA-256 与 epoch 身份）。
- 真实运行配置核对（训练启动后 `--verify-run`）：两臂 `skip_test_eval=true`、`init_ckpt=''`、`use_amp=false`、`pair_aux_arm=none`、`atom_feat_mode` 分别为 `legacy3`/`legacy3_const`，其余与 B7 冻结配方一致。

## 3. 运行记录

命令（顺序执行，单张 V100-32GB，FP32）：

```bash
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat legacy3 \
  --skip_test_eval --tag _eidprop100_s42
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat legacy3_const \
  --skip_test_eval --tag _eidconst100_s42
```

- 产物：`output/ablation_m1_eidprop100_s42/`、`output/ablation_m1_eidconst100_s42/`（各 best/latest 两个 checkpoint），`results/history_m1_eid{prop,const}100_s42.csv`（逐轮损失、valid 指标、耗时、显存），`results/train_m1_eid{prop,const}100_s42.log`。
- 资源实测：两臂均约 `189.7 s/轮`，峰值显存约 `7140 MiB`；每臂约 5.3 GPU 小时，合计约 10.5 GPU 小时（与预估一致）。
- 异常：无 NaN/Inf、无 OOM、无中断/续跑、无 test 路径访问。两臂均完整跑满 100 轮。

学习曲线关键读数（runner valid，`balanced_score` / eDOS 中位 R² / phDOS 中位 R²）：

| epoch | A100 | B100 |
|---|---|---|
| 25 | 0.6795 / 0.5089 / 0.7399 | 0.6696 / 0.5078 / 0.7337 |
| 35 | 0.6572 / 0.5206 / 0.7442 | 0.6550 / 0.5221 / 0.7358 |
| 50 | 0.6429 / 0.5290 / 0.7293 | 0.6376 / 0.5328 / 0.7201 |
| 75 | 0.6315 / 0.5309 / 0.7265 | 0.6267 / 0.5312 / 0.7235 |
| 100 | 0.6257 / 0.5292 / 0.7239 | 0.6290 / 0.5327 / 0.7169 |

完整逐轮曲线见两个 history CSV。两臂全程互有交替，未见 B 晚期追赶或退化的系统性模式；两臂 phDOS 都在 ep35 附近见顶后缓慢回落（A：0.7442→0.7239），eDOS 缓升，100 轮预算内 valid 仍缓慢变化，按文档记为「固定预算比较」。

## 4. valid 判读（`results/eid_s42_valid_verdict.json`）

选点规则同 runner：valid `balanced_score = 0.5×MAE_edos_median + 0.5×MAE_phdos_median` 最小。**A100 best=ep73（0.62277）；B100 best=ep87（0.62447）**。第 100 轮 runner 读数：A100 eDOS/phDOS 0.5292/0.7239（balanced 0.62575）；B100 0.5327/0.7169（0.62897）。

主表：两臂各自 best checkpoint 的 Q1 valid 逐样本指标（2,000 次配对 bootstrap，95% 区间；Δ = B − A）：

| 指标（中位 R²） | A100 | B100 | Δ | Δ 95% CI | 失败率 A/B | Δ失败率 pp (95% CI) |
|---|---|---|---|---|---|---|
| eDOS oracle（主） | 0.53232 | 0.53674 | +0.00441 | [−0.00555, +0.01242] | 8.43% / 8.78% | +0.35 (−0.48, +1.17) |
| eDOS blind | 0.48447 | 0.48390 | −0.00057 | [−0.00864, +0.01054] | 12.06% / 12.24% | +0.17 (−0.74, +1.12) |
| phDOS oracle | 0.72352 | 0.72114 | −0.00238 | [−0.01125, +0.00556] | 5.88% / 5.58% | −0.30 (−0.99, +0.35) |
| phDOS blind | 0.71309 | 0.71183 | −0.00126 | [−0.01212, +0.00669] | 6.44% / 6.10% | −0.35 (−1.08, +0.30) |

第 100 轮（`checkpoint_latest`）oracle/blind：A100 0.52916/0.47837/0.72385/0.71945；B100 0.53270/0.48475/0.71689/0.71185（顺序同上表）。与 best 读数方向一致。

- 四项 Δ 的区间均含 0，点估计绝对值均在项目描述线 `0.02` 以内；失败率差绝对值均 <1pp。这是**单次运行（seed 42）的初步证据**，按设计第 6 节不称「稳定增量」，也不称「元素身份已足够」。
- oracle 与 blind 在 eDOS 上方向不一致（oracle B 略高、blind B 略低），按设计补报 γ/η 误差定位尺度路径：γ 绝对误差中位 A 0.0492 / B 0.0497（log-ratio 0.0858/0.0852），η 绝对误差中位 A 0.0247 / B 0.0241（log-ratio 0.0300/0.0295）。两臂尺度误差几乎相同，oracle/blind 差距主要来自尺度路径本身，而非两臂差异。
- 部署保护：B 臂 eDOS blind 与 phDOS oracle/blind 均未越线（中位 R² 下降 <0.02、失败率上升 <1pp），`pass=true`。
- 元素频次分层（最稀有元素 train 原子槽次数 <500：239 条；其余 2,074 条）：稀有组 B 较高（eDOS oracle Δ +0.0188、phDOS oracle Δ +0.0212），其余组 eDOS oracle Δ +0.0059、phDOS 约 −0.003。仅作线索，不单独判胜。
- 同组成 pair（固定 320 对）：目标谱差 TV 中位 0.26975；预测谱差 TV 中位 A100 0.03820、B100 0.03840（几乎相同）；「预测差−目标差」TV 中位 A100 0.26876、B100 0.26595。作为背景：历史 B7 预测谱差 TV 约 0.01604——两臂新模型的同组成对比度都比 B7 大，但仍远低于目标，TV 增大不代表更准确。

逐样本明细 `results/eid_s42_valid_samples.csv`（2,313 行），pair 明细 `results/eid_s42_valid_pairs.csv`。

## 5. A100 与历史 B7 的同口径比较（背景对照）

同一 `_predict` 口径重算 B7（ep33 冻结 checkpoint），并与其历史逐样本 CSV 交叉核对（最大绝对差 <1e-6，口径一致）。Δ = A100 − B7：

| 指标（中位 R²） | B7 | A100 | Δ | Δ 95% CI | Δ失败率 pp |
|---|---|---|---|---|---|
| eDOS oracle | 0.51498 | 0.53232 | +0.01734 | [+0.00696, +0.02578] | +2.16 |
| eDOS blind | 0.47314 | 0.48447 | +0.01133 | [−0.00018, +0.02290] | +2.94 |
| phDOS oracle | 0.74380 | 0.72352 | −0.02028 | [−0.03102, −0.01243] | +2.08 |
| phDOS blind | 0.73787 | 0.71309 | −0.02478 | [−0.03293, −0.01432] | +2.08 |

- **调度差异警示**：B7 为 35 轮、warmup+cosine 35 轮调度；A100 为 100 轮、100 轮调度从零训练。差异混合了训练长度与学习率调度，不能单独归因于「多训练 65 轮」。
- 事实：A100 的 eDOS 中位 R² 提升约 +0.017（oracle 区间不含 0），但 phDOS 中位 R² 下降约 0.02–0.025（区间不含 0），且四项失败率全部上升约 2–3 个百分点。A100 的 eDOS 提升部分伴随尾部变差（失败率上升），phDOS 退化与其曲线在 ep35 后回落一致。
- 结论语言：这是同 valid 口径下的描述性比较；eDOS/phDOS 权衡明显，属设计第 6 节「任务间权衡」情形。

## 6. 结论与边界

1. 在当前 Q1/M1/100 轮配方下，seed 42 单次运行未观察到超过项目描述线的两臂差异（eDOS oracle Δ = +0.004，四项区间均含 0）；B 的部署保护通过。这是初步证据，跨初始化结论需要 seed 43/44（未执行，待用户决定）。
2. 本实验不能证明三项性质含 `Z` 之外的信息；若 B 变差也只能归因于「保留数值分支、消除元素间差异」这一整体操作。B 未变差本身不构成「删除分支」的依据。
3. 不能检验未见元素泛化（valid 元素均在 train 出现）；不涉及几何 encoder 路径；单元素材料的同元素原子共享初始特征的机制未被触及。
4. 100 轮内两臂 valid 仍在缓慢变化，属固定预算比较，非「充分收敛」保证。

## 7. 基线升级建议（仅建议，待用户决定）

不建议直接将 A100 升为统一基线：相对 B7，eDOS 中位 R² 提升约 +0.017，但 phDOS oracle/blind 下降约 0.02–0.025（越过 0.02 描述线）、四项失败率上升 2–3pp，且该差异混合了调度变化，不能归因于训练长度。若后续实验以 eDOS 为主指标并接受 phDOS 权衡，可由用户决定以 A100 为基线并在 `docs/status.md` 显式记录该权衡；或先补 seed 43/44 与调度对照（例如 35 轮调度×100 轮预算的拆分实验）再定。任何基线更新由用户确认后执行。

## 8. 交付物索引

- 实现：`utils/atom_feature.py`、`run_ablation_experiments.py`、`tools/eval/element_identity_preflight.py`、`tools/eval/element_identity_valid_verdict.py`。
- manifest：`results/eidprop100_s42_manifest.json`、`results/eidconst100_s42_manifest.json`。
- 训练：`results/history_m1_eidprop100_s42.csv`、`results/history_m1_eidconst100_s42.csv`、`results/train_m1_eid{prop,const}100_s42.log`；checkpoint 在 `output/ablation_m1_eid{prop,const}100_s42/`（SHA-256 已记入 manifest）。
- 判读：`results/eid_s42_valid_verdict.json`、`results/eid_s42_valid_samples.csv`、`results/eid_s42_valid_pairs.csv`。
- 异常/中断说明：无。
