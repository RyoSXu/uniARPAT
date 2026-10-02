# 纯原子序号表示（Z100）单种子临时 baseline：实验设计记录

日期：2026-09-30。状态：**已实施并完成判读**（本轮提示词授权 M1、seed 42、100 轮、`--atom_feat z_only`）。
当前方案与代码职责见[元素初始化说明](design-element-initialization.md)；本页保留历史 Z100 实验协议。
本页是实施前冻结的实验设计记录；结果与判读见
[`../logs/log-2026-10-01-eid-zonly-baseline.md`](../logs/log-2026-10-01-eid-zonly-baseline.md)（结论：未达采用标准，eDOS oracle 失败率上升 1.43pp 越线）。
本候选对应 [`design-element-identity-control.md`](design-element-identity-control.md) 第 7 节预留的「删分支检验精简架构」追问；旧 B7 只作历史参考，不是本轮采用门槛。

后续记录：本页第 6 节边界指出 Z100 与 A100 不共享参数初值（删除模块改变随机数消耗），差异不能单独归因于三项性质。
该混淆由 [`design-element-identity-zonlyproj.md`](design-element-identity-zonlyproj.md) 的 ZP100 追问收窄：
`--atom_feat z_only_proj` 在 Z100 表示上加 `Linear(512→512)` 投影，共享参数复制自 Z100 seed-42 未训练初值、
投影=单位阵/零偏置、训练随机状态与数据顺序对齐。结果见
[`../logs/log-2026-10-02-eid-zonlyproj.md`](../logs/log-2026-10-02-eid-zonlyproj.md)：相对 Z100 未观察到可区分改善
（四项 Δ 区间全跨 0），相对 A100 仍败在 eDOS oracle 失败率 +1.04pp，A100 保留为对照。

## 1. 要回答的问题

在 Q1/M1 双谱配方与 100 轮预算下，仅用可学习的原子序数 embedding（不输入质量、半径、电负性三项显式性质）能否达到与 A100 相当的 valid 精度，
从而作为后续实验更简单的元素表示 baseline？

## 2. 冻结方案（Z100）

元素输入退化为：

```
atom_src = atom_norm(tok_emb(atom_idx))
```

1. 保留现有原子序数 embedding `tok_emb`、`atom_norm`、输入索引与 padding/mask 契约；`atom_idx = src[:, 2:]` 不变。
2. `z_only` 模式不实例化、不调用数值性质编码器 `AtomFeatureEncoder`、`num_norm` 与 `fuse_proj`；不保留伪装成纯 Z 的常数分支，
   不新增 MLP、投影、门控或额外元素特征。
3. 晶体几何继续进入现有 encoder；encoder、decoder、谱头、尺度头（H1 eta/gamma）及损失保持原样。
4. `legacy3`、`legacy3_const`、`mendeleev24` 原路径的模块创建顺序、行为与 checkpoint 加载兼容性保持不变。
5. 不缩小隐藏维度、不减少层数、不调整任务权重或 checkpoint 选择规则（valid `balanced_score = 0.5×MAE_edos_median + 0.5×MAE_phdos_median` 最小）。

实现入口：`run_ablation_experiments.py::build_arg_parser` 的 `--atom_feat` 增加 `z_only`；
`model/transformer.py::Transformer` 构造与 forward 按模式分支；`utils/atom_feature.py::AtomFeatureEncoder` 显式拒绝 `z_only`，
防止误入性质表读取路径（z_only 不应经过性质表读取）。

## 3. 随机数消耗事实（预先记录）

删除 `num_emb_encoder`（Linear 3→512）、`num_norm`、`fuse_proj`（Linear 1024→512）会改变初始化时随机数的消耗：
构造期各模块初始化抽签以及 `Transformer._reset_parameters` 按模块创建顺序对所有 dim>1 参数（含 `tok_emb`）的
xavier 扫描落点都会移动。同 seed 下 z_only 与 A100 **不**共享参数初值逐值相同（实测连 `tok_emb` 与 encoder 权重都不同）。
本实验评价**简化架构整体效果**，不把差异单独归因于三项性质。同模式（z_only 自身）同 seed 重新初始化应可复现，这是训练前检查项之一。

## 4. 训练前检查（工具：`tools/eval/eid_zonly_preflight.py`）

- z_only 构造后 `num_emb_encoder/num_norm/fuse_proj` 不存在（为 `None` 且不进 `state_dict`），其余模块键齐全；
- forward 中 encoder 实际输入逐值等于 `atom_norm(tok_emb(atom_idx))`；
- monkeypatch 性质表读取为异常后，z_only 构造与前向仍成功；`AtomFeatureEncoder(feat='z_only')` 抛错；
- 双谱与 eta/gamma 输出形状 `edos[4,128]/phdos[4,64]/eta[4,2]`，正反向数值有限；
- 同模式同 seed 初始化可复现；实测可训练参数量相对 A100 减少恰为 **527,872**（`Linear(3,512)=2048` + `LayerNorm(512)=1024` + `Linear(1024,512)=524,800`）；
- 旧模式行为与加载兼容性：`legacy3`/`legacy3_const`/`mendeleev24` seed-42 初始 `state_dict` 哈希与改动前快照逐值一致；
  A100/B100 checkpoint 可 `strict=True` 载入各自模式；z_only 不能载入 legacy3 checkpoint（应显式失败）；
- Q1 train/valid 数据、划分、网格与训练配方与 A100 一致；有效配置 FP32、dropout=0.05、`skip_test_eval=true`、`init_ckpt=''`、pair_aux 关闭；
- tag `_eidzonly100_s42` 对应目录、history、checkpoint 未被占用（防 runner 自动续跑）。

溯源：manifest `results/eidzonly100_s42_manifest.json` 记录命令、计划 CLI、数据/划分/性质表指纹、代码版本与修改文件哈希、
检查结果；训练后 `--finalize` 补记 best/latest checkpoint SHA-256 与 epoch 身份。manifest 是溯源记录，不代替检查本身。

## 5. 运行与监控

```bash
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat z_only \
  --skip_test_eval --tag _eidzonly100_s42
```

从零训练，不加载 A100、B100 或旧 B7 权重。预算约 5.3 GPU 小时（以 A100/B100 实测约 189.7 s/轮、峰值约 7140 MiB 估计），
产物：`output/ablation_m1_eidzonly100_s42/`、`results/history_m1_eidzonly100_s42.csv`、`results/train_m1_eidzonly100_s42.log`。
每约 5 轮监控 epoch、训练损失与分量、valid 指标、最佳 epoch、耗时、显存与产物更新；启动、25/50/75/100 轮及最终判读同步。
遇 NaN/Inf、OOM、异常退出、配置错位或意外续跑，保留记录并报告，不自动改参数、重启或追加训练。

## 6. valid 判读与采用标准

判读工具：`tools/eval/eid_zonly_valid_verdict.py`（独立产物名，不覆盖 A100/B100 的 manifest、逐样本结果或判读文件）。
只用 Q1 valid，不构建或使用 test。选点规则保持 `balanced_score`（已知其受原始 MAE 尺度影响，本轮记录该限制，不改规则）。
主结果用各自 best，另报第 100 轮。Z100、A100、B100 同 valid 样本、同 oracle/blind 公式；A100/B100 逐样本结果在身份、
样本 ID、配置与指纹核对通过后复用。报告四项中位 R²（eDOS/phDOS × oracle/blind）、四项失败率（逐材料 R²<0 比例）、
best epoch、balanced_score、参数量与运行成本；Δ = Z100−A100 及 2,000 次逐样本配对 bootstrap 95% 区间；
学习曲线对齐 25/35/50/75/100 轮；固定 320 对同组成材料的预测谱差 TV 与对比误差（结构响应诊断，不单独决定采用）。

**工程采用标准（单种子临时 baseline）**：相对 A100，四项中位 R²下降均 <0.02，四项失败率上升均 <1 个百分点，
且实现、配置与数值检查全部通过。达标则登记为「单种子临时 baseline」并保留 A100/B100 历史对照；未达标完整报告、不改门槛、不启动下一轮。
bootstrap 区间与 seed 波动边界如实说明；达标不表述为严格等价、稳定非劣或「三项性质永远无用」。
