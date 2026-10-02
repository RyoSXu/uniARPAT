# 元素身份对照实验：开发、运行与判读协议

状态：**seed 42 的 A100/B100 已实施并完成判读**；seed 43/44 未执行。本页保留当时的实验协议，
结果见[运行与判读记录](../logs/log-2026-09-30-element-identity-control.md)。
当前用户已固定 ZP 用于后续结构研究，见[元素初始化说明](design-element-initialization.md)；
历史协议不构成追加训练的授权。

## 1. 要回答的问题

在当前 Q1/M1 双谱配方中，可学习的原子序数表示已经存在。问题是：额外输入的原子质量、原子半径、电负性三项**显式元素先验**，是否在固定训练预算下改善 eDOS 谱形，并且不以损害 blind eDOS 或 phDOS 为代价？两臂均从零训练 100 轮，记录完整学习曲线，区分早期收敛差异与 100 轮预算内的最佳结果。

三项性质是原子序数 `Z` 的确定函数。实验检验的是其输入方式带来的优化与泛化效果，不能证明这些性质含有 `Z` 之外的信息，也不能证明可学习 embedding 实际学到了哪种化学规律。下文的「身份臂」是**保留数值分支、消除其中元素间差异**的操作定义；它不是删除分支后的精简模型。

## 2. 经仓库核对的事实

| 事项 | 事实与依据 |
|---|---|
| 元素输入 | `datasets/dataset.py::get_elements` 读取原子序数缓存；`model/transformer.py::Transformer.forward` 去掉前两个哨兵槽后，以 `tok_emb(Z)` 和 `AtomFeatureEncoder(Z)` 两路编码，各经 LayerNorm，再拼接并由 `fuse_proj(1024→512)` 融合。 |
| 当前性质表 | `utils/atom_feature.py::PeriodicTable` 读取 `utils/periodic_table_v2.csv` 的质量、半径、电负性。CSV 有 `Z=1..118` 共 118 行；各列填补后按全表 min–max 归一化，第 0 行补零，得到 `[119,3]`。118 个三元组均不同。半径缺失 19 个，填 `209.464646...`；电负性缺失 15 个，填固定值 `1.18`，**不是列均值**。质量没有缺失，故其误用半径填充值的 `fillna` 当前不触发。 |
| 输入索引边界 | B7 的 `tok_emb = nn.Embedding(118,512)` 接受索引 `0..117`；性质表虽含第 118 号元素，当前 B7 输入契约不能编码 `Z=118`。本实验 Q1 的最大 `Z=93`，因此不影响两臂对照。 |
| 数据口径 | `docs/data.md` 与本地 `elements_{train,valid}.npy`：Q1 train 18,706、valid 2,313；train 有 85 种元素，valid 有 82 种，valid 中没有训练未见的元素。划分来自 `index/split_v2.yaml`。 |
| B7 配方 | `output/ablation_m1_e9ctl/config_used.yaml`：M1、seed 42、35 epoch、batch 32、FP32、`legacy3`、`norm=sumnorm`、`loss_form=sumnorm_klw`、H1 `scale_mode=eta`、AdamW `lr=5e-5`。`run_ablation_experiments.py` 按 valid `balanced_score = 0.5×MAE_edos_median + 0.5×MAE_phdos_median` 保存最佳 checkpoint；B7 选中 epoch 33。 |
| 历史曲线与资源 | `results/history_m1_e9ctl.csv`：B7 eDOS valid 中位 R² 从 epoch 25 的 `0.51203` 到 epoch 33 的 `0.51498`；每轮中位耗时约 `189.2 s`，峰值显存约 `7140 MiB`（约 `7.0 GiB`）。35 轮曲线不能确定身份臂在更长训练后能否追上。 |
| 既有 100 轮配置的边界 | `output/ablation_m1_b5/config_used.yaml` 虽标记 `epochs=100`、Q1 与 `legacy3`，但其 `dropout=0.0`，不同于 B7 的 `0.05`，且未找到可核对的完整 history；它不能代替本实验的 100 轮对照。 |
| 历史实验边界 | `docs/logs/log-2026-09-14-C1pilot挂起.md` 的 C1.1 在 pre-Q、10 轮、另一种输入结构下比较了 24 维性质方案；它没有检验本页的「真实三项 vs 常数三项」问题，指标也不能与 Q1 直接比较。 |

性质表文件当前 SHA-256：`8cfba286b85f175c73431011a5c24bb8b110223ed60e81bae0357a02af9318fc`；划分文件当前 SHA-256：`8880b49652f72d5b5c5690afcc06ccaf643e3072c4789f58920256df42e43c41`。实施时应重新计算并与记录一致，不因本页写有哈希就跳过核对。

## 3. 冻结的首轮对照

| 臂 | CLI 模式 | 元素性质查找表 | 其他可训练模块 |
|---|---|---|---|
| A：性质臂 | `--atom_feat legacy3` | 现有真实 `[119,3]` 表 | B7 M1 原样 |
| B：身份臂 | 新增 `--atom_feat legacy3_const` | 第 0 行零；第 1..118 行均为同一个固定向量 `c` | 与 A 同形状、同参数数目、同初始化流程 |

**固定向量的预注册定义**：令 `F = PeriodicTable().atom_feature_map()`，即完成当前缺失填补和全表归一化后的 `float32` 张量；取 `c = F[1:119].mean(dim=0)`，当前约为 `[0.49465635, 0.39238882, 0.40330893]`。使用全表 118 行的**逐元素等权均值**，排除 padding；不要改成前 103 行、只取 Q1 元素、按样本频率加权或手写四舍五入后的常数。这个选择使线性投影前的全表均值可追溯，**不使 LayerNorm 后的激活分布与 A 相同**；不同 `c` 的方向可能改变训练轨迹。记录计算公式、实际 `float32` 值与表哈希。

在 B 中，所有非 padding 原子的数值分支输出相同，经融合后只能贡献共享向量。`feature_map[0]` 为零并不表示 `Linear` 投影后的 padding 表示为零；padding 仍按原模型掩码处理。B 保留形状与参数量，但数值分支的**有效容量和梯度**与 A 不同。若 B 变差，只能归因于本次操作整体，不能进一步断言「物理语义」是原因。后续若要检验语义，可另做固定置换表臂；若要决定是否删除分支，可另做精简架构臂。二者均不属于首轮。

两臂固定 Q1 划分、E0/P0 网格、M1 encoder/decoder/head、SumNorm 损失、H1 尺度头、batch 32、AdamW、学习率、warmup+cosine、`dropout=0.05`、FP32、数据顺序策略、无增强、无 bucket、无 mask、`--skip_test_eval`。两臂从零开始，均不传 `--init_ckpt`，使用相同 seed。`run_ablation_experiments.py` 用 `cfg.epochs` 构建 warmup+cosine：本实验两臂都运行 **100 轮调度**，不能把旧 B7 的 35 轮权重续训 65 轮当作等价的 A 臂。

首轮 seed 42，重新训练 A100（真实性质基线）与 B100（常数性质臂），各 100 轮：按 B7 耗时估计每臂约 **5.3 GPU 小时**，合计约 **10.5 GPU 小时**；每臂峰值显存约 `7.0 GiB`。当前 runner 每臂保留 best/latest 两个约 854 MB 的 checkpoint，首轮两臂约需 **3.4 GB** checkpoint 空间。若要对「稳定增量」给出跨初始化证据，再在相同协议下做 seed 43、44 的两组配对，额外约 **21.0 GPU 小时**，三组共约 **31.5 GPU 小时**。成本均是历史估计，应以实际机器监控值报告。

100 轮仍是**固定预算比较**，不是数学上的「充分收敛」保证。如任一臂到第 100 轮仍明显上升，先报告「100 轮预算内结果，收敛状态未明」；延长训练需重新确定两臂共同的预算与调度。旧 B7 `_e9ctl` 的 35 轮 checkpoint 只作历史参考；A100 是本实验重新训练的对照，A100 与旧 B7 的差异也混合了训练长度和学习率调度变化，不能单独解释为多训练 65 轮的收益。

## 4. 实现契约与前置核对

### 最小开发范围

1. `utils/atom_feature.py::AtomFeatureEncoder`：增加显式 `legacy3_const` 分支，要求 `input_dim == 3`；先沿当前 `legacy3` 路径取得 `F`，再复制并替换 `F[1:]` 为上面的 `c`，保留 `F[0]`。默认 `legacy3` 路径逐值不变。新分支不得新增随机抽样或改变可训练层的创建顺序。
2. `run_ablation_experiments.py::build_arg_parser`：把 `legacy3_const` 加入现有 `--atom_feat` choices。`utils/experiment_config.py::ExperimentConfig.atom_feat` 已是字符串字段；`model/transformer.py` 对非 `mendeleev24` 模式已使用 3 维性质层，并且只对 `mendeleev24` 把 `tok_emb` 零初始化，因此本方案通常无需改这两处。实现者应核对真实调用链，不增加多余分支。pair-aux 的冻结校验继续拒绝新模式。
3. valid 判读工具：可扩展现有工具，也可新增一个薄脚本；复用 `tools/eval/periodic_manybody_valid_verdict.py` 的逐样本 oracle/blind 计算与配对 bootstrap 逻辑，但须移除其中固定的 B7 路径、epoch 33、`use_periodic_manybody` 和「A 臂预测须等于历史 B7」断言。新训练的 A 臂不是冻结 B7 checkpoint。

`feature_map` 是普通属性，**不进入 `state_dict`**。相同键和 `strict=True` 加载不能证明 checkpoint 属于正确臂；判读工具必须先核对每臂 `config_used.yaml`、seed、有效模式、原表哈希、常数值、模型 checkpoint 的模型名与 epoch。将运行时的表指纹、代码版本和配置哈希写入实验 manifest；训练完成后补记两臂 checkpoint 的 SHA-256，判读时校验后再加载。这些记录提供溯源，不能代替实现前的表与模式检查。不要把 B7 checkpoint 当作 A 臂，也不要在同一输出目录复用不同配方。

### 实施后、训练前的检查

- 用两个新模式构造模型，检查 `feature_map` 均为 `[119,3]`，A 与现行表逐值一致，B 的第 0 行为零且第 1..118 行逐值相同；实际 `c` 与公式一致。
- 同一 seed 下两臂初始 `state_dict` 的键、形状与**可训练参数值**逐值一致，参数总量相同；检查两臂数据样本 ID 顺序一致。两臂输入不同，**不要求首 batch loss 相等**。
- 检查正反向数值有限、M1 双谱输出及 H1 头形状正确；同臂重新初始化结果可复现。确认新模式没有触发 `mendeleev24` 专属的 token 零初始化。
- 预先检查目标 tag 未有 `checkpoint_latest.pth`、history 或不兼容配置。runner 会自动续跑已有 tag；若目录已占用，先核对来源并换全新 tag，不覆盖或删除已有产物。
- 在真实运行配置中核对 `skip_test_eval=true`、`init_ckpt=''`、`use_amp=false`、`pair_aux_arm=none`、模型与超参一致，并确认数据 manifest / split / 性质表哈希。

## 5. 执行与监控

只有实现、前置检查与用户对预算/判据的确认完成后，才执行以下**命令模板**。每个 seed 使用一对专属 tag；先以 seed 42 的 A100/B100 配对实验为首轮，不因中途 valid 读数提前停训或改变配方。

```bash
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat legacy3 \
  --skip_test_eval --tag _eidprop100_s42
python3 run_ablation_experiments.py --model M1 --epochs 100 --seed 42 --batch_size 32 \
  --lr 5e-5 --norm sumnorm --scale_mode eta --atom_feat legacy3_const \
  --skip_test_eval --tag _eidconst100_s42
```

训练产物分别为 `output/ablation_m1_eidprop100_s42/`、`output/ablation_m1_eidconst100_s42/` 的配置与 checkpoint，以及 `results/history_m1_eidprop100_s42.csv`、`results/history_m1_eidconst100_s42.csv`。seed 43/44 若获准，保持配方不变，只替换命令中的 seed 和 tag 后缀。

监控时在首轮、之后约每 5 轮和收尾记录两臂进度：epoch、训练损失与各分量、valid `balanced_score`、eDOS/phDOS 中位 R²与失败率、当前最佳 epoch、单轮耗时、峰值显存、checkpoint/history 是否持续写入。按相同 epoch 展示两臂学习曲线，重点保留 25/35/50/75/100 轮读数，用来判断 B 是否较晚追上 A；**最终评估各用自己依据相同规则选出的 best checkpoint**。B7 历史曲线只用于背景参考，不能直接替代重新训练的 A100。

遇到 NaN/Inf、OOM、数据或配置错位、异常重启/自动续跑、checkpoint 与 tag 不符、test 路径被访问等情况，停止该组结果的科学判读并保留原始记录。当前 resume 路径不恢复训练 RNG 状态；任一臂中断续跑后不应声称仍与另一臂保持严格的随机流配对。需要重做时用**新 tag**成对从零重跑，不覆盖旧产物。不要以监控曲线临时调参、换常数值或改判定门槛。

## 6. valid 判读与结果语言

### 模型选择与指标

- 每臂各取其 **100 轮内 valid `balanced_score` 最小**的 checkpoint 作为主结果，同时报告第 100 轮的验证指标和最佳轮次，以便观察晚期是否追赶或退化。epoch 33 只是 B7 的历史选点，不是新实验的强制选点。选点指标与科学主指标分开报告。
- 主指标：Q1 valid **eDOS oracle 中位 R²**，`Δ = B − A`；这里的 oracle 使用真实谱总量恢复预测谱，主要比较给定正确总量下的谱形能力。同步报告 oracle 失败率（`R² < 0`）差，单位为百分点。
- 同步报告 eDOS blind、phDOS oracle/blind 的中位 R²和失败率；blind 按现有 H1 `γ/η` 公式重建。若 oracle 与 blind 方向不同，再报告 `γ/η` 误差以定位尺度路径。phDOS 与 blind 的退化会影响**是否采用 B 臂**，不会抹去 eDOS oracle 上已经观察到的事实。
- 对每个 seed，用相同 valid 样本索引对两臂逐样本 R²作 2,000 次配对 bootstrap；每次计算「B 样本中位数 − A 样本中位数」及失败率差，报告 95% 区间。该区间只反映固定 checkpoint 下的 valid 样本重采样，不覆盖训练 seed 波动、选点不确定性或未知数据分布。
- 机制诊断：按固定的 `results/edos_spectral_support_q1_valid_pairs.csv` 的 320 对样本 ID 计算目标、两臂预测的同组成 eDOS 谱差 TV，以及「预测谱差向量 − 目标谱差向量」的 TV。仅核对固定 pair 的 ID 与目标口径，不要求新 A 与历史 B7 预测相同。元素频次分层固定为「样本所含最稀有元素在 train 的原子槽出现次数 `<500`」与其余样本；当前 valid 两组分别为 239 和 2,074 条。报告两组各自差异与样本数，仅作线索。TV、元素频次分层与 embedding 相似性均不单独判胜。

### 判读规则

`docs/decisions.md` 的 `|Δ中位 R²| < 0.02`、`|Δ失败率| < 1 个百分点` 是项目**单次比较的描述线**，本页沿用作实际效应尺度；不把它称为显著性检验或严格等效性证明，也不另造缺乏依据的 `0.01` 门槛。

| 观察 | 可报告的结论 |
|---|---|
| 只有 seed 42 | 报告两臂差值、区间、学习曲线和「单次运行的初步证据」。无论差值大小，都不称「稳定增量」或「元素身份已经足够」。 |
| 三个配对 seed 的 eDOS oracle `Δ` 均 `≤ −0.02`，区间均不跨 0，且失败率没有达到反向的 1 个百分点差异 | 现有三项显式性质在**当前 Q1/M1/100 轮配方**下，呈现跨三个初始化一致的正增量证据；报告三个原始差值与范围。 |
| 三个配对 seed 的 eDOS oracle `Δ` 均 `≥ +0.02`，区间均不跨 0，且失败率没有达到反向的 1 个百分点差异 | 常数臂在该配方下呈现一致优势；这不证明「删除性质分支」也会更好。 |
| 三个 seed 的 eDOS oracle 差值及各自区间均落在 `(-0.02,+0.02)`，失败率差均在 `(-1,+1)` 个百分点内 | 三次运行均未观察到超过项目描述线的 eDOS oracle 差异；**不**据此证明性质在其他预算、架构或数据条件下无用。 |
| 种子方向不一、区间跨判定线、oracle/blind 分歧或任务间权衡明显 | 分别说明 eDOS oracle 观察、尺度与 phDOS 结果；总体采用决定标为混合或未决。先核对学习曲线、尺度头与数据分层，再提出下一步，不事后更改本轮门槛。 |

**部署保护另判**：考虑采用 B 时，其 eDOS blind 与 phDOS oracle/blind 在每个 seed 上不得出现中位 R²下降 `≥0.02` 或失败率上升 `≥1` 个百分点；若越线，记录为部署/双任务权衡，即使 B 的 eDOS oracle 主指标更好，也不直接替换现有方案。每一项仍报告原始数值和区间。任何结果都不自动修改 B7 默认配置。

本实验不能检验未见元素泛化，因为 Q1 valid 元素全部在 train 中出现；也不改变 `TransformerEncoderLayer` 中几何只进入 attention score 的路径。单元素材料的同元素原子仍共享同一初始特征，故本实验不能直接修复那一几何机制。`docs/logs/log-2026-09-28-model-accuracy-upper-bound.md` 的 `+0.00539` 仅是**只完美修复 37 条单元素样本**时全体 blind eDOS 中位 R²的反事实上界，不能当作本实验对所有材料的收益上界。已有同组成 TV 的目标约 `0.270`、B7 预测约 `0.016` 是背景观察；本实验中 TV 增大本身不代表预测更准确。

## 7. 交付记录与后续决定

交付物应包含：实现差异与前置检查结果；两臂逐 seed 的完整命令、配置、原表与 split 哈希、常数向量的实际值、代码版本、训练日志和 history；两臂 best checkpoint 的 epoch 与 valid 选点分数；逐样本 oracle/blind 指标、配对区间、同组成 pair 与元素频次诊断；异常或中断说明；最终结论及适用边界。仅把已实际运行的检查称为「已验证」。

首轮之外的两个决定由用户另作：是否追加 seed 43/44 取得跨初始化证据；若结果需要追问，是用**打乱表**检验性质值与元素的真实对应，还是用**删分支**检验精简架构。两种追问改变了问题和成本，不从首轮结果自动启动。当前不改性质表数值、Q1 数据契约或几何 encoder；test 不参与本实验的选型。

后续记录：「删分支」追问以 [`design-element-identity-zonly.md`](design-element-identity-zonly.md) 的 Z100（`--atom_feat z_only`）单候选实施，
结果见 [`../logs/log-2026-10-01-eid-zonly-baseline.md`](../logs/log-2026-10-01-eid-zonly-baseline.md)：seed 42 单次运行未达工程采用标准
（eDOS oracle 失败率上升 1.43pp 越线），A100 保留为对照，未登记 baseline。「打乱表」追问未执行。
Z100 判读指出的初始化混淆（删除分支改变随机数消耗、与 A100 不共享初值）由
[`design-element-identity-zonlyproj.md`](design-element-identity-zonlyproj.md) 的 ZP100（`--atom_feat z_only_proj`，
与 Z100 共享初值与随机流的对齐初始化）追问，结果见
[`../logs/log-2026-10-02-eid-zonlyproj.md`](../logs/log-2026-10-02-eid-zonlyproj.md)：相对 Z100 未观察到可区分改善，
相对 A100 仍败在 eDOS oracle 失败率（+1.04pp），A100 继续保留为对照。
