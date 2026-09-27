# 设计：联合边内容（候选1）资源核查与 Q1 valid-only M1×10 pilot

> 关联：[状态与计划](../status.md)（阶段 B 已完成并 park）、
> [候选取舍草案](design-model-upgrade-candidates.md)（候选1）、
> [实施与 CPU 验收日志](../logs/log-2026-09-27-joint-content-implementation.md)、
> [兼容性审查](../logs/log-2026-09-27-joint-content-flash-review.md)、
> [G2a 预检审查](../logs/log-2026-09-20-g2a-预检审查.md)、
> [G2a 阈值更正](../logs/log-2026-09-20-g2a-阈值更正.md)、
> [G2a pilot](../logs/log-2026-09-20-g2a-pilot.md)、
> [eDOS slope-loss pilot](../logs/log-2026-09-23-eDOS-slope-loss-pilot.md)。
>
> 本文只覆盖候选1的**下一阶段（资源核查 + Q1 valid-only M1×10 pilot）**，给出一个推荐方案，
> 不堆叠可选项。下文保留执行前冻结的设计与行动表；它本身不是授权记录。
> 事实以代码、冻结日志和实际文件为准；推断与待批准提案逐条标注。日期 2026-09-27，
> 仓库 HEAD `45c2e85`。
>
> **执行结果（2026-09-28）：**阶段 A 通过；阶段 B 经用户批准后完成。`joint vs control`
> 的 blind eDOS `Δmedian=−0.00055`、`Δfail=+0.043pt`，全部指标落在平局线，正式裁决
> `tie`。候选1按第 9 节 park，不复现、不进入 M1×35；详见
> [正式日志](../logs/log-2026-09-28-joint-content-pilot.md)。

## 0. 事实、推断与待批准提案

- **事实（本轮只读核对）：**
  - `run_ablation_experiments.py::setup_ablation_seed` 只在 `train_and_eval` 开头调用一次
    （第 191 行），发生在 `ConfigBuilder`/`builder.get_model()`（第 290/296 行）之前；
    模型构造后、`init_ckpt` 载入后都没有再重置 RNG。
  - `utils/builder.py::ConfigBuilder.get_sampler` 训练集使用
    `DistributedSampler(..., shuffle=True, seed=0)`；`run_ablation_experiments.py` 每个 epoch
    调用 `sampler.set_epoch(epoch)`（第 415–419 行）。
  - `run_ablation_experiments.py` 已实现 `--g2_content_mode {radial,joint}`、`--use_g2`、
    `--init_ckpt`、`--skip_test_eval`；`content_mode="joint"` 必须显式配合 `--use_g2`
    （`validate_g2_config`，第 64–80 行）。
  - `init_ckpt` 路径只消费 checkpoint 的 `['model']` 子字典（第 304–312 行）；joint 走
    `load_joint_initial_state`，radial/control 走 `strict=False`。
  - `output/ablation_m1_e9ctl/checkpoint_best.pth`：SHA-256
    `cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40`，`epoch=33`、
    `model_name=M1`、`seed=42`、无 `use_amp` 键（FP32 旧格式），含 `optimizer` 与
    `best_val_score` 键但 `init_ckpt` 不读取它们；state dict 205 键，无任何
    `encoder.g2_msgs.*` 键。配套 `config_used.yaml` 生成于 G2 开关加入前，因而没有
    `use_g2` 字段；G2 关闭由 state dict 无 G2 键与历史代码边界共同确认，不能伪称配置显式为 false。
  - 历史 `output/ablation_m1_g2ctl/checkpoint_best.pth`（ep10，205 键，无 G2）与
    `output/ablation_m1_g2edge/checkpoint_best.pth`（ep10，259 键，含径向 G2）存在，但两臂
    都不是从 B7 初始化。
  - 冻结证据文件存在：`results/edos_spectral_support_q1_valid_pairs.csv`（320 对，含
    `sample_index_a/b`、`oracle_target_tv`、`b7_predicted_tv`）、
    `results/edos_error_attribution_q1_valid_samples.csv`（valid 逐样本）、
    `results/g2_edge_audit_q1.csv`、`results/c2_1b_valid_samples.csv`、
    `data/train4ARPAT/manifest.json`（valid n=2313）。
  - 历史资源事实（G2a，见阈值更正日志）：V100 batch 32 无 OOM，稳态完整训练单步耗时
    +29.5%，pilot 实测平均每轮 +27.3%、峰值显存 +16.5%（+3.8%~+7.4% 为门禁口径）。这些是
    成本报告，不是精度门槛。
- **推断（非事实）：**候选1的联合内容函数是表达差别／归纳偏置假设；G2a 已生效却平局是强反证，
  因此本 pilot 结果更可能是平局；此推断不预判结论，只影响下一关卡顺序（见第 11 节）。
- **实施状态（2026-09-28）：**RNG 前置、资源/判决工具、CPU 合同、阶段 A 与阶段 B 均已完成；
  结果为平局并 park。未读取 test。

## 1. 项目决策与服务对象

- **本轮要服务的项目决策：**判断**联合内容函数整体**是否值得进入 M1×35 确认。
- **归因边界（冻结）：**本轮只支持或否定“该内容函数整体改版”的价值，**不得**解释成单独证明
  接收端条件（`h_i` 参与内容）或参数容量（新增 `W_i/phi1/phi2`）是根因。三臂的比较角色固定为：
  - `joint vs control`＝**采用判据**（边路径 + 内容函数改版的组合包相对 B7 等价语义）；
  - `joint vs radial`＝**内容函数整体的归因参照**（只换内容函数）；
  - `radial vs control`＝**新基线复核**（在 B7 初始化 + 新代码线下重验 G2a 平局）。
- **会改变后续安排的结果：**资源门禁只决定能否安全申请 pilot；pilot win 才能把候选1推进到
  另行设计的复现，平局/退化则 park 候选1且不自动转候选2。因此本设计满足“结果会改变模型选择或
  验证安排”的立项条件。

## 2. 两阶段顺序（不堆叠）

1. **阶段 A：资源可运行性核查（同一 V100、batch 32、同一固定批次的完整训练单步）。**
   只有阶段 A 通过才进入阶段 B。
   - 每个臂**只跑一次**完整 `train_one_step`（前向 + 损失 + 反向 + 该步优化器更新，依
     `model.model.train_one_step` 的实际定义）；**不做优化步累计训练**，不循环多步，不做
     steady-state 平均。
   - **不读取 test**，不评估任何拆分，不产生精度结论。
   - 输入为**同一固定批次**：Q1 train split、`shuffle=False` 的 batch 0、`augment` 关闭、
     batch 32；三臂逐字节相同。该批次来自既有训练池，只读 train，不读 test。
   - 报告：单步时间、峰值显存，以及相对 control（B7 等价语义）与 radial（G2a 语义）的比值，
     并对照历史 G2a 事实。单次冷启动时间只作粗略成本线索，不冒充稳态均值或精确整轮预测。
     **不编造百分比承诺，不擅设成本淘汰线。**
   - **硬停止仅限：**（a）OOM；（b）NaN/Inf 或非有限损失；（c）固定 batch 32 下无法完成完整
     训练单步。（a）–（c）记录为资源不可运行，不裁边、不缩 batch、不改 cutoff。当前没有获批的
     pilot 资源预算，因此阶段 A 完成后必须报告测量值与粗略整项成本，**暂停并由用户决定是否批准
     阶段 B**；成本高本身不得自动 park。
2. **阶段 B：Q1 valid-only M1×10 三臂 pilot。** 仅当阶段 A 通过后执行，判据见第 8 节。

## 3. 三臂同代码基线（唯一新 tag；命令未授权执行）

所有臂只在 `--use_g2`／`--g2_content_mode` 上不同；其余完全一致。

| 臂 | 角色 | 关键 CLI |
|---|---|---|
| `_jcctl` | control（G2 关，B7 等价语义） | 无 `--use_g2` |
| `_jcrad` | radial（原 G2a 内容，新基线复核） | `--use_g2` |
| `_jcjoint` | joint（联合内容，采用判据） | `--use_g2 --g2_content_mode joint` |

- 固定配方：Q1、M1、10 epoch、seed 42、batch 32、lr 5e-5、dropout 0.05、SumNorm
  KL/W1/Huber、H1 eta/gamma、FP32、**不用** `--use_amp`／`--use_bucket_batch`、
  `--skip_test_eval`。
- 三臂都从 `output/ablation_m1_e9ctl/checkpoint_best.pth`（B7 epoch 33）初始化共享参数
  （`--init_ckpt`）。
- **唯一新 tag**：`_jcctl`／`_jcrad`／`_jcjoint`（已核对 `output/`、`results/` 无同名目录或
  history 文件）。

**命令草案（阶段 B 未授权执行；`--reset_rng_after_init` 已实现并通过合同测试）：**

```bash
# control（G2 关）
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _jcctl \
  --seed 42 --batch_size 32 --lr 5e-5 --dropout 0.05 --norm sumnorm \
  --scale_mode eta --eta_sup_w 1.0 --delta_edos 0.09375 --delta_phdos 19.6875 \
  --skip_test_eval --reset_rng_after_init \
  --init_ckpt output/ablation_m1_e9ctl/checkpoint_best.pth

# radial（G2a 内容）
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _jcrad \
  --seed 42 --batch_size 32 --lr 5e-5 --dropout 0.05 --norm sumnorm \
  --scale_mode eta --eta_sup_w 1.0 --delta_edos 0.09375 --delta_phdos 19.6875 \
  --skip_test_eval --reset_rng_after_init --use_g2 \
  --init_ckpt output/ablation_m1_e9ctl/checkpoint_best.pth

# joint（联合内容）
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _jcjoint \
  --seed 42 --batch_size 32 --lr 5e-5 --dropout 0.05 --norm sumnorm \
  --scale_mode eta --eta_sup_w 1.0 --delta_edos 0.09375 --delta_phdos 19.6875 \
  --skip_test_eval --reset_rng_after_init --use_g2 --g2_content_mode joint \
  --init_ckpt output/ablation_m1_e9ctl/checkpoint_best.pth
```

长训练按项目约定用 `setsid + nohup`；本设计不授权执行。

**为什么不能用历史 `_g2ctl`／`_g2edge` 作因果对照：**（i）两臂不是从 B7 初始化，而与本轮三臂
“B7 ep33 warm-start”的起点不同，初始化差异与内容函数差异混淆；（ii）历史运行早于
`g2_content_mode` 代码线，逻辑上是另一代码版本；（iii）历史 pilot 的选型、配方与本次并非同一
运行内的成对对照。历史结果只能作背景参照，不能作因果对照。

## 4. 随机公平性：代码前置已完成

- **事实：**现 runner 在模型构造后**没有**重置训练 RNG。模块数量不同（control 无 G2；
  radial 每层 3 个 Linear 的 6 张参数 + `alpha` + 2 缓冲；joint 每层 6 个 Linear 的 12 张参数 +
  `alpha` + 2 缓冲）会消费不同数量的随机数，导致 dropout 等前向随机数流在进入训练时已经错位。
- **同时保留的事实：**训练集 `DistributedSampler` 使用自身生成器（`seed=0 + epoch`），每 epoch
  已调用 `set_epoch`，其打乱顺序与全局 RNG 无关；这一行为必须保持。
- **结论：**仅相同 `--seed 42` **不足以**保证三臂随机公平；**不得**声称相同 seed 已足够。
- **已实现的代码前置：**在 `init_ckpt` 载入完成之后、
  创建训练迭代器/首次前向之前，重跑同一套种子设置（`random.seed`、`np.random.seed`、
  `torch.manual_seed`、`torch.cuda.manual_seed_all`），使三臂在同一 RNG 状态进入 epoch 0。
  提案以显式开关（草案名 `--reset_rng_after_init`，默认关闭）实现，避免改变既有未使用该开关的
  运行轨迹；三臂必须同时开启。
- **合同测试要求（CPU、合成）：**模块数量不同的两个合成模型在 `init_ckpt` 载入 + 重置后，后续
  随机抽样逐值相同；未开启开关时行为不变；`DistributedSampler.set_epoch` 语义保持不变。
- 明确边界：joint 新增分支的**参数初值**是模型构造期随机初始化的新参数，本前置不试图使不同
  架构的新参数初值相同（这是架构差异的一部分）；本前置只对齐**训练随机数流**。

## 5. 预飞检查（三臂每次启动前，冻结）

1. **checkpoint 身份：**`output/ablation_m1_e9ctl/checkpoint_best.pth` 的 SHA-256 必须等于
   `cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40`，且 `epoch=33`、
   `model_name=M1`、`seed=42`；`config_used.yaml` 记录 `epochs=35`、`norm=sumnorm`、
   `scale_mode=eta`，且作为旧配置不含 `use_g2`；state dict 必须没有 G2 键。
2. **共享键逐值相同：**三臂构造后，state dict 中所有**非** `encoder.g2_msgs.` 键必须与 B7
   载入张量 `torch.equal` 逐值一致。
3. **G2 alpha 全零：**radial/joint 各 6 层 `encoder.g2_msgs.<l>.alpha` 全为 `0`。
4. **新增键集合符合预期：**
   - control：`encoder.g2_msgs.*` 键数为 0；
   - radial：每层恰为 `{alpha, rbf_centers, rbf_width, W_v.weight, W_v.bias, W_g.weight,
     W_g.bias, W_o.weight, W_o.bias}`（9 键 × 6 层）；
   - joint：radial 键集并上每层 `{W_i.weight, W_i.bias, phi1.weight, phi1.bias, phi2.weight,
     phi2.bias}`（15 键 × 6 层）。
5. **optimizer 无恢复状态：**新 tag 的 `output/ablation_m1_<tag>/` 目录在启动前不存在；
   启动日志不得出现 `Resumed from epoch`；B7 checkpoint 的 `optimizer` 键不进入初始化
   （`init_ckpt` 只读 `['model']`）。
6. **前向可运行：**首步前向/反向完成后，`loss/loss_edos/loss_phdos/loss_eta` 全部有限；
   radial/joint 的 `alpha.grad` 存在且有限。

## 6. checkpoint 选择（正式判决口径）

- **正式判决固定使用三臂 epoch 10 的 `checkpoint_latest.pth`**，**不使用各臂 best**：B7 的
  best 落在 epoch 33，而 pilot 只训练 10 epoch，按 best 选择会在不同臂间引入不同的隐式 epoch
  选择。`checkpoint_best.pth` 只作训练安全产物，不用于判决。
- 恢复运行只允许**同 tag、同配置**严格恢复（runner 既有恢复保护），并在日志记录中断与恢复
  点；不同 tag/配置不得混用 checkpoint。

## 7. 主判决（正式判据，Q1 valid-only）

- **样本：**全体 Q1 valid n=2313；三臂**同序样本配对**（按 valid index/`mpid` 对齐，顺序必须
  逐值相等）；2000 次 bootstrap；**绝不读取 test**。
- **失败定义：**`failure = R² < 0`（沿用项目定义）。
- **主指标：**blind eDOS 中位 R² 与失败率。
- **win（`joint vs control`，全部同时满足）：**
  1. blind eDOS `Δmedian R² ≥ +0.02`；
  2. blind eDOS 失败率增量 `< +1 pt`；
  3. oracle eDOS `Δmedian R² ≥ 0`；
  4. phDOS oracle `Δmedian R² ≥ −0.02` 且失败率增量 `< +1 pt`；
  5. phDOS blind `Δmedian R² ≥ −0.02` 且失败率增量 `< +1 pt`。
- **裁决口径：**以上**点估计**是唯一裁决门槛；bootstrap 区间只报告，不额外制造第二套准入规则。
- **辅助比较（不进入采用判据）：**`joint vs radial` 报告内容函数整体归因；`radial vs control`
  报告新基线复核。若 `radial vs control` 出现异常（越线 win 或越线退化），只作记录与提示，不
  改变本节的采用判据。

## 8. 辅助机制读出（只解释，不推翻判决，不开启新诊断链）

全部复用既有定义，三臂同口径：

1. **同组成材料对的带符号谱差预测误差：**复用冻结配对表
   `results/edos_spectral_support_q1_valid_pairs.csv`（320 对）的配对，按既有口径计算各臂的
   预测谱差 TV 及其对目标谱差的误差；不新增配对/筛选规则。
2. **谱支持距离四分位：**复用 `logs/log-2026-09-26-edos-spectral-support.md` 的 train 目标谱
   最近邻 TV 四分位分组，报告各臂组内 Δmedian。
3. **高粗糙度分组：**复用 train eDOS roughness p90 阈值 `0.351600` 的分组。

以上只解释；**不**以配对误差移动替代整体判决，**不**开启新的诊断链。

## 9. 结果 → 行动穷尽表

| 结果 | 行动 |
|---|---|
| win（第 7 节全部条件满足） | 保持默认不变；**仅提议**另行设计第 10 节的下一步（推荐 seed 43/44 复现），本设计不授权 35 或改默认 |
| 只优于 radial 但平 control | **park**；结论写“内容函数整体优于 G2a 内容，但相对 B7 仍平局”，不采用 |
| 整体平局（`|Δmed|<0.02` 且 `|Δfail|<1pt`） | **park**，默认关闭 |
| 退化／保护失败（`Δmed≤−0.02` 或 `Δfail≥+1pt` 或任一保护项越线） | **park**，记录“该预算下假设未获支持”，无新假设不重跑 |
| 数值技术失败（NaN/Inf、非有限损失） | 记为技术未完成，**不是**机制反证；排查后决定是否重跑，不作科学 park |
| 资源不可运行（OOM／固定 batch 32 无法完成单步） | 阶段 A 停止；不裁边、不缩 batch、不改 cutoff，默认关闭 |
| 预算成本需用户决定（预测整轮/整项超预算） | **暂停并请用户决定**，不得自动 park |

**非 win 一律不自动进入：**候选2、角模块、数据扩充、超参／隐藏宽度／cutoff／层数扫描、小样本
拟合。所有边界外混合结果都结束本次方案。

## 10. 训练后若 win 的下一步（推荐顺序；本轮不授权）

- **推荐：先做 seed 43/44 的 M1×10 复现，再决定是否进入 M1×35 确认。**
- **理由（基于当前证据）：**（i）候选1此前最强证据是 G2a 平局，任何 10-epoch win 都在历史平局
  尺度边缘，可能来自单 seed；（ii）三臂都 warm-start 自单个 B7 ep33 checkpoint，10 epoch 的
  适应有限，warm-start 与随机初始化的交互可能产生 seed 相关效应；（iii）单 seed 复现的算力远
  低于三臂 35-epoch 成对确认，先用它筛掉 seed 噪声更经济。
- **单 seed 局限：**seed 42 是单一实现，固定 seed 的三臂差**不**估计 seed 方差，win 可能是
  seed-42 特有；不得据此宣称泛化收益。
- **晋级规则（若走 seed 复现）：**在 seed 43、44 上按第 7 节同一 valid-only 判据只重跑 control
  与 joint；seed 42 的 radial 臂已经完成内容函数归因，复现阶段不再为它重复付费。**两个新 seed
  都必须满足完整 joint vs control win 判据**，才晋级另行设计 M1×35 确认；任一 seed 不满足即
  park 并记录单 seed 局限。具体复现设计另行成文并经用户批准，本轮不授权。

## 11. 开工前最小代码／工具（已完成）

1. **RNG 前置：**第 4 节的 `init_ckpt` 后重置、显式开关与 CPU 合同测试已完成。
2. **资源核查工具 `tools/eval/joint_content_resource_gate.py`（新，不得放 `/tmp`）：**复用
   `tools/eval/g2_resource_gate.py` 的生产配置、隔离子进程、CUDA Event 与显存统计方式，并增加
   B7 初始化和 content mode；在同一固定 Q1 train batch、batch 32、V100 上，对
   control/radial/joint 各跑**一次**完整 `train_one_step`。报告冷启动时间、峰值显存及相对比值；
   不读 test、不累计优化步、不循环，且不把冷启动时间写成稳态成本。
3. **判决工具 `tools/eval/joint_content_pilot_verdict.py`（新，不得放 `/tmp`）：**基于
   `tools/eval/edos_error_attribution.py::run_audit`（valid、`expected_epoch=10`、
   `verify_reference_r2=False`）与 `tools/eval/edos_slope_pilot_verdict.py` 的成对/bootstrap
   逻辑，输出三臂同序配对 Δmedian／Δfail 与第 8 节辅助读出；点估计裁决，区间只报告。
4. **测试与 CI：**为 1/2/3 添加 CPU 合同测试，并纳入 `tools/ci/check-static.sh`。
5. **不加**任何扫描开关（宽度固定 256、cutoff 固定 5.5、层数固定 6+6）。

## 12. 正式产物路径与验证清单

- **正式产物（`results/`）：**
  - `results/history_m1_jcctl.csv`、`results/history_m1_jcrad.csv`、
    `results/history_m1_jcjoint.csv`；
  - `results/joint_content_pilot_q1_valid.json`、三组 `*_vs_*.csv`、三臂 valid 逐样本 CSV、
    `*_auxiliary_readouts.csv` 与 `*_composition_pairs.csv`；
  - `results/joint_content_resource_gate_v100.json`、`.csv`。
- **检查点（`output/`，本地便利产物）：**
  `output/ablation_m1_jcctl/`、`output/ablation_m1_jcrad/`、`output/ablation_m1_jcjoint/` 下的
  `checkpoint_latest.pth`（判决用）、`checkpoint_best.pth`（安全用）、`config_used.yaml`。
- **开工前验证清单：**第 5 节预飞检查全部通过；第 4 节 RNG 前置已实现且合同测试通过；阶段 A
  无 OOM/NaN 且完整单步可在固定 batch 32 完成；三臂 `--skip_test_eval` 生效；tag 唯一。
- **阶段 B 完成验证：**三臂各 10 epoch 完成；判决只读 valid；三臂 epoch10 逐样本顺序逐值对齐；
  bootstrap 2000 次可复现（固定种子）。

## 13. 明确不处理的范围

- 不改业务代码、测试、结果、`output/`、数据、依赖；不改 `status.md`、`index.md`、
  `decisions.md`。
- 不运行训练、GPU、真实数据前向、缓存重建或 test 评估；不实现第 11 节工具。
- 不进入候选2／角模块／数据扩充／超参扫描／小样本拟合；不因本轮结果自动改默认。
