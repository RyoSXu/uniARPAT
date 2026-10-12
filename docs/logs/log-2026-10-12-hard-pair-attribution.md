# 2026-10-12 — C1 小样本拟合困难材料对归因诊断

## 范围与协议

紧接[小样本拟合日志](log-2026-10-12-local-structure-small-fit.md)的两对几乎学不动的材料，
本轮做四项只读/短诊断：对应性核对、C1 输入可区分性、阶段表示差异、三损失梯度几何。
只用修复版 `crystal_q1_v1` 的 train、seed 42；不评 valid/test，不改模型公式、数据划分、
原 checkpoint，没有 Git 提交。焦点材料对为困难对 `mp-aaaaaafe / mp-aaacevps`（H1，单原子 Al，
candidate skill 0.027）与 `mp-aaabipws / mp-aaabjbnl`（H2，Cl16Zn4Rb8，skill 0.016）。

参照对按「candidate 臂谱差误差减少至少 80%（skill≥0.8）且与困难对可比」选取：
R1 `mp-aaabijzd / mp-aaacfxzp`（Rb4 单元素，target TV 0.172，skill 0.838）、
R2 `mp-aaaaaity / mp-aaabkqhp`（F12K4Ca4 氟化物，target TV 0.0446 ≈ H2 的 0.0467，skill 0.840）。
两对 control 臂 skill 0.664/0.674，一并报告；若按"两臂都 ≥80%"的严格口径，候选集是
`mp-aaaaalsd/mp-aaacrxgg` 等 5 对，本轮未深析，全部 16 对的输入指纹已落盘备查。

**成本**：输入/来源核对约 41 s（CPU）；GPU 合计约 8 分钟（隔离复现两臂各 600 步 197.1 s +
198.8 s，校准与分析约 30 s），低于 15 分钟预算。逐阶段证据含完整原子状态与逐能量 token。

**隔离复现声明**：原 600 步诊断的临时权重未保存。本轮按原选样（seed 42 均匀组成组）、
原初值（best ep34，`initial_state_hash` 等于归档值）、原损失与 600 步预算隔离复现并保存
诊断权重 `diagnostic_state_{control,candidate}_s600.pth`。**本文"拟合后/复现 s600"指这次
协议忠实复现的权重，不是归档运行（其权重不存在）的权重。**复现直接使用归档校准权重
λ=3.677382707595825；独立重校准三次得 3.6773810–3.6773824（差 ≤1.7e-6，仅记录数值波动，来源未定位）。

## 复现保真度（限定一切拟合后结论）

同协议、同种子、同 RNG 起止摘要下，复现轨迹与归档轨迹在 step 0 一致到 1e-9，随后漂移：
step 100 约 1e-4–5e-3，step 400–600 最大 0.023–0.029；末端 `pair_error_mean` 差
0.0111（control，0.0938 vs 0.0827）／0.0080（candidate，0.0588 vs 0.0508），两臂方向与
量级复现（candidate 仍明显优于 control、误差仍从 0.2294 下降）。`final_state_hash` 与归档
不同。**事实：本次同协议复现没有逐值重现归档轨迹。**GPU 原子加序的非确定性是可能解释，
没有用确定性重放或逐算子对照定位漂移来源，不能把这一解释写成已证实原因。
方向性观察保留，逐数值结论受复现漂移限制。

## 任务 1 — 材料 ID／结构／标签／归一化／能量网格对应

- 选样逐值等于归档 `selected_train_pairs.csv`；32 材料的 eDOS/phDOS 目标与归档
  `targets.npz` 逐值差 0.0；SumNorm 行和 1±1e-7，无负值 bin；能量网格等于
  `grids.json` E0/P0 边界中点（128/64 bins，差 0.0，批内各行相同）。
- 材料 ID ↔ train 行 ↔ 结构 ↔ 标签对齐逐条核对。发现并修正本轮工具自身一处行映射缺陷：
  dataset 行经 `structure_indices` 映射到存储行（train 视图隔离 2 条质量不合格记录），
  焦点材料 store_row = train_row 或 train_row+2，逐材料登记。
- 标签来源可查项：eDOS 任务 ID（`mp:aaadqxse` 等，任务索引 `delta_edos_index.parquet`）、
  MP 数据库 2026.04.13、抓取 2026-09-12、run_type=GGA；有原始 DOS 记录者无自旋向下通道
  （推断为非自旋极化，属推断非字段）。**未知项：磁态、kpoints、cutoff、smearing、泛函细节、
  弛豫状态，8 个焦点材料全部未知。**
- 标签-结构对应：8 个焦点材料的 DOS 任务 nsites/物种与存储结构全部一致；但 DOS 任务自身
  体积/原子与存储结构存在差异（H2 两端 1.69/1.57 Å³/atom，R1 左 2.73，H1 右 0.59，
  H1 左 0.0000），有原始记录者边多集合与存储结构不等。按既有先例登记事实、不判标签错误。
- phDOS 来源对内不一致：H1 两端均 jarvis（`ph_weight` 0.5）；H2 左 mp/pheasy、右 phonondb。
  后续代码复核确认：`utils/dataset.py` 未把 `ph_weight` 放入 batch，`model/model.py` 的
  基础 phDOS 损失按统一 `lambda_ph=1.0` 加权。这不证明来源差异造成失败。

## 任务 2 — C1 输入可区分性（周期邻居、元素、距离）

C1 实际消费 6.0 Å 内全部周期镜像的（元素对，距离）边多集合与 16 通道 RBF。逐对比较：

| 对 | 边数 L/R | 多集合相等 | 平均\|Δd\| | 距离直方图 TV |
| --- | ---: | --- | ---: | ---: |
| hard_H1 | 54/58 | 否 | 不可配对 | 1.00（完全不相交，fcc vs bcc 原胞） |
| hard_H2 | 728/728 | 否 | 0.021 Å（最大 0.097） | 0.19 |
| ref_R1 | 48/36 | 否 | 不可配对 | 1.00 |
| ref_R2 | 1016/992 | 否 | 0.05–0.45 Å 逐元素对 | 0.29 |

**H1 的 C1 输入强可区分**（单原子 Al 的 fcc/bcc 原胞，54 对 58 条镜像边、距离集合不相交），
"输入太相似"不成立。**H2 的输入只微弱可区分**：逐元素对边数完全相同，平均距离差
0.021 Å；该距离大小本身不证明有效信号不可学习。原子重排与幺模等价晶胞变换下 C1 边多集合不变：8/8 焦点材料
（全部 32 材料）通过；完整模型输出的等价晶胞/超胞表达仍未验收，本轮不扩大该声明。

## 任务 3 — 局部／全局／decoder／输出的差异流（完整张量）

`stage_tensors.npz` 保留 8 材料 × 3 态的完整原子状态 [N,512]、逐能量 decoder token
（eDOS [128,512]／phDOS [64,512]）与预测谱；下表 rel = 逐特征差异 RMS／逐特征表示 RMS
（原子态用最优匹配，decoder token 沿能量轴对齐）。pooled 摘要只作参考，不替代完整张量。

**2026-10-12 复核更正**：原工具把逐原子欧氏距离除以逐特征 RMS，导致原子态 rel 比
decoder 的口径大 `sqrt(512)=22.6274` 倍。现已修正 `atom_set_diff`，用保存的完整张量重算
下表，不重训、不改原证据目录。统一缩放不改变原子的匹配。原归档 `model_report.json` 保留
错误数值供追溯；新代码使用 `per-feature-rms-v2`，距离、最大距离、Chamfer 和 pooled RMS
也统一到逐特征单位。**旧表不能用于定位 global→decoder 瓶颈。**更正证据另存于
[损失隔离日志](log-2026-10-12-local-structure-loss-isolation.md)所列新目录。

| 态／对 | atom_local rel | atom_global rel | decoder eDOS token rel | 输出 pred_tv / target_tv | skill / cos |
| --- | ---: | ---: | ---: | ---: | ---: |
| 起点 H1 | 0.00895 | 0.02483 | 0.02077 | 0.001 / 0.263 | 0.00 / 0.08 |
| 起点 H2 | 0.00219 | 0.00478 | 0.00420 | 0.001 / 0.047 | 0.00 / 0.01 |
| 起点 R1 | 0.02884 | 0.04312 | 0.03404 | 0.002 / 0.172 | 0.00 / 0.14 |
| 起点 R2 | 0.03874 | 0.10084 | 0.10582 | 0.011 / 0.045 | 0.02 / 0.10 |
| 复现 s600 cand H1 | 0.01347 | 0.12466 | 0.07401 | 0.016 / 0.263 | 0.02 / 0.14 |
| 复现 s600 cand H2 | 0.00235 | 0.01147 | 0.00719 | 0.001 / 0.047 | 0.02 / 0.82 |
| 复现 s600 cand R1 | 0.05334 | 0.82480 | 0.65385 | 0.170 / 0.172 | 0.85 / 0.99 |
| 复现 s600 cand R2 | 0.04458 | 0.29176 | 0.20867 | 0.051 / 0.045 | 0.84 / 1.00 |

- 起点四对的预测谱差都≈零响应，但**表示差异非零，困难对不例外**。这只能排除所检查
  完整表示完全相同，不能证明它包含可预测谱差的信息，更不能证明可迁移性。
- 拟合后参照对把差异经全局层放大并读出到逐能量 token（R1 token rel 0.034→0.654），
  输出谱差与目标一致。H1 的 global/token rel 为 0.125/0.074，输出仍近零响应。
  不同层的坐标、归一化和表示用途不同，即使统一距离单位，差异范数变小也不等于
  有效信息丢失，**无法据此定位 decoder 瓶颈**。H2 的 token rel 为 0.007，方向 cos 0.823
  但幅度只有目标 2%：
  方向部分正确、幅度塌缩。
- control_s600 形状相同（H1 skill 0.032、H2 0.001），困难模式不依赖显式谱差监督。

## 任务 4 — 三损失在局部／全局／decoder 参数上的梯度

eval 模式、FP32，对 `sumnorm_klw` eDOS 基础损失、`edos_pair_contrast_loss` 谱差损失、
phDOS 基础损失分别 `autograd.grad`；参数组：atom_entry / local / global / decoder / head
（全参数被且仅被一组覆盖，工具内断言）。下表为逐对批次（pair/e = 谱差梯度范数／eDOS 基础
梯度范数；cos 为组内梯度方向余弦）：

| 态／对 | 组 | pair/e | cos(e,pair) | cos(pair,p) |
| --- | --- | ---: | ---: | ---: |
| 起点 H1 | local | 0.042 | **−0.52** | **−0.54** |
| 起点 H2 | local | 0.009 | −0.26 | −0.14 |
| 起点 R1 | local | 0.022 | **+0.33** | +0.14 |
| 起点 R2 | local | 0.525 | +0.22 | −0.38 |
| s600 cand H1 | local | 0.289 | **−0.97** | **−0.95** |
| s600 cand H2 | local | 0.170 | −0.49 | −0.82 |
| s600 control H1 | local | 1.068 | **+0.99** | **+0.99** |
| s600 control H2 | local | 0.075 | +0.03 | −0.28 |
| s600 cand R1 | local | 8.81 | +0.12 | +0.77 |
| s600 cand R2 | local | 8.21 | **+0.88** | +0.17 |

- 起点：困难对的**未加权**谱差梯度比基础 eDOS 梯度弱 25–100 倍（H2 最弱，pair/e≈0.01）；H1 还与
  eDOS/phDOS 基础梯度反向。R1 起点同样弱但方向兼容（+0.33），R2 信号强（pair/e≈0.5）。
- 拟合后：参照对谱差梯度主导（pair/e 2–9）且与基础损失兼容；H1 即使梯度变大也几乎完全
  反向（cos −0.97/−0.95）。实际乘 λ=3.67738 后，candidate s600 的 local pair/e 为
  H1 1.064、H2 0.624；不能沿用起点的「弱 25–100 倍」描述终点。
  **关键反例：control s600 的 H1 仍失败（skill 0.032），但余弦为 +0.989/+0.994。**
  已观察到的冲突不是 H1 失败在所有状态下的共同特征，不能据此确认根因。
- 全批（32 材料）拟合后 cos(e,pair) 为负（local −0.67）：显式谱差监督与单材料拟合存在
  权衡，与 10-12 候选臂单材料 eDOS 谱形 TV 变差（0.051→0.088）一致。
- 观察：H1 两材料的 phDOS 梯度远大于 eDOS 梯度（起点 |p| 4.5–6.8 vs |e| 0.14–0.28，
  jarvis 标签）。这只比较分量梯度尺度，不证明实际 AdamW 更新由 phDOS 主导。
- 这些是 eval 模式的三个分量梯度，未包括 H1 eta/gamma 监督，也未考虑 dropout 和
  AdamW 动量/预条件；不是完整训练目标或实际更新方向的因果检验。

## 判读与下一步

**得到支持**：H1 的径向输入明显不同，H2 的径向距离更接近；两对完整表示都不完全相同。
部分参照对已能拟合谱差，困难对仍失败。两对在已检查状态下的梯度大小/方向不同，
提供后续干预线索，但**未证明它们失败的机制不同、decoder 有瓶颈或梯度冲突是根因**。

**不能排除**：(1) 标签-结构对应问题（体积差、phDOS 来源不一致、磁态与计算条件未知）是
困难来源之一；(2) decoder 读出/表达限制 vs 600 步固定 lr 的优化不足——冲突是复现 s600 态
的性质，更长训练或不同权重下未必持续；(3) 轨迹级 GPU 非确定性（drift 1e-2）对逐数值结论
的污染；(4) 单 seed、单起点。**梯度冲突与拟合失败同时出现，不证明冲突是根因**（也可能是
"拟合不动→残差结构产生冲突"的反向因果）。

**下一项已由用户确认**：同一 16 对/32 材料、共同 best ep34 初值、相同 AdamW/lr/dropout/RNG，
比较原完整联合损失与仅谱差损失，各 600 步，所有参数保持可训练，不同时冻结 encoder 或
缩小材料集。若困难对改善，只支持此次去掉其他目标对拟合有帮助；不能定位哪一损失、
证明梯度冲突是唯一原因或证明泛化。若仍失败，表达/优化/预算/标签对应等解释均保留。
固定使用 seed 42；未另开重复训练。该对照随后已完成，两对仍失败，未确认根因；
实施与结果见[损失隔离日志](log-2026-10-12-local-structure-loss-isolation.md)。

## 实现与复核

- `tools/eval/local_structure_hard_pair.py`（`--phase inputs|model`，`--reuse-states`）：
  复用 `select_pairs`/`evaluate`/`new_model` 与既有谱差损失；新增 `edge_arrays`/`compare_edges`
  （C1 边多集合指纹与逐对比较）、`FullStageExtractor`（完整原子态与逐能量 token 钩子）、
  `gradient_geometry`（分 loss、分参数组的梯度范数与方向）、`reproduce_arm`（隔离复现并
  保存诊断权重、逐步对照归档 trace）。
- 执行中修正两处本轮工具自身缺陷并留痕（旧输出在 `/tmp/opencode/hard_pair_backup/`）：
  能量网格比较口径（应比边界中点而非边界）、dataset 行到存储行的 `structure_indices` 映射。
- 新增 `tests/pipeline/test_local_structure_hard_pair.py` 6 项并接入 `check-static.sh`；
  完整检查通过：324 项测试、ruff、语法与仓库检查（隐藏 CUDA 的 CPU 口径）。
- 独立复核 `independent_verification.py.txt` → `verification.json`：边多集合比较、阶段/输出
  指标、逐对拟合行、归档 trace 差值、证据指纹共 5 组重放全部通过。
- 后续审核发现上述复核没有覆盖原子态的跨阶段单位一致性；已新增相同张量的 atom/token
  单位一致测试、特征重复不改变距离测试及非法形状测试。六组状态/困难对的 local 梯度另经
  独立逐参数 FP64 内积复算，与保存数值在 FP32 容差内一致，未做 optimizer 更新。
- 原 checkpoint、配置、train 数据指纹未变；RNG 起止摘要两臂一致；没有 Git 提交。

证据目录 `results/runs/hard_pair_diagnosis_c1_s42_20261012/`：`inputs_report.json`
（含标签来源与未知项）、`model_report.json`（复现对照、阶段差异、梯度几何）、
`focus_pairs.csv`、`pairs_input.csv`、`materials.csv`、`edge_fingerprints.npz`、
`stage_tensors.npz`、`diagnostic_state_{control,candidate}_s600.pth`、
`trace_repro_*.csv`、`predictions_repro_*_s600.npz`、`verification.json`、
`artifact_manifest.json`、`executed_tool.py.txt`。

状态：困难对有输入/表示/梯度线索，根因未定位；距离单位与因果判读已更正。
