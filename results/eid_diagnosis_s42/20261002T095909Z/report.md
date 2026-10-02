# 纯原子序号模型训练差距诊断（B100 / Z100 / ZP100）

日期：2026-10-02。执行依据：[`docs/design/design-element-identity-diagnosis.md`](../../../docs/design/design-element-identity-diagnosis.md)。
产物目录：`results/eid_diagnosis_s42/20261002T095909Z/`（正式运行）；
`results/eid_diagnosis_s42/20261002T093859Z/` 为一次交叉核对中止的尝试，按规则原样保留。
诊断工具：`tools/eval/element_identity_diagnosis.py`（新增，未改动任何生产路径）。

**一句话结论**：纯 Z 两臂的落后**在相同训练样本上就已存在**（拟合差距，非纯泛化问题），
优化器、梯度、参数更新均无机械缺陷，也**不缺元素信息**（B100 的性质分支本来就是常数）；
差距与“B100 入口参数化 + 全体共享参数初值抽签”的混合路径相伴，而这些差异的可见痕迹
在最后一层原子状态的摘要统计上基本收敛——现有证据**尚不能定位根因**，只把解释面收窄到
“入口/架构参数化 vs 共享初值轨迹”这一对，并给出一个最小验证对照（第 7 节）。

---

## 1. 执行与核对（事实）

- 三阶段均完成：A（同口径评估）、B（初值/参数/优化器/当前梯度）、C（表示/注意力）。
- 身份核对：三臂 manifest / `config_used.yaml` / checkpoint SHA-256、split、数据 manifest、
  周期表指纹全部与历史记录一致；当前源码独立重建的 seed-42 初始 `state_dict` 指纹
  与记录**逐值一致**（B100 `efca7088…`、Z100 `4eda…`、ZP100 `4d7ae7…`）。
- 各阶段前后模型 `state_dict` 指纹不变；未调用 `optimizer.step()`/训练入口；未构建或读取 test；
  未追加 seed。旧产物哈希前后一致（manifest `old_artifacts_unchanged: true`）。
- 计算：V100-32GB，单模型驻留 GPU；固定批次 0.64 s，峰值显存 1.57 GB（评估）；阶段 A 256 s、
  B 55 s、C 92 s（实测）。
- 样本冻结：train 子集 2,048（`default_rng(20261002)`，排序）、探针 128（`default_rng(20261003)`
  自 train 子集抽取）、valid 全量 2,313；探针共 1,490 个真实原子（127 个多元素材料、1 个单元素材料）。
  评估 batch 32、探针 batch 8，`shuffle=False`、FP32。
- 已知核对限制（**未放宽容差**）：与 `results/eid_zproj_s42_valid_samples.csv` 的 24 组逐样本 R²
  交叉核对中 22 组通过预注册 `rtol=1e-5, atol=1e-6`；`ZP100_r2_phdos_oracle`（best）与
  `ZP100_r2_phdos_blind`（best）不通过（最大差 4.8e-6 / 5e-2，涉及 ≤6 个样本）。口径（公式、
  输入、批次顺序）逐项核对一致；定位实验（`valid_crosscheck_regime.json`）显示：把 ZP100/best
  作为新进程的**冷启动首遍**评估时可复现历史 CSV 至 ≤2e-7（float32 往返量级），而稳定态数值
  偏移 ≤4.8e-6——历史工具先评 best，其 best 列即冷态数值。这是 GPU 前向的 kernel 区制噪声
  （实测包络 ~8e-6），不是口径差异。另有一个近平坦 phDOS 目标样本（valid idx 1235，R²≈−2.3e6）
  的 R² 病态放大，原始差 0.05 属浮点表示噪声，按预注册 rtol 通过。

## 2. 阶段 A：同口径拟合与泛化（事实）

数值为逐材料中位（损失）/中位 R²（括号内失败率 R²<0）；`samples.csv` 26,166 行、`metrics_phase_a.json`。

| split | 臂 | 快照 | L_e | L_p | L_eta | L_total | eDOS oracle | eDOS blind | phDOS oracle | phDOS blind |
|---|---|---|---|---|---|---|---|---|---|---|
| train | B100 | latest | 0.0922 | 0.0118 | 0.00154 | 0.1124 | +0.749(0.2%) | +0.729(1.3%) | +0.990(0.0%) | +0.987(0.3%) |
| train | Z100 | latest | 0.0971 | 0.0144 | 0.00170 | 0.1208 | +0.741(0.2%) | +0.716(1.8%) | +0.988(0.0%) | +0.984(0.3%) |
| train | ZP100 | latest | 0.0975 | 0.0142 | 0.00168 | 0.1207 | +0.740(0.3%) | +0.716(1.8%) | +0.987(0.0%) | +0.983(0.3%) |
| valid | B100 | latest | 0.1967 | 0.2731 | 0.00226 | 0.5459 | +0.533(9.0%) | +0.485(12.4%) | +0.717(5.9%) | +0.712(6.3%) |
| valid | Z100 | latest | 0.1979 | 0.2805 | 0.00250 | 0.5627 | +0.522(9.9%) | +0.471(12.7%) | +0.704(6.2%) | +0.697(7.2%) |
| valid | ZP100 | latest | 0.1994 | 0.2821 | 0.00240 | 0.5535 | +0.523(9.9%) | +0.479(12.7%) | +0.711(6.0%) | +0.698(6.7%) |
| valid | B100 | best | 0.1968 | 0.2680 | 0.00225 | 0.5416 | +0.537(8.8%) | +0.484(12.2%) | +0.721(5.6%) | +0.712(6.1%) |
| valid | Z100 | best | 0.1967 | 0.2770 | 0.00248 | 0.5617 | +0.526(9.9%) | +0.473(13.0%) | +0.705(6.0%) | +0.697(6.7%) |
| valid | ZP100 | best | 0.1997 | 0.2772 | 0.00240 | 0.5465 | +0.525(9.5%) | +0.481(12.6%) | +0.710(6.1%) | +0.699(6.7%) |

配对 bootstrap（2,000 次，`default_rng(20261004)`，固定顺序 split→snapshot→比较组→4 项 R²、
同格内 4 项损失均值；单一 RNG 流属口径解释，已记入 manifest/`metrics_phase_a.json`）：

- **train 子集（同训练样本）**：Z100−B100 / ZP100−B100 的四项中位 R² 差全部为负且 CI 不含 0
  （latest：−0.0085/−0.0130/−0.0026/−0.0029 与 −0.0090/−0.0131/−0.0029/−0.0033）；
  损失均值差同向（latest L_e +0.0067、L_p +0.0070、L_total +0.0142）。
- **valid**：差距放大，best 快照 Z100−B100 为 eDOS oracle −0.0113、blind −0.0111、
  phDOS oracle −0.0158（CI [−0.026,−0.005]）、blind −0.0152；损失均值差集中在 **L_p**
  （best：Z100 +0.0189、ZP100 +0.0260），L_e 差 ≈0。
- **ZP100−Z100**：多数 CI 含 0（train best 的 eDOS blind +0.0069、phDOS +0.0014/0.0019 除外），
  与既有“区间含 0、非等价”结论一致。
- 尺度路径：gamma/eta 绝对误差中位三臂几乎相同（gamma ≈0.049–0.051，eta ≈0.024–0.026）。
- 失败翻转（valid）为双向：如 latest Z100−B100 eDOS oracle “B 成功 Z 失败”65 例、“B 失败 Z 成功”
  45 例、都失败 164 例；稀有元素组（239 例）与常见组翻转率相近（2.5% vs 2.8%）；原子数分层
  无明显单侧集中（`failure_flips.csv`）。

## 3. 阶段 B：初值、参数、优化器、当前梯度（事实）

- 初值指纹三臂全部与记录一致；ZP100 严格按 Z100 共享初值 + 恒等/零偏置投影构造。
- 参数变化（init→latest，组内 RMS）：所有模块都发生了实质更新，无“冻住”的部件。
  例：`decoder` ~0.0082、`spectrum_heads` ~0.0073、各层 `rp_proj` ~0.014–0.015、
  `scale_head` ~0.014；Z100/ZP100 的 `atom_norm` 更新 RMS（0.0135）约为 B100（0.0047）的 3 倍，
  B100 的 `num_norm` 0.024。ZP100 的 `atom_proj` 更新 RMS 0.00535（初值 RMS 0.0442）。
- embedding 行分层：train 真实 Z 行（85 行）相对变化 0.10–0.12；padding(0) 与未出现行（32 行）
  约 0.015（≈纯权重衰减漂移基线），三臂相近——元素行确实在学，不是衰减假象。
  事实备注：`token_num=118`（行 0–117），哨兵 token 值 126/127 只占 `src` 前两行晶格槽位，
  从不进入 `tok_emb`；“哨兵行”类别为空属预期。
- 优化器：单参数组，逐参数核对通过（条目数=参数数：B100 203 / Z100 197 / ZP100 199），
  `betas=(0.9,0.99)`、`eps=1e-8`、`weight_decay=0.01`、`initial_lr=5e-5`；
  Adam step 统一（latest 58,500 = 100×585；best = best_epoch×585，与 manifest 一致）。
  所有参数 `requires_grad=True` 且都有一阶/二阶状态（`optimizer_state.csv`）。
- 当前 eval 梯度（128 条探针、16 批、init/latest 六快照，`gradients.csv`）：
  各项损失对各模块梯度有限、非零；latest 各臂量级相近（如 `tok_emb` L_e 0.056–0.069、
  `decoder` 0.76–0.83）。`None` 梯度全部是结构性未使用路径（L_eta 不经 decoder/谱头、
  L_e/L_p 不经尺度头、L_e 不经 phDOS 头），不是梯度缺失；非有限计数为 0。
  eDOS/phDOS 梯度余弦中位 ≈0（−0.04…+0.01），负余弦批次比例 0.3–0.8，三臂无系统差异。

## 4. 阶段 C：表示、共享偏移、注意力（事实）

入口 `atom_src` 与 6 层 encoder 输出的逐材料统计见 `activations.csv`；打分/权重见 `attention.csv`。
hook 不改变前向输出（与稳定态参照差 0.0），层前向重算与正式输出逐值一致（差 0.0）。

- 入口标量均值/方差（真实原子）：B100 best −0.069 / 1.865；Z100 −0.0001 / 0.971；
  ZP100 −0.0005 / 1.579（init：B100 1.369、Z/ZP 0.997）。
- 共享偏移 b_shared（设计公式，含 `atom_norm.bias`）：**B100 init |b|=19.1（逐维 RMS 0.844）**，
  训练后 15.8/0.70；Z100 0→0.21（RMS 0.0094）；ZP100 0→0.52（RMS 0.023）。
  跨臂方向：Z100↔ZP100 余弦 0.936；B100↔两者 ≈0.067。
- 异元素原子对余弦（多元素材料内）：B100 入口 0.277 → layer5 0.022；Z100 0.007 → −0.024；
  ZP100 0.033 → −0.025。同元素对入口恒为 1.0（设计事实，非坍缩），层深处仍 ≈0.99–1.0。
  材料内原子间方差：layer5 三臂都 ≈0.53–0.54（B100 0.528、Z100 0.541、ZP100 0.541）。
- 注意力（8 head 平均，非 padding query/key，逐 query 中心化）：几何打分标准差 / 特征打分标准差
  的中位比值 layer0：B100 0.51、Z100 0.42、ZP100 0.34；layer5：0.50–0.53（三臂接近）。
  归一化熵 layer0：B100 0.29（最大权重中位 0.72）明显比 Z100 0.46（0.53）/ZP100 0.375（0.61）
  集中；layer5 三臂 0.40–0.42。近零分母（单有效 key 查询）按标志计数、比例记 N/A。

## 5. 推断（基于以上事实，非因果证明）

1. **差距源自拟合/优化层面，泛化是放大项**：train 子集同口径损失与 R² 已落后且 CI 不含 0；
   valid 上 L_p 差（+0.019…0.026）远大于 L_e 差（≈0），失败翻转双向、共享失败尾部大。
2. **“缺元素信息”不成立**：B100 的数值分支输入是预注册常数，三臂入口的元素依赖部分都只是
   `tok_emb(z)` 的变换；B100 相对纯 Z 的入口优势只可能是**参数化/初值/常数通路**，不是元素知识。
3. **“没训练/梯度坏了/优化器漏参数”不成立**（第 3 节）。
4. **“纯 Z 任务冲突更严重”不成立**：eDOS/phDOS 梯度负余弦比例三臂相当。
5. **入口差异的可见痕迹在深层摘要统计中大体收敛**（层 5 的方差、异元素余弦、几何/特征打分比值
   三臂接近），但性能差距仍在：说明差距不是由这些摘要量承载，更像早期轨迹差异被锁定；
   B100 首层注意力更集中、入口共享偏移大，是目前唯一稳定的入口侧差异。
6. ZP100 的入口函数族可覆盖 B100 的入口族（`W_proj·LN(tok)+b_proj` ⊇ `W_Z·LN(tok)+const`），
   但其投影在 100 轮内只移动了 ~0.005 RMS、共享偏移只学到 |b|=0.52：**可表达 ≠ 可达**。

## 6. 候选解释：支持 / 反证 / 待补证据

| 解释 | 支持 | 反证 | 待补证据 |
|---|---|---|---|
| X1 纯 Z 拟合/优化不足 | train 同样本损失/R² 落后（CI 不含 0） | 优化器/梯度/参数更新无缺陷；不支持“扩容量”（ZP100 加 26 万参数无收益） | 逐轮参数/梯度轨迹（本诊断无历史轨迹） |
| X2 泛化/数据支持差距为主 | valid 差距更大、失败尾部大 | train 已有同向差距 | 多 seed 波动 |
| X3 B100 入口参数化/常数共享偏移关键 | B 入口共享偏移大（|b|=19→16）、入口方差更大、首层注意力更集中 | ZP100 入口族覆盖 B 却无收益；ZP 的 b_proj 学不动；B 的 b_shared 训练中反而变小 | 入口/架构因果对照（见第 7 节） |
| X4 共享初值抽签/优化轨迹 | Z100≈ZP100（同共享初值）；B/Z 全部共享初值不同；深层摘要收敛但差距仍在 | 单 seed，不能证明 B 初值更优 | 同上对照；跨 seed 稳定性 |
| X5 eDOS/phDOS 任务冲突 | 各臂负余弦批次 0.3–0.8 | 三臂无系统差异，Z/ZP 不更冲突 | — |
| X6 容量不足 | B 多 527k 参数 | ZP100 增参无收益，差 <1% | — |
| X7 训练/加载契约缺陷 | — | 逐参数核对全部通过 | — |

## 7. 推荐的唯一最小验证对照（本轮不实施）

**“B 架构 + Z100 共享初值”交换对照**（区分 X3 与 X4）：

- 构造：`legacy3_const`（B100 同架构，含常数分支/`num_norm`/`fuse_proj`），
  `setup_ablation_seed(42)` 后按 B100 原路径构造（数值分支与 `fuse_proj` 保持 B100 自身初值），
  再把**共享参数**（`tok_emb`/`atom_norm`/encoder/decoder/谱头/尺度头）逐值复制自 Z100 seed-42
  未训练参考（复用 `utils/zproj_alignment.py` 的复制与指纹协议）。唯一被替换的因子：共享参数的
  初值抽签。
- 控制变量：M1、seed 42、100 轮、SumNorm KL/W1/Huber、H1 eta/gamma、FP32、dropout 0.05、
  同数据顺序（`DistributedSampler(seed=0)`）、不做 test 评估、不动既有臂与门槛。
- 判据：Q1 train 子集与 valid 上同口径 L_e/L_p 与四项中位 R²（同一配对 bootstrap）。
  若该臂相对 Z100 的中位 R² 提升 ≥（B100 相对 Z100 提升）的 50%（四项中至少三项 CI 不含 0），
  支持 X3（入口/架构参数化）；若提升 ≤10% 或 CI 含 0，支持 X4（共享初值/轨迹）。
- 成本：约 190 s/epoch × 100 ≈ **5.3 GPU 小时**（与三臂历史一致），峰值显存 ~7.2 GB；
  加实现与判读各一次。这只是建议，不改变现有 baseline 门槛。

## 8. 局限与未知

- 单 seed；bootstrap 区间只覆盖固定 checkpoint 的样本重采样，不含 seed 波动与选点不确定性；
  区间跨零不证明等价。
- 无逐轮参数/历史梯度，训练轨迹只能由 init/best/latest 三点与 history 粗粒度推断。
- 交叉核对 2/24 列未过预注册容差（第 1 节），已定位为 GPU 前向区制噪声而非口径差异；
  容差未放宽，该限制如实保留。近平坦目标样本的 R² 病态（原始差 0.05）属浮点表示噪声。
- 表示/注意力只覆盖入口与 encoder 各层；decoder 与谱头内部行为未做分解。
- 共享偏移分解是参数恒等式，不是因果证明；摘要统计（余弦/方差/熵）不能单独判定新架构。
- 当前 eval 梯度不代表历史训练梯度（历史训练含 dropout、参数逐批变化）。

## 9. 产物清单

`results/eid_diagnosis_s42/20261002T095909Z/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 代码/配置/数据/checkpoint 哈希、初始指纹、样本索引与 ID 哈希、环境、实际 batch、各阶段状态 |
| `samples.csv`（26,166 行）、`metrics_phase_a.json`、`failure_flips.csv` | 三臂×两快照×两 split 逐材料损失/指标、配对 bootstrap、失败翻转分层 |
| `parameter_changes.csv`、`optimizer_state.csv`、`metrics_phase_b.json` | 参数变化（含 embedding 行分层）、优化器逐参数核对、梯度/冲突汇总 |
| `activations.csv`、`attention.csv`、`metrics_phase_c.json` | 入口/各层表示统计、层 0/5 逐 head 注意力打分与权重、共享偏移分解 |
| `valid_crosscheck_regime.json` | 交叉核对偏差来源的定位实验（冷态/稳定态 vs 历史 CSV） |
| `run.log` | 实际执行记录；`*.v1.*` 为修复 entry 聚合分母 bug 前的存档（`activations.csv`/`attention.csv` 与 v1 逐位相同） |

工具符号（`tools/eval/element_identity_diagnosis.py`，均为新增）：
`build_data`（样本冻结与 loader）、`evaluate_split/_evaluate_split_once`（同口径损失+oracle/blind 指标，
双遍取稳定态）、`build_init_models`（三臂未训练重建与指纹）、`parameter_change_rows`/
`embedding_row_categories`/`optimizer_rows`/`gradient_rows`（阶段 B）、`ActAccumulator`/
`AttnAccumulator`/`recompute_layer`/`shared_offset`（阶段 C）、`regime_check`（交叉核对定位）。
复用：`element_identity_valid_verdict.load_arm/predict 口径/paired_comparison`、
`eid_zonly_valid_verdict.load_z100`、`eid_zproj_valid_verdict.load_zp100`、
`utils/zproj_alignment.state_hash/identity_projection_`、`model/losses.sumnorm_klw_loss`、
`utils/metrics.per_sample_spectral_metrics`。未改动任何生产代码、配置、数据或旧产物。
