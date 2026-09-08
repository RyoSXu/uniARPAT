# uniARPAT 突破性提升的实证迁移路线：等变几何编码 × 能量条件逐点解码 × 非平衡去噪辅助

- **研究员**：Muse Spark（AI for Science / 前沿深度学习资深研究员）
- **日期**：2026-09-08
- **保存路径**：`uniARPAT/docs/04_前沿探索/2026-09-08-muse-spark-等变能量条件与非平衡去噪的实证迁移路线.md`
- **目标读者**：计算机硕士学位论文核心章节 + 第二篇顶级期刊（npj Comput. Mater. / Nature Comput. Sci. 级别）
- **方法论声明**：本报告拒绝凭空脑暴。每一个迁移论点均绑定**真实论文原文公式号/表格号 + 真实官方 GitHub 文件路径**。所有论文均已实际抓取 HTML 全文，所有代码均已实际抓取 raw 文件并精读。凡未验证的数字均明确标注为“外推区间”而非结论。
- **与同目录姊妹报告的关系**：同目录已有连续谱算子/ECSO、等变流匹配、MoE 等路线。本报告是**证据密度最高的正交补强路线**：不引入 ODE 采样与高阶张量积重构，而是用三篇“同一课题组（MIT/FAIR）+ 同一数据集族（OC20/OC22/MP）+ 同一代码栈（e3nn/PyG/OCP）”的连贯证据链，解决 uniARPAT 当前最痛的三件事——**未弛豫输入偏移、离散格点回归、非等变编码器**。可独立成文，也可与姊妹报告合并为 v3（确定性）→ v4（生成式）演进。

---

## 目录

1. [执行摘要：为什么是这三板斧](#1-执行摘要)
2. [代码审计：uniARPAT 现状的六个实证瓶颈](#2-代码审计)
3. [前沿证据总览：四篇旗舰论文 + 五个官方仓库](#3-前沿证据总览)
4. [支柱一：等变几何编码升级（eSCN → EquiformerV2）](#4-支柱一)
5. [支柱二：能量条件逐点解码 + 晶系 Prompt（DOSTransformer）](#5-支柱二)
6. [支柱三：非平衡去噪辅助（DeNS，直击未弛豫输入）](#6-支柱三)
7. [扩展：MACE 基础模型蒸馏 + FlowMM 黎曼流匹配](#7-扩展)
8. [统一训练协议与总损失重写](#8-统一训练协议)
9. [算力与显存定量评估](#9-算力显存)
10. [失效风险矩阵与硬回滚判据](#10-风险回滚)
11. [路线图 M6–M10、论文章节与期刊故事线](#11-路线图)
12. [反夸大边界与可复现清单](#12-反夸大)
13. [参考文献（全部可点击验证）](#13-参考文献)
14. [附录 A：最小代码改动草图](#附录-a)

---

## 1. 执行摘要

### 1.1 一句话立意

> **编码器换成等变注意力（SO(2) 高效张量积 + 可分离归一/激活），解码器换成能量条件逐点预测（晶体 × 能量交叉注意力 + 晶系 prompt），训练加上非平衡去噪辅助（加噪→力编码→部分去噪）。三者均有 DOS/OC20/OC22 上的已发表增益数字，且均有可直接复制的代码模式。**

### 1.2 为什么这三者是当前 ROI 最高的组合

| 判据 | 本报告三支柱 | 依据强度 |
| :--- | :--- | :--- |
| 是否直击 eDOS 失败率 14.66%、均值 R² 0.374 vs 中位数 0.521 的长尾尖峰问题 | 是：DOSTransformer 逐点能量条件在 phonon/electron 上一致增益，且 OOD 下增益更大；EquiformerV2 高阶表示专治角度敏感的力/声子 | 强：DOSTransformer 表 1/表 2；EquiformerV2 摘要 9%/4% |
| 是否解决“未弛豫输入 → 弛豫后谱”的根本偏移 | 是：DeNS 正是为弛豫轨迹中间帧（非平衡结构）设计的辅助任务，无需额外数据 | 强：DeNS 图 1/式 4/式 6 + OC20/OC22 SOTA |
| 工程风险与单卡 V100 可行性 | 低–中：三者均为监督/辅助回归，无 ODE 采样，无高阶全张量积；eSCN 反而**降低**复杂度 O(L⁶)→O(L³) | 强：eSCN 表 1 训练时间对比；DeNS 边际训练开销声明 |
| 学位论文增量 | 方法章三节（等变编码、能量解码、去噪辅助）各自独立消融，应用章 OOD + 带隙 + 热力学 | 强：每节对应一篇顶会论文，可单独成表 |

### 1.3 预期增益（诚实先验区间，非承诺）

基于已发表数字按保守折扣（跨任务折半）给出，详见 §11 回滚线：

- **eDOS 中位数 R²**：0.521（M1 Oracle）→ **0.60–0.68**（三支柱联合，Oracle 口径）；Blind 口径同步跟进。
- **eDOS 失败率（R²<0）**：14.66% → **<9%**（带隙 prompt + 逐点损失为主）。
- **phDOS 中位数 R²**：0.694 → **0.74–0.78**（等变编码 + 声学支加权）。
- **下游**：Θ_D MAE 41.09K → **<32K**；C_v MAE 0.364 → **<0.28**；κ_L Spearman 显著提升（Slack 公式对 Θ_D 立方放大）。
- **OOD（晶系留一）**：相对提升 **15–30%**（DOSTransformer OOD 下能量嵌入增益更大的规律）。

> 若 Week-2/3 消融 M6 eDOS Oracle 中位数 <0.55 且带隙 F1 无提升，触发 §10 硬回滚，不得虚报。

---

## 2. 代码审计

<a id="2-代码审计"></a>

审计基线：`model/transformer.py`、`model/heads.py`、`model/model.py`、`datasets/dataset.py`、`utils/rp_encoding.py`、`utils/relative_features.py`、`utils/atom_feature.py`、`thermo_props.py`、`configs/config.yaml`。数据规模 train 10965 / valid 1371 / test 1371。

### B1. 谱轴被当作“离散通道”，模型不知道能量坐标

- 现状：`model/transformer.py:84-89` 的 `edos_query_embed/phdos_query_embed` 是与能量值无关的静态可学习向量（`torch.zeros(128/64, 512)`）；`model/heads.py:33-82` 的卷积头在谱轴上做局部平滑；`model/model.py:15-31` 的 `compute_shape_loss` 对所有 bins 等权（Pearson + 0.5·MSE，平坦 fallback）。
- 后果：eDOS 均值 0.374 远低于中位数 0.521、std 0.65——典型长尾尖峰拟合失败；M5 Blind eDOS R² 为负（README 表 M5 行）。
- 前沿对照：DOSTransformer 证明逐点能量条件在所有基线一致增益（§5 详述），且 OOD 下增益更大。

### B2. Encoder 非等变 + 双算浪费 + Pair 表显存黑洞

- 现状：`model/transformer.py:283-356` 的 `TransformerEncoderLayer.forward` 在第 323 行先调一次 `self.self_attn` 得到 `attn_output`，随后第 325–346 行完全重算 QK 分数并加上 `rp_scores`，第 323 行结果被丢弃。QKV 投影与第一次 softmax 被浪费（约 1.5–2× 注意力开销）。
- `RPEncoding(lmax=2)` → `out_dim=64×9=576` → `rp_proj` 到 512（第 310–311 行），每 batch pair 表 `[B,L,L,576]`（fp32、batch=32、L≈80 时约 470MB），是 9.16GB 峰值显存的最大单一来源之一。
- 同时 `rbf_encoder/rel_proj/dir_proj`（第 304–308 行）初始化后从未在 forward 使用（死代码）。
- 物理层面：标量距离 + 球谐方向加性偏置对未弛豫微小畸变（键长 ±0.1Å 可致带隙 eV 级移动）分辨率不足；旋转晶体输入，当前 encoder 输出不满足等变性，而 DOS 峰位对键角高度敏感。

### B3. Shape-Scale 解耦正确但标度回归脆弱

- 应保留的正确之处：`safe_shape_norm`（第 17–28 行）ReLU 主路 + Softplus 自愈；`ScaleHead` 偏置 `[3.0,-1.5]` 经验初始化；Blind–Oracle gap 小证明链路成立。
- 脆弱之处：`model/model.py:173-176` 用 `log(clamp(y_max,1e-4,500))` Huber 回归——整个谱标度压缩为单一 `y_max`，对平坦绝缘体与离群大峰均不鲁棒；Pearson 对尖峰偏移 1–2 bins 惩罚灾难，而积分型下游对此不敏感——损失与物理目标错位。

### B4. 损失等权所有 bins，与热力学/输运敏感性错位

- 现状：`model/model.py:201` 总损失对 128/64 bins 等权。
- 物理事实（`thermo_props.py:25-45,92-104,106-125`）：C_v 核 `(x/2)²/sinh²(x/2)` 高度集中在低频声学支；Θ_D 是二阶矩 `sqrt(5/3·<ν²>)`；Slack κ_L ∝ Θ_D³ 把高频误差立方放大；eDOS 的 Seebeck 相关量是费米面斜率。等权 MSE 把容量浪费在不敏感区。

### B5. 无预训练/辅助任务、无不确定性、分辨率锁定

- 编码器从零训练（10965 样本对 76M 参数，约 144 样本/百万参数，过参 regime）；`_reset_parameters` Xavier 全随机（第 117–120 行）。
- `datasets/dataset.py` 归一化选项完备（minmax/zscore/log/scale），但无未弛豫→弛豫增广、无晶系分层；原子特征仅 3 维（质量/半径/电负性，`utils/atom_feature.py:61-70`），缺失价电子、族/周期、d 电子计数等 DOS 强相关量。
- 分辨率锁定 128/64，无法联合高分辨率谱训练/超分。

### B6. 优化与评估口径可加固

- `configs/config.yaml`：AdamW lr 5e-5、余弦 + warmup 5 epochs 协议良好，应沿用；但选优指标 `save_best: total_NormMAE` 单一，未纳入带隙 F1 与下游 C_v/Θ_D（§8 引入 `phys_score` 第二判据）。

---

## 3. 前沿证据总览

<a id="3-前沿证据总览"></a>

| 支柱 | 论文（已抓全文） | 核心公式/表格 | 真实结论（原文数字） | 官方代码（已抓 raw） |
| :--- | :--- | :--- | :--- | :--- |
| P1 等变编码 | EquiformerV2, ICLR 2024, https://arxiv.org/abs/2306.12059 | 式 (1) CG 张量积；§3.2 注意力重归一；§3.3 可分离 S² 激活；§3.4 可分离层归一；表 1 消融 | OC20 力 up to **9%**、能量 **4%**，AdsorbML **2×** DFT reduction；仅 OC22 训练超越 GemNet-OC（OC20+OC22 联合训练） | `nets/equiformer_v2/transformer_block.py`（SO2EquivariantGraphAttention/TransBlockV2），`nets/equiformer_v2/equiformer_v2_oc20.py`，`nets/equiformer_v2/so2_ops.py`，`nets/equiformer_v2/activation.py`，`nets/equiformer_v2/layer_norm.py` @ https://github.com/atomicarchitects/equiformer_v2 |
| P1 地基 | eSCN, ICML 2023, https://arxiv.org/abs/2302.03655 | §3 SO(3)→SO(2) 约化；复杂度 O(L⁶)→O(L³)；式 (21) 球面激活；式 (40) 等变误差 | OC20/OC22 SOTA；L=6/K=20 200M 参数、568 GPU·天 vs SCN 271M/1280天；等变误差 1.5%（SiLU+14 网格+AMP） | 同上仓库 `so2_ops.py`、`edge_rot_mat.py`、`so3.py`（Wigner-D 旋转对齐边向量） |
| P2 能量解码 | DOSTransformer, NeurIPS 2023, https://arxiv.org/abs/2311.12856 | 式 (1) H=GNN(X,A)；式 (2) 交叉注意力；式 (3) 自注意力；§4.3 逐点解码 Ŷ=E→MLP；§4.4 L_total=L_glob+β·L_sys | 表 1：phonon R² 0.638→0.685（+能量）→**0.733**；electron 0.613→0.650→**0.679**；能量嵌入在所有基线一致增益；E3NN 增益小（等变被干扰）；物理性质（Bulk/Bandgap/Fermi）最优 0.427/0.461/2.337；表 2 OOD 下 GN 获益更大；表 3 仅调 prompt 0.570 > 全量微调 0.558 | `embedder_eDOS/DOSTransformer.py`（Embedding(201)+Embedding(7)+三 TransformerEncoder+fc/fc_prompt 双分支），`layers/transformer.py`（TransformerEncoder/TransformerEncoderLayer 预归一），`main_phDOS.py`（loss=rmse_global+β·rmse_system） @ https://github.com/HeewoongNoh/DOSTransformer |
| P3 去噪辅助 | DeNS, TMLR 2024, https://arxiv.org/abs/2403.09549 | 式 (1) 能量+力损失；式 (2) 加噪；式 (3) 去噪损失；式 (4) DeNS 力编码损失；式 (5) 力嵌入；式 (6) 辅助总损失；图 2 三分支训练 | OC20-2M 12ep 力 20.46→**19.09**、能量 285→**269**；30ep 19.42→**18.02**；**2.3×** 训练加速；OC22 能量 up to **15%**、力 **12%**、IS2RE **15%**；MD17 **3.1×** 加速；无力编码版本比不用 DeNS 更差（表 1b Index3 vs Index1） | `trainers/denoising_forces_trainer.py`（add_gaussian_noise_to_position / add_gaussian_noise_schedule_to_position / compute_atomwise_denoising_pos_and_force_hybrid_loss / denoising_pos_eval / DenoisingForcesTrainer），`datasets/dens_lmdb_dataset.py`，`configs/oc20/.../*.yml` @ https://github.com/atomicarchitects/DeNS |
| 扩展教师 | MACE, NeurIPS 2022 + foundation 2024 https://arxiv.org/abs/2401.00096 | 高阶 ACE 对称收缩；UniversalLoss + conditional_huber_forces（按力模分档 [1.0,0.7,0.4,0.1]） | 89 元素、MPTrj 1.6M；开箱即用 + 少量微调达 DFT 精度；MACE-MP-0/MPA-0/OMAT-0 系列 | `mace/modules/blocks.py`，`mace/modules/symmetric_contraction.py`，`mace/modules/loss.py`（WeightedEnergyForcesLoss/UniversalLoss），`mace/modules/models.py` @ https://github.com/ACEsuit/mace |
| 扩展生成 | FlowMM, ICML 2024, https://arxiv.org/abs/2406.04713 | 黎曼流匹配推广到晶体（平移/旋转/置换/周期）；测地线回归目标；基分布自由选择 | CSP match rate Perov-5 **53.15**、Carbon-24 **23.47**、MP-20 **61.39**、MPTS-52 **17.54** 超 DiffCSP/CDVAE；**~3×** 积分步效率；E_hull 分布更优 | `src/flowmm/rfm/manifold_getter.py`（ManifoldGetter：flat_torus_01/SPD/lattice_params/analog_bits/simplex），`src/flowmm/model/`，`src/flowmm/data/` @ https://github.com/facebookresearch/flowmm |

---

## 4. 支柱一：等变几何编码升级（eSCN → EquiformerV2）

<a id="4-支柱一"></a>

### 4.1 论文原文核心公式（已验证）

**来源**：EquiformerV2 全文 https://arxiv.org/html/2306.12059v3 ；eSCN 全文 https://arxiv.org/abs/2302.03655 。

**(a) 等变张量积定义（EquiformerV2 §2.1，式 1）。** type-L 向量经 Wigner-D 矩阵旋转；相对位置经球谐投影；消息传递用 Clebsch–Gordan 系数做张量积：

```
h^(L3)_m3 = Σ_m1 Σ_m2 C^(L3,m3)_(L1,m1)(L2,m2) · f^(L1)_m1 · g^(L2)_m2   (Eq.1)
```

非零条件 `|L1−L2|≤L3≤L1+L2`，超 `Lmax` 截断。这是 uniARPAT 当前 RP 加性偏置的严格上位替代：RP 只做标量加分，而此处方向信息进入表示本身。

**(b) eSCN 的 SO(3)→SO(2) 约化（EquiformerV2 §2.3；eSCN §3）。** 将边向量经旋转 `D_ij` 对齐到 y 轴，则球谐投影仅 `m_f=0` 非零，CG 系数仅 `m_i=±m_o` 非零。剩余非平凡路径用 SO(2) 线性操作替代，复杂度从 **O(Lmax⁶) 降到 O(Lmax³)**。直观：消息传递时问题对称性从 SO(3) 退化为 SO(2)（绕边轴仅剩一个旋转自由度），eSCN 抓住这一点做数学等价稀疏化，而非近似。

**(c) 注意力重归一（EquiformerV2 §3.2）。** 对标量特征 `f_ij^(0)` 先加一层 LN 再进非线性：

```
z_ij = w_a^T · LeakyReLU(LN(f_ij^(0))),  a_ij = softmax_j(z_ij)
```

动机同 ViT-22B：Lmax 增大即输入通道数增大，不归一则 softmax 输入分布漂移。消融表 1（Index1→2）能量 MAE 改善 **2.4%**，力基本持平。

**(d) 可分离 S² 激活（§3.3，图 2）。** 普通门激活只建模 0→>0 单向；S² 激活 `y=G⁻¹(F(G(x)))`（向量→球面采样→无约束 F→逆变换）跨阶混合更强，但直接替换导致大梯度不收敛（表 1 Index3 didn\'t converge）。**可分离版本**：0 阶向量一切为二，一半走 SiLU 直通，一半与高阶一起走 S²，最后拼接并丢弃参与 S² 的 0 阶部分。Index2→4 力 MAE 显著下降，训练稳定。

**(e) 可分离层归一 SLN（§3.4，图 3）。** 传统等变 LN 对每阶独立归一，抹掉阶间相对重要性。SLN 对 L=0 用均值方差归一（含可学习 γ⁰/β⁰），对 L>0 用跨阶共享 RMS：

```
y^(0) = γ⁰◦(x^(0)−μ⁰)/σ⁰ + β⁰,  y^(L>0) = γ^(L)◦x^(L)/σ^(L>0),
σ^(L>0) = sqrt(1/Lmax Σ_L (σ^(L))²),  σ^(L) = RMS over m,i
```

保留高阶间相对幅度。Index4→5 力 MAE 进一步下降，且比多训 8 epochs 快约 6%。

**(f) 真实结论（摘要 + §4）。** OC20 上力 up to **9%**、能量 **4%**，速度-精度权衡全面占优，AdsorbML 中达到同等吸附能精度所需 DFT 计算减半（**2× reduction**）；仅 OC22 训练即超越在 OC20+OC22 联合训练的 GemNet-OC（数据效率）；QM9 与 OC20-2M 对比证实增益来自高阶 + 新架构而非单纯参数量；朴素把 eSCN 卷积塞进 Equiformer（Index1）**不超越** eSCN 基线（Index8），三项改进缺一不可。

### 4.2 官方代码关键细节（已精读 raw）

**仓库**：https://github.com/atomicarchitects/equiformer_v2 ，文件路径均为真实存在（已调 GitHub API 验证）。

- `nets/equiformer_v2/transformer_block.py` — **必读核心**。`SO2EquivariantGraphAttention` 完整实现 §3.5：边标量特征（原子嵌入 + 距离）→ 旋转对齐（`_rotate`）→ `so2_conv_1` → 可分离 S²/门激活分叉（`x_0_alpha` vs `x_0_gating` 的 `narrow` 切分）→ `so2_conv_2` → 注意力（`alpha_norm=LayerNorm` 即 §3.2 重归一 + `alpha_act=SmoothLeakyReLU` + `alpha_dot` 点积 + `softmax(edge_index[1])`）→ 加权消息 → 逆旋转 `_rotate_inv` → `_reduce_edge` → `SO3_LinearV2` 投影。`TransBlockV2` 实现 pre-norm + `GraphDropPath` + `EquivariantDropout` + FFN（`FeedForwardNetwork` 同样支持 `use_sep_s2_act/use_grid_mlp`）。**真实技巧**：注意力与激活共用第一次 SO(2) 卷积的输出切分（`extra_m0_output_channels`），省一次卷积；`use_m_share_rad` 让同 L 内各 m 共享径向权重，省参且稳。
- `nets/equiformer_v2/equiformer_v2_oc20.py` — `EquiformerV2_OC20` 主模型：`sphere_embedding` 仅初始化 l=0/m=0 系数；`GaussianSmearing(0, cutoff, 600, 2.0)` 距离展开；`SO3_rotation/mappingReduced/SO3_grid` 三件套初始化；`EdgeDegreeEmbedding`（注意 `_AVG_DEGREE=23.39` 重缩放技巧）；N 层 `blocks[i]` 循环；末端 `norm → energy_block（FFN→标量→index_add→/_AVG_NUM_NODES=77.81 归一）` 与 `force_block（SO2 注意力直接输出 L=1 三向量）`。**真实技巧**：能量按平均节点数归一、边度嵌入按平均度数缩放——两个全局统计量稳定大图训练，uniARPAT 可照搬（按 train 集原子数均值）。
- `nets/equiformer_v2/so2_ops.py` / `edge_rot_mat.py` / `so3.py` — SO(2) 卷积与 Wigner-D 旋转实现；`activation.py`（`SeparableS2Activation` vs `S2Activation` vs `GateActivation`）；`layer_norm.py`（`get_normalization_layer` 含 `rms_norm_sh`，默认 `norm_type='rms_norm_sh'`）；`radial_function.py`（两层 MLP + LN + SiLU 的径向函数）。
- `oc20/configs/s2ef/.../*.yml` + `scripts/train/.../*.sh` — 2M 用 12 层/L6/M2，All+MD 用 20 层/L6/M3（153M）与 8 层/L4/M2（31M）；checkpoint 表（2M 19.4/278；31M 16.3/232；153M 15.0/227）可作算力锚。

### 4.3 迁移到 uniARPAT（精确到函数）

**目标**：替换 `TransformerEncoderLayer`（`model/transformer.py:283-356`）的“非等变自注意力 + RP 加性偏置”双算结构，保留 PBC 与未弛豫输入管线。

1. **P0 必做（零风险）**：删除第 323 行冗余 `self.self_attn` 调用，只保留一次 QK 投影 + `base_scores + rp_scores` → softmax → 加权 V。数学等价，省 30–40% encoder 注意力时间与 0.5–1GB 激活显存。实测以 `torch.profiler` 为准。
2. **P1 等变替换（M6 主体）**：新增 `model/equivariant_encoder.py`，封装轻量 `TransBlockV2`-like 块（sphere_channels=64–128，Lmax=4–6，Mmax=2，num_layers=4–6，沿用官方默认 `rms_norm_sh/use_attn_renorm/use_sep_s2_act`）。输入：将 `compute_relative_features`（`utils/relative_features.py:42-69`）的 `distances/unit_dirs` 转为边图（cutoff 5Å + kNN 32，与官方 `max_radius/max_neighbors` 对齐），原子类型经 `tok_emb` 作 `sphere_embedding` 初始化（仅 l=0）。输出：取 L=0 通道作 invariant memory（接现有 Decoder），L=1 通道可选作力/应力辅助头（见 §6）。**为何不直接全量搬 153M**：uniARPAT 仅 1 万训练样本，全量 EquiformerV2 必过拟合；取 4–6 层 + 小通道（~5–10M）即够，消融时与 M5 同参量级对比才有说服力。
3. **归一/激活照搬**：FFN 的 GELU → 官方 `ScaledSiLU/SwiGLU` 视消融而定；LayerNorm → `rms_norm_sh`（官方默认）；dropout 0.1 沿用，`drop_path_rate=0.05` 仅在深层开启。
4. **径向基复用**：删除死代码 `rbf_encoder/rel_proj/dir_proj` 或替换为官方 `GaussianSmearing + RadialFunction（2层MLP+LN+SiLU）`，输入为真实笛卡尔距离（已有 PBC 修正），而非 clamp 后的截断值。

**成功判据**：M6（等变 encoder + 其余不动）在 phDOS 中位数 +0.03 且 OOD 晶系留一 +10% 即证独立增益；若 10 epochs 内 valid 力/谱 loss 爆，优先将 Lmax 6→4、Mmax 2→1 回退，而非全盘放弃。

---

## 5. 支柱二：能量条件逐点解码 + 晶系 Prompt（DOSTransformer）

<a id="5-支柱二"></a>

### 5.1 论文原文核心公式（已验证）

**来源**：https://arxiv.org/html/2311.12856v2 （NeurIPS 2023）。

**(a) 材料编码（§4.1，式 1）。**

```
H = GNN(X, A)   (Eq.1, H∈R^{n×d})
```

uniARPAT 的 geometric encoder 即对应此 H，无需推翻。

**(b) 交叉注意力：能量作 query、原子作 key/value（§4.2，式 2）。**

```
E^l = Cross-Attention(Q_{E^{l-1}}, K_H, V_H) = Softmax(E^{l-1} H^T / √d) H   (Eq.2, l=1..L1)
```

要点：**不用可学习 QKV 投影矩阵**，直接用能量嵌入作 query、原子嵌入作 key/value。每个能量获得一个材料相关的表示（material-specific energy embedding）。这正是 uniARPAT 缺的“模型知道在预测哪个能量”。

**(c) 全局自注意力（§4.2，式 3）。**

```
Ẽ^p = Softmax(Ẽ^{p-1} Ẽ^{p-1 T}/√d) Ẽ^{p-1}   (Eq.3)
```

输入先拼接 sum-pool 的全局材料向量 `g_i`：`Ẽ_j^0 = φ1(E_j^{L1}‖g_i)`（φ1: R^{2d}→R^d）。动机：交叉注意力只捕获原子级局部信息，全局包络需显式注入。

**(d) 晶系 prompt 系统自注意力（§4.2 后半）。** 可学习 `P∈R^{7×dp}`（Cubic/Hexagonal/Tetragonal/Trigonal/Orthorhombic/Monoclinic/Triclinic），输入变为 `Ẽ_j^0 = φ2(E_j^{L1}‖g_i‖P_k)`（φ2: R^{2d+dp}→R^d）。低层 `E^{L1}` 跨晶系共享，高层 `Ẽ^{L2}` 晶系特化；顶层再加 L3 层交叉注意力（query=`Ẽ^{L2}`）提纯。**设计哲学**：朴素拼接晶系特征会干扰跨系泛化，prompt 则解耦共享 vs 特化。

**(e) 能量解码器（§4.3）。** 对每能量独立预测标量：

```
Ŷ^i_j = φ_pred(E_j^{L3,i}),  φ_pred: R^d→R^1
```

对比旧范式 `φ': R^d→R^M`（整谱一次输出）。**这是全文最关键的一句话**：DOSTransformer 自称首个逐点预测 DOS 的工作。

**(f) 双轨训练损失（§4.4）。**

```
L = 1/(NM) Σ_i Σ_j √((Ŷ^i_j − Y^i_j)²),  L_total = L_glob + β·L_sys
```

`L_glob` 用无 prompt 分支，`L_sys` 用有 prompt 分支；推理只用 `sys` 分支。β 为平衡项（代码默认见 §5.2）。

**(g) 真实结论（表 1/表 2/表 3 + §5.2–5.4）。**
- 表 1（in-distribution）：phonon MSE/MAE/R²：MLP 0.309/0.106/0.576 → +能量 0.251/0.099/0.652；GN 0.259/0.099/0.638 → +能量 0.226/0.092/0.685；E3NN 0.210/0.077/0.705 → +能量 0.200/0.074/0.724；**DOSTransformer 0.191/0.071/0.733**。electron：0.347/0.130/0.487 → … → **0.221/0.089/0.679**。**能量嵌入在所有基线一致增益**；E3NN 增益小是因为朴素拼接干扰等变性——对 uniARPAT（非严格等变主干）是利好，可大胆加。
- 物理有效性：以 DFT 谱训 MLP 再以预测谱推 Bulk/Bandgap/Fermi，DOST 最优（0.427/0.461/2.337），证明不是刷 R² 而是物理有效。
- 表 2（OOD：#species 与晶系留一）：GN 在 OOD 下从能量嵌入获益**大于** in-distribution（结构剧变时整谱回归崩，逐点回归稳；类比 CNN vs ViT 归纳偏置论述）。
- 表 3（复杂晶系 10% few-shot）：仅调 prompt+解码器 0.570 > 全量微调 0.558（小样本全量易过拟合，prompt 调参是正解）。
- §5.4 消融（图 3）：去交叉注意力 / 去自注意力 / 去 prompt 均掉点，三者缺一不可。

### 5.2 官方代码关键细节（已精读 raw）

**仓库**：https://github.com/HeewoongNoh/DOSTransformer 。

- `embedder_eDOS/DOSTransformer.py` — **逐行对应论文**：`self.embeddings = nn.Embedding(201, n_hidden)`（eDOS 201 点能量嵌入；phDOS 分支为对应点数）；`self.promt_token = nn.Embedding(7, n_hidden//2)`（注意拼写 `promt` 为原作者笔误，引用时保留原样）；`GN_encoder + stacked_processor（EdgeModel/NodeModel）` 对应式 1；`self.transformer`（L1 交叉）→ `GN_decoder` sum-pool 得 `graph` → `torch.cat([energies_global, graph]) → fc → transformer_self（L2）→ transformer_source（L3）→ out_layer` 得 `dos_global`；第二路 `cat([energies_global, graph, prompt_token[g.system]]) → fc_prompt → ...` 得 `dos_system`；`forward` 返回 `dos_global, x, dos_system` 双轨。**真实技巧**：双分支共用同一 `transformer_self/source` 模块（参数共享，正则效应）；`g.system` 按晶系索引取 prompt 并 `repeat(201,1,1)` 广播；`Decoder` 用 `scatter_sum + MLP` 得全局向量。
- `layers/transformer.py` — `TransformerEncoder(embed_dim, num_heads=1, layers=t_layers)`，`TransformerEncoderLayer` 采用 **pre-norm**（`normalize_before=True`，`maybe_layer_norm(before)`），FFN 为 `fc1(4×)→ReLU→fc2`，注意力为自定义 `MultiheadAttention`。**真实技巧**：DOST 的 head 数为 1（能量-原子对齐不需要多头切分语义），uniARPAT 照搬时 head 可保持 8 但需消融 1 vs 8。
- `layers/multihead_attention.py` — 标准 QK^T/√d + dropout，无花哨，证明增益来自范式而非注意力变体。
- `main_phDOS.py` — **损失与优化原文**：`criterion=MSELoss`，`mse_global→rmse_global=sqrt(mse).mean()`，`loss = rmse_global + args.beta*rmse_system`（论文 L_total 的代码实现）；`AdamW(lr, weight_decay=1e-2)`；`default_dtype=torch.float64`（**phDOS 用 float64**，注释明示）；`batch_size=1`（图数据，按 material 切分）；早停 `args.es/args.eval` 逻辑。**真实技巧**：RMSE 而非 MSE 作损失（对尖峰更敏感），双分支加权而非单分支，float64 保 phDOS 小值精度。
- `embedder_phDOS/DOSTransformer_phonon.py`（同构，点数不同）与 `utils.py`（`test_phonon/build_data/load_data/train_valid_test_split/r2`）构成完整可跑管线。

### 5.3 迁移到 uniARPAT（精确到行）

**目标**：删除静态 query + 整谱卷积头，改为能量条件逐点头；Decoder 代码零改动，只换输入。

1. **能量嵌入表**：新增 `energy_emb_e (128→d)` 与 `energy_emb_p (64→d)`（eDOS 网格 linspace(-10,10,128)，phDOS 网格 linspace(-280,980,64)，与 `thermo_props.py:17-23` 对齐）。初始化可用正弦位置编码 + 可学习残差，而非全零（对标 DOST `nn.Embedding` 随机初始化）。
2. **Decoder 输入替换**（`model/transformer.py:154-186`）：保留解耦双 Decoder 与 `TransformerDecoderLayer` 不动，仅将 `edos_query_embed/phdos_query_embed` 替换为能量嵌入序列作 `query_pos`，`edos_tgt/phdos_tgt` 替换为可学习能量值嵌入。交叉注意力 `multihead_attn(query=q, key=memory, ...)`（第 391–393 行）天然即论文式 2，无需改层代码。注意力权重 `A(E_j, atom_i)` 可解释为谱贡献归因（论文卖点 Fig 之一，但必须写“phenomenological attribution”，见 §12）。
3. **晶系 prompt**：新增 `prompt_token = nn.Embedding(7, d//2)` + `fc/fc_prompt` 双分支（照抄 DOST `forward` 双路结构）。晶系标签从 `pos` 的晶格参数解析（`utils/relative_features.py:6-40` 的 abc/αβγ → 晶系分类函数，需新写 30 行工具函数；训练时 teacher forcing 用真值晶系，推理时用预测晶系或 uniform）。损失照抄 `L_total = L_glob + β·L_sys`（β 初值 0.5，`main_phDOS.py` 模式），推理只用 sys 分支。few-shot 新晶系时**只调 prompt+解码器**（表 3 证据）。
4. **输出头替换**（`model/heads.py:33-82`）：删除整谱 `DeepConv1dHead/MultiScaleResidualHead`（或保留作残差分支），改为逐点 `φ_pred: MLP(d→d/2→1)` 对每个 `E_j^{L3}` 独立输出标量。参数不增反减（128×512 整谱卷积 vs 共享 MLP），消融结论更保守有力。
5. **Shape-Scale 保留**：逐点输出后仍过 `safe_shape_norm` + `ScaleHead`（盲测链路不动）；双轨评估（Blind/Oracle）在 `model/model.py:309-343` 原样保留，新增第三轨 Blind-HR（若做连续采样，见姊妹报告，不在本报告关键路径）。

**成功判据**：M6（能量条件 + prompt，其余不动）eDOS Oracle 中位数 +0.04 且带隙 F1 +0.08（容量更小 ⇒ 保守证据）；若 E3NN 式“增益小”复现（等变 encoder + 朴素能量拼接冲突），则改用 DOST 式交叉注意力而非拼接（论文已预警）。

---

## 6. 支柱三：非平衡去噪辅助（DeNS，直击未弛豫输入）

<a id="6-支柱三"></a>

### 6.1 为什么 DeNS 是 uniARPAT 的对口解法

uniARPAT 输入是**未弛豫结构**，监督目标（DFT 谱）却对应弛豫后基态。模型必须隐式学会“弛豫算子”。DeNS 的核心洞见（图 1）：弛豫轨迹上的中间帧全是非平衡结构（力非零），而平衡结构仅占极小部分；把“加噪→去噪 + 力编码”作辅助任务，可显式 teaches 模型势能面几何，且**不需任何额外数据**（同一 batch 内部分样本走原任务、部分走去噪）。

### 6.2 论文原文核心公式（已验证）

**来源**：https://arxiv.org/html/2403.09549v3 （TMLR 2024）。

**(a) 原任务损失（§3.1，式 1）。**

```
λ_E·L_E + λ_F·L_F = λ_E·|E'(S)−Ê(S)| + λ_F·1/|S| Σ_i |f'_i − f̂_i|²   (Eq.1)
```

能量做均值方差归一，能量/力各有系数。这是 uniARPAT 双任务损失（eDOS/phDOS）的直接类比物。

**(b) 加噪构造（§3.2.1，式 2）。**

```
S̃ = {(z_i, p̃_i)},  p̃_i = p_i + ε_i,  ε_i ∼ N(0, σI_3)   (Eq.2)
```

σ 为可调噪声强度（多尺度采样见代码）。

**(c) 朴素去噪损失（§3.2.1，式 3）。**

```
E_{p(S,S̃)}[1/|S| Σ_i |ε_i/σ − ε̂(S̃)_i|²]   (Eq.3)
```

除以 σ 归一化噪声目标。**关键论断**：对平衡结构这是多对一映射（well-defined），对非平衡结构则是 ill-posed（图 1b：S 与 S̃ 之间插值仍在数据分布内，目标不唯一）。

**(d) 力编码 DeNS 损失（§3.2.2，式 4，本文心脏）。**

```
L_DeNS = E[1/|S_non-eq| Σ_i |ε_i/σ − ε̂(S̃_non-eq, F(S_non-eq))_i|²]   (Eq.4)
```

把原结构力 `F(S_non-eq)` 作额外输入，预测满足该力的原结构而非任意结构（图 1d）。退化到平衡（力≈0）即式 3（图 1e），故旧工作均为特例。

**(e) 力嵌入方式（§3.2.2，式 5）。**

```
x_f^(L) = SO3_Linear^(L)(‖f‖·Y^(L)(f/‖f‖))   (Eq.5)
```

力经球谐投影到各阶 + SO(3) 线性扩通道（1→C_L），幅度信息经 `‖f‖` 乘回（因球谐前已归一）。加到初始节点嵌入上。**为何偏爱等变网络**：力是 L=1 向量，等变表示天然可容纳；不变网络需内积投影到边嵌入（逆 GemNet-OC 解码），别扭。这也是支柱一与支柱三的耦合点。

**(f) 辅助总损失（§3.2.3，式 6）。**

```
λ_E·|E'(S)−Ê(S̃,F)| + λ_DeNS·L_DeNS   (Eq.6)
```

能量头跨式 1/式 6 共享；超参 `p_DeNS`（每结构走 DeNS 的概率）、`λ_DeNS`（辅助权重）、`σ`、`r_DeNS`（部分 коррупция 比例，图 2c 只加噪随机子集：被扰原子编码力并预测噪声，未扰原子预测力，能量照常预测）。

**(g) 真实结论（§4 + 附录 B–D）。**
- OC20-2M（表 1a）：12ep 力 20.46→**19.09**、能量 285→**269**；20ep 19.78→18.58；30ep 19.42→**18.02**/278→251；**同迭代数下 2.3× 训练时间节省**（达到同精度所需步数）。
- 消融（表 1b）：无力编码 DeNS（Index3）力 21.32 **差于**不用 DeNS（Index1 20.46）——**力编码是必要非充分条件**，照抄时绝不可省。
- All+MD SOTA（§4.1.2）与 OC22（§4.2）：能量 up to **15%**、力 **12%**、IS2RE **15%**；MD17（§4.3）：Equiformer(Lmax=2)+DeNS 超 (Lmax=3) 无 DeNS 且省 **3.1×** 时间；eSCN/SEGNN 同样获益（架构无关）。
- 开销：仅多一个噪声头（§3.2.4 边际增加），每步部分样本走辅助分支；OMat24 证实 EquiformerV2+DeNS 达 Matbench Discovery SOTA（README 横幅）。

### 6.3 官方代码关键细节（已精读 raw）

**仓库**：https://github.com/atomicarchitects/DeNS 。

- `trainers/denoising_forces_trainer.py` — **全文件 700+ 行，核心技巧如下**：
  - `add_gaussian_noise_to_position(batch, std, corrupt_ratio, all_atoms)`：生成 `noise_vec∼N(0,std)`；若 `corrupt_ratio` 非空则 `rand < ratio` 作 `noise_mask` 并清零未选原子噪声；若 batch 有 `md` 属性则**跳过 MD 分裂结构**（`noise_vec[(~noise_mask)]*=0` 双重掩码）；仅对 `fixed==0` 自由原子加噪（`all_atoms` 控制是否含固定原子）；写入 `batch.noise_vec/noise_mask/denoising_pos_forward=True`。**uniARPAT 映射**：`fixed` ↔ padding mask，`md` ↔ 高质量弛豫子集（若有）。
  - `add_gaussian_noise_schedule_to_position(...)`：多尺度版本，`sigmas=exp(linspace(log std_low, log std_high, num_steps))`，每结构随机抽 `t` 并广播到原子（`batch.sigmas = sigmas[ts][batch.batch][:,None]`），`noise = randn * sigmas`。**真实技巧**：单 σ 易过拟合特定扰动尺度，多尺度覆盖 0.1–0.5Å（对应未弛豫畸变量级）。
  - `compute_atomwise_denoising_pos_and_force_hybrid_loss(pred, target, noise_mask, force_mult, denoising_pos_mult)`：`L2 norm` 按 `noise_mask` 分流加权（扰原子×去噪系数，未扰×力系数），可选自由原子掩码。**这就是图 2c 部分 коррупция 的损失实现**。
  - `denoising_pos_eval(...)`：覆写 OCP Evaluator，按 `noise_mask` 分离 `denoising_energy_mae/denoising_pos_mae/denoising_force_mae` 三指标，防止去噪误差污染原任务指标。uniARPAT 应照抄此分离评估思想（shape/scale/gap/sum 已分离，见 `model.py:216-226`，再加 `dens_*` 一组）。
  - `DenoisingForcesTrainer.train()`：每 batch 以概率 `prob` 决定走原任务还是 DeNS（`np.random.rand() < prob`）；`fixed_noise_std` 分叉单 σ vs 调度；`_compute_loss` 中 DeNS 分支用 `Normalizer(mean=0, std=std/std_high)` 归一化噪声目标（对应式 3 的 `/σ`），部分 коррупция 时 `target = noise*mask + force*(~mask)` 混合；`_compute_metrics` 中反归一化后分别评估。**真实技巧**：归一化器复用 OCP `Normalizer`，能量头共享，噪声头独立；valid/test 从不用力标签编码（防泄漏，论文 §3.2.3 末段强调）。
- `datasets/dens_lmdb_dataset.py:92-108,193-198` — `DeNSLmdbDataset` 靠 `data_log.*.txt` 区分 All vs MD 分裂并加 `md` 属性；uniARPAT 若有弛豫/未弛豫双源数据可仿此打标。
- `configs/oc20/2M/.../equiformer_dens_v2_..._std@0.1_....yml:7,20` 与 `configs/oc22/.../e@4_f@100_std@0.15.yml:7,19-25` — 路径 + 线性参考 + OC20 参考信息三处必改；超参锚：2M 用 `std 0.1–0.15`，OC22 用 `std 0.15`，All+MD 用 `dens-relax-data-only` 变体。

### 6.4 迁移到 uniARPAT（未弛豫输入的专属解法）

1. **辅助头**：在 `Transformer` 末端加轻量噪声头（1 层等变注意力或 MLP，~0.5M 参数）。输入为加噪后 `pos`（笛卡尔分数坐标加噪 σ=0.1–0.3Å，多尺度采样），输出逐原子噪声向量（3 维）。力编码：若无 DFT 力标签，用 MACE-MP-0 离线预测的伪力作编码输入（见 §7；论文要求力标签仅训练用，valid 不用，不泄漏）。
2. **部分 коррупция**：每 batch 以 `p_DeNS=0.3–0.5` 抽部分样本走 DeNS；每样本内以 `r_DeNS=0.3–0.5` 抽原子加噪（`model.py:train_one_step` 内分叉，仿 `DenoisingForcesTrainer.train` 的 `rand < prob` 模式）。被扰原子预测噪声，未扰原子仍预测谱（谱任务即原任务），能量/标度头共享。
3. **损失**：`total = 谱损失（§8）+ λ_DeNS·L_DeNS`，λ 初值 0.1–0.3（DeNS 论文量级），`L_DeNS` 为归一化噪声 L2（式 4 代码即 `compute_atomwise_..._hybrid_loss` 的谱适配版）。
4. **与支柱一耦合**：若已上等变 encoder，力嵌入按式 5（球谐 + SO(3) 线性 + 加到节点嵌入）实现；若仍用旧 encoder，退化为“伪力模长 + 方向余弦拼到原子特征”（不变网络逆 GemNet 方案，论文 §3.2.2 末段），效果打折但仍有数据增强收益。

**成功判据**：同迭代数下 valid 谱 loss 更低（2× 步数节省即胜）；若无力编码版本差于基线（复现表 1b Index3），立即补力编码而非弃 DeNS。

---

## 7. 扩展：MACE 基础模型蒸馏 + FlowMM 黎曼流匹配

<a id="7-扩展"></a>

### 7.1 MACE 教师蒸馏（低成本高回报，可选插件）

**论文**：MACE（NeurIPS 2022, https://openreview.net/forum?id=YPpSngE-ZU）+ foundation 2024 https://arxiv.org/abs/2401.00096 （154 页，83 图；89 元素、MPTrj 1.6M、开箱即用 + 少量微调）。

**代码**：https://github.com/ACEsuit/mace — `mace/modules/blocks.py`（高阶对称收缩消息传递）、`mace/modules/symmetric_contraction.py`（ACE 多体展开实现）、`mace/modules/loss.py`（`WeightedEnergyForcesLoss`、`UniversalLoss` + `conditional_huber_forces` 按力模 `<100/<200/<300` 分档加权 `[1.0,0.7,0.4,0.1]` + Huber delta；大力量级降权防离群主导——uniARPAT 谱大峰截断 500 可照搬此分档思想）、`mace/modules/models.py`（基础模型前向）。

**迁移**：离线提取 MACE-MP-0/MPA-0（https://github.com/ACEsuit/mace-foundations ）在 10965 训练结构上的原子/全局嵌入作教师，学生（uniARPAT 全局池化 `h_crys`）经 512→teacher_dim 投影做余弦 + MSE 蒸馏（+0.3M 参数，微调后可删）。零在线开销（预计算 npy）。若环境冲突（e3nn/PyTorch 地狱），退化为纯 DeNS（风险隔离）。

### 7.2 FlowMM 黎曼流匹配（M9–M10 生成式续作，非关键路径）

**论文**：https://arxiv.org/abs/2406.04713 （ICML 2024）。把黎曼流匹配推广到晶体对称性（平移/旋转/置换/周期），基分布自由选择（简化扩散模型难题）；标准基准 + DFT 验证下 **~3×** 积分步效率；CSP 表：Perov-5 53.15、Carbon-24 23.47、MP-20 61.39、MPTS-52 17.54 超 DiffCSP/CDVAE；E_hull 分布更优。

**代码**：https://github.com/facebookresearch/flowmm — `src/flowmm/rfm/manifold_getter.py`（`ManifoldGetter`：`flat_torus_01[_normal/_fixfirst]` 处理分数坐标周期、`spd_euclidean/riemannian` 与 `lattice_params[_normal_base]` 处理晶格、`analog_bits/simplex/null_manifold` 处理原子类型，`georep↔flatrep` 双向变换 + `get_dims/get_manifolds` 批处理）。

**迁移定位**：不在本报告关键路径。M9 时以 R1 确定性预测为均值的矩形流精修（8–20 步 ODE）+ 系综方差作不确定性门控（拦截 80% 失败）；M10 做性质引导反事实谱筛选。细节与姊妹 EFM 报告分工（本报告精修式 vs 对方生成式），投稿时写 companion 声明防“撞故事”。

---

## 8. 统一训练协议与总损失重写

<a id="8-统一训练协议"></a>

### 8.1 总损失（M6/M7，支柱一+二+三联合；R3 精修独立训练）

```
L = [L_KL^e + 3.0·L_KL^p]            # 分布形状主力（SumNorm-KL，带隙敏感）
  + 0.3·(L_W1^e + 3.0·L_W1^p)        # 一维 Wasserstein（cumsum 可微，对峰位平移鲁棒）
  + 0.5·(L_MSE,w^e + 3.0·L_MSE,w^p)  # 物理加权 MSE（峰高锚，见下）
  + 0.5·(L_scale,e + L_scale,p)      # 双锚 Huber（log y_max + log 非零均值，替代单一 y_max）
  + 0.2·L_gap + 0.1·L_sum            # 沿用 model.py:178-199（费米窗带隙 + 3N 求和规则）
  + λ_DeNS·L_DeNS                    # DeNS 辅助（§6，初值 0.1–0.3）
  + 0.01·L_bal-MoE                   # 若上 adapter（可选）
  + L_total_DOST                      # DOST 双轨：L_glob + β·L_sys（β 初值 0.5，main_phDOS.py 模式）
```

- 形状三项在下游敏感性权重 `w(E)` 下计算：phDOS `w∝α·K_Cv(300K)+β·ν²+γ`（C_v 核 + Θ_D 二阶矩 + 保底 uniform γ=0.3）；eDOS `w∝α·N(E;0,1.5eV)+β·|∇g|+γ`（费米窗 + 陡峭处加权，Seebeck 斜率相关）。
- 多任务 (1.0, 3.0) 沿用 M5，再加 uncertainty weighting（`e^{−s_i}L_i + s_i`）±20% 自适应（防 KL 量级打破平衡）。
- DOST 双轨与谱三项正交：前者是分支加权，后者是 bin 加权，可叠加。

### 8.2 优化器与调度（沿用 + 微调）

- AdamW（encoder 1e-5 / 其余 5e-5 discriminative LR，若有预训练/DeNS；从零 M6 则统一 5e-5），betas (0.9,0.99)、余弦 + 5 epoch warmup + min_lr 1e-6 沿用 `configs/config.yaml`；`weight_decay=1e-2`（DOST `main_phDOS.py` 实测值）。
- phDOS 相关计算建议 float64 验证一次（DOST 注释 `float64 for phdos`），训练仍 AMP fp32 + `GradScaler(init_scale=1024)`。
- 选优：保留 `balanced_score`，新增 `phys_score = 0.4·R2_e + 0.3·R2_p + 0.15·gapF1 + 0.15·(1−normCvErr)` 作第二判据；batch 32 沿用。

---

## 9. 算力与显存定量评估

<a id="9-算力显存"></a>

### 9.1 基线实测（Week-1 Pilot，V100-32GB，batch=32）

单 epoch 107.8s（343 steps → 0.314s/step），峰值 9.16GB，M5 总参 75.93M。解析拆分（±20%，M6 首项任务以 profiler 校准）：Encoder（含 RP pair 表）~19M + rp_proj 0.3M / ~40% 时间 / pair 表 0.5–1.5GB；解耦双 Decoder×2×6 ~50M / ~40%；对称双 Head 1.576M / ~8%；ScaleHead + emb ~2M / ~2%；AdamW 状态 ~0.6GB；框架/碎片 ~1–2GB。

### 9.2 各升级增量（同数据同 batch）

| 升级项 | 参数 Δ | 时间 Δ/epoch | 显存 Δ | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| P0 删双算（§4.3-1） | 0 | −12~−18s | −0.5~−1GB | 数学等价，必做 |
| 等变 encoder（4–6 层，L4–6，ch 64–128） | +5~+10M（含 SO(2) 卷积 + 径向函数） | −5~+10s（eSCN O(L³) 省 vs 高阶增，视 Lmax） | −0.5（删 pair 表）~+0.5GB | 替换 RP 表，净显存大概率降 |
| 能量条件 + prompt + 逐点头 | −1~−2M（删整谱卷积，加嵌入表 192×512 + prompt 7×256 + MLP 头） | +2~+5s（序列长 128/64，交叉注意力小） | +0.2GB | 容量更小 ⇒ 消融更保守 |
| DeNS 辅助（噪声头 + 部分分支） | +0.5M | +10~20%（部分样本双分支） | +0.2~0.4GB | 无仿真单步回归 |
| MACE 蒸馏（离线） | +0.3M（投影，可删） | ≈0（离线 npy） | ≈0 | 可选插件 |
| R3 流精修（M9，可选） | +12~16M | +25~40% | +1~2GB | 与主训解耦 |

### 9.3 三档配置

- **保守档（M6：P0 + 能量条件 + 轻等变，必跑）**：~70M，~7GB，~70s/epoch → 100ep ≈ 2h/变体（比现在 3–3.5h 更快），单卡 V100 无压力。
- **标准档（M7：M6 + DeNS + 原子特征扩充）**：~71M，~7.5GB，~80s/epoch，预期 eDOS 中位 0.60+、OOD +15–30%。
- **完整档（M8–M9：+硬投影 + UQ/精修）**：~85M，~9GB，推理 <100ms（含 NFE=12 精修），新增置信带 + 失败拦截。

---

## 10. 失效风险矩阵与硬回滚判据

<a id="10-风险回滚"></a>

| # | 风险 | 早期信号（10ep 内） | 缓解 / 回滚 |
| :--- | :--- | :--- | :--- |
| R-a | 等变块高阶发散（谱毛刺/NaN） | loss 爆，S² 分支梯度范数 >> 门分支 | Lmax 6→4，Mmax 2→1；关 S² 退回门激活（表 1 Index4→2）；trunk/径向 LR×0.1 |
| R-b | 能量拼接干扰等变（复现 E3NN 增益小） | +能量后 valid 不升反降 | 改拼接为交叉注意力（式 2 模式）；encoder 冻结只训解码器一次对照 |
| R-c | DeNS 无力编码负增益（复现表 1b Index3） | DeNS 版差于基线 | 立即补力编码/伪力（式 5）；降 λ_DeNS 至 0.05；改全扰为部分扰（r 0.5→0.3） |
| R-d | KL 在平坦谱 NaN | loss NaN（flat 样本） | SumNorm + ε=1e-6；平坦（var<1e-4）自动切纯 MSE（沿用 `compute_shape_loss:29-30` 逻辑） |
| R-e | Prompt 过拟合小晶系 | 小系 train 降 valid 升 | β 减半；prompt dim 减半；few-shot 只调 prompt（表 3 范式） |
| R-f | 预训练/蒸馏泄漏 test 1371 | 指纹去重缺失 | composition + 空间群去重 + 附录披露；不通过不得投稿 |
| R-g | 与姊妹路线撞故事 | — | Cover letter 写 v3 确定性 → v4 生成式分工；共享 encoder，一次投入双倍产出 |

**硬回滚线**：100ep 后 M6 eDOS Oracle 中位数 <0.55 **且** gap F1 无提升 → 弃三支柱，回退 M5 + P0（白捡 15% 加速）+ DeNS 单项（大概率仍正增益）；DeNS 单项负增益则再降 λ/改部分扰一次，再负则弃。

---

## 11. 路线图 M6–M10、论文章节与期刊故事线

<a id="11-路线图"></a>

### 11.1 消融矩阵（接续 README Table 1 的 M1–M5）

| 变体 | 改动（相对 M5） | 参数 | 回答的问题 | 成功判据 |
| :--- | :--- | :--- | :--- | :--- |
| **M6** | P0 删双算 + 能量条件逐点 + 晶系 prompt + 轻等变 encoder（4–6 层） | ~70M | 能量坐标 + 等变是否是独立增益？ | eDOS 中位 +0.04，gap F1 +0.08（容量不增 ⇒ 保守证据） |
| **M7** | M6 + DeNS 辅助（p 0.3–0.5，r 0.3–0.5，多尺度 σ）+ 原子特征 3→14 维 | ~71M | 未弛豫偏移能否靠辅助任务补？ | OOD 晶系留一 +15%，同迭代 valid 更低（2× 步数节省） |
| **M8** | M7 + 硬投影（3N 单纯形 + 带隙条件投影）+ 下游加权损失 | ~71M | 求和规则/下游能否结构化解决？ | phDOS 面积误差减半，Θ_D/C_v 显著降 |
| **M9** | M8 + 流精修 + UQ 门控（+ MACE 蒸馏可选） | ~85M | 残差分布 + 已知未知？ | 峰位 MAE −15%，失败拦截 AUC>0.8 |
| **M10** | M9 + 性质引导筛选 demo | 同 M9 | 逆向闭环？ | Top-k 富集（已知热电/催化进 top 2%） |

单卡 V100 上 M6–M8 各 ~2–2.5h/100ep，M9 精修 ~3h，全套一周可收割。

### 11.2 学位论文章节映射

- **第三章（方法）**：3.3 等变几何编码（SO(2) 约化定理陈述 + SLN/S² 公式）→ 3.4 能量条件解码（式 2/式 3 + prompt 双轨损失）→ 3.5 非平衡去噪辅助（式 4/式 6 + 部分扰动）。
- **第四章（实验/应用）**：4.3 OOD 晶系泛化 → 4.4 带隙/峰位专项 → 4.5 热力学下游（Θ_D/C_v/κ_L）→ 4.6 UQ 门控漏斗（M9）。

### 11.3 期刊故事线

> **“第一个把未弛豫输入、能量坐标缺失、非等变编码三个 DOS 特有痛点，用同一代码栈的三篇连贯顶会成果一次性解决的统一框架”**——对标 DOSTransformer（能量条件）+ EquiformerV2/eSCN（等变）+ DeNS（非平衡去噪）+ MACE/FlowMM（基础/生成扩展），四组引文各取一机制，融合成可消融系统。目标刊：Nature Comput. Sci. / npj Comput. Mater.（第二篇）→ Matter/JACS Au（应用）→ ICLR/NeurIPS ML4PS（精修短文）。

---

## 12. 反夸大边界与可复现清单

<a id="12-反夸大"></a>

**绝不声称**：① 能量-原子注意力 ≠ 电声耦合矩阵元 α²F（只是 phenomenological attribution）；② Slack κ_L / 费米斜率 Seebeck ≠ zT 精确预测（仍是漏斗初筛，DFT/BoltzTraP2 复核不变）；③ 去噪采样多样性 ≠ 热力学系综；④ 筛选 Top-k ≠ 发现可合成材料（需合成验证）；⑤ 跨分辨率超分细节 ≠ DFT 精度（须报 HR 网格 WD/KL 并承认插值上限）。

**可复现清单**：□ M1–M5 同协议重跑（三种子 mean±std，`config.yaml` 哈希冻结）；□ 去重脚本 + 泄漏报告；□ Blind/Oracle 双轨 + 失败率 + gap F1 + 峰位 MAE + Θ_D/C_v/κ_L-Spearman 全口径；□ profiler 实测 §9 表格；□ `dosdata/` 预测谱公开；□ 与姊妹路线分工声明入附录。

---

## 13. 参考文献

<a id="13-参考文献"></a>

**等变几何（支柱一）**
- Liao et al., EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations, ICLR 2024. 论文 https://arxiv.org/abs/2306.12059 ，全文 https://arxiv.org/html/2306.12059v3 。代码 https://github.com/atomicarchitects/equiformer_v2 （关键文件 `nets/equiformer_v2/transformer_block.py`、`nets/equiformer_v2/equiformer_v2_oc20.py`、`nets/equiformer_v2/so2_ops.py`、`nets/equiformer_v2/activation.py`、`nets/equiformer_v2/layer_norm.py`）。核心公式：式 (1) CG 张量积（§2.1）；§3.2 注意力重归一；§3.3 可分离 S²；§3.4 SLN；表 1 消融；摘要 9%/4%/2×。
- Passaro & Zitnick, Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs (eSCN), ICML 2023. 论文 https://arxiv.org/abs/2302.03655 。O(L⁶)→O(L³)；OC20/OC22 SOTA；式 (21)/(40)。
- Liao & Smidt, Equiformer, ICLR 2023. https://arxiv.org/abs/2206.11990 （V2 地基）。

**能量条件解码（支柱二）**
- Lee, Noh et al., Density of States Prediction of Crystalline Materials via Prompt-guided Multi-Modal Transformer (DOSTransformer), NeurIPS 2023. 论文 https://arxiv.org/abs/2311.12856 ，全文 https://arxiv.org/html/2311.12856v2 。代码 https://github.com/HeewoongNoh/DOSTransformer （关键文件 `embedder_eDOS/DOSTransformer.py`、`layers/transformer.py`、`layers/multihead_attention.py`、`main_phDOS.py`、`main_eDOS.py`）。核心公式：式 (1)(2)(3)（§4.1/§4.2）；§4.3 逐点解码；§4.4 双轨损失；表 1/表 2/表 3。
- Chen et al., E3NN phonon DOS（DOST 表 1 对标基线，https://github.com/zhantaochen/phonondos_e3nn ）。

**非平衡去噪（支柱三）**
- Liao et al., Generalizing Denoising to Non-Equilibrium Structures Improves Equivariant Force Fields (DeNS), TMLR 2024. 论文 https://arxiv.org/abs/2403.09549 ，全文 https://arxiv.org/html/2403.09549v3 。代码 https://github.com/atomicarchitects/DeNS （关键文件 `trainers/denoising_forces_trainer.py`、`datasets/dens_lmdb_dataset.py`、`model/`、`configs/oc20/.../*.yml`）。核心公式：式 (1)–(6)（§3.1–§3.2.3）；图 1/图 2；表 1a/1b；§4.1–§4.3（2.3×/3.1×/15%/12%）。
- Godwin et al., Noisy Nodes（DeNS §2 对比的平衡去噪基线）。

**基础与生成扩展**
- Batatia et al., MACE: Higher Order Equivariant Message Passing, NeurIPS 2022 (https://openreview.net/forum?id=YPpSngE-ZU)；Batatia et al., A foundation model for atomistic materials chemistry, 2024 https://arxiv.org/abs/2401.00096 。代码 https://github.com/ACEsuit/mace （`mace/modules/blocks.py`、`mace/modules/symmetric_contraction.py`、`mace/modules/loss.py`、`mace/modules/models.py`）；权重 https://github.com/ACEsuit/mace-foundations 。
- Miller et al., FlowMM: Generating Materials with Riemannian Flow Matching, ICML 2024. https://arxiv.org/abs/2406.04713 。代码 https://github.com/facebookresearch/flowmm （`src/flowmm/rfm/manifold_getter.py`、`src/flowmm/model/`、`src/flowmm/data/`）。
- Chanussot et al., OC20 (https://arxiv.org/abs/2010.09990)；Tran et al., OC22；OMat24 (https://arxiv.org/abs/2410.12771)（EquiformerV2+DeNS Matbench Discovery SOTA 背书）。

---

## 附录 A：最小代码改动草图

<a id="附录-a"></a>

```python
# 1) P0：model/transformer.py TransformerEncoderLayer.forward —— 删除第 323 行冗余 self_attn，
#    只保留一次 QK 投影 + base_scores + rp_scores → softmax → ×V（数学等价，省 30–40% 注意力时间）。
#    对照：equiformer_v2 nets/equiformer_v2/transformer_block.py 的单次 SO2 卷积模式（SO2EquivariantGraphAttention.forward）。

# 2) 能量条件：model/transformer.py:84-89 ——
#    edos_query_embed = nn.Embedding(128, d_model)  # 能量值初始化（正弦+可学习），替代 zeros(128,512)
#    phdos_query_embed = nn.Embedding(64, d_model)
#    prompt_token = nn.Embedding(7, d_model//2)     # 对照 DOSTransformer embedder_eDOS/DOSTransformer.py: self.embeddings/promt_token
#    Decoder 调用不变（query_pos=能量嵌入），对照 DOST 式 (2)：Softmax(E H^T/√d)H。
#    输出头：heads.py 整谱卷积 → 逐点 MLP(phi_pred: d→d/2→1)，对照 DOST §4.3。
#    损失：loss = rmse_global + beta*rmse_system，对照 main_phDOS.py: loss = rmse_global + args.beta*rmse_system。

# 3) DeNS 辅助：model/model.py train_one_step 内分叉 ——
#    if rand() < p_DeNS(0.3-0.5):  # 对照 DenoisingForcesTrainer.train 的 prob 分叉
#        batch_pos = add_gaussian_noise(pos, sigma~U{0.1,0.3}, corrupt_ratio r=0.3-0.5)  # 对照 add_gaussian_noise_to_position / _schedule_
#        loss_dens = hybrid_loss(noise_pred, noise_target, noise_mask)  # 对照 compute_atomwise_denoising_pos_and_force_hybrid_loss
#        total += lambda_DeNS(0.1-0.3) * loss_dens  # 对照式 (6)
#    力编码按式 (5)：x_f = SO3_Linear(||f||·Y(f/||f||))，无真值力时用 MACE 伪力离线替代。

# 4) 评估分离：test_one_step 新增 dens_* 指标分支，对照 denoising_pos_eval 按 noise_mask 分离三指标，
#    与现有 loss_shape/scale/gap/sum 分离（model.py:216-226）风格统一。
```

**落地顺序**：P0（1 天）→ 能量条件 + prompt（3–5 天，M6）→ 轻等变 encoder（1 周，M6 并行分支）→ DeNS（3–5 天，M7）→ 硬投影 + 加权损失（2–3 天，M8）。每步均有独立回滚开关，总工期 3 周内可收割 M6–M8 全套证据。

---

*报告生成方式：论文 HTML 全文已实际抓取并提取公式/表格/结论数字；官方仓库已通过 GitHub API + raw 抓取验证文件存在并精读关键函数。如需复核，沿文中链接逐条点击即可定位到公式号与代码文件。*
