# uniARPAT 下一代突破路线：能量条件连续谱算子（ECSO）与几何基础预训练

- **研究员**：Muse Spark（AI for Science / 前沿深度学习资深研究员）
- **日期**：2026-09-08
- **保存路径**：`uniARPAT/docs/04_前沿探索/2026-09-08-muse-spark-连续谱神经算子与几何基础预训练.md`
- **目标读者**：上海大学材料基因组工程研究院计算机系硕士生（学位论文第三章 + 第二篇顶刊）
- **与同目录姊妹报告的关系**：同目录已有一篇 `2026-09-08-Gemini-3.1-Pro-等变流匹配与物理混合专家联合能谱生成机制.md`，主攻“等变编码器 + 从噪声出发的条件流匹配生成 + 物理 MoE”。本报告是**正交互补**的另一条主线：**不推翻现有 Shape-Scale 盲测链路**，而是把“离散格点回归”升级为“连续谱算子回归”，把“从零训练的几何编码器”升级为“去噪基础预训练编码器”，把“单点预测”升级为“矩形流精修 + 不确定性门控 + 逆向设计”。两条路线可独立成文，也可合并为 uniARPAT-v3/v4 的两阶段演进（本报告 = v3 确定性升级，姊妹报告 = v4 生成式升级）。

---

## 目录

1. [执行摘要：为什么是“连续谱算子 + 基础预训练”](#1-执行摘要为什么是连续谱算子--基础预训练)
2. [代码审计：当前架构的五个瓶颈（附行号）](#2-代码审计当前架构的五个瓶颈附行号)
3. [前沿全景 2024–2026：四类可迁移机制](#3-前沿全景-20242026四类可迁移机制)
4. [旗舰路线 R1：能量条件连续谱算子解码器（ECSO）](#4-旗舰路线-r1能量条件连续谱算子解码器ecso)
5. [路线 R2：几何基础预训练（去噪 + 对比 + 物理引导 + 教师蒸馏）](#5-路线-r2几何基础预训练去噪--对比--物理引导--教师蒸馏)
6. [路线 R3：矩形流谱精修、不确定性门控与逆向设计](#6-路线-r3矩形流谱精修不确定性门控与逆向设计)
7. [路线 R4：效率与可靠性现代化（低风险高确定性）](#7-路线-r4效率与可靠性现代化低风险高确定性)
8. [统一训练协议与总损失重写](#8-统一训练协议与总损失重写)
9. [算力与显存开销定量评估](#9-算力与显存开销定量评估)
10. [失效风险矩阵与回滚判据](#10-失效风险矩阵与回滚判据)
11. [路线图：M6–M10 消融、论文章节映射与期刊故事线](#11-路线图m6m10-消融论文章节映射与期刊故事线)
12. [反夸大边界与可复现清单](#12-反夸大边界与可复现清单)
13. [参考文献](#13-参考文献)
14. [附录 A：最小代码改动草图](#附录-a最小代码改动草图)
15. [附录 B：下游敏感性加权函数的物理推导](#附录-b下游敏感性加权函数的物理推导)

---

## 1. 执行摘要：为什么是“连续谱算子 + 基础预训练”

### 1.1 一句话立意

> **把 uniARPAT 从“预测 128+64 个离散格点”重构为“学习从晶体结构到连续谱函数 $G:\ \mathcal{C} \to (E \mapsto y(E))$ 的神经算子”，并用大规模无标注晶体去噪预训练解决“未弛豫输入分布偏移”这一根本难题。**

这一定位同时满足“计算机算法创新深度”（神经算子 + 几何自监督是 2024–2026 顶会核心议题）与“材料能谱真实物理增益”（尖峰保真度、带隙分类、声学支权重直接决定 $C_v/\Theta_D/\kappa_L$/Seebeck 下游精度）。

### 1.2 为什么这是当前最值得攻关的主题（四条判据）

| 判据 | 本报告路线 | 姊妹报告路线（EFM+MoE） | 判断 |
| :--- | :--- | :--- | :--- |
| 是否直击 eDOS 失败率 14.66% 的主因（尖峰抹平、带隙伪态） | ✅ 是：逐点能量条件 + 谱偏置 trunk + 分布损失，直接建模 van Hove 奇点 | ✅ 是：生成式 ODE 保留高频 | 两条都打中，但本路线**保留现有回归评估口径**（$R^2$/MAE 可直接对比 M1–M5），审稿人验证成本最低 |
| 是否解决“未弛豫输入”这一论文独特性 | ✅ 是：DeNS 非平衡去噪预训练正是为弛豫轨迹中间态设计的 | ⚠️ 部分：等变编码器对畸变更敏感，但仍需标注数据 | 本路线更对口 |
| 工程风险与单卡 V100 可行性 | ✅ 低–中：R1 甚至**减少**参数，R2 为一次性预训练 | ⚠️ 中：ODE 采样 + 高阶张量积，FP16 下 NaN 风险 | 本路线先行，姊妹路线随后，符合 5–6 周时间表 |
| 学位论文章节增量 | ✅ 第三章方法创新（算子视角）+ 第四章应用（超分/外推/不确定性），每节都有独立消融证据 | ✅ 第四章生成式应用 | 两者合并即完整 v3+v4 拼图 |

### 1.3 预期增益（诚实区间，非承诺）

基于 DOSTransformer（能量嵌入在所有基线上一​​致带来增益，尤其 OOD）、Mat2Spec（SumNorm-KL 在带隙分类上显著优于 WD/MSE）、FEDONet（混沌/宽频系统相对 $L^2$ 误差从 76%→21% 量级改善，但那是 PDE 基准，不可直接平移）、DPF/CDSSL（2–10% 全量提升，低数据区 up to 10%）等已发表证据，按保守折扣给出：

- **eDOS 中位数 $R^2$**：0.521（M1 Oracle）→ **0.60–0.68**（R1+R2 联合，Oracle 口径）；Blind 口径同步跟进，Blind–Oracle gap 维持 ≤0.06。
- **eDOS 失败率（$R^2<0$）**：14.66% → **<9%**（主要来自带隙 F1 提升 + 尾部分布损失）。
- **phDOS 中位数 $R^2$**：0.694 → **0.74–0.78**（声学支加权 + 求和规则硬投影）。
- **下游**：$\Theta_D$ MAE 41.09K → **<32K**；$C_v$ MAE 0.364 → **<0.28 J/(mol·atom·K)**；$\kappa_L$ 相对排序（Spearman）显著提升（Slack 公式对 $\Theta_D$ 是立方放大，见附录 B）。
- **外推能力**（新增证据维度）：晶系留一（OOD）eDOS $R^2$ 相对提升 **15–30%**（DOSTransformer 在 OOD 下能量嵌入增益更大的规律）。

> 以上为基于文献外推的**先验区间**，Week-2/3 消融若 eDOS Oracle 中位数 $<0.55$ 即触发 §10 回滚预案，不得虚报。

---

## 2. 代码审计：当前架构的五个瓶颈（附行号）

审计基线：`model/transformer.py`、`model/heads.py`、`model/model.py`、`datasets/dataset.py`、`thermo_props.py`；数据规模：train 10965 / valid 1371 / test 1371，原子位宽 80（`elements_*.npy` 形状 `[N, 82]`，前 2 列为晶格参数行，见 `transformer.py:124-129`）。

### B1. 谱轴被当作“离散通道”，模型不知道“能量坐标”的存在

- 现状：`transformer.py:84-89` 的 `edos_query_embed/phdos_query_embed` 是**与能量值无关的可学习静态向量**（`torch.zeros(128/64, 512)`）；`heads.py:33-82` 的两个卷积头在谱轴上做局部平滑；`model.py:15-31` 的形状损失对所有 bins 等权。
- 后果（与基线数据自洽）：eDOS 均值 $R^2$（0.374）远低于中位数（0.521），标准差 0.65——典型**长尾尖峰拟合失败**分布；Pilot 中 eDOS 在第 2–5 轮卡在 MAE≈3.72 平台（多尺度卷积迟迟学不会奇点定位）。
- 前沿对照：DOSTransformer（NeurIPS 2023，Heewoong Noh et al.）证明**把能量作为第二模态输入、逐点预测 $y(E_j)$** 在所有基线（MLP/GN/E3NN）上一致带来增益，且 OOD 下增益更大；其消融指出 E3NN 增益小恰是因为朴素拼接能量会干扰等变性——这对 uniARPAT（RP 几何偏置 + 非等变主干）是利好：我们没有严格等变性包袱，能量条件可大胆加。

### B2. Encoder 几何偏置存在“双算浪费”与表达天花板

- 现状：`transformer.py:283-356` 的 `TransformerEncoderLayer.forward` 先调用 `self.self_attn` 得到 `attn_output`（第 323 行），随后在第 325–346 行**完全重算一遍** QK 分数并加上 `rp_scores`，第 323 行的结果被丢弃。QKV 投影与第一次 softmax 的 FLOPs/显存被白白浪费（约 1.5–2× 注意力开销）。
- 表达层面：`RPEncoding(lmax=2)` → `out_dim = 64×9 = 576` → `rp_proj` 到 512（第 310–311 行），每个 batch 的 pair 表 `[B, L, L, 576]`（fp32 下 batch=32、L=80 时约 **470MB**，是 9.16GB 峰值显存的最大单一来源之一）；同时 `rbf_encoder/rel_proj/dir_proj`（第 304–308 行）初始化后**从未在 forward 中使用**（死代码），`max_ell=3` 的 `dir_proj` 亦然。几何通道“ paying full price, getting half value”。
- 物理层面：标量距离 + 球谐方向的加性偏置对**未弛豫结构的微小畸变**（键长 ±0.1Å 可致带隙 eV 级移动）分辨率不足；原子特征仅 3 维（质量/半径/电负性，`atom_feature.py`），缺失价电子数、族/周期、d 电子计数等 DOS 强相关量（2025 年 GT-PDOS 工作证实 s/p/d/f 价电子计数是显著增益特征）。

### B3. Shape-Scale 解耦正确但“标度回归”仍是单点脆弱估计

- 现状正确之处（应保留）：`safe_shape_norm`（第 17–28 行）的 ReLU 主路 + Softplus 自愈退避是对的；`ScaleHead` 偏置 `[3.0, -1.5]` 经验初始化是对的；Blind–Oracle gap 仅 0.05（Pilot 第 10 轮）证明链路成立。
- 脆弱之处：`model.py:173-176` 用 `log(clamp(y_max, 1e-4, 500))` 做 Huber 回归——**整个谱的绝对标度被压缩为单一标量 $y_{\max}$**，对绝缘体（谱系 legitimately 接近平坦、$y_{\max}$ 小且噪声敏感）和离群大峰样本（截断到 500）都不鲁棒；且形状损失 Pearson 项对尖峰位置偏移 1–2 bins 的惩罚是灾难性的（相关系数崩塌），而物理下游（积分量）对此类偏移其实不敏感——**损失函数与物理目标错位**。

### B4. 损失函数“等权所有 bins”，与热力学/输运敏感性错位

- 现状：`model.py:201` 总损失对 128/64 bins 等权；`thermo_props.py` 的 $C_v$ 核函数 $(x/2)^2/\sinh^2(x/2)$ 高度集中在低频声学支，$\Theta_D$ 是二阶矩（$\langle\nu^2\rangle$），Slack $\kappa_L \propto \Theta_D^3$ 把高频误差立方放大；eDOS 的 Seebeck 相关量是费米面斜率 $|d\ln g/dE|$。等权 MSE 把容量浪费在物理不敏感区。
- 数据证据：$C_v$ 相对误差仅 1.4%（per-atom），说明积分量对形状误差有平均效应——反过来，只要**按敏感性加权**训练，就能以很小的 $R^2$ 代价换取大的下游增益，这是“物理增益”的最低垂果实。

### B5. 无预训练、无不确定性、无连续分辨率

- 现状：编码器从零训练（10965 样本对 76M 参数，约 144 样本/百万参数，典型的过参 regime）；`_reset_parameters` 用 Xavier 全权重随机化（第 117–120 行），几何知识全靠监督信号从 DOS 反推，OOD 晶系必然崩（失败率 7–15% 且无检测）。
- `dataset.py` 的归一化选项（minmax/zscore/log/scale_factor） excellent，但**没有未弛豫→弛豫的增广**、没有晶系分层采样；`test_cif` 分支全零 target 意味着盲测链路已预留，只缺不确定性门控。
- 分辨率锁定 128/64：无法利用高分辨率 DFT 谱（如 MAD 的 4806 点、MP 新版 401 点）做联合训练，也无法做谱超分——而 2026 年 PET-MAD-DOS / DOSSIER 已展示跨分辨率/跨数据集统一 DOS 建模是可行且加分的。

---

## 3. 前沿全景 2024–2026：四类可迁移机制

### F1. 连续物理表征：神经算子（DeepONet / FNO / FEDONet）

- **来源**：DeepONet（Lu et al. 2019）分支-主干定理；FNO（Li et al. 2020）谱卷积；**FEDONet**（Sojitra et al. 2025/2026 JCP）：trunk 加随机傅里叶嵌入 $\gamma(v)=[\sin(2\pi Bv),\cos(2\pi Bv)]$，在 Burgers/KS 等宽频混沌系统上相对 $L^2$ 大幅下降（KS：76%→21%），机理是**谱预条件**（缓解 MLP 谱偏置）+ 严格扩大假设空间；LFNO（NeurIPS 2025 ML4PS）把瞬态/稳态分解与傅里叶稳态算子结合；IS-FNO（2025）用近可逆 lifting/projection 稳定长程 rollout。
- **迁移点**：DOS 正是“输入函数（结构）→ 输出函数（谱）”的算子学习问题；R1 几乎逐字搬运 FEDONet 的 trunk 嵌入 + DeepONet 分支-主干结构，只需把 PDE 的时空坐标换成**能量/频率坐标**。

### F2. 生成式精修：流匹配（Flow Matching / Rectified Flow）

- **来源**：Lipman et al. 2022–2024 条件流匹配；材料侧 **FlowMM**（Miller et al. 2024，CSP 上 50 步达到 DiffCSP 1000 步的 match rate）、**CrystalFlow**（Nat. Commun. 2025，~10× 推理加速）、**MatterGen**（Nature 2025，S.U.N. 翻倍，RMSD 降 5–10×，adapter 条件微调范式）、**FlowLLM**（2024，LLM 先验 + RFM 精修：稳定性 +300%，S.U.N. +50%，弛豫步数 192→38）、DMFlow（2026，无序材料黎曼流匹配）。
- **迁移点**：不做“从噪声生成谱”（那是姊妹报告的 EFM 路线），而做 **“以确定性预测为均值的矩形流精修”**（rectified-flow refiner）：训练是无仿真的单步回归（便宜），推理是 10–20 步 ODE（可接受），天然给出**采样集成不确定性**与**性质条件引导**（classifier-free guidance 指向低 $\kappa_L$/目标带隙/d 带中心）。

### F3. 数据工程预训练：去噪 + 对比 + 物理引导 + 教师蒸馏

- **来源**：**DPF**（AAAI 2025：38 万无标注晶体上掩码原子类型 + 扰动位置/晶格，JARVIS/MP 全面提升，小数据 bulk/shear modulus 增益最大）；**CDSSL**（2024：加噪边距离重构，49 配置中 37 胜）；**DeNS**（Liao et al. 2024：把去噪推广到**非平衡结构** + 力编码，OC20/OC22 SOTA，**正是未弛豫输入的对口解法**）；**DSSL**（2024：掩码生成 + 对比全局 + 微观性质预测 valence-electron/atomic-stiffness 物理引导）；**UNATE**（2026：去噪自编码 + 对比，嵌入转移而非权重转移，低数据区 +10%，权重直接迁移反而 -106%——重要警示）；**PET-MAD-DOS**（Digital Discovery 2026：无约束 transformer + MAD 通用 DOS，bespoke 仅 2× 更优，LLPR 系综 UQ）；**DOSSIER**（2026：化学语言模型 + UMIP 教师跨模态蒸馏，1k 样本下 -11% 误差；Mat2Spec 基准 R2 0.53–0.58 区间是我们的对标锚）。
- **迁移点**：uniARPAT 编码器用 MP/Alexandria/GNoME 无标注结构做三任务去噪预训练；DeNS 思想处理未弛豫（把“弛豫轨迹中间帧”当作天然非平衡数据）；DOSSIER 式蒸馏用 MACE/MatterSim/PET-MAD 嵌入做教师；UNATE 警示决定我们采用“**嵌入转移 + 低 LR 全量微调**”而非冻结。

### F4. 序列与架构效率：Transformer++ 处方 + 状态空间模型的启示

- **来源**：Mamba/Mamba-2/Mamba-3（2023–2026：选择性 SSM、复数状态追踪、MIMO 提高解码算术强度；LongMamba/MambaExtend 揭示全局通道衰减机制）；Scaling Laws for Neural Material Models（2025：**EquiformerV2 持续领先无约束 transformer**，显式对称性在材料领域不是可有可无）；Mat2Spec（Nat. Commun. 2022：**SumNorm-KL 在带隙分类上最好**，WD 反而被尖峰拖垮——直接指导 R1 的损失选择）。
- **迁移点**：不盲目换 Mamba（谱轴才 128/64，二次注意力根本不是瓶颈；瓶颈是 pair 表，见 §9），而是取 Transformer++ 处方中**零风险**的部分（SwiGLU/RMSNorm/ fused SDPA），以及 Mamba 文献的**全局/局部分通道诊断思想**（定位 eDOS 尖峰 vs 基线分别由哪些通道承载，指导 MoE-adapter 设计）。

---

## 4. 旗舰路线 R1：能量条件连续谱算子解码器（ECSO）

### 4.1 核心思想（一句话）

> **删除“静态 query 嵌入 + 整谱卷积头”，改为“晶体分支（branch）× 能量主干（trunk）的逐点预测”**：对任意能量坐标 $E$，$\hat{y}(E) = \langle b(\text{crystal}),\, t(E) \rangle + \text{head}(b, t)$。模型从此知道“它在预测哪个能量”，且可在任意网格（128/401/4806）上查询/训练/超分。

这正是 DOSTransformer“多模态能量嵌入 + 逐点预测”与 FEDONet“分支-主干 + 傅里叶 trunk”的交汇点，也是与姊妹报告 EFM 路线最本质的区别（本路线是**确定性算子**，对方是**随机性生成**）。

### 4.2 架构拓扑

```text
[晶体结构 (Atoms + Coords + Cell)]
                │
                ▼
[几何编码器 Encoder（保留，R2 预训练后）]  ── memory H ∈ R^{B×L×512}
                │
                ├──► [全局池化 h_crys] ──► [ScaleHead（保留，盲测标度）]
                │
                ▼
[晶体分支 Branch Net]                       [能量主干 Trunk Net ×2（eDOS/phDOS 独立）]
h_crys ──► b_e, b_p ∈ R^{B×r}              E_j ──► γ(E_j) ──► t_e(E_j), t_p(E_j) ∈ R^{r}
(r=256，见 §4.3)                            γ = 随机傅里叶嵌入 + 可学习残差（见 §4.3）
                │                                         │
                └─────────────┬───────────────────────────┘
                              ▼
              [逐点融合头：点积 + 小 MLP 残差]
              ŷ_shape(E_j) = σ( b·t(E_j)/√r + MLP([b; t(E_j); b⊙t(E_j)]) )
                              │
                              ▼
              [物理投影层（硬约束，见 §7/R4）]
              非负 / phDOS 首点置零 / 可选 3N 单纯形投影
                              │
                              ▼
              [连续谱函数，可在任意网格查询]
              训练网格 128/64 ＋ 随机连续采样 E~U（见 §4.4）
```

**与现有代码的对应关系**（最小侵入）：

1. 删除 `transformer.py:84-89` 的静态 `edos_query_embed/edos_tgt/phdos_query_embed/phdos_tgt`（或保留其作为 branch 的初始化先验，见风险 R-anchoring）。
2. 保留解耦双 Decoder，但其 cross-attention 的 query 从“静态谱槽位”改为“**能量嵌入序列**”：`query_pos = TrunkNet(E_grid)`，`tgt = learned_energy_value_emb(E_grid)`。即 DOSTransformer 的 cross-attention 能量-原子交互， migrant 到现有 `TransformerDecoderLayer`（第 358–399 行）——**Decoder 代码零改动，只换输入**。
3. 替换 `heads.py` 的整谱卷积头为逐点头（参数见 §9：不增反减）。
4. `safe_shape_norm` 从“整谱 max 归一”改为**可选**（见 §4.5 的可辨识性讨论）：连续算子下形状/标度解耦有更干净的数学形式。

### 4.3 数学机理

**（a）分支-主干算子（DeepONet 形式）。** 设晶体 $c$ 的分支向量 $b(c) \in \mathbb{R}^r$，能量 trunk $t(E) \in \mathbb{R}^r$，则谱算子

$$
G_\theta(c)(E) \;=\; \sum_{k=1}^{r} b_k(c)\, t_k(E) \;+\; \phi\big([b(c);\, t(E);\, b(c)\odot t(E)]\big),
$$

其中 $\phi$ 为 2 层残差 MLP（256→256→1，GELU）。第一项是秩-$r$ 全局基展开（捕捉谱包络、带隙位置、声学/光学分支划分），第二项拟合局部残差（尖峰高度、精细劈裂）。$r=256$ 与当前 Head 隐层 256 对齐，容量可比，消融公平。

**（b）傅里叶 trunk（FEDONet 谱预条件）。** 能量坐标先归一化到 $[-1,1]$（eDOS：$E/10$；phDOS：$2(\nu+280)/1260-1$），再做：

$$
\gamma(E) = \big[\sin(2\pi B E),\; \cos(2\pi B E),\; E\big], \quad B_{ij} \sim \mathcal{N}(0, \sigma^2),
$$

$B \in \mathbb{R}^{m \times 1}$（$m=128$，$\sigma$ 为可调带宽，eDOS 取大 $\sigma\approx 8$ 以分辨尖峰，phDOS 取小 $\sigma\approx 2$ 以偏好平滑——**双带宽是关键超参**，见 §10），后接 $t(E) = W_2\,\text{GELU}(W_1 \gamma(E)) + W_{\text{skip}} E$。机理：普通 MLP 有谱偏置（优先学低频），随机傅里叶嵌入把高频基显式放入假设空间，FEDONet 的 NTK 分析证明这严格扩大可表函数类并改善条件数——对应到 DOS 就是 **van Hove 尖峰不再被抹平**，且 trunk 只有 ~0.2M 参数（见 §9）。

**（c）谱相关头（可选 FNO-1D 残差）。** 在逐点预测后，对整条谱做一次轻量傅里叶残差修正（mode=16，width=64，~0.1M 参数）：

$$
\hat{y} \leftarrow \hat{y} + \mathcal{F}^{-1}\big(R \cdot \mathcal{F}(\hat{y})\big)_{|\omega|\le 16} + \text{Conv1d}_{k=3}(\hat{y}),
$$

捕捉“峰-峰关联”（如 eDOS 成键/反键峰对、phDOS 声学-光学带隙），FNO 理论保证分辨率不变性（换网格不用重训）。

**（d）能量-原子 cross-attention（DOSTransformer 机制）。** 现有 `TransformerDecoderLayer.multihead_attn`（第 391–393 行）天然就是能量 query 对原子 memory 的 cross-attention；只需把 `query_pos` 设为 trunk 嵌入。注意力权重 $A(E_j, \text{atom}_i)$ 本身是**可解释的**：费米面处高权重原子 = 对带隙贡献最大的 Wyckoff 位——可直接画 Fig（论文卖点之一）。

### 4.4 训练：连续采样 + 分布损失 + 物理加权（三项缺一不可）

**（a）连续能量采样（超分训练）。** 每个 batch，除固定网格（128/64，保证与 M1–M5 可比）外，额外采样 $K=32$ 个**连续随机能量** $E' \sim \mathcal{U}$，目标值由 DFT 谱三次样条插值得到（数据 pipeline 一次性预处理，无 DFT 重算）。损失同时作用于固定网格 + 随机点。这带来两项免费增益：① 模型见过“峰之间”的位置，峰位偏移 1 bin 不再灾难；② 推理时可输出 401/4806 点高清谱（与 PET-MAD-DOS/DOSSIER 的跨分辨率范式接轨）。

**（b）分布损失（Mat2Spec 教训：KL > WD > 纯 MSE 用于带隙）。** 在 SumNorm 分布空间 $p = y/\sum y$ 上：

$$
\mathcal{L}_{\text{spec}} = \underbrace{\text{KL}(p_{\text{true}} \| p_{\text{pred}})}_{\text{带隙/小值敏感，gap 分类主力}} + \lambda_{\text{w}} W_1(p_{\text{true}}, p_{\text{pred}}) + \lambda_{\text{mse}} \text{MSE}_{\text{shape}},
$$

- KL 对小值误差惩罚重 → 带隙伪态被强力压制（Mat2Spec 表 2：SumNorm-KL 的 gap F1 最好）；
- $W_1$（一维 Wasserstein = 排序后 CDF 差的 $L^1$，`torch.cumsum` 可微实现，$O(n\log n)$ 可忽略）对**峰位整体平移**鲁棒，弥补 Pearson 对 1–2 bins 偏移的过度惩罚；
- 保留 MSE 项锚定峰高。三者权重初值 $(1.0, 0.3, 0.5)$，eDOS/phDOS 外层再乘现有 (1.0, 3.0) 任务权重。

**（c）下游敏感性加权（附录 B 推导）。** 逐 bin 权重 $w(E)$ 取下游核的归一化包络：

- phDOS：$w_{\text{ph}}(\nu) \propto \alpha \cdot K_{C_v}(\nu; 300\text{K}) + \beta \cdot \nu^2 + \gamma$（$K_{C_v}$ 是 $C_v$ 被积核，$\nu^2$ 是 $\Theta_D$ 二阶矩核，$\gamma=0.3$ 保底 uniform 防塌）；
- eDOS：$w_{\text{e}}(E) \propto \alpha \cdot \mathcal{N}(E; 0, 1.5\text{eV}) + \beta \cdot |\nabla_E g_{\text{true}}|_{\text{norm}} + \gamma$（费米窗 + 陡峭处加权，Seebeck 斜率相关）。

最终逐点损失 $\sum_j w_j \ell_j / \sum_j w_j$。这是“用物理换 $R^2$”的最诚实杠杆：即使 uniform $R^2$ 只 +0.03，加权下游指标可 +10–20%。

### 4.5 Shape-Scale 在连续算子下的干净数学形式（解决 B3）

连续谱下不可辨识性 $y(E) = s \cdot \phi(E)$（$s>0$，$\max\phi=1$）依然存在，但可做得更干净：

1. **形状头输出后做连续 max 归一**：$\phi(E) = \text{ReLU}(f(E))/(\max_{E\in\text{grid}} \text{ReLU}(f(E)) + \epsilon)$（`safe_shape_norm` 逻辑逐点保留，Softplus 自愈保留）；
2. **标度头不变**（`ScaleHead` + 全局池化），但把回归目标从单一 $y_{\max}$ 改为**双锚点** $(\log y_{\max},\, \log \bar{y}_{>0})$（峰值 + 非零均值），Huber 损失对两者求和——绝缘体（$y_{\max}$ 小）靠均值锚稳定，大峰样本（截断 500）靠峰值锚定位，双锚互为备份；
3. 训练时对形状用**谱插值后的连续 max**（避免网格 max 的采样抖动），推理时用查询网格 max。Blind–Oracle 双轨评估（`model.py:309-343`）原样保留，新增第三轨 **Blind-HR（高分辨率盲测）**：在 401 点插值网格上评 WD/KL，证明连续性不是摆设。

### 4.6 预期增益与消融位（M6）

- **M6 = M5 + ECSO（Decoder 层数 6→4，卷积头→逐点头+傅里叶 trunk）**：参数**减少** ~8–10M（见 §9），若 eDOS Oracle 中位数 $R^2$ +0.05 且带隙 F1 +0.1，即证“能量条件”的独立增益（与容量无关——因为容量反而小了，结论更保守有力，沿用 Table 1 的“保守判定”逻辑）。
- 物理增益：峰位 MAE（top-5 峰位置误差）与 gap F1 应先于 $R^2$ 改善——要求 Fig 中必须同时报告这三者，防止 $R^2$  washing。

---

## 5. 路线 R2：几何基础预训练（去噪 + 对比 + 物理引导 + 教师蒸馏）

### 5.1 为什么预训练是“未弛豫输入”问题的对口解法

uniARPAT 的输入是**未弛豫结构**，监督目标（DFT 谱）却对应**弛豫后电子/振动基态**。模型必须隐式学会“弛豫算子”。DeNS（Liao et al. 2024）的核心洞见正是：**弛豫轨迹上的中间帧就是天然的非平衡训练数据**，用“加噪→去噪 + 力编码”可显式 teaches 模型势能面几何。OMat24（Boltzmann rattled + AIMD + rattled relaxed）的存在使大规模非平衡数据触手可及。DPF（38 万结构）与 Scaling Laws（2025）的结论共同指向：**材料领域 scaling law 依然有效，且对称性显式建模在同等规模下占优**——我们的 RP 偏置正好是“弱对称性归纳”，预训练可将其推向“近似等变”而不付 eSCN 的复杂度税。

### 5.2 三任务预训练协议（作用于 Encoder，不动 Decoder/Heads）

```text
[无标注晶体 (MP 15万 + Alexandria 子集 + GNoME/OQMD 去重，~30-40万)]
        │
        ├── Task A: 掩码原子类型重构（mask 15%，DPF/UNATE 协议）
        │     输入：原子类型 mask + 真实位置/晶格 → 输出：118 类交叉熵
        │     （DPF 证明这是最稳定、迁移最强的单任务）
        │
        ├── Task B: 坐标/晶格去噪（CDSSL/DeNS 协议）
        │     输入：位置加噪 σ=0.1-0.5Å（多尺度采样）+ 晶格加噪
        │     输出：逐边距离重构 L2（CDSSL 式，避免直接坐标回归的规范问题）
        │     若有力标签（OMat 子集）：加 DeNS 力编码分支（可选）
        │
        └── Task C: 物理引导微观性质 + 对比（DSSL/UNATE 协议）
              C1: 价电子数/原子刚度回归（DSSL 的 VE/AS，先验：带隙↔价电子，弹性↔刚度↔声子）
              C2: 同结构双扰动 InfoNCE（全局不变性，温度 τ=0.1）
```

总预训练损失 $\mathcal{L}_{\text{pre}} = \alpha\mathcal{L}_{\text{mask}} + \beta\mathcal{L}_{\text{denoise}} + \gamma\mathcal{L}_{\text{micro}} + \delta\mathcal{L}_{\text{cl}}$，初值 $(1.0, 1.0, 0.5, 0.2)$（DSSL 消融：mask 与对比是互补的，单对比不行）。

### 5.3 教师蒸馏（DOSSIER 式跨模态，低成本高回报）

- 教师：开源 MACE-MP-0（或 MatterSim 开放 checkpoint）的原子嵌入 / PET-MAD 的结构嵌入，**离线**提取 10965+α 训练结构的教师向量（一次性，无需教师推理进训练循环）。
- 学生（uniARPAT Encoder 全局池化 $h_{\text{crys}}$）用余弦 + MSE 蒸馏对齐（DOSSIER 在 1k 样本下 -11% 的证据）。成本：几乎零显存增量（预计算 npy），+1 个 512→teacher_dim 投影头（~0.3M 参数，微调后可删）。
- 若教师不可用（版本/pytorch 冲突，见 MOFSimBench 2025 综述的 e3nn/PyTorch 地狱警示），退化为纯自监督三任务——**蒸馏是可选插件，不是关键路径**（风险隔离）。

### 5.4 知识转移策略（UNATE 警示的直接应用）

UNATE（2026） ablation：嵌入转移 +2.32%，权重直接初始化 -106%（任务错配导致灾难）。对策：

1. **不断层转移**：预训练后**全量微调**（不冻结），但 Encoder 用 **0.2× LR**（discriminative LR：encoder 1e-5，decoder/heads 5e-5），warmup 5 epochs（沿用 `config.yaml` 余弦 + warmup 协议）；
2. **回放锚**：微调前 10 epochs 保留 5% 预训练去噪损失作正则（multi-head replay，MACE fine-tuning 文献策略），防灾难遗忘；
3. **诊断门**：若 valid 上 mask 重构准确率在微调后塌（proxy：取 encoder 输出训线性 probe），触发回滚到“嵌入转移”（冻结 encoder，只训 decoder+heads+ECSO）——UNATE 证明这仍有 +2–3%。

### 5.5 数据管线（与现有 `dataset.py` 的衔接）

- 预训练 Dataset 复用 `Dos_Dataset` 的 elements/positions 读取（只用前两项，target 置空，走 `test_cif` 分支模式），增广（mask/noise/双扰动）在 collate 中在线做，不落地、不污染现有 npy。
- 去重：用 `train_index/valid_index/test_index.npy` 对 MP/Alexandria 做结构指纹去重（composition + space group），防止预训练泄漏 test 1371（**泄漏检查必须写入论文附录**）。
- 晶系分层：7 大晶系分层采样（呼应 DOSTransformer 的晶系 prompt 思想），低对称三斜/单斜过采样 2×（失败率重灾区）。

---

## 6. 路线 R3：矩形流谱精修、不确定性门控与逆向设计

### 6.1 定位：与姊妹报告 EFM 的分工（重要，避免评审质疑“换汤不换药”）

| 维度 | 姊妹报告 EFM（生成式） | 本报告 R3（精修式） |
| :--- | :--- | :--- |
| 起点分布 | 高斯噪声 $X_0\sim\mathcal{N}(0,I)$ | **R1 确定性预测** $\hat{y}_{\text{base}}$ 为条件均值，流只学残差分布 |
| ODE 长度 | 50–100 步（需蒸馏） | **8–20 步**（残差流路径短，可用 rectified straight path） |
| Shape-Scale 链路 | 需重建（生成后投影） | **完整保留**（精修作用于 shape 空间 $[0,1]$，scale 不动） |
| 不确定性 | 采样多样性 | **采样集成方差** + conformal 区间（可直接用于审稿人要的 error bar） |
| 逆向设计 | 从性质生成结构 | **从目标谱/性质检索+引导精修**（更务实，见 §6.4） |

两者是“v4 生成”与“v3.5 精修”的关系，可同论文两节，也可分两篇（本报告 R3 单独即一篇 ML4PS/NeurIPS workshop + 期刊方法节）。

### 6.2 数学机理（条件矩形流，Conditional Rectified Flow）

固定晶体条件 $c$（R1 的 $h_{\text{crys}}$ + branch 向量），在 shape 空间学时变向量场 $v_\theta(y_t, t \mid c)$，$t\in[0,1]$：

$$
y_t = (1-t)\, y_0 + t\, y_1, \quad y_0 \sim \mathcal{N}(\hat{y}_{\text{base}}(c),\, \sigma_0^2 I),\; y_1 \sim p_{\text{data}}(\cdot\mid c),
$$

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,y_0,y_1}\big\| v_\theta(y_t, t \mid c) - (y_1 - y_0) \big\|^2.
$$

- 以基预测为先验均值（FlowLLM“好先验 + RFM 精修”思想的谱版本）：路径接近直线，NFE 8–20 即够（FlowMM/CrystalFlow 的 3–10× 加速证据外推）。
- $v_\theta$ 架构：轻量 1D DiT（4 层，width 256，AdaLN 注入 $(t, c)$，~6–8M 参数），eDOS/phDOS 各一（解耦原则延续）。
- 每步投影：$y_t \leftarrow \Pi(y_t)$（非负 + phDOS 首点零 + 可选单纯形归一），IS-FNO“近可逆/投影稳定 rollout”的思想在采样环路中的对应物。
- 训练成本：无仿真单步回归，**不展开 ODE**，+25–40% 时间（见 §9）；推理：Euler/Heun 10–20 步，每步 ~3ms（V100）→ 额外 30–60ms，仍 <100ms/材料（高通量漏斗可接受）。

### 6.3 不确定性门控（把 14.66% 失败率变成“已知未知”）

1. **系综方差**：对同一晶体采 $N=8$ 条精修轨迹（不同 $y_0$），逐 bin 方差 $\sigma^2(E)$ 即不确定性。valid 集上校准：失败样本（$R^2<0$）的平均 $\bar{\sigma}$ 应显著高于成功样本（AUC 目标 >0.8，否则 R3 判负）。
2. **Conformal 区间**：在 valid 1371 上做 split-conformal（逐 bin 残差分位数），给出 $90\%$ 覆盖的谱带——论文 Fig 可直接画“预测谱 ± 置信带 vs DFT”，这是 Mater. Sci. 审稿人最爱的图之一（PET-MAD-DOS 的 LLPR 系综 UQ 同理）。
3. **部署门控**：$\bar{\sigma} > \tau$ 的材料自动路由到 DFT 精算队列（呼应研发方案 §7 三级漏斗的第 2→3 级接口），$\tau$ 在 valid 上按“拦截 80% 失败、误伤 <20% 成功”选定。高通量场景下这是比单纯提 $R^2$ 更实在的贡献。

### 6.4 逆向设计（务实两步走，不画“结构生成”大饼）

- **Step 1（谱→性质检索）**：用 R1+R3 预测全库谱，计算目标描述符（带隙、费米斜率、$\kappa_L$、d 带中心），Top-k 检索——零新模型，立即可做（DOSSIER 的 NiPt₃ 式筛选范式：1 万二元 + 1.3 万高熵筛选，已知催化剂排进 top 2%）。
- **Step 2（性质引导精修）**：MatterGen adapter 范式下放到谱精修：在 $v_\theta$ 上加 classifier-free guidance $ \tilde{v} = v_{\text{uncond}} + \gamma (v_{\text{cond}(p^*)} - v_{\text{uncond}})$，$p^*$ 为目标性质（如 $\kappa_L<2$、带隙 1–2eV），生成“反事实谱”回答“谱长什么样才有目标性质”，再用谱相似性反查结构候选。**绝不声称**生成了可合成结构（反夸大边界 §12）。

---

## 7. 路线 R4：效率与可靠性现代化（低风险高确定性）

按“021”优先级排序（0 = 立刻做，2 = 可选）：

**P0-1 修复 Encoder 双算浪费**（`transformer.py:323-346`）：删第一次 `self.self_attn` 调用，直接由 QKV 投影 + `base_scores + rp_scores` 走一次 softmax。预期 -30–40% encoder 注意力时间，-0.5–1GB 激活显存，**零精度变化**（数学等价，只是删冗余）。这是全报告 ROI 最高的单行级改动。

**P0-2 Transformer++ 处方（无损）**：FFN GELU→SwiGLU（dim_ff 2048→1536 保持参数近似，门控分支 +50% FFN 参数但 P0-1 省下的预算覆盖），LayerNorm→RMSNorm（解码器），`nn.MultiheadAttention`→ fused SDPA（`F.scaled_dot_product_attention`，V100 无 Flash 但有 memory-efficient kernel；A100/4090 上 Flash-2 全开）。预期 -15–25% 显存，+10–20% 速度（Transformer++ 在 Mamba 论文中即最强基线处方）。

**P0-3 原子特征扩充**（`atom_feature.py`）：3 维 → 12–16 维（+ 价电子总数、s/p/d/f 价电子、族、周期、电离能、共价半径、d 电子计数、金属/非金属标志），`AtomFeatureEncoder` 输入 dim 对应改，proj 层 +~4k 参数可忽略。2025 GT-PDOS 工作的直接证据支持。

**P1-1 晶系 prompt + 轻量 MoE-adapter**（DOSTransformer prompt + DeepSeek-MoE 轻量版）：7 晶系可学习 prompt token 拼到 memory 前（7×512 参数，可忽略）；Decoder FFN 旁挂 4 专家 adapter（每专家 512→128→512，~0.13M×4×12 层 ≈ 6M），top-1 路由，负载均衡 auxiliary loss 0.01。专治三斜/单斜长尾，不碰主干容量口径（adapter 可单独消融）。

**P1-2 硬物理投影层**（把软损失变成硬保证）：`phys_phdos` 输出后做 $3N$ 单纯形精确投影（$\min\|y-\hat{y}\|^2$ s.t. $\sum y\Delta\omega = 3N$, $y\ge 0$，Duchi et al. simplex projection $O(n\log n)$，64 维可忽略）；eDOS 带隙投影（费米窗内若真值判金属则跳过——推理时用辅助带隙分类头 $p_{\text{gap}}$，F1>0.85 才投影，防误伤金属，呼应 `model.py:178-193` 的条件掩码思想从训练延伸到推理）。

**P2 死代码清理**：`rbf_encoder/rel_proj/dir_proj/max_ell`（第 304–308 行）要么接入（ ablation：RP + RBF 双通道求和，+0.5M 参数）要么删除（减 confusion）。建议删除 + 文档记录（除非 M6 消融显示 RBF 通道独立增益）。

---

## 8. 统一训练协议与总损失重写

### 8.1 总损失（M6/M7，R1+R4；R3 精修阶段独立训练）

$$
\begin{aligned}
\mathcal{L} ={}& \underbrace{\mathcal{L}_{\text{KL}}^{\text{e}} + 3.0\,\mathcal{L}_{\text{KL}}^{\text{p}}}_{\text{分布形状（SumNorm-KL 主力）}} + 0.3\big(\mathcal{L}_{W_1}^{\text{e}} + 3.0\,\mathcal{L}_{W_1}^{\text{p}}\big) + 0.5\big(\mathcal{L}_{\text{MSE,w}}^{\text{e}} + 3.0\,\mathcal{L}_{\text{MSE,w}}^{\text{p}}\big) \\
&+ 0.5\,(\mathcal{L}_{\text{scale,e}} + \mathcal{L}_{\text{scale,p}})_{\text{双锚 Huber}} + 0.2\,\mathcal{L}_{\text{gap}} + 0.1\,\mathcal{L}_{\text{sum}} + 0.01\,\mathcal{L}_{\text{bal-MoE}},
\end{aligned}
$$

- 所有形状项在 §4.4(c) 的 $w(E)$ 加权下计算；$\mathcal{L}_{\text{MSE,w}}$ 即现有 `compute_shape_loss` 的 Pearson 部分被 KL/$W_1$ 接管后的残留（保留作峰高锚，权重可 anneal 到 0.2）；
- 多任务权重 (1.0, 3.0) 沿用 M5（phDOS 易学，压住 eDOS 梯度噪声），但引入 **uncertainty weighting**（Kendall et al.：$\mathcal{L} = \sum_i e^{-s_i}\mathcal{L}_i + s_i$，$s_i$ 可学习）做 ±20% 自适应微调，防止 KL 量级变化打破平衡；
- R3 精修损失独立：$\mathcal{L}_{\text{CFM}}$ + 0.1 端点形状锚（防 ODE 漂移）。

### 8.2 优化器与调度（沿用 + 微调）

- AdamW（encoder 1e-5 / 其余 5e-5，R2 预训练后 discriminative LR；从零跑 M6 则统一 5e-5），betas (0.9, 0.99) 沿用 `config.yaml`；
- 余弦 + 5 epoch warmup + min_lr 1e-6 沿用；R1 trunk 的 $B$ 矩阵 LR ×0.1（防高频发散，FEDONet 实践）；
- batch 32 / AMP 沿用；`GradScaler(init_scale=1024)` 在 RMSNorm+SwiGLU 下复测（若 overflow 改 2048）；
- 选优指标：`balanced_score` 保留，**新增 `phys_score = 0.4·R2_e + 0.3·R2_p + 0.15·gapF1 + 0.15·(1−normCvErr)`** 作 checkpoint 第二判据（防 $R^2$ 单一指标 washing，见 B4）。

---

## 9. 算力与显存开销定量评估

### 9.1 基线实测（Week-1 Pilot，V100-32GB，batch=32）

单 epoch 107.8s（343 steps → 0.314s/step），峰值 9.16GB，M5 总参 75.93M。按模块拆分（基于形状的解析估计，±20%，需 `torch.profiler` 实测校准——已列为 M6 首项任务）：

| 模块 | 参数 | 时间占比估计 | 显存大户 |
| :--- | :--- | :--- | :--- |
| Encoder×6（含 RP pair 表） | ~19M + rp_proj 0.3M | ~40%（pair 表 + 双算浪费） | pair 表 ~0.5–1.5GB（峰值） |
| 解耦双 Decoder×2×6 | ~50M | ~40% | cross-attn logits $[B·H, 192, 80]$ 小 |
| 对称卷积双 Head | 1.576M | ~8% | 小 |
| ScaleHead + 其他 | 0.1M + emb ~2M | ~2% | 小 |
| 优化器状态（AdamW fp32 m+v） | — | — | ~0.6GB |
| 框架/碎片/CUDA 上下文 | — | ~10% | ~1–2GB |

### 9.2 各路线增量（batch=32，V100-32GB，同数据）

| 升级项 | 参数 Δ | 时间 Δ/epoch | 显存 Δ | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **R4-P0-1 删双算** | 0 | **−12 ~ −18s**（−11~−17%） | **−0.5 ~ −1.0GB** | 数学等价，必做 |
| R4-P0-2 SDPA+SwiGLU+RMSNorm | +1.5M（FFN 门控） | −5 ~ −10s | −0.5 ~ −1.0GB | V100 上 SDPA 为 mem-efficient 非 flash，A100 上收益翻倍 |
| R4-P0-3 原子特征 3→14 维 | +0.01M | ≈0 | ≈0 | 可忽略 |
| **R1 ECSO（Decoder 6→4 + 逐点头 + 双 trunk + FNO-1D）** | **−8 ~ −12M**（砍 2×2 层 decoder ≈ −17M，加 trunk+头+ FNO ≈ +5–7M + MoE-adapter 另计） | −15 ~ −25s（含 P0-1 叠加后总量 −25 ~ −40s，即 **65–80s/epoch**） | −1.0 ~ −2.0GB（总量 → **~7GB**） | 容量更小 ⇒ 消融结论更保守有力 |
| R4-P1-1 晶系 prompt + 4-expert adapter（top-1） | +6M（激活 +1.5M） | +3 ~ +6s | +0.3 ~ +0.6GB | 可独立开关 |
| R2 预训练（一次性，30–40 万结构 × 20–30 epochs） | 0（复用 encoder） | 一次性 **12–30h** V100（MP-15万 ≈12h，全量 ≈30h，可断点） | 同训练 | 蒸馏 teacher 向量离线预计算，零在线开销 |
| R2 微调（ discriminative LR + 5% replay） | +0.3M（teacher 投影，可删） | +5%（replay 前 10 epochs） | +0.2GB | — |
| **R3 流精修训练**（DiT-1D×2，~12–16M） | +12 ~ +16M | +25 ~ +40%（无仿真单步回归） | +1.0 ~ +2.0GB | 与 R1 解耦训练（冻结 R1） |
| R3 推理（NFE=12 Heun） | — | +35 ~ +70ms/材料（基线 <30ms → **<100ms**） | ≈0（逐 batch） | 高通量漏斗可接受；UQ 系综 ×8 仅对被门控 flag 的尾部触发 |

### 9.3 综合三档配置（给导师/算力管理员的一页纸）

- **保守档（M6：R1+R4-P0，必跑）**：~64M 参数，~7GB，~70s/epoch → 100 epochs ≈ **2h/变体**（比现在 3–3.5h 更快），单卡 V100 无压力，预期 eDOS 中位数 +0.05、失败率 −4~6pts。
- **标准档（M7：M6+R2 预训练微调+P1-1 adapter）**：~70M，~7.5GB，~75s/epoch + 一次性预训练 12–30h，预期 eDOS 中位数 0.60–0.66、OOD +15–30%。
- **完整档（M8：M7+R3 精修+UQ 门控）**：~84M，~9GB，训练 +30%，推理 <100ms，新增置信带 + 失败拦截 80% + 逆向筛选 demo（期刊 Fig 5/6 素材）。

---

## 10. 失效风险矩阵与回滚判据

| # | 风险 | 概率×影响 | 早期信号（第 10 epoch 前可见） | 缓解 / 回滚 |
| :--- | :--- | :--- | :--- | :--- |
| R-a | 傅里叶 trunk 高频发散（eDOS 全谱毛刺，valid loss 爆） | 中×高 | 随机点损失 ≫ 网格损失；谱 TV 爆表 | trunk LR×0.1 + $\sigma$ 减半 + 对 $\|B\|$ 施 L2；回滚：trunk 退化为可学习 sinusoid（固定频率）或纯 MLP 能量嵌入（DOSTransformer 原始形态） |
| R-b | 连续采样插值目标引入伪监督（样条在尖峰间过冲为负/振铃） | 中×中 | 随机点损失不降，网格点正常 | 改 PCHIP 保形插值 + 负值裁零 + 随机点权重 ×0.5；回滚：关连续采样（K=0），R1 退化为固定网格能量条件（仍保留 DOSTransformer 增益主体） |
| R-c | KL 在平坦谱上 NaN/梯度爆（B3 的分布版） | 低×高 | loss NaN（现有 flat fallback 只覆盖 Pearson） | SumNorm 加 $\epsilon=10^{-6}$ + 平坦谱（var<1e-4）自动切换纯 MSE（沿用 `compute_shape_loss` 第 29–30 行逻辑到 KL）；已有代码模式可复制 |
| R-d | $W_1$ 对多峰谱的“峰合并”病理（把双峰拉成单峰以省传输 cost） | 中×中 | 双峰样本 valid WD 降但峰计数错 | $\lambda_w \le 0.3$ 上限 + 峰计数 auxiliary 监控；回滚：$\lambda_w=0$（Mat2Spec 本来就显示 WD 非必需，KL 才是 gap 主力） |
| R-e | 预训练→微调负迁移（UNATE -106% 警示） | 中×高 | 微调 valid 首轮差于 M6 同轮 | discriminative LR + replay 已备；回滚 Ladder：① 关 replay ② encoder LR 再 ×0.2 ③ 冻结 encoder（嵌入转移，仍有 +2–3% 保底）④ 弃预训练（M6 照发） |
| R-f | 预训练数据泄漏 test 1371 | 低×灾难 | 指纹去重报告缺失 | 强制 composition+SG 指纹去重 + 附录披露 + 可复现脚本；泄漏检查不通过不得投稿 |
| R-g | ODE 精修漂移（多步后谱面积/带隙丢） | 中×中 | NFE↑ 反而 WD↑ | 端点形状锚 0.1 + 每步投影 $\Pi$ + Heun 二阶；回滚：NFE=1（单步去噪自编码，仍有集成方差可用）或弃 R3（R1+R2 照发，R3 移 workshop） |
| R-h | Conformal 覆盖不达标（谱带过宽/过窄） | 低×中 | valid 覆盖率 ≠90%±3% | 逐晶系分层 conformal（低对称单独分位数）；不达标则只报系综方差，不报 conformal（降级不断线） |
| R-i | 与姊妹 EFM 路线“撞故事”（审稿人问区别） | 中×中 | — | 本报告 §6.1 分工表 + 联合 roadmap（v3 确定性 → v4 生成式）写入投稿 cover letter；两路线共享 R2 预训练 encoder（一次投入，双倍产出） |

**Week-2/3 硬回滚线**（写进实验记录，不容商量）：100 epochs 后若 M6 eDOS Oracle 中位数 $<0.55$ **且** gap F1 无提升 → 弃 R1 全套，回退 M5 + R4-P0（白捡 15% 加速）+ R2（预训练大概率仍正增益）；若 R2 微调负增益 → 冻结 encoder 重跑一次，再负则弃。

---

## 11. 路线图：M6–M10 消融、论文章节映射与期刊故事线

### 11.1 消融矩阵（接续 Table 1 的 M1–M5，每格都是独立证据）

| 变体 | 改动（相对 M5 Full） | 参数 | 回答的问题 | 成功判据 |
| :--- | :--- | :--- | :--- | :--- |
| **M6** | R1 ECSO（Dec 6→4 + 能量条件 + 傅里叶 trunk + 逐点头 + KL/$W_1$ + 物理加权）+ R4-P0 | ~64M | 能量坐标建模是否是独立增益？ | eDOS 中位 +0.04，gap F1 +0.08（容量更小 ⇒ 保守证据） |
| **M7** | M6 + R2 预训练微调 + 原子特征 14 维 | ~70M | 未弛豫表示能否靠无监督补？ | OOD 晶系留一 +15%，全量 eDOS 中位 0.60+ |
| **M8** | M7 + R4-P1（晶系 prompt + MoE-adapter + 硬投影） | ~76M | 长尾晶系/求和规则能否结构化解决？ | 失败率 <9%，phDOS 面积误差减半 |
| **M9** | M8 + R3 流精修（冻结 base）+ UQ 门控 | ~90M | 残差分布建模 + 已知未知？ | 峰位 MAE −15%，失败拦截 AUC>0.8，conformal 90%±3% |
| **M10** | M9 + 性质引导反事实谱 + 全库筛选 demo | 同 M9 | 逆向筛选闭环？ | Top-k 富集（已知热电/催化剂进 top 2%，DOSSIER 范式） |

> M6–M8 构成**第二篇顶刊正文**（方法 + 消融 + 下游）；M9–M10 构成**第四章应用 + 第三节 Precision 筛选**（UQ + 逆向）；姊妹 EFM 报告对应 **v4 生成式续作**（第三篇）。单卡 V100 上 M6–M8 各 ~2–2.5h/100epochs，M9 精修 ~3h，全套一周内可收割（Week-2/3 窗口内）。

### 11.2 学位论文章节映射

- **第三章（方法篇）增量小节**：3.3 连续谱算子视角（branch-trunk + 傅里叶 trunk 定理陈述）→ 3.4 去噪几何预训练（DeNS 非平衡动机 + 三任务）→ 3.5 物理加权损失与硬投影（敏感性核推导放附录 B）→ 3.6 流精修与不确定性（可选入第四章）。
- **第四章（应用篇）增量小节**：4.3 超分谱（401 点）与跨分辨率证据 → 4.4 OOD 晶系泛化 → 4.5 UQ 门控漏斗 → 4.6 逆向筛选案例（2–3 候选 + DFT 复核接口，沿用研发方案 §7 三级漏斗）。

### 11.3 期刊故事线（一句话卖点 + 对标）

> **“第一个把晶体 DOS 预测重构为连续神经算子、并用非平衡去噪预训练解决未弛豫输入偏移的统一框架”**——对标 DOSTransformer（能量条件）+ Mat2Spec（分布学习）+ PET-MAD-DOS/DOSSIER（通用/跨分辨率）+ FEDONet（谱 trunk），四篇引文各取一机制，融合成一个可消融的系统；姊妹 EFM 作 companion/next-step 引用，评审看到的是“有计划的研究纲领”而非单点 trick。目标刊：Nature Comput. Sci. / npj Comput. Mater.（第二篇）→ Matter/JACS Au（应用）→ ICLR/NeurIPS ML4PS（R3 方法短文）。

---

## 12. 反夸大边界与可复现清单

**绝不声称**（沿用研发方案 §2.2 并扩展）：① 跨模态注意力 ≠ 微观电声耦合矩阵元 $\alpha^2F(\omega)$（R1 的能量-原子注意力亦然，只是“谱贡献归因”，论文必须写“phenomenological attribution”）；② Slack $\kappa_L$ / 费米斜率 Seebeck 趋势 ≠ $zT$ 精确预测（仍是漏斗初筛描述符，DFT/BoltzTraP2 复核不变）；③ 流精修采样多样性 ≠ 热力学系综（是模型认知不确定性，不是物理温度系综）；④ 逆向筛选 ≠ 发现可合成材料（是“DFT 复核候选队列”，合成验证才算发现）；⑤ 超分谱的高频细节 ≠ DFT 精度（必须报 HR 网格上的 WD/KL 且承认插值目标上限）。

**可复现清单**（投稿前逐项打勾）：□ M1–M5 同协议重跑（balanced_score + phys_score 双选优，`config.yaml` 冻结哈希）；□ 预训练去重脚本 + 泄漏报告；□ 三随机种子 mean±std（至少 M6–M8）；□ Blind/Oracle/HR 三轨指标 + 失败率 + gap F1 + 峰位 MAE + $\Theta_D$/$C_v$/$\kappa_L$-Spearman 全口径；□ profiler 实测 §9 表格（替换“估计”字样）；□ `dosdata/` 预测谱 + conformal 带公开；□ 与姊妹 EFM 路线的分工声明写入附录。

---

## 13. 参考文献

**神经算子与连续表征**

- Lu et al., DeepONet, Nat. Mach. Intell. 2019/2021（分支-主干算子定理）。
- Li et al., Fourier Neural Operator, ICLR 2021（谱卷积，分辨率不变性）。
- Sojitra–Dhingra–San, FEDONet: Fourier-Embedded DeepONet, JCP 2026 / arXiv 2509.12344（随机傅里叶 trunk + 谱预条件 NTK 分析；KS 76%→21%）。
- Cao et al., Laplace Neural Operator, Nat. Mach. Intell. 2024；LFNO, NeurIPS ML4PS 2025（瞬态/稳态分解思想来源）。
- IS-FNO, arXiv 2512.19439（近可逆 lifting/projection 稳定 rollout，投影精修的思想来源）。

**DOS 预测直接对标**

- Chen et al. (Kong), Mat2Spec, Nat. Commun. 2022（谱预测开山；SumNorm-KL 最优带隙分类；WD 被尖峰拖垮的关键教训）。
- Noh–Lee et al., DOSTransformer, NeurIPS 2023（能量条件逐点预测 + 晶系 prompt；能量嵌入在所有基线一致增益、OOD 增益更大）。
- Chen et al., E3NN phonon DOS, 2021（phDOS 基准与数据划分沿用）。
- Fung et al., Physically Informed eDOS, Chem. Mater. 2022（物理信息 eDOS）。
- Wu et al., Graph Transformer + 价电子特征 PDOS, J. Phys. Chem. A 2025（原子特征扩充的直接证据：s/p/d/f 价电子计数）。
- Mazitov–Ceriotti et al., PET-MAD-DOS, Digital Discovery 2026（通用 DOS + LLPR 系综 UQ；bespoke 仅 2×；跨分辨率 4806 点范式）。
- DOSSIER, arXiv 2608.24513（化学语言模型 + UMIP 教师蒸馏 1k 下 -11%；Mat2Spec R2 0.53–0.58 锚；NiPt₃ top-2% 筛选范式）。

**流匹配与生成式材料设计**

- Lipman et al., Flow Matching / Riemannian FM, 2022–2024（CFM 无仿真目标）。
- Miller et al., FlowMM, arXiv 2406.04713（50 步 ≈ DiffCSP 1000 步；黎曼流匹配处理周期边界）。
- Zeni et al., MatterGen, Nature 2025（S.U.N. 翻倍；adapter 条件微调范式）。
- Sriram et al., FlowLLM, arXiv 2410.23405（LLM 好先验 + RFM 精修：稳定 +300%；弛豫步 192→38——R3 先验选择的直接依据）。
- CrystalFlow, Nat. Commun. 2025（~10× 于 DiffCSP；少步 ODE 充分性证据）；DMFlow, arXiv 2602.04734（无序统一表示 + 球面重参Simplex 约束技巧）。

**几何预训练与基础模型**

- Shen et al., DPF, AAAI 2025（38 万无标注；掩码原子类型最稳；小数据 bulk/shear 增益最大）。
- CDSSL, arXiv 2408.17255（加噪边重构；49 配 37 胜）。
- Liao et al., DeNS, arXiv 2403.09549（非平衡去噪 + 力编码；OC20/OC22 SOTA——未弛豫问题的对口解）。
- DSSL, arXiv 2401.05223（掩码 + 对比 + VE/AS 物理引导；mask⊕对比互补，单对比不行）。
- Sola et al., UNATE, arXiv 2605.25866（嵌入转移 +2.3%，权重直迁 -106%——§5.4 的直接依据）。
- MoleVers, arXiv 2411.03537（两阶段预训练：掩码+极端去噪 → DFT/LLM 辅助性质精修）。
- Batatia et al., MACE, NeurIPS 2022；Yang et al., MatterSim, arXiv 2405.04967；Liao et al., EquiformerV2, ICLR 2024（教师库 + 对称性 scaling 证据）；Scaling Laws for Neural Material Models, arXiv 2509.21811（EquiformerV2 > 无约束 transformer 的 scaling 结论）；MOFSimBench, npj Comput. Mater. 2025（20 个 uMLIP 横评；e3nn/PyTorch 依赖地狱警示——蒸馏离线化的理由）。

**效率架构**

- Gu–Dao, Mamba, arXiv 2312.00752；Mamba-2 (SDD), ICML 2024；Mamba-3, arXiv 2603.15569（复数状态 + MIMO 算术强度——解码诊断思想来源；本报告不换主干，只取诊断与处方）。
- Dao, FlashAttention-2, 2024（SDPA 融合依据）。
- DeepSeek-V2/V3, Mixtral（稀疏 MoE 缩放定律——R4-P1 adapter 的轻量来源；重 MoE 见姊妹报告）。

---

## 附录 A：最小代码改动草图

> 草图非最终代码，M6 首周按此实现 + profiler 校准 §9。

**A1. 能量 trunk（新增 `model/trunk.py`，~60 行）。**

```python
class FourierTrunk(nn.Module):  # 双实例：eDOS(σ=8) / phDOS(σ=2)
    def __init__(self, m=128, d=512, r=256, sigma=8.0):
        super().__init__()
        self.register_buffer("B", torch.randn(m, 1) * sigma)  # 不训练（FEDONet 协议），LR 备注见 §8.2
        self.net = nn.Sequential(nn.Linear(2*m+1, d), nn.GELU(), nn.Linear(d, r))
        self.skip = nn.Linear(1, r)
    def forward(self, E):  # E: [B, N, 1] 归一化到[-1,1]
        proj = 2*math.pi*E @ self.B.T            # [B,N,m]
        g = torch.cat([torch.sin(proj), torch.cos(proj), E], -1)
        return self.net(g) + self.skip(E)        # [B,N,r]
```

**A2. Decoder 输入替换（`transformer.py:156-186` 改 ~15 行）。**

```python
# 旧：edos_query = self.edos_query_embed.unsqueeze(0).repeat(B,1,1)
E_e = norm_energy_grid(128, kind="edos").to(device)          # [128,1] 常量
E_p = norm_energy_grid(64,  kind="phdos").to(device)
q_e, q_p = self.trunk_e(E_e).expand(B,-1,-1), self.trunk_p(E_p).expand(B,-1,-1)
hs_edos, _ = self.edos_decoder(tgt=q_e, memory=memory, ..., query_pos=q_e)
# head：逐点 Branch(x)·Trunk(E) + MLP 残差（替代 heads.py 整谱卷积）
shape_e = pointwise_head(hs_edos, branch_e(h_crys), q_e)     # [B,128]
```

**A3. P0-1 双算修复（`transformer.py:319-349` 删 ~8 行）。**

```python
# 删：attn_output, attn_weights = self.self_attn(q,k,v,...)  # 结果被丢弃
# 留：QKV 投影复用 self.self_attn.in_proj_weight/in_proj_bias 切分，
#     base_scores + rp_scores → 一次 softmax → 一次 bmm(v)
```

**A4. 一维 $W_1$（`model.py` 加 ~6 行）。**

```python
def w1_loss(p, q):  # p,q: [B,N] SumNorm 分布
    return torch.mean(torch.abs(torch.cumsum(p,-1) - torch.cumsum(q,-1)), dim=-1).mean()
```

**A5. 3N 单纯形投影（`thermo_props.py` 或 `heads.py` 加 ~10 行，Duchi et al.）。**

```python
def simplex_project_3N(y, N_atom, dw=20.0):  # y: [B,64] 非负
    target = 3.0*N_atom/dw
    return project_onto_simplex(y, target)   # 排序+阈值，O(n log n)
```

---

## 附录 B：下游敏感性加权函数的物理推导

**B1. $\kappa_L$ 对 $\Theta_D$ 是立方放大。** Slack 方程（`thermo_props.py:106-125`，$A=3.1\times10^{-6}$）：$\kappa_L \propto \Theta_D^3$ ⇒ $\delta\kappa_L/\kappa_L \approx 3\,\delta\Theta_D/\Theta_D$。$\Theta_D$ 又来自二阶矩 $\langle\nu^2\rangle$（第 92–104 行）：$\Theta_D \propto \sqrt{\langle\nu^2\rangle}$ ⇒ 高频 10% 误差 → $\Theta_D$ ~5% → $\kappa_L$ ~16%。**结论**：phDOS 高频光学支值得 $\nu^2$ 加权，哪怕 uniform $R^2$ 不动。

**B2. $C_v(300K)$ 核集中在低频。** $K(x)=(x/2)^2/\sinh^2(x/2)$，$x=1.4388\,\nu/T$：$T=300K$ 时 $x=1 \Leftrightarrow \nu\approx 208\text{cm}^{-1}$，$K$ 在 $\nu<400\text{cm}^{-1}$ 占主导权重。**结论**：声学支（0–200cm⁻¹，约前 10–15 bins）值得独立加权 + 峰位精度要求最高——这正是当前 20cm⁻¹ 粗网格最亏的地方，也是超分（附录 A 连续采样）物理回报最大的地方。

**B3. eDOS 费米窗主导输运趋势。** Mott 公式 $S \propto -\partial\ln\sigma(E)/\partial E|_{E_F}$，$\sigma(E) \propto g(E)$（常数弛豫时间近似）⇒ 费米面 ±1.5eV（约 bins 53–75）+ 大梯度处（van Hove 边）是 Seebeck/带隙描述符的信息集中区。**结论**：$w_e(E)$ 取费米高斯 + 梯度项（§4.4c），金属/绝缘体分类错误在此窗内代价最大——与条件带隙损失（`model.py:178-193`）同窗，形成训练-评估闭环。

---

*报告完。本报告与姊妹 EFM-MoE 报告共同构成 uniARPAT-v3/v4 双轨纲领；建议优先立项 M6（R1+R4-P0）：改动最小、速度反增、证据最硬，一周内可出第一张新 Fig。*
