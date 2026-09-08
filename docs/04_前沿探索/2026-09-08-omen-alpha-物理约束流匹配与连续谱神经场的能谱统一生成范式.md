# uniARPAT 前沿探索报告：物理约束流匹配（Physics-Constrained Flow Matching）与连续谱神经场（Spectral Neural Field）驱动的能谱统一生成范式

- **报告日期**：2026 年 9 月 8 日
- **生成模型**：Omen Alpha（opencode-go/omen-alpha）
- **报告主题**：将 2024–2026 年生成式流匹配、物理硬约束推理、隐式神经表征与多模态预训练四条前沿主线系统性迁移至 uniARPAT，提出第三代能谱建模范式 **uniARPAT-SpecFlow**
- **报告性质**：深度调研 + 完整技术方案（架构拓扑 / 数学机理 / 算力显存评估 / 失效风险分析 / 消融设计）
- **关联代码库**：`uniARPAT/model/transformer.py`、`model/heads.py`、`model/model.py`、`datasets/dataset.py`、`thermo_props.py`

---

## 目录

1. [研究动机：uniARPAT 现状代码级诊断与瓶颈定位](#1-研究动机uniarpat-现状代码级诊断与瓶颈定位)
2. [前沿文献系统检索与机制剖析（2024–2026）](#2-前沿文献系统检索与机制剖析20242026)
3. [主题选择论证：为什么是"物理约束流匹配 + 连续谱神经场"](#3-主题选择论证)
4. [总体架构设计：uniARPAT-SpecFlow](#4-总体架构设计uniarpat-specflow)
5. [核心数学机理推导](#5-核心数学机理推导)
6. [算力、显存与推理时延全量评估](#6-算力显存与推理时延全量评估)
7. [数据工程与预训练方案](#7-数据工程与预训练方案)
8. [消融实验设计（M6–M10）与评测协议升级](#8-消融实验设计m6m10与评测协议升级)
9. [潜在失效风险全景分析与缓解预案](#9-潜在失效风险全景分析与缓解预案)
10. [执行路线图与论文定位](#10-执行路线图与论文定位)
11. [参考文献](#11-参考文献)

---

## 1. 研究动机：uniARPAT 现状代码级诊断与瓶颈定位

### 1.1 现有架构的实际形态（基于代码精读）

通过对核心代码的逐行审读，uniARPAT 当前的完整计算图如下：

```
输入（未弛豫晶体：elements [B,L] + pos [B,L+2,3]）
   │
   ├─ tok_emb (nn.Embedding, 118→512) ⊕ num_emb (AtomFeatureEncoder) → fuse_proj
   │
   ├─ ARPAT 共享几何 Encoder ×6 层
   │    └─ 每层：标准 MHA + RPEncoding 相对位置打分注入（rp_scores 与 base_scores 相加后 softmax）
   │
   ├─ 解耦双 Decoder ×6 层（eDOS 128 queries / phDOS 64 queries，均为可学习参数化 query+tgt）
   │
   ├─ PostDecoderGatedCrossAttention（α_e=α_p=0 零初始化门控）
   │
   ├─ MultiScaleResidualHead (eDOS, 788,225 params) / DeepConv1dHead (phDOS, 787,969 params)
   │
   ├─ safe_shape_norm（ReLU 归一化 + Softplus 自愈退避 + phDOS 边界 bin 置零）
   │
   ├─ ScaleHead（global_masked_pool → 3层MLP → log_scales → exp）
   │
   └─ 复合物理损失（model.py:164-203）：
        L_shape (Pearson+MSE, 平坦谱退避)
      + 3.0·L_shape_p + 0.5·(L_scale Huber)
      + 0.2·L_gap（费米面条件带隙二次惩罚）
      + 0.1·L_sum（3N 声子态数守恒, 软惩罚）
```

### 1.2 三个尚未解决的结构性瓶颈（自评）

**瓶颈 A：确定性回归范式 vs. 未弛豫输入的固有病态多解性。**
当前前向是纯确定性映射 `structure → DOS`（`transformer.py:122-223`）。但任务输入是**未弛豫晶体结构**——同一个输入结构可能弛豫到多个局域极小（磁性序、多形体系、Jahn-Teller 畸变等），对应**多个合法但不同的 DOS**。MSE/Pearson 回归在这种一对多映射下必然学习到"条件均值谱"，表现为：峰位被抹平（smearing）、带隙边缘被模糊、尖锐 van Hove 奇异点被系统性低估。这直接解释了基线中 eDOS 中位 R²=0.521 但均值仅 0.374、失败率 14.66% 的重尾分布——失败样本正是多解性最强的体系。

**瓶颈 B：物理约束全部是"软惩罚"，无一处"by construction"。**
`loss_sum`（3N 守恒）权重仅 0.1，`loss_gap` 是条件二次惩罚，非负性靠 `p_n[p_n<0]=0` 在评测端裁剪（`model.py:288`）。这意味着模型完全可能输出违反守恒律的谱，然后靠评测端"物理化妆"通过指标。2026 年 ICML Oral 工作 DeepSciReasoner/DeepDOSReasoner（Cornell Gomes Lab）已经证明：**把非负性与总态数守恒做成构造性硬约束（CDF 守恒输运格式）**，可将 eDOS R² 从 0.645 提升至 0.691、phDOS R² 提升 15%，且仅需 25% 训练数据即可匹配最强基线。这是对我们当前软惩罚路线的直接超越威胁，也是必须正面回应的竞争压力。

**瓶颈 C：固定 128/64 离散 bin 表征与下游热力学积分的分辨率耦合。**
`thermo_props.py` 的 Cv/Sv/Fvib 全部在 64 个固定频率 bin（`linspace(-280, 980, 64)`，步长 20 cm⁻¹）上做离散 Riemann 求和。谱的连续物理本质（g(ω) 是频率的连续函数）被 bin 网格截断：Debye 温度、低温 Cv（T < 50K 时积分核 `x²/sinh²(x/2)` 只采样前几个 bin）对 bin 分辨率极度敏感。同时费米面附近态密度斜率 `|d ln g/dE|`（热电筛选描述符，Week-4 计划项）目前只能在离散 bin 上做数值差分，噪声极大。

### 1.3 机遇窗口判断

当前（2026 年 9 月）恰好出现三个时间窗口叠加：
1. **DeepSciReasoner 刚刚确立"硬约束谱推理"新范式**（ICML 2026 Oral），但它是**确定性**的——多解性问题、不确定性量化、逆设计能力全部空白；
2. **流匹配（Flow Matching）在材料领域全面爆发**（CrystalFlow / FlowMM / LiFlow / QHFlow / MatterGen），但**尚无任何工作把"测度空间守恒输运"与流匹配结合用于能谱生成**——这是一个明确的空白交叉点；
3. uniARPAT 已有的 **Shape-Scale 解耦、门控跨模态、解耦双解码器** 基础设施，恰好是承载上述两项技术的理想骨架。

---

## 2. 前沿文献系统检索与机制剖析（2024–2026）

### 2.1 主线一：流匹配 / 扩散生成范式在材料科学的渗透

| 工作 | 发表 | 核心机制 | 对 uniARPAT 的可迁移点 |
| :--- | :--- | :--- | :--- |
| **MatterGen** (Zeni et al.) | *Nature* 639, 624–632 (2025) | 等变 score 网络联合去噪原子类型/坐标/晶格；性质条件微调（adapter 注入性质 embedding） | 性质条件生成范式：我们的 eDOS/phDOS 本身就是最丰富的"性质条件" |
| **CrystalFlow** (Luo et al.) | *Nat. Commun.* 16, 9267 (2025) | 晶体流匹配生成，OT 直线路径，采样快于扩散 | 直线路径 FM 训练目标可直接移植到谱空间 |
| **FlowMM** (Miller et al.) | arXiv:2406.04713, ICLR 2025 | 黎曼流匹配处理环面（分数坐标）几何 | 流匹配可处理非欧空间——谱的单 simplex 同样是非欧约束空间 |
| **LiFlow** (Fu et al.) | *npj Mach. Intell.* / arXiv:2410.01464 (2025) | Propagator + Corrector 双流匹配子模型；Maxwell-Boltzmann 自适应先验 | **"物理先验作为流起点"**思想：物理先验分布替代高斯先验可大幅缩短输运路径 |
| **QHFlow** (Kim et al.) | arXiv:2505.18817 (2025) | 高阶等变流匹配生成 DFT 哈密顿量，替代确定性回归 | **直接先例**：把"谱类结构化物理量"从回归改为条件生成，性能与泛化双提升 |
| **CrysLLMGen** (Das et al.) | NeurIPS 2025, arXiv:2510.23040 | LLM 生成中间表示 + 等变扩散精修的混合框架 | 混合"确定性粗估 + 生成式精修"的两段式思路 |
| **条件流匹配逆设计** (Felder et al.) | arXiv:2609.00863 (2026) | CFM 用于 PDE 约束逆设计，OT 耦合降低路径曲率 | 以目标 DOS 为条件反向生成结构候选的逆设计通道 |
| **FlowLLM** (Sriram et al.) | arXiv:2410.23405 (2024) | 以 LLM 生成的粗糙结构作为流匹配的**基础分布**（base distribution），稳定性率提升 3 倍 | 关键启示：**数据相关的先验分布 ≫ 高斯先验**——我们将用"物理解析先验谱"做流起点 |

**机制本质提炼**：流匹配在 ODE 意义下回归边际速度场
$$\mathcal{L}_{FM} = \mathbb{E}_{t, x_1 \sim p_{data}} \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|^2, \quad x_t = (1-t)x_0 + t x_1$$
其对**一对多条件分布** $p(x|c)$ 学习的是完整分布而非条件均值，天然解决瓶颈 A 的多解性；同时 rectified flow / 一致性蒸馏支持 1–4 步采样，推理成本可控。

### 2.2 主线二：物理硬约束"by construction"推理（直接竞品解剖）

**DeepSciReasoner / DeepDOSReasoner**（Wang et al., Cornell, ICML 2026 Oral；demo: gomes-lab.github.io/deepdosreasoner-demo）两阶段：

- **Stage I 结构感知谱解码**：edge-aware 晶体图 Transformer + **atom-to-energy source attention**——每个能量 bin 作为 query 对原子级 embedding 做选择性聚合，使每个谱坐标物理性地"扎根"于在该能量贡献态的原子（无监督即可恢复元素分辨谱）；
- **Stage II 约束保持物理推理**：把归一化 DOS 视为概率密度，学习一个**连续性方程输运**（no-flux 边界），用 **CDF 上风格式**沿能量轴搬运谱质量——每一步输出都**构造性满足**非负 + 总态数守恒（质量只能重新分布，不能凭空产生或消灭）；
- **实测量化**：eDOS R² 0.645 → 0.691；eDOS Wasserstein 距离 ↓14.3%；phDOS R² ↑15%；VB-gap 识别 MCC ↑18.6%；~10.5 ms/晶体；25% 数据即匹配最强基线。

**对其的关键超越点分析（本研究方案的立足点）**：
1. 其 Stage II 是**确定性事后精修**（deterministic post-hoc refinement）——输运的起点仍是单一回归输出，多解性问题原封不动；
2. 其输运在**离散能量网格**上用 upwind 数值格式实现，格式误差在尖锐峰处引入数值扩散（numerical diffusion），恰恰伤害最需要保真的峰锐度；
3. 其 e/ph 两谱推理是**独立**的，没有跨模态耦合（uniARPAT 的核心差异化资产）。

### 2.3 主线三：隐式神经表征（INR）/ 连续场表征用于物理谱

| 工作 | 核心机制 | 可迁移点 |
| :--- | :--- | :--- |
| **Phy-CoSF** (arXiv:2605.13583, 2026) | 连续光谱场重建：波长坐标 Fourier 高频编码 + 强度合成头，光谱超分辨 | 谱作为连续函数 $g(E)$ 而非固定 bin 向量的表征范式 |
| **SSIF** (ICLR 2025 submission) | 空间-谱联合隐式函数：单模型覆盖任意分辨率，物理响应函数易于嵌入 | 任意分辨率谱解码 + 物理算子（积分/微分）解析可得 |
| **PEINR** (ICML 2025) | 物理增强 INR 高保真流场重建 | 物理增强项与 INR 的组合训练策略 |
| **AVC / Vekua Cascade** (arXiv:2512.11776) | 解析基函数 + 可微线性求解器对抗 spectral bias | 谱高频（尖锐峰）拟合的谱偏差（spectral bias）问题及其解析基缓解方案 |
| **Fourier Features / SIREN**（经典基础） | 坐标高频编码 | 能量坐标编码 $\gamma(E) = [\sin(2\pi \omega_k E), \cos(2\pi \omega_k E)]_k$ 的带宽设计应与物理展宽核匹配 |

**机制本质提炼**：把 DOS 从"K 维向量"升级为"坐标条件函数" $g_\theta(E; c)$，则**任意分辨率求值、解析积分求热力学量、解析微分求费米面斜率、跨分辨率预训练**全部在一次框架下解锁。

### 2.4 主线四：多模态对齐预训练与等变基础模型

- **MultiMat**（Newton/Cell Press 2025, arXiv:2312.00111）：把晶体结构、DOS、电荷密度、文本四模态对齐到共享潜空间做对比预训练；DOS 模态本身就是预训练模态之一，下游检索/性质预测 SOTA。→ 证明"eDOS/phDOS 潜空间对齐"预训练具有可迁移价值。
- **EPT**（*Nat. Commun.* 2026）/ **EquiformerV2/V3**：等变 Transformer 在 OMat24（1.1 亿非平衡结构）上做去噪预训练，**对非平衡/未弛豫结构的显式覆盖**正是我们输入分布的核心特征。→ 数据工程方向。
- **MACE-MP / UMA**（NeurIPS 2025）：通用原子基础模型家族。→ 可作为冻结几何先验的蒸馏教师（可选增强）。
- **Mat2Spec**（*Nat. Commun.* 13, 949, 2022）：概率嵌入 + 监督对比学习的谱预测——早期的"谱分布化"尝试，但其分布仅是嵌入层面的高斯，未触及输出空间。

### 2.5 检索结论：空白交叉点定位

```
                确定性回归          硬约束推理(DeepSciReasoner)     生成式(FM/Diffusion)
物理谱预测        ✗ 已被超越              ✓ 新SOTA(但确定性)           ✗ 完全空白 ←【本研究落点】
多解性/不确定性     ✗ 无法处理             ✗ 未处理                     ✓ FM天然解决
守恒律 by construction ✗ 软惩罚            ✓ 有                        ✗ 无 ←【我们的融合创新】
跨模态 e-ph 耦合   ✓ uniARPAT独有          ✗ 无                        ✗ 无
连续谱表征/任意分辨率 ✗ 固定bin            ✗ 固定bin                    ✗ 无
```

**选定主题**：**"测度空间守恒流匹配 + 连续谱神经场"**——把守恒律从"损失函数里的软惩罚"升级为"生成流的几何约束"，把谱从"离散 bin 向量"升级为"连续能量坐标场"，把模型从"确定性回归器"升级为"条件生成分布"，三者在一个统一架构中闭环。

---

## 3. 主题选择论证

### 3.1 四个候选主题的对比淘汰

| 候选主题 | 算法创新深度 | 物理增益 | 工程可行性 | 竞争风险 | 判定 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| A. 纯等变升级（E3NN/Equiformer 骨干替换） | 中 | 中 | 高 | EPT/EquiformerV3 已做完，增量小 | ❌ 淘汰：撞车且创新薄 |
| B. 纯 LLM 机制移植（MoE/长上下文/RoPE 到晶体） | 中 | 低-中 | 中 | CrysLLMGen 等已占位；谱任务无序列语义 | ❌ 淘汰：物理增益弱 |
| C. 硬约束谱推理复刻 + 改进 | 中 | 高 | 高 | 与 DeepSciReasoner 正面同质 | ❌ 单独不可发：novelty 不足 |
| **D. 物理约束流匹配 + 连续谱神经场** | **高（生成流 × 测度几何 × 神经场三重交叉）** | **高（多解性/守恒/分辨率/不确定性四增益）** | 中（单卡 V100 可承载，§6 论证） | **空白交叉点，且与 C 形成代差** | ✅ **选定** |

### 3.2 主题 D 的四重物理增益与三重算法增益闭环

**物理增益**：
1. **多解性建模**：输出 $p(\text{DOS}|\text{未弛豫结构})$ 全分布，条件均值只是分布的一阶矩——可按需取中位数/众数/分位数，评测与筛选策略解耦；
2. **守恒律构造性成立**：每个采样谱都严格非负且总态数守恒，评测端 `p_n[p_n<0]=0` 这类"物理化妆"彻底退役；
3. **任意分辨率与解析算子**：连续场表征使 Cv(T)、Θ_D、费米面斜率变为解析可微算子，低温热力学精度摆脱 bin 网格截断；
4. **不确定性量化**：采样 N 条 ODE 轨迹 → 逐 bin 置信区间 → 高通量筛选自动获得"可信度"维度，失败率（R²<0 样本）可被前置预警。

**算法增益**：
1. **测度空间流匹配**：在密度单纯形 / 分位数函数空间上定义流，FM 理论与最优输运（Brenier 1D 单调重排）结合，属全新问题设定；
2. **守恒生成流**：把 DeepSciReasoner 的"确定性 CDF 输运"推广为"随机性守恒流"——约束在流的每一步、每一个中间样本上成立，而非仅终点；
3. **跨模态联合流**：e-ph 两个谱空间上的**耦合速度场**，门控交叉注意力在流中间态上生效——把 uniARPAT 既有的零初始化门控机制从"特征调制"升级为"分布层面的电声耦合唯象表征"（严格保持 Anti-Overclaiming 边界：不声称微观 $\alpha^2F(\omega)$ 矩阵元）。

---

## 4. 总体架构设计：uniARPAT-SpecFlow

### 4.1 架构总拓扑图

```
输入 (elements, pos) —— 未弛豫晶体
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  第一级：ARPAT 共享几何 Encoder（保持不变, 6层, ~50M）          │
│  输出 memory [B, L, 512] + h_crystal [B, 512]              │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  第二级：连续谱神经场查询构造（Spectral Field Queries, 新增）   │
│                                                           │
│  eDOS: K_e=128 个"能量坐标 token"                          │
│    q_i = W_q·γ(E_i) + W_c·h_crystal + PE_type(i)           │
│    γ(E) = [sin(2πω_k E), cos(2πω_k E)]_{k=1..32}           │
│    ω_k ~ 与物理展宽核带宽匹配的可学习频率组                    │
│                                                           │
│  phDOS: K_p=64 个"频率坐标 token"（同构）                    │
│  ※ 关键升级：query 不再是自由参数，而是能量坐标的函数           │
│    → 天然支持推理期任意加密网格（128→1024 bins 超分辨）         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  第三级：守恒流匹配速度场网络（Conservative FM Velocity）      │
│                                                           │
│  状态: 谱潜 token 序列 z_t = concat(z_e,t, z_p,t)           │
│        z_e,t ∈ R^{K_e×d}, z_p,t ∈ R^{K_p×d}                │
│        （z 携带当前流时间 t 的谱估计的量化函数表征，§5.2）       │
│                                                           │
│  v_θ(z_t, t, c) =                                          │
│    t 嵌入: AdaLN-Zero 调制（FiLM: γ(t), β(t)）              │
│    自注意力: z_e,t 内部 + z_p,t 内部（复用解耦双解码器）        │
│    跨模态: 零初始化门控交叉注意力（α_e, α_p 保留!）            │
│    交叉编码: 对 memory 的 cross-attention（能量token读原子）   │
│    ↓                                                       │
│  输出头: 分位数函数增量场 ΔQ_e, ΔQ_p（§5.2, 单调性硬约束）     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  第四级：物理后端（可微热力学 + 不确定性）                      │
│  · 分位数→密度可微直方图: g(E_i), g(ω_j)                     │
│  · 解析热力学: Cv(T), Sv(T), Fvib(T), Θ_D（autograd）       │
│  · N 次采样集成: 逐bin μ±σ, Wasserstein 分位数带              │
│  · 筛选描述符: |d lng/dE| (解析), κ_L (Julian-Slack 保留)    │
└─────────────────────────────────────────────────────────┘
```

### 4.2 模块详细设计

#### 4.2.1 连续谱神经场查询（Spectral Field Queries）

替换 `transformer.py:84-89` 的自由参数 `edos_query_embed` / `phdos_query_embed`：

```python
class SpectralFieldQuery(nn.Module):
    def __init__(self, d_model=512, num_freqs=32, coord_range=(-10.0, 10.0)):
        super().__init__()
        self.omega = nn.Parameter(torch.linspace(1.0, 32.0, num_freqs).unsqueeze(0))  # 可学习频率
        self.coord_scale = nn.Parameter(torch.tensor(coord_range))
        self.proj = nn.Linear(2 * num_freqs + d_model, d_model)

    def forward(self, E, h_crystal):          # E: [B, K] 能量/频率坐标; h_crystal: [B, d]
        E_n = (E - self.coord_scale[0]) / (self.coord_scale[1] - self.coord_scale[0])
        arg = 2 * math.pi * self.omega * E_n.unsqueeze(-1)     # [B, K, F]
        gamma = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # [B, K, 2F]
        hc = h_crystal.unsqueeze(1).expand(-1, E.shape[1], -1)       # 全局晶体条件广播
        return self.proj(torch.cat([gamma, hc], dim=-1))             # [B, K, d]
```

设计要点：
- **坐标条件化**保证"能量相近的 bin 特征相似"，归纳偏置与谱的连续性物理一致，替代 CNN 头隐式学到的局部性；
- **全局晶体条件**注入使每个 bin 知道"我该在哪个化学环境下取值"——对标 DeepDOSReasoner 的 atom-to-energy source attention，但通过 cross-attention 到 memory 达成同等信息通路；
- 可学习频率 $\omega_k$ 采用对数均匀初始化（1–32 周期），上限与 DFT 谱的 Gaussian 展宽核（通常 σ ≈ 0.05–0.1 eV）匹配，规避 spectral bias（AVC 分析的教训：无界高频会引入对初始化的病态敏感）。

#### 4.2.2 守恒参数化输出层（State-Conservation Head）

**核心数学事实**：phDOS 的总态数由晶格动力学**严格确定**为 $3N$；eDOS 在给定能量窗内的总态数由价电子数确定。这意味着 uniARPAT 现有 ScaleHead 回归的自由标度，**在物理上根本不是自由量**。当前代码用 0.1 权重的软损失 `loss_sum`（`model.py:199`）去近似一个**已知的解析约束**——这是范式级浪费。

升级为构造性参数化：设网络输出 logits $z \in \mathbb{R}^K$，

$$g(E_i) = N_{\text{states}} \cdot \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau) \cdot \Delta E}, \qquad \sum_i g(E_i)\Delta E = N_{\text{states}} \; \text{(精确成立)}$$

其中 $N_{\text{states}}^{\text{ph}} = 3N_{\text{atom}}$（由输入 mask 解析可得，代码已有 `valid_atoms`），$N_{\text{states}}^{e}$ 由化合价电子数表（`utils/periodic_table_v2.csv` 已具备元素数据）解析计算。

**与现有 `safe_shape_norm` 的关系（关键工程决策）**：
- softmax 参数化严格非负 + 严格守恒，但**无法表达精确零**（带隙处有 $e^{z/\tau} > 0$ 的指数小残尾）；
- `safe_shape_norm` 可表达精确零但不守恒。
- **融合方案 "Gated Softmax"**：引入逐 bin 门控 $m_i \in [0,1] = \sigma(\lambda \cdot \text{net}_i)$（λ=10 初始化陡峭化），输出 $g_i = N_{\text{states}} \cdot m_i \cdot \text{softmax}_i(z)$。带隙时门控饱和到 0 → 精确零；非隙时 m≈1 → 守恒误差 $\le (1-\bar{m})N_{\text{states}}$，可加一项 $\mu \sum_i (1-m_i)$ 的门控稀疏正则驱动"除带隙外全开"。此设计**同时继承** P0-C 修复的带隙精确零语义与新增的守恒语义，且对现有 `safe_shape_norm` 完全向后兼容（消融 M6b 单独验证）。

#### 4.2.3 测度空间守恒流匹配（Quantile Flow）——本方案最核心创新

**问题**：直接在密度值空间 $\{g_i\}$ 上做流匹配，ODE 积分的中间态会越出单纯形（出现负密度/质量漂移），FM 的"分布保真"与"物理约束"冲突。

**解法：把流定义在 1D 最优输运（单调重排）的自然参数化——分位数函数空间上。**

设谱形状为归一化密度 $s(E) = g(E)/N_{\text{states}}$，$\int s\,dE = 1$。其分位数函数：
$$Q(u) = \inf\{E : \text{CDF}(E) \ge u\}, \quad u \in [0, 1]$$

**Brenier 定理的一维形式**：两个等质量密度之间的最优输运映射是单调重排 $T = Q_1 \circ Q_0^{-1}$。**McCann 插值**定义测度空间上的测地线：
$$Q_t(u) = (1-t)\, Q_0(u) + t\, Q_1(u)$$

三个关键性质（逐条推导见 §5.2）：
1. **每个 $t$ 的 $Q_t$ 仍是合法分位数函数**（单调递增）⇒ 对应密度严格非负、质量恒为 1；
2. **带隙 = 平台段**：绝缘体的零密度区间在 $Q$ 中表现为能量平台（$Q$ 在对应 $u$ 区间上导数为零的补集跳变），即**带隙是分位数表示的原生特征**，而非像密度表示那样需要"精确零"这种脆弱结构；
3. **尖锐峰 = 陡峭段**：van Hove 奇异点对应 $Q$ 的高斜率区，线性插值不会像密度空间 MSE 那样把峰"平均掉"。

**网络侧实现**：谱状态 $z_t$ 的每个 token 携带分位数节点对 $(Q_t(u_k), u_k)$，$u_k = (k+0.5)/K$。速度场网络输出**单调增量**：
$$Q_{t+\Delta t}(u_k) = Q_t(u_k) + \Delta t \cdot \Delta Q_\theta(\cdot), \quad \Delta Q \text{ 通过 } \text{cumsum}(\text{softplus}(\delta)) \text{ 强制增量单调性}$$

流匹配目标（分位数空间直线路径）：
$$\mathcal{L}_{QFM} = \mathbb{E}_{t \sim U[0,1],\, u} \left\| v_\theta\big(Q_t(u), t, c\big) - \big(Q_1(u) - Q_0(u)\big) \right\|^2$$

**物理先验基础分布（对标 FlowLLM 思想）**：$Q_0$ 不取均匀分位数（对应均匀密度），而取**化学解析先验谱**的分位数：
- phDOS：Debye 模型 $\propto \omega^2$（截止频率由 $\bar{M}, V_{\text{atom}}$ 估计——`thermo_props.py` 已有全部输入量）与原子对分布频率尺度的混合；
- eDOS：按组成元素各自数据库平均 eDOS（MP 元素级统计，离线预计算）加权和的高斯展宽谱。
数据相关先验使输运路径大幅缩短（LiFlow/FlowLLM 均实测 2–3 倍收敛加速），并把模型容量集中用于学习"先验之外的结构特异性修正"。

**跨模态联合流（uniARPAT 差异化核心）**：eDOS 与 phDOS 的分位数 token 拼接后进同一速度场网络，中间经零初始化门控交叉注意力。物理含义：在分布层面建模电声关联（如重元素软模 ↔ 窄带隙倾向），$\alpha_e, \alpha_p$ 的最终收敛值本身成为可解释的电声耦合唯象强度指标（延续 M5 Pilot 中 $\alpha \approx -0.0015$ 的观测并赋予新意义）。

#### 4.2.4 可微热力学后端

`thermo_props.py` 的循环实现改为全向量化 + autograd：

$$C_v(T) = 3R \int \left(\frac{x}{2}\right)^2 \mathrm{csch}^2\!\left(\frac{x}{2}\right) g(\omega)\, d\omega, \quad x = \frac{1.4388\,\omega}{T}$$

由于 $g(\omega) = N \cdot m(\omega) \cdot \text{softmax}(z(\omega))$ 是网络输出的解析函数，$\partial C_v / \partial \theta$ 可精确反传——使得 **3N 守恒、Dulong-Petit 高温极限（$C_v \to 3R$, per-atom）、低温 Debye $T^3$ 律**全部可作为**解析物理正则项**参与训练（而当前仅靠间接的谱形损失隐式逼近）。费米面斜率 $\partial_E \ln g |_{E_F}$ 同理由 autograd 解析获得，直接服务 Week-4 热电粗筛（对比现方案 bin 域数值差分）。

#### 4.2.5 推理与不确定性量化

```
盲测推理（每材料）:
  Q_0 ← 物理解析先验分位数
  for s in 1..S:                       # S = 8 个独立先验扰动种子
      for step in ODE (RK45, 10 步):   # rectified 路径 + 一致性蒸馏后可减至 1–4 步
          v = v_θ(Q_t, t, c)
      谱_s = 分位数→密度
  μ, σ = mean, std over S 条谱          # 逐 bin 不确定性
  筛选置信度 = 1 - σ/μ 的聚合统计        # 高通量漏斗第 1 级新增"可信度"闸门
```

---

## 5. 核心数学机理推导

### 5.1 条件流匹配目标的无偏性

对任意条件概率路径 $p_t(x|c)$ 与生成向量场 $v_\theta$，FM 损失
$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t, x \sim p_t(\cdot|c)} \left\| v_\theta(x, t, c) - u_t(x|c) \right\|^2$$
与其条件版本 $\mathcal{L}_{CFM} = \mathbb{E}_{t, x_1 \sim p_{data}, x_0 \sim p_0} \| v_\theta(x_t, t, c) - (x_1 - x_0)\|^2$（线性插值 $x_t = (1-t)x_0 + t x_1$）具有**相同的梯度期望**：
$$\nabla_\theta \mathcal{L}_{FM} = \nabla_\theta \mathcal{L}_{CFM}$$
（证明即 Lipman et al. 2022 引理 1 的逐点加权版本，条件结构 $c$ 不进入插值定义因此不影响该结论。）这保证我们可以在不构造边际速度场的情况下直接回归条件速度——训练只需"采样一对 (先验谱， 真值谱)"。

### 5.2 分位数线性插值的守恒性证明

**命题**：设 $s_0, s_1$ 为 $[E_{\min}, E_{\max}]$ 上积分为 1 的非负密度，$Q_0, Q_1$ 为其分位数函数。对任意 $t \in [0,1]$，$Q_t(u) = (1-t)Q_0(u) + t Q_1(u)$ 是某个积分为 1 的非负密度的分位数函数。

**证明**：
(1) *单调性*：$Q_0, Q_1$ 均单调不减，非负系数凸组合保持单调不减，故 $Q_t$ 单调不减；
(2) *边界*：$Q_0(0) = Q_1(0) = E_{\min}$（支撑下界），同理上界，故 $Q_t$ 边界保持；
(3) *存在对应密度*：单调右连续函数 $Q_t$ 定义一个概率测度 $\mu_t$ 为 Lebesgue 测度在 $Q_t$ 下的推前（pushforward）：$\mu_t(A) = \text{Leb}\{u : Q_t(u) \in A\}$。该测度的密度 $s_t = d\mu_t/dE$ 处处非负且 $\int s_t dE = \int du = 1$。∎

**推论 1（非负性硬约束）**：$s_t \geq 0$ 对所有 $t$ 严格成立——守恒不需要惩罚项，它是插值几何的内禀性质。
**推论 2（质量守恒硬约束）**：$\int s_t\, dE = 1$ 对所有 $t$ 成立——推前定义直接给出，无需任何数值格式。对比 DeepSciReasoner 的 CDF-upwind 离散格式：其守恒成立到格点精度且有数值扩散；分位数插值的守恒是**连续精确**的。
**推论 3（带隙的原生性）**：若 $s_1$ 在 $(a, b)$ 严格为零而 $s_0$ 在该区间有质量，则 $Q_1$ 在 $[\text{CDF}_1(a), \text{CDF}_1(b)]$ 上取常数 $b$（平台），$Q_t$ 随 $t$ 增长逐步把该平台"压平"——即输运把质量推出带隙区间。**反方向同样成立**：金属化过程（带隙消失）对应 $Q$ 平台的抬升。带隙开合在分位数空间是**平滑的一维几何形变**，而在密度空间是"零与非零的尖锐切换"——后者的梯度信号（训练 `loss_gap` 时）稀疏且不稳定，前者的梯度稠密平滑。这从数学上解释了为何带隙预测（eDOS 失败率 14.66% 的主因之一）在当前范式下属硬骨头。

### 5.3 分位数→密度可微直方图（离散化桥）

数值实现需把连续 $Q$ 映射回固定能量网格 $\{E_i\}_{i=1}^K$。采用核软分箱：
$$s(E_i) = \sum_{k=1}^{K} w_k \, \mathcal{K}_h\!\big(E_i - Q(u_k)\big) \cdot \Delta u, \quad \mathcal{K}_h = \mathcal{N}(0, h^2), \quad \sum_k w_k \Delta u = 1$$
权重 $w_k$ 可学习（初始化均匀 $w_k = 1$），带宽 $h$ 从展宽核尺度退火至 0.5 bin。该算子对 $Q$ 和 $w$ 均可微；总质量误差为 $O(h^2)$ 的核偏差，可在输出端做一次 $L_1$ 重归一（封闭形式，可微）将误差归零。

### 5.4 速度场网络的训练-推理一致性

推理 ODE：$dQ/dt = v_\theta(Q, t, c)$。由于 (i) $Q$ 的合法域（单调、边界固定）是凸集，(ii) $v_\theta$ 输出被 cumsum-softplus 约束为保序增量场，**欧拉/RK45 积分的每一个中间态都保持在合法域内**——这与 §4.2.3 的推论共同构成"**全流程构造性守恒**"：训练目标合法、速度场保序、积分路径合法、采样终点合法。任何一步都不依赖"事后投影"或"评测端裁剪"。

### 5.5 损失函数总纲（替换 `model.py:164-203`）

$$\mathcal{L} = \underbrace{\mathcal{L}_{QFM}^{e} + 3.0\,\mathcal{L}_{QFM}^{p}}_{\text{分位数流匹配}} + \underbrace{0.3\,\mathcal{L}_{\text{mom}}}_{\text{矩匹配辅助: 谱均值/方差/熵}} + \underbrace{0.2\,\mathcal{L}_{\text{edge}}}_{\text{带边加权(带隙边缘聚焦)}} + \underbrace{0.1\,(\mathcal{L}_{C_v}^{\text{DP}} + \mathcal{L}_{\text{integ}})}_{\text{解析物理正则}}$$

- $\mathcal{L}_{\text{mom}}$：对 $Q_t$ 的样本分位数与真值分位数做 L1 对齐（补充 FM 之外的模式覆盖信号，缓解少量样本下方差塌缩）；
- $\mathcal{L}_{\text{edge}}$：对真值谱梯度的强变化 bin（带边/van Hove）加权 2×，聚焦峰锐度；
- $\mathcal{L}_{C_v}^{\text{DP}} = (C_v(1000K) - 3R)^2 / (3R)^2$：Dulong-Petit 渐近正则（解析可得，无需真值 Cv 标签）；
- 原有 `loss_scale` 退役：标度由 $N_{\text{states}}$ 解析给出（§4.2.2），ScaleHead 仅保留为辅助头做交叉验证输出（消融 M6c 检验两种标度来源的一致性，不一致率本身成为有价值的诊断指标）。

---

## 6. 算力、显存与推理时延全量评估

### 6.1 参数量与显存核算（基准：V100-32GB，d_model=512）

| 组件 | 当前 M5 | SpecFlow 变化 | 说明 |
| :--- | ---: | ---: | :--- |
| 共享 Encoder（6层） | ~50M | 0（不变） | 复用 |
| 双 Decoder（6层×2） | ~26M | 0（**转为速度场骨干**） | 不新增，职能转换 |
| 门控交叉注意力 | ~2.1M | 0（不变，移入速度场网络内部） | 复用 |
| 谱神经场查询构造 | 0（自由参数 query ~0.1M） | **+0.35M** | `SpectralFieldQuery` ×2 |
| t 嵌入 + AdaLN-Zero | 0 | **+0.2M** | 时间调制（每层 γ/β 生成器） |
| 分位数头 + Gated Softmax 头 | ~0.79M×2（旧 Head） | **+0.5M** | 替换旧输出头 |
| ScaleHead（辅助） | 74k | +0（保留） | |
| **总计** | **75.93M** | **≈76.9M（+1.3%）** | 模型规模几乎不变 |

**训练显存**：Pilot 实测峰值 9,382 MB。新增开销三项：① t 采样与 AdaLN 调制（<50MB）；② 每样本 $S{=}1$ 的先验谱采样与插值缓存（$K_e{+}K_p{=}192$ tokens × 512 dim，<10MB）；③ 分位数 cumsum 反传图（<30MB）。**预估峰值 ≈ 9.8–10.5 GB（+8–12%）**，32GB 卡余量充足；RTX 4090/3090（24GB）亦可承载。

**训练耗时**：每步新增一次先验构造（numpy 级，<1ms）+ cumsum 算子（<2ms）。速度场网络即原双解码器前向，**计算图规模与 M5 等价**。预估 107.8 s/epoch → **约 115–125 s/epoch（+8–15%）**，百轮单变体 3.2–3.5 小时，M6–M10 五组消融 ≈ 17–18 小时（V100 单卡一夜 + 一个白天）。

**推理时延（对比表）**：

| 方案 | 每材料时延 | 出处/依据 |
| :--- | ---: | :--- |
| DFT（phDOS+eDOS） | ~10⁵–10⁶ s | 文献通用口径 |
| DeepDOSReasoner | 10.5 ms | 其 demo 页声明（~10⁷× 加速） |
| uniARPAT M5（确定性） | ~15 ms（估） | 单次前向 |
| **SpecFlow（确定性蒸馏，1 步）** | **~18 ms** | 一致性蒸馏后单前向 |
| **SpecFlow（8 采样 × 10 步 ODE）** | **~1.2 s** | 15 万级筛选场景用蒸馏版，精筛用完整版 |
| SpecFlow（15 万材料全漏斗） | 蒸馏版 ≈ 45 min | 完全在 Week-4 筛选预算内 |

### 6.2 训练稳定性预算

- AMP：沿用 `GradScaler(init_scale=1024)`；分位数 cumsum-softplus 对 fp16 数值友好（单调累积无减法灾难性消去）；Gated Softmax 中 λ=10 的 sigmoid 在 fp16 下饱和安全；
- 先验谱构造在 CPU 端 DataLoader worker 内完成（离线缓存训练集真值谱与先验谱的插值对只需存索引，无 IO 瓶颈）。

---

## 7. 数据工程与预训练方案

### 7.1 非平衡结构去噪预训练（对齐 EPT/OMat24 思想，轻量版）

uniARPAT 的输入分布是**未弛豫结构**，而训练标签 DOS 来自弛豫结构。构造自监督一致性任务：

1. 对训练集晶体施加 rattle 扰动（σ ∈ {0.01, 0.03, 0.05, 0.1} Å）+ 晶格畸变（±2%），生成非平衡视图；
2. **物理不变性假设**：小幅结构扰动下 DOS 仅平滑变化（Debye-Waller 型效应），用 mean-teacher 一致性损失 $\|g_\theta(\text{perturbed}) - \text{sg}[g_{\bar\theta}(\text{clean})]\|^2$ 蒸馏；
3. 该预训练**无需任何新标签**，直接对冲"未弛豫输入 → 弛豫标签"的域差，预期对 eDOS 失败率（14.66%）有显著削减——因为相当比例失败样本疑为高内应力结构。

### 7.2 e-ph 潜空间对比对齐（MultiMat 思想的谱版本）

在共享 encoder 的 $h_{\text{crystal}}$ 与双解码器池化谱嵌入 $h_e, h_p$ 之间加 InfoNCE：同一材料的 $(h_e, h_p)$ 互为正对，batch 内其他材料为负对。目的：把"电声在潜空间的可关联性"显式结构化，为跨模态联合流提供更好的初始几何（零初始化门控 $\alpha$ 仍保证训练起点严格解耦，符合 M4 的消融叙事）。

### 7.3 物理先验谱库离线构建

- phDOS 先验：Debye $g \propto \omega^2\Theta_D^{-3}$ + 权重与对分布谱的混合，参数由 $\bar{M}, V_{\text{atom}}, N$ 解析给出（thermo_props.py 现成输入）；
- eDOS 先验：元素级平均 eDOS（从 MP/训练集聚合）按组成加权 + 0.3 eV 高斯展宽；
- 一次性构建，全部样本预缓存为 128+64 维先验分位数向量。

---

## 8. 消融实验设计（M6–M10）与评测协议升级

### 8.1 消融矩阵（衔接 M1–M5 叙事线）

| 变体 | 谱表征 | 推理范式 | 守恒机制 | 先验分布 | 跨模态 | 核心验证假设 |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **M5**（已有） | 离散 bin + safe_shape_norm | 确定性回归 | 软惩罚 | — | 零初始化门控 | （基线） |
| **M6** | 连续谱场查询 + Gated Softmax | 确定性回归 | **构造性守恒** | — | 门控 | 谱场+守恒头的独立增益（对标 DeepSciReasoner 同类增益） |
| **M7** | M6 | **分位数流匹配** | 流内构造性 | 均匀 | 门控 | 生成式 vs 确定式的分布增益 |
| **M8** | M6 | 分位数流匹配 | 流内构造性 | **物理解析先验** | 门控 | 数据相关先验的路径缩短增益 |
| **M9** | M6 | 分位数流匹配（e-ph **联合**流） | 流内构造性 | 物理先验 | **流内门控** | 跨模态分布在流层面的耦合增益 |
| **M10** | M6 | M9 + 非平衡预训练 + e-ph 对齐 | 流内构造性 | 物理先验 | 流内门控 | 完整系统（论文主结果） |

### 8.2 评测协议升级（新增分布指标）

在现有逐样本 MAE/MSE/R²（中位/均值/失败率）全口径基础上新增：

1. **Wasserstein-1 距离**（逐样本，与 DeepSciReasoner 直接可比）：$W_1 = \int |F_{\hat{g}}(E) - F_g(E)|\,dE$——由分位数表示**闭式可得**（$W_1$ 在 1D 等于分位数函数差的 L1 范数），这是本表征的免费红利；
2. **带隙判别 MCC**（VB-gap precision/recall，对齐 DeepDOSReasoner 的 0.904 precision / MCC+18.6% 报告口径）；
3. **不确定性校准**：预测区间覆盖率（PICP@90）与区间宽度（MPIW）——要求实际覆盖率 ≈ 90%；
4. **热力学精度**：Cv(300K)、Θ_D、Sv(500K) MAE（现协议保留），新增**低温 Cv(T<50K)** 专项（检验连续表征对 bin 截断误差的消除）；
5. **峰锐度指标**：谱峰半高宽（FWHM）误差分布与逐峰位偏差——直接量化"条件均值抹平"问题的解决程度。

---

## 9. 潜在失效风险全景分析与缓解预案

| 风险编号 | 风险描述 | 机理 | 概率 | 缓解预案 |
| :---: | :--- | :--- | :---: | :--- |
| **R1** | FM 采样方差过大，单样本质量劣于 M5 均值 | 未弛豫输入的多解性是**真实**的，但真值只有弛豫后的单一标签：模型可能把"标签噪声"误学为"真实多模态" | 高 | ① $\mathcal{L}_{\text{mom}}$ 矩匹配锚定一阶矩；② 推理默认输出**分布中位数谱**（对多模态稳健）+ 保留确定性蒸馏头作为"保守通道"；③ 评测双轨制：单点指标用中位数谱，分布指标用全采样 |
| **R2** | 分位数表示在离散网格上的直方图重构噪声伤害峰锐度 | 核软分箱带宽 h 与峰宽竞争 | 中 | 带宽退火调度（2 bin → 0.5 bin）+ 可学习权重 $w_k$；FWHM 指标全程监控 |
| **R3** | Gated Softmax 门控塌缩（全 0 或全 1） | sigmoid 饱和区梯度消失 | 中 | λ 温水化（1→10 线性升温）+ 门控熵正则 + 保留 `safe_shape_norm` 作为结构等价回退分支（自愈思想复用，tests/test_scale_norm.py 的测试矩阵同步迁移） |
| **R4** | eDOS 总态数 $N_{\text{states}}^e$ 的解析值与数据集实际归一化不一致 | eDOS 能量窗 [-10,10] eV 不必然包含全部价态；数据集创建时的归一化约定需核实 | 中 | **Week-1 前置核验项**：对训练集逐样本比对 $\sum g \Delta E$ 与化合价电子数，若存在系统性比例因子，将 $N_{\text{states}}^e$ 改为"解析值 × 可学习全局标定常数"（每模态仅 1 个参数，无泄漏风险） |
| **R5** | ODE 数值积分误差累积导致带隙泄漏 | RK 步长在 $Q$ 平台边缘的高曲率区产生过冲 | 中 | RK45 自适应步长 + 平台区检测（$|dQ/du| < \epsilon$ 时局部减步）；一致性蒸馏天然抑制多步误差 |
| **R6** | 8 采样集成的推理成本拖垮 15 万级筛选 | 全量 ODE 采样 × 集成 | 低 | 两级推理：漏斗第 1 级用 1 步蒸馏 + 4 采样（~72 ms/材料）；第 2 级对 Top-1 万候选启用完整 8×10 采样 |
| **R7** | 联合流训练中 e-ph 梯度串扰重启负迁移 | M1→M2 的教训：共享梯度通路导致任务干扰 | 中 | 门控零初始化语义完整保留（$\alpha$ 从 0 起步）；备用方案：e/ph 速度场解耦训练 30 epoch 后再解冻联合层（课程式） |
| **R8** | 审稿质疑"分布指标提升来自平滑而非真多模态" | 缺乏多解性 ground truth | 中 | 构造**合成多解性验证集**：对已知多形体系（如 TiO₂ 三相）用同一未弛豫输入 + DFT 三相标签做分布覆盖检验（覆盖率/模式数），这是最有说服力的直接证据，亦是独立可发表的分析实验 |
| **R9** | 训练不收敛 / FM 损失与矩损失冲突 | 多目标梯度冲突 | 低-中 | 沿用 M5 的应急回炉判据风格：若 20 epoch 内 $\mathcal{L}_{QFM}$ 未达先验基线 1.5×以内，冻结流训练退回 M6 确定性路线（M6 本身是完整可交付成果，方案无"全输"分支） |

**风险组合评估**：R1 是主风险，但方案设计了 M6（确定性守恒）作为独立可交付层——即使生成式部分增益不及预期，M6 对标 DeepSciReasoner 的"守恒谱场解码器 + uniARPAT 跨模态"组合仍构成完整贡献；M7–M10 为增量上探。整体呈现"保底不输、上不封顶"的结构。

---

## 10. 执行路线图与论文定位

### 10.1 六周路线图（衔接现有 Week-2 M1–M5 消融节奏）

| 周次 | 里程碑 | 验收判据 |
| :--- | :--- | :--- |
| Week-2（并行） | M1–M5 百轮消融照常收割；同时完成 R4 的 $N_{\text{states}}$ 核验与物理先验谱库构建 | Table 1 填平；先验谱库离线校验通过 |
| Week-3 | M6（守恒谱场确定性）实装 + 10 epoch Pilot | Pilot 显存 ≤11GB、无 NaN、R4 核验报告归档 |
| Week-4 | M7/M8 分位数流匹配训练；一致性蒸馏 1 步版 | M7 vs M6 的 W-1 距离与 R² 对照；蒸馏版与 10 步版单点偏差 < 2% |
| Week-5 | M9/M10 全量百轮 + 不确定性校准评测 | PICP@90 ∈ [0.85, 0.95]；FWHM 误差报告 |
| Week-6 | R8 多形体系合成多解性验证 + 高通量漏斗联调（对接 thermo_props 与 Julian-Slack 筛选） | 15 万候选全漏斗 < 1.5 h；Top 候选进入 DFT 复算 |
| Week-7+ | 论文成文（方法章 + 应用章闭环） | 投稿目标：npj Computational Materials / Nature Communications |

### 10.2 论文叙事定位

- **方法学贡献**（计算机算法深度）：首次提出**测度空间守恒流匹配**（quantile-flow），将 1D 最优输运的McCann插值几何与条件流匹配结合，实现"训练-速度场-积分-采样"全流程构造性物理约束；附赠 $W_1$ 距离闭式化、带隙的原生分位数表征等理论红利。
- **物理学贡献**（材料能谱真实增益）：未弛豫结构的谱分布建模、构造性守恒消除软惩罚范式、解析可微热力学链路（低温 Cv 精度、费米面斜率描述符）、不确定性感知的高通量热电粗筛。
- **与 ARPAT（npj 2026）/uniARPAT 的传承**：Encoder、门控跨模态、Shape-Scale 物理哲学一脉相承；ScaleHead 从"回归标度"进化为"解析守恒"——恰好构成学位论文第三章"从解耦到守恒、从回归到生成"的方法论演进主线。

---

## 11. 参考文献

1. Zeni, C. et al. A generative model for inorganic materials design (MatterGen). *Nature* **639**, 624–632 (2025).
2. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M. & Le, M. Flow matching for generative modeling. *ICLR* (2023).
3. Liu, X., Gong, C. & Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR* (2023).
4. Luo, X. et al. CrystalFlow: A flow-based generative model for crystalline materials. *Nat. Commun.* **16**, 9267 (2025).
5. Miller, B.K., Chen, R.T.Q., Sriram, A. & Wood, B.M. FlowMM: Generating materials with Riemannian flow matching. *ICLR* (2025), arXiv:2406.04713.
6. Fu, X. et al. Flow matching for accelerated simulation of atomic transport in materials (LiFlow). *npj Mach. Intell.* / arXiv:2410.01464 (2025).
7. Kim, S. et al. High-order equivariant flow matching for density functional theory Hamiltonian prediction (QHFlow). arXiv:2505.18817 (2025).
8. Sriram, A. et al. FlowLLM: Flow matching for material generation with large language models as base distributions. arXiv:2410.23405 (2024).
9. Das, K. et al. LLM meets diffusion: A hybrid framework for crystal material generation (CrysLLMGen). *NeurIPS* (2025), arXiv:2510.23040.
10. Wang, Y. et al. DeepSciReasoner / DeepDOSReasoner: Physics-grounded reasoning for density of states prediction. *ICML* (2026, Oral); demo: gomes-lab.github.io/deepdosreasoner-demo.
11. Kong, S. et al. Density of states prediction of crystalline materials via prompt-guided multi-modal transformer (DOSTransformer). arXiv:2311.12856.
12. Fung, V. et al. Density of states prediction for materials discovery via contrastive learning from probabilistic embeddings (Mat2Spec). *Nat. Commun.* **13**, 949 (2022).
13. Al-Fahdi, M. et al. Rapid prediction of phonon density of states by crystal attention graph neural network. *Mater. Today Phys.* **50**, 101632 (2025).
14. Chen, Z. et al. Direct prediction of phonon density of states with Euclidean neural networks. *Adv. Sci.* (2021).
15. Lee, S. et al. Multimodal foundation models for material property prediction and discovery (MultiMat). *Newton* (2025), arXiv:2312.00111.
16. Jiao, R. et al. An equivariant pretrained transformer for unified 3D molecular representation learning (EPT). *Nat. Commun.* (2026).
17. Passaro, S. & Zitnick, C.L. Reducing SO(3) convolutions to SO(2) for efficient equivariant GNNs (EquiformerV2 基础). *ICML* (2023); EquiformerV3: arXiv (2026), OMat24 基准.
18. Wood, B.M. et al. UMA: A family of universal models for atoms. *NeurIPS* (2025).
19. Jayasundara, D. & Patel, V.M. Implicit neural representations: A signal processing perspective. arXiv:2604.15047 (2026).
20. Phy-CoSF: Physics-guided continuous spectral fields reconstruction and spectral super-resolution. arXiv:2605.13583 (2026).
21. SSIF: Physics-inspired implicit representations for spatial-spectral super-resolution. *ICLR* (2025) submission, OpenReview.
22. The Adaptive Vekua Cascade: A differentiable spectral-analytic solver for physics-informed representation. arXiv:2512.11776 (2025).
23. Shen, L. et al. PEINR: A physics-enhanced implicit neural representation for high-fidelity flow field reconstruction. *ICML*, PMLR 267 (2025).
24. Felder, J. et al. Conditional flow matching for ML-based inverse design problems. arXiv:2609.00863 (2026).
25. InvDesFlow-AL: Active learning-based workflow for inverse materials design. *npj Comput. Mater.* (2025).
26. McCann, R.J. A convexity principle for interacting gases. *Adv. Math.* **128**, 153–179 (1997).（McCann 插值）
27. Santambrogio, F. Optimal Transport for Applied Mathematicians. Birkhäuser (2015).（1D Brenier 映射与单调重排）
28. Lipman, Y. et al. Flow matching guide and code. arXiv:2412.06264 (2024).
29. 本组前期工作：ARPAT—Unified representation and joint learning for electronic and vibrational band structures of crystals. *npj Comput. Mater.* (2026), DOI: 10.1038/s41524-026-02199-3.

---

## 附录 A：与现有代码的对接映射表（实施索引）

| 新模块 | 对接点 | 修改性质 |
| :--- | :--- | :--- |
| `SpectralFieldQuery` | `transformer.py:84-89`（替换自由参数 query） | 新增类，替换初始化 |
| `QuantileState`（分位数状态封装） | `Transformer.forward` 的 decoder 输入 | tgt 输入从参数改为流状态 |
| `ConservativeVelocityDecoder` | `TransformerDecoder` 外包一层 t-AdaLN | 后者代码基本复用 |
| `QuantileHistogram` | 新增（heads.py） | 可微直方图 + L1 重归一 |
| `GatedSoftmaxHead` | `safe_shape_norm` 的超集 | 保留原函数为回退分支 |
| `physical_prior_quantile` | 数据管线离线缓存（dataset.py） | CPU 端预计算 |
| `thermo_props` 向量化 | `thermo_props.py:25-80` 循环改 einsum | autograd 化 |
| 评测扩展（W-1/PICP/FWHM） | `model.py:test_one_step` + `evaluate_and_plot.py` | 新增指标函数 |

*报告完 — Omen Alpha, 2026-09-08*
