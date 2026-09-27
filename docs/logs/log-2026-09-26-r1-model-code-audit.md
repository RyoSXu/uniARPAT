# 日志：2026-09-26 — R1 结构到双谱的信息通路代码审计

## 一页结论

马尚酱，本轮只读源码、已有日志和测试定义，补充上一轮几何调研的接口证据；没有运行模型、
加载 checkpoint、训练、拟合探针或读取新的 test 标签。唯一写入是本日志。角色为 R1 代码机制
审计；实际模型菜单显示名无法由本 agent 独立核验，不把编排建议的模型名冒充实际运行元数据。

1. **B7 的严格盲点成立，但不能外推为全体 Q1 根因。**原子初始特征只依赖元素；普通 encoder
   的几何只改变 softmax 权重，不直接改变 Value。eval 下单元素相同 Value 的加权平均仍相同，
   后续 decoder 和 H1 没有独立几何入口。已有冻结 B7 合成核验支持这个推导。但 valid 单元素仅
   37/2313（1.60%）；多元素的几何权重本来就能改变元素混合，不能称所有结构信息都未进入网络。
2. **G2a 已经修复“消息内容完全不含几何”的缺口，新方案必须超出它。**其每层残差是发送端
   Value × 径向门控 × cutoff，再按入度平方根归一化求和。它保留多镜像和跨胞自边，可改变同质
   节点内容；已有通路核验说明影响确实到达输出。尚未显式建模的是接收端—发送端—边的联合非线性
   内容、边与边的角关系／有约束的方向消息。两者是不同假设，不能一次全换后归因。
3. **优先研究“联合边条件内容”的表达差别，但当前不足以选定精度方案。**这个问题适用于多元素
   局部环境，不局限于 1.6% 的单元素子集；不过 G2a 平局、已有径向内容已生效、普通多层网络能间接
   合成环境，都是强反证。角度／晶胞重表达和谱读出仍是候选解释，encoder 不是已定位的唯一瓶颈。
   本轮没有任何结果支持精度提升，也不建议恢复连续训练集拆层诊断。

可复用的核心边界是 `src/pos/mask`、G2 多镜像边、`[B,L,512]` 原子 memory、现有共享 decoder、
双谱 logits 和 H1。新增模块的旋转／重表达性质不会自动成为完整网络性质：82 格式消去外部整体
朝向，但换晶格基底可能改变 B7 的单镜像选择和内部方向。不得把这两种变换混为一谈。

交接状态：**closed（本角色审计完成）；候选仍待综合和独立评审，未授权实施／训练。**

## 范围、来源与快照

- 已按顺序阅读 `docs/status.md`、`docs/index.md`、`docs/workflow.md`、`docs/glossary.md`、
  `AGENTS.md`，以及编排设计 P0/P1、几何调研；涉及结构输入时补读 `docs/data.md` 和
  `index/z0_REPORT.md`。本报告的数字来自既有日志，没有重新计算模型指标。
- HEAD：`45c2e8561e61d3d7a10d0d25c2c975e664bb6e14`。
- 开始与写入前的已有改动相同：`M docs/index.md`、`M docs/status.md`；未跟踪
  `docs/design/design-model-research-agent-orchestration.md`、
  `docs/logs/log-2026-09-26-agent-orchestration.md`、
  `docs/logs/log-2026-09-26-geometry-message-research.md`。相关源码没有未提交修改。
- 不修改这些文件及其他 agent 的交付，不执行 Git 写操作；共享页面由 R0 收口。
- 下文“代码事实”是源码可直接确认的行为；“推导”是公式结论；“假设”需要未来独立验证。
  本轮未重跑历史测试；读到测试定义不等于本轮测试通过。

## 证据附录 A：完整调用链和张量

| 模块／调用链 | 文件 → 符号／行号 | 实际信息流 |
|---|---|---|
| CIF → 82 行结构 | [utils/b7_cif_inference.py](../../utils/b7_cif_inference.py) → `structure_to_b7_inputs`，87–139 | 保留输入晶胞；`src[0:2]=126,127`，之后是原子序数，零为 padding；`pos[0]=(a,b,1/c)`，`pos[1]=(α,β,γ)`，之后为分数坐标。最多 80 原子，不自动标准化为原胞。 |
| 缓存 → batch | [datasets/dataset.py](../../datasets/dataset.py) → `Dos_Dataset.__getitem__`，112–153；[model/model.py](../../model/model.py) → `basemodel.data_preprocess`，143–201 | 元素和位置来自缓存；`mask=(inp==0)`。真实谱及尺度统计是训练／评估目标，不传给 encoder；网格中心是常量元数据，B7 默认 query 不使用它。 |
| 原子初值 | [model/transformer.py](../../model/transformer.py) → `Transformer.forward`，340–359；[utils/atom_feature.py](../../utils/atom_feature.py) → `AtomFeatureEncoder.forward`，118–123 | 去掉两个哨兵，元素 embedding 与元素质量／半径／电负性投影分别 LayerNorm，再拼接投影为 `[B,L,512]`。同元素初值相同；默认无坐标／晶胞内容注入。 |
| 结构 → 规范内部晶胞 | [utils/relative_features.py](../../utils/relative_features.py) → `build_cell_from_lattice`，5–45 | 从长度／角度重建行向量晶胞 `A`，恢复 `c=1/inv_c`；返回 `A:[B,3,3]` 与 `f:[B,L,3]`。 |
| B7 几何 | 同文件 → `compute_relative_features`，47–74；[utils/rp_encoding.py](../../utils/rp_encoding.py) → `RPEncoding.forward`，19–40 | 单个逐分量折回位移 `δ=f_i-f_j-round(f_i-f_j)`，`r=δA`；距离和方向经 64 个径向基 × l=0,1,2 球谐，成为 `[B,L,L,576]`。encoder 显式选择 lmax=2，不能误读 RPEncoding 构造默认的 lmax=3。 |
| B7 encoder | `TransformerEncoder.forward`，574–611 → `TransformerEncoderLayer.forward`，695–755 | 六层共用几何基，每层独立 `rp_proj`。几何只参与打分；Value 为当前原子状态。注意力聚合后残差、LayerNorm、逐原子前馈。输出 `[B,L,512]`。 |
| G2 可选支路 | [utils/g2_periodic_edges.py](../../utils/g2_periodic_edges.py) → `build_g2_edges`，73–181；`PeriodicEdgeMessage.forward`，57–80 | 一次枚举所有 cutoff 内周期有向边；每个普通 encoder 层之后调用独立的 G2 消息模块。边在同一前向各层复用；接收当前层节点内容，产生径向条件残差。默认关闭。 |
| query → 共享 decoder | `Transformer.forward`，440–487 → `TransformerDecoderLayer.forward`，830–860 | 默认 eDOS/phDOS 各有可学习 query 与 target，形状 `[128,512]`／`[64,512]`；同一 decoder 分别跑两次。Self-attention 使用 target+query 的 Q/K，target 的 V；cross-attention 使用 query 相关 Q、memory 的 K/V。`pos` 虽被传入，层内没有用它。 |
| decoder → 双谱 | `Transformer.__init__`，320–323／`forward`，505–509；[model/heads.py](../../model/heads.py) → `CNN`，129–155 | 默认 eDOS 单层 Conv1d，phDOS 六层 Conv1d；输出 `[B,128]`／`[B,64]` logits。卷积沿谱轴，不是原子轴。 |
| memory → H1 | `Transformer.forward`，525–528 → `global_masked_pool`，heads.py 324–334 → `EtaHead`，277–300 | 对有效原子平均池化 `[B,512]`，MLP+sigmoid 输出 `[B,2]` 的 eta_ph/gamma_e。H1 绕过 decoder，依赖同一 memory。 |
| logits/H1 → blind DOS | `predict_b7_blind`，b7_cif_inference.py 173–217 | `p=softmax(logits)`；`DOS_e=p_e*N_val*gamma/ΔE`，`DOS_p=p_p*3*N_atoms*eta/Δω`。N_val 和 N_atoms 来自 CIF；oracle 评估则使用真实目标总量，不能混报。 |

实际 B7 参数依据 `utils/b7_cif_inference.py:33–53::B7_TRANSFORMER_PARAMS`：512 维、8 头、
6 层 encoder／共享 6 层 decoder、legacy3、legacy CNN、dropout 0.05、H1；不以可编辑模板或
其他历史分支推断默认行为。SumNorm 损失见 [model/losses.py](../../model/losses.py):30–52
`sumnorm_klw_loss`；默认无 coverage loss mask。这里没有更改数据契约。

## 证据附录 B：几何究竟进入哪里

### B1. B7：权重进入，内容没有独立边项

每头的核心公式（省略逐原子前馈和残差）是：

`s_ij = h_i·h_j/sqrt(d_head) + h_i·P(RBF(d_ij)⊗Y_l(u_ij))`

`a_ij = softmax_j(s_ij), m_i = Σ_j a_ij h_j`。

代码事实：`q=k=v=src`（703）；没有独立的 Q/K/V 学习投影和注意力输出投影，`rp_proj`
是逐层的几何投影；其后逐原子前馈会混合通道。注意力几何项没有和 base term 一起除
`sqrt(d_head)`（712、715–721），不能假定它的尺度已校准；是否饱和或主导未知，未查权重或激活。

推导：若所有有效 `h_j=v`，`Σa_ij h_j=v`。eval 的逐原子 LayerNorm／前馈保留相等性，
归纳到末层；decoder 的常量 memory 加权、H1 平均池化都无法恢复被抹去的几何。因此单元素
且固定元素／原子数的结构变化不会改变理论输出（浮点误差除外）。训练 dropout 可随机打破
节点相等，但不是可复现的结构编码，不能据此声称 eval 盲点被修复。

多元素第一层仍可用几何调整各元素 Value 的总权重；同一元素多个邻居只通过其总注意力质量
起作用。后续层的同元素状态可能已随环境不同。因此“每层所有 Value 都只是元素”是错误扩张。
已有 [近期审阅日志](log-2026-09-26-recent-commit-review.md) 给出冻结 B7 合成核验及单元素比例，
也给出 258 个结构匹配判为不同的多元素 valid 对，目标／预测 TV 中位数 0.28229/0.01713。
它支持条件响应不足的线索，不能单独定位其发生在 encoder 或读出。

### B2. G2a：内容、求和和配位都已进入

`m_i = deg_i^(-1/2) Σ_(j,T) c(d_ijT) [W_v h_j ⊙ sigmoid(W_g RBF(d_ijT))]`

`h'_i = h_i + alpha_l W_o m_i`，其中 `deg` 下界是 1。

代码事实：边在 5.5 Å 内严格保留，排除同胞自边 `(i=i,T=0)`，保留跨胞自镜像；无 top-k。
同一距离的门控在给定层不依赖接收元素或发送元素，但 Value 本身依赖发送端当前状态。
每层消息接在普通层之后，故 `h_j` 可以已经包含 B7 方向信息；不能把整个 G2 网络称作纯径向模型。
`W_o` 在聚合后混合通道，并有偏置；有边 batch 中无入边节点仍可能收到 `alpha*W_o(0)`。
整个 batch 边为空则直接返回 h。padding 的非零隐藏状态本身不是污染证据，关键是是否参与后续聚合。

推导：同质 Value 不再必然消去径向门控；若各边消息相同，聚合随 `sqrt(deg)` 缩放，保留配位数
信号。它不是归一化门控平均，也不是 `Σgate*v/Σgate`。但对相同的发送状态—距离多重集合仍
给出相同分支消息；分支没有直接接收 `h_i`、边角、向量／张量中间量。

边数是硬 cutoff 的计数，因此“单条消息乘光滑 cutoff”并不证明完整聚合随结构严格光滑：
边出入 cutoff 时 `deg` 变化会重缩放其余边。这是合同边界，未测其数值影响，不另立优化任务。

最强已有反证：[G2a pilot](log-2026-09-20-g2a-pilot.md) 的 Q1/test/ep10/seed42，
oracle 双任务 Δmedian 为 −0.0066/−0.0048，耗时 +27.3%、显存 +16.5%，已 park。
后来 [冻结通路核验](log-2026-09-26-g2-structure-path-audit.md) 在 Q1/valid/ep10 看到
encoder／decoder 相对 RMS 0.05368/0.06374、eDOS TV 0.01469，但独立训练 edge−control 的
eDOS oracle Δmedian 仅 −0.000709，区间跨零。不能再以“G2 完全没传到读出”为升级理由。

### B3. 输出端：几何只以 memory 中的内容间接出现

decoder 的 `pos` 参数未使用；没有 energy–atom–geometry 的独立三方边项。不同谱点靠自己的
query、target 以及 token 间 self-attention 选择 memory。B7 默认 learned target 不是全零输入：
构造时虽以零分配，`_reset_parameters`（335–338）会对二维参数作 Xavier 初始化。
R1b 则同时把 query 换成坐标 MLP、target 置零（424–438）；其失败不能只归因于坐标 MLP。

H1 的平均池化会消除原子总数的显式尺度，符合 eta/gamma 作为比率的语义；blind 后处理另外乘
N_atoms/N_val。不能只凭“用了平均”认定 H1 应改为求和。softmax 输出处允许 logit 整体平移等价，
但这不是结构谱差小的充分原因。现有源码未显示查询饱和、memory 实际秩或 H1 分辨率，均未知。

## 证据附录 C：结构、掩码、不变性与兼容边界

| 合同 | 可确认的边界／迁移含义 |
|---|---|
| 全局旋转／反射 | 82 格式仅保存晶格长度、角度和分数坐标。对晶胞和原子一起作全局刚体变换，输入相同，内部仍重建同一规范晶胞；不能因为球谐被线性混合就断言该输入的全局旋转不变性失效。若直接旋转内部方向张量，B7 的任意 `rp_proj` 不保证标量不变；两者不是同一操作。 |
| 换晶格基底／等价晶胞重表达 | 重建内部朝向可能改变；B7 的逐分量 round 也不是一般斜晶胞的欧氏最近镜像搜索。即使单镜像距离偶然相同，方向打分也没有不变性保证。`tests/test_g2_periodic_edges.py:1–22` 明确把基底交换／幺模变换（整数可逆且行列式±1的基底变换）限定为边距离和独立径向模块合同，完整模型未获此保证。 |
| G2 完整枚举 | `K=ceil(R/s_min(A)+0.5)`；先把分数差中心化，再枚举立方 shift 集。若某镜像距离<R，则分数位移范数<R/s_min，每个坐标加上最多0.5的中心差足以落入该界；因此是保守界，不依赖固定平移盒。病态晶胞可能使枚举昂贵，退化晶胞拒绝；未测本轮成本。 |
| 周期整数平移／全局平移 | 分数差抵消整体平移，round 通常消去整格平移。半格精确边界有舍入择边细节，有限精度也有容差，不声称逐位通用合同。G2 保留全部镜像，`shifts` 会随表达改变，距离多重集才是合适比较对象。 |
| 原子重排 | 元素、坐标、mask 同步重排时，原子编码按相同规则置换，decoder 对 memory 集合读取、H1 平均给不变图级输出（eval、忽略舍入）。已有冻结通路日志提供实际容差记录，本轮没有重跑。 |
| 超胞 | G2 的局部邻居集合有物理等价依据，但 B7 单镜像稠密支路、输入原子数上限及后处理尺度均需另验。CIF 入口明确把超胞作为不同输入，不承诺与原胞数值互换；不能借模块性质悄悄改变此公共合同。 |
| mask/padding | True=padding；encoder 仅遮 key 列，没有清零 padded query 行；G2 明确去掉 padded 发送／接收端；decoder 的 memory mask 与 H1 有效原子平均最终阻断 padded 内容。新聚合必须继续用 mask，不可直接平均全 L。全空输入不在 CIF 合同内；有限 -1e9 mask 不是全空 attention 安全语义的证明。 |
| 距离截断 | B7 距离 clamp 到11 Å，不是10 Å邻接 cutoff；所有有效原子对仍参加注意力。超过11 Å时变量 `unit_dirs` 在返回时未必单位长，但球谐调用 normalize=True；径向距离已饱和。迁移时不要把默认10与G2的5.5混成同一图。 |
| 向量复用 | G2 返回 `batch,dst,src,shifts,distances,k,L`；现有消息只消费 `batch,dst,src,distances`，不消费 shifts。若要方向可用 `(f_src-f_dst+T_true)A` 重建；若要边角还要按共同中心配对边，并保持镜像身份。B7 位移方向是 i−j，G2 是 j−i+T；奇数阶球谐符号不能直接混用。 |
| 梯度 | `build_g2_edges` 受 `no_grad` 装饰；其返回距离不支持通过几何回传坐标梯度。当前 DOS 任务不需要力，但不能把它直接宣称为可微势能／力接口。 |
| checkpoint | `use_g2=False` 不创建 g2_msgs，原 B7 state-dict 键保持；打开会新增参数，不能 strict-load 老权重后假装全量兼容。零初始化 alpha 可作同权重起点，但须核对缺失键与输出，不能泛化成任何新模块零残差均兼容。`load_b7_model` 明确 strict=True；恢复模块 `restore_ablation_checkpoint` 使用默认 strict 的 `load_state_dict`。变输入几何而不变键也可能改变旧checkpoint语义。 |

可复用接口的最小图是：结构输入 → 既有周期边／mask → 候选原子更新 → `[B,L,512]` memory
→ 既有 decoder 与 H1。若新增角度或方向，推荐先保留这个输出形状，但“保留形状”不等于
数值兼容、不变性完整或训练分布兼容。不得跨归一化方案复用 checkpoint。

## 证据附录 D：最多三个瓶颈候选

### 候选 1（优先研究）：几何与化学状态的联合消息内容不足

- **事实与位置：**`TransformerEncoderLayer.forward:703–745` 仅几何权重；
  `PeriodicEdgeMessage.forward:65–80` 是逐通道可分的“发送状态线性项 × 径向门控”，接收状态未
  直接进入消息，聚合前没有接收／发送／边的联合非线性内容。
- **假设：**不同元素组合对相同距离的谱贡献需要接收端条件及非线性相互作用，当前有限层／维度
  的分解形式不容易高效学到。这可覆盖多元素完整 Q1，而非只修单元素；但发生频率和收益未知。
- **替代解释：**各层普通注意力及 G2 输入状态已经有环境依赖，聚合后的前馈可合成类似关系；
  数据支持不足、训练目标或读出可能更重要。不能用代码形式差异宣称不可表达。
- **影响范围／与 G2a 的差别：**只研究消息内容函数在相同周期边、cutoff、聚合、层位置、memory
  合同下的变化；不是重跑G2或把更多角度／新聚合同时加入。将来参数匹配与成本仍需明确。
- **最强反证：**G2a 内容已生效却未赢；纯谱差联合适配也未改善留出，排除“只要增加结构响应就赢”。
- **会改变取舍的结果：**若参考公式仅重写成发送项×径向门控，就淘汰“新增联合能力”的理由；若
  真有不同表达，才可形成待批准的独立valid设计。未来全Q1 valid、完整谱目标的成对结果若平局／
  退化则park，不能因同组成子集TV变大晋级；若预注册整体收益与保护通过才考虑确认。这里不启动实验。

### 候选 2：周期表示和方向使用未提供一致的环境几何合同

- **事实与位置：**`compute_relative_features:61–72` 只取一个逐分量镜像；
  `RPEncoding.forward:31–38` 与 `rp_proj` 将有方向分量输入任意标量打分；G2虽枚举完整镜像，
  只把距离送入消息。完整B7+G2没有显式的边—边角关系或有约束的等变更新。
- **假设：**某些相同／近似径向分布而不同角环境，以及同晶体不同基底表达，难以被当前通路一致
  地处理。影响可跨元素类别；但Q1是否存在足量相关例子、它们是否主导错误未知。
- **替代解释：**规范化输入可能已减少表达差异，B7成对方向加多层注意力也可能间接编码角关系；
  单镜像不完整并不代表监督任务所需信息必然不足。
- **影响范围／与 G2a 的差别：**候选必须明确研究不变角内容还是替换B7几何路径；两者不能打包。
  仅附加一个不变残差不会修复保留支路的重表达敏感。角邻居对成本通常随度数平方增长，未实测。
- **最强反证：**G2完整多镜像+径向内容并未赢，因此“只修周期边”理由已不够；目前没有已定位的
  Q1角度混叠或基底表达误差贡献证据，完整几何并不是当前DOS精度必要且充分的证明。
- **会改变取舍的结果：**若候选仅在内部旋转不变而CIF输入本来已消去全局朝向，不以此作为收益
  理由；若不能保持分支/全网合同区分或成本上限就修改/搁置。未来独立valid未获整体收益则停止，
  合成结构区分能力只作技术合同。无需再启动train小样本拆层诊断。

### 候选 3（保留竞争解释，当前不优先改读出）：谱任务对 memory 的使用／监督未形成可迁移差异

- **事实与位置：**`TransformerDecoderLayer.forward:844–848` 只读memory；
  `Transformer.forward:525–528` 的H1亦只读memory；`CNN.forward`沿谱轴映射，
  `sumnorm_klw_loss`针对完整谱形。已有多元素谱差很小，但G2干预的影响确实穿过decoder。
- **假设：**表示中有可用的细粒度结构差异，实际query读取与谱目标更倾向共同谱形，或谱标签分布
  缺少足够训练支持。这是竞争解释，不是已证明的“decoder丢信息”。
- **替代解释：**memory可能缺少与目标相关的信息，现有谱差可能含来源差异，短预算优化或标签
  支持不充分也可解释。R1a技术平局、R1b组合失败、R2b失败均不支持任意替换读出。
- **影响范围／与 G2a 的差别：**针对完整谱形的可迁移使用方式，概念上作用于全体双谱；应保持
  encoder/geometry而只改变明确的读出或训练因素。不能同时更改encoder和损失并归因读出。
- **最强反证：**冻结读出谱差训练valid恶化，联合适配也未胜；纯谱差目标破坏共有谱形。
  [路线复盘](log-2026-09-26-g2-probe-route-review.md) 已撤回连续拆层。
- **会改变取舍的结果：**论文若只有图级性质或仅依赖训练时真实谱输入，不足以迁移部署；若能提供
  保持CIF-only推理、完整谱目标的机制并区别于已失败探针，才保留后续候选。现有证据不足以再次
  改query、单纯加谱差目标或恢复probe。未来整体valid收益而非train拟合才改变优先级。

这三项是瓶颈假设，不是三项已授权实施方案。排序仅是代码审计给R0的研究优先级；若跨来源
证据不能补足“为什么影响完整Q1”的理由，合法结论仍是暂不改模。

## 给论文 agent 的三个机制问题

1. 在**相同周期边与聚合**上，参考实现的消息内容是否真的加入 `h_i,h_j,e_ij` 联合非线性？
   用明确公式说明何种环境能区分于G2a逐通道乘法，哪些情况又等价；接收端是否只影响门控？
2. 哪个角度／方向片段能保留 `[B,L,512]` 标量memory，并给出模块而非整网的重表达性质？
   在保留B7方向支路时，哪些定理不能继承？请区分同质Value归一化平均与角度更新实际创造的内容。
3. 直接DOS的结构—谱对齐机制能否提供上述几何假设的竞争解释？训练和推理各需什么输入，是否
   保留完整谱监督，凭什么不同于已失败的纯谱差读出probe？不因它能拟合train就推荐。

## 快照 SHA-256

下表为本轮读到的实际文件内容；不是仅凭HEAD指认快照。相关源码在写入前仍无Git差异。
引用自检时发现R0同步更新了编排设计（初读哈希为
`640d936d054e06fd8b729b4f7b57eb877b57d4aa22c906f82c655673553b39ba`），已补读其P0及验证边界，
写报告授权与本任务一致；下表登记结束时版本。14个本地Markdown引用均存在，源码哈希未改变。

| 文件 | SHA-256 |
|---|---|
| model/transformer.py | `8bb1235c0ef9e8b626b93c13112d561bd650f8887ef809a78e4fdc3314db269b` |
| utils/relative_features.py | `b7402c099c98284bd98d270e5b999bfded12cd2d894c75f8fe5cbf809d47cd52` |
| utils/rp_encoding.py | `f87c4c310f1d706f2f07d1a4dc8f2c5d11fd7e50da84640a0ad206336cf4a993` |
| utils/g2_periodic_edges.py | `e6dba018b05b8804c5e9679d8c978180f1a7738cdcfb43009bd5aea6e4ec811e` |
| utils/atom_feature.py | `b71bb2c3d0d472ea3cd7b33b169c80e5b6ba1aa802fcfb557637efaa407df491` |
| model/heads.py | `595c5fde69f26c7d4fe73a1a60406076cc4c84845fa3d0285dbae4a0cb1388c2` |
| datasets/dataset.py | `6740d3808a3a07a549a39ff0dd77743361870c260ac60135fec74102b3fca191` |
| model/model.py | `966712abed392d68582210cb8356d7103228a754d5a7cb787571216b5376ef65` |
| model/losses.py | `834fd50eaa1bbac138e2ccaaa1356bcddab76ce5d32815683ae4c4b6da5c173b` |
| utils/b7_cif_inference.py | `f3a044648df8cc9684eeb14ced02495933340840239da3e174c799975a3948ea` |
| tests/test_g2_periodic_edges.py | `b1254c76ac23e48f02280ba1211dcf17254c895352bf9f24999f8a7b18eed8e8` |
| utils/ablation_checkpoint.py | `8660241a8fbd615e153d999631174aca103b1ce0a20cc7eba1eb55f8f1ab982f` |
| run_ablation_experiments.py | `04eb02c8c9db9c49d2b7f258d159ae8b1460c0fad4b6f04d12118cf60c55bd4b` |
| utils/experiment_config.py | `b1afa0f6211021d20022b8313997052935efa9803fa81a4e38d6958147ffd84c` |
| docs/design/design-model-research-agent-orchestration.md | `1cd33721ac18e3f0d5d87ca234c0389d0cdb576bc875b173a40641a5ebc78588` |
| docs/logs/log-2026-09-26-geometry-message-research.md | `fcd64ada127cf3fde00ca3676fb94b521db4c700e6d920e87ed8a1bd3f5b2fd1` |

## 验证与交接

- 新增文件仅本日志；新增、删除或修改的生产符号：**无**。
- 实际核验：只读源码、配置入口、已有日志、测试定义；核对HEAD／工作区状态与上表哈希；
  检查本日志本地Markdown引用和文档空白差异。不将历史测试结果算作本轮执行。
- 未验证：完整Q1上的几何混叠发生率、任一新机制的精度、参数/显存/耗时、旧checkpoint新模块
  加载、数值旋转／重表达／padding测试、注意力饱和或memory的信息充分性。无模型或GPU运行。
- 论文机制的原始来源复核由R3承担；本报告只提出代码问题，不重复上一轮综述，不依据未读论文
  推断ComFormer或其他模型必须采用。当前没有资料缺失阻止这份代码交付。
- 由R0汇总本报告与R2/R3并独立评审，最多形成两个具体候选；共享status/index/decisions由R0
  处理。本角色不自行推进下一阶段，不修改默认方案，不声称研究已带来精度收益。
