# 日志：2026-09-26 — 周期几何消息的第一轮文献与实现对照

## 范围

- 马尚酱确认开始重新研究模型本身，工作方式为“理解项目缺陷 → 学习相关机制 → 判断可迁移内容”。
  本轮完成消息机制的第一轮对照；不是架构实施设计，也不授权训练或数据扩充。
- 只写本日志和 `docs/status.md`。保留已有 `docs/decisions.md` 改动及 opencode 的
  `log-2026-09-26-data-support-feasibility.md`，不修改模型、依赖、数据契约、缓存或检查点。
- 同时复核 A 线候选数据的材料 ID 数量，以回答是否需要继续数据工作；没有重跑其原始谱质量审计。

## 证据与调用链

### 1. 当前模型的边界

- 几何输入：`utils/relative_features.py::compute_relative_features` 将每个原子对折回一个
  分数位移，再转换为实空间距离和方向；它没有完整周期多重边集合。
- 编码器：`model/transformer.py::Transformer.forward` 产生元素初始特征，调用
  `TransformerEncoder.forward`；`utils/rp_encoding.py::RPEncoding` 提供径向基与球谐特征，
  `TransformerEncoderLayer.forward` 将其用于注意力打分。该层直接取 `q=k=v=src`。
- 若同元素原子的 Value 全部相同，softmax 加权满足 `sum_j a_ij v_j = v`，几何改变权重仍不能
  改变传递内容。已有冻结 B7 核验见 `log-2026-09-26-recent-commit-review.md`。
  这是表达边界；单元素仅占 valid 1.60%，不能据此解释全体精度。
- 表征到输出：`TransformerDecoderLayer.forward` 从原子 memory 读取内容，两个任务分别调用
  同一 decoder；既有输出头和 H1 继续负责谱形及 blind 尺度。
- G2a：`utils/g2_periodic_edges.py::build_g2_edges` 保留周期镜像，包括跨胞自镜像；
  `model/transformer.py::PeriodicEdgeMessage.forward` 在普通层后加入径向残差。
  它已能使同元素节点获得几何相关的消息，故下一项不能仅以“修复同元素盲点”作为新理由。

### 2. 消息公式的对照

下式仅提取消息机制，省略前馈、归一化和残差等细节。`h_i` 为原子隐状态，`e_ij` 为边特征，
`d_ij` 为距离，`c(d)` 为截断权重，`deg_i` 为入边数。

| 路径 | 核心消息／聚合 | 本轮判断 |
|---|---|---|
| B7 | `sum_j softmax(score(h_i,h_j,geometry)) * h_j` | 几何控制权重；相同 Value 的加权平均不产生新的环境内容。 |
| G2a | `sum_(j,T) c(d) * W_v(h_j) * sigmoid(W_g RBF(d)) / sqrt(deg_i)` | 径向几何已进入内容且保留配位信号；没有显式角度或等变向量消息。该具体方案已 park。 |
| 经典 ALIGNN 参考实现 | 原子更新含 `sum_j gate_ij * Vh_j / (sum_j gate_ij + eps)`；另在线图更新键与角度表示 | 线图值得学习，但不能仅凭“使用角度”就断言直接替换原子更新会修复同质 Value 问题。 |
| ComFormer 节点消息 | `gate(h_i,h_j,e_ij) * phi(h_i,h_j,e_ij)`，再求和 | 边特征同时进入权重和内容的联合非线性变换；区别于 G2a 的发送端特征乘径向门控。 |

- 对 ALIGNN 的限制判断来自公式：同质邻居、忽略分母 `eps` 时，门控在归一化平均中消去；
  实际代码有 `eps=1e-6`、BatchNorm 和额外边更新。本轮没有执行上游模型，不把这个推导写成
  “完整 ALIGNN 对结构完全不敏感”或其精度结论。
- ComFormer 两种几何处理需要区分：iComFormer 用与周期参考向量相关的角度更新边；
  eComFormer 经张量积传递中间的方向表示，再产生标量原子特征。“等变”指内部方向特征随
  坐标变换按约定变化；最终标量 DOS 需要不变的输出。
- 只把一个几何模块接在 B7 后面，不能自动继承整套 ComFormer 的表示完整性或不变性。
  B7 原方向敏感路径若仍存在，完整输出的晶胞重表达合同仍须单独检查。

### 3. 文献与参考代码

本轮阅读原论文及作者代码；以下来源用于机制比较，未作 Q1 上的运行／精度复现。

1. [ComFormer，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/0ab51646ca369140c3c3ece011b66587-Paper-Conference.pdf)：
   第 3 节为周期表示，第 4 节为消息机制。其标量性质基准不等于本项目 DOS 验收。
2. [ComFormer 消息实现](https://github.com/divelab/AIRS/blob/4a16c68a7da707c521019067dec51c227c10de45/OpenMat/ComFormer/comformer/models/transformer.py)：
   固定参考提交 `4a16c68a7da707c521019067dec51c227c10de45`；符号为 `ComformerConv.message`、
   `ComformerConv_edge.forward`、`ComformerConvEqui.forward`、`TensorProductConvLayer.forward`。
3. [ComFormer 模型拼接](https://github.com/divelab/AIRS/blob/4a16c68a7da707c521019067dec51c227c10de45/OpenMat/ComFormer/comformer/models/comformer.py)：
   `iComformer.forward`／`eComformer.forward`；现有图级池化和输出不能原样替代 B7 的原子 memory 接口。
4. [经典 ALIGNN 参考实现，v2025.4.1](https://github.com/usnistgov/alignn/blob/v2025.4.1/alignn/models/alignn.py)：
   `EdgeGatedGraphConv.forward`、`ALIGNNConv.forward`。本轮结论限于该版本，不涵盖其他重实现。
5. [ALIGNN phDOS 研究](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.023803)：
   提供直接谱预测的应用背景；没有据其指标推断在 Q1 上的收益。

### 4. 迁移边界与尚缺证据

- 优先复用已有周期边枚举、原子掩码、decoder 和 H1 接口；任何移植都需要解释输出如何恢复为
  `[batch, atoms, 512]` 的 memory。不能把图级单标量预测直接等同于本项目双谱预测。
- 引入角度边关系可能增加邻居对数量；等变张量积也有额外计算成本。本轮未测 Q1 运行耗时、
  显存或参数预算，不能宣称低成本。G2a 已有约 27% 耗时增量是现有资源警示。
- 本机模块发现检查：`torch`、`e3nn`、`torch_geometric`、`dgl` 可发现，`torch_scatter`、
  `torch_sparse` 不可发现。未安装依赖，也未验证这些库的运行兼容性。
- 固定 AIRS 快照的消息文件还出现 `comforemr` 导入拼写及 `super(MatformerConv, self)`
  与当前类名不一致。它可用于核对计算机制，但本轮没有证明该快照能直接运行。
- 下一轮研究应明确候选新增的是角度／方向环境表达、联合边条件消息还是主干替换；
  不能把这几项与归一化、图构造全部同时改变后，将收益归因于其中一项。
- 比较对象继续固定 Q1、完整谱形目标、现有读出与 H1。研究阶段不预先固定训练预算或启动命令。

### 5. 对 A 线新增数量的有限复核

只读取 `/root/home/newstudy/getdata/raw/` 的 ID 账本，以及
`/root/home/newstudy/getdata/v2_release/v2_processed.parquet` 的 `mpid` 列。

设 `S` 为 `census_effective_ids.json`，`P` 为 `census_phonon_map_canonical.json` 中值非空的材料，
`C` 为现有成品 24,988 条材料，`E` 为 `census_edos_materials.txt`，`A` 为 `edos_absent.json`。
本轮用 Python 集合交差运算复算如下：

| 项目 | 实测数量／关系 |
|---|---:|
| `S` | 154,373 |
| `P` | 26,609 |
| `C` | 24,988，全部在 `S` 中 |
| `C ∩ P` | 18,644 |
| `C \ P` | 6,344 |
| `S \ (P ∪ C)` | 121,420 |
| `A` | 7,965，全部在 `P` 内 |
| `A ∩ E` | 7,965 |

- 121,420 的材料集合候选数量可以复核；它不是经过读取、质量过滤、标签兼容性与新颖度核验的
  可训练数量。7,965 条待落实记录也在 eDOS 普查名单内，因此名单命中不足以证明标签可用。
- 本轮没有复核隔离样本的逐条原始谱；“未找到当前可行修复路径”不能扩大成所有来源均绝无可回收。
  中位数全掩不能证明每条全掩，也不能单独确定 NBANDS 是所有记录的原因。
- 把旧增量约 95% 落实率移用到已缺失群体，需要额外依据；本轮不接受它作为 7,500 条可靠新增的承诺。
- A 日志中“train 升、valid 不动 → 标签不可约”的归因过强。该结果最多说明指定扩充方案未获
  验证集收益，仍可能涉及选样、优化、表示或过拟合。
- eDOS-only 追加会改变双谱样本的出现次数与两个任务的有效监督量。相同 optimizer steps
  不自动等于相同计算成本或相同 phDOS 监督；将来设计需明确这些数量和损失归一化口径。
- 建议 A 暂不追加落实审计或数据加工，保留为后备证据；这是本轮排序建议，不代替用户批准
  新数据活动。没有修改 opencode 的日志或其已记录结果。

## 结论

- 状态：closed（本轮对照单元完成）；整体模型调研继续，尚未选定实施方案。
- 优先深读 ComFormer 的“几何同时进入消息权重与内容”及角度／等变更新，逐项对照 G2a。
  经典 ALIGNN 用于学习键角组织与核查聚合，不凭模块名称预先推荐整套移植。
- 新方案必须说明超出 G2a 的具体假设，以及为什么可能影响完整 Q1 valid；单元素敏感性恢复、
  表征有变化或合成例子通过均不能替代精度对照。
- 没有新增精度结果，没有修改默认模型，没有新训练、外部数据采集、依赖安装或 Git 提交。

## 验证与交接

- 实际操作：代码与文献只读审阅、上述材料 ID 集合复算、模块发现检查、文档差异与引用检查。
  无代码改动，未运行模型测试或上游参考程序。
- 下一项研究单元：讲清周期表示、消息内容与聚合的职责，评估至多两种具体迁移形式的
  表达差别、接口、成本与独立验证方式；选定设计后再请求实施所需的架构／训练决定。
- `status.md` 记录本轮已启动模型研究，保留历史小样本路线已结束的结论；不改 `decisions.md`。
