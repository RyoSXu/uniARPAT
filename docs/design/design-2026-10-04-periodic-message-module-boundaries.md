# 训练前的周期几何与局部消息模块边界

更新日期：2026-10-06。状态：**独立 A/B+ 与物理记录已接入完整新模型；基本接口与公式保持，首轮对照已暂停。**
依据为 [ZP 约定](design-2026-10-02-element-initialization.md)、[A/B 候选](design-2026-10-04-periodic-message-candidates.md)
及 [B+ 量化对照](../logs/log-2026-10-03-periodic-message-high-order.md)。
本页不选择最终 Encoder 路线、局部/全局编排或训练预算。历史诊断包装器仅作复核参照，
新实验使用独立核心；完整接入与无更新检查见[接入日志](../logs/log-2026-10-05-local-global-encoder.md)。
用户随后确认的标量状态、外层残差、标准全局注意力、更新前归一化和独立参数原则，
以及几何只进入局部聚合的限制，见[局部—全局接入设计](design-2026-10-04-local-global-encoder.md)。
A/B+、编排与深度仍待实验选型，不改变本页所记录的已实施局部接口。
随后用户授权的[计算提速](../logs/log-2026-10-05-local-message-speedup.md)仅调整 A 的全量邻边对分块和
角度首层线性项复用，B+ 免去无用的主机分组准备；公开接口、参数名/形状、科学公式与默认选项保持。
有限同权重输出/梯度及完整 E(3) 检查另记录于该日志，浮点结果按容差比较。

## 提取前的依赖与保留的历史实现

下表是提取的代码依据；原型、旧 Encoder 和已有实验产物仍保留，作为冻结的历史数值参考。
新核心不再依赖这些对象，实际调用与验收见下节及[解耦日志](../logs/log-2026-10-03-periodic-message-modules.md)。

| 位置 | 事实及对模块边界的影响 |
|---|---|
| [`Transformer._encode_atoms`](../../model/transformer.py)，433 行 | ZP 的 `Embedding → LayerNorm → Linear` 不读几何；方法仍在大 `Transformer` 类里，参数保留顶层名称以维持历史 checkpoint 键 |
| [`Transformer.forward`](../../model/transformer.py)，461 行 | 拆哨兵、掩码、宏观几何、图构建、Encoder 调用和读出等由同一类编排；接入局部模块需限定在编排处，不能让它读取该类内部字段 |
| [`ProbeGeometry` / `PeriodicMessageProbe` / `ProbeZP`](../../archive/code/tools/eval/periodic_message_prototypes.py) | 诊断入口在一个文件里组合了几何准备、局部网络和随机元素入口；局部 forward 已接收原子特征而非原子编号，但包装器只允许 CPU |
| [`periodic_edge_vectors`](../../archive/code/model/periodic_manybody.py)，25 行 | 几何工具放在历史多体 Encoder 文件内；新几何组件不应因取位移而依赖旧 Encoder 实现 |
| [`HighOrderMessageProbe`](../../archive/code/tools/eval/periodic_message_high_order.py)，98 行 | 为受控初值比较而要求一个原版 B 对象并复制权重；这种实验构造协议应与独立训练组件的构造分开 |
| [`TransformerEncoderLayer.rp_proj`](../../model/transformer.py)，795 行 | 当前方向分量进入任意线性投影打分；局部输出不变并不能替整条路径提供 E(3) 保证 |
| [`validate_zproj_config`](../../tools/train.py)、[`prepare_reference`](../../utils/zproj_alignment.py) | 训练入口绑定历史 ZP100/Z100 配方、种子指纹和整模型初始化；新模块不能直接沿用该历史实验协议 |

## 已实施的局部职责与后续边界

```text
原子编号 → ZP 初始化 → h0 ──────────→ 局部消息 A/B+ → h_local → 后续全局 Encoder → decoder/head
晶胞、坐标、padding_mask → 周期几何记录 ──↗
```

图示说明接口依赖，**不固定最终局部/全局的顺序、深度或交替方式**。
局部消息在科学职责上属于 Encoder 的结构更新部分；实现上应是可独立调用、测试和替换的子组件。
用户所指的“后面 Encoder”，在这份边界中是后续的全局上下文混合，不应反过来成为局部网络的依赖。

| 组件 | 接收与返回 | 自己负责 | 边界 |
|---|---|---|---|
| ZP 原子初始化 | `Z → h0[B,L,D]` | 元素身份的标量特征，保持既定公式 | 不构建邻居或做几何更新；本小单元不重命名现有参数 |
| 周期几何构建 | `结构, padding_mask, radius_A → records` | 六字段、真实位移/距离、完整镜像和输入校验 | 不读取元素 embedding、Encoder 参数、谱标签或实验配置 |
| 局部结构消息 | `h, records, padding_mask → h_local[B,L,D]` | 径向/角度编码、消息与状态更新、计数入口、标量读出 | 不包含 ZP、全局注意力、decoder/head 或损失；不查看上下游对象内部属性 |
| 后续全局 Encoder | `h_local, padding_mask → memory` | 全局信息交换 | 不依赖 A/B+ 的内部通道、球谐阶数或张量积实现 |
| 编排/适配代码 | 调用上述公开接口 | 对齐原子槽、device/dtype、配置和模型参数注册 | 保持薄层，不重新实现邻居、消息公式或元素入口 |

已实现的局部接口为：

```python
h_local = local_message(h, records, padding_mask)
```

2026-10-06 增加向后兼容的可选 `prepared_geometry`：
`prepare_shared_message_geometry(records, route)` 生成结构量，同次前向的局部调用可共享。
消息模块核对物理记录、路线与张量版本，再重新计算所有可学习消息；后续全局模块不接收此对象。
固定结构的进程内 GPU 缓存位于组装调用链，按当前 batch 重组记录及其特征，不改变上述基本调用方式。
坐标求导时仅共享本次可导几何，不使用持久缓存；失效、预算与验收见
[几何缓存日志](../logs/log-2026-10-06-local-geometry-cache.md)。

- 当前 `D=512`；`PeriodicLocalMessage("A")` / `PeriodicLocalMessage("B+")` 使用相同外部
  shape 和语义，内部 `512↔64` 投影属于局部模块。构造器可显式指定外部 `d_model`，
  内部通道和两块深度保留已验收配方；本轮数值迁移验收使用 512 维。
  后续修改内部通道不应修改 ZP 或后续 Encoder；这不是旧原型接口的原地替换。
- `records` 使用[既定六字段](design-2026-10-03-periodic-neighbor-records.md)，明确中心/邻居索引、整数镜像、
  中心指向邻居镜像的 `r[E,3]` 和 `d[E]`（Å）。它携带显式半径，不让局部块另取一个隐式默认半径。
  原子槽与掩码由调用者对齐，`True` 表示 padding；空记录、跨胞自身、多镜像规则保持不变。
- 几何构建输出物理记录；RBF、平滑权重及球谐的具体使用由局部实现或其内部共用特征器负责。
  记录契约不绑定 16 个 RBF、某个球谐阶数或某个下游网络。
- 公开输出为反射偶不变标量，方向状态在模块内部保留。可提供明确的诊断输出，但后续 Encoder
  不以它作为私有接口。后续如需跨模块传递等变状态，应另立契约。
- 构造组件只接收本组件的配置；B+ 不要求先创建一个 B 模型。研究中的共享初值、种子、
  权重复制和哈希记录属于实验构造工具，不进入组件的 forward。
- 核心几何/消息实现不导入 `tools/eval`、训练 runner 或旧完整 Encoder；诊断工具调用核心。
  几何向量工具已由几何工具层提供；旧定义暂保留冻结，而不是修改旧证据所指源码。

**独立不等于冻结或截断梯度。**ZP 的可学习参数仍按项目约定训练，`h0 → h_local → memory`
的反向链应连通，不在接口处 `detach`。参数各归所属模块，不能偷偷共享同一个 Parameter；
编排层必须将可训练组件注册进完整模型，使优化器能够收集其参数。

## 本轮提取的文件与调用方式

| 文件 | 实施职责 |
|---|---|
| [`utils/periodic_geometry.py`](../../utils/periodic_geometry.py) | `PeriodicNeighborRecords` 六个物理字段，半径与原子槽掩码元数据；校验、完整枚举适配、真实位移和显式 device/dtype 转换 |
| [`model/periodic_messages.py`](../../model/periodic_messages.py) | `PeriodicLocalMessage` 共同接口；RBF/权重/计数的内部准备，A 的无序邻边对及 B+ 的等变更新和三/四阶统计 |
| [`model/__init__.py`](../../model/__init__.py) | 按需解析原有 `basemodel` / `Transformer` 名称，避免导入局部核心时加载完整模型与训练工具 |
| [`periodic_message_module_check.py`](../../tools/eval/periodic_message_module_check.py) | 实验层复制冻结无训练权重，核对固定真实结构、回归和原证据哈希；不作为新训练初始化协议 |
| [`test_periodic_message_modules.py`](../../tests/model/test_periodic_messages.py) | 独立导入、数值/梯度迁移、掩码与空邻域、参数归属和最小 ZP→局部→下游容器验收 |

```python
from utils.periodic_geometry import build_periodic_records
from model.periodic_messages import PeriodicLocalMessage

records = build_periodic_records(pos, padding_mask, radius=6.0)
local_message = PeriodicLocalMessage("B+").to(device=h.device, dtype=h.dtype)
records = records.to(device=h.device, dtype=h.dtype)
h_local = local_message(h, records, padding_mask.to(h.device))
```

示例中的 `h` 是调用者提供的 ZP 或当前原子标量状态；不需要把 ZP/Encoder 对象传给局部模块。
模块须注册在组装模型上，如 `self.local_message = ...`，而不是每次 forward 重新构造。
几何构建可以在调用者指定的 device 上执行；核心和记录支持显式 `.to`。提取阶段已验收 CPU
FP32/FP64；后续[GPU 局部验收](../logs/log-2026-10-03-periodic-message-gpu.md)也已通过，
包括 V100 上两 dtype 的数值/梯度、固定 batch 1/4/8 和较密单结构的 FP32 成本。
AMP、完整训练 batch 与整轮吞吐另验收。提取阶段未改变数据格式与完整 Transformer 调用链；
随后完整调用链的新增分支见接入设计与日志，不修改本页的独立局部公式。

记录只含 `batch/dst/src/shifts/vectors/distances` 六个物理字段和 `radius/padding_mask` 元数据，
不含 RBF、球谐或可学习参数。`from_edges` 可接收任意正确物理框架的向量；它校验对齐、
距离、严格半径、索引和填充，不独立证明调用者提供记录的完整性或去重。
`build_periodic_records` 复用既有完整镜像枚举器，在入口拒绝非有限半径/有效坐标等异常。
向量方向、镜像语义和距离不因新包装改变；旧公共枚举器的其他调用方没有自动获得新防护。

RBF、平滑权重、计数和同中心分组在局部 forward 内派生；B+ 直接构造两块及 16 个额外输入，
不要求 B 实例。诊断层显式 `load_state_dict` 的严格迁移只用于证明公式没有变化，
默认构造的随机初值不承诺重演旧实验的 seed 协议；完整对照的共同初值协议见接入设计。
`return_debug=True` 仅用于核查状态和计数；下游正式接口仅使用同形状的不变标量输出。

有效原子的空邻域仍经过自身标量更新，不保证返回原输入；填充输出严格为零。
接口不 `detach` 原子特征或调用者提供的连续几何。既有枚举器的边选择与距离返回使用
`no_grad`，本轮不增加通过邻居集合变化或全部位置字段求导的承诺。

不能直接把当前 CPU 诊断包装器导入生产模型，也不能仅把局部块挂在现有方向打分路径之前
就宣称整个模型满足 E(3)。随后的新路径通过接入独立全局块处理该旁路，历史 Encoder 保留。
本页没有批准迁移 checkpoint、修改历史初始化指纹、改变数据池或运行新训练。

## 验收与进入训练的条件

**独立模块验收条件（实际结果见解耦日志）：**

- 无需 ZP 或完整 Encoder 对象即可用张量独立构造和调用局部块；参数归属清楚，无训练脚本反向依赖。
- 更换 A/B+ 只更换局部实例，ZP 和下游测试组件源码、输入/输出接口不变。
- 固定初值时，核心与原诊断原型的 CPU 前向及必要参数/输入梯度在既定容差内一致。
- 距离/位移、镜像、计数、掩码、全 padding、空邻域、两 dtype 的既有科学契约保持。
- 从标量读出反向到局部参数和 ZP 参数都有有限梯度，输入不被原地污染，参数未漏注册。
  该检查可无 optimizer 执行，不等于训练收益验收。

**随后独立的小单元：**

1. **已完成：**独立 CPU/GPU FP32/FP64 迁移、局部对称性/梯度与固定小批次内存/前向反向成本。
   本次成本不确定实际训练 batch 上限，AMP 与正式模型成本仍另验收。
2. **完整接入已实施：**新路径退出 `rp_proj`，追踪两类谱、尺度与后处理；全链论证和有限
   CPU/GPU 输出检查见接入日志。原子重排、晶胞表达及超胞按各自契约另报。
3. **共同初值与四组对照代码已实施：**不套用历史整模型 seed 指纹。实际训练预算、深度、
   valid 选点方式及采用门槛待决定；模型选择只用 train/valid，按既定 ZP100/B7 参考报告收益。

这些条件完成后，才有可解释、可归因的实际训练对照。内聚性保证研究中的模块可替换，
不要求每个模块单独训练，也不要求本轮同时完成全局 Encoder 的全部解耦。
