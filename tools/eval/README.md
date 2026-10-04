# 评估工具

可复用的分析与结论判定脚本放在这里。脚本必须读取受版本控制的结果 CSV，说明输入数据口径
（Q1 或 pre-Q）和 oracle/blind 模式，并写出可复现的摘要。不得把一次性结论逻辑放在 `/tmp`。

请使用描述性名称，例如 `c5_pilot_verdict.py`。图和临时输出应保留在此目录外，除非它们本身是
有意纳入项目的产物。

## 元素身份与纯原子序号实验

`element_identity_preflight.py` / `element_identity_valid_verdict.py`：A100/B100 的实施核查、
manifest 与 Q1 valid 判读；`eid_zonly_*` 对应 Z100，`eid_zproj_*` 对应 ZP100。
其中 `*_monitor.py` 读取进程、history 与产物状态，`*_preflight.py` 会执行数值检查并写入
实验记录；历史 preflight 命令不能直接用来覆盖已有 manifest。

`element_identity_diagnosis.py`：对已有 B100/Z100/ZP100 做 train/valid 拟合、参数与优化器、
当前梯度、表示及注意力诊断，设计见 `docs/design/design-element-identity-diagnosis.md`。
正式结果目录为 `results/eid_diagnosis_s42/20261002T095909Z/`，包含追溯快照；中断目录
`20261002T093859Z/` 按用户决定保留本地并忽略。原报告的解释边界见
`docs/design/design-element-initialization.md`，当前标准为 ZP100 baseline 与 B7 最优指标参考。

## 既有结构与读出诊断

`periodic_message_gpu_check.py`：独立 A/B+ 的 GPU 数值与资源验收，沿用冻结无训练权重、固定
16 个 Q1 train/valid 结构、严格 6.0 Å 与完整镜像。FP32/FP64 比较 CPU/GPU 和 GPU 原生记录，
检查局部旋转/反射、实际平移、表达一致性、填充、输入/参数梯度和最小 ZP→局部→标量测试读出链。
FP32 固定 batch 1/4/8 与较密单结构使用同步墙钟比较前向及无更新反向；CPU 单线程，
预热一次、测量三次，结构构建/传输另列。主预算默认 720 秒、CPU 2 GiB、CUDA 分配器 8 GiB。
`confirm_periodic_message_gpu_timing.py` 补做三次预热、五次 GPU 反向测量，预算 90 秒，保留首轮峰值；
`verify_periodic_message_gpu_check.py` 必须随后运行，独立重算各项判据、覆盖、中位数与哈希，
核对 JSON/JSONL/CSV，避免只相信汇总通过标记。四项 CPU 回归位于 `test_periodic_message_gpu_check.py`。
不读取谱标签、test、checkpoint，不使用 AMP、optimizer 或更新参数；blind/oracle 不适用，拒绝覆盖。
通过只覆盖当前局部模块和有限批次，不证明完整模型 E(3)、谱准确率或端到端训练成本。
命令、966 项结果、预热补测及边界见[GPU 验收日志](../../docs/logs/log-2026-10-03-periodic-message-gpu.md)。

`periodic_message_module_check.py`：用户批准提取后的独立 A/B+ 接口验收。当前可复用核心为
`model/periodic_messages.py`，物理记录来自 `utils/periodic_geometry.py`；本工具将冻结无训练
原型的权重显式迁移到核心，核对原固定 16 个 Q1 train/valid 结构的 FP32/FP64 几何与状态，
另做小结构旋转/反射和必要回归。新核心不导入诊断、ZP 或完整 Encoder；B+ 可直接构造。
CPU 检查预算默认 180 秒、2 GiB，无谱标签/optimizer/参数更新，blind/oracle 不适用，拒绝覆盖。
旧原型/对照脚本保持冻结，用于历史复核，后续新增消息实验应调用新核心。
这项迁移不启动训练、不处理完整 Encoder，也不提供 GPU 性能或最终谱不变性证据；
完整结果见[解耦日志](../../docs/logs/log-2026-10-03-periodic-message-modules.md)。

`periodic_message_high_order.py` / `periodic_message_high_order_compare.py`：用户批准的独立 B+
增强与 A/原版 B 对照。复用八通道径向权重，以三/四阶球谐矩计算角度对不变量，原版源码与
权重不变。固定 16 个 Q1 train/valid 结构、五个无训练种子、两 dtype；严格 6.0 Å，全镜像。
主模式检查局部对称性/一致性、固定径向角度与关闭新增输入的消融、物理剪切及三轮 CPU
前向/无更新反向；`rss-A` / `rss-B` / `rss-Bplus` 在新进程测同一案例的 RSS。
`verify_periodic_message_high_order.py` 独立核对产物与旧证据哈希；
`plot_periodic_message_high_order.py` 从复核后的数据导出 PNG/SVG。
入口需已有初轮 precheck manifest；不读取谱标签、test、checkpoint，不创建 optimizer、不训练，
blind/oracle 不适用；拒绝覆盖。受控 `c=1` 高阶反例的改善与正常截断近边界仍无可分辨响应
分别报告，不能当作谱准确率、完整模型不变性或 GPU 成本证据。
完整结果见 [2026-10-03 高阶角度对照](../../docs/logs/log-2026-10-03-periodic-message-high-order.md)。

`periodic_message_prototypes.py` / `periodic_message_precheck.py`：6.0 Å 两套独立局部消息原型，
共同随机 ZP 入口，在 Q1 train/valid 固定 16 个真实结构与合成例上检查局部 E(3)、重排、
换基、超胞、距离/角度/计数入口和资源。CPU 数值计算单线程，固定权重；反向检查不更新
参数，不创建 optimizer，不读谱标签、test 或 checkpoint，不接入生产模型。
`periodic_message_controls.py` 重现原权重哈希，固定径向字段隔离角度响应，并在新进程测 RSS；
`verify_periodic_message_precheck.py` 独立复核产物、摘要、覆盖及哈希，不重运行神经网络。
无训练诊断不适用 blind/oracle 谱模式；拒绝覆盖已有产物。840 项既定检查通过不能消除当前
B 的低阶角度盲点与两者近截止弱响应，也不证明完整模型不变性或 DOS 收益。结果与范围见
[`2026-10-03 CPU 预检`](../../docs/logs/log-2026-10-03-periodic-message-precheck.md)。

`periodic_geometry_acceptance.py`：显式使用 6.0 Å，复用现有枚举和向量重建函数，对 Q1 train/valid
全量记录检查字段、反向位移、真实距离、填充及计数；在固定样本上用独立逐轴枚举核对完整性。
物理平移/旋转/反射单列为 E(3) 检查，原子重排、整数坐标镜像、换基和 2×1×1 超胞单列为
表达一致性。诊断显式维护物理框架，近截止的键集差异保留为受限证据；保留逐样本结果与哈希，
拒绝覆盖结果目录，不加载谱标签、checkpoint 或执行模型。几何证据不构成完整模型的对称性验收。
正式结果与范围见 [`2026-10-03 几何验收`](../../docs/logs/log-2026-10-03-periodic-geometry-acceptance.md)。

`periodic_neighbor_radius_audit.py`：仅加载 Q1 train/valid 的元素、位置及样本编号，显式指定
候选半径，复用现有周期枚举和计数函数，比较空邻域、前三个距离壳层的完整覆盖及 CPU 枚举
成本。壳层采用按起始距离分组的绝对容差，另报容差敏感性；稀疏结构自适应扩大参考半径，
未解析的壳层标为截尾。结果含原子与结构明细、独立半径计时、摘要及输入/源码哈希，拒绝
覆盖已有目录。不加载谱标签或执行模型，几何统计不证明 DOS 收益，也不自动采用某个半径。
`plot_periodic_neighbor_radius_audit.py` 从上述摘要生成覆盖与成本图；全量结果与取舍见
[`2026-10-03 审查日志`](../../docs/logs/log-2026-10-03-periodic-neighbor-radius-audit.md)。

`g2_structure_path_audit.py`：冻结 G2 epoch 10 两臂，仅评估 Q1 train/valid，并在内存中关闭
G2 残差，追踪 encoder／decoder／谱输出响应与误差；设计见
`docs/design/design-g2-structure-path-audit.md`。默认拒绝覆盖已有诊断结果。

`g2_frozen_readout_probe.py`：固定 G2 encoder，复用原 eDOS 读出，对比正确对应与组内随机对应的
谱差训练；在训练中未出现的约化组成上验收。设计与固定预算见
`docs/design/design-g2-frozen-readout-probe.md`；拒绝覆盖已有运行目录及结果。

`g2_encoder_adaptation_probe.py`：沿用冻结读出配对任务，只开放现有encoder更新；相同初值、
读出优化规则及1620步预算，对比已保存的冻结matched结果。blind尺度从更新后的表征重算；
设计见 `docs/design/design-g2-encoder-adaptation-probe.md`；源文件、数据与旧结果均校验哈希。

`g2_small_fit_probe.py`：标签无关抽取16个train组成对，先核对输入区分性，再比较冻结／联合
实际网络各2000步的小样本拟合；保存首次全部达到逐对门槛的权重与末步结果，只检验训练可拟合性。
设计见 `docs/design/design-g2-small-fit-probe.md`；只加载train，拒绝覆盖已有产物。
`plot_g2_small_fit.py`从完成后的正式history绘制各检查点的最差pair残留和通过数量，不改变判据。

`g2_value_fit_probe.py`：复用同一16对缓存和两臂对照，只更新6个G2消息模块与原读出；
训练前检查跨冻结层梯度，结束后逐值检查冻结边界并重载最终／首次拟合权重。
设计见 `docs/design/design-g2-value-fit-probe.md`；旧执行快照与当前helper分别登记哈希。
若2000步已完成而结果导出中断，可用`--finalize-existing`从现有最终权重汇总；先核对旧执行
快照与保护输入，不执行训练。缺失的中间逐对轨迹会明确标记，不能由末步预测补造。

`g2_non_value_fit_probe.py`：冻结G2消息参数，只训练其余encoder与读出的16对互补臂；
复用同一训练、梯度边界和重载实现，保留冻结模块的反向传播。比较包含前三个臂，并保留
已有缺失明细标记；设计见`docs/design/design-g2-non-value-fit-probe.md`。

`g2_last_layer_fit_probe.py`：固定G2与前5个普通encoder层，只训练最后一层及读出；沿用同一
16对和2000步。首次／最终权重重载时核对冻结参数及缓冲区；设计见
`docs/design/design-g2-last-layer-fit-probe.md`。
