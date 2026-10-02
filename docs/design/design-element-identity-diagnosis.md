# 纯原子序号模型的训练差距诊断

日期：2026-10-02。状态：**已实施并完成 A/B/C 三阶段诊断**。本页保留当时的执行协议。
正式产物为 [`results/eid_diagnosis_s42/20261002T095909Z/`](../../results/eid_diagnosis_s42/20261002T095909Z/report.md)，
包含两项 valid 指标交叉核对未过原容差的限制；原报告的部分因果解释和下一步建议尚未确证。
经复核的结论边界与当前 ZP 决策见[元素初始化说明](design-element-initialization.md)。
没有执行原报告建议的交换初始化训练。

## 1. 目标与范围

目标是调查 B100、Z100、ZP100 的差距从哪里产生，为能独立初始化、从零训练的纯原子序号 baseline 提供依据。
本轮使用已有日志、配置、checkpoint 和 Q1 train/valid；重建未训练模型用于快照比较，不启动训练。

需要回答：

1. 纯 Z 模型在相同训练样本上也拟合得更差，还是差距主要出现在 valid？
2. embedding、归一化、投影和 encoder 是否参与计算与优化，参数及梯度有什么差异？
3. 入口表示差异是否延续到注意力和后续原子状态？
4. 哪些解释受到支持，哪些仍需要新的因果对照？

**不预设常数分支、共享向量或更多元素性质是答案。** 偏移大小、梯度大小、注意力差异均是诊断线索，不能单独证明根因。

### 执行边界

- 允许新增独立诊断工具、诊断结果和报告；保留已有工作区改动。
- 不修改模型、训练入口、损失实现、数据、配置、性质表、既有实验文档与产物。
- 不调用 `optimizer.step()`、`train_one_step()`、训练入口或学习率调度更新；不续训、不补 seed。
- 不构建或读取 test 数据；不使用已有 test 结果。不得直接运行默认读取 test 的旧诊断工具。
- 不折叠旧权重生成新候选，不注入 B100 常数，不在本轮提出后立即实施新架构。
- A100 保留为本轮既有性能对照；本轮以 B100 为诊断参照，不重新登记 baseline，也不混入旧 B7 门槛。

背景：[元素身份对照](design-element-identity-control.md)、[Z100 设计](design-element-identity-zonly.md)、
[ZP100 设计](design-element-identity-zonlyproj.md)、[ZP100 运行记录](../logs/log-2026-10-02-eid-zonlyproj.md)。

## 2. 已有事实与尚不能下的结论

以下依据来自本轮核查的代码、history、manifest 和 checkpoint；执行时重新核对，不把文档当作实际检查结果。

| 项目 | B100 | Z100 | ZP100 |
|---|---|---|---|
| 元素模式 | `legacy3_const` | `z_only` | `z_only_proj` |
| 输出目录 | `output/ablation_m1_eidconst100_s42/` | `output/ablation_m1_eidzonly100_s42/` | `output/ablation_m1_eidzproj100_s42/` |
| manifest | `results/eidconst100_s42_manifest.json` | `results/eidzonly100_s42_manifest.json` | `results/eidzproj100_s42_manifest.json` |
| history | `results/history_m1_eidconst100_s42.csv` | `results/history_m1_eidzonly100_s42.csv` | `results/history_m1_eidzproj100_s42.csv` |
| best epoch | 87 | 82 | 89 |
| latest epoch | 100 | 100 | 100 |
| 第 100 轮训练损失 | 0.169239 | 0.184485 | 0.184557 |

- 三者均为 M1、seed 42、100 轮、SumNorm KL/W1/Huber、H1 eta/gamma、FP32、隐藏维度 512、dropout 0.05。
- 已保存 best/latest 两个 checkpoint；没有完整逐轮参数或历史梯度。逐轮 history 记录训练模式下的批次平均损失。
- 第 100 轮已有 AdamW 状态均记录 58,500 次更新；有效参数为 `betas=(0.9,0.99)`、`eps=1e-8`、
  `weight_decay=0.01`、末轮 LR 为 `1e-6`，`grad_clip=0`。这些状态不证明每个部件学得有效。
- B100 与 Z100/ZP100 的共享初值不同；ZP100 与 Z100 的共享初值及训练随机状态对齐，新增投影为单位矩阵/零偏置。
- ZP100 对 Z100 的四项 valid 中位 R² 差值区间均含 0。这不是等价性证明，也不排除其他初始化方案。
- encoder 当前直接使用 `q=k=v=src`，几何进入注意力打分。ZP100 的投影及 LayerNorm 都已有偏置参数。

损失较高尚不能定位为容量不足、优化不足或泛化问题；单 seed 也不能证明 B100 的优势跨初始化稳定。

## 3. 实现入口与复用规则

建议只新增 `tools/eval/element_identity_diagnosis.py`，不修改现有生产路径。

| 用途 | 已有实现 |
|---|---|
| 模型及入口、encoder | `model/transformer.py::Transformer`、`TransformerEncoderLayer` |
| 数据和批次契约 | `datasets/dataset.py::Dos_Dataset` |
| 纯谱形损失 | `model/losses.py::sumnorm_klw_loss` |
| eta/gamma 监督口径 | `model/model.py::basemodel.train_one_step` 的损失计算部分，仅阅读与复现公式，不调用函数 |
| 物理空间逐材料指标 | `utils/metrics.py::per_sample_spectral_metrics` |
| B100 加载与 oracle/blind 公式、bootstrap | `tools/eval/element_identity_valid_verdict.py::load_arm/predict/paired_comparison` |
| Z100 加载 | `tools/eval/eid_zonly_valid_verdict.py::load_z100` |
| ZP100 加载 | `tools/eval/eid_zproj_valid_verdict.py::load_zp100` |
| 初值指纹及恒等投影 | `utils/zproj_alignment.py::state_hash/identity_projection_` |

只复用核查过的纯函数或加载函数，不运行旧工具的 `main()`；旧工具可能写旧产物或计算其他历史臂。
在诊断工具中封装无更新的批次前向/损失计算，复用已有损失和指标，避免复制整个训练流程。

加载前核对模式、seed、epoch、配置与 checkpoint SHA-256，以及 split、数据 manifest 和 B100 常数表指纹。
配置使用各臂实际 `config_used.yaml`，不能重新读当前默认模板代替历史有效配置。
当前源码和诊断脚本也记录 SHA-256；源代码不匹配历史实现且不能确认兼容时，先报告问题，不继续归因。

## 4. 固定数据与样本

Q1：train 18,706，valid 2,313。只构建 `Dos_Dataset` 的这两个 split；使用原 SumNorm、原生产网格和尺度口径，
关闭增强、log 变换和重新归一化。所有快照使用同一批样本、同一顺序。

在任何模型预测前冻结样本：

```python
train_indices = np.sort(
    np.random.default_rng(20261002).choice(18706, size=2048, replace=False)
)
probe_indices = np.random.default_rng(20261003).choice(
    train_indices, size=128, replace=False
)
```

- 训练集评估使用这 2,048 条；valid 使用完整 2,313 条。
- 梯度和激活探针使用固定 128 条 train 样本，不按模型表现或失败样本替换探针。
- 保存数组下标及对应 `train/train_index.npy`、`valid/valid_index.npy` 的材料 ID；核对数量、唯一性和对应顺序。
- 从完整 `Dos_Dataset` 使用 `torch.utils.data.Subset`，不要使用其 `choice` 参数截取数据：现有实现未同步截取全部
  尺度统计、mask 与价电子元数据，可能错配标签和统计量。
- 标准评估 batch size 为 32；探针为 8，均 `shuffle=False`、`num_workers=0`、单进程、FP32。
  若显存不足，仅允许拆小 batch，保留样本及顺序并记录变更；valid 数值复核仍需通过。
- 按实际批次 `src[:,2:]` 的非 padding 原子数统计。不要把两个哨兵或 padding 计为原子。

## 5. 三个执行阶段

### 阶段 A：同口径拟合与泛化比较

先完成三臂各自 latest=epoch100 的比较，再补 best；共六个 checkpoint。
所有模型 `eval()`，前向使用 `inference_mode()`，关闭 dropout。一次前向同时计算以下量：

1. 每材料 `L_e`、`L_p`、`L_eta` 和 `L_total`；按材料汇总均值、中位数、p90。
2. eDOS/phDOS × oracle/blind 四项物理空间 MAE、中位 R²、失败计数/率。
3. gamma、eta 绝对误差和预测/真值；仅作为尺度路径诊断。

模型构建和权重加载放在普通上下文；`inference_mode()` 仅包围评估前向，不能让后续梯度探针使用不可求导的 inference tensor。

`L_e/L_p` 复用 `sumnorm_klw_loss`；参数从历史模型配置读取，缺省值按 `basemodel.__init__` 解析并记录。
当前配方为 `use_mask=False`、`w_w1=1`、`w_huber=1`、`huber_delta=0.02`、`lambda_ph=1`、`eta_sup_w=1`：

```text
eta_true   = clamp(S_ph * 19.6875 / (3 * N_atoms), 0, 1)
gamma_true = clamp(S_e  * 0.09375 / N_valence, 0, 1)
L_eta     = 两个尺度通道平方误差的均值，保留原训练公式的有限值处理
L_total   = L_e + lambda_ph * L_p + eta_sup_w * L_eta
```

SumNorm 数据集的 max 槽为原始谱的窗口总和 `S_e/S_ph`，不是归一化后谱的最大 bin。
尺度头输出顺序是 `[eta_ph, gamma_e]`。缺失或非法必需元数据应报告，不能静默改监督公式。

oracle/blind 预测和反归一化严格复用现有判读公式；失败定义为单材料物理谱 `R²<0`，不能重定义为阈值误差。
valid 的六组逐样本 R²须与 `results/eid_zproj_s42_valid_samples.csv` 中对应 best/epoch100 列交叉核对：
同 ID 顺序，`rtol=1e-5, atol=1e-6`；不匹配先检查口径，不能放宽容差让检查通过。

对每个 split、checkpoint 口径分别计算 Z100−B100、ZP100−B100、ZP100−Z100 的配对差异。
每组固定 2,000 次样本 bootstrap，R²/失败率复用已有实现；损失均值差按同材料配对重采样。
bootstrap 独立使用 `np.random.default_rng(20261004)`，固定循环顺序为 train/valid、latest/best、上述三组比较、
eDOS oracle/eDOS blind/phDOS oracle/phDOS blind，损失均值另按 `L_e/L_p/L_eta/L_total` 顺序。

train/valid 分布不同，重点比较各臂相对 B100 的差值，不能把跨 split 的原始数值之差直接判为过拟合。
训练日志含 dropout 且参数在 epoch 内变化，与这里的固定模型 eval 损失不同，不要求两者数值一致。

同时报告 valid 的失败翻转：B100 成功而纯 Z 失败、反向翻转及两者都失败；附原子数和 train 元素频次分层，
不据此重选探针或 checkpoint。已有 history 绘制损失分量和 valid 曲线；`alpha_e/alpha_p` 是可选门控记录，
不能误读成当前任务权重。

阶段 A 完成后同步结果与优先解释，再进入 B/C；即使差距不明确，也如实完成其描述性核查。

### 阶段 B：初值、参数与当前梯度

重建三臂 seed-42 未训练模型，只在内存中保留；不是新训练或权重迁移候选。

- B100：从其有效 transformer 配置独立构造，初始指纹应等于既有兼容性记录
  `efca7088573a371fefe589a144ea66694273192498c9e7c15f2d111b717e06e0`。
- Z100：从其有效配置独立构造，指纹应等于 manifest 的
  `4eda5029e186d2f3ef60adeb0616b0a170ee3ca8fad80bad6e3e46cd9735ca32`。
- ZP100：按其实际协议复制 Z100 未训练共享参数、设置投影单位矩阵/零偏置；初始指纹应等于 manifest 的
  `4d7ae7aa41e283b1c95acd55682ed0551dbc5820031b90f99dd339a33b38e1d2`。
- 禁止用普通 `Transformer(z_only_proj)` 的默认随机初值代替 ZP100 的实际初值。
  可复用指纹、参数复制和恒等初始化辅助；不得运行会覆盖原 manifest/对齐记录的前置检查命令。

比较 init→best、init→latest 的参数绝对变化 RMS、相对范数变化，并记录参数数量/权重 RMS。
初始范数为零时，相对变化写 `null` 并注明原因，保留绝对变化；净变化不等于历史每一步更新。

按模块归组：`tok_emb`、`atom_norm`、`atom_proj`、B 的数值分支/`num_norm`/`fuse_proj`、各层 encoder
的 `rp_proj` 与其他参数、decoder、谱头、尺度头；不存在的模块标记 N/A，不当作零梯度。
embedding 中 train 出现的真实 Z、padding/哨兵、未出现行分开统计，避免把权重衰减误当成元素学习。
真实 Z 从原子槽 `elements[:,2:]` 获取；哨兵按槽位识别，不把 Z=1/2 等真实元素行当成哨兵。

核对每个参数的 `requires_grad`、优化器参数组归属、Adam step/一阶与二阶统计。
通过构造模型的 `named_parameters()` 原始顺序与 checkpoint 的参数组 ID 建立映射，核对数量和状态形状；
不能按排序后的 `state_dict` 猜测映射。优化器状态只读，不执行恢复后的 step。

在 init/latest 六个快照、相同 128 条探针上，保持 `eval()` 并使用 `torch.enable_grad()`，逐批计算：

- `L_e`、加权 `lambda_ph*L_p`、加权 `eta_sup_w*L_eta` 各自梯度；可用 `autograd.grad`，不执行参数更新。
- 各模块梯度范数及 RMS、None/零/非有限梯度数量；区分梯度大小与参数实际净变化。
- 共同模块上 eDOS/phDOS 梯度余弦；记录每批值及负余弦批次比例，零范数时记 N/A。
  梯度冲突只是一条线索，不单凭负余弦认定任务干扰是根因。

清理图和梯度；当前 eval 梯度不代表历史训练梯度或 dropout 下的梯度。单凭偏置小不能判定模块没训练。

### 阶段 C：表示与注意力的实际行为

使用相同 128 条 train 探针，检查 init/best/latest 九个快照。
仅使用不改变返回值的临时 hook 或独立重算，完成后移除 hook；不得修改正式 forward。

**表示。** 报告入口及各 encoder 层输出的均值、RMS、材料内原子间方差和余弦相似性。
只统计真实原子；分开统计同元素/异元素对、单元素/多元素材料，并注明材料数、原子数和配对数。
同元素入口相同是设计事实，不能误诊为 embedding 坍缩；通道旋转使坐标均值本身不适合单独跨模型判优。

**共享项。** 若拆解入口共享偏移，必须包含 `atom_norm.bias`，不能只比较投影 bias：

```text
Z100:  b_shared = beta_atom
ZP100: b_shared = W_proj * beta_atom + b_proj
B100:  b_shared = W_Z * beta_atom + W_c * LN_num(A*c + b_num) + b_fuse
```

同时记录真正的入口激活均值和方差；这个参数分解不是因果证明。B100 数值分支常数仅在分析其原行为时读取，
不用于初始化、注入或构造任何候选。

**注意力。** 至少检查第一层和最后一层，记录各 head 的分布。按实际实现计算：

```text
S_feature(i,j) = q_i · k_j / sqrt(64)
S_geometry(i,j) = q_i · rp_proj(rp_base_ij)
S_total = S_feature + S_geometry
```

不额外缩放几何项。统计排除 padding query/key；每个 query 先减去有效 key 均值，再比较两个打分项的标准差及比值。
softmax 对行公共偏移不敏感，未中心化的 RMS 可能把不起作用的公共项当成主导信号。
分母近零时记录标志及原值，比例记 N/A，不能用任意 epsilon 放大后断言某项主导。

使用正式 mask 重算注意力，报告最大注意力权重与归一化熵 `H/log(N_keys)`；单有效 key 时熵比例记 N/A。
结合各层原子状态的差异变化判读：几何打分大或注意力变化，不等于几何改变了最终表示；相同 V 的加权平均
可能保持不变。不得单凭注意力熵、共享偏移大小或余弦相似性决定新架构。

## 6. 判读与下一步决策

报告必须把事实、推断、未检验假设分开，每个解释列出支持、反证和待补证据。

| 观察 | 优先追查 | 仍不能直接推出 |
|---|---|---|
| 相同 train 样本的损失/R²也落后，valid 同方向 | 拟合和优化；结合参数变化及梯度检查 | 必须扩大容量或增加常数分支 |
| train 接近而 valid 落后 | 泛化、失败尾部、数据支持；保留抽样不确定性 | 一定过拟合 |
| 参数未进入优化器、梯度缺失或状态映射异常 | 先核实训练/加载契约，提出修复范围 | 直接改架构可解决 |
| 同初值 ZP/Z 差异小，但与 B 有差异 | B 的整体初值及训练参数化仍是混合解释 | 某个入口偏移是必要因素 |
| 入口或注意力差异未延续到原子状态/性能 | encoder 更新路径或其他模块可能限制收益 | 几何或元素信息永远无用 |
| best/latest 结论改变，谱形和尺度/MAE指标不同步 | 训练阶段与原选点规则 | 重新挑 checkpoint 后旧候选就达标 |

置信区间仅为固定 checkpoint 的样本重采样，不含 seed 波动和选点不确定性；区间跨零不证明等价。
训练子集、当前梯度和多项描述性统计也不能替代因果实验。已有数据不足时，允许结论为“尚不能定位”。

最终最多推荐 **一个优先的最小验证对照**，说明它区分哪两个解释、需要控制哪些变量、预计成本及判据。
这只是下一步建议，本轮不实施、不训练、不调整既有 baseline 门槛。

## 7. 产物、检查与运行节奏

全部新结果放在独立目录 `results/eid_diagnosis_s42/<run_id>/`，`run_id` 使用实际 UTC 时间。
保留每次运行和中间结果；重跑只能写新目录，不能覆盖已有实验或诊断记录。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 代码版本及脏工作区、配置/数据清单/split/checkpoint 哈希、初始指纹、样本索引/ID、环境、实际 batch 与资源、各阶段完成状态 |
| `samples.csv`、`metrics.json` | 三臂、两类训练后快照、两个 split 的逐材料损失/指标及配对统计；保留 arm/snapshot/split/sample_index/mpid |
| `parameter_changes.csv`、`optimizer_state.csv` | 参数变化及经过核对的优化器统计 |
| `gradients.csv` | init/latest、模块、损失项、探针批次的梯度与冲突统计 |
| `activations.csv`、`attention.csv` | 快照、层/head、材料分类及表示/注意力统计 |
| `report.md`、`run.log` | 通俗结论、证据与局限、下一最小对照建议、实际执行记录；图表可放同目录 |

必要核对：

1. 输入身份/哈希、初始指纹、样本 ID 和 valid 指标复核全部有实际结果。
2. 各阶段前后模型 `state_dict` 指纹相同；无参数更新。原配置、manifest、checkpoint、history 和旧结果文件未改变。
3. 损失、指标、梯度和激活有限；本来不适用的量保存 `null`/N/A 和原因，不写 JSON NaN/Inf。
4. 注意力排除 padding，hook 不改变 forward 输出，重算使用原 mask；阶段 C 的重复前向与正常前向在
   `rtol=1e-5, atol=1e-6` 内一致。诊断异常先修复新诊断工具，不能改正式模型让核对通过。
5. 三个阶段均有结果或明确的阻塞/缺失说明；不存在 test 计算、训练、候选模型变更或旧产物覆盖。

单次只在 GPU 放一个模型，CPU 加载其余快照；释放前向图和 hook 缓存，不保存大型全量注意力张量。
粗略预计计算在 **0.5–1 GPU 小时**量级，尚未实测，实施时间另计；先计时一个固定批次并外推，记录实际耗时和显存。
不要抢占其他运行或杀进程。若需要扩大计算范围，先报告新增成本，由用户决定。

同步节点：实现完成与实际预算、阶段 A 完成、B/C 完成、最终报告；运行中不超过 60 秒无进度沟通。
遇身份错位、数值异常或计算中断，保留产物和日志，先定位诊断工具问题；需修改生产代码或扩大实验时暂停并报告。
完成报告后停止，不自动启动下一实验。
