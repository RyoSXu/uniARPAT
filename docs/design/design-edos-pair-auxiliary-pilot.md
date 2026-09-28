# 设计：完整谱锚定的 eDOS 同组成谱差辅助 Pilot

状态：**Stage B 已完成并 park；未读取 test，不进入 M1×35。**路线证据见
`../logs/log-2026-09-28-next-step-gate-review.md`，实现与资源证据见
`../logs/log-2026-09-28-edos-pair-aux-stage-a.md`，正式结果见
`../logs/log-2026-09-28-edos-pair-aux-pilot.md`。本设计只定义一个可证伪候选，不把外部方法名、
训练内响应或子集改善当作采用理由。

## 项目决策

判断在保留 B7 完整双谱主损失与 H1 的前提下，增加 eDOS 同组成谱差辅助监督，能否把已测的条件
响应不足转化为完整 Q1 valid 的 blind eDOS 收益，同时保护 oracle eDOS、phDOS 和单谱形质量。

成功才提议 M1×35 长预算确认；任何非成功结果都 park。设计不服务于定位某一层，也不允许结果引出
新的训练集诊断。

## 单一因素与两臂

共同配置：B7 epoch 33 同一 checkpoint warm-start、Q1、M1、10 epoch、seed 42、batch 32、FP32、
非分桶、SumNorm KL/W1/Huber、H1 eta/gamma、dropout 0.05、相同 optimizer steps、相同 checkpoint
选择规则，载入后重置 RNG；跳过 test loader 和 test inference。

| 臂 | 主 batch | 辅助 pair 前向 | pair loss 梯度 |
|---|---|---|---|
| `_pcctl` | 原 Q1 train loader | 执行 | 乘零，不更新 |
| `_pcaux` | 与 control 同序 | 同一 pair、同一调用时点 | 按冻结权重加入 |

两臂都做相同的辅助前向以对齐耗时、显存和 dropout RNG 消耗。唯一实验因素是 pair loss 梯度。
不新增模型头、参数、encoder 输入或推理输出；默认 `pair_ratio=0` 时不建立辅助 loader，保持原行为。

## 冻结 pair 计划

- 只用 Q1 train 的 `elements_train.npy` 和 `train_index.npy` 建组，不访问 valid/test 标签。
- pair 两端必须具有相同**绝对元素计数**；不同绝对计数组不直接配对。统计与轮换按约化组成组等权。
- 现有 2,591 个合法 pair、1,198 个约化组成组作为冻结宇宙。每个 epoch 每组确定性选择一个 pair；
  组内候选按 `sha256(seed, reduced_group, mpid_a, mpid_b)` 固定排序，再按 `epoch % 候选数` 循环，
  从而保证多候选组逐轮轮换；epoch 计划哈希另含 epoch。全宇宙和选择策略有独立冻结哈希；不按目标
  TV 筛选。
- 每个辅助 batch 为 16 pair／32 个材料实例，共 75 个 batch；均匀插入约 585 个主 batch，单个主 step
  最多附加一个辅助 batch。两个臂使用逐项相同计划和计划 SHA-256。
- 辅助 batch 只产生 pair loss；所有材料的完整主损失仍由原主 loader 每 epoch 恰好暴露一次，数据权重
  与主优化步数不变。

## 辅助损失与一次性校准

预测先由 eDOS logits 做 softmax 得到 `p`，目标 `y` 已为 SumNorm。对 pair `(a,b)`：

`L_pair = mean_pairs 0.5 * sum_bins |[(p_a-p_b) - (y_a-y_b)]|`。

它就是既有带符号谱差误差的 TV 形式，直接同时约束方向和幅度；不使用仅放大预测 TV 的目标。
总体损失为 `L_total = L_main + lambda_pair * L_pair`，其中 `L_main` 完整保留生产双谱与 H1 各项。

- `pair_ratio=0.10` 固定；不扫描。
- 在 B7 初值、eval 模式和冻结的首个辅助 batch 上，用现有
  `calibrate_additive_loss_weight` 令 pair loss 对全部可训练 Transformer 参数的梯度范数等于同批 eDOS
  主损失梯度范数的 10%。校准不执行 optimizer step。
- 校准输入、两项梯度范数、`lambda_pair` 和计划哈希写入配置与 checkpoint。非有限／零梯度或
  `lambda_pair` 不在 `[1e-4, 1e4]` 时记技术失败并停止，不改比例或换 batch。
- 恢复必须严格核对 `pair_ratio`、`lambda_pair` 与计划哈希；不能用不同 pair 计划续训。

## 实现边界（获批后）

实施前再次搜索复用；当前已确认可复用 `composition_keys`／配对口径、
`calibrate_additive_loss_weight`、checkpoint 原子写入和 valid 配对判决工具。

拟议调用链：

1. `utils/pair_aux_batches.py`：生成并哈希冻结 pair 计划，按 epoch 提供 75 个 pair batch；不读目标谱。
2. `model/losses.py`：新增纯函数 `edos_pair_contrast_loss`。
3. `model/model.py::basemodel`：增加默认关闭的 `pair_ratio/lambda/plan` 状态；训练 step 可选接收已处理的
   辅助 pair batch，control 也完成同一前向，但只有 candidate 加入梯度。
4. `run_ablation_experiments.py`：仅显式 flag 构造辅助 loader、调度相同调用、记录配置；禁止与 AMP、
   bucket batch、G1/G2、联合内容或其他实验 flag 组合。
5. `utils/ablation_checkpoint.py`：保存／恢复校准值和计划哈希；旧 checkpoint 在 flag 关闭时兼容。
6. `tools/eval/edos_pair_aux_verdict.py`：只读 Q1 valid，复用冻结的 117 组／163 pair 和全体 320 pair；
   不提供 test 入口。

默认关闭时不得改变 DataLoader、RNG、模型 state dict、forward 输出、训练损失或 checkpoint payload。

## 分阶段门禁

### A. 实现与前置，不训练

2026-09-28 已通过。校准 `lambda_pair=0.2593688071`，control/candidate 校准前主输出最大绝对差为 0；
16 个代表性主 step 中插入 2 个辅助 batch，时间比最大 `1.0992x`，峰值显存比最大 `0.9991x`。
详细证据及首次显存失败后的同一步顺序反传修复见对应日志。这里没有精度或泛化结论。

- CPU 合同：绝对计数组、约化组、确定性轮换、每组每 epoch 一 pair、无标签选样、计划哈希、两臂计划
  相同、pair loss 符号／置换性质、ratio=0 兼容、校准失败、checkpoint 恢复与 test 隔离。
- B7 单 batch：control/candidate 在加入 pair 梯度前的主输出逐值相同；control 的辅助前向不改变梯度。
- V100 资源门禁只跑固定少量 step，不形成精度结果：平均 step 耗时 `<=1.25x`、峰值显存
  `<=1.10x` 同配置无辅助参照；任一失败即停止，不缩减 pair 覆盖率救场。

### B. Q1 valid-only 两臂 Pilot（已完成）

正式使用 epoch 10 `checkpoint_latest.pth`，不按 valid 挑 epoch。2,313 条 valid 同序配对，2,000 次
bootstrap 只报告区间，点估计与预注册门槛共同裁决。

2026-09-28 执行结果为 `park`：blind eDOS 中位 R²只提高 `0.00159`，未达到 `+0.02` 主门槛；
117 组／163 pair 的组等权谱差误差相对下降 `0.465%`（95% bootstrap 区间
`0.104%…0.880%`），未达到 `5%` 机制门槛。谱形、共享任务和 paired-256 三项保护门均通过。
依行动表不进入 M1×35，不扫描，不追加诊断。

## 验收与行动

必须同时满足：

1. **总体主门槛：**candidate−control 的 valid blind eDOS 中位 R² `>=+0.02`，失败率 `<+1pt`。
2. **谱形保护：**oracle eDOS 中位 R² `>=0`，失败率 `<+1pt`。
3. **共享任务保护：**phDOS oracle 与 blind 中位 R² 均 `>=−0.02`，失败率均 `<+1pt`。
4. **机制门槛：**train 未见组成的 valid 117 组／163 pair，按组成组等权的谱差 TV 误差相对下降
   `>=5%`，配对 bootstrap 95% 区间下限 `>0`。
5. **配对子集保护：**上述 256 个材料的 oracle 与 blind eDOS 中位 R²均不得低于 control 超过 0.02，
   失败率不得增加 1pt 或以上。

行动表：

- 五项全部通过：记录 pilot win，只提议 M1×35 长预算确认设计；不自动训练或改默认。
- 总体平局，即使机制指标改善：park，说明辅助项改变了子集但没有解决项目主指标。
- 总体改善但机制未过：不按本假设晋级；记录意外主效应并 park，不追加定位诊断。
- 任一保护越线或总体退化：park；不扫描 ratio、pair 频率、损失形式或 sampler。
- 数值、恢复、输出错位等技术失败：不作科学结论；只允许在设计不变时修复并重新申请执行。

## 明确不处理

- 不执行逐层响应距离、冻结层、小样本拟合、InfoNCE、标签编码器、假负例过滤、角特征或新增数据。
- 不读取 test，不比较历史 test，不用训练集 pair 改善宣称泛化。
- 不把一次 10 epoch seed 42 pilot 当作最终收益；没有通过就不做 seed 扩展。
