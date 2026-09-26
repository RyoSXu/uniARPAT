# 日志：2026-09-26 — 最后一个普通encoder层的16对拟合通过

## 范围与单条结论

- 马尚酱批准固定G2和前5个普通encoder层，只训练最后一个普通encoder层与读出。
  预注册见 `docs/design/design-g2-last-layer-fit-probe.md`；沿用同一16对／32个Q1 train材料、
  原G2 ep10初值与2000步，本轮只新增`last_layer`臂。
- **从该初值出发，最后一个普通encoder层与读出的联合更新，已经足以拟合这16对谱差；
  无需继续更新前5层或G2消息参数。**第2000步首次在预定检查点达到16／16，末步权重重载复现。
  判定为`last_layer_updates_sufficient_for_small_fit`。
- 全部普通encoder开放时第700步即通过，本轮直到预算最后一个检查点才通过；只有一个
  全部通过的检查点，不能声称后续训练稳定。两臂等步数、等数据暴露，不声称等算力或精度等价。
- 第6层仍直接接收距离／方向，末尾G2仍接收周期边；因此通过只缩小了需要更新的参数范围，
  没有证明前5层完整保留了目标几何信息，也没有定位信息究竟在哪一层丢失。

## 实施地图与调用链

固定atom_src／几何输入 → 固定前5层及其G2 → 可训练第6个普通encoder层
→ 参数固定但保持反向传播的末尾G2 → 可训练原eDOS读出；当前memory经冻结H1重算blind尺度。

| 文件 | 本轮符号与职责 |
|---|---|
| `tools/eval/g2_last_layer_fit_probe.py` | 新增`main`，选择`last_layer`范围、设计及四个旧臂，调用公共运行器 |
| `tools/eval/g2_small_fit_probe.py` | `configure_encoder_scope`增加只开放`layers[-1]`；沿用`fit_arm`的目标、优化器、2000步及逐点保存 |
| `tools/eval/g2_value_fit_probe.py` | `load_reference`区分旧入口与共享helper快照；`gradient_boundary_check`支持最后层并报告实际层号；`verify_checkpoints`重载后增加冻结参数及缓冲区逐值核对；`run_probe`增加最后层边界与结果汇总，继承历史记录缺口 |
| `tools/eval/plot_g2_small_fit.py` | `plot_results`增加最后层曲线，复用正式CSV |
| `tests/test_g2_value_fit_probe.py` | `TestG2ValueFit`增加4项最后层测试，扩展范围切换检查 |
| `tools/eval/README.md` | 登记新入口；已有CI包含该测试模块 |

新增测试为`test_last_layer_gradient_scope_keeps_actual_layer_index`、
`test_detaching_final_frozen_g2_breaks_last_layer_gradient`、
`test_last_layer_training_keeps_prefix_and_messages_unchanged`及
`test_reload_rejects_a_witness_with_changed_frozen_prefix`。

编码器可更新2,397,184／17,734,662个参数（13.52%），仅`encoder.layers.5.*`；读出可更新
25,357,825个参数。最后层包含注意力Q/K/V、相对位置投影、前馈和归一化，尚未拆分这些部分。
G2仍参与前向，最后一个G2的输入与输出随第6层适配而变化。

## 运行合同与边界核验

- 标签`g2_last_layer_fit_q1`；选样seed20260926、初值seed42、FP32、eval模式、关闭dropout。
  完整16对batch重复2000步；AdamW lr5e-5、betas0.9／0.99、weight_decay0.01，读出与
  可训练encoder分别裁剪1.0。未改变选样、初始化、目标或预算。
- 目标为`mean((128*((p_a-p_b)-(q_a-q_b)))**2)`。逐对残余MSE≤零谱差MSE的1%，且
  残余TV≤目标TV的10%，要求16对同时满足；第0步及每100步检查。
- 初始最大预测TV差1.97e-7、blind尺度相对误差2.58e-7、读出梯度相对RMS2.01e-5、损失
  相对误差1.88e-7，全部低于设计门槛。第6层梯度范数0.095971，冻结参数没有梯度。
- 训练后第6层改变2,397,181个值，相对L2变化0.056330；读出改变25,357,807个值，相对L2
  变化0.028542。前5层、所有G2、其他encoder状态和H1逐值保持。
- 训练183.73秒，训练加收尾190.62秒；峰值已分配显存1.870 GiB，V100 32GB。
- 仅使用train；未评估valid、test或phDOS。沿用此前独立logits目标可行性对照，不重复训练。

```bash
setsid nohup python3 -u tools/eval/g2_last_layer_fit_probe.py > output/g2_last_layer_fit_q1.log 2>&1 < /dev/null &
python3 tools/eval/plot_g2_small_fit.py --prefix results/g2_last_layer_fit_q1
```

缓存SHA256：`98ec2b34d588ad9aec16a14e2afd3eeb7c4555d019fa50a7d37d67ae20e158ed`。
首次通过权重：`f121ae09d335a4f3d9523c4d5711f9a7d7e5b0b642be88afb2bde213b360bbd9`；
末步权重：`839b28f35686f3981f2fe093814b6d52653c664002cf809f9390df0f2560d11e`。
二者均为第2000步的同一训练状态，分别保存并重载核验，不能当成两个独立成功检查点。

## 五种更新范围的结果

所有臂均训练读出。四个旧臂直接引用既有正式结果，本轮只运行最后一行。

| encoder可更新范围 | 检查点最多通过 | 首次16／16 | 末步通过 | 末步最差pair MSE残留 | 末步最差pair TV残留 |
|---|---:|---:|---:|---:|---:|
| 无 | 8／16 | 无 | 8／16 | 73.5313% | 83.7287% |
| 全部 | 16／16 | 700步 | 16／16 | 0.1295% | 2.0992% |
| 仅G2消息 | 10／16 | 无 | 4／16 | 60.1476% | 79.3594% |
| 除G2之外的encoder | 16／16 | 700步 | 16／16 | 0.2219% | 4.3076% |
| 最后一个普通encoder层 | 16／16 | 2000步 | 16／16 | 0.5601% | 5.2246% |

残留以各pair的零谱差为参考，门槛分别1%和10%。本轮末步总体MSE残留0.0248%，总体
TV残留1.5883%；逐对门槛全部通过，最差MSE与TV均为pair12。

- 第1400～1800步均14／16，第1900步15／16；最后未通过的pair15在1900步仍有MSE／TV
  残留6.5927%／11.1218%，2000步降到0.1836%／5.1559%，使全部材料对通过。
- 先前最难的pair0（`mp-aaaaabjd`／`mp-aaaaarxw`）本轮末步MSE／TV残留为
  0.0215%／1.3988%；冻结encoder和仅G2更新的末步TV残留为83.7287%／79.3594%。
- 通过次数随训练曾回落；没有按中间均值、材料子集或改变门槛宣称成功。

### 固定32材料的eDOS单谱指标

| 臂 | oracle 中位R²／失败率 | blind 中位R²／失败率 |
|---|---:|---:|
| 原G2 ep10 | 0.567433／3.125% | 0.547878／3.125% |
| 冻结encoder | 0.245919／31.250% | 0.212284／28.125% |
| 全encoder更新 | 0.640421／0.000% | 0.501260／0.000% |
| 仅G2更新 | 0.399469／25.000% | 0.472620／28.125% |
| 冻结G2、更新其余encoder | 0.622165／0.000% | 0.522578／0.000% |
| 最后一个普通encoder层 | 0.544679／15.625% | 0.457336／21.875% |

口径：Q1 train固定32材料、各臂第2000步。纯谱差目标未锚定共有谱形；本轮oracle／blind
失败率均高于原模型，谱差成功不能替代单谱质量，更不能据此晋级生产方案。

## 验证与证据完整性

- 相关测试21／21通过：`python3 -m unittest tests.test_g2_value_fit_probe tests.test_g2_small_fit_probe`。
  `bash tools/ci/check-static.sh`为103／103通过；新文件显式Ruff、`git diff --check`通过。
- 两层实际网络测试验证冻结前缀、G2和H1不变；主动切断末尾冻结G2时，最后层梯度检查
  报错；篡改首次通过权重中的冻结前缀时，重载检查拒绝该权重。
- 首次通过／末步权重各严格重载，冻结参数与缓冲区改变数均0，各前向5次均16／16。
  最大重复TV差分别4.37e-7／4.77e-7；末步重载对原保存预测的最大TV差4.19e-7，均低于1e-5。
- 本轮71个受保护输入哈希一致；入口、两个共享helper、设计共4份执行快照字节一致。
  旧臂CSV记录逐项对照未改变；新臂21×16＝336行逐对记录完整，均为训练时记录。
- 继承仅G2臂缺失304行中间逐对明细的记录和`recomputed_endpoint`标记；新JSON的
  `reference_artifact_deviations`保留19个缺失检查点，不补造历史。
- 正式前缀`results/g2_last_layer_fit_q1`：JSON、checkpoint_verification.json、selection16行、
  history105行、pair_history1376行、samples192行、final_pairs96行、summary6行、learning.png。
  `output/g2_last_layer_fit_q1`保存执行快照、每点记录、首次通过／最终权重与最终预测。

## 下一步

- 已有一个明确反例：拟合这16对并不要求继续更新前5层或G2；最后一个普通encoder层与
  读出的联合适配足够。仍不能把限制归因到最后层的某一个组件，也不能声称仅读出永远不可能拟合。
- 按预注册的通过分支，建议下一项固定最后层注意力及相对位置投影，只开放最后层前馈／
  归一化与读出，保持相同选样、初值、目标和预算；检验几何注意力参数的继续更新是否也可省去。
  这是待确认的下一项，不在本轮自动启动。即使通过，冻结权重也不等于关闭几何作用。
- 本轮日志、status与index同步；小样本相关工作尚未提交。进入全体准确率候选之前，仍须
  恢复完整谱形约束，并检查blind与phDOS保护。
