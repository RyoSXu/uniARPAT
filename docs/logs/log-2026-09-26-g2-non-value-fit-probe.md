# 日志：2026-09-26 — 冻结 G2 消息的16对互补拟合通过

## 范围与结论

- 马尚酱批准冻结G2消息、只训练其余encoder与读出的互补对照。预注册见
  `docs/design/design-g2-non-value-fit-probe.md`；同一16对／32个Q1 train材料、原G2 ep10
  初值和2000步，仅增加`non_g2`一个训练臂。
- **单条结论：从这个已训练的G2 ep10模型出发，更新其余encoder与读出已足以拟合这16对，
  无需继续更新G2消息参数。**第700步首次16／16；700～2000步的14个预定检查点均16／16。
  首次与末步权重严格重载后，各5次前向仍全部通过。
- 判定`non_g2_updates_sufficient_for_small_fit`。这排除了本次适配必须更新G2消息参数的
  解释；没有排除G2模块本身的作用，也没有证明普通encoder的某一层丢失了信息或全体泛化改善。

## 实施地图与调用链

固定atom_src／几何输入 → 可训练普通encoder层 → 参数固定但保持反向传播的G2残差
→ 可训练原eDOS读出；当前memory经冻结H1重算blind尺度。

| 文件 | 本轮符号与职责 |
|---|---|
| `tools/eval/g2_non_value_fit_probe.py` | 新增`main`：选择`non_g2`、本轮设计和三臂旧对照，调用已有运行器 |
| `tools/eval/g2_small_fit_probe.py` | `configure_encoder_scope`增加互补范围；复用`fit_arm`的目标、优化器、步数和逐点保存 |
| `tools/eval/g2_value_fit_probe.py` | `reference_path`／`load_reference`支持对照前缀并核对旧执行版本；`gradient_boundary_check`／`verify_checkpoints`／`run_probe`显式接收实验臂，复用同一梯度、冻结、重载及汇总规则 |
| `tools/eval/plot_g2_small_fit.py` | `plot_results`增加冻结G2的曲线 |
| `tests/test_g2_value_fit_probe.py` | `TestG2ValueFit`增加3项互补臂测试，扩展范围切换检查 |
| `tools/eval/README.md` | 补充互补入口说明；现有CI自动包含该测试模块 |

本轮普通encoder可更新14,383,104／17,734,662个参数（81.10%），包括普通注意力的Q/K/V
投影、相对位置投影、前馈和归一化；读出25,357,825个参数继续训练。G2的3,351,558个参数
冻结。G2输入来自变化中的普通encoder，因此其输出仍会改变；这不是关闭G2的消融。

## 运行合同与检查

- 标签`g2_non_value_fit_q1`，选择seed20260926、初始化seed42，FP32、eval模式、关闭dropout。
  同一完整16对batch重复2000步。AdamW lr5e-5、betas0.9／0.99、weight_decay0.01；读出与
  可训练encoder分别裁剪1.0。与旧臂等更新步数、等数据暴露，不声称等算力。
- 纯谱差MSE目标不变；每对残余MSE≤零谱差MSE的1%，残余TV≤目标TV的10%，16对同时
  通过。每100步记录一次，保存首次通过及末步权重；不按valid选择。仅train，未评估valid、
  test或phDOS，共有谱形无锚定。
- 初始预测最大TV差1.90e-7、blind尺度相对误差2.58e-7；读出梯度相对RMS2.05e-5、损失
  相对误差0，均通过预定等价门槛。6个普通encoder层梯度范数0.07296～0.09597，均有限非零；
  G2及H1没有梯度，检查后清零再训练。
- 训练后普通encoder改变14,383,088个值（相对L2变化0.022273），读出改变25,357,787个值
  （0.020750）；G2与H1状态逐值不变。源文件、数据、缓存及旧结果哈希保护检查通过。
- 训练循环220.80秒，训练及收尾227.34秒，峰值已分配显存2.683 GiB，V100 32GB。

```bash
setsid nohup python3 -u tools/eval/g2_non_value_fit_probe.py > output/g2_non_value_fit_q1.log 2>&1 < /dev/null &
python3 tools/eval/plot_g2_small_fit.py --prefix results/g2_non_value_fit_q1
```

缓存SHA256：`98ec2b34d588ad9aec16a14e2afd3eeb7c4555d019fa50a7d37d67ae20e158ed`。
首次通过权重：`5fa3807b42dc22c71c6f68542c0f1a6856d2f2d147ca05b8bec1e95e7614e185`；
最终权重：`e10733781a5d295f4353747ff761d06817d6a0d69d31f92893c14f5c5f1af07f`。

## 四种更新范围的结果

所有臂都训练读出。三个旧臂直接引用既有结果，本轮只运行最后一行。

| encoder可更新范围 | 检查点最多通过 | 首次16／16 | 末步通过 | 末步最差pair MSE残留 | 末步最差pair TV残留 |
|---|---:|---:|---:|---:|---:|
| 无 | 8／16 | 无 | 8／16 | 73.5313% | 83.7287% |
| 全部 | 16／16 | 700步 | 16／16 | 0.1295% | 2.0992% |
| 仅G2消息 | 10／16 | 无 | 4／16 | 60.1476% | 79.3594% |
| 除G2之外的encoder | 16／16 | 700步 | 16／16 | 0.2219% | 4.3076% |

残留相对于各pair的零谱差参考，门槛分别1%和10%。本轮末步总体MSE残留约0.0330%，
总体TV残留1.8232%；逐对门槛全部满足。全encoder臂末步MSE更低，不能把本轮通过门槛
解释为两种更新范围在拟合精度上等价。

- 第600步仅pair15未通过，MSE／TV残留6.5811%／11.1210%；第700步全部通过，最差
  MSE／TV残留0.3706%／3.7996%。之后所有预定检查点保持16／16。
- 先前最难的pair0（`mp-aaaaabjd`／`mp-aaaaarxw`），冻结encoder／仅G2／冻结G2并更新
  其余encoder的末步TV残留为83.7287%／79.3594%／1.5226%。全encoder为0.1808%。
  这对在本轮也获得拟合，成功不依赖继续调整G2消息权重。
- 末步最差MSE为pair14，最差TV为pair11；没有用均值掩盖未通过的材料对。

### 固定32材料的eDOS单谱指标

| 臂 | oracle 中位R²／失败率 | blind 中位R²／失败率 |
|---|---:|---:|
| 原G2 ep10 | 0.567433／3.125% | 0.547878／3.125% |
| 冻结encoder | 0.245919／31.250% | 0.212284／28.125% |
| 全encoder更新 | 0.640421／0.000% | 0.501260／0.000% |
| 仅G2更新 | 0.399469／25.000% | 0.472620／28.125% |
| 冻结G2、更新其余encoder | 0.622165／0.000% | 0.522578／0.000% |

口径：Q1 train固定32材料、各臂第2000步；无phDOS结果。这些补充指标不构成泛化或部署
收益。本轮仍使用纯谱差目标，尚未验证完整谱形损失及blind／phDOS保护。

## 验证、可复现性与记录完整性

- 相关测试17／17通过：`python3 -m unittest tests.test_g2_value_fit_probe tests.test_g2_small_fit_probe`。
  `bash tools/ci/check-static.sh`为99／99；新文件显式Ruff、`git diff --check`均通过。
- 新增测试实际训练两层encoder，核查G2／H1逐值不变、普通encoder及读出更新；主动切断
  冻结G2输出时，必须发现早期普通encoder断梯度。每个检查点的pair明细落盘检查通过。
- 第700／2000步checkpoint各重载前向5次，逐对判定一致且均16／16；最大重复TV差分别
  3.69e-7／2.82e-7，低于1e-5数值容差。末步与原保存预测最大TV差2.73e-7。
- 新臂全部21×16＝336行逐对轨迹完整，均标记`recorded_during_training`。旧仅G2臂的
  304行中间明细缺口及`recomputed_endpoint`标记保持原样，JSON单列
  `reference_artifact_deviations`，没有填补历史缺失。
- 改公共helper前，先按上轮记录的哈希保存当时精确版本为
  `output/g2_value_fit_q1/small_fit_helper_finalized.py`；旧训练、旧收尾和本轮执行代码
  分别校验与保留。原设计、结果及权重未被改写。
- 正式前缀`results/g2_non_value_fit_q1`：JSON、checkpoint_verification.json、selection16行、
  history84行、pair_history1040行、samples160行、final_pairs80行、summary5行和learning.png。
  output保存入口／helper／设计快照、每点记录、首次通过与最终权重、最终预测。

## 下一步

- 参数更新范围已收窄：这批材料的成功不要求再调整G2消息；普通encoder与读出联合适配
  已足够。普通encoder仍有注意力、相对位置投影、前馈及归一化等多个部分，不能单独定罪某个模块。
- 按设计，建议下一项固定G2及前5个普通encoder层，只开放最后一个普通encoder层与读出，
  保持同一16对、初值和2000步；检验更早层的参数更新是否也可以省去。尚未启动。
- 本轮日志、status、index已同步；研究范围仍是训练可拟合性，默认B7与全体实验队列保持现状。
  小样本相关工作尚未提交。
