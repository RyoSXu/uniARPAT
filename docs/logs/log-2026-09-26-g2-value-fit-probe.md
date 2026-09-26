# 日志：2026-09-26 — 仅 G2 消息更新的固定16对拟合

## 范围与唯一结论

- 马尚酱批准继续收窄encoder更新范围。本轮按
  `docs/design/design-g2-value-fit-probe.md`只新增一个训练臂：G2消息与读出可更新，其余
  encoder及H1权重冻结；复用上轮同一16对／32个Q1 train材料、原G2 ep10初值和2000步。
- 状态：**closed；指定预算内未获得全部16对的拟合见证**。21个预定检查点最多10／16通过，
  末步4／16；全encoder对照第700步已16／16。结论限于这套初值、优化设置和预算。
- 不能据此判定G2没有作用或容量不足，也不能把差距单独归因于注意力、前馈或归一化。
  末段明显回退，优化不稳定仍是解释之一。本轮有明确记录的产物缺口，见下文。

## 实施地图

调用链：固定atom_src／几何输入 → 普通encoder层（权重固定，保留反向传播）→ 各层G2残差
→ 原eDOS读出；当前memory同时供冻结H1重算blind尺度。

| 文件 | 新增或修改的符号及职责 |
|---|---|
| `tools/eval/g2_small_fit_probe.py` | 新增`configure_encoder_scope`，选择frozen／joint／g2_only更新范围；`fit_arm`仅向优化器传递可训练encoder参数，并逐检查点写入pair明细 |
| `tools/eval/g2_value_fit_probe.py` | `reference_path`／`load_reference`核对旧数据与执行快照；`gradient_boundary_check`检查每层G2梯度及冻结边界；`verify_checkpoints`重载权重并检查数值波动和逐对判定；`run_probe`／`main`训练或仅恢复汇总 |
| `tools/eval/plot_g2_small_fit.py` | `plot_results`增加G2-only曲线，读取完整汇总轨迹 |
| `tests/test_g2_value_fit_probe.py` | 7项测试：更新范围切换、跨冻结层梯度、主动切断梯度的反例、实际参数更新与重载、伪拟合见证拒绝、数值容差、导出失败后的逐对轨迹保留 |
| `tools/ci/check-static.sh`、`tools/eval/README.md` | 纳入测试与入口说明 |

训练时encoder可更新参数3,351,558／17,734,662（18.90%），其余14,383,104个参数冻结；
读出25,357,825个参数继续训练。6层G2各558,593个参数。普通encoder权重冻结不等于注意力
激活固定：G2改变前向状态后，后续注意力仍会重新计算。

## 运行合同与证据

- 标签`g2_value_fit_q1`；选择seed20260926、初始化seed42，原M1／G2 ep10 FP32检查点。
  完整16对batch各步重复，共2000步；eval模式，关闭dropout。AdamW lr5e-5、betas(.9,.99)、
  weight_decay.01，读出与可训练encoder分别裁剪1.0。梯度裁剪范围随更新参数集合缩小。
- 目标仍为纯谱差MSE；逐对要求残余MSE≤零谱差MSE的1%，残余TV≤目标TV的10%，16对
  同时通过。共有谱形无锚定。仅train，valid／test未评估，phDOS未评估。
- 初始预测最大TV差1.89e-7、blind尺度相对误差2.36e-7；读出梯度相对RMS2.02e-5、
  损失相对误差2.51e-7，均通过既定门槛。6层G2梯度范数0.00380～0.01213，均有限非零。
- 训练后G2改变3,351,557个值（相对L2变化0.062891），读出改变25,357,799个值；其余
  encoder状态和H1逐值不变。原源文件、数据、缓存和旧结果保护检查通过。
- 训练循环耗时198.60秒；收尾恢复的时间／显存单列，不冒充训练峰值。没有增加训练步数
  或重新训练，也没有用第1900步替换末步结果。

```bash
setsid nohup python3 -u tools/eval/g2_value_fit_probe.py > output/g2_value_fit_q1.log 2>&1 < /dev/null &
python3 -u tools/eval/g2_value_fit_probe.py --finalize-existing > output/g2_value_fit_q1_finalize.log 2>&1
python3 tools/eval/plot_g2_small_fit.py --prefix results/g2_value_fit_q1
```

训练缓存SHA256：`98ec2b34d588ad9aec16a14e2afd3eeb7c4555d019fa50a7d37d67ae20e158ed`。
最终权重SHA256：`9e1ba7ad7a91aefe286a808edc9c479adca60b7740be34cda81ecfe0a40a16a0`。
训练前设计与代码保留原快照，收尾恢复代码另存`finalizer_executed.py`；正式JSON分别记录
训练时与收尾时的输入哈希。

## 结果

所有训练臂均训练读出，以下差别仅是encoder的可更新范围。旧对照来自上轮正式CSV。

| 更新范围 | 预定检查点最多通过 | 首次16／16 | 2000步通过 | 末步总体MSE残留 | 末步最差pair MSE残留 | 末步最差pair TV残留 |
|---|---:|---:|---:|---:|---:|---:|
| encoder冻结 | 8／16 | 无 | 8／16 | 1.44894% | 73.5313% | 83.7287% |
| 全encoder | 16／16 | 700步 | 16／16 | 0.001719% | 0.1295% | 2.0992% |
| 仅G2消息 | 10／16 | 无 | 4／16 | 1.85504% | 60.1476% | 79.3594% |

残留均相对于对应零谱差参考；总体MSE残留是MSE均值之比，验收仍按每对分别判断。

- 第1900步G2-only的总体MSE／TV残留为0.43212%／3.81844%，均值已较小，但最差pair
  为8.23438%／12.74236%，只有10／16通过。第2000步总体MSE上升约4.29倍，退回4／16。
  完整21点汇总轨迹保留，未按均值或最好检查点改写结论。
- 末步通过pair索引3、4、7、8。冻结读出臂已通过的1、2、6、14在G2-only末步未通过；
  冻结读出臂未通过的8对在G2-only末步也均未通过。这是末步比较，不能替代丢失的中间逐对轨迹。
- pair0（`mp-aaaaabjd`／`mp-aaaaarxw`）仍是最差pair：冻结／仅G2／全encoder的TV残留
  为83.7287%／79.3594%／0.1808%。本预算下，仅更新G2未恢复这对的谱差拟合。

### 32个train材料的单谱指标

| 臂 | eDOS oracle 中位R²／失败率 | eDOS blind 中位R²／失败率 |
|---|---:|---:|
| 原G2 ep10 | 0.567433／3.125% | 0.547878／3.125% |
| 冻结encoder | 0.245919／31.250% | 0.212284／28.125% |
| 全encoder | 0.640421／0.000% | 0.501260／0.000% |
| 仅G2消息 | 0.399469／25.000% | 0.472620／28.125% |

口径：Q1 train固定32材料、各臂第2000步；oracle／blind分列，phDOS无结果。这些是训练
小样本的补充指标；纯谱差目标未保护共有谱形，不能据此晋级全体准确率方案。

## 收尾故障、修复与产物缺口

1. 训练2000步和最终权重保存均完成，随后原重载检查以最大预测TV差≤1e-7比较两次前向并
   抛错。实际同一权重连续5次前向最大差约6.8e-7，各次均4／16通过，存在可复现数值波动。
2. 把**重载一致性**容差改为1e-5，与原初始链路一致性标准相同；拟合门槛未变。同时每个
   重载checkpoint连续前向5次，要求逐对通过向量一致。正式核验最大重复TV差6.62e-7，
   与恢复预测最大差6.34e-7；4／16判定一致。测试另确认1e-3量级的预测变化仍被拒绝。
3. 当时`fit_arm`每100步只立即保存汇总history，逐对轨迹留在内存，报错使第100～1900步
   的19×16＝304行明细丢失。**这部分未恢复，未重跑补造。**汇总history包含全部21点，
   足以核对各点未达全对门槛，但无法追溯第1900步具体是哪6对未过。
4. 新实现已逐检查点保存pair明细；新增模拟末尾导出失败测试。`--finalize-existing`仅从
   完成2000步的权重恢复汇总，核对原执行快照、汇总末步指标与冻结边界，保留原训练来源。
5. 新臂正式pair_history仅含重新前向获得的0／2000步端点32行，标记
   `measurement_source=recomputed_endpoint`；旧对照672行完整保留。JSON中的
   `artifact_deviations`列出全部缺失步数。结果有效性与记录完整性分别报告。

## 验证与正式产物

- 相关测试：`python3 -m unittest tests.test_g2_value_fit_probe tests.test_g2_small_fit_probe`，
  修复后14／14通过；静态门禁`bash tools/ci/check-static.sh`，96／96通过。
- 新文件显式Ruff检查通过，`git diff --check`通过；最终权重严格加载及5次实际前向核验通过。
  学习曲线从完整汇总history生成并人工查看。
- 正式前缀`results/g2_value_fit_q1`：JSON、checkpoint_verification.json、selection16行、
  history63行、pair_history704行（含上文缺口）、samples128行、final_pairs64行、summary4行，
  以及learning.png。检查点与训练／恢复日志留在`output/g2_value_fit_q1*`。

## 下一步

- 建议以已完成的全encoder臂为对照，**仅冻结G2消息参数，开放其余encoder与读出更新**，
  仍用同一16对、初值、目标和2000步。若通过，排除“拟合这批材料必须更新G2消息参数”；
  若不通过，保留协同适配及优化设置解释。这个互补臂尚未启动。
- 先完成这组可更新参数集合的互补比较，再考虑普通encoder内部的分层定位。
  当前没有全体准确率pilot的晋级依据，默认B7维持当前版本。
- `status.md`、`index.md`已同步。本轮及上轮小样本工作均未提交；既有`docs/decisions.md`
  改动保持原样。
