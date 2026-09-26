# 日志：2026-09-26 — 冻结 G2 表征的结构谱差读出实验

## 范围与预定假设

- 马尚酱确认执行冻结 encoder、重新训练既有读出的机制实验；设计见
  `../design/design-g2-frozen-readout-probe.md`。
- 固定 `_g2edge` M1 epoch10／seed42／FP32／Q1／SumNorm。复用原decoder、eDOS query、target
  token和CNN head；两臂唯一差别是训练时表征与目标正确对应，或在同绝对组成组内每batch随机置换。
- 目标是检验现有表征中的结构谱差是否可被读出，以及这种学习能否迁移到留出的约化组成。
  机制探针不直接替换生产模型，不将本轮结果解释为整体准确率晋级。

## 数据与预算

- train：2,591对／3,115材料／1,198约化组成组。
- 主验收valid：163对／256材料／117约化组成组，均未出现在encoder原始Q1 train中。
- 补充valid：全部320对／394材料／168约化组成组。按绝对计数组的原划分不重叠，但其中51个
  约化组成在train出现过，故不纳入主验收。只根据元素输入作此划分，没有按误差或标签筛选。
- 每臂10轮、每batch16对，共1,620步；FP32、AdamW lr5e-5、betas0.9/0.99、weight decay0.01、
  梯度裁剪1.0，dropout关闭。最终权重固定取最后一步，不通过valid选取。
- 训练目标为带符号概率谱差的MSE（乘128调整数值尺度），约化组成等权；正式判据用带符号谱差
  TV。它不约束共有谱形，因此单样本oracle/blind准确率必须同时报告。

## 实现与调用链

- `tools/eval/g2_frozen_readout_probe.py`：`main` → `run_probe` 组织完整运行。
  - `composition_keys`／`prepare_pair_plan` 核对材料ID、绝对组成和约化组成隔离；`group_weights`
    实现组等权，`shuffled_donors` 为组内均匀置换（允许固定点且batch内一致）。
  - `extract_cache` 复用原数据加载、预处理和 `FeatureProbe`；`FrozenEdosReadout` 复制现有
    decoder／query／target／CNN；`memory_batch`／`predict` 使用冻结原子特征。
  - `contrast_mse`／`train_arm` 实现两臂优化；`positive_logit_control` 检查目标与softmax损失
    可优化，不能据此证明完整decoder优化充分。
  - `metric_tables`／`summarize_results`／`verdict_from_comparisons` 汇总样本、配对和组等权
    比较，并按训练前写下的阈值判定。复用既有成对bootstrap与原子checkpoint保存函数。
- `tests/test_g2_frozen_readout_probe.py`：9项合成测试，涵盖组成隔离、对应随机化、冻结边界、
  前向等价、梯度、相同更新预算、损失及结论判据。`tools/ci/check-static.sh` 已纳入该模块。
- 入口、设计与工作状态同步到 `tools/eval/README.md`、`docs/index.md`、`docs/status.md`。

## 命令、验证与产物

通过 `setsid + nohup` 后台运行：

```bash
python3 -u tools/eval/g2_frozen_readout_probe.py --device cuda
```

- 本地运行目录：`output/g2_frozen_readout_q1/`；控制台日志：`output/g2_frozen_readout_q1.log`。
- 正式结果前缀：`results/g2_frozen_readout_q1`；保存samples、pairs、summary、comparisons、
  history、plan六份CSV及JSON结论。源checkpoint与冻结特征须在运行前后保持哈希一致。
- 训练前验证：新增9/9项测试通过；新代码Ruff通过；CI静态检查、编译及75/75项CPU测试通过。
  完整数据训练结果将在下节记录；本轮未执行完整仓库测试套件。

## 结果

两臂各1,620步完整结束，合计25,357,825个可训练读出参数；缓存、训练和结果汇总共279.5秒。
没有按valid调参、选择中途checkpoint或延长预算。

### 1. 训练内有部分改善，留出组成明确退化

谱差误差均为约化组成组内平均后再组等权平均，越低越好。

| 范围 | 读出 | 带符号谱差TV误差 | 训练目标对应的谱差MSE |
|---|---|---:|---:|
| train，1198组 | original | 0.268993 | 1.300601 |
| train，1198组 | matched | 0.252549 | 1.081616 |
| train，1198组 | shuffled | 0.269346 | 1.316645 |
| 主valid，117组 | original | 0.283055 | 1.519184 |
| 主valid，117组 | matched | 0.290771 | 1.660440 |
| 主valid，117组 | shuffled | 0.279835 | 1.520336 |

- **train matched相对original：**TV误差下降6.11%，绝对改善0.016444，组bootstrap95%区间
  `0.014829…0.018120`；MSE下降16.84%。有可测的训练内学习，但TV未达预定10%门槛。
- **主valid matched相对original：**TV误差增加2.73%；按“参考−matched”定义的改善为
  `−0.007716`，区间`−0.012075…−0.003272`，全部低于零。相对shuffled误差增加3.91%，
  改善区间`−0.016815…−0.005378`。
- 主valid零谱差参考误差为0.279784；matched比它恶化3.93%，改善区间
  `−0.016892…−0.005431`。三个参考均未达到预定的5%改善，更没有通过区间条件。
- 主valid的MSE也增加9.30%，退化方向不依赖从训练MSE改用验收TV。全部320对的补充valid
  结果同向：matched／original的TV误差为0.283776／0.277656，恶化2.20%。
- 主valid预测谱差TV中位数从0.007882增加到0.023419，约2.97倍；目标TV中位数0.281320。
  增大的响应没有对应更准确的留出结构谱差。

### 2. 单材料谱形付出了明显代价，不能作为候选模型

下表为配对材料子集的eDOS中位R²／失败率（%），不是全体Q1指标。

| 范围 | 读出 | oracle中位R²／失败率 | blind中位R²／失败率 |
|---|---|---:|---:|
| train，3115材料 | original | 0.52577／4.205 | 0.48900／8.443 |
| train，3115材料 | matched | −0.55290／75.859 | −0.59338／76.212 |
| train，3115材料 | shuffled | −0.00045／52.488 | −0.01265／80.835 |
| 主valid，256材料 | original | 0.46869／3.516 | 0.43450／8.203 |
| 主valid，256材料 | matched | −0.84783／77.734 | −1.00699／79.297 |
| 主valid，256材料 | shuffled | 0.00015／48.047 | −0.01255／79.688 |

本次只优化两谱之差，没有约束它们的共有谱形；这是设计预先指出的自由度。该代价进一步阻止
直接部署探针，不能据此推断输入或encoder没有可用信息。phDOS未作为本轮训练或保护验收目标。

### 3. 实际运行核验与缓存兼容修复

- 源模型与缓存读出的最大概率谱TV：train `3.5063e-7`、valid `2.3150e-7`，通过`1e-5`门槛。
- 独立logits正对照MSE由0.726470下降到`3.4367e-6`，残留比例`4.7307e-6`，通过下降99%的门槛。
  它仅验证目标和softmax损失可优化，不证明完整decoder在1620步内已充分优化。
- 运行结束时15份已登记输入及原始冻结缓存哈希一致，包括源checkpoint、数据、脚本与预注册设计。
  正式结果为10,527行样本、8,733行材料对、9行汇总、9行比较、20行history、2,911行计划。
- 结束后的缓存回读发现材料ID保存成NumPy字符串，`weights_only=True`不接受该类型。原缓存从
  本轮已核对哈希的可信文件回读成功；现将入口中的ID显式转为Python字符串（仅一行序列化修复）。
  原始输入缓存保留，另存`output/g2_frozen_readout_q1/frozen_features_portable.pt`；它通过
  `weights_only=True`回读，所有特征、目标、原预测张量逐位相同，ID文本与索引相同。
- **执行版本可追溯：**数值运行使用的脚本保存在
  `output/g2_frozen_readout_q1/script_executed.py`，哈希与主结果JSON一致；当前入口多了上述
  `str(ids[split][index])`转换。两版本与两缓存哈希、回读结果记录在
  `results/g2_frozen_readout_q1_cache_compatibility.json`。没有重新训练或更改数值结果。

## 结论

- **状态：complete；固定读出干预未获支持。**机器判定为
  `readout_intervention_not_supported`，按训练前阈值执行，没有因train出现改善而降低门槛。
- **唯一机制结论：**在固定G2 ep10表征、现有读出、谱差目标和本轮预算下，正确对应训练能部分
  降低已见组成的谱差误差，但不能将该改善迁移到未见组成，实际留出误差比原读出更高。
- 因而本轮不支持“直接重训该读出就能释放可泛化结构谱差信息”的具体方案；既不能宣称完全
  没有学习，也不能宣称encoder无信息、全部读出方案无效或标签一定有错。仍可能涉及表征与读出的
  协同适配、训练目标或优化预算，这些解释未在本轮中被分别识别。
- 不将本探针晋级为全体准确率pilot，B7默认与G2原pilot的park状态保持。所有生成检查点保留。

## 交接

- 本项授权范围已完成，status与index同步完成状态；不改全局实验纪律和默认模型。
- 下一项候选干预可只改变encoder是否参与更新，用同一配对任务检验表征与读出是否需要协同适配。
  这是待讨论方案，未获新训练授权；本轮负结果不足以认定encoder就是根因。若转向全体准确率
  候选，必须恢复完整谱形约束并验证blind／phDOS保护，不能直接沿用这个纯谱差目标。
