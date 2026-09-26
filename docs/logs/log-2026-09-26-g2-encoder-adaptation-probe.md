# 日志：2026-09-26 — G2 encoder 与读出的联合适配实验

## 范围与假设

- 马尚酱在冻结读出结果后确认继续。预注册见
  `../design/design-g2-encoder-adaptation-probe.md`；本轮唯一因素是encoder是否更新。
- 沿用G2 M1 ep10／seed42／FP32／Q1初值、配对任务和纯谱差目标；新增joint臂与上轮冻结encoder
  的matched臂比较。original、shuffled和零谱差为附加参考，不重新训练它们。
- train为2591对／3115材料／1198约化组成；主valid为原Q1 train未见的117约化组成、163对、
  256材料。全部valid320对／394材料只作补充。未读取test，valid不用于选择预算或checkpoint。
- 预算仍为10轮配对任务、16对/batch、保留尾batch，共1620步；不是完整Q1数据的10个epoch。
  encoder和读出均eval模式，关闭dropout而启用梯度；各自AdamW lr5e-5、betas0.9/0.99、
  weight_decay0.01，梯度各自裁剪1.0。元素嵌入和H1头参数冻结。

## 改动与调用链

`Transformer.forward的真实encoder输入 → 原encoder → 原eDOS读出 → 谱差损失`

- `tools/eval/g2_encoder_adaptation_probe.py`：新增独立机制入口，生产模型文件未改。
  - `split_encoder_inputs`／`extract_inputs`：以观察钩子保存实际atom_src、几何和G2边；去除
    padding并重编号，不引入标签特征。`encoder_batch`为新batch补齐原子并重编边的batch索引。
  - `AdaptiveEdosProbe`：复制既有encoder、`FrozenEdosReadout`及冻结H1；encoder更新后，
    blind尺度由新memory重新计算，未复用过期尺度。
  - `gradient_equivalence`：首个train batch检查新路径与旧冻结缓存的读出损失／梯度等价。
  - `clip_and_step`／`train_joint`：分别裁剪encoder和读出，保留上轮读出优化规则和更新预算。
  - `summarize_joint`／`verdict_from_comparisons`：按约化组成配对比较四项参考，执行预定门槛。
  - `run_probe`／`main`：限制输入、复核旧结果与数据哈希，保存执行副本及独立产物，拒绝覆盖。
- `tests/test_g2_encoder_adaptation_probe.py`：7项边界与判据测试；纳入
  `tools/ci/check-static.sh`。导航与入口说明更新至index、status和tools/eval/README。
- 上轮冻结读出代码和正式数值产物、本轮生产模型与默认方案均保留原状。

## 命令、技术检查与产物

以setsid + nohup运行：

```bash
python3 tools/eval/g2_encoder_adaptation_probe.py --device cuda
```

- 输出目录：`output/g2_encoder_adaptation_q1/`；正式控制台日志：
  `output/g2_encoder_adaptation_q1_run.log`；正式结果前缀：`results/g2_encoder_adaptation_q1`。
- 第一次入口检查将数据加载文件误写为`dataset.py`，实际应为`datasets/dataset.py`。该次在
  建立运行目录和开始训练之前停止；修正路径后执行本次。失败日志保存在
  `output/g2_encoder_adaptation_q1.log`，没有训练重试、改预算或覆盖已有checkpoint。
- 新增测试7/7通过；新代码Ruff和git diff空白检查通过；仓库静态门禁与CPU合同测试82/82通过。
  完整仓库测试套件未运行，本次未改变生产训练调用链。
- 真实配对材料初始前向：最大预测TV差train `3.9241e-7`、valid `3.9178e-7`；blind尺度最大
  相对误差分别`7.3506e-7`、`5.4557e-7`，均低于`1e-5`。
- 首batch读出梯度相对RMS差`2.4763e-5`，低于`1e-3`；两路径损失同为`1.01339209`。
  通过后开始1620步训练。这里只证明实现足以用于该对照，不代表机制已成立。

## 结果

固定1620步完成，机器判定为`joint_adaptation_not_supported`。本轮新增17,734,662个可训练
encoder参数，读出仍为25,357,825个参数；总计43,092,487。没有根据valid延长训练或选择中途权重。

### 1. 相对冻结读出只有有限改善，没有胜过原始模型

谱差误差先按约化组成组内平均，再组等权，越低越好。

| 范围 | 臂 | 谱差TV误差 | 谱差MSE |
|---|---|---:|---:|
| train，1198组 | original | 0.268993 | 1.300601 |
| train，1198组 | 冻结matched | 0.252549 | 1.081616 |
| train，1198组 | 联合joint | 0.251299 | 1.048577 |
| 主valid，117组 | original | 0.283055 | 1.519184 |
| 主valid，117组 | 冻结matched | 0.290771 | 1.660440 |
| 主valid，117组 | 联合joint | 0.287445 | 1.518717 |
| 主valid，117组 | shuffled | 0.279835 | 1.520336 |

- **直接因果对照joint vs 冻结matched：**train的TV误差下降0.495%，改善95%区间
  `−0.000361…0.002901`；主valid下降1.144%，绝对改善0.003326，区间
  `−0.002115…0.008588`。两处区间都跨零，未达到预定训练10%／留出5%门槛。
- **joint vs original：**train TV误差下降6.58%，区间`0.015742…0.019662`；主valid却
  增加1.55%，改善区间`−0.008881…−0.000572`全部低于零。因此不能把略胜退化的冻结matched
  当成释放了可泛化信息。
- 主valid的零谱差参考为0.279784；joint比它误差增加2.74%，改善区间
  `−0.012504…−0.003312`。相对shuffled也增加2.72%，四项参考均未通过留出门槛。
- **MSE与TV不能混为一个结论：**joint的train／主valid MSE相对冻结matched分别下降3.05%／
  8.54%；主valid MSE已接近original（仅低0.031%），而TV仍更差。不能说联合适配完全没有作用，
  也不能挑选MSE改写原定主判据。
- 主valid预测谱差TV中位数original／matched／joint为`0.007882／0.023419／0.037642`，
  目标为`0.281320`；joint响应是original的4.78倍，谱差方向余弦中位数为
  `0.005930／0.008086／0.042166`。响应和方向出现变化，但整体谱差仍没有预测准确。
- 全部valid320对的TV误差original／matched／joint为`0.277656／0.283776／0.283612`；
  joint相对matched改善仅0.058%，区间`−0.004295…0.004629`，未改变主验收结论。

### 2. 单材料谱形进一步退化

下表仅为配对材料的eDOS中位R²／失败率（%），Q1、源ep10后机制训练1620步，不能替代全体Q1
指标。H1权重冻结，但joint的blind尺度由更新后的memory重新计算。

| 范围 | 臂 | oracle med／fail | blind med／fail |
|---|---|---:|---:|
| train，3115材料 | original | 0.52577／4.205 | 0.48900／8.443 |
| train，3115材料 | 冻结matched | −0.55290／75.859 | −0.59338／76.212 |
| train，3115材料 | 联合joint | −0.77453／81.284 | −0.83904／79.872 |
| 主valid，256材料 | original | 0.46869／3.516 | 0.43450／8.203 |
| 主valid，256材料 | 冻结matched | −0.84783／77.734 | −1.00699／79.297 |
| 主valid，256材料 | 联合joint | −1.12740／85.547 | −1.16413／86.719 |

- 主valid逐材料`oracle R² − blind R²`的p50／p90／p99：original为
  `0.00977／0.18283／1.44632`，matched为`0.01534／1.64986／7.02528`，joint为
  `−0.09595／1.88980／7.22724`。p50为负不意味着blind整体优越，两种尺度下准确率均已明显退化。
- 纯谱差目标没有约束共有谱形，原定设计就存在这项自由度。两臂的单谱退化表明当前探针不能
  晋级准确率候选；不能将它解释为几何输入必然无效。phDOS未验收。

### 3. 运行完整性与代价

- 本轮缓存、检查、训练、评估和汇总总计244.3秒；联合训练194.6秒，上轮冻结matched训练
  116.0秒，约1.68倍。此次对齐的是更新次数和数据暴露，**不是等FLOP或等耗时**。
- CUDA峰值分配显存5.78GiB。encoder／readout状态相对L2变化为4.60%／4.72%，H1状态逐位
  不变。所有43份登记输入和新encoder输入缓存运行前后哈希一致，包含旧结果、源checkpoint、
  数据、依赖代码与两轮设计。
- 保存执行脚本与设计副本、`encoder_inputs.pt`、`joint_final.pth`、`joint_predictions.pt`。
  正式CSV为14,036行samples、11,644行pairs、12行summary、12行comparisons、10行history、
  2,911行plan；其中保留上轮三项参考以便逐行对照。
- 三份新张量产物均通过`weights_only=True`回读；执行脚本、设计副本的哈希与登记值及当前原件
  一致。新CSV中的旧参考逐行复核通过（10,527行samples、8,733行pairs，数值容差1e-12）；
  index表中25个文档引用均存在。

## 结论与交接

- **状态：closed，本轮授权任务完成；联合适配方案未获支持。**在同初值、相同谱差目标和1620步
  预算下，开放encoder更新没有给主valid带来明确的TV收益；比原始模型仍差，并进一步损害单谱
  准确率。不晋级该方案，不自动延长训练或启动超参扫描，B7默认保持。
- **能排除的解释范围：**“只要开放现有encoder更新，这套训练就能解除冻结读出的限制”未获
  支持。不能扩大成encoder无信息、读出无用或已找到几何丢失层；一次训练的组bootstrap也不涵盖
  训练seed不确定性。MSE的有限改善作为连续效应完整保留。
- **实验鉴别力仍有边界：**两轮在train的TV改善都只有约6%，尚未证明实际网络能充分拟合这项
  任务；上轮独立logits正对照只能证明目标可优化。因此，继续用全配对集合跑一个新模块仍难以
  区分“训练没有学会”和“表征中不可读出”。这是对后续设计的限制，不是新的根因结论。
- **下一项建议：**先在固定8–16个train材料对上，对实际网络做冻结／联合两臂可拟合性检验，
  并先核对输入是否可区分。冻结可拟合则排除这批样本的绝对不可读出；仅联合可拟合则继续检查
  encoder适配的作用；两者都失败则先检查实际优化和输入别名，而非宣称标签错误。需单独规定
  小样本选择、拟合门槛和预算；本轮没有启动这项训练。
- **建议的规则修改，待用户决定：**以后要用失败的训练来限制表征／读出假设，先要求实际待测
  网络通过小样本可拟合性正对照，不能用独立logits可拟合替代。本轮只记录建议，未写入AGENTS
  或workflow。
- status、index同步完成状态与下一项建议；全部checkpoint保留，本轮改动未提交。
