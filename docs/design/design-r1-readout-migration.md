# 设计：R1 非卷积读出层迁移

> 对应 `docs/status.md` 的下一项。R1 是内部工作名，不改变已有项目编号；R1a 与 R1b 必须顺序执行，
> 每次启动训练前都要先将上一关的结论写入日志和 `status.md`。

## 目的与边界

当前 M1 使用固定长度的可学习 decoder query，并用沿 bin 轴的 Conv1d 输出头生成谱。卷积只适用于
等距、固定长度网格。R1 的目标是建立可处理可变 query 长度的非卷积接口；它**不**改变 Q1、E0/P0、
窗口、bin、标签重分箱、SumNorm KL/W1/Huber 或 H1。网格、窗口和 bin 另属后续数据表示实验。

R1 不假定必然提高 B7 分数。准确率的合并标准仍是 win；同时另设技术门槛，以判断非卷积接口是否
足够不伤性能、可作为下一阶段的载体。

## R1a：参数匹配的逐点读出头

### 假设

decoder 的 self-attention 已经在 token 间交换信息，输出 Conv1d 的局部 bin 耦合未必是必要的。将每个
token 表示 `h_i` 独立映射为谱值，可在不改变固定 query 或网格的前提下检验此假设。

### 唯一改动

- eDOS：以逐点 MLP `512→3→1`（GELU）替代现有单层三点卷积；1,543 参数，对照头为 1,537 参数。
- phDOS：以逐点 MLP `512→2704→2704→2704→2704→2704→1`（中间 GELU）替代六层三点卷积；
  30,647,137 参数，对照头为约 30.68M，容量差低于 0.2%。
- 两个任务仍使用原有固定 query、decoder、目标函数及 H1；MLP 在所有 bin 共享权重，不接收坐标。
- `--r1a_point` 默认关闭；关闭路径的状态字典、参数数和输出必须与当前 M1 完全相同。

### 关卡与判决

1. 先做单测：关闭路径等价、逐点置换合同、参数匹配、有限梯度、现有测试通过、真实 Q1 CPU 单步冒烟。
2. 运行 `_r1actl` 与 `_r1apoint`，均为 M1×10、seed 42、Q1、E0/P0、H1、0.05 dropout。
3. **准确率 win：**至少一个任务中位 R²提高至少 0.02，且任一任务失败率不恶化超过 1pt、另一任务
   不有害退化；才运行 `_r1along` M1×35。
4. **技术通过但不合并：**两任务都在平局线内、无 NaN/OOM、epoch 时间和显存增幅均不超过 15%。可进入
   R1b，但 R1a 不成为默认模型。
5. **技术失败：**任一任务出现有害退化或成本超过门槛，R1 全部 park；不得以增宽、加层等方式补跑。

## R1b：坐标生成 query

### 前置条件

仅当 R1a 技术通过后启动，并固定 R1a 的逐点头。若 R1a 获得准确率 win，R1b 使用其确认后的配置；若
R1a 仅技术通过，则仍明确标为非默认载体。

### 假设与唯一改动

将固定长度的 `edos_query_embed/phdos_query_embed` 及固定长度 target 表替换为任务独立的坐标 query
生成器 `q_e(x)`、`q_p(x)`：归一化坐标经 plain MLP `1→128→512` 产生 query；decoder target 为同长度
零张量。逐点头只读取 decoder 输出，不再额外加入坐标。

因此，改变 bin 数或不均匀坐标时，decoder 能接受相应长度的 query；但本关仍只在原 E0/P0 标签训练和
评估。R1b 不使用 Fourier/RFF，以免重开已 park 的 Q2 因素。

### 关卡与判决

1. 单测：关闭路径等价；任意 `[B,E]`/`[B,P]` 坐标长度可前向；同坐标确定性；坐标梯度与 query MLP
   梯度存在；E0/P0 原网格 CPU 冒烟有限。
2. 运行 `_r1bctl` 与 `_r1bcoord`，均为 M1×10、Q1、原 E0/P0、seed 42；唯一变量为固定 query 到坐标
   query 的替换。
3. 准确率 win 的 35 epoch 规则同 R1a。若仅技术通过（平局且成本增幅≤15%），可进入网格/窗口/bin
   数据表示设计，但仍不得替换 B7 默认模型；若失败，R1b park，网格实验不启动。

## 夜间交接与监控规则

1. **夜间训练前：**运行相关单测、`python3 -m unittest discover tests`、真实 Q1 CPU 单步冒烟；任一失败
   则不启动 GPU。
2. 每一臂使用唯一 tag、`setsid + nohup`，输出到对应 `output/ablation_*`；不得删除任何 checkpoint。
3. 每个 epoch 监控日志、GPU 显存、NaN/Inf、checkpoint 和训练存活；崩溃只允许从相同 tag 的最新
   checkpoint 续跑一次。第二次崩溃即停止并记录失败。
4. 一关完整结果落盘后，agent 必须先写 `docs/logs/log-日期-r1*.md`、更新 `status.md`，再按上述明确
   门槛决定是否启动下一关；不得同时并跑不同关卡，也不得为填满 GPU 重开 park 项。
5. blind 与 oracle：每个取得 accuracy win 的臂都要重跑 H1 blind 指标及 oracle–blind gap；未获 win
   的 pilot 至少在日志中明确标注 oracle 口径和 blind 未重跑原因。

## 预期 GPU 用量

- R1a 成对 pilot 约 1 小时；若 win，35 epoch 确认约 2 小时。
- R1b 只在 R1a 技术通过后运行，成对 pilot 约 1 小时；若 win，确认约 2 小时。
- 该流程的最大有意义用量约 6 小时。额外 GPU 时间只能用于预先批准的多 seed 复现，不能用无结论的
  超长训练填充。
