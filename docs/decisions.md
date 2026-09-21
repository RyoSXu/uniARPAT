# 已确定的决策

本页记录当前已经确定的选择，不是时间线。已完成实验写入 `logs/`，其机器可读证据写入
`results/`。术语定义见 `glossary.md`。

## 当前科学与数据决策

1. **参考与数据：**Q1 上的 B7 `_e9ctl` 是唯一的当前对照。pre-Q 结果仅作历史记录，
   必须明确标注。
2. **默认训练目标：**使用含 KL、W1 和 Huber 项的 SumNorm；eDOS 与 phDOS 均报告中位 R²
   和失败率。
3. **盲推理：**使用 H1 eta/gamma 头。报告 blind 分数与 oracle–blind gap 分布；不得使用
   全局尺度常数。
4. **数据完整性：**缺失值保持缺失，coverage mask 不进入损失；除非记录新的数据决策，Q1
   已排除记录继续排除。
5. **网格约定：**生产使用 E0 + P0。非均匀网格需要逐点或能量条件读出层，不得与当前卷积头
   搭配。
6. **输入约定：**生产推理只能使用从 CIF 得到的信息。任何外部数据、预训练权重或由标签推导
   的特征，均需单独获批的实验。
7. **读出层与 Query 结构（R1）：**
   - 逐点 MLP 输出头（R1a）在参数匹配下与 Conv1d 头在统计平局线内等价（eDOS/phDOS Δmed 为 −0.010/−0.010，训练速度提升 2.7%，显存降低 2.0%），技术通过并可作为非卷积载体，但当前保持默认关闭。
   - 标量单点 MLP 坐标 query 生成器（R1b：1→128→512 plain MLP）在离散 E0/P0 网格上表现出显著性能退化（eDOS Δmed −0.055，phDOS Δmed −0.036，Cv MAE 恶化 0.16），证明仅靠一维标量经简单 MLP 映射得到的 query 流形严重退化了 cross-attention decoder 的频率/能量特化能力。该方案已 park，代码保持默认关闭，下游网格/窗口/bin 重分箱实验不启动。
8. **结构表征与周期边消息（G2）：**
   - 周期多镜像边条件消息模块（G2a，全量保守枚举 $T \in [-K,K]^3$、$R=5.5\text{ \AA}$、不裁边、6 层径向残差）在 Tesla V100、batch 32 生产配方测得峰值显存比值 1.165x（无 OOM）及单轮耗时比值 1.273x（+27.3%）。
   - Q1 M1×10 成对 pilot 实测表明：G2a 相对控制臂的 eDOS/phDOS Oracle 中位 $R^2$ 变化为 −0.0066/−0.0048，失败率变化为 +0.22/+0.44pt；Blind 中位 $R^2$ 变化为 −0.0024/−0.0033，失败率变化为 +0.26/+0.57pt；$C_v$ MAE 变化为 +0.0104。
   - 双任务指标均严格落在平局线（$|\Delta\text{med}| < 0.02$ 且 $|\Delta\text{fail}| < 1.0\text{ pt}$）内，未达到正向突破门槛（任一任务 $\ge +0.02$）。G2a 正式 park，保持默认关闭，不进入 35-epoch 确认，不展开超参扫描。
9. **结构信号诊断：**Q1 test 中，同约化化学式但缓存结构表示不同的 176 个组（402 样本）具有显著
   组内谱形变化：eDOS/phDOS 中位 TV 为 0.252/0.320；谱形离散度与 B7 组中位 R²呈负相关
   （Spearman −0.507/−0.650）。因此，G2a 的无收益只否定该特定周期消息模块，不能推出结构信息无用。
10. **R2a decoder 缩减：**Q1 M1×10 中，共享 decoder 从 6 层缩减至 3 层相对控制的 Oracle
    eDOS/phDOS 中位 R²为 −0.0164/−0.0011，失败率为 +0.48/+0.35pt，均未越过负向平局线；训练
    耗时为 0.821x、峰值显存为 0.812x。它不是 accuracy win，B7 6 层仍为基线和默认，但 3 层获准
    作为后续原子加性 phDOS 读出的低成本载体；不进行 35-epoch 确认或深度扫描。
11. **R2b 原子加性 phDOS：**Q1 M1×10 中，固定 P0 的原子 token 非负贡献求和读出相对 R2a 3 层
    控制的 Oracle eDOS/phDOS 中位 R²为 −0.0081/−0.0328，失败率为 −0.35/+0.00pt；phDOS 越过
    −0.02 保护下界，故 park。虽参数、每轮耗时、显存分别降 52.0%/22.6%/17.6%，不启动确认或头部
    扫描。该结论仅否定这个低容量、总谱弱监督的原子加性读出组合，不能推出结构信息无用。
12. **C2.1b 损失归因：**B7 epoch 33 在 Q1 valid 的冻结审计未显示 eDOS/phDOS 主谱形的负梯度冲突
    （最后 encoder／共享 decoder FFN 的中位余弦为 0.074/0.009），H1 eta 也未压制主任务。高熵 phDOS
    更难但已获得更高当前损失，不能导出有方向的再加权机制；默认 SumNorm KL/W1/Huber 与 H1 保持，
    不得以此重开 L3 式权重扫描。详见 `logs/log-2026-09-21-c2-1b-loss-attribution.md`。

## 实验纪律

1. 每次实验只改变一个因果因素，并在开始前写明成功判据。
2. 先用 10 个 epoch 的 pilot 筛选，再以等量算力确认胜出方案。
3. 不得在不同归一化方案或不同数据口径之间复用检查点。
4. 保存结果 CSV 与完整命令；本地检查点及其复制的配置仅为便利产物，不是事实记录。

## 工程与协作

1. 原始数据不纳入 Git；清单、数据划分、参考表、代码和结果 CSV 应纳入版本控制。
2. 当前状态与活跃工作、已确定决策、数据约定和任务证据，分别存放于 `status.md`、本页、
   `data.md`、`logs/` + `results/`。
3. 只有在已记录替代路径和相关测试后才能退役旧路径。被替代的细节可从 Git 历史恢复。
4. **C4 AMP：**CUDA FP16 AMP 仅由 `--use_amp` 显式开启，B7 默认仍为 FP32。V100 B7 batch-32 门禁的
   数值误差满足预注册门限，单步耗时/显存为 FP32 的 0.457x/0.949x；AMP 与 FP32 checkpoint 不得混合
   恢复，也不得据此提高 batch size 或声称准确率提升。详见 `logs/log-2026-09-21-c4-amp.md`。
5. **E5 评估边界：**生产 `evaluate_split` 的 SumNorm 物理谱评估与历史 `basemodel.test_one_step` 的
   raw-logit／M5／导出语义不等价，保持隔离；两者只共享 `utils.metrics.per_sample_spectral_metrics`。
   生产实验不得通过 legacy 评估入口报告结果。详见 `logs/log-2026-09-21-e5b-evaluation-boundary.md`。
6. **E6 分桶 batch：**`--use_bucket_batch` 仅在训练集上以固定 20-batch 窗口分桶并动态裁剪尾部原子
   padding，默认关闭。V100 Q1 batch-32 资源门禁的原子槽／单步耗时／峰值显存比为 0.171x/0.539x/1.000x；
   它改变 batch 组成和优化顺序，不能据此提高 batch size 或解释 accuracy 差异。详见
   `logs/log-2026-09-21-e6-bucketed-batches.md`。
7. **E7 lint/CI：**`bash tools/ci/check-static.sh` 是缓存无关的最小回归门禁：对所有受跟踪 Python 文件
   运行 Ruff 致命规则、编译并执行 42 项合成 CPU 合同测试；`.github/workflows/ci.yml` 在 push/PR 复现该
   命令。Q1 数据集成测试仍只在本地完整套件执行。详见 `logs/log-2026-09-21-e7-lint-ci.md`。
8. **B7 CIF 盲推理：**`b7_cif_infer.py` 是 B7 `_e9ctl` 的唯一 CIF 导出入口。它只接受 M1、seed 42、
   epoch 33 的严格 state dict，并以 Z0 `N_val(CIF)`、H1 gamma/eta 和 E0/P0 固定 bin 重建 blind 谱；
   不得以旧 M4 `cif2dos.py` 或标签尺度声明 B7 推理。详见
   `logs/log-2026-09-21-b7-cif-blind-inference.md`。
9. **D4 phDOS 标签形状审计：**以 Q1 train p90 固定的负频坐标质量阈值（0.129287）在 B7 test 划出
   230 条 high 样本，其 phDOS 失败率为 16.09%（other 2.09%，差 14.00pt、bootstrap 95% CI
   9.41–18.73pt）；尖峰集中度不满足门槛、coverage 外目标质量为零。此结论只授权原始来源／稳定性审计，
   不等同于虚频真值，也不授权标签处理或训练。详见
   `logs/log-2026-09-21-d4-phdos-spike-imaginary-audit.md`。

## 参考测量值

- **B7 oracle 测试集，Q1：**eDOS 中位 R² 0.518 / 失败率 5.73%；phDOS 0.741 / 3.50%；
  Cv 平均绝对误差 0.30。
- **B7 blind 测试集，Q1：**eDOS 中位 R² 0.480 / 失败率 8.66%；phDOS 0.735 / 4.20%。
  oracle–blind gap 的 p50/p90/p99：eDOS 为 0.010/0.150/0.826，phDOS 为
  0.001/0.035/0.559。

这些数值是证据，并非目标保证。完整精度见受版本控制的 B7 结果 CSV。
