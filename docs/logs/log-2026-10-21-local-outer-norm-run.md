# 日志：2026-10-21 — 完整 A 外层归一化对照 LN2

状态：**35轮完成并独立核验；LN2未达采用条件，继续保留A64参考，不自动替换I1**。
对应[设计](../design/design-2026-10-21-local-outer-norm.md)与[准备记录](log-2026-10-21-local-outer-norm-preparation.md)。

## 范围与运行

只改变完整A两次局部调用的512维外层LayerNorm位置：A64参考为 `h + L(LN512(h), R)`，LN2为 `LN512(h + L(h, R))`；四个64维内部块继续post-norm。使用认证的A64结果和冻结train/valid导出，不重训A64，也不读取A64/LN1训练权重初始化候选。

固定条件：`crystal_q1_v1` train/valid 18,704/2,313；联合eDOS/phDOS，seed42、35轮、batch32、lr5e−5、FP32；valid两任务blind中位R²等权最大选点。候选 `init_ckpt` 为空，初始化协议 `local-outer-norm-init-v1`，287个张量与A64共同未训练初值逐值一致，SHA256为 `2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。实际执行记录四个内部块及两次外层调用均为post-norm。

执行命令：

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_outer_norm_compare.py \
  --tag-prefix outer20261021 \
  --preflight results/runs/local_outer_norm_20261021/preflight_final.json \
  --out results/runs/local_outer_norm_20261021 --run
```

监督器PID 1318、对照入口PID 1319、训练进程PID 1798。开始于 **2026-10-21 10:13:18 +08:00**，结束于 **14:47:28 +08:00**，总墙钟 **4:34:10**；队列状态 `complete`、退出码0。每5分钟监测，共54个快照；无OOM、Traceback、NaN/Inf或监测异常。训练日志记录test loader和自动test评估均跳过；没有读取test。

前三轮ETA记录为15:39。该估算偏保守：监测器把包含验证/检查点开销的墙钟轮速与额外诊断预留相加，重复计入部分轮次开销；实际全流程14:47:28结束。该偏差未改变训练配置或预算。

## 完成核验

- `history`恰有epoch 1–35且连续；按每轮 `0.5 × eDOS blind中位R² + 0.5 × phDOS blind中位R²` 重算，最大值 **0.6142674088** 出现在ep31。`checkpoint_best.pth`为ep31且分数匹配；`checkpoint_latest.pth`为ep35。
- seed、轮数、batch、学习率、`balanced_blind_r2`及LN2契约均与预检一致；完整分析报告 `analysis_complete=true`。
- A64的配置、best/latest checkpoint、history及冻结导出哈希均未改变。参考best为ep31、latest为ep35；未重训参考。

## 主结果：完整 valid、blind

失败率按单材料blind R²<0计算；样本数均为2,313。

| 模型/选点 | eDOS中位R² | eDOS失败率 | phDOS中位R² | phDOS失败率 | `balanced_blind_r2` |
|---|---:|---:|---:|---:|---:|
| A64，ep31 | 0.481721 | 8.4738%（196） | 0.740719 | 4.2802%（99） | 0.611220 |
| LN2，ep31 | 0.484188 | 8.6035%（199） | 0.744347 | 4.4099%（102） | 0.614267 |
| LN2−A64 | +0.002467 | +0.1297 pp | +0.003627 | +0.1297 pp | **+0.003047** |

联合差值的材料配对bootstrap 95%区间为 **[−0.001024, +0.007236]**（2,000次、seed20261016、两任务共用材料抽样索引），包含零。两任务R²下降≤0.02及失败率增加≤1 pp保护线均通过；但联合区间下界不大于零，**未达设计采用条件**。区间条件于选定checkpoint，不涵盖训练seed不确定性。Oracle仅为诊断；不自动替换I1、不追加训练。

## 同组成谱差诊断

下表为组成组等权的差异谱误差变化 `LN2−A64`；负值表示LN2误差较小，正值表示较大。eDOS近零目标差对（`target_tv < 1e−6`）按既定规则排除。区间用2,000次组成组bootstrap、seed42；仅作诊断，不参与选点，也未作多重比较校正。

| 划分 | 组成分组 | 任务 | 组成组数 | 平均误差变化 | 95%组成组区间 |
|---|---|---|---:|---:|---:|
| train | absolute | eDOS | 1,330 | −0.000077 | [−0.000386, +0.000234] |
| train | absolute | phDOS | 1,331 | +0.000138 | [−0.000744, +0.001001] |
| train | reduced | eDOS | 1,790 | −0.000092 | [−0.000344, +0.000146] |
| train | reduced | phDOS | 1,791 | +0.000193 | [−0.000525, +0.000909] |
| valid | absolute | eDOS | 170 | **+0.001951** | **[+0.000294, +0.003969]** |
| valid | absolute | phDOS | 170 | +0.001136 | [−0.001891, +0.003985] |
| valid | reduced | eDOS | 180 | +0.001616 | [−0.000001, +0.003540] |
| valid | reduced | phDOS | 180 | +0.001619 | [−0.001276, +0.004403] |

未检出train组成谱差改善；valid绝对组成eDOS误差增加的区间不含零，其余valid区间包含零。此诊断不足以定位机制或替代主判据。

## 成本与证据

- history中每轮训练计时中位数411.98秒、合计14,433.92秒；PyTorch记录的峰值分配显存25,151.63 MiB（24.56 GiB）。5分钟间隔的54次 `nvidia-smi` 采样最高31,424/32,768 MiB；这是采样峰值，不是连续监测得到的精确瞬时峰值。
- 启动预检的学习率轨迹探针产生PyTorch scheduler顺序/弃用warning；来源为预检调用轨迹探针，不是训练异常。训练日志无OOM、Traceback或非有限值。
- 原始结果与监测记录：`results/runs/local_outer_norm_20261021/analysis/report.json`、`run_control.json`、`monitor.log`、`runner.log`、`ln2_train.log`；训练history及best/latest位于对应 `results/reports/history_m1_lon_outer20261021_post_k2_g6_staged_s42_e35.csv` 和 `results/runs/ablation_m1_lon_outer20261021_post_k2_g6_staged_s42_e35/`。
- Codex追加独立核验：`analysis/verification_codex.py`及`verification_codex.json`，`passed=true`。
  认证69份当前源码、63份历史源码、35轮选点及实际归一化记录；29份输入产物保持不变。
  从保存谱数组复算物理指标及168,136个失败标志，并独立核对16份配对表、24个组成组区间、12个材料区间和联合区间；
  联合区间误差为0、物理指标中位数最大误差`3.98e−7`、配对量最大误差`1.43e−14`，均通过。
  本轮核验未执行模型前向、参数更新或test加载，未执行Git提交。

## 结论与后续

本轮仅能说明：在固定单seed、35轮配方下，LN2的联合blind点估计高于A64，但材料区间仍含零；valid绝对组成eDOS谱差诊断偏差不利。**不满足采用条件，保留A64局部参考和I1历史性能参照**；不自动追加实验或替换I1。残差连接与全局头数/FFN仍未完成选型，不能据此断言总体最优或归因退化机制。
