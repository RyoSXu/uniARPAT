# 术语表（第一次出现就按这里写）

## 口径（每次报数字必带）
- `pre-Q`：旧缓存 20040/2477/2471，γ地板0.002。B5/B6/H1/S1 均为此口径，存档勿引为现行。
- `Q1`：干净池 18706/2313/2287，γ地板0.106+。B7 起一律此口径。注意 `Q1-quarantine`（数据隔离1682出池）与 `Qc1/Qc2`（模型坐标臂，`--q1_coord/--q2_fourier`）是两件事，禁止裸写 Q1。
- `oracle/blind`：oracle 用真值和槽逆归一化，blind 用 η/γ 头自预测。`gap = oracle_R² − blind_R²`，报 p50/p90/p99。
- 数字模板：`e med/fail + p med/fail + (test|valid, epN, oracle|blind, Q1|pre-Q)`。例：`e 0.518/5.73% (test, best ep33, oracle, Q1)`。`pt/pp` 表百分点。

## 五态（不合并的5种说法统一）
- 合并 win：med 涨超 draw 线且 fail 不恶化，进默认。
- park：打平，代码+单测留存，默认 off（如 Q1/Q2）。
- 挂起：待长跑确认（如 G1 pilot 后）。
- 死刑：机制证伪，永不重做（如 C2.3 loss 掩膜、Eg 回归）。
- 关闭：暂不做可重开（如 Q 组关闭）。

draw 线：`|Δmed| < 0.02 且 |Δfail| < 1pt` 视为打平。

## 人话缩写
- med/fail：中位 R² / R²<0 失败率。mean R² 停用。
- ctl/exp：对照臂/实验臂，必带后缀如 `ctl_B7-35`、`ctl_Q1-10`。
- NBANDS 截断：MP-DOS 任务能带数不足，全谱电子数 <50%，eDOS 形状毒药，见 Q1 D1–D4。
- E_F 错位：窗内仅 0.01~0.1 e⁻/原子的尾巴样本，盲 gap p99 主因，转 S1/Eg 队列。
- 和槽/Δ：盒平均之和×Δ 还原面积；Δ_e=0.09375 eV，Δ_p=19.6875 cm⁻¹。
- balanced-score：选 best ckpt 的 valid 综合分，公式见 runner，跨轮数不可比。
