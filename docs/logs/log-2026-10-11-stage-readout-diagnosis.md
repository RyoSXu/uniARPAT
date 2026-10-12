# 日志：2026-10-11 — 冻结模型逐阶段结构谱差可读性诊断（C1 主 / C2 参）

## 范围

- 任务与假设：补上一轮 pooled 探针的三个缺口，重新检验「同组成 eDOS 谱形差」在冻结模型
  **逐阶段**是否线性可读。三个缺口与本轮对应修正：①探针 `fit_intercept=True` 不保证交换材料
  反号 → **禁用截距**并显式验证反号/零差；②绝对组成只有 27 个 train 对 → 从**完整有效 train**
  重新配对（绝对 2,591 对/1,331 组、约化 7,837 对/1,791 组），只对参与配对的材料提取表示；
  ③decoder 在能量轴上 mean/RMS 池化丢失谱位置 → decoder 取 **eDOS decoder 逐能量 token
  `hs_edos [B,128,512]`**，用逐能量位置线性读出（每位置 512→1，共 65,536 参数）保留能量维度。
- 只读诊断：C1（主）/C2（参）`balanced_oracle_mae` best checkpoint（seed 42、35 轮），
  模型始终 eval、FP32、参数冻结；只拟合诊断用小读出。**只用 train/valid，不读 test**，
  不改变数据划分、不重训、不 Git 提交。
- 改动的文件（全部新增，不改 `CODE_FILES`，旧工具与报告原样保留）：
  [`stage_readout_diagnosis.py`](../../tools/eval/stage_readout_diagnosis.py)（独立补充入口）、
  [`check_stage_readout_diagnosis.py`](../../tools/checks/check_stage_readout_diagnosis.py)
  （独立复核：独立池化钩子、独立加权 ridge 求解、独立配对公式）、
  [`test_stage_readout_diagnosis.py`](../../tests/pipeline/test_stage_readout_diagnosis.py)
  （14 项测试），并在 [`check-static.sh`](../../tools/checks/check-static.sh) 注册测试。
  复用 [`lg_structure_diagnosis.py`](../../tools/eval/lg_structure_diagnosis.py) 的
  `pair_errors`/`paired_group_interval`/`loader`、
  [`local_structure_diagnosis.py`](../../tools/eval/local_structure_diagnosis.py) 的
  `grouped_mean_frame`/`run_directory`、
  [`lg_train_subset_diagnosis.py`](../../tools/eval/lg_train_subset_diagnosis.py) 的
  `same_composition_pairs`/`composition_keys_by_position`、
  [`local_structure_compare.py`](../../tools/eval/local_structure_compare.py) 的
  预检/完成态核对，以及 `lg_valid_export` 的 `load_run`/`select_best`。

## 执行与产物

独立 tmux 会话 `uniarpat_stage_diag_20261010`（2026-10-11 11:55 创建，调用结束后进程继续存活至
自然结束）；正式结果 `results/runs/stage_readout_diagnosis_c1c2_20261010/`（目录后缀沿用系列日期，
实际执行窗口 2026-10-11；V100、**1,884.7 s ≈ 31.4 min**、CUDA 分配峰值 **1,599.7 MiB**、
`device=cuda`、`seed=42`，见 `report.json`）：

```sh
PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/tefscloud/miniconda3/envs/torch/bin/python -B \
  tools/eval/stage_readout_diagnosis.py --out-dir results/runs/stage_readout_diagnosis_c1c2_20261010
PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/tefscloud/miniconda3/envs/torch/bin/python -B \
  tools/checks/check_stage_readout_diagnosis.py \
  --diagnosis-dir results/runs/stage_readout_diagnosis_c1c2_20261010 \
  --out results/runs/stage_readout_diagnosis_c1c2_20261010/verification_replay.json
```

正式运行前短测 4 轮（`stage_readout_diagnosis_c1c2_20261010_smoke*/`，各 79 材料、约 4 min）：
smoke 管线与独立复核 925 项检查先在短测上通过；期间修复 4 个实现缺陷并保留现场——
①decoder 批量 Gram 批次轴取错（`np.matmul` 沿样本轴批处理而非能量位置轴）；②CV 中逐 alpha
重复重建 Gram（正式规模不可承受，改为每折每 alpha 集合一次求解）；③设计矩阵构造顺序不一致
（float32 相减 vs 先缩放后作差，病态 Gram 下放大到 1e-6 失配，统一为先按 train 尺度缩放后作差，
使 CV 预测与逐对 CSV 数学同源）；④复核脚本按完整划分重建配对在 smoke 口径下不匹配
（manifest 增记 `split_scope`）。另确认 GPU 原子加法使重复运行特征有 ~1e-6 相对末位抖动
（与历史日志一致），复核因此一律从落盘特征重算拟合与指标，特征本身以容差比对。

校验与状态：`manifest.json` 记录源码、数据 manifest 与全部 train/valid 载荷、两个
`checkpoint_best.pth`（C1 best ep 34、C2 best ep 35，`balanced_oracle_mae` 选点）、
53 个产物文件的 sha256，材料 ID/组成组/配对数与 `split_scope`；
`report.json` 记录各阶段输入维度、读出参数量、正则网格与 CV 轨迹。两模型抽取前后
`state_hash` 一致、checkpoint sha 不变、钩子全部摘除；重复短测（同单线程设置）数值稳定，
GPU 特征重提取比对容差 2e-4 相对。**本轮无输出干预，无需恢复动作。**
全仓 `check-static.sh`（GPU 隐藏）通过：**312 项测试**、ruff/语法检查与仓库检查
（733 个候选文件）全部成功，退出码 0（`stage_readout_diagnosis_c1c2_20261010_check_static.log`）。
独立复核 `verification_replay.json` **959 项检查全部通过**：独立池化钩子重提取 16 项特征/
预测比对、独立加权求解重放 12 个阶段×口径的 CV 轨迹与选定 α、48 项 vs 零响应/模型差值、
逐对误差列与组均/CI/近零计数重算、fit_scope=train-only、材料顺序与配对表重建、
state_hash 与钩子摘除；最大失配仅为各容差上限的 5%。

## 证据

口径：`crystal_q1_v1` repaired、eDOS **单位和为 1 谱形空间**（blind/oracle 尺度不在本轮范围）；
主误差 `E_pair = 0.5 × Σ|预测谱差 − 真实谱差|`；**绝对组成**为主口径，约化组成为一致性检查；
组均误差＝组成组等权平均，置信区间按组成组重采样 2,000 次（seed 42），共享材料的配对不当独立样本。
全部结论限定 **seed 42、35 轮、MAE 选点**。近零目标谱差（TV < 1e-6）单独处理：train 各口径
1 对（TV 恰为 0），valid 0 对；主指标不含近零对。

### 1. 配对与数据身份（完整有效 train / 完整 valid）

| 划分/口径 | 配对数 | 组成组数 | 参与材料 |
|---|---:|---:|---:|
| train 绝对 | 2,591 | 1,331 | 3,115 |
| train 约化 | 7,837 | 1,791 | 4,751 |
| valid 绝对 | 320 | 170 | 420 |
| valid 约化 | 360 | 180 | 420 |

绝对配对是约化配对的子集；提取只覆盖参与配对的材料（train 4,751、valid 420），配对仍由完整
划分生成。重算的 valid 预测与上一轮 `local_structure_attribution_20261009/analysis/` 导出
逐材料一致（C1/C2 最大绝对差 3.1e-7 / 2.2e-7，GPU 末位量级），数据与 checkpoint 身份见 manifest。

### 2. 读出协议与正则选择（train 内分组 CV，锁定后评 valid）

无截距线性读出，交换材料严格反号、相同表示严格零差（逐项 0.0）；差分特征按 train 材料 RMS
缩放（折内拟合、不减均值）；配对按组成组等权（组内权重 1/|组|，均值 1）；正则网格
`1e-8 … 1e14` 含接近零响应的强收缩对照，边界只按 train 扩展。

| 阶段 | 输入维度/材料 | 读出参数 | C1 选定 α（绝对/约化） | 说明 |
|---|---:|---:|---|---|
| local_last（末次局部更新后完整外层原子态，mean+RMS 池化） | 1,024 | 131,072 | **1e22 / 1e22** | 两次扩展后仍在边界，CV 单调趋向零响应 |
| global_last（末层全局更新后原子态，同法池化） | 1,024 | 131,072 | **1e22 / 1e22** | 同上 |
| decoder_edos（eDOS decoder 逐能量 token） | 65,536（128×512） | 65,536 | **1e4 / 1e4** | 内点，非平凡解 |

C2 选择相同（1e22 / 1e4）。**事实**：原子阶段的 train 分组 CV 误差随 α 单调下降直至零响应极限
（C1 绝对 local CV：α=1e2 → 0.2767、1e4 → 0.2710、1e6 → 0.2700、1e8 → 0.2699619=零响应极限，
扩展至 1e22 不再变化）；decoder 阶段 CV 有真实内点极小（α=1e4：0.26884 vs 零响应极限 0.26996）。

### 3. 主结果（C1，绝对组成；组均 E_pair，[95% 组重采样区间]）

| 方法 | train 组均 | train skill | train 余弦中位 | valid 组均 | valid skill | valid 余弦中位 | valid vs 零响应 |
|---|---:|---:|---:|---:|---:|---:|---|
| 严格零响应 | 0.26997 | 0 | — | 0.27483 | 0 | — | — |
| local_last 读出 | 0.26997 | 0 | — | 0.27483 | 0 | — | 0.00000 |
| global_last 读出 | 0.26997 | 0 | — | 0.27483 | 0 | — | 0.00000 |
| decoder_edos 读出 | 0.26641 | +0.0051 | 0.0878 | 0.27739 | −0.0102 | 0.0138 | **+0.00256 [+0.00083,+0.00431]** |
| 原模型读出 | 0.25570 | +0.0311 | 0.1310 | 0.27957 | −0.0179 | 0.0435 | **+0.00474 [+0.00062,+0.00940]** |

（vs 零响应＝该方法减零响应的组均 E_pair 差，负＝更好；原子阶段读出经 CV 收缩后**恰为**零响应，
差值 0。目标 TV 中位 train/valid = 0.26584/0.26975；模型预测 TV 中位 = 0.01958/0.02379，
幅度比 0.074/0.088，与上一轮 7%–13% 一致。）

约化组成一致性检查（C1）：原子阶段同样收缩为零响应；decoder 读出 train 优于零响应
（−0.00349 [−0.00434,−0.00273]），valid **劣于**零响应（+0.00402 [+0.00189,+0.00635]）；
模型 valid vs 零响应 +0.00465 [+0.00046,+0.00951]。C2 同形状（decoder 读出 train −0.00653、
valid +0.00261 [+0.00012,+0.00528]）。**C2 与 C1 在 valid 各阶段差异均不可分辨**
（decoder 读出差 +0.00005 [−0.00138,+0.00156]；原子阶段读出两者同为零响应）。

valid 上「优于原模型」的形式差异：原子阶段读出减模型 = −0.00474 [−0.00940,−0.00062]，
decoder 读出减模型 = −0.00218 [−0.00582,+0.00107]。**但原子阶段读出即零响应，该差异只反映
模型自身比零响应更差**；按「先超过零响应才算成功预测结构差异」的判据，没有阶段读出成功。

### 4. train→valid 落差

decoder 读出是唯一非平凡读出，其 train 优势在 valid 反号：C1 绝对 train −0.00356
[−0.00456,−0.00263] → valid +0.00256 [+0.00083,+0.00431]；方向余弦 train 0.0878 → valid 0.0138
（模型：0.1310 → 0.0435）。组内同组成 train 配对多 8 倍（2,591 vs 320）仍未产生可迁移方向，
与上一轮「方向不可迁移」判读一致，且不能再归因于 train 对数太少。

### 5. 对上一轮探针结论的修正（适用范围）

- 上一轮「冻结表示线性探针 valid 不优于读出/零响应」的结论**在其三个缺口下不成立为强阴性**；
  本轮补齐缺口后结论在更强口径下**重申**：valid 上无阶段读出优于零响应。
  但适用范围仍只限于**线性读出 + 本文的汇总/逐位置协议**：原子阶段是 mean/RMS 有损汇总，
  **阴性线性探针不能证明完整表示没有信息**；decoder 逐能量读出也未测跨能量线性映射。
- 上一轮 C1−C0/C2−C0 的 valid 差值（+0.00474/+0.00393）与本轮「模型−严格零响应」逐位一致
  （C2 +0.00393 [−0.00215,+0.01186]），确认 C0≈零响应与本口径的连续性。
- 「pooled 摘要层面不支持后续利用不足」的表述应修正为：**在无截距、组等权、充分正则的
  协议下，逐阶段都读不出可迁移的谱差方向；信息被后续删除或被最终读出/监督浪费这两种
  简单故事都不符合观察形状**（见结论）。

## 结论

状态：**pending（诊断完成，阶段间定位不成立；根因仍偏向泛化/标签条件，未唯一确定）**。
全部限定 seed 42、35 轮、MAE 选点、repaired Q1、谱形空间；区间只表示组成组变动，
**不表示跨训练种子稳定**。

1. **valid 上无阶段读出优于严格零响应（事实）**：原子阶段读出被 train 内分组 CV 选到强收缩
   （α=1e22，扩展后仍在边界）即零响应；decoder 逐能量读出 valid 显著劣于零响应
   （+0.00256 [+0.00083,+0.00431]，绝对口径；约化 +0.00402 [+0.00189,+0.00635]）。
2. **没有阶段成功预测结构差异（事实）**：形式上原子阶段读出误差低于模型（−0.00474
   [−0.00940,−0.00062]），但其读出即零响应，只反映模型比零响应更差；decoder 读出与模型不可分辨。
3. **证据优先指向「仍无法定位为信息删除/读出浪费」，更支持「无可迁移的线性可读方向」（推断）**：
   若局部/全局阶段信息被后续混合破坏，形状应为原子阶段读出强而 decoder 弱——实际原子阶段连
   train 内都无；若 decoder 表示中的信息被最终读出/监督浪费，decoder 读出应在 valid 超过模型与
   零——实际与模型不可分辨且劣于零。decoder 表示的 train 可读性（唯一内点正则）不迁移。
   阶段间差异只解释为该读出协议下的可读性差异，不宣称信息被删除。
4. **幅度/方向判读延续上一轮**：预测幅度比 0.074–0.088，方向余弦 valid 0.014–0.044；
   唯一非平凡读出也呈现同样的 train→valid 落差。

## 交接

- **推荐下一项单独对照（等待用户决定后才启动，本轮不执行）：同组成组内留出对照。**
  - 做法：完整 train 按绝对组成分组，组内 ≥2 的 1,331 组/3,115 条（≥3 的 288 组/1,029 条），
    每组固定规则留出 1 条作 pseudo-valid，其余照常训练；C1 同配方（seed 42、35 轮、batch=32、
    lr=5e-5、MAE 选点、同初值协议）重训一臂，评估留出材料对的 eDOS 谱差组均 skill 与全量 valid。
  - 判断标准：组内留出配对 skill 显著 >0（组重采样 CI 不含 0）而全量 valid ≈0 → 覆盖/
    可辨识性为主因；组内留出也 ≈0 → 覆盖被排除，转向标签条件审计与配对差分监督/更强归纳偏置。
    本轮证据（同组成 train 对多 8 倍仍无可迁移方向）使该对照更有区分力：它直接检验
    「同组成且训练中见过配对双方」时结构响应是否可学。
  - 成本：单臂约 1.3–1.5 h V100 + 划分与分析约 0.5 h；**需要修改 train/valid 划分契约，须用户批准**。
- 次选（无需训练、约 10 分钟）：对 12+1 个严格匹配对做标签来源核查设计（磁态/计算设置元数据），
  估计同组成谱差的可达上限；本轮不自行执行。
- 更新：`docs/status.md`、`docs/index.md` 已记录本轮状态与上一轮探针结论的适用范围；
  未执行 Git 提交。
