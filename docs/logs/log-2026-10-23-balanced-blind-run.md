# 日志：2026-10-23 — 新版等权 blind 全谱联合训练

## 范围

- 完整 A64 联合 eDOS/phDOS；seed42、35轮、batch32、lr=5e-5、FP32，从头训练，仅 train/valid。
- 固定数据、mask、双任务失败数门槛、R²下降≤0.02保护线及预算。未读 test，不替换 I1/A64/LR2，不追加实验、不提交 Git。
- 检验对象是**尺度、loss 与选点构成的完整修正包**；不把结果单独归因于loss、总量头或Encoder。

## 证据

### 配置、实现与运行

- 短检 `results/runs/spectral_balance_preflight_20261023_final_v2/report.json`：`passed=true`；启动前核对52个源码SHA-256均一致，训练后复核仍全部一致。冻结协议只用有效train 18,704条拟合常数：`C_e=5.542719504890618`、`C_p=0.03628462802488803`；valid为2,313条。
- 训练命令取自 `results/runs/spectral_balance_20261023_final_v2/report.json.command`；显式设置 `PYTHONUTF8=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。`config_used.yaml`核实 `use_amp=false`、`skip_test_eval=true`、`scale_mode=per_atom`、`selection_metric=balanced_blind_energy`、I3/staged K2/G6；训练日志确认 test loader/自动test评估已跳过。
- 正常完成35轮。history连续覆盖ep1–35；重算整段selection contract后，eligible为ep26、29、32、33，按联合误差最小选中**best ep32**、score 0.284902333；`latest`为ep35且不合格，不作best替代。best/latest均与配置契约一致。
- 每轮盲误差、R²中位/P10、失败率、eligible、best/latest、耗时和峰值显存见 `results/runs/spectral_balance_20261023_final_v2/analysis/epoch_monitor.csv`；原始训练输出见 `training.stdout.log`，完整history见 `results/reports/history_m1_sbal_energy20261023_a64_s42_e35.csv`。仅使用现有 `lg_valid_export.py::export` 对完整valid重新评分并导出逐材料CSV与谱；导出器核验best契约、完整cohort和冻结分数。
- 选点保护线在ep32：eDOS失败183/2313（7.912%，上限196），中位R² 0.462119（下限0.461721）；phDOS失败63/2313（2.724%，上限99），中位R² 0.721804（下限0.720719）。

### 固定 A64 对照

`E_i,t = Δ_t × Σ_b(((ŷ_i,t,b − y_i,t,b)/N_i)^2) / C_t`。这里“归一化 blind 误差”指结构预测谱以冻结的train谱能量常数归一；“材料配对 bootstrap”指每次对同一组材料索引同时重采样两个模型和两任务。A64为原选中点ep31（旧规则`balanced_blind_r2`）；新模型ep32按`balanced_blind_energy`选中，两者原始选点规则不同。

| 模型（valid，n=2313） | eDOS mean E / P90 E | phDOS mean E / P90 E | eDOS中位/P10 R²；失败率 | phDOS中位/P10 R²；失败率 |
|---|---:|---:|---:|---:|
| 固定 A64 ep31 | 0.341748 / 0.671951 | 0.215207 / 0.453023 | 0.481721 / 0.047093；8.474% | 0.740719 / 0.309374；4.280% |
| 新包 best ep32 | 0.349311 / 0.655063 | 0.220494 / 0.451283 | 0.462119 / 0.054350；7.912% | 0.721804 / 0.327193；2.724% |

| 配对差值（candidate−A64） | 点差 | 95%区间 | 判读 |
|---|---:|---:|---|
| eDOS mean E | +0.007563 | [−0.002017, +0.016802] | 不确定 |
| phDOS mean E | +0.005287 | [+0.001005, +0.009501] | 恶化 |
| 等权 mean E | **+0.006425** | **[+0.001268, +0.011474]** | **恶化** |
| eDOS中位R² | −0.019602 | [−0.027729, −0.010445] | 恶化 |
| eDOS P10 R² | +0.007257 | [−0.019170, +0.034333] | 不确定 |
| phDOS中位R² | −0.018916 | [−0.026364, −0.010365] | 恶化 |
| phDOS P10 R² | +0.017818 | [−0.001704, +0.038341] | 不确定 |
| eDOS失败率（pp） | −0.562041 | [−1.556420, +0.389105] | 不确定 |
| phDOS失败率（pp） | −1.556420 | [−2.248163, −0.951146] | 改善 |
| eDOS P90 E | −0.016888 | [−0.052892, +0.033627] | 不确定 |
| phDOS P90 E | −0.001740 | [−0.025590, +0.019604] | 不确定 |

Bootstrap为2,000次、seed20261023、单位为材料；同一抽样索引供两模型、两任务和全部指标共用。材料顺序SHA为`a9dc110fcb17b2741c5ae9305ff4f27dbb9b94259dbac2998270d41ccf7ff64d`，索引矩阵SHA为`5217d058174e9955d7f137c987e8ef2a97bfc00f8541fe8621646964667a9230`。区间仅表示这两个**已选checkpoint**上的valid材料抽样不确定性，不包含训练seed波动、选点不确定性或valid反复使用偏差。

### 诊断与待核材料

- Oracle仅作诊断：A64→新包的oracle mean E分别为eDOS 0.320111→0.337782、phDOS 0.208709→0.214154；不参与选点或主结论。
- 同组成谱差仅作诊断。绝对组成有320个材料对/170组，约化组成有360对/180组。形状空间预测TV中位数（目标TV保持不变）在绝对组成eDOS为0.02599→0.02002、phDOS为0.04387→0.02732；详见`analysis/same_composition_diagnostics.json`，不据此推断总体优劣。
- 审计中的5条材料全部保留在正式2313条valid评分内。下表为每材料的等权单材料E；完整每任务误差、原`N_val`及组成参考值见`analysis/flagged_material_contributions.csv`。

| 材料 | eDOS E：A64→新包 | phDOS E：A64→新包 | 等权单材料E：A64→新包 |
|---|---:|---:|---:|
| mp-aaaaadad | 0.61445→0.57940 | 0.24911→0.23928 | 0.43178→0.40934 |
| mp-aaaaarbn | 0.80342→0.49480 | 0.21389→0.13414 | 0.50865→0.31447 |
| mp-aaacioqq | 1.01496→0.97570 | 0.07188→0.08870 | 0.54342→0.53220 |
| mp-aaacpjlg | 4.50239→4.54891 | 0.22801→0.33650 | 2.36520→2.44270 |
| mp-aaacrkqc | 0.84173→0.35800 | 0.07995→0.12626 | 0.46084→0.24213 |

五条材料对完整等权mean E的合计贡献为A64 0.0018633（0.669%）、新包0.0017038（0.598%）；仍全部计入正式分数。`mp-aaacpjlg`在两任务上合计误差上升，其他四条的合计误差下降。

## 结论

- 状态：**本轮训练与核验完成；新包不替换现有基线**。
- 在同一train归一化协议下，equal-task mean E的配对区间整体高于零，故本次valid结果判为**恶化**；eDOS mean E单项不确定，phDOS mean E恶化。两任务中位R²均恶化，phDOS失败率改善，eDOS失败率及P10/P90差异不确定。选点门槛是点估计保护，并不等价于配对差异无损。
- 这是完整修正包对旧A64原选中点的比较；旧A64按不同规则选点，不能单独归因loss、总量头或Encoder。单seed及材料抽样区间不能证明跨seed稳定性。5条待核材料和现有mask均未删改；未评估test，不自动替换I1/A64/LR2，不追加实验或提交。

## 成本与证据

- Tesla V100-SXM2-32GB；完整训练墙钟约15,938.8秒（4.43 GPU小时），35轮训练计时合计14,429.1秒，中位412.2秒/轮。history峰值已分配显存25,146 MiB；监测到的`nvidia-smi`占用最高31,442/32,768 MiB，未发生OOM。best/latest各约913 MB。训练末尾明确记录test评估跳过。
- 运行证据：`results/runs/spectral_balance_preflight_20261023_final_v2/report.json`、`results/runs/spectral_balance_20261023_final_v2/report.json`、`results/runs/spectral_balance_20261023_final_v2/training.stdout.log`、`results/reports/history_m1_sbal_energy20261023_a64_s42_e35.csv`、`results/runs/ablation_m1_sbal_energy20261023_a64_s42_e35/{config_used.yaml,checkpoint_best.pth,checkpoint_latest.pth}`。
- valid导出：`results/runs/spectral_balance_20261023_final_v2/analysis/{candidate_valid_samples.csv,candidate_valid_summary.json,candidate_valid_spectra.npz}`。
- 配对分析与逐轮监测：`results/runs/spectral_balance_20261023_final_v2/analysis/{paired_compare.py,paired_comparison.json,paired_bootstrap_differences.csv,epoch_monitor.csv,flagged_material_contributions.csv,same_composition_diagnostics.json}`。固定A64输入见`results/runs/local_internal_residual_20261021/analysis/A64_{summary.json,valid_samples.csv,valid_spectra.npz}`。

## 交接

- 新协议本组完成；不自动启动下一实验。保持I1及既有参考不变，是否将新协议作为后续基线由用户决定。
- `status.md`仅更新简短结论；本日志保留详细运行、指标与解释边界。

## Codex独立核验与谱形/总量拆解

本次仅重算已有train常数、history、checkpoint元信息及valid冻结预测，未做模型前向、训练或test评估。
52份预检源码哈希、准备报告身份和9份结果证据哈希匹配；配置、模型/数据契约与预检一致。
独立重算完整35轮history得到eligible ep26/29/32/33、best32/latest35；第32轮也是全部35轮中
联合mean E最低的一轮，并非仅因约束而错过本次训练中误差更低的轮次。
完整2313条valid的ID与目标一致，两任务所有R²<0标记一致；11项配对bootstrap区间全部复现。

额外作一次解析误差拆解。记每原子目标为x、预测为a q、sum(q)=1；a为逐箱密度和，
乘箱宽后才是窗口离散面积。对冻结谱形q，用标签求最优非负a*=dot(q,x)/dot(q,q)，则：

`E = (Δ/C) × [ ||a* q − x||² + (a − a*)² ||q||² ]`。

第一项是该谱形即使总量最优仍留下的误差；第二项是当前总量偏离此最优值的额外误差。
这是借助标签的诊断，不是blind输出，也不参与选点。

| valid mean E分量 | 固定A64 | 新包 | 新包−A64，配对95%区间 |
|---|---:|---:|---:|
| eDOS：冻结谱形下的最优误差 | 0.304748 | 0.316806 | +0.012059 [+0.004500, +0.019324] |
| eDOS：总量偏离最优的额外误差 | 0.037000 | 0.032504 | −0.004496 [−0.010588, +0.001532] |
| phDOS：冻结谱形下的最优误差 | 0.192669 | 0.202405 | +0.009735 [+0.006231, +0.013239] |
| phDOS：总量偏离最优的额外误差 | 0.022537 | 0.018090 | −0.004448 [−0.006199, −0.002883] |

重采样仍按材料配对2000次、seed20261023，与主分析相同ID排序和抽样索引；两分量逐材料相加
精确还原正式blind E。结果支持本轮退化涉及谱形，不能仅归于总量预测。
第二项依赖各自谱形的最优值，不能将其降低直接解释为总量头独立改进；
这些输出端诊断也不能在loss、Encoder、head耦合或优化动态之间作因果定位。

五条待核材料对联合误差变化的合计贡献为−0.000160，即它们整体降低了新包相对旧模型的差距，
不能用这五条解释本轮总体退化。仅作敏感性核查、暂不计这五条时，剩余材料mean E差仍为+0.006599；
正式指标保留全部材料，不据此改变标签契约。

下一步建议固定骨架与评价协议，优先隔离谱形监督和新总量接入的影响；另定对照后再训练。
当前结果不证明应恢复不合理的总量硬上限，也不证明应更换局部A公式、全局层数或直接延长预算。
