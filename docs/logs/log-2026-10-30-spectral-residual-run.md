# 日志：2026-10-30 — LR2迁移E＋S正式运行与核验

对应[单候选设计与冻结判据](../design/design-2026-10-30-spectral-residual.md)及[准备与短检](log-2026-10-30-spectral-residual-preparation.md)。

## 范围与冻结条件

- 唯一候选为LR2迁移当前E＋S配方：seed42、35轮、batch32、lr=`5e-5`、FP32；保留完整A消息，只关闭四个内部加法。
- 对照复用已完成的A64 E＋S ep25；不加载旧训练权重、不重训对照、不追加候选/seed/轮数。
- 使用准备报告中的原始command和protocol；`init_ckpt`为空，`skip_test_eval=true`。best按`balanced_blind_energy`及冻结的双任务保护选择，latest只用于恢复。
- 仅valid用于选点和比较；本轮没有test评估。五条待核材料仍纳入2,313条正式评分。

## 启动、契约与完整性

- 启动前准备报告73项`input_sha256`、短检54项`source_sha256`全部一致；短检绑定的准备报告SHA-256为
  `01d9c59467b8bf3b0033e85f412fee2e1df89ac54746cfaa3c558fafd7bff24f`，与实际报告一致。训练结束后复核仍为73/73、54/54。
- 执行的argv与`results/runs/spectral_residual_20261030/report.json::command`逐项相同：

  ```text
  /opt/tefscloud/miniconda3/envs/torch/bin/python /root/home/newstudy/uniARPAT/tools/train.py --epochs 35 --batch_size 32 --lr 5e-05 --seed 42 --tag _sbal_lr2es20261030_lr2_s42_e35 --data_dir /root/home/newstudy/uniARPAT/data/crystal_q1_v1 --atom_feat z_only_proj --skip_test_eval --scale_mode per_atom --spectral_balance_config /root/home/newstudy/uniARPAT/results/runs/spectral_residual_20261030/protocol.json --spectral_shape_aux --local_route LR2 --local_layout staged --local_depth 2 --global_depth 6 --selection_metric balanced_blind_energy --q1_coord
  ```

- 环境：`PYTHONUTF8=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。
- 正式训练进程PID `11126`，监测supervisor PID `11122`。启动时间`2026-10-30T02:45:12Z`（本地10:45:12），结束时间`2026-10-30T07:15:31Z`（本地15:15:31），退出码0。
- history连续覆盖ep1–35，全部数值有限；eligible为ep18–35。`checkpoint_best.pth`为ep29、score=`0.2719145269`；`checkpoint_latest.pth`为ep35。best与冻结history选点及模型契约一致；latest含optimizer与RNG状态。正式状态为selected，不是`no_eligible_best`。
- 运行无OOM、非有限值、异常栈或监督器告警。ep5日志有一条PyTorch `scheduler.step(epoch)`弃用警告，未中断运行。日志确认test loader及自动test评估按请求跳过。
- 启动时依照当时的执行指令使用了脱离会话的supervisor。后来收到“不要脱离会话”的补充要求时，没有向进程发送信号或重启；之后核验发现它已自然完成。valid导出和分析均以前台命令执行。

## 训练成本与逐轮记录

- Tesla V100-SXM2-32GB。35轮`epoch_time_s`合计14,637.9秒（4.07 GPU小时），中位417.8秒/轮；进程wall time 16,219秒（4小时30分19秒）。
- history记录PyTorch峰值分配25,146.0 MiB；538次约30秒间隔的`nvidia-smi`采样峰值31,442/32,768 MiB。采样显存包含缓存/保留内存，与PyTorch分配峰值不是同一口径。
- 首轮实测414.98秒后估计余34轮约14,109秒；正式逐轮loss、valid指标、epoch耗时和峰值分配在history及`execution/epoch_observations.jsonl`，GPU采样和完整stdout分别在`execution/gpu_samples.jsonl`、`execution/train_full.log`。

## 完整valid与主结果

使用现有`tools.eval.lg_valid_export.export`导出正式best ep29的完整valid。候选与A64 E＋S ep25均为2,313条；材料ID及顺序一致，eDOS/phDOS物理标签及归一化标签逐项一致（最大绝对差均为0），原子数与完整valid结构及`n_atoms_valid.npy` sidecar一致。

blind指标如下；E越低越好，R²越高越好：

| 模型／轮次 | 任务 | mean E | median E | P90 E | median R² | P10 R² | 失败数／率 |
|---|---|---:|---:|---:|---:|---:|---:|
| A64 E＋S／25 | eDOS | 0.341684 | 0.164856 | 0.640704 | 0.489122 | 0.084454 | 171／7.393% |
| LR2 E＋S／29 | eDOS | 0.334862 | 0.159836 | 0.646775 | 0.491335 | 0.075186 | 174／7.523% |
| A64 E＋S／25 | phDOS | 0.214229 | 0.129866 | 0.453855 | 0.735590 | 0.339375 | 74／3.199% |
| LR2 E＋S／29 | phDOS | 0.208967 | 0.125936 | 0.425845 | 0.743409 | 0.326770 | 72／3.113% |

联合mean E为A64 E＋S `0.27795645`、LR2 E＋S `0.27191453`，差值方向为LR2−A64，点差`−0.00604192`。
材料配对bootstrap 2,000次、seed `20261030`，95%区间`[−0.01163879, −0.00102599]`。同一抽样矩阵用于eDOS、phDOS及联合指标；所有上述指标的差值区间均保存在`analysis/primary_comparison.json`。

冻结保护以历史A64 ep31为阈值：eDOS失败数`174≤196`、median R² `0.491335≥0.461721`；phDOS失败数`72≤99`、median R² `0.743409≥0.720719`，均通过。eDOS mean E差`−0.006822`、区间`[−0.016970,+0.002643]`含0；phDOS mean E差`−0.005261`、区间`[−0.009479,−0.001251]`低于0。部分尾部指标并未一致改善，详见完整bootstrap结果。

按预设规则，联合mean E区间上界严格小于0且双任务保护通过，**支持LR2在本次运行中改善主目标**；这不自动采用，工作参考仍为A64 E＋S，待用户决定是否并入。

## 仅作诊断的结果

- Oracle mean E（eDOS/phDOS）：A64 E＋S `0.329052/0.207130`，LR2 E＋S `0.320680/0.202628`。不参与选点或主判断。
- 同组成signed谱差使用既有绝对/约化组成分组；先按组成聚合，再按组成组配对bootstrap 2,000次、seed42，不把组内相关材料对当独立样本。差值为LR2−A64，负值表示组均差异误差较低：

  | 分组／任务 | 材料对／组成组 | 组均误差差 | 95%组bootstrap区间 |
  |---|---:|---:|---:|
  | absolute／eDOS | 320／170 | −0.002325 | [−0.005041,+0.000537] |
  | absolute／phDOS | 320／170 | −0.004154 | [−0.009430,+0.000802] |
  | reduced／eDOS | 360／180 | −0.002451 | [−0.004926,+0.000269] |
  | reduced／phDOS | 360／180 | −0.004619 | [−0.009642,+0.000145] |

  四个区间均包含0；本诊断不构成同组成谱差改善证据，也不改变主结论。
- 五条待核材料全部保留在正式valid评分中。其联合mean E贡献之和：A64 E＋S `0.00135752`（完整mean E的0.4884%），LR2 E＋S `0.00163086`（0.5998%）；差为`+0.00027333`。本次总体改善并非由这五条材料带来。

## 证据边界与文件

- 主要运行记录：`results/runs/spectral_residual_20261030/execution/{launch.json,execution.json,verification.json,first_epoch_eta.json,epoch_observations.jsonl,gpu_samples.jsonl,train_full.log,end_time.txt,exit_code.txt}`。
- 训练产物及history：`results/runs/ablation_m1_sbal_lr2es20261030_lr2_s42_e35/{config_used.yaml,checkpoint_best.pth,checkpoint_latest.pth}`、`results/reports/history_m1_sbal_lr2es20261030_lr2_s42_e35.csv`。
- valid导出与分析：`results/runs/spectral_residual_20261030/analysis/{candidate_valid_samples.csv,candidate_valid_summary.json,candidate_valid_spectra.npz,cohort_check.json,cohort_ids_atoms.csv,primary_comparison.json,diagnostics_only.json,same_composition_pairs.csv,flagged_material_contributions.csv,analysis_inputs.json,analysis_report.json,paired_lr2_analysis.py}`。
- 配对bootstrap区间条件于已选checkpoint，不能表示跨seed训练波动，也不消除valid反复选型偏差。本轮是单seed、单候选；Oracle、同组成谱差和五条材料仅为诊断。没有评估test，也没有追加训练。
- 结论：**通过本次预设改善判据；是否把LR2并入当前工作配方由用户决定。**未改训练源码，未执行Git提交。

## 独立核验（Codex，2026-10-30）

从正式checkpoint和原始导出重新核验，没有直接复用本轮分析脚本的计算结果：

- 73项准备输入、54项短检源码及23项分析输入指纹均匹配；protocol与A64 E＋S对照逐字相同。
- 从准备配置重建模型契约，与运行配置、短检契约一致；实际生产参数相对对照只改`local_route`。
  记录的共同初值指纹相同；`init_ckpt`为空、FP32、跳过test。
- 独立用每轮失败数和R²保护重建资格：ep18–35合格，最小联合E为ep29；与正式best一致。
  latest为ep35，同模型／数据契约，含完整RNG和优化器状态，AdamW步数20,475＝585批×35轮。
- 由best ep29再次前向完整2,313条valid，联合mean E为0.2719145247，与原导出相差−5.22×10⁻⁹；
  两任务失败数仍为174／72，标签完全相同，按既有选点与导出容差通过。
  两次FP32前向的原始输出并非逐位相同：eDOS物理谱最大绝对差约4.58×10⁻⁴，
  两任务谱形最大绝对差约2.09×10⁻⁷／9.54×10⁻⁷；未把浮点复跑解释为逐位相同。
- 直接由物理谱、真实原子数与冻结train常数，用NumPy FP64复算E及R²；材料配对2,000次、seed20261030，
  联合差−0.0060419185，95%区间[−0.0116387892,−0.0010259905]，相对下降2.1737%。
  eDOS／phDOS均值差分别−0.0068224756／−0.0052613615，与归档结论一致。
- 从完整valid结构重新生成绝对／约化组成配对，逐谱计算signed差异误差，再按组成组bootstrap，
  复现320对／170组、360对／180组及四个区间，均包含0。
- 5条待核材料对本次联合E差贡献为+0.0002733313，因此总体改善并非由它们带来。
  全部保留在正式评分中；去除5条只作诊断，剩余材料均值差为−0.0063289310。
- history成本及538条GPU采样与日志一致；训练stdout只有一次启动记录，未见恢复、OOM或异常栈，
  最终明确跳过test评估。没有新增训练。

独立产物：`results/runs/spectral_residual_20261030/analysis/independent_codex_20261030/`
中的`verification.json`、`verify_independent.py`及`valid_{samples.csv,summary.json,spectra.npz}`。
本轮仅同步运行日志、状态、设计与路线图；不改变工作参考或checkpoint选择。

## 采用决定（2026-10-31）

用户在独立核验汇报后确认更新并提交：当前研究工作点采用完整A64的LR2形式＋E＋S，
关闭四个内部显式加法，保留内部post-norm、外层pre-norm残差、完整状态读出与其余已冻结设置。
参考运行是`ablation_m1_sbal_lr2es20261030_lr2_s42_e35`，best ep29，模型契约为
`joint-spectral-balance-residual-model-v1`，冻结protocol为`results/runs/spectral_residual_20261030/protocol.json`。

原A64 E＋S ep25保留匹配对照，I1保留历史性能参照；双任务保护仍基于旧A64 ep31的196／99条失败上限
与各自中位R²下降不超过0.02，不因工作点更新而重设。选点、loss、数据及既有checkpoint契约均不重写。
研究入口须显式指定LR2及E＋S相关开关，历史默认保留。

采用仅基于本次单seed总体误差改善，接受已报告的eDOS失败多3条和两任务R² P10点估计下降，
不宣称同组成谱差已改善或跨seed稳定。当前固定局部模块工作点，不追加局部公式对照；
下一阶段训练须围绕尚未解决的结构谱差另定问题及协议。
