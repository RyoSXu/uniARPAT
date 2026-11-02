# 日志：2026-11-01 — LR2＋E＋S中段编排正式对照

## 范围

- 唯一干预：将第二次完整局部调用从 `L2` 移到 `G3` 后；候选顺序为
  `L1 G1 G2 G3 L2 G4 G5 G6`。对照为已完成的 staged LR2＋E＋S ep29，不重训对照。
- 仅 seed42、35轮、batch32、lr `5e-5`、FP32；保持 per-atom 归一化、loss、数据、选点、模型宽度和其他结构不变。仅训练 train、使用完整 valid 选点和诊断；未评估 test。
- 正式 run：`results/runs/ablation_m1_sbal_midpoint20261101_lr2_midpoint_s42_e35/`。

## 证据

### 准备报告、短检与训练

- 准备报告 SHA-256：
  `ae4875cb90bfc034f020e2da3993c50745122c71535a8e0c505d5a9ade7b4a0e`。
- 短检报告 SHA-256：
  `36161023212ecc22f90f333a95ed50d653e7023847bcb0deee88a318366738d0`；短检内的
  `preparation_sha256` 与准备报告实测哈希相同，`passed=true`。
- 启动前重算了短检绑定的56项源码和准备报告绑定的75项输入，均无哈希差异。
- 进程 PID 19804 于本地时间约16:27启动，20:57正常退出；中间无中断，无 OOM、NaN 或契约错误。
  每隔一段时间记录进程、训练日志、history 和 GPU 状态，共110条采样，见 run 目录 `monitor.jsonl`。
- GPU 为 Tesla V100-SXM2-32GB。训练history中各轮峰值已分配显存最大 `25135.19 MiB`；监控采样的
  nvidia-smi 已用显存最高 `31400 MiB`。history的35轮训练耗时合计 `14618.82 s`（不含valid轮末开销），
  墙钟约4小时30分。日志有一条 PyTorch scheduler `epoch` 参数弃用提示；训练继续且无数值异常。

### 实际执行命令

命令 argv 直接取自已核验的准备报告，未改配方：

```bash
/opt/tefscloud/miniconda3/envs/torch/bin/python /root/home/newstudy/uniARPAT/tools/train.py --epochs 35 --batch_size 32 --lr 5e-05 --seed 42 --tag _sbal_midpoint20261101_lr2_midpoint_s42_e35 --data_dir /root/home/newstudy/uniARPAT/data/crystal_q1_v1 --atom_feat z_only_proj --skip_test_eval --scale_mode per_atom --spectral_balance_config /root/home/newstudy/uniARPAT/results/runs/spectral_layout_20261101/protocol.json --spectral_shape_aux --local_route LR2 --local_layout midpoint --local_depth 2 --global_depth 6 --selection_metric balanced_blind_energy --q1_coord
```

运行环境保留准备日志记录的 `PYTHONUTF8=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
`OPENBLAS_NUM_THREADS=1`。stdout/stderr 写入 run 目录 `training.log`；原始启动信息在 `launch.json`。

### History、best 与完整 valid

- `results/reports/history_m1_sbal_midpoint20261101_lr2_midpoint_s42_e35.csv` 有连续第1–35轮、数值有限。
  History SHA-256：`2582ef5a92e644fdaba2030ab472387c25b6bd9a34a5bd250d0a20ff370d2f7b`。
- 按 `balanced_blind_energy` 从history重新选点，共19轮通过固定保护，其中联合E最小的 best 为 **ep24**，分数 `0.2703912144`。
  `checkpoint_best.pth` 的轮次、分数及模型／数据契约均与history和 `config_used.yaml` 一致；
  `checkpoint_latest.pth` 为ep35且契约相同，只用于恢复，不作为best。
  固定双任务保护通过：eDOS/phDOS失败数 **177/70**（上限196/99），中位R²
  **0.483277/0.743491**（下限0.461721/0.720719）。
- `tools/eval/lg_valid_export.py::export` 对ep24重新导出了完整valid 2,313条，并检查history选点、
  checkpoint与契约、valid数据契约和选中分数。完整产物：
  `analysis/candidate_valid_summary.json`、`candidate_valid_samples.csv`、`candidate_valid_spectra.npz`。
- 独立运行核验见 `analysis/verification.json`；无测试集评估，训练末尾日志明确记录
  `Test loader and automatic test evaluation skipped by request`。

### 与 staged ep29 的总体误差配对比较

以下均为完整valid、blind、每原子能量；R²是中位/P10，失败数按逐材料 R²<0 计。E 的中位/P90用于描述，
checkpoint仍只按预设的等权mean E和固定保护选取。

| 任务／布局 | mean E | median E | P90 E | R² median / P10 | 失败数／率 |
|---|---:|---:|---:|---:|---:|
| eDOS staged ep29 | 0.334862 | 0.159836 | 0.646775 | 0.491335 / 0.075186 | 174 / 7.52% |
| eDOS midpoint ep24 | 0.335965 | 0.163742 | 0.664901 | 0.483277 / 0.070399 | 177 / 7.65% |
| phDOS staged ep29 | 0.208967 | 0.125936 | 0.425845 | 0.743409 / 0.326770 | 72 / 3.11% |
| phDOS midpoint ep24 | 0.204817 | 0.129414 | 0.424737 | 0.743491 / 0.353577 | 70 / 3.03% |

材料配对bootstrap 2,000次、seed20261101，同一材料索引用于eDOS、phDOS和联合E：

| 差值 midpoint−staged | mean E差 | 95%区间 |
|---|---:|---:|
| eDOS | +0.001104 | [−0.005512, +0.007638] |
| phDOS | −0.004150 | [−0.007827, −0.000660] |
| 联合 | −0.001523 | [−0.005324, +0.002099] |

midpoint best通过固定双任务保护，但联合E差区间上界大于0；按预设判据不支持总体改善。
phDOS单任务E差的区间低于0，不替代联合主判据。此结果不触发采用，当前staged ep29工作点保留。

### 同组成signed谱形差（独立诊断，不用于选点）

在sum-normalized shape空间以有符号材料对差构造误差
`0.5*L1((pred_i-pred_j)-(target_i-target_j))`；按absolute/reduced组成配对，先在组成组内平均，
再以组成组为单位bootstrap 2,000次、seed42。eDOS目标差小于 `1e-6` 的配对按既有诊断口径排除；本次四项均为0条。

| 组成／任务 | 组数／材料对 | target TV median | predicted TV median staged→midpoint | zero-response skill staged→midpoint | 误差差 midpoint−staged [95%区间] |
|---|---:|---:|---:|---:|---:|
| absolute eDOS | 170 / 320 | 0.269746 | 0.032574→0.029184 | 0.09%→0.25% | −0.000452 [−0.002956, +0.001886] |
| absolute phDOS | 170 / 320 | 0.281847 | 0.048682→0.041065 | 2.19%→3.02% | −0.002524 [−0.006490, +0.001547] |
| reduced eDOS | 180 / 360 | 0.268463 | 0.029386→0.026552 | 0.00%→−0.02% | +0.000075 [−0.002236, +0.002462] |
| reduced phDOS | 180 / 360 | 0.264821 | 0.045823→0.038860 | 2.40%→3.02% | −0.001901 [−0.005787, +0.001880] |

四项 midpoint−staged 区间均包含0，未检出本候选改善同组成谱差。midpoint reduced-phDOS 相对零响应的
组误差差区间为[−0.018010, −0.000163]，是“优于零响应”的诊断；不能解读为优于staged，后者的配对区间含0。
材料对bootstrap区间条件于已选checkpoint，不代表训练seed方差或消除valid反复选型偏差。

### 归档与复核

- run目录保存 `launch.json`、`training.log`、110条 `monitor.jsonl`、最终35轮history和checkpoint。
- `analysis/` 保存best核验、完整valid summary/samples/spectra、四种组成/任务的两模型pair CSV、总体及同组成报告，
  以及run-scoped复算脚本 `analyze_layout.py`。
- `analysis/analysis_validation.json` 以导出样本和pair CSV独立重算，总体E的三个bootstrap区间、四个组成组误差区间
  均与归档报告逐值匹配；包括15个以上归档结果的SHA-256清单。
- 初次分析脚本执行在写完报告后，终端摘要打印处有一次变量名错误；修复打印逻辑后正常读取归档报告，独立复算验证通过，
  未重跑训练、未覆盖分析结果。
- 主要哈希：准备报告 `ae4875…7b4a0e`；短检 `361610…6738d0`；
  `checkpoint_best.pth` `e4d35f…f2ba334`；`checkpoint_latest.pth` `be49af…971674`；
  完整valid samples `ca8813…00ab87c`；spectra `74e1b3…77fc7c7`。

## Codex独立核验（2026-11-02）

在原analysis下新增隔离目录 `independent_codex_20261102/`，保存核验脚本、完整valid重新导出及
`verification.json`；未覆盖OpenCode产物、改模型／配方、重训或评估test。

- 逐项验证准备报告75项输入、短检56项源码，以及原分析归档的输入／产物SHA-256，全部匹配。
  实际命令、配置、共同初值和顺序契约一致；与staged对照的训练配置仅布局及同内容protocol路径不同。
- history连续1–35轮且数值有限；直接按固定失败数和R²保护重新筛选19轮，等权mean E最小为ep24。
  best与history一致，latest为ep35，保存优化器20,475步及完整训练RNG，数据／模型契约一致。
- 从best重新运行完整valid 2,313条，联合E为 `0.2703912195`，与原导出差 `+3.52e-9`；
  两任务失败数仍为177／70。比较使用既有聚合分数容差 `rtol=3e-6, atol=1e-7`。
  FP32逐点预测未要求位级相同，谱形最大绝对差eDOS `1.34e-7`、phDOS `7.75e-7`；目标逐点一致。
- 由保存的物理原谱、实际原子数独立重算FP64 E及R²，再按固定材料顺序与seed20261101重采样；
  总体三项区间与归档报告一致。直接从valid晶体组成重建absolute／reduced的320／360材料对，
  按组成组、seed42复算四项谱差区间，也全部吻合。
- 补齐五条待核电子谱材料的贡献诊断：仍全部纳入正式评分，对联合mean E差贡献 `−0.00011527`；
  其余材料贡献 `−0.00140805`，二者按完整valid分母相加为总差。该诊断不改变保护线、选点或不升级结论。

本次核验支持归档结果可复现；不证明跨seed稳定、其他顺序无效或结构谱差问题已解决。

## 结论

- 状态：**closed；不升级工作点**。
- 事实依据：midpoint ep24满足预设valid保护且完整valid重算通过；总体联合E材料配对点差虽为负，95%区间含0；
  四种同组成谱差口径的midpoint−staged区间也均含0。
- 因此，staged LR2＋E＋S ep29继续作为工作点。本次只约束seed42、指定配方和所测顺序，不说明其他顺序或seed。
  依设计不追加epoch、seed或顺序扫描；test未评估。

## 交接

- `docs/status.md` 与 `docs/index.md` 已更新；正式训练、完整valid和两类判读均链接到本日志与run归档。
- 不改配方、不重训对照、不执行Git提交。
