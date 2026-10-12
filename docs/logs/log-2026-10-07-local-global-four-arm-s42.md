# 日志：2026-10-07 — A/B+ × staged/alternating 四组首轮对照（seed=42）

创建日期：2026-10-07。状态：**pending**（选型未定，等待采用门槛决定）。
四组各 35 轮全部完成；仅使用 train/valid，未读取 test，未改 batch/lr/预算，未启用 AMP。
本日志只记录本对照的执行、结果与口径；执行方式优化证据见[几何缓存日志](log-2026-10-06-local-geometry-cache.md)。

## 范围

- 任务与假设：在共同初值 `local-global-init-v1` 下比较 A/B+ 局部路线 × staged/alternating 编排，
  以 `balanced_oracle_mae` 在 valid 上选点；假设为路线或编排对 eDOS 结构条件预测存在可辨差异。
- 改动的文件或配置：**无代码改动**（HEAD `de913f3`，工作区保持干净）。仅生成运行产物、原始日志与文档。
  四组 `config_used.yaml` 由既有入口写入 `runtime.local_global_execution` 执行记录。

## 证据

- 命令、标签、随机种子和 epoch 预算：

  ```bash
  python tools/eval/local_global_compare.py \
    --local_depth 2 --epochs 35 --batch_size 32 --lr 5e-5 \
    --selection_metric balanced_oracle_mae \
    --seeds 42 --tag_prefix s42sel20261005 --run     # 收尾以 --summarize 复读
  ```

  K=2（局部调用 2 次、内部局部块 4 个、全局注意力+FFN 各 2 次）、seed=42（单 seed）、
  batch=32、lr=5e-5、SumNorm/H1、`z_only_proj`、`q1_coord`、FP32。tag：
  `_lg_s42sel20261005_k2_{a,bp}_{staged,alternating}_s42`。
- A/staged 为恢复续跑：`checkpoint_latest` epoch=21 只读核对（239 参数张量 + optimizer 239 状态项 +
  python/numpy/torch_cpu/torch_cuda RNG + model/data contract，探针 `resume_probe.json`）后，
  按既有 resume 路径恢复权重、optimizer、RNG 并快进 scheduler，**第 22 轮起更换执行方式**（见下）；
  其余三组从共同初值起训 35 轮。
- 执行方式（各组 `config_used.yaml.runtime.local_global_execution`，A/staged `from_epoch: 22`，其余 `from_epoch: 1`）：
  关闭角度重计算（A 的 4 个 block 均 `false`；B+ 无对应 block，记为空表）、`pair_chunk=65536`（A）、
  `shared_geometry=true`、GPU 结构缓存开启（`cell-frac-content-v1`、lazy、上限 16 GiB = `min(16GiB,half_device)`）。
- 结果文件与测试：`results/reports/history_m1_lg_..._s42.csv`（每组 35 行）与
  `results/runs/ablation_m1_lg_..._s42/`（config/best/latest）；原始日志与探针
  `results/runs/local_global_compare_s42sel20261005_20261006/`（`queue_run_attempt1/2.log`、
  `summarize.json`、`resume_probe.*`、`post_crash_state.*`）。
- 指标（valid，各自 best epoch，`balanced_oracle_mae` 选点；中位数，fail 为 %）：

  | 组 | best ep | sel | eDOS blind R²/MAE/fail | eDOS oracle R²/MAE/fail | phDOS blind R²/MAE/fail | phDOS oracle R²/MAE/fail |
  |---|---|---|---|---|---|---|
  | A/staged | 32 | 0.6620 | 0.4616 / 1.3665 / 11.28 | 0.5012 / 1.3169 / 8.26 | 0.7142 / 0.00708 / 4.84 | 0.7277 / 0.00703 / 4.32 |
  | A/alternating | 34 | 0.6607 | 0.4542 / 1.3615 / 11.11 | 0.5007 / 1.3143 / 8.13 | 0.7108 / 0.00719 / 4.89 | 0.7226 / 0.00697 / 3.89 |
  | B+/staged | 25 | 0.6652 | 0.4547 / 1.3802 / 10.94 | 0.4977 / 1.3234 / 7.39 | 0.7092 / 0.00715 / 4.76 | 0.7204 / 0.00704 / 4.02 |
  | B+/alternating | 31 | 0.6581 | 0.4564 / 1.3745 / 11.63 | 0.5018 / 1.3093 / 7.39 | 0.7139 / 0.00710 / 5.14 | 0.7241 / 0.00697 / 4.15 |

  数据口径：crystal_q1_v1 修复版（18,704 / 2,313 / 2,287），valid 集 2,313 条；blind/oracle 拆分。
  泛型 `r2_*`/`mae_*` 列经核对与 oracle 列逐值相同。
- 路线、编排与交互（单 seed，差值均小于或接近轮间波动）：
  - 路线：A vs B+ 的 sel 差值在两种编排下符号相反（staged +0.0033、alternating −0.0025），
    eDOS oracle R² 中位四组极差仅 0.0041，未见可辨路线效应；B+ 整轮耗时约为 A 的 0.39 倍。
  - 编排：alternating 的 sel 在两条路线下均略低（A −0.0013、B+ −0.0071），方向一致但幅度小。
  - 交互：仅见上述幅度差异，不能与种子/轮间噪声区分。
- 耗时（CSV `epoch_time_s` 只计训练段；整轮墙钟含 valid 与 checkpoint，自日志时间戳）：
  - A×2（热缓存）：训练段 410.7–415.9 s，整轮墙钟约 455 s；B+×2（热缓存）：159.5–161.7 s，约 177 s。
  - 首轮缓存填充分开计：B+ 冷轮 299.1 / 291.1 s（填充成本约 +131–138 s，逐百步均匀变慢与惰性填充一致）；
    A 冷轮（A/staged 22 轮、A/alternating 1 轮）410.6 / 411.5 s，与热轮无可辨差异，填充被较重局部计算掩盖。
  - **A/staged 分段口径**：1–21 轮为旧执行（训练段 1415–1433 s，中位 1417.5），22–35 轮为新执行（410.6–415.9），
    两段不可混算；队列墙钟 attempt1（含 14+35+35 轮）约 7 h55 min，attempt2（35 轮）约 1 h44 min。
- 显存（逐轮 `peak_vram_mb`，PyTorch 分配口径）：A 22.5–24.2 GiB（缓存开启后较旧执行 4.2–4.4 GiB 增约 15–16 GiB，
  与 15.28 GiB 全量缓存口径吻合）；B+ 7.2–8.1 GiB。B+ 的缓存静态量远小于 A（推断：缓存内容随路线的共享几何需求不同，
  未独立证实）；nvidia-smi 读数含分配器保留显存，不作跨阶段占用比较。
- 异常与中断（保留现场、按用户决定重跑，未改任何参数）：attempt1 在第四组启动时于
  `tools/train.py:468` 读取 `tools/training-defaults.yaml` 抛 `UnicodeDecodeError('ascii')` 退出。
  事实：该文件含 UTF-8 长破折号且自 10-04 未改动；同一进程前三组读取成功；新进程默认编码为 UTF-8。
  推断：长命进程默认文本编码中途降为 ASCII（等价于进程内 C locale 被切换），触发点未定位，未证实。
  对前三组产物无影响；attempt2 原命令重跑，前三组 resume 空转跳过（`range(35,35)`，无 checkpoint/配置改写），
  第四组从头训练完成。

## 结论

- 状态：pending（四组完成，采用门槛与路线/编排选型未定）。
- 单 seed=42 结果：四组 `balanced_oracle_mae` 0.6581–0.6652，eDOS oracle R² 中位 0.4977–0.5018，
  phDOS oracle R² 中位 0.7204–0.7277，组间差异均在轮间波动量级，不能证明任一组合更优。
  **单 seed 不能宣称跨初始化稳定；A/staged 第 22 轮起更换执行方式（保存激活/65536 分块/共享几何/缓存），
  其 1–21 与 22–35 轮成本分段报告，指标轨迹未受影响（同一 RNG/optimizer 连续恢复）。**
- 历史参考（仅参考、不可直接比较）：ZP100 `_eidzproj100_s42` best ep89（100 轮预算）eDOS R²/MAE/fail
  0.5246/1.2592/9.47、phDOS 0.7099/0.00698/6.14、balanced 0.6331；B7 `_e9ctl` best ep33（35 轮预算）
  eDOS 0.5150/1.3197/6.27、phDOS 0.7438/0.00680/3.80、balanced 0.6633。
  **口径差异**：历史为原 Q1 数据（未修复版），新四组为 crystal_q1_v1 修复版；历史无 blind/oracle 拆分；
  ZP100 预算 100 轮、B7 35 轮；历史执行方式为旧实现（无缓存/小分块，训练段约 189 s）。
  同 35 轮预算下新四组 balanced 与 B7 同量级（0.6581–0.6652 vs 0.6633），eDOS/phDOS R² 中位略低于 B7，
  但受数据口径差异影响，不作优劣结论。

## 交接

- 下一项关卡工作：确定采用门槛与路线/编排选型（是否需要多 seed 复核）；eDOS/phDOS 采用判据仍待决定。
- 对 status、backlog 和 decisions 的更新：`status.md` 更新为四组首轮完成、记录分段口径与异常；
  `index.md` 增本日志入口。历史参考口径差异已在本日志注明。
