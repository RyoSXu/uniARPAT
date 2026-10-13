# O0/O1 旧全局骨架归因：实施与训练前检查

日期：2026-10-12。用户已确认先实施入口与短测，再交由 OpenCode 正式训练。
问题、初值、判读及成本边界见[设计](../design/design-2026-10-12-legacy-backbone-attribution.md)。

状态：新增两组入口、真实 V100 预检与全仓检查已通过，尚未启动正式训练。
保留本轮开始时已有 `docs/status.md`、`docs/index.md` 与 G0 执行日志改动；没有提交 Git。

## 调用链与变更

- 模型：`LocalGlobalEncoder(O0/O1)` → `LegacyContentGlobalLayer` →
  `TransformerEncoderLayer(geometry_scoring=False)`；保留原 `q=k=v`、FFN、post-norm，
  无 `rp_proj/RPEncoding`。O1 的局部层与几何准备/cache 复用 C1。
- 初值：`initialize_comparison_model` → `initialize_legacy_backbone_model`；
  保留原 baseline 的未训练张量，O1 的全部局部张量复制未训练 C1，记录摘要及独立契约。
- 执行：`legacy_backbone_compare` 只训练 O0/O1；复用 G0/C1/原 baseline，沿用选点、
  checkpoint/optimizer/RNG 恢复与异常停止。默认只打印命令。
- 分析：五组完整 valid 逐材料指标、谱与同组成配对，报告六项对比含交互；
  总体指标按材料配对，谱差按组成组重采样。逐样本导出新增可选谱出口。

新增主要符号：`LegacyContentGlobalLayer`、`validate_legacy_backbone_config`、
`legacy_backbone_configs`、`baseline_parameters`、`old_source_key`、
`initialize_legacy_backbone_model`、`legacy_backbone_contract`、
预检 `check/code_fingerprints`，实验入口 `reference_identity/verify_preflight/verify_completed/`
`factorial_report/composition_contrasts/analyze/execute`。

## 验证记录

- 新增 11 项检查已通过：旧内容层与原层几何打分清零后的输出/梯度一致；
  整个 O0 与原 baseline 打分清零后的完整输出一致；初值逐张量复制与 RNG 保持；
  无旧几何路径；E(3)、重排与空行；冻结配方；恢复与契约隔离；材料/组成组统计、
  差值与交互方向；完成组不重训；异常停止队列。
- 全仓 Ruff、内存语法编译及 **352 项测试通过**，仓库规范检查通过（757 个 prospective files）；
  日志在 `results/runs/legacy_backbone_attribution_20261012/check_static.log`。
- 真实 V100 预检 `results/runs/legacy_backbone_attribution_20261012/preflight_final.json`，
  耗时 **113.9 秒**，绑定 31 个源码与 C2/G0/C1/原 baseline 的配置、best/latest 指纹。
  原 baseline 与 C1 的未训练初始化记录完整复现；O0/O1 共享 **195 个剩余张量**，
  O1 的 **52 个局部张量**从 C1 逐值复制，初始化未改变调用方 RNG。
  总参数 O0 **69,248,324**，O1 **69,590,468**。
- 每组两批真实 train（batch32，352/380 原子；第二批微批 `[23,9]`）更新：全部参数梯度存在且有限。
  旧 `RPEncoding.forward` 由报错 mock 禁止调用，两组均通过；一批 valid 的 blind/oracle 指标有限。
- 一批真实 valid、六种 E(3) 操作、完整 eDOS/phDOS/H1/形状及重建谱：O0 两种 dtype 最大差为零；
  O1 FP32 最大相对差 **2.34e-6**，FP64 **3.98e-15**，低于预设 1e-4/1e-9 阈值。
  checkpoint 在内存中序列化并恢复，权重、AdamW 状态与 RNG 全部逐值核对通过。
- 逐样本导出新增可选谱文件出口后，只读复算已有 G0 的完整 **2,313 材料**：逐项指标与归档 CSV
  完全一致，谱与材料顺序一致；绝对组成 320 对/170 组、约化组成 360 对/180 组，
  绝对组成预测谱差在 1e-7 数值量级。临时导出在 `/tmp`，结束后移除，既有产物保持。

## 固定批次成本与正式执行

以下为同批两次预热后三次完整 optimizer step 的中位数，含 AdamW、当批缓存和记录构建。
不能当成完整训练轮耗时、全量缓存显存或正式精度证据。

| 组 | batch32 / 352 原子 | batch32 / 380 原子（23+9） | 当批 CUDA 最大分配峰值 |
|---|---:|---:|---:|
| O0 | 0.1741 s | 0.2403 s | 3,778 MiB |
| O1 | 0.1914 s | 0.2691 s | 3,920 MiB |

单卡两组预留 **3–4 小时**，量级与已有 G0/C1 接近；由前几轮实测更新剩余时间，配方固定。
预检更新的模型均已丢弃，正式组从未训练参考开始。

新 tag（seed42、35 轮，旧六层骨架，staged）：
`_oldg_attrib20261012_o0_k0_g6_staged_s42_e35`、
`_oldg_attrib20261012_o1_k2_g6_staged_s42_e35`。

由 OpenCode 在独立后台会话运行并监测：

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/tefscloud/miniconda3/envs/torch/bin/python -B \
/root/home/newstudy/uniARPAT/tools/eval/legacy_backbone_compare.py \
  --tag-prefix attrib20261012 \
  --preflight /root/home/newstudy/uniARPAT/results/runs/legacy_backbone_attribution_20261012/preflight_final.json \
  --out /root/home/newstudy/uniARPAT/results/runs/legacy_backbone_attribution_20261012 --run
```

监测 `queue.json`、`o0_train.log/o1_train.log`、两组完整 history、实际进程与 checkpoint。
先用每组前三轮估算剩余时间；每 5–10 分钟核对完成轮数、有限 loss/指标、OOM/traceback、显存。
状态无异常时只追加监测记录，完成、失败或需用户处理时汇报。异常停止，不自动重试或改配方。
完成组复核后自动复用 G0/C1/原 baseline，分析输出在同目录 `analysis/report.json`；
报告旧几何路径、骨架整体差异、C1 的两种净效应与交互，同组成谱差另报。
更新执行日志与 `docs/status.md`，不自动启动下一轮或做 Git 提交。

新旧骨架比较仍包含 Encoder 参数化、容量和内部初值；上述检查不证明正式训练的准确率，
不唯一定位整体差距，不证明跨种子稳定性。正式训练始终从未训练参考开始。
