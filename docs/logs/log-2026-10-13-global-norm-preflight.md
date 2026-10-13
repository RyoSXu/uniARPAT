# N1 全局归一化位置对照：实施与预检

日期：2026-10-13。依据及判读见[设计](../design/design-2026-10-13-global-norm-attribution.md)。

当前状态：入口、初始化/恢复隔离、定向测试、真实 GPU 预检与全仓检查均已通过。
正式 N1 训练尚未开始；保留本轮开始时已有全部改动，没有提交 Git。

## 变更地图

- 全局层：`ScalarGlobalLayer(norm_first=False)` 添加 post-norm 计算；默认 pre-norm 分支保持。
  `LocalGlobalEncoder(N1)` 使用同 C1 局部消息与缓存，无旧几何打分。
- 初始状态与契约：`global_norm_config` 固定既有配方，
  `initialize_global_norm_model` 复用 C1 完整未训练状态，`global_norm_contract` 隔离同形状 checkpoint。
- 调度与分析：`global_norm_compare.reference_identity/verify_preflight` 绑定当前代码与历史证据，
  `execute/analyze/comparison_report/composition_contrasts` 只运行 N1，复用 C1/O1/原 baseline，
  分别报告材料指标与同组成谱差；恢复与完成组验收复用已有 `verify_completed`。
- 准入：`check_global_norm.check/code_fingerprints` 使用临时模型，记录真实损失、梯度、
  有限 E(3)、恢复、固定批次成本与初始状态摘要。
- 新增 10 项定向测试，并纳入 `check-static.sh`。

主要变更文件：`model/local_global_encoder.py`、`model/transformer.py`、
`utils/local_global_experiment.py`、`tools/train.py`；
新增文件见设计，既有局部核心和输出模块保持。

## 验证与交接

- 新增 10 项定向测试通过：post-norm 输出/共享参数与输入梯度匹配标准层，默认 pre-norm
  保持原公式；全部初始张量与 C1 一致；有限 E(3)/重排/空行与 C1 缓存映射；冻结配方
  与 CLI；同形状 checkpoint 拒绝；统计符号与组成组重采样；完成组不重训与异常停止。
- 全仓 Ruff、内存语法编译、**362 项测试**通过（163.8 秒），仓库规范检查通过
  （764 个 prospective files）；日志为
  `results/runs/global_norm_attribution_20261013/check_static.log`。
- 真实 V100 预检通过，耗时 **89.2 秒**，绑定 **34 个源码指纹**与 C2/G0/C1/原 baseline/O1
  参考身份和既有结果证据；报告为
  `results/runs/global_norm_attribution_20261013/preflight_final.json`。
  N1 **271 个状态张量**全部复制未训练 C1，整模型 hash 与 C1 的原始初始化记录一致；
  总参数 **75,894,212**，容量相同。调用方 CPU/CUDA RNG 保持，实际训练开始 seed 状态已记录。
- 两批真实 train，batch32、352/380 原子，第二批按 `[23,9]` 微批：全部参数梯度存在且有限，
  loss 为 2.8967/2.4909。完整 valid 一批 blind/oracle 有限，旧 RP 路径 mock 禁止调用通过。
- 六种 E(3) 操作、完整输出/尺度/重建谱：最大相对差 FP32 **2.90e-6**、FP64 **4.44e-15**，
  低于 1e-4/1e-9 准入阈值。权重、AdamW 状态与 RNG 的内存恢复逐值核对通过。
- 预检初次失败来自我把 CLI 的 `dropout=None` 误当作实际值；它表示继承配置，解析后的
  C1/N1 均为 0.05。第二次失败来自预检误用训练容器的 optimizer dict，已按既有
  `model.optimizer['transformer']` 接口及 state 比较修正。完整重跑通过，模型配方保持，
  两次失败只涉及临时模型，没有创建正式 history/checkpoint。
- 修正后的真实准入再次只读执行通过：当前 34 个源码、参考证据与最终配置匹配；
  正式 N1 状态为 pending，正式 run 目录无文件、history 不存在。

同批两次预热后三次完整 optimizer step 的中位成本（含当批记录/缓存与 AdamW）：

| 批次 | 中位耗时 | CUDA 最大分配 / 保留峰值 |
|---|---:|---:|
| batch32，352 原子，padded40 | 0.1969 s | 4,023 / 4,362 MiB |
| batch32，380 原子，padded94，微批23+9 | 0.2781 s | 3,813 / 4,408 MiB |

成本只代表固定批次，不包含全量缓存或整轮验证等开销。正式训练单组预留约 **1–1.5 小时**，
沿用 35 轮，不据短测声称精度改善；临时更新的模型均已丢弃。

正式 tag：`_gn_norm20261013_n1_k2_g6_staged_s42_e35`。
由 OpenCode 独立后台会话执行并监测：

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/tefscloud/miniconda3/envs/torch/bin/python -B \
/root/home/newstudy/uniARPAT/tools/eval/global_norm_compare.py \
  --tag-prefix norm20261013 \
  --preflight /root/home/newstudy/uniARPAT/results/runs/global_norm_attribution_20261013/preflight_final.json \
  --out /root/home/newstudy/uniARPAT/results/runs/global_norm_attribution_20261013 --run
```

核对进程、`queue.json/n1_train.log`、35 轮 history 与 best/latest；前三轮估 ETA，
其后每 5–10 分钟检查进度、有限指标、异常与资源。异常停止，不自动重试或变更配方。
完成组复核后自动分析 N1/C1/O1/原 baseline；报告 **N1−C1** 为直接归一化位置对照，
其他两项为整包参考，同组成谱差单列。区间含零不当作等价，单种子不宣称跨初始化稳定。
更新执行日志与状态页，不自动开始下一因素或做 Git 提交。
