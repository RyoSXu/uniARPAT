# 日志：2026-10-06 — 大角度分块与 GPU 结构缓存

创建日期：2026-10-06。状态：**closed**。用户已批准采用并整理提交。本轮为执行方式优化，正式训练继续停止。
原 A/staged 完成 21/35 轮，其余三组 pending；seed=42、K=2、batch=32、lr=5e-5、选点指标及预算保持。
不读取谱标签或 test、不执行 optimizer step、不改写原 checkpoint、配置或 history。

## 变更地图与决定

调用链为 `train_and_eval → Transformer → PeriodicGeometryCache → LocalGlobalEncoder → PeriodicLocalMessage`。
缓存未命中时构建当前结构记录，命中时重组当前 batch 的记录；局部模块计算可学习消息，后续全局和读出仍只读原子标量。

| 文件 | 符号与职责 |
|---|---|
| [`local_global_encoder.py`](../../model/local_global_encoder.py) | `LOCAL_PAIR_CHUNK=65536`；`LocalGlobalEncoder.forward` 在一次前向共享结构特征；`LocalResidualLayer` 仍用独立参数计算每次消息 |
| [`periodic_messages.py`](../../model/periodic_messages.py) | `prepare_shared_message_geometry` 准备结构量；`PeriodicLocalMessage.forward(prepared_geometry=...)` 核对来源并复用；`InvariantMessageBlock._angle_chunk` 复用角度基/截断因子；B+ 复用球谐 |
| [`periodic_geometry_cache.py`](../../model/periodic_geometry_cache.py) | `PeriodicGeometryCache.get/_assemble` 保存单结构记录与静态特征，再映射到当前 batch；`geometry_cache_budget`、LRU 与工作空间估算控制显存 |
| [`transformer.py`](../../model/transformer.py) | `enable_local_geometry_cache` 配置进程内执行缓存；`forward` 命中或回退；`_apply` 在模型移动/转换 dtype 前清空旧条目 |
| [`train.py`](../../tools/train.py) | `configure_local_global_execution` 仅为 CUDA、无增强对照启用；`write_local_global_execution` 记录大分块、共享几何、容量策略及生效轮次 |
| [`test_periodic_geometry_cache.py`](../../tests/model/test_periodic_geometry_cache.py) | 失效、重排/填充、预算/资源回退、推理后训练、全部参数/坐标梯度、随机流与参数更新响应 |
| [`test_local_global_comparison.py`](../../tests/pipeline/test_local_global_comparison.py) | 真实 runner 的合成恢复与运行设置记录；增强/CPU 绕过与科学契约/RNG 保持 |
| [`check_local_global.py`](../../tools/checks/check_local_global.py) | `--geometry-cache` 对正式缓存路径检查四组全链有限 E(3)、完整梯度与成本 |
| [`check_local_message_speed.py`](../../tools/checks/check_local_message_speed.py) | 历史源码参照显式保留旧 4096 分块/不共享几何，维持历史核验入口 |

独立局部模块的默认 4096 分块与重计算选项保持；正式组装使用 65536、保存角度激活。
没有新增可学习参数、state_dict 键、训练 CLI 或科学配置字段，`local-global-init-v1` 和 checkpoint 契约保持。

## 缓存边界与显存策略

键使用 FP32/FP64 晶格矩阵与有效分数坐标的精确内容，dtype、设备、半径作为缓存上下文。
不依赖 dataset ID，不改变数据/输入格式。更换坐标或晶格重新构建；改变 padding、batch 顺序或拼接方式重新映射。
原子身份由当前原子状态参与消息，缓存不保存元素投影、可学习权重、消息、输出或优化器状态。
坐标/晶格需要梯度、混合空行、旧式几何输入、dtype/设备不匹配时回到普通路径；CPU 或增强训练不启用持久缓存。
同次前向的共享结构量保留坐标/晶格的可导图，不对其 detach。

采用延迟填充，跨 epoch 复用。CUDA 默认上限为 `min(16 GiB, 设备显存/2)`；当前 V100 32 GiB 为 16 GiB。
条目超预算不持久保存，LRU 淘汰不删除结构或邻边对。单结构条目不永久保存逐邻边对的 batch 接收者索引，避免额外约 3 GiB。
对即将计算的完整邻边对另估工作空间，在内存不足时先释放缓存；估算保守程度不等于所有 batch 均保证可运行。
缓存构建/重组分配 OOM 时清空可选缓存并回到普通路径；普通模型本身 OOM 仍报错，不更改模型公式或自动缩小有效 batch。
推理中填充的持久张量仍为普通 tensor，随后可用于训练反向；切换模式不缓存学习结果。

恢复时在 `config_used.yaml.runtime.local_global_execution` 记录例如：

```yaml
- from_epoch: 22
  route: A
  recompute_angles: [false, false, false, false]
  pair_chunk: [65536, 65536, 65536, 65536]
  shared_geometry: true
  geometry_cache:
    enabled: true
    key: cell-frac-content-v1
    population: lazy
    max_bytes: 17179869184
    budget_policy: min(16GiB,half_device)
    workspace_guard: local-pairs-v1
```

CPU/增强记录关闭及原因。旧配置无该记录仍能恢复；不追记前 21 轮，不把执行字段加入科学契约。
本轮未写入原实验配置；正式恢复会由既有入口记录。

## 数值与成本证据

先行探针采用临时适配器，源码与 JSON 保留在忽略目录；它们用于决定，不冒充正式代码的测量。

- `results/runs/local_global_pair_chunk_20261006/`、`local_global_pair_chunk_20261006_fp64/`：4096→65536。
  同真实训练权重，普通/尾部前向+反向为 1752.80→478.29 ms / 3243.96→807.14 ms，提速 3.66 / 4.02 倍。
  FP32 输出最大绝对差 1.526e-5；逐点参数梯度检查未过 190/239 与 37/239，整体相对 L2 为 2.778e-5 / 2.668e-6。
  FP64 输出约 2e-14、梯度约 2.84e-14，239 项梯度检查均通过。未放宽原阈值，未承诺大分块逐位相同或最终精度相同。
- `results/runs/local_global_geometry_cache_20261006/`、`local_global_geometry_cache_validation_20261006/`：
  两种缓存的结构身份、四组两 dtype 数值与有限 E(3)、失效/梯度回退；A 固定同分块确定性梯度相同，B+ 有极小非逐位差异但通过既有容差。
- `results/runs/local_global_geometry_cache_full_20261006/`：临时适配器全量 train/valid 静态张量 15.278 GiB，构建 212.56 s；
  真实 optimizer 保留，正常/尾部训练批耗时减少 18.12% / 9.26%，固定 valid 结构前向减少 43.11%，峰值 allocated 22.825 GiB。

正式实现的独立复查见 `results/runs/local_global_geometry_cache_adopted_20261006/`。
`checkpoint_probe.py` 保存实际检查源码，`checkpoint_cache_fp32.json` 保存恢复、全部梯度、随机流、全量缓存与成本证据。
数值比较固定同分块 65536，只在诊断阶段启用确定性 CUDA；生产入口未开启该设置，计时恢复普通执行。
使用只读恢复的第 21 轮权重与完整 AdamW 状态，普通 `[32]`、尾部 `[31,1]`，沿用既有有效 batch 和反向加权。
共享几何、冷缓存和热缓存与不共享参照的输出及全部 239 个参数梯度相同，随机流相同。
模型/缓冲区与 optimizer 每个状态保持；原配置、best/latest checkpoint 与 history 的 SHA256 保持。

正式实现全量装入 **18,704 train + 2,313 valid = 21,017** 条目，占 15.278 GiB，构建 234.73 s。
当前两批次没有预算/工作空间淘汰或 OOM 回退；推理模式填充后可直接执行训练反向。
每项预热两次、重复三次，完整前向+反向与有限梯度检查，含真实 optimizer 状态，无更新：

| 批次 | 65536 不共享 ms | 同次前向共享 ms | 全量热缓存 ms | 相对不共享耗时减少 | 热缓存峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|
| 普通 32 | 484.37 | 471.57 | 389.05 | 19.68% | 22.828 |
| 含 160 原子、按 `[31,1]` | 802.63 | 792.11 | 728.78 | 9.20% | 22.798 |

这张表只比较同为 65536 的执行方式；不与先行 4096 跨进程成本混为严格同时测量。
正式延迟填充把上述构建成本分摊在最先访问各结构时；首轮冷缓存与后续热缓存吞吐应分开报告。
不将固定批次短测换算为整轮或完整队列加速保证。

最终兼容复查又增加空 batch 与模型/结构 dtype 不一致时的普通路径保护；26 项相关检查通过，
未更改以上正常 FP32 批次的计算路径。最终全链报告 `e3_fp32.json/e3_fp64.json` 使用当前正式源码与缓存。
覆盖 A/B+ × 分段/交替、六类 E(3) 操作、谱 logits/形状/尺度及 blind 重建和全部参数连接；
FP32 另查 192 原子结构压力例；有限 E(3) 输出最大绝对差 FP32 为 1.609e-6、FP64 为 3.580e-15。
两份最终全链报告的全部源码哈希与本次正式实现一致。
最终 `check-static.sh` 的 Ruff、语法、**268 项回归通过（109.345 s）**，仓库工作区政策检查通过。
四组历史 Git 源码参照的 CPU 前向/反向兼容检查通过。
7 份改动文档的 126 个本地引用及 `git diff --check` 通过；原始静态日志保存在同一证据目录，不与先行用例数相加。

## 结论与恢复

用户采用的是相同数学公式的执行方式优化。大分块有 FP32 计算顺序差异；缓存仅存结构量、学习计算每次重做。
准确率、整轮吞吐和所有实际 batch 的显存仍待正式训练验收。有限数值/E(3) 与回归检查不能证明最终 R²/MAE 相同。
训练保持停止。本轮不修改原队列 tag、预算或参数；以后恢复原 seed=42 队列时，A/staged 从完整第 21 轮恢复，下一轮为 22。
先前与后续轮次的成本必须按执行设置分段报告。
