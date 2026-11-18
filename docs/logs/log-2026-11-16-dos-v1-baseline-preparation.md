# 日志：2026-11-16 — dos_v1 h2 基线准备

状态：**历史35轮准备，未开始正式训练；已被[100轮新准备](log-2026-11-16-dos-v1-h2-e100-preparation.md)取代**。[事先配方及追加](../design/design-2026-11-16-dos-v1-baseline.md)。

## 完成事项

- 工作区开始时干净。确认 dos_v1 的 v2 标签、部分标签训练和完整 valid 选点可复用现有训练器；未修改模型、数据或损失实现。
- 新增准备与独立监督工具。监督工具未实际启动，其运行生命周期仍须执行 agent 在正式授权启动时核验。
- 公开 manifest 含 test 的样本数元信息，但 `files` 仅列 train/valid。检查全部公开文件指纹，不读取封存 test。
- 新拟合 train 常数并冻结协议：eDOS 49,184、phDOS 24,639 条有效标签。精确常数、源码与数据指纹见准备 `report.json`。
- 本机可用 Python：`/home/jovyan/miniconda-recovery/envs/uniarpat/bin/python`，PyTorch 2.5.1+cu124，GPU 为 RTX 4090 D（24 GB）。
- 普通 train 批次 6 批 × 2 次临时更新完成；不保存 checkpoint。

## 失败及边界

将训练集中原子数最大的 32 个晶体（136～160 原子）组成压力批次，通过生产 `train_one_step` 的自动分块路径更新时，
在局部角度消息计算处发生 `torch.OutOfMemoryError`：额外申请 32 MiB 失败，PyTorch 已分配 22.68 GiB，GPU 剩余约 31.69 MiB。
进程退出，显存已释放，没有训练 checkpoint。train/valid 目标一致性检查尚未执行，普通批次计时未落盘，完整成本预检未完成。

这是人为构造的最大晶体组合压力检查，不证明正常随机顺序必定 OOM；但表明当前自动分块无法保证这组完整晶体可在 24 GB 上运行。
未自动调整分块、batch、精度或其他参数，未重试。

本次证据位于 `results/runs/dos_v1_baseline_20261116/`，其中 `preflight.json` 明确 `passed=false`；监督工具会拒绝该准备。
准备命令：

```bash
PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/jovyan/miniconda-recovery/envs/uniarpat/bin/python tools/eval/dos_baseline_prepare.py
```

## 验证

运行 `python -m unittest tests.data.test_partial_labels tests.data.test_spectral_balance tests.pipeline.test_spectral_balance tests.pipeline.test_phdos_head tests.pipeline.test_checkpoint_selection`：
50 项，49 通过、1 跳过（测试自身前置条件）；这些测试通过不等于 GPU 成本预检通过。

## 初次短检后提交用户的选项

1. 推荐：保持有效 batch32 与科学配方，批准较小计算分块预算的新预检；现有代码按整批的各任务标签数累积梯度，
   但分块会改变 dropout 的随机数消费，不能保证与原轨迹逐值一致。可能变慢，具体预算与成本需重新核验。
2. 换更大显存 GPU，保持现有计算分块，再做新预检；需要对应资源。

尚不提供可执行训练交接提示词，避免将失败准备交付为可启动实验。

## 同日继续：用户选择选项 1

显式计算预算设为 `pair_budget=51200`，有效 batch32、35 轮、seed42、FP32、h2 与所有科学配方保持。
首次新准备在旧结构对照校验中被拒绝（显式预算不同于历史默认），未进入 GPU，
`results/runs/dos_v1_baseline_pb51200_20261116/` 留存其协议和失败说明。

修正 `legacy_balance_config`：只对经过数据核验的 v2 协议，将历史 backbone 参照的计算预算恢复为其历史默认；
实际 cfg、training_record、data/checkpoint 契约仍记录 51200。v1 仍由冻结校验拒绝非默认预算。
`lg_valid_export.export` 现在传递保存配置的 `pair_budget`，使导出与训练、验证一致。
准备／监督工具改为显式指定新目录，准备工具失败时自动落盘失败状态。

新准备命令：

```bash
PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/jovyan/miniconda-recovery/envs/uniarpat/bin/python tools/eval/dos_baseline_prepare.py --out results/runs/dos_v1_baseline_pb51200_r2_20261116 --tag _sbal_dosv1basepb51200r2_20261116_h2_s42_e35 --pair-budget 51200
```

### 新预检结果

目录 `results/runs/dos_v1_baseline_pb51200_r2_20261116/`：`preflight.json` 为 `passed=true`，
准备、协议、有效模型组装、数据与源码指纹、源码快照、训练命令均已保存。

- 有效 train：58,105 行，eDOS 49,184、phDOS 24,639；valid 7,474 行，eDOS 6,342、phDOS 3,080。
- C：eDOS 12.231741379815725，phDOS 0.039093343576505346；只拟合 train。
- 参数 47,759,556；没有修改模型结构。
- 普通 20 批×2 次临时更新：冷／热每批 0.4409／0.4005 秒；每轮 1816 批。
- 最大 32 晶体组合更新通过，分为 16 块、每块 2 个，耗时 3.306 秒；全短检峰值 15668.7 MiB（约 15.3 GiB）。
- eval 模式 lr=0 的部分标签训练能量与验证联合 E 相差 2.22e-7。
- 35 轮纯训练外推 7.07 小时，不含验证、存盘、全量冷缓存成本；不是全程资源保证。
- 进程退出后 GPU 显存释放；没有训练 checkpoint、没有读取 test。

新增监督工具拒绝失败预检、变更源文件、未显式恢复、只扫描本次日志与完成状态的单元测试；
新增部分标签不均匀分布时，分块前后梯度与单次优化更新一致的测试（eval 模式关闭 dropout）。
首次新增梯度测试因沿用了 v1 fixture 被正确拒绝，已改成 v2 部分标签 fixture；最终同一相关测试集 57 项，56 通过、1 跳过。

长时训练仍交执行会话。完成训练后用 `lg_valid_export.export(..., spectra_out=...)` 导出完整 valid 原谱与逐材料结果，
按 valid ID 对齐原标签、原子数与缺失标志，用 NumPy 的 float64 独立复算两任务 E，核对 best/history 与标签数量，
补充来源、争议、虚频分组，最后更新状态与日志。不将跨版本分数差解释为数据扩充的因果收益。
