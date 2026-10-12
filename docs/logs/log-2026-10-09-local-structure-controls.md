# C0/C1 同骨架局部结构对照：实施与执行

日期：2026-10-09。问题与既有诊断见
[最小对照设计](../design/design-2026-10-09-local-structure-attribution.md)及
[无更新诊断日志](log-2026-10-09-local-structure-diagnosis.md)。
本轮在用户确认“继续”后实施 C0 无几何、C1 简单径向，复用 C2 B+ 分段 L2/G6 的 35 轮结果。
只推进已确认的 seed=42、35 轮配方；不改变 A/B+ 核心公式、ZP、全局层/decoder、损失或数据。
保留工作区已有的深度解耦、历史对照与诊断改动，本轮不提交 Git。

## 实现与调用链

`tools/train.py` → `Transformer.forward` → 同一 `LocalGlobalEncoder` → 两次局部调用
（每次两块）→ 六层标准全局注意力/FFN → 现有六层共享 decoder → eDOS/phDOS 与 H1。
C0 在 `Transformer.forward` 中直接传 `records=None`；C1 使用原 6 Å 严格截断全镜像记录，
准备距离/计数，随后仅执行标量消息。两者的外部原子状态均为 512 维不变标量。

| 文件 | 本轮新增/修改符号与职责 |
|---|---|
| [`local_structure_controls.py`](../../model/local_structure_controls.py) | 新增 `CONTROL_ROUTES`、`PointwiseControlBlock`、`RadialControlBlock`、`ControlLocalMessage`；无几何/仅径向的两块局部内容 |
| [`periodic_messages.py`](../../model/periodic_messages.py) | `_prepare_message_geometry` 增加默认保持的 `directions`；`prepare_shared_message_geometry` 对 C1 不创建方向/角度特征，A/B+ 分支保持 |
| [`local_global_encoder.py`](../../model/local_global_encoder.py) | `LOCAL_ROUTES`、`LocalResidualLayer.__init__` 选择控制模块；`LocalGlobalEncoder.__init__/forward` 对 C0 禁止几何输入与准备，更新计数保持 |
| [`transformer.py`](../../model/transformer.py) | `Transformer.__init__/forward/enable_local_geometry_cache` 接纳 C0/C1；C0 跳过记录与缓存，旧模型路径保持 |
| [`periodic_geometry_cache.py`](../../model/periodic_geometry_cache.py) | `PeriodicGeometryCache.__init__/_assemble` 接纳 C1 的纯径向准备结果，保留预算/失效保护 |
| [`local_structure_experiment.py`](../../utils/local_structure_experiment.py) | 新增 `validate_structure_control_config`、`is_shared_control_parameter`、`parameter_counts`、`initialize_structure_control`、`structure_control_contract`、`load_control_reference`、`structure_control_configs`；固定配方、共同未训练权重、独立契约、参考数据/预算核查 |
| [`local_global_experiment.py`](../../utils/local_global_experiment.py) | `validate_local_global_config`、`initialize_comparison_model`、`comparison_contract` 对 C0/C1 分派新协议，旧协议保持 |
| [`train.py`](../../tools/train.py) | `build_arg_parser` 增加 C0/C1；`configure_local_global_execution` 对 C0 关闭缓存；`train_and_eval` 记录实际训练开始/恢复 RNG，原 seed reset 不变 |
| [`check_local_structure_controls.py`](../../tools/checks/check_local_structure_controls.py) | 新增 `check`、`step_cost`、`code_fingerprints`；真实配方/初值、前后向、两类谱对称性、恢复、GPU 成本及训练准入证据 |
| [`local_structure_compare.py`](../../tools/eval/local_structure_compare.py) | 新增 `verify_preflight`、`verify_completed_control`、`execute`、`analyze`；顺序训练与状态记录，完成后同选点配对分析，默认只打印命令 |
| [`test_local_structure_controls.py`](../../tests/pipeline/test_local_structure_controls.py) | 新增 `TestLocalStructureControls` 的 7 项边界/初值/对称性/参数量/缓存/契约检查；注册到 [`check-static.sh`](../../tools/checks/check-static.sh) |

各块公式和全部参数量见设计文档。C0/C1 总参数只差 68，共同可训练部分均为 75,686,852；
共享张量摘要均为 `fa82d71a2cce1b9d…`。C2 与两控制之间仍有消息形式和容量差异。

## 实际验证

正式预检：`results/runs/local_structure_attribution_20261009/preflight_r1.json`，V100 32GB、FP32/FP64，
约 65.7 秒；临时模型上若干真实 batch 更新用于验证损失、梯度及成本，丢弃临时权重，
未写新的 checkpoint，也未用预检权重作为训练初值。第一次成功预检保留为 `preflight.json`，
随后补齐源数据身份和代码指纹覆盖，再以 `preflight_r1.json` 复核并作为队列准入依据。

- C2 全部初始化记录复现，包括整模型 `dbdbbdf50a076083…`、共享 K2 `9746b5825aab96ba…`、
  扩展全局 `944ad93d990eca8b…`、非局部共同部分 `0e3f9c7bc57bb4a5…`。
- 实际 runner 组装配置除 `local_route` 外与 C2 完全一致。train/valid 为 18,704/2,313 条，
  batch=32，585/73 批；固定 35 轮 warmup 5＋余弦至 1e-6 的完整学习率轨迹保存在预检中。
  原模板里的 `epochs=100` 不代表实际 runner 的 35 轮学习率轨迹。
- C0/C1 的训练开始 RNG 重建摘要相同：`d7e698b5c9fabbd10…`；记录 sampler seed=0 的
  epoch 0/1 全样本顺序摘要。C2 历史开始 RNG 未单独落盘，只能依据历史代码/执行协议复核。
- 两个真实 train batch，C0 loss 约 2.93/2.56、C1 约 2.90/2.52，全部参数梯度存在且有限。
  第一批 32 结构/352 原子，第二批 32/380 原子并按原协议拆为 `[23,9]`，未改变梯度累积。
- 真实 valid 批的六种 E(3) 操作，eDOS/phDOS/H1 scale 及 shape/blind 重建谱：C0 最大差为 0；
  C1 FP32 最大相对差约 1.85e-6，FP64 约 3.08e-15，分别低于 1e-4/1e-9 检查阈值。
- 真实权重、AdamW 状态与全部 RNG 经内存 checkpoint 序列化/恢复逐值复核；使用 `weights_only=True`。
- 单元测试：新增 7 项＋相关深度/对照 16 项全部通过；全仓静态/语法及 **292 项回归通过**，
  仓库规范检查通过。初次测试夹具遗漏 Q1 谱网格，补齐夹具后通过；未放宽模型要求或对称性阈值。

## 成本与执行

以下是同批次两次预热后、三次完整 optimizer step 的中位耗时。含模型、已分配 AdamW 状态
和当批已填充的缓存，不含数据读取、CUDA 上下文或全量缓存；不能当成整轮吞吐/显存上限。

| 分支 | 第一批耗时 | 第二批耗时 | 当批最大 CUDA 分配峰值 |
|---|---:|---:|---:|
| C0 | 0.1899 s | 0.2657 s | 3,912.9 MiB |
| C1 | 0.1957 s | 0.2795 s | 4,017.5 MiB |

C0 不建立缓存，C1 沿用 `min(16GiB,half_device)` 的有预算径向缓存。
只新增两组各 35 轮；训练前外推总计约 3 小时，实际耗时以首轮/整轮日志修正，不保证此估算。

```bash
PYTHONUTF8=1 python tools/eval/local_structure_compare.py \
  --tag-prefix attrib20261009 \
  --preflight results/runs/local_structure_attribution_20261009/preflight_r1.json \
  --out results/runs/local_structure_attribution_20261009 --run
```

新增 tag：`_lsc_attrib20261009_k2_g6_c0_staged_s42_e35`、
`_lsc_attrib20261009_k2_g6_c1_staged_s42_e35`。队列和日志位于
`results/runs/local_structure_attribution_20261009/queue.json`、`queue.log`、`c0_train.log`、`c1_train.log`。
运行中不改这些分支的源码/配置；异常即停、不自动重试，旧 checkpoint 不删除或覆盖。

队列完成后自动以共同 `balanced_oracle_mae` best checkpoint 分析 train 固定 2,048 条与 valid 全量：
三组四项 blind/oracle 指标、R²<0 失败率、逐材料配对区间、同绝对/约化组成有符号谱差误差和
按组成组重采样区间；还复算 C2 并核对既有逐材料结果与失败标志。
分析输出 `analysis/report.json` 与逐材料/谱/配对文件。完整学习轨迹保留在原 history CSV 中。

**状态（2026-10-10 复核）：C0/C1 均完成 35/35 轮，自动三组分析已完成，GPU 当前空闲。**
C0 在第 30 轮后中断并恢复的经过保留于下节；最终结果见“正式结果与复核”。
首次启动时已确认 C0 完成第 1/35 轮并进入第 2 轮，首轮 train 120.8 秒，验证约 5 秒。
实际保存的 C0 初始化记录、契约、数据和训练开始 RNG 与正式预检逐项一致。
首次队列 PID 21217、C0 训练 PID 21257，后续进度同时核对真实进程、history 与 checkpoint。
当前证据只证明所执行的检查通过，不证明新方案准确率、更大输入或全部物理变换均已验收。
单 seed、35 轮对照不能证明跨初始化稳定性，也不能据此否定局部结构的一般价值。

## 中断与恢复（2026-10-09）

用户询问进度时发现首次队列与训练进程均已退出、GPU 空闲，旧 `queue.json` 仍写着 running。
C0 history 完成至第 30 轮，第 31 轮日志最后到 300/585 批，C1 未启动。训练日志无异常 traceback，
原执行会话也已失效；退出原因不能确定，不将其归因于模型、CUDA OOM 或用户停止。
现场记录保留在 `interruption_20261009.json`，完整性检查在 `resume_verification_20261009.json`。

只读核查通过：代码与正式预检指纹一致，初始化、模型/数据契约一致；latest 为第 30 轮，
best 为共同 MAE 选点的第 29 轮，均与 history 对齐；latest 全部浮点权重有限，
271 项 optimizer 状态及 Python/NumPy/Torch CPU/CUDA 随机状态齐全。
由原 runner 恢复权重、优化器、随机状态并快进原 35 轮学习率轨迹，从第 31 轮重放未完整保存的一轮。

没有修改科学源码或配置。执行方式改为独立 `tmux` 会话 `uniarpat_lsc_attrib20261009`，
启动脚本 `results/runs/local_structure_attribution_20261009/resume_queue.sh`，追加原日志。
独立后台探针在启动工具返回后保持运行并完成，其服务器父进程为 PID 1；正式队列使用同一方式。
本次恢复为调查后的人工恢复，未增加自动异常重试，也没有删除已有 checkpoint 或从头重训。
已实际观察恢复后的第 31 轮运行，前 100/200/300 批的 loss 日志在四位小数精度下与中断前相同；
恢复记录明确为 `from_epoch=31`、`method=resume`。这支持恢复生效，不等同于逐位复现全部训练状态。
恢复后已完成第 31 轮并进入第 32 轮，独立后台会话与实际训练进程均确认存活。

## 正式结果与复核（2026-10-10）

队列于 **2026-10-09 21:30:43（北京时间）**完成，时间由 `queue.json` 的 Unix 时间转换。
C0/C1 history 均连续包含第 1–35 轮，latest 均为第 35 轮；原选点 `balanced_oracle_mae`
的 best 分别为 **29 / 34**，复用 C2 best 为 **35**。三组均为 seed=42、35 轮、
`crystal_q1_v1`、SumNorm/H1、L2/G6 staged，仅 train/valid。

主报告：`results/runs/local_structure_attribution_20261009/analysis/report.json`。
以下为完整 valid 2,313 条的逐材料 R² 中位数；blind 使用模型预测尺度，oracle 使用真实尺度。

| 分支 | eDOS blind | eDOS oracle | phDOS blind | phDOS oracle | eDOS blind 失败率 |
|---|---:|---:|---:|---:|---:|
| C0 无几何 | 0.44757 | 0.49527 | 0.70140 | 0.71061 | 11.76% |
| C1 简单径向 | 0.46090 | 0.50455 | 0.71201 | 0.72223 | 10.77% |
| C2 当前 B+ | 0.46668 | 0.50529 | 0.71442 | 0.72585 | 11.72% |

事实（区间为逐材料配对 bootstrap 95%，条件于本次选中的 checkpoint）：

- **C1 相对 C0 的总体收益可分辨**：四项 R² 均提高且区间不含零。
  eDOS blind ΔR² **+0.01333** `[+0.00481,+0.02284]`、oracle **+0.00929**
  `[+0.00129,+0.01902]`；blind 失败率 **−0.994 pp** `[-1.902,-0.086]`。
  两种 eDOS MAE 同时改善，phDOS 失败率差异未分辨。
- **C2 相对 C1 的 eDOS 额外收益未分辨**：blind ΔR² **+0.00578**
  `[-0.00463,+0.01396]`、oracle **+0.00074** `[-0.00682,+0.00966]`；
  eDOS MAE 和失败率差异的区间也均含零。C2 有部分 phDOS 收益：blind/oracle MAE 区间不含零，
  oracle 失败率降低 **0.649 pp** `[-1.254,-0.043]`；四项 R² 的差异均未分辨。
- C2 相对 C0 的四项 R² 均改善且区间不含零。本次同骨架对照不支持将几何分支概括为
  “完全无效”；总体验证收益与同组成结构区分能力需分别判断。

### 同组成结构区分

使用共同固定 train 子集 2,048 条与完整 valid，保持此前预定的组成组等权统计。
下面是绝对组成的 eDOS 有符号谱差误差：每组先求配对误差均值，再按组成组重采样。
负差值表示谱差预测改善，共享材料的多个配对不当作独立样本。

| 对比 | train 27 组的平均误差差值 [95% 区间] | valid 170 组的平均误差差值 [95% 区间] |
|---|---|---|
| C1−C0 | −0.02257 `[-0.04932,-0.00166]` | **+0.00474** `[+0.00062,+0.00940]` |
| C2−C0 | −0.02080 `[-0.04390,-0.00206]` | +0.00393 `[-0.00215,+0.01186]` |
| C2−C1 | +0.00178 `[-0.01153,+0.01740]` | −0.00081 `[-0.00555,+0.00453]` |

约化组成的 valid eDOS 结论一致：C1−C0 为 +0.00465 `[+0.00046,+0.00951]`，
C2−C0 为 +0.00397 `[-0.00180,+0.01183]`，C2−C1 为 −0.00069 `[-0.00509,+0.00420]`。
C1 的同组成谱差预测在 train 上改善，在 valid 上按预定组均值口径反而变差；
C2 相对无结构响应参照的 valid 收益未分辨。phDOS valid 谱差误差对比的区间均含零。

valid 320 对绝对同组成材料的目标 eDOS TV 中位数为 **0.26975**，C1/C2 预测 TV
仅 **0.02379 / 0.03004**；两者虽有结构响应，仍未准确还原目标差异。
C0 预测 TV 约 1.3e-7，是无结构响应的数值波动，接近零的 skill 不作微小正负收益解读。
配对误差中位数与组成组等权均值是不同统计量，采用判断沿用预定组均值及其区间。

### 与原六层 Encoder baseline 的性能参考

补充报告：`analysis/paired_vs_original_baseline_20261010.json`。复用原 baseline 的真实 MAE best
（第 34 轮），核对同 seed、35 轮、数据契约与完整 2,313 材料身份。它用于整体架构性能参照，
不能把新旧 Encoder 的差异单独归因于局部内容。

原 baseline 四项 R² 为 **0.47292 / 0.50991 / 0.72723 / 0.73521**。
C1 相对它的 blind eDOS ΔR² **−0.01202** `[-0.02070,-0.00236]`、blind phDOS **−0.01522**
`[-0.02504,-0.00450]`，eDOS blind 失败率 **+1.989 pp** `[+0.908,+3.070]`。
C2 的 eDOS R² 差异区间含零，但 phDOS R² 仍更低、eDOS blind 失败率仍更高。
因此当前 C1/C2 尚不能据此替换原 baseline。

### 实际成本与独立核查

| 分支 | 每轮 train 中位耗时 | 35 轮 train 耗时总和 | 最大 CUDA 分配峰值 |
|---|---:|---:|---:|
| C0 | 120.1 s | 1.173 h | 4,461 MiB |
| C1 | 131.8 s | 1.313 h | 6,759 MiB |
| C2 已有结果 | 172.5 s | 1.714 h | 8,679 MiB |

这些时间为 history 的训练阶段统计，不含验证、保存、分析或中断等待。C1 每轮比 C2 少约 23.6%，
可作为成本较低的后续诊断参照；这不是最终生产选型。

本轮没有新训练或模型修改。只读核查证据保存在 `analysis/verification_20261010.json`：
代码/参考/数据指纹通过、实际训练初值与开始 RNG 通过、两组完整预算与 best 选点通过；
三组 × train/valid 的 CSV 重新计算 R²/MAE/失败率，由 NPZ 重新计算绝对/约化组成的
有符号谱差误差与组成组平均差值，均与主报告一致，差异仅为浮点舍入量级。
自动分析还复算 C2 valid，原逐材料指标和失败标志复现。

**推断与下一步建议：**简单几何内容带来了总体收益，复杂 B+ 的 eDOS 额外收益仍未显示。
接下来应进一步定位同组成结构响应的泛化、训练覆盖与监督/读出利用，C1 可作为简化参照。
本次不证明具体根因、跨训练种子稳定性或角度的一般价值，未启动下一轮训练。
