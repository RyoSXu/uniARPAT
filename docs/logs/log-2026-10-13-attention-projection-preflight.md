# 2026-10-13 — 投影初始化与学习能力对照：实施与准入

## 范围

用户确认继续检查可学习 Q/K/V/O 及其初始化。协议与判读见 [设计](../design/design-2026-10-13-attention-projection-attribution.md)。
本轮只准备 I1/I0 两组对照，复用已完成 D1/O1/原 baseline；代码与短测由 Codex 完成，正式训练继续交给 OpenCode。
两组沿用 seed42、35轮、batch32、lr5e-5、FP32、`balanced_oracle_mae`、修复版 `crystal_q1_v1`，仅 train/valid。
短测的参数更新只存在于随后丢弃的模型；不写正式 checkpoint、不读 test、不提交 Git。

**恒等投影**是保持输入特征不变的映射：Q/K/V/O 权重为单位矩阵，偏置为零；**固定投影**指这些权重不被优化器更新，梯度仍可传回输入。
I1 与 D1 比较初始映射，I0 与 I1 比较是否允许投影继续学习。冻结也减少有效可训练容量，不能把学习能力和容量分别归因。

## 已确认现状与实现

未训练权重核对显示：D1/O1 的 199 个全局层以外的张量相同；48 个保留的全局 FFN/归一化张量中，24 个 FFN 权重/偏置初值不同，归一化张量相同。
因此本轮保持未训练 D1 的 FFN，避免把它与投影因素一起更换。

调用链：`attention_projection_compare` → `attention_projection_configs` → `tools/train.py` →
`initialize_attention_projection_model/attention_projection_contract` → `LocalGlobalEncoder(I1/I0)`。

| 文件 | 符号与职责 |
|---|---|
| `utils/attention_projection_experiment.py`（新增） | `attention_projection_configs`、`validate_attention_projection_config` 固定两组配方；`d1_reference_parameters`、`initialize_attention_projection_model` 重建未训练 D1 并复制全部非投影初值；`set_identity_projections` 设置初始恒等映射；`is_projection_parameter`、`validate_projection_model` 校验冻结范围和映射；`attention_projection_contract` 隔离初始化/恢复协议。 |
| `model/local_global_encoder.py`、`model/transformer.py` | `LOCAL_ROUTES`、`LocalGlobalEncoder.__init__`、`Transformer.__init__` 接受 I1/I0；两组仍使用 C1 局部、post-norm、概率 dropout=0；I0 在创建优化器前仅冻结 Q/K/V/O。 |
| `utils/local_global_experiment.py`、`tools/train.py` | `validate_local_global_config`、`initialize_comparison_model`、`comparison_contract` 分发新协议；`build_arg_parser` 接受新组；`train_and_eval` 记录训练开始 RNG。旧组默认和历史契约保持。 |
| `tools/eval/attention_projection_compare.py`（新增） | `reference_identity`、`evidence_fingerprints`、`verify_preflight` 绑定历史 D1 与当前准入；`comparison_report`、`composition_contrasts` 标明两个直接对照；`analyze` 导出五组 valid；`execute` 仅按 I1→I0 训练，异常即停。 |
| `tools/eval/legacy_backbone_compare.py`、`tools/eval/global_norm_compare.py` | 既有 `execute`、`analyze` 增加可选回调与已完成组参数，供新队列复用；旧默认行为保持；已完成组不得覆盖冻结参考。 |
| `tools/checks/check_attention_projection.py`（新增）、`tools/checks/check_global_norm.py` | `code_fingerprints`、`check` 合并两组真实批证据；共享预检 `check` 支持 I1/I0，核查固定映射更新后保持且无 AdamW 状态。 |
| `tests/pipeline/test_attention_projection.py`（新增）、`tools/checks/check-static.sh` | 16 项定向检查覆盖共同初值、恒等公式、输入/共同梯度、固定投影更新、有限对称性、恢复与跨组拒绝、统计配对、准入变更拒绝、完成不重训和失败即停；接入既有全仓检查。 |

I0 保留参与计算的固定映射，不删除参数键或切换注意力实现。总参数量/键/形状与 I1/D1 相同，固定参数不产生梯度、不建 AdamW 状态。
新模型契约为 `attention-projection-model-v1`，初始化协议为 `attention-projection-init-v1`；I0/I1/D1/N1 checkpoint 不混用。

## 验证记录

定向检查首次执行 50 项时，恒等公式检查因测试夹具未显式提供 `activation` 而报错；已按模型默认 `gelu` 修正，并使用非均匀读出核对输入梯度。该项单独重跑通过，未改生产模型行为。
Ruff 与差异空白检查通过。全仓检查与真实 V100 两组预检随后执行；最终结果记录见下方收尾。

历史 D1/N1 的预检、报告和队列保留原指纹；新预检重新绑定当前源码与参考产物，不改写旧科学证据。

## 最终准入与证据

- **全仓检查通过**：实际运行 `tools/checks/check-static.sh`，Ruff、内存语法编译、391 项测试和仓库边界检查全部通过，测试耗时 214.744 秒。新增的 16 项投影检查在此轮均通过；此前失败的夹具项已修正，未把首次失败写成通过。
- **真实 V100 预检通过**：[preflight_final.json](../../results/runs/attention_projection_attribution_20261013/preflight_final.json)，总耗时 239.12 秒；绑定 40 份当前源码、7 组历史参考，两个候选均未有正式历史或 checkpoint。两组嵌套 `d1_initialization` 与已完成 D1 的未训练初始化记录完全相同，不使用已训练 D1 权重作起点。
- **初值和容量**：两组均有 271 个状态张量、75,894,212 个总参数，完整初始状态 SHA-256 均为 `47c689c450504d3ebbf4051664bbccfea48cd639cc574b2a761e1206ae07a370`。复制 D1 的 247 个非投影初始张量，仅 12 个投影权重张量初值改变。I1 全部可训练；I0 固定 24 个投影权重/偏置张量，可训练参数为 69,590,468，其余参数全部保持可训练。
- **真实训练模式和梯度**：相同的两批 train，各 32 个结构，352/380 个原子，实际微批为 `[32]`、`[23,9]`。六层注意力均在 train 模式；所有可训练参数的梯度存在且有限。I1 损失为 2.88304305 / 2.50405803，I0 为 2.88304305 / 2.50869317。第一批共同起点相同、更新后轨迹开始分开；这些损失只用于有限性检查，不作为准确率比较。
- **固定映射和恢复**：I0 经真实更新和成本短测后恒等映射仍逐值保持，无固定投影梯度或 AdamW 状态；输入和局部更新仍有梯度。两组模型权重、AdamW 与 RNG 的内存保存/恢复逐值通过，I0 固定标志保持；定向检查验证跨组 checkpoint 拒绝。
- **有限对称性**：真实 valid 批次的两类谱及 blind 重构在所测平移、旋转、反射下通过。I1 的 FP32 最差相对差 2.12×10⁻⁶、FP64 3.93×10⁻¹⁵；I0 分别为 2.05×10⁻⁶、4.04×10⁻¹⁵，均低于 1e-4 / 1e-9 阈值。定向检查另覆盖原子重排、空行与 C1 几何缓存；有限检查与架构不变性依据分开判读。
- **只读初值核对**：[initialization_comparison.json](../../results/runs/attention_projection_attribution_20261013/initialization_comparison.json) 再次独立重建 D1/O1，确认 199 个非全局张量相同，48 个保留全局张量中的 24 个 FFN 权重/偏置不同。未替换 FFN 或读取已训练权重作为初值。
- **正式入口准入通过**：[admission.json](../../results/runs/attention_projection_attribution_20261013/admission.json) 实际调用 `verify_preflight`，源码、数据和参考身份均通过，耗时 47.93 秒；`summarize` 显示 I1/I0 都为 pending。训练开始 RNG 同为 `d7e698b5c9fabbd10ec02394dbe8df9eca101cfcea47c0f5fbe259ee5d1a6dbf`，与原协议相同。没有启动正式队列或训练进程。

预检捕获配置时会打印训练入口横幅，但 `captured_config` 在真实模型/训练构建前受控停止，不执行 35 轮训练；只读学习率轨迹辅助仍产生已记录的 PyTorch scheduler 警告，正式优化器/调度执行顺序未改。

### 固定两批的成本

| 组 | 352 原子批更新中位耗时 | 380 原子批更新中位耗时 | 两批 torch 峰值分配最大值 |
|---|---:|---:|---:|
| I1 | 0.1961 s | 0.2843 s | 4,023.6 MiB |
| I0 | 0.1956 s | 0.2917 s | 3,950.7 MiB |

含热身后的完整优化器更新与几何缓存，每批计时重复 3 次。该短测不代表完整 epoch 耗时或整轮峰值显存；两组实测成本相近，不据此宣称固定投影提速。

## 交接

当前只准备，不启动正式训练。产物目录 `results/runs/attention_projection_attribution_20261013/`，tag prefix `proj20261013`；两组正式训练预计合计约 3 小时，实际以前三轮为准。
实施与准入已完成，可交给 OpenCode 按固定队列运行。科学问题仍待正式结果回答；本轮只有实现/协议/有限数值检查通过。
默认只打印命令，`--run` 才训练；完成组核验后跳过，已有分析目录不覆盖，失败即停、不自动重试。
总体比较 84 项、组成组比较 28 项中，优先判读 I1−D1 和 I0−I1，其他比较为整体参考；区间含零不证明等价，同组成谱差和总体指标分开报告。
