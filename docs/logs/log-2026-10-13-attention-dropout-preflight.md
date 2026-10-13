# 2026-10-13 — D1 概率 dropout 单因素实施与训练前准入

问题与判读规则见[设计](../design/design-2026-10-13-attention-dropout-attribution.md)，
参考结果见[N1 独立核验](log-2026-10-13-global-norm-run.md)。用户已确认继续。

## 目标与范围

在 N1 上只关闭六层全局注意力概率 dropout，保持 post-norm、C1 局部消息、learned
Q/K/V/O、所有其他 dropout、下游模块和全部未训练初值。新增 D1 独立身份，复用
N1/O1/原 baseline 做配对。正式训练交由 OpenCode，本轮不启动正式训练、不读取 test，
不改损失/数据/轮次/编排，不提交 Git，保留此前工作区改动。

## 变更地图

- 模型：`model/local_global_encoder.py` 的 `LOCAL_ROUTES/LocalGlobalEncoder` 接入 D1，
  局部映射为 C1，全局保持 post-norm，仅设置 `attn.dropout=0`；
  `model/transformer.py` 的 `Transformer` 接受 D1。
- 初始化/契约：`utils/attention_dropout_experiment.py` 的
  `validate_attention_dropout_config/attention_dropout_config/n1_reference_parameters/`
  `initialize_attention_dropout_model/attention_dropout_contract` 固定配方、复制完整未训练
  N1 状态并记录实际 dropout；`utils/local_global_experiment.py` 三个入口增加分派。
- 训练：`tools/train.py` 的 CLI 和训练开始 RNG 记录包含 D1，其余训练流程复用。
- 队列/分析：`tools/eval/attention_dropout_compare.py` 核验已完成 N1、绑定原证据、
  检查新预检，复用 `global_norm_compare` 的 `comparison_report/composition_contrasts/`
  `analyze/execute`；直接对照为 D1−N1，另报 D1−O1/D1−原 baseline。
- 真实短测：`tools/checks/check_attention_dropout.py` 绑定当前源码，复用
  `check_global_norm.check(route="D1")`；历史 N1 默认入口保持，旧预检不改写。
  `_train_batch_updates` 显式开启训练模式，再检查真实批次的 loss 与所有参数梯度。
- 测试：`tests/pipeline/test_attention_dropout.py` 的 `TestAttentionDropout` 共 13 项，
  已加入 `tools/checks/check-static.sh`。设计、状态和导航同步。

## 已执行检查

- 定向测试：`tests.pipeline.test_attention_dropout` 与 `tests.pipeline.test_global_norm`
  合计 22 项通过，耗时 20.656 秒。验证了初值/其他 dropout/eval 保持、手动干预的
  train 输出与梯度/RNG 匹配、有限对称性、错误契约拒绝、配对和失败停止。
- 修改涉及的 Python 文件 Ruff 检查通过，`git diff --check` 通过。
- 补训练模式检查前的全仓检查：`tools/checks/check-static.sh` 的 Ruff、内存语法编译、374 项测试全部通过，
  测试耗时 173.861 秒；仓库策略检查覆盖 771 个 prospective 文件，通过。
- N1 分析兼容：从已有导出重新运行默认分析，36 项总体对比连同 scope/选点/配对元数据
  精确复现原报告。12 项同组成分析与本轮开始前函数在相同 CSV 上逐项完全一致，
  与原报告最大差 9.55×10⁻¹⁸，来自 CSV 浮点往返，不影响区间含零判断。
- 补训练模式检查后的定向回归：D1/N1 合计 23 项通过（D1 13 项），耗时 21.996 秒；
  当前全部维护源码 Ruff 通过，217 个 Python 文件内存编译通过。最后的改动只涉及
  预检训练模式及对应检查，未再次重复全部无关测试；模型/训练/分析源码沿用全仓通过版本。

## 预检模式疏漏与更正

首次 GPU 预检已跑完，但核查发现前两个更新步骤处于 eval：`basemodel.__init__`
默认调用 `.eval()`，预检直接调用 `train_one_step` 没有先 `.train()`。因此首次两批
loss 恰好与 N1 的历史预检相同，不能作为训练期概率 dropout 已被检验的证据。
首次结果完整保留为 `results/runs/attention_dropout_attribution_20261013/preflight_eval_mode.json`，
不用于最终准入。

已将两批更新提取为 `_train_batch_updates`，显式 `.train()` 并核对六层注意力训练
模式；报告增加模式记录，同时增加从 eval 出发的回归检查。**正式训练**的
`tools/train.py::train_and_eval` 每轮已有明确 `.train()`，本问题仅影响预检中的这
两批梯度检查，不改写旧训练结论、旧预检或 checkpoint。既有热成本探针已经显式
`.train()`；E(3) 推理检查按设计仍使用 eval。

小模型补充实际运行确认：相同初值和 seed、训练模式下，原 N1 与 D1 输出不同；
eDOS/phDOS/eta 最大绝对差分别约 0.6561/0.00299/0.05727。该运行只确认训练期干预
有效，不构成精度证据。修正后的真实 GPU 预检已重跑通过，最终证据见下段。

## 最终真实 GPU 预检

产物：`results/runs/attention_dropout_attribution_20261013/preflight_final.json`，
schema `attention-dropout-preflight-v1`，`passed: true`，耗时 **100.812 秒**。

- **冻结身份**：37 份当前源码指纹；六个参考 run（C2/G0/C1/原 baseline/O1/N1）
  配置与 best/latest 文件保持。D1 的组装配置除 route 外与 N1 完全相同。
- **完整初值**：271 个状态张量、75,894,212 个参数与未训练 N1 相同，完整 N1 初始化
  摘要复现历史记录。状态 SHA-256 为 `86daf5a33c86b64849ea6e9be705a3785063deb6f0a4ebc35e387b37e38c8f5a`。
  初始化保留调用方 RNG；训练开始 RNG 复现 N1 的 `d7e698b5c9fa…`。
- **实际训练模式**：报告记录 `training_mode: true`，六层注意力模式全为 true；
  两个真实 train 批各 32 个结构、352/380 个原子，微批次为 `[32]` / `[23,9]`。
  两批 loss 为 2.909973/2.510457，所有可训练参数梯度存在且有限；只使用 train/valid。
- **几何与对称性**：旧 `RPEncoding` 路径被断言禁止；真实 valid 批 blind/oracle 指标
  有限。六项 E(3) 操作下输出最坏相对差 FP32 1.912×10⁻⁶、FP64 4.524×10⁻¹⁵，
  通过约定容差；不把有限操作检查等同于完整数学证明。
- **恢复**：内存序列化后，权重、AdamW 状态与 RNG 完全恢复；未写正式 checkpoint。
- **热成本**：完整 optimizer 步中位 0.195835 / 0.279052 秒，峰值分配
  4,023.582 / 3,813.735 MiB，峰值保留 4,352 / 4,398 MiB。
  这是两批资源短测，不能据此宣称整轮提速或精度保持。
- **证据保护**：原 N1 预检与报告 SHA-256 保持；第一版 D1 预检保留为
  `preflight_eval_mode.json`，其旧源码指纹不能通过最终入口的准入。

调度轨迹的只读复算沿用既有 `learning_rate_trace`，日志中有 PyTorch 对该复算调用顺序
和显式 epoch 参数的警告；本轮未改调度或正式训练的更新顺序。

## 当前状态

D1 实现与真实预检完成，正式训练 pending，尚无 D1 精度结论。
最终实际调用 `attention_dropout_compare.verify_preflight` 通过（39.346 秒），核对当前
37 份源码、冻结配置、六组参考与历史证据；明确核对两批更新及六层注意力处于训练模式。
随后 `--summarize` 对应的只读状态仍为 pending，未生成正式训练 history/checkpoint。
最终仓库策略检查 771 个文件通过，四份本轮文档的 138 个相对链接、空白与差异检查通过；
原 N1 预检/报告哈希不变，未提交 Git。
关闭概率 dropout 会改变后续训练随机流，直接对照反映当前协议下的净效应。
正式训练成本参考 N1 约 1.5 小时，训练期间按真实耗时更新估计。
