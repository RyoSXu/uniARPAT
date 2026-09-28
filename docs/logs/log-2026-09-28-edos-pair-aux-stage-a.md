# eDOS 同组成谱差辅助 Stage A 实现与资源门

## 范围

本轮只执行 `design-edos-pair-auxiliary-pilot.md` 的 Stage A：默认关闭的实现、CPU 合同、B7 初值
无优化步校准、固定少量 V100 资源门和 valid-only 判决工具。没有启动 epoch 训练，没有读取 valid/test
预测结果，更没有读取 test 标签。

## 冻结实现

- `utils/pair_aux_batches.py` 只接收 Q1 train 的元素数组和材料 ID。全量复算为 2,591 个合法 pair、
  1,331 个可配对绝对组成组、1,198 个被覆盖的约化组成组和 3,115 个材料。
- 每个 epoch 每个约化组循环选择一对；16 pair／batch，共 75 batch，均匀插入 585 个主 step。选择不
  接收或读取谱标签。冻结哈希覆盖全部 2,591 个候选和选择策略；epoch 哈希覆盖当轮 1,198 对。
- `edos_pair_contrast_loss` 实现预注册的带符号谱差 TV。control 和 candidate 运行相同的 pair 前向；
  前者乘零，后者加入校准权重。两个反向在同一 optimizer step 前顺序累加，主 step 数不变。
- runner 只允许 `_pcctl`／`_pcaux`、Q1、M1、10 epoch、seed 42、batch 32、FP32、SumNorm、H1 和
  B7 epoch 33 指定 checkpoint；核对 checkpoint SHA-256 与身份并严格加载。AMP、bucket、G1/G2、
  其他读出／结构候选和 test loader 均被前置拒绝。
- checkpoint 与 `config_used.yaml` 同时记录并严格核对 `pair_ratio`、`pair_lambda` 和冻结计划哈希；默认
  关闭时旧 payload 不增加字段。
- `edos_pair_aux_verdict.py` 只有 Q1 valid 入口，冻结 epoch 10 latest、2,313 同序样本、117 组／163 对
  机制集、256 材料保护集和五项联合判据；没有 test 参数。

## 校准与资源结果

冻结 B7 checkpoint SHA-256 为
`cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40`。epoch 0 首个 pair batch、
eval 模式、无 optimizer step 的结果：

| 项目 | 结果 |
|---|---:|
| base 梯度范数 | 0.8361804485 |
| pair 梯度范数 | 0.3223905265 |
| `lambda_pair` | 0.2593688071 |
| control/candidate 主输出最大绝对差 | 0 |
| control pair 梯度范数 | 0 |
| candidate pair 梯度范数 | 0.1487605274 |
| 校准后参数／buffer | 逐值未变 |
| 校准后 optimizer state 条目 | 0 |

首版把主图和 pair 图同时保留，时间比 `1.1010x` 通过但显存比 `1.8143x` 失败。没有调低覆盖率或放宽
门槛；改为同一 optimizer step 内先反传主损失、释放主图，再反传 pair 损失，数学上仍是梯度相加。
按 75/585 频率在 16 个代表性主 step 中插入 2 个辅助 batch，重跑结果：

| 臂 | 平均 step ms | 时间比 | 峰值 MiB | 显存比 |
|---|---:|---:|---:|---:|
| 无辅助 | 329.7448 | 1.0000 | 7148.7412 | 1.0000 |
| control | 362.0883 | 1.0981 | 7141.7412 | 0.9990 |
| candidate | 362.4380 | 1.0991 | 7141.7412 | 0.9990 |

两项均通过 `<=1.25x`／`<=1.10x`。正式机器可读结果见
`results/edos_pair_aux_stage_a_v100.json`。这只是可运行性与成本证据，不是准确率结果。

## 审查与验证

- OpenCode DeepSeek V4.1 Flash、MiMo-V2.6-Flash 和 GLM-5.3-Flash 均完成部分只读核查，但在写入前
  出现连接重置或长期无输出，未产生代码改动；协调者据已核实口径实施。
- 独立代码审查指出并已修复两个 P1：校准原先可能推迟到第 7 个主 step；运行门原先没有完整锁定
  B7/Q1/seed/epoch 配方。修复后训练 step 拒绝未校准状态，runner 在任何优化前校准并严格核验配方。
- 相关契约测试 40 项通过；最终全库 `python3 -m unittest discover tests` 为 270 项通过。
  `ruff`、compileall、`git diff --check` 均通过。

## 结论与下一关

Stage A 通过，仅说明方案实现、校准、恢复、隔离和资源成本满足预注册条件。下一项可能工作只有 Stage B
的 `_pcctl`／`_pcaux` 两臂 M1×10 valid-only pilot，必须另行批准；批准前不启动训练。Stage B 任何
非 win 结果都按设计 park，不扫描 ratio、频率、loss、sampler 或层级。
