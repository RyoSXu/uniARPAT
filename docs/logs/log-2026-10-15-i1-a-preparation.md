# 日志：2026-10-15 — 修正联合骨架的完整 A 对照准备

对应[实验设计](../design/design-2026-10-15-i1-a-encoder.md)。状态：prepared，正式训练尚未启动。
本轮仅一次性模型接收有限更新，没有完整 epoch、正式 checkpoint 或 test loader。

## 改动与调用链

- `model/local_global_encoder.py` / `model/transformer.py`：`LocalGlobalEncoder` / `Transformer`
  增加 I3；实际几何与局部消息为完整 A，全局设置沿用 I2。
- `utils/i1_a_experiment.py`：`i1_a_config` / `validate_i1_a_config` 固定联合配方；
  `i2_reference_parameters` / `initialize_i1_a_model` 重建共同初值；
  `validate_i3_model` 检查骨架与 A 执行设置，`i1_a_contract` 隔离 checkpoint。
- `utils/local_global_experiment.py` / `tools/train.py`：接入配置、初始化、契约与 RNG 分派；
  `write_local_global_execution` 记录 I3 的真实角度重计算、分块和缓存设置。
- `tools/checks/check_i1_a.py`：`check` 运行真实数据有限更新与成本测试，`code_fingerprints` 冻结 48 个源文件。
- `tools/eval/i1_a_compare.py`：`reference_identity` 验证完成 I2/I1 与历史导出；
  `verify_preflight` 冻结当前协议；`execute` 只训练 I3；`analyze` 仅对 I3 新增推理，
  复用 12 份 I1/I2 导出；`structural_report` 汇总 24 组统计与 16 组候选减参考的谱差比较。
- `tests/pipeline/test_i1_a.py`：11 项新增回归测试，接入 `tools/checks/check-static.sh`。

调用链：比较入口 → 固定配置 → 训练入口 → I3 初始化/契约 → 完整 A → 六层全局 →
共享 decoder / 两类谱 → 完整 train/valid 三模型配对分析。局部公式与联合损失沿用现有实现。

## 已执行验证

- 有效 train/valid 18,704 / 2,313；两批 train 共 64 结构，全部可训练参数梯度存在且有限；
  一个 32 结构 valid 批的 blind/oracle 评估有限。
- I2 完整未训练初始化记录、训练起始 RNG、数据视图与首批采样顺序重建一致。
  239 个共同张量逐值复制，共 75,686,852 个可训练参数。
  I3 / I2 总参数为 76,077,252 / 75,923,660，局部内容为 390,400 / 236,808。
- 4 个真实 valid 结构上的 FP32 / FP64 E(3) 最坏相对差为 1.61e−6 / 5.86e−15，
  小于 1e−4 / 1e−9；检查两类输出、eta、归一化谱形与重建 blind 谱。
  原子重排与 NaN padding 另由合成结构单元测试覆盖。
- 权重、优化器和训练 RNG 的内存 checkpoint 恢复一致；拒绝 I2 契约。
  单元测试另外覆盖 I1/旧 A/旧 B+ 契约隔离、配方漂移、失败停止、完成组不重训、
  导出逐字节复用和仅新增 I3 推理。11 项测试通过（39.9 s）。
- `tools/checks/check-static.sh` 通过：Ruff、语法、414 项测试（279.2 s）与 794 个候选仓库文件的边界检查。
  当前预检入口再次核验通过，I3 run/history 尚不存在；5 份本轮文档的 63 个本地引用检查通过，状态页保持 45 行。

历史 I2 的 source 指纹保持原样，不要求它等于新增 I3 后的源码；新预检锁定当前代码。
I2/I1 的 config、checkpoint、history、历史预检、完成队列与独立核验记录均重新核对，
导出必须与 `verification_codex.json` 的指纹匹配。没有修改或放宽 I2 原有预检入口。

## 实测成本与预算推断

V100 32GB，完整预检 67.0 s。每个成本批次暖身 2 次，测量 3 次完整优化器更新，FP32、batch32：

| 批次 | 原子总数 / padding 宽度 | 实际 microbatch | 中位耗时 | 峰值分配 / 保留显存 |
|---|---|---|---:|---:|
| 普通 1 | 352 / 40 | `[32]` | 0.362 s | 7,808 / 8,478 MiB |
| 普通 2 | 380 / 94 | `[23,9]` | 0.513 s | 7,534 / 8,538 MiB |
| 含 train 最大结构 | 510 / 160 | `[31,1]` | 0.593 s | 7,805 / 9,070 MiB |

这是少量缓存后的批次实测；最大原子数不保证是最昂贵的邻居/角度组合，
也未覆盖完整数据缓存和所有长尾批次，不能据此保证整轮峰值或耗时。
同批历史 I2 短检为 0.234 / 0.356 s，完整 35 轮训练实际 1.72 h；
本次 A 的普通批次约为其 1.44–1.55 倍。结合历史 A 的较重长尾，
I3 35 轮训练**暂按约 3–5 h 规划**，这是推断，分析另计。正式首轮完成后重估，不自动改变预算。

[最终预检](../../results/runs/i1_a_encoder_20261015/preflight_final.json) SHA256：
`6b7a5c3ee47927a95e9126b1b4177b0aa3d014c347fabfde0eddff999d8ac75e`。
训练结束须核对 35 轮、latest/best 契约、队列完成和 `analysis_complete=true`。

## 下一步解释线索

已核验的 I2 绝对组成 eDOS 导出显示：train/valid 的预测谱差 TV 中位数为 0.0322 / 0.0312，
目标分别为 0.2658 / 0.2697；方向余弦中位数为 0.1724 / 0.0618。
这提示应同时关心结构响应幅度与方向，不能仅增加响应幅度就期待 valid 改善。
两 split 的组成组不同，这也未证明特定层丢失信息或联合训练是唯一原因。
完成 A 后围绕完整模块的传播接口与联合谱差监督定下一项干预，单任务对照保持搁置。
