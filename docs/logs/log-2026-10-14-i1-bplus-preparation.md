# 日志：2026-10-14 — I1 骨架完整 B+ 对照准备

对应[实验设计](../design/design-2026-10-14-i1-bplus-encoder.md)。代码、短检和运行入口已准备；
正式训练与监控交给 OpenCode，只新增 I2 的 seed42、35 轮，复用完成的 I1。
本次未运行完整 epoch、未创建正式 checkpoint，未建立 test loader。

## 改动与调用链

- `LocalGlobalEncoder` / `Transformer` 新增 `I2` 路由，实际几何/消息使用完整 `B+`；
  全局保持 I1 的 post-norm、概率 dropout=0、可学习恒等投影初值。
- `utils/i1_bplus_experiment.py`：`i1_bplus_config`、`validate_i1_bplus_config` 固定实验协议；
  `initialize_i1_bplus_model` 复制共同初值，`validate_i2_model` 检查骨架与完整 B+；
  `i1_bplus_contract` 隔离 checkpoint。接入既有配置/初始化/契约分派与训练 RNG 记录。
- `tools/checks/check_i1_bplus.py`：`check` 做真实数据有限更新、E(3)、恢复和成本预检。
- `tools/eval/i1_bplus_compare.py`：`reference_identity` 验证完成 I1 与当前数据；
  `verify_preflight` 冻结代码/参考指纹；`execute` 复用训练队列；`analyze` 复用总体 valid 导出并
  增加完整 train 导出、同组成谱差及零差响应比较。`group_summary` 使用组成组等权；
  `assert_paired_targets` 核对两模型的 ID 与目标，`export_train` 只做冻结推理。
- 新测试 `tests/pipeline/test_i1_bplus.py` 接入既有静态检查入口；状态、索引、基线约定与路线图同步。

调用链：`i1_bplus_compare` → 固定配置 → `tools/train.py` → I2 初始化/契约 →
`LocalGlobalEncoder(B+)` → 完整 train/valid 配对分析。局部公式、数据、损失与 decoder 配方沿用现有实现。

## 已执行验证

- 真实有效 train/valid 视图 18,704 / 2,313；两批 train 共 64 结构、一个 valid 批共 32 结构。
  仅一次性模型接收有限优化器更新，全部可训练参数梯度存在且有限，blind/oracle 评估有限。
- 历史 I1 **完整未训练**初始化记录与训练起始 RNG 重建一致；共同部分 239 个张量逐值复制，
  不加载已训练权重初始化 I2。实际采样顺序与历史预检一致。
- I1 / I2 参数为 75,894,212 / 75,923,660，差 29,448；共同可训练参数 75,686,852。
  局部内容分别为 207,360 / 236,808，因此仍按完整局部方案判读，不声称纯夹角归因。
- 真实结构 FP32 / FP64 E(3) 的最坏相对差分别为 1.82e−6 / 5.89e−12，
  低于 1e−4 / 1e−9；范围包含两类输出及重建的 blind/oracle 谱。
- 权重、优化器、训练 RNG 恢复一致；I2 拒绝 I1 checkpoint 契约。
  单元测试另外覆盖旧 B+ 契约拒绝、原子重排/NaN padding、分组权重、目标错配、
  train/valid 汇总、完成组不重训、失败停止，以及源码变化后拒绝分析。
- `tools/checks/check-static.sh` 通过：Ruff、语法、403 项测试（237.7 s）与仓库文件边界检查。

检查过程中纠正了采样种子误判：训练随机种子为 42，既有 `DistributedSampler` 固定为 0；
预检改为与历史顺序核对，没有改变 sampler。零长度 buffer 的空存储指针也从非共享存储断言中单列。

## 成本、证据与交接

V100 32GB 上最终预检用时 65.4 s。两批完整优化器更新的暖态中位数分别为 0.234 / 0.356 s：
32 结构、352 / 380 原子，宽度 40 / 94，实际 microbatch 为 `[32]` / `[23,9]`。
峰值分配显存 4,247 / 4,010 MiB；这是短检，不代表全数据缓存后的整轮峰值。
历史 I1 的中位轮时 131.4 s、35 轮合计 1.31 h。
据两批暖态成本与历史预算，I2 的 35 轮训练**估计约 2–3 h**，后处理另需数分钟；
这是外推，首轮后重估，不自动改变 batch、lr、epochs 或缓存策略。

本地[最终预检](../../results/runs/i1_bplus_encoder_20261014/preflight_final.json)冻结模型、代码、
I1 config/best/latest/history 与历史预检指纹。默认入口只打印一条 I2 命令，`--run` 才训练。
结束须同时核验 35 轮、latest/best 契约、队列完成与 `analysis_complete=true`；中途异常停止，不自动重试。
完整训练结果、全数据峰值与结构谱差收益仍待运行；等价晶胞/超胞一致性与 AMP 未在本轮验收。
