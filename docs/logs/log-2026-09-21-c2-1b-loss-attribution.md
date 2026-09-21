# 工作日志 2026-09-21：C2.1b 验证集损失归因审计

## 范围

- **问题：**检验 B7 的 SumNorm `KL + W1 + Huber` 与 H1 eta 是否在共享 encoder／decoder 上有可定位的量级失衡或 eDOS/phDOS 梯度冲突。
- **范围：**仅冻结 B7 `_e9ctl` 的最佳 checkpoint（epoch 33），仅 Q1 验证集 2,313 条；不构造 test loader、不调用 optimizer、不写 checkpoint。验证集只用于决定是否存在可证伪的下一项训练假设，不能报告为泛化改进。
- **与 L3 的边界：**L3 已证实删除 KL、W1 或 Huber 没有 accuracy win；本项不进行权重扫描，只有发现有方向的机制才允许另立单因素设计。

## 证据

- 新增 `tools/eval/c2_1b_loss_attribution.py`。它从 B7 保存的有效配置重建模型，逐样本分解生产 SumNorm 损失，并均匀抽取 32/73 个验证 batch，读取最后 encoder FFN 与共享 decoder FFN 的梯度范数和余弦关系。
- 命令：

  ```bash
  python3 tools/eval/c2_1b_loss_attribution.py
  python3 -m unittest discover tests
  ```

- 损失分解逐样本重构生产 `sumnorm_klw_loss`，新增 2 项合同测试；全套 **78/78** 通过。审计确认 checkpoint epoch=33、split=`valid`、样本数 2,313、所有适用的损失、梯度和分层量均有限。
- 机器可读结果为 `results/c2_1b_valid_attribution_summary.json`、逐样本表、梯度 batch 表和分层表。

| 验证集统计 | eDOS | phDOS |
|---|---:|---:|
| KL 中位损失 | 0.1808 | 0.2172 |
| W1 中位损失 | 0.0190 | 0.0121 |
| Huber 中位损失 | 0.000015 | 0.000082 |
| 主谱形总损失中位数 | 0.2003 | 0.2336 |

- eDOS/phDOS 主谱形梯度的中位余弦为 **0.074**（最后 encoder FFN）和 **0.009**（共享 decoder FFN），没有达到预注册的负冲突阈值 −0.10。
- W1 并未完全失活：相对 KL 的中位梯度范数比在 eDOS 为 12.4% / 29.3%（encoder/decoder），在 phDOS 为 6.9% / 6.8%；它与 KL 也不共线。但这只说明 W1 在参与优化，不能给出增大或减小权重的方向，且 L3 已经表明删除 W1 无收益。Huber 梯度仅为 KL 的约 $10^{-4}$ 量级。
- H1 eta 在 encoder 探针相对两主谱形梯度中位数的比值为 5.7%，低于 10% 门槛；它不经过 decoder，且与两主谱形梯度的中位余弦接近零。
- phDOS 的高熵谱确实更难：目标熵最低/最高四分位的归一化谱形 R²中位数为 0.895/0.599（差 −0.297），熵与总损失的 Spearman 相关为 +0.328。困难样本已经得到更大的当前损失，故该观察本身不能推出应上调还是下调其权重。

## 结论

- **状态：PARK。**未发现可操作的负任务冲突、辅助项压制或与 L3 不同的有方向损失机制；默认 SumNorm `KL + W1 + Huber` 与 H1 eta 保持不变，不启动 C2.1b 的 10-epoch pilot，也不做常规权重扫描。
- 高熵 phDOS 是可复用的错误分层事实，不是损失改动的因果证据。只有未来出现针对该类谱的独立结构、数据或物理机制时，才能以新设计重开。

## 后续

- C2.1b 已结束；下一项转入独立工程队列，优先为 C4 AMP 的设计与数值／资源门禁。它只降低后续实验成本，不声称准确率提升，也不启动训练。
- 已更新 `status.md`、`decisions.md` 和索引；默认训练与数据约定未变。
