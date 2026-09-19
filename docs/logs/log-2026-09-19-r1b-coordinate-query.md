# 工作日志 2026-09-19：R1b 坐标生成 query

## 改动与验证

- 在已技术通过的 R1a 逐点读出头基础上，实现 R1b 坐标生成 query：
  - 将固定长度的 `edos_query_embed/phdos_query_embed` 与可学习 target 替换为任务独立的坐标 query 生成器 $q_e(x)$、$q_p(x)$（plain MLP `1→128→512`，中间 GELU）。
  - Decoder target 置为同长度的全零张量，使 decoder 仅依靠连续坐标生成的 query 从晶体 memory 中抽取谱特征。
  - 参数变化：移除 196,608 个固定 query/target 离散参数，新增 132,608 个坐标 MLP 参数，模型可训练参数量由 71.117M 降至 71.053M（差异仅 −0.09%）。
  - `--r1b_coord` 默认关闭；关闭路径保持原有固定 query 与 target，状态字典与输出完全不变。
- 单测验证：
  - 关闭路径等价性（Bitwise exact）
  - 任意 `[B,E]`/`[B,P]` 坐标长度前向（测试任意非对称维度如 E=37, P=19 及高密网格）
  - 同坐标确定性（同一坐标输入输出 bitwise 严格相同）
  - 坐标与 query MLP 梯度回传（MLP 权重与坐标输入 $x$ 自身均接收有限非零梯度）
  - 真实 Q1 数据 CPU 单步冒烟通过
  - 全套 47 项单元测试通过
- 训练口径：`_r1bctl`（固定 query + R1a 逐点头）与 `_r1bcoord`（坐标 query + R1a 逐点头）均为 Q1、M1×10、seed 42、原 E0/P0、SumNorm KL/W1/Huber、H1 eta/gamma、dropout 0.05 的 oracle 测试口径。

## 结果

| 臂 | eDOS 中位 R² / 失败率 | phDOS 中位 R² / 失败率 | Cv MAE | 单轮耗时 | 峰值显存 |
|---|---|---|---:|---:|---:|
| `_r1bctl` | 0.4649 / 4.72% | 0.7122 / 1.22% | 0.2956 | 183.8s | 8,730 MiB |
| `_r1bcoord` | 0.4103 / 4.63% | 0.6763 / 1.66% | 0.4552 | 183.7s | 8,734 MiB |

- 增量（坐标 query − 固定 query 对照）：
  - eDOS：中位 R² −0.0546 / 失败率 −0.09pt
  - phDOS：中位 R² −0.0359 / 失败率 +0.44pt
  - Cv MAE：+0.1596
- 指标分析：
  - eDOS 中位 R² 恶化 0.0546（远超平局线 `|Δ中位 R²| < 0.02` 门槛）；
  - phDOS 中位 R² 恶化 0.0359（超过平局线门槛）；
  - 定容热容 Cv MAE 由 0.2956 显著恶化至 0.4552；
  - 失败率虽未明显恶化，但谱形质量出现系统性有害退化。
  - 物理与表征归因：将离散可学习 query 投影限制在标量坐标的连续 1D 流形（plain MLP 1→128→512）上，严重削弱了 decoder 在 cross-attention 阶段针对不同能区/频区探测晶体结构局部特征的表达自由度；相比之下，R1a 证实了输出端的逐点映射无害，瓶颈在于单纯依靠标量坐标生成 cross-attention query。

## 结论

**R1b 技术失败（Technical Fail），R1b park，默认关闭。**
根据 `docs/design/design-r1-readout-migration.md` 设定的准则：
- 出现有害退化，R1b 不具备准入资格，不得通过增宽或加深盲目调优；
- 依赖于 R1b 的后续非均匀网格与守恒重分箱数据表示实验暂不启动；
- 基准保持当前 B7 默认方案（M1 + 固定 query + legacy 头）。
下一阶段工作按 `docs/status.md` 重排后的待办顺序执行后续独立候选队列。
