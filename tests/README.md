# 测试与验证

| 目录 | 主要职责 |
|---|---|
| `model/` | 输出头、局部消息、全局模块、归一化、几何缓存、梯度及回归 |
| `data/` | 完整 FP32 晶体、变长 batch、周期邻居与谱能量定义 |
| `pipeline/` | 准备/恢复、选点、E＋S、导出、物理验收、路径和 Git 边界 |
| [archive/code/tests/](../archive/code/tests/) | 历史兼容与退役候选测试，保留原语义 |

按改动选择相关模块，例如只检查选点和 E 定义：

```bash
python -m unittest tests.pipeline.test_checkpoint_selection tests.data.test_spectral_balance
```

需要完整检查时使用已安装运行依赖及 Ruff 的环境：

```bash
python -m unittest discover -s tests -t .
python -m unittest discover -s archive/code/tests -t .
bash tools/checks/check-static.sh
```

静态入口执行 Ruff、内存语法编译、列出的 CPU/GPU 契约测试与仓库文件边界检查；其名称不表示所有测试都无数据依赖。
`.github/workflows/ci.yml` 当前为手动触发，不把文档中的历史通过记录当本次 CI 已运行。

## 已知前置条件

- `TestCompleteQ1Cache.test_full_storage_dtype_order_labels_and_repairs` 在新 manifest 存在时会读取旧 Q1 原数组作逐项对照。
  2026-11-03 本机缺少 `data/train4ARPAT/train/train_index.npy`，此项无法完成；其他依赖的跳过/失败由各测试实际前置条件决定。
- 不能以新缓存替代、伪造旧文件或修改条件掩盖缺失。完整迁移历史结果见[原始实施记录](../docs/logs/log-2026-10-04-crystal-structure-input.md)。
- GPU 短检和完整训练需相应方案与资源；短批通过不保证全程显存/吞吐或研究假设成立。

报告具体执行的命令、通过/跳过/失败数量及数据条件；文档更新通常检查链接、示例与差异，不为此重跑完整训练。
