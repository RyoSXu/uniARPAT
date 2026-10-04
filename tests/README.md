# 测试分类

| 目录 | 验证内容 |
|---|---|
| `model/` | 当前模型、输出头、归一化、局部消息和回归 |
| `data/` | 周期邻居枚举和独立物理记录 |
| `pipeline/` | checkpoint 恢复、CIF 推理、支持库、路径和 Git 数据边界 |

历史候选测试集中于 [archive/code/tests](../archive/code/tests/)，不混入当前分类。

```bash
python3 -m unittest discover -s tests -t .
python3 -m unittest discover -s archive/code/tests -t .
bash tools/checks/check-static.sh
```

静态检查包含相关历史兼容契约，不加载完整训练或真实 checkpoint。
依赖机器本地 Q1 的检查在缓存缺失时明确跳过；CPU/GPU 成本与完整模型训练需要独立方案。
测试通过只说明对应已执行检查通过，不能扩展为整个研究假设或训练复现已完成。
