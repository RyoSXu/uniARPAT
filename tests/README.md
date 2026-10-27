# 测试分类

| 目录 | 验证内容 |
|---|---|
| `model/` | 当前模型、输出头、归一化、局部消息、谱形loss、几何缓存失效/预算/梯度及回归 |
| `data/` | 完整 FP32 结构缓存、变长 batch/梯度累积、周期邻居和独立物理记录 |
| `pipeline/` | checkpoint恢复与约束选点、blind全谱协议、谱形辅助接线、CIF推理、支持库、路径和Git数据边界 |

历史候选测试集中于 [archive/code/tests](../archive/code/tests/)，不混入当前分类。

```bash
python3 -m unittest discover -s tests -t .
python3 -m unittest discover -s archive/code/tests -t .
bash tools/checks/check-static.sh
```

静态检查包含相关历史兼容契约，不加载完整训练或真实 checkpoint。
数据依赖以各测试前置条件为准，并非缺少任意缓存都会跳过：完整迁移对照还需要历史Q1原数组。
当前服务器缺少`data/train4ARPAT/train/train_index.npy`，该迁移检查无法完成；
不能以新缓存替代或伪造旧文件。新协议及当前数据路径的相关回归结果见
[119项检查记录](../docs/logs/log-2026-10-25-spectral-shape-preparation.md)。
CPU/GPU成本与完整模型训练需要独立方案；文档同步不代表重新执行模型测试。
测试通过只说明对应已执行检查通过，不能扩展为整个研究假设或训练复现已完成。
