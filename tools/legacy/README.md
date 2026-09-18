# 已退役入口

本目录只读保存 E8 期间被替换的入口。不得调用、更新这些文件，也不得把它们作为新工作的范例。

| 已退役文件 | 退役原因 | 支持的替代方案 |
|---|---|---|
| `train.py`、`train_script.sh`、`test.py`、`test_script.sh` | 依赖已删除的数据路径和过时的输出布局。 | `run_ablation_experiments.py` |
| `run_pilot_10epochs.py` | 固定使用已退役的 M5 scale head。 | `run_ablation_experiments.py --model M1 --epochs 10 --tag _pilot` |
| `test_cif.py` | 依赖已删除的 v1 CIF 缓存。 | `cif2dos.py` |
| `evaluate_and_plot.py` | 读取过时的检查点和预测文件名。 | 训练入口的评估功能加 `results/history_*.csv` |

重新引入已退役路径需要新的设计和兼容性测试。
