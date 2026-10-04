# 已退役入口

本目录保留已退役入口及旧 v1 数据检查脚本，不作为当前运行入口。若需复现，先独立恢复旧输入和环境，按迁移登记适配旧路径、指定新输出并完成兼容验证；相关训练仍需单独授权。

| 已退役文件 | 退役原因 | 支持的替代方案 |
|---|---|---|
| `train.py`、`train_script.sh`、`test.py`、`test_script.sh` | 依赖已删除的数据路径和过时的输出布局。 | `tools/train.py` |
| `run_pilot_10epochs.py` | 固定使用已退役的 M5 scale head。 | `tools/train.py --model M1 --epochs 10 --tag _pilot` |
| `test_cif.py` | 依赖已删除的 v1 CIF 缓存。 | `tools/infer_b7_cif.py`（B7 blind） |
| `evaluate_and_plot.py` | 读取过时的检查点和预测文件名。 | 训练入口的评估功能加 `results/reports/history_*.csv` |
| `v1_csv2npy.py`、`v1_data_check.ipynb` | 原来混放在本地数据归档中的 v1 实现，字节原样保留。 | 当前 Q1 数据约定与已批准的恢复计划 |

重新引入已退役路径需要新的设计和兼容性测试。
