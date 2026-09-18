# tools/legacy — 已退役入口（冻结存档，只读）

> 执行 Backlog E8（入口去重）。以下文件不再维护、不再引用，保留仅为历史可查。
> 现行入口：根目录 `run_ablation_experiments.py`（训练/评估）+ `cif2dos.py`（盲推）。

| 文件 | 退役原因 | 替代 |
|---|---|---|
| `train.py` + `train_script.sh` | 旧通用训练链，引用的 `data/csv2npy.py` 已不存在，输出旧布局 `output/config/...` | `run_ablation_experiments.py` |
| `test.py` + `test_script.sh` | 同上旧测试链（`training_options.yaml` + `checkpoint_best.pth` 口径） | runner 内建 `evaluate_split` |
| `run_pilot_10epochs.py` | 硬编码已作废的 M5 ScaleHead（零梯度），pilot 已由 `--epochs 10` 接管 | `run_ablation_experiments.py --model M1 --epochs 10 --tag _xxx` |
| `test_cif.py` | 读不存在的 `./data/train4w2023/test_cif/`（v1 路径） | `cif2dos.py`（E8 留一的胜者） |
| `evaluate_and_plot.py` | 读旧布局 `output/config/.../training_options.yaml` + 旧 `pred_edos.npy` 命名 | runner 内建评估 + `results/history_*.csv` |

恢复任一文件即视为重开 E8，需重新上会。
