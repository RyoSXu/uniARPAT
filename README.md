# uniARPAT

从晶体结构预测电子态密度和声子态密度（eDOS 与 phDOS）。当前研究重点是同组成、不同结构材料的谱形差异预测。
当前模型、阶段结论和下一步见[项目状态](docs/status.md)，其他说明见[文档索引](docs/index.md)。

## 开始使用

```bash
pip install -r requirements.txt
python3 -m unittest discover tests
python3 tools/train.py --help
```

当前默认数据为完整结构修复版 `data/crystal_q1_v1/`。训练前检查本地清单：

```bash
python3 -c "import json; print(json.load(open('data/crystal_q1_v1/manifest.json')))"
```

数据字段、有效划分和旧 Q1 的兼容边界见[数据约定](docs/data.md)。训练配方和预算按获准设计执行，使用新的唯一标签；不同归一化方案不共用 checkpoint。
未经明确批准，不运行 `--model all --epochs 100`。实验只用 train/valid 选型，具体规则见[已确定约定](docs/decisions.md)。

## 代码与文件

模型在 `model/`，支持库在 `utils/`，训练、推理和诊断入口在 `tools/`，测试在 `tests/`。
设计与证据分别放在 `docs/design/`、`docs/logs/`；目录职责和历史兼容路径见[仓库约定](docs/repository.md)。

原始数据、权重、逐样本结果和日志保留本地；Git 只收录获准的小型元信息与证据，单文件上限 2 MiB。
新克隆需要启用文件检查钩子：

```bash
git config core.hooksPath tools/checks
```

## B7 CIF 推理

以下入口用于历史 B7 `_e9ctl`，要求 epoch-33 checkpoint，不适用于当前 I1 诊断模型：

```bash
python3 tools/infer_b7_cif.py --cif structure.cif \
  --checkpoint results/runs/ablation_m1_e9ctl/checkpoint_best.pth \
  --output results/runs/b7_cif_blind
```

输出为 blind 模式的 NPZ、JSON 元数据和 `summary.csv`；入口拒绝不匹配的 checkpoint、无序 CIF 和超过 80 原子的 CIF 单胞。

协作遵循 [AGENTS.md](AGENTS.md) 与[工作流程](docs/workflow.md)。
