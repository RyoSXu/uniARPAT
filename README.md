# uniARPAT

uniARPAT 从未弛豫的晶体结构预测电子态密度和声子态密度（eDOS 与 phDOS）。
这是一个研究型代码库：可复现性、清晰的实验对比和可审计的数据边界，与模型改动同等重要。

## 当前基线

当前参考实验为 B7 `_e9ctl`：使用 Q1 清洁数据池的 M1 模型、总和归一化的
KL/W1/Huber 损失、H1 eta/gamma 盲推理头，以及 0.05 dropout。35 个 epoch 中最佳
检查点为第 33 个：eDOS 的中位 R² 为 0.518（失败率 5.73%），phDOS 为 0.741（失败率
3.50%）。指标定义与报告规则见 [`docs/glossary.md`](docs/glossary.md)，正在进行的
工作见 [`docs/status.md`](docs/status.md)。

## 从这里开始

```bash
pip install -r requirements.txt
python3 -m unittest discover tests
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _pilot
```

已批准的参考训练请使用唯一的标签：

```bash
python3 run_ablation_experiments.py --model M1 --epochs 35 --tag _e9ctl
```

未经明确批准，不得运行 `--model all --epochs 100`。不得在不同归一化方案之间复用
检查点。训练前必须确认 Q1 缓存及其清单：

```bash
python3 -c "import json; print(json.load(open('data/train4ARPAT/manifest.json')))"
```

## 目录说明

```text
uniARPAT/
├── run_ablation_experiments.py  # 训练与评估入口
├── cif2dos.py                   # 从 CIF 预测 DOS 的入口
├── thermo_props.py              # 热力学后处理库
├── model/ datasets/ utils/      # 模型、数据集适配层与通用工具
├── configs/                     # 受版本控制的模板默认值
├── tests/                       # 单元测试与回归测试
├── tools/
│   ├── data/                    # 受保护的数据获取、处理与审计工具
│   ├── eval/                    # 可复用的结论判定与分析脚本
│   └── legacy/                  # 只读的已退役入口
├── docs/                        # 当前工作文档、日志和设计模板
├── data/                        # 本地缓存；清单和小型索引会纳入版本控制
├── index/                       # 固化的数据划分与参考表
├── results/                     # 受版本控制的正式实验 CSV
└── output/                      # 本地检查点与运行日志（忽略）
```

## 文档与协作

以 [`docs/index.md`](docs/index.md) 作为文档导航。贡献者和 agent 必须遵循
[`AGENTS.md`](AGENTS.md) 与 [`docs/workflow.md`](docs/workflow.md)：推进实验前先记录
结论；每个完成的任务写一份日志；当前状态与历史记录分开维护。数据边界由
[`docs/data.md`](docs/data.md) 定义。

## 推理

`cif2dos.py` 是用于兼容 M4 的旧入口。它可以对 CIF 文件运行兼容的 M4 检查点，但尚未
接入当前 B7 M1 盲推理模型的导出流程；不得据此宣称得到 B7 推理结果。提供兼容的检查点
和 CIF 文件后，再选择输出目录：

```bash
python3 cif2dos.py --cif structure.cif --weights model.pth --output predictions/
```

可运行 `python3 cif2dos.py --help` 查看支持的选项。检查点必须与训练时的模型和归一化
配置相匹配。
