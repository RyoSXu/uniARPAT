# uniARPAT

uniARPAT 从未弛豫的晶体结构预测电子态密度和声子态密度（eDOS 与 phDOS）。
这是一个研究型代码库：可复现性、清晰的实验对比和可审计的数据边界，与模型改动同等重要。

## 当前基线

当前研究基线是 ZP100 `_eidzproj100_s42`：Q1/M1、seed 42、100 轮、best epoch 89；
元素入口为 Embedding → LayerNorm → Linear。B7 `_e9ctl` 是最优指标参考：使用 Q1 清洁数据池的 M1 模型、总和归一化的
KL/W1/Huber 损失、H1 eta/gamma 盲推理头，以及 0.05 dropout。35 个 epoch 中最佳
检查点为第 33 个：eDOS 的中位 R² 为 0.518（失败率 5.73%），phDOS 为 0.741（失败率
3.50%）。指标定义与报告规则见 [`docs/glossary.md`](docs/glossary.md)，正在进行的
工作见 [`docs/status.md`](docs/status.md)。

## 从这里开始

```bash
pip install -r requirements.txt
python3 -m unittest discover tests
python3 tools/train.py --help
```

训练配方与预算以获准设计为准，使用新的唯一标签；保留已有 `_e9ctl` 和 `_eidzproj100_s42` 产物。

未经明确批准，不得运行 `--model all --epochs 100`。不得在不同归一化方案之间复用
检查点。训练前必须确认 Q1 缓存及其清单：

```bash
python3 -c "import json; print(json.load(open('data/train4ARPAT/manifest.json')))"
```

## 目录说明

```text
uniARPAT/
├── model/           # 当前模型计算
├── utils/           # 数据集及可导入支持库
├── tools/           # train.py、infer_b7_cif.py、training-defaults.yaml
│   ├── data/        # 获取和处理工具
│   ├── eval/        # 当前诊断、验收和绘图
│   └── checks/      # 静态检查与 Git 载荷拦截
├── data/
│   ├── reference/   # 固定参考表和数据契约
│   └── train4ARPAT/ # 当前 Q1，本地数组；Git 仅清单和统计
├── results/
│   ├── reports/     # 经审查的当前摘要和有限证据
│   ├── figures/     # 当前正式图像
│   └── runs/        # 当前权重、日志、逐样本和中间产物（本地）
├── docs/            # 当前文档；design/logs 各自带模板
├── tests/           # model/data/pipeline 三类当前测试
└── archive/         # code/docs/data/results/metadata 五类历史材料
```

原始数据、数组、权重、逐样本文件和日志保持本地；报告只允许精确开放的小证据，单文件上限 2 MiB。
本机 Git 钩子检查提交和待推送历史；新克隆需要设置 `git config core.hooksPath tools/checks`。

完整目录职责、旧路径映射、文档日期和 Git 忽略规则见 [`docs/repository.md`](docs/repository.md)。
默认输入/输出按仓库根目录定位；显式指定的自定义相对路径按启动目录解释。

## 文档与协作

以 [`docs/index.md`](docs/index.md) 作为文档导航。贡献者和 agent 必须遵循
[`AGENTS.md`](AGENTS.md) 与 [`docs/workflow.md`](docs/workflow.md)：推进实验前先记录
结论；每个完成的任务写一份日志；当前状态与历史记录分开维护。数据边界由
[`docs/data.md`](docs/data.md) 定义。

## 推理

旧 M4 入口 `cif2dos.py` 已退役，历史可从 Git 恢复，不得据此宣称得到 B7 推理结果。

当前 B7 `_e9ctl` 的 CIF-only blind 推理使用独立入口。它要求明确的 epoch-33 checkpoint，
从 CIF 的元素组成计算 Z0 价电子数，并使用 H1 eta/gamma 头恢复 E0/P0 窗口内的物理总量：

```bash
python3 tools/infer_b7_cif.py --cif structure.cif \
  --checkpoint results/runs/ablation_m1_e9ctl/checkpoint_best.pth \
  --output results/runs/b7_cif_blind
```

输出目录包含每个 CIF 的 `*_b7_blind.npz`（bin 中心与谱）、同名 JSON 元数据和 `summary.csv`。
该结果是 blind 而非 oracle；入口拒绝不匹配的 checkpoint、无序 CIF 和超过 80 原子的 CIF 单胞。
