# Mat2Spec 外部基线（移植）

Mat2Spec（Kong et al., Nat. Commun. 2022）在本项目数据划分上的移植。方案、口径与预算见
[设计文档](../../docs/design/design-2026-11-10-mat2spec-baseline.md)；本页只记录来源、目录和每一处改动。

## 来源与许可

| 内容 | 来源 | 许可 | 本地位置 |
|---|---|---|---|
| Mat2Spec 代码 | `github.com/gomes-lab/Mat2Spec`，提交 `5ec842f104d91a944e63b334ad150e14ce1b8698`（2022-04-18），2026-11-10 浅克隆 | MIT（`upstream/MIT LICENSE.txt`，Copyright (c) 2021-2022） | `upstream/`（未改动；逐文件 SHA-256 见 `UPSTREAM_SHA256.txt`，`.git` 已移除以免成为嵌套仓库） |
| CGCNN 元素嵌入表 `atom_init.json` | `github.com/txie-93/cgcnn`，`data/sample-regression/atom_init.json`（Mat2Spec 仓库不附带，其 README 指向 CGCNN） | MIT（`upstream_cgcnn/LICENSE`，Copyright (c) 2018 Tian Xie） | `upstream_cgcnn/`，SHA-256 `cb90bd22…0f1894` |

`upstream/` 与 `upstream_cgcnn/` 是下载内容，只作数据读取和比对，不从其中运行解释器、不加入 `sys.path`。
`*.json` 被仓库 `.gitignore` 忽略：若以后提交，`atom_init.json` 需加例外或在说明中保留下载步骤（未提交，待用户决定）。

## 目录

| 路径 | 内容 |
|---|---|
| `port/mat2spec_model.py` | 上游 `Mat2Spec_Codes/Mat2Spec/Mat2Spec.py` 的副本，仅含下表 7 处改动；`port/mat2spec_model.diff` 是完整差异 |
| `port/pytorch_stats_loss.py` | 上游同名文件的逐字节副本（Wasserstein 损失） |
| `adapter/data.py` | 自写：`crystal_q1_v1` 的 train/valid → Mat2Spec 输入；拒绝读取 test |
| `adapter/train.py` | 自写：单臂训练（复现上游训练循环）、逐轮 valid 评分、两种 best、恢复与状态文件 |
| `adapter/project_metrics.py` | 自写：按本项目口径评分（直接调用项目的 E 与 R² 函数） |
| `adapter/evaluate.py` | 自写：`selfcheck` / `export` / `report` |
| `adapter/run_queue.py` | 自写：脱离会话的顺序排队 |
| `../../tests/baselines/test_mat2spec_port.py` | 12 项测试 |

未使用的上游文件：`data.py`、`utils.py`、`file_setter.py`、`SinkhornDistance.py`、`train_Mat2Spec.py`、`test_Mat2Spec.py`（数据加载与训练循环由 `adapter/` 按其逻辑重写，见下）。

## 对上游模型文件的改动（共 7 处，全部在 `port/mat2spec_model.py`，带 `[port N]` 注释）

| # | 改动 | 原因 | 是否改变计算 |
|---|---|---|---|
| 1 | 删除 `from torch_scatter import scatter_add` | 未使用；当前环境无 torch_scatter | 否 |
| 2 | 删除 `from Mat2Spec.utils import *` 与 `SinkhornDistance` 导入；Wasserstein 损失改为相对导入 | 包路径不同；Sinkhorn 仅在注释掉的代码中用到 | 否 |
| 3 | `device = set_device()` 内联为等价的一行 | 随 #2 | 否 |
| 4 | 删除模块级 `sinkhorn = SinkhornDistance(...)` | 未使用 | 否 |
| 5 | `softmax(alpha, edge_index_i, size_i)` → `softmax(alpha, edge_index_i, num_nodes=size_i)` | PyG ≥2 的第三个位置参数是 `ptr`，原写法会被误解 | 否（恢复 PyG 1.5 语义） |
| 6 | `GNN.forward` 不再读取 `data.cluster` | 上游读出后从未使用；因此不必做逐材料谱聚类 | 否 |
| 7 | `GAT_Crystal` 的 `MessagePassing.__init__` 显式传 `node_dim=0` | PyG 1.5 默认沿第 0 维聚合，PyG ≥2 默认 `-2`（对三维消息即注意力头维，直接报错） | 否（恢复 PyG 1.5 语义） |

移植环境：现有 `torch` 环境（PyTorch 2.2.1＋cu121、PyG 2.6.1、pymatgen 2025.5.2），未新建环境。
#5、#7 由测试 `test_gat_layer_matches_explicit_pyg15_semantics` 用显式公式逐项核对（中心原子为 `edge_index[0]`，按中心原子做 softmax 与求和）。
无法在 PyTorch 1.4/PyG 1.5 上并排运行，因此“与原环境数值一致”是按文档语义与显式公式核对的推断，不是实测。

## 自写适配相对上游流程的差异

数据与图：

- 结构来自本项目的 FP32 `cell`／`frac_coords`，目标为 E0 128 箱 eDOS、P0 64 箱 phDOS（上游为 −4～4 eV 128 箱、约 0～1000 cm⁻¹ 51 箱），输出维度相应为 128／64。
- 近邻：与上游相同的规则（8 Å 内按距离取前 12 个，不足时用原子 0、距离 9 Å 填充；41 维高斯展开）。实现改用 `get_neighbor_list`＋稳定排序以提速，
  与上游原语句 `get_all_neighbors`＋`sorted` 在抽查的 valid/train 结构上（随机 85 个，加上全部 10 个触发填充的稀疏晶体）索引与距离完全一致；等距近邻的先后取决于 pymatgen 版本，无法与上游当年的 2022.0.14 对齐。
  触发填充的只有 train 9 个晶体（33 个原子）和 valid 1 个晶体（4 个原子）。
- 不计算 `cluster`（见 #6）；用自写的 GPU 批处理代替 PyG `DataLoader`（测试核对与逐晶体拼接一致）。

目标：

- **标签单位默认取每原子谱 `y/N`**（`--target_unit per_atom`）。上游直接用整胞 eDOS（states/eV）；本项目的 E 按每原子定义，且 Mat2Spec 的图读出是按原子取均值、看不到原子数，故 blind 总量取“每原子预测 × N”。`--target_unit cell` 保留上游字面做法。SumNorm 形状不受此影响。
- Standard（逐箱 z-score）的均值／标准差**只用有效 train 拟合**（上游用全体数据，含 valid/test）。标准差为 0 的箱置 1（本数据两种谱均无此情况）。
- SumNorm／MaxNorm 的训练目标分母加 `1e-8`（上游只在验证分支加）：train 有 7 条、valid 有 1 条全零 phDOS，按上游训练分支会得到 NaN。全零谱的目标为全零，对 KL 项贡献为 0。

训练：

- 与上游一致：AdamW（lr 1e-3，weight decay 1e-2）、StepLR 每 1/4 总轮数减半并按迭代步进、200 轮、seed 1、打乱、保留最后不满的批、`compute_loss` 及其权重（1／1.1／0.1）、dropout 0.5、K=10、3 层 4 头 128 维、标签分支输入未归一化的标签。
- 不开启 `torch.autograd.set_detect_anomaly(True)`（调试开关，只影响速度）。
- 每轮 valid 用确定性预测（潜变量取均值，即上游 `test_Mat2Spec.py` 的做法）；上游训练循环内的验证因 `args.train` 未复位而带采样噪声。
- 保存两种 best：`best_upstream.pth`（上游判据：valid `mae_x_ori` 最小）与 `best_project.pth`（本项目判据：valid E 最小，Standard 臂用 blind E，其余用 oracle E）。
- phDOS 的 batch：上游脚本为 8（其 phDOS 训练集仅 1,220 条），本移植默认 256；见设计文档的配方一节。
- 新增：非有限值即停、状态与心跳、指纹（配置／数据／协议／源码）、从 `latest.pth` 恢复、拒绝覆盖。

## 常用命令

```bash
PY=/opt/tefscloud/miniconda3/envs/torch/bin/python
$PY -m unittest tests.baselines.test_mat2spec_port
$PY baselines/mat2spec/adapter/evaluate.py selfcheck
```

训练、导出与报告命令见设计文档。
