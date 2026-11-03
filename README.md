# uniARPAT

从完整晶体结构联合预测电子态密度（eDOS）与声子态密度（phDOS）。当前研究目标是让同组成、不同结构材料的谱差预测获得可泛化收益。

接手先读[当前状态](docs/status.md)与[开发交接](docs/handoff.md)；按[文档导航](docs/index.md)查契约和证据。
当前研究工作点为 **完整 A64 的 LR2 形式＋E＋S、staged L2/G6**。总体误差已有单 seed 改善证据，同组成谱差尚未检出明确改善。

## 环境与数据

```bash
python -m pip install -r requirements.txt ruff
git config core.hooksPath tools/checks
python tools/train.py --help
```

Git 保存代码、文档和有限元信息；训练数据、checkpoint 与原始预测须单独恢复。
当前输入为 `data/crystal_q1_v1/`，有效 train/valid/test 为 18,704/2,313/2,287。
先按[交接清单](docs/handoff.md)校验数据和所需产物，再执行获准方案。
数据字段、来源和恢复方式见[数据约定](docs/data.md)与[中英文数据卡](docs/dataset-card-zh.md)。

命令行保留历史默认，裸训练命令不代表当前配方；当前 loss、选点、显式开关及 checkpoint 边界见[固定契约](docs/decisions.md)。
新实验使用唯一标签，选型只用 train/valid。长时间实验由准备 agent 提供执行提示词，交给用户指定的新 agent 运行并监控，见 [AGENTS.md](AGENTS.md)。

## 开发入口

- [代码地图与 Git 边界](docs/repository.md)：模型、数据、训练与导出的调用关系。
- [研究流程](docs/workflow.md)：方案、预检、实验交接和结果核验。
- [测试说明](tests/README.md)：可用检查及本地数据依赖；完整检查需要相应历史数据。
- [分析工具](tools/eval/README.md)、[数据工具](tools/data/README.md)：用途、原协议与执行边界。

历史 B7 CIF 入口 `tools/infer_b7_cif.py` 仅支持旧 `_e9ctl` ep33，不能加载当前 LR2 checkpoint；当前尚无等价的 LR2 通用 CIF 命令行入口。
