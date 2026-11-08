# 开发交接

核对日期：2026-11-03。先读[状态](status.md)明确当前问题，再按本页恢复工作；[代码地图](repository.md)与[固定契约](decisions.md)按需查阅。

## 接到任务先确认

1. 检查 `git status --short` 与当前提交，保留已有改动；明确本轮是调查、实施还是实验执行。
2. 当前研究工作点是 **staged LR2＋E＋S、phDOS 头 2 层（h2）、best ep24**，不是裸 CLI 默认或历史 A64/H1。当前主问题是同组成谱差泛化；历史候选不自动续跑。
3. 查相关设计和最终运行日志。文档记录了历史结果，不保证所引用的机器本地数组、权重或环境仍存在。
4. 核对数据与所需产物后，再做短时测试/预检。长时间任务按[交接流程](workflow.md#长时间实验交给执行-agent)生成提示词，由用户指定新执行 agent。

## 代码之外必须保存什么

| 内容 | 当前路径（相对项目根） | 用途 |
|---|---|---|
| 完整修复版数据 | `data/crystal_q1_v1/` 与 `data/reference/` | 33 个数组＋manifest、网格与必要参考；只克隆 Git 不含训练数组 |
| 已采用 LR2 准备与协议 | `results/runs/spectral_residual_20261030/{report.json,protocol.json,source_snapshot/}` | 固定命令、数据/源码身份、train 常数和双任务保护 |
| h2 工作点模型 | `results/runs/ablation_m1_sbal_phead20261106_h2_s42_e35/`；准备与协议 `results/runs/phdos_head_20261106/{report.json,protocol.json}` | best ep24；分析 `results/runs/phdos_head_20261106/analysis/` |
| LR2 6 层头模型与配置（R42，匹配对照） | `results/runs/ablation_m1_sbal_lr2es20261030_lr2_s42_e35/` | `config_used.yaml`、best ep29、latest ep35；latest 含优化器/RNG |
| LR2 逐轮历史 | `results/reports/history_m1_sbal_lr2es20261030_lr2_s42_e35.csv` | 完整 ep1–35、资格保护与独立重建选点 |
| 匹配 A64 E＋S 对照 | `results/runs/ablation_m1_sbal_shape20261025_shape_a64_s42_e35/` | best ep25；不是旧 A64 ep31 的固定选点保护参照 |
| 正式导出与核验 | `results/runs/spectral_residual_20261030/analysis/` | 完整 valid 的逐材料表、原谱、配对结果和独立核验 |
| 后续诊断 | `results/runs/spectral_layout_20261101/`、`crystal_model_invariance_20261102/`、`cutoff_boundary_census_20261102/`（均在 `results/runs/` 下） | 已结束的编排对照、物理检查和边界统计 |

本次核对时，上述 LR2 准备报告、协议、best，以及三项后续诊断报告均在本机；它们不是 Git 附带资产。
迁机时另存权重、协议、配置、history、原谱与核验材料，按指纹验证；摘要不能替代原始证据。
复用准备工具还须按报告的 `input_sha256` 补齐其引用的对照导出与认证记录；上表是定位清单，不保证覆盖所有历史依赖。
若仅有 Git＋数据，可检查代码和从头设计新对照，但不能声称已恢复旧 checkpoint、继续原训练或复现原预测。
原协议已经冻结的常数和保护参照不能用新拟合结果冒充；缺少必需文件时先恢复或另定方案。
2026-10-18 的[平台迁移日志](logs/log-2026-10-18-platform-migration-handoff.md)仅作历史快照，W1 断点和旧迁移包不代表当前交接状态。

## 最小环境与数据检查

在已安装项目依赖的 Python 环境运行；本机可用解释器为 `/opt/tefscloud/miniconda3/envs/torch/bin/python`，它不是跨机器要求。

```bash
python -m pip install -r requirements.txt ruff
git config core.hooksPath tools/checks
python tools/train.py --help
python tools/checks/check_repository.py --worktree
python - <<'PYDATA'
from utils.crystal_cache import crystal_data_contract
contract = crystal_data_contract('data/crystal_q1_v1', normalization='sumnorm')
assert contract is not None, '完整修复版 Q1 manifest 缺失'
print(contract)
PYDATA
```

此检查验证 manifest 与所有数组指纹；`sumnorm` 是 loader 的谱形归一化标记，不代替 per_atom／E＋S 模型协议认证。
完整字段、排除两条材料及有效划分另见[数据约定](data.md)。来源许可和发布准备保留原核查日期，不能把本地发布候选当已发布附件。

## 开发与实验边界

- 先按[测试说明](../tests/README.md)选相关检查；部分完整迁移测试需要当前机器缺少的旧 Q1 数组，不能伪造文件使其通过。
- 新实验先确定问题和预算，生成新的准备/预检记录与标签；旧目录拒绝覆盖，旧训练命令不是新授权。
- 复用证据先验证数据、源码、协议与 checkpoint 身份；不能通过改写冻结哈希绕过不匹配。当前代码新增工具不等于旧协议失效，按各验证器实际依赖判定。
- 提交须用户明确要求；推送需另有明确授权。Git 仅保存获准元信息和小型证据，权重、数组、逐材料明细本地备份。
