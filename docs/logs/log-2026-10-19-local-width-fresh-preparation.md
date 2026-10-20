# 2026-10-19 迁移后从零重建 A64/W1 容量对照

状态：**入口与短检完成，正式训练未启动**。用户确认继续使用原V100 32GB配置和完整数据。
旧W1停在9轮，无容量结论；新平台只保留Git源码和冻结数据，采用新标签，从头训练两组，不恢复旧权重。

## 范围与入口

- A64：完整A，标量64/MLP隐藏128；W1：完整A，128/256。
- 固定seed42、各35轮、batch32、lr5e−5、FP32、修复版数据、staged L2/G6与原联合损失。
  更新公式、残差、块数、全局设置、原子对微批策略和缓存策略均不变。
- 两任务平等：valid `balanced_blind_r2` 最大选点；分别报告eDOS/phDOS blind中位R²和失败率。
  W1−A64仅判断容量增益；**本轮不重训I1，也不作相对I1的基线升级结论**。Oracle和同组成谱差仅诊断。
- 联合增益95%材料配对区间下界>0且双任务R²下降≤0.02、失败率增加≤1 pp，才满足容量升级数值条件。
  材料bootstrap 2,000次、seed20261016，两任务共用索引；谱差按组成组bootstrap 2,000次、seed42。

| 文件与符号 | 职责 |
|---|---|
| `tools/eval/local_width_fresh_compare.py`：`comparison_configs`、`validate_preflight`、`execute` | 直接复用版本内冻结配方，不读取历史配置/权重/报告；绑定源码、环境、数据、共同初始化及恢复检查，串行运行两组 |
| 同文件 `analyze`、`comparison_report`、`structural_report` | 对新训练的两组完整train/valid推理，报告容量对照与谱差诊断，保留配置/权重/历史哈希 |
| `tools/checks/check_local_width_fresh.py`：`check`、`code_fingerprints` | 复用现有几何、恢复和计时检查；从无历史产物的状态构建两组并做有限批次更新 |
| `tests/pipeline/test_local_width_fresh.py` | 配方、无权重初始化、旧协议拒绝、环境/源码/数据漂移、失败停止、不自动续跑和两组分析的回归；登记既有静态入口 |

模型实现及旧入口保持原样。旧 `local_width_compare.py` 仍要求历史A64/I1/I2认证，不用于这次迁移后的从零实验。
新入口默认只打印两条训练命令；`--run`才训练，`--summarize`只读进度，`--analyze`只分析完成的运行。

## 实际验证与成本

在临时干净副本中，仅复制Git跟踪的源码/元信息和当前33个数据载荷，**不复制 `results/` 或 `archive/results/`**。
新入口完整GPU预检通过，耗时85.22秒；随后在当前工作区重新认证该报告通过，锁定63份源码。
证据保存在 `results/runs/local_a_width_fresh_20261019/` 的 `preflight_final.json` 与 `clean_restore_verification.json`。
预检SHA256：`e1a4e0cc155bd58bf8537568ce57c62bb9c6eaf07c0256fd43177c212cca4d01`。

- 实际train/valid视图18,704/2,313，数据manifest及所有载荷校验通过；test仅做文件完整性校验，无模型评估。
- A64完整未训练初值由代码重建；W1记录的窄模型初始化与其一致，非局部共同张量哈希相同，训练起始RNG和批序一致。
- 两组真实训练梯度和blind/oracle输出有限；FP32/FP64完整输出及物理谱的E(3)/原子重排检查通过。
- 两组权重、AdamW、RNG恢复一致；异宽度与异选点契约拒绝；只对临时模型更新，不产生正式checkpoint或epoch历史。
- 预检同时锁定当前源码、Python/主要依赖版本、CUDA运行时、GPU型号/显存；在另一环境使用前须重新预检。
- 59项相关CPU回归通过（114.71秒，含新增9项）；全源码Ruff、新增3份Python内存编译、830个候选文件边界检查通过。
  5份相关文档的84处本地引用存在，差异空白检查通过。本轮未提交Git；新平台需先同步这两份新入口。

相同真实批次，预热后的完整AdamW步中位耗时：

| 批次（结构/原子/槽宽） | A64秒 | W1秒 | A64峰值MiB | W1峰值MiB |
|---|---:|---:|---:|---:|
| 32/352/40 | 0.360 | 0.456 | 7,808 | 11,691 |
| 32/380/94，微批23+9 | 0.508 | 0.638 | 7,534 | 11,334 |
| 32/510/160，微批31+1 | 0.584 | 0.785 | 7,805 | 11,675 |

短测不覆盖整轮缓存增长。[迁移记录](log-2026-10-18-platform-migration-handoff.md)的W1实际训练峰值为30.42 GiB，
不能以本次11.42 GiB推断整轮上限。依据原A64约4.64 h和扩宽耗时，**两组35轮合计规划10–13 h单卡**；
这是历史与短测推断，正式运行仍需监测显存、进度和非有限值。OOM或异常即停，不自动改变batch、精度或预算。

## 新平台交接

先拉取包含本次入口的代码，恢复完整 `data/crystal_q1_v1/` 及必要参考文件，在**新平台本机**执行：

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/checks/check_local_width_fresh.py --tag-prefix migr20261019 \
  --out results/runs/local_a_width_fresh_20261019/preflight_final.json

PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_width_fresh_compare.py --tag-prefix migr20261019 \
  --preflight results/runs/local_a_width_fresh_20261019/preflight_final.json \
  --out results/runs/local_a_width_fresh_20261019 --run
```

若本机已有相同预检，先核验后复用；源码/环境不一致或存在部分训练产物时停下报告，不覆盖、不自动续跑。
使用安装了项目依赖的Python环境，不硬编码旧服务器解释器路径。OpenCode记录PID、起止时间及退出码，完成或异常时汇报。
不使用test调参，不追加I1、seed、预算或变体，不自动提交或替换基线。
