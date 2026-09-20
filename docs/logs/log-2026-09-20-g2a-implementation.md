# 日志：2026-09-20 — G2a 实施与预检（代码、单测、Q1 边数审计；未训练）

## 范围

- 任务与假设：按 `docs/design/design-g2-periodic-multi-image-message.md` 实施 G2a 单因素模块：
  保留 B7 六层 dense attention，仅增加全量周期多镜像 `(i,j,T)` 径向边条件 Value 残差
 （`R=5.5 Å`、不裁边、六层零初始化 `alpha`、64 中心 RBF、`1/sqrt(d_i)` 聚合）。
- 改动的文件或配置：
  - 新增 `utils/g2_periodic_edges.py`：精确周期多重边构建（`K=ceil(R/s_min+0.5)` 保守枚举、
    中心胞 `delta` 稳定化、真值 `T` 存储）、RBF/cutoff 辅助函数、纯 CPU 接口；
  - `model/transformer.py`：新增 `PeriodicEdgeMessage`、每层独立残差、`use_g2/g2_r_cut` 开关、
    与 G1 互斥断言、关闭路径无新增属性；
  - `utils/experiment_config.py`、`run_ablation_experiments.py`：仅增加默认关闭的 `--use_g2`，
    `g2_r_cut` 固定 `5.5` 写入 `config_used.yaml`，不暴露邻居数/平移范围扫描参数；
  - 新增 `tests/test_g2_periodic_edges.py`：设计第 1–5 关卡确定性合同 + Q1 CPU 冒烟；
  - 新增 `tools/eval/g2_edge_audit.py`：不读标签，遍历 Q1 三划分输出 `results/g2_edge_audit_q1.csv`。
- 未改动：默认 YAML、Q1 缓存、G1 文件、训练损失、既有结果文件；未启动任何训练。

## 证据

- 命令、标签、随机种子和 epoch 预算：未运行 `run_ablation_experiments.py` 训练（`_g2ctl/_g2edge`
  均未启动）。仅运行 `python3 -m unittest discover tests` 与
  `python3 tools/eval/g2_edge_audit.py --batch 64`（CPU）。
- 结果文件与测试：
  - `python3 -m unittest discover tests`：64/64 通过（含新增 13 项 G2a 合同），约 23 秒；
  - `python3 -m unittest tests.test_g2_periodic_edges -v`：13/13 通过；
  - `results/g2_edge_audit_q1.csv`：23,306 行（train 18,706 / valid 2,313 / test 2,287），列为
    `split,idx,n_atom,K,n_edges,mean_indegree,max_indegree`；
  - 关闭路径：`use_g2=False` 时 `encoder` 无 `g2_msgs` 属性，参数量与 B7 一致；
    开启后新增 3,351,558 参数（六层各约 0.559M）：B7 71,078,914（71.079M）→ G2 74,430,472
   （74.430M，+4.7%，低于 5% 预算）；
  - `alpha=0` 的 eval 前向在 `edos/phdos` 上与 B7 逐元素 bitwise 相等（`torch.equal`）；
    `alpha` 首步取有限非零梯度，更新一次后 `W_v/W_g/W_o` 取有限非零梯度；
  - Si diamond primitive fixture（`a=5.431/sqrt(2) Å`、角 `60°`、`R=5.5`）：每接收原子
    34 条入边（18 自镜像 + 4 首壳 `2.351692 Å` 异原子边），全图 68 有向边（36 自镜像 +
    8 首壳），`K=3`，与设计一致；
  - 枚举完整性：在随机斜晶胞、小晶胞、高长宽比晶胞上与 `K+3` 暴力枚举的
    `dst/src/T/distance` 逐条一致，无 `i=j,T=0`；
  - 物理合同：反向边配对等距、严格 `<R`、quintic 值域端点、padding 排除、输出有限、
    `frac+integer` 距离集合一致（容差 `1e-5`）；
  - 等价合同：平移、联合置换、基矢交换、幺模变换后距离多重集一致（容差 `1e-5`，
    实测基矢交换约 `5e-07`、幺模约 `1e-06`），置换下消息等变、整模型读出不变。
- 指标：`e med/fail + p med/fail + (test|valid, epN, oracle|blind, Q1|pre-Q)` —— 本轮无训练，
  故无指标结果。
- Q1 边数审计摘要（`R=5.5 Å`，CPU，无退化晶胞报错）：
  - train：每结构边数 p50/p95/p99/max = 250/1576/2729.5/6344（max 行 10088，`n_atom=80`）；
    每原子入度（27,4505 个原子 pooled）p50/p95/p99/max = 41/68/83/140，均值 40.27；
    `K` p50/p95/p99/max = 2/3/4/7（max 行 15136）；
  - valid：边数 250/1509.6/3196.64/6704（max 行 1056，`n_atom=80`）；入度 41/68/85/122；
    `K` 2/3/4/6；
  - test：边数 250/1596.4/2924.48/5600（max 行 1151，`n_atom=56`）；入度 41/68/86/122；
    `K` 2/3/4/7；
  - train 有 5 个零边结构（idx 3876/9743/9918/13321/15491），属大晶胞孤立像，G2 消息退化为
    恒等（`E=0` 直接返回 `h`），无 NaN 风险。
- 首次使用 `docs/glossary.md` 未定义的术语时，给出中文解释：本轮无新增术语；
  `indegree` 指接收端真实入边数 `d_i`，`K` 指每样本枚举半宽。

## 结论

- 状态：pending。
- 原因：实现、单测、Q1 边数审计三项已按设计通过；但设计要求的 batch 32 资源门禁
  （同一 V100、同一批次比较 B7 与 G2 单步峰值显存和耗时，任一超 1.25 倍或 OOM 即停）
  尚未执行，故 `_g2ctl/_g2edge` 的 Q1 M1×10 成对 pilot 不得启动。

## 交接

- 下一项关卡工作：在同一 V100、batch 32、同一输入批次下实测 B7 与 G2 的单步峰值显存和
  耗时（先 CPU 验证数值合同已完成，不用作成本结论）；通过后才运行
  `python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _g2ctl` 与
  `--tag _g2edge --use_g2`。
- 对 status、backlog 和 decisions 的更新：未改 `status.md`（当前仍为“G2a 设计完成，
  等待实施预检”；待资源门禁通过后再更新预检状态）；未改 `decisions.md`（尚无可用结论）。
