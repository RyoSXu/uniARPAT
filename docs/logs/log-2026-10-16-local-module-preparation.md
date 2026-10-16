# 2026-10-16 局部 A/B+ 选型准备与短检

对应[选型方案](../design/design-2026-10-16-local-module-selection.md)。本轮完成单 A 新 blind 入口与短检；
正式35轮训练尚未启动。I1/B+复用已核验结果，模型公式、宽度、残差、损失和全局编排均未改变。

## 实现与来源

| 调用位置 | 文件与职责 |
|---|---|
| 配置、参考认证与运行 | `tools/eval/local_module_compare.py`：`candidate_config`、`reference_identity`、`verify_preflight`、`execute`；默认只打印单 A 命令 |
| 报告 | 同文件 `comparison_report`、`analyze`：A−B+、B+−A与A−I1；路线判断与基线升级分开，仅对A重新推理 |
| 真实短检 | `tools/checks/check_local_module.py`：`check`、`code_fingerprints`；复用现有完整A检查与原子重排检查 |
| 共享检查与统计 | `check_i1_a.check`允许显式注入新协议并检查选点契约隔离；`global_norm_compare.comparison_report`允许指定统计seed，默认仍为20261013 |
| 回归 | `tests/pipeline/test_local_module.py`，登记到既有静态检查入口 |

新入口认证既有预检、独立核验、完成队列、配置、best/latest、history、预测导出及当前train/valid数据身份。
历史源码指纹保持原样；新预检另锁当前57份源码。旧实验入口默认配方和checkpoint契约保持原样。
不完整运行拒绝自动续跑，失败即停；已完成候选须经身份核验，分析目录禁止覆盖。

## 实际核验

预检：[preflight_final.json](../../results/runs/local_module_selection_20261016/preflight_final.json)，
SHA256：`098127032d46b77b8cffebb8e2d5d0de9db46856c1d60fb4cd416c8875193572`。
V100 32GB实测67.93秒；临时模型做有限批次更新，未产生正式checkpoint或epoch历史。

- 与历史A比较，完整未训练初始化记录及组装后的模型、损失、优化器配方完全相同；实验配置只改标签和选点。
- train/valid为18,704/2,313；sampler seed=0，首批材料顺序和训练起始RNG与已完成新I2一致。
- 两批真实训练数据的损失与全部可训练参数梯度有限；完整blind/oracle验证输出有限。
- 四个真实valid结构的平移、旋转、反射及原子重排通过。E(3)最大相对误差：FP32为1.31e−6、FP64为4.58e−15；
  原子重排分别为1.09e−6、5.01e−15，低于对应1e−4/1e−9容限。
- 权重、AdamW状态及RNG恢复一致；拒绝I2契约和旧Oracle选点的A契约。
- 运行入口重新认证该预检通过，57份当前源码指纹一致。候选正式目录为空，35轮历史尚不存在。
- 新增13项回归通过：选点与旧契约隔离、双向路线判断、失败率保护、参考漂移拒绝、仅A推理、失败停止与不自动续跑。
- 全仓静态入口通过：Ruff、内存语法编译、451项测试（337.66秒）与811个候选文件边界检查；4份文档的76处本地引用通过。

以上检查不代表完整等价晶胞/超胞验收，也不提供新的准确率结论。

## 成本

以下为含AdamW更新、缓存预热后的单批中位数与峰值分配；均为FP32。

| 批次 | 结构数/总原子数/最大槽宽 | 微批 | 耗时 | 峰值MiB |
|---|---|---|---:|---:|
| 普通1 | 32/352/40 | 32 | 0.363 s | 7,808 |
| 普通2 | 32/380/94 | 23+9 | 0.512 s | 7,534 |
| 含最大结构 | 32/510/160 | 31+1 | 0.592 s | 7,805 |

短测不覆盖全数据的角度对数量和缓存增长。正式规划仍以历史A队列4.64 h、峰值25,146 MiB为依据，
预留 **4.5–5.5 h单卡**；不把短测7.6 GiB当作整轮显存上限，不据两批数据声称整轮加速。

## 交接

产物根目录 `results/runs/local_module_selection_20261016/`，标签前缀 `locsel20261016`。
正式候选目录为 `results/runs/ablation_m1_lms_locsel20261016_i3_k2_g6_staged_s42_e35/`。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/tefscloud/miniconda3/envs/torch/bin/python -B tools/eval/local_module_compare.py \
  --tag-prefix locsel20261016 \
  --preflight results/runs/local_module_selection_20261016/preflight_final.json \
  --out results/runs/local_module_selection_20261016 --run
```

OpenCode记录PID、起止时间和退出码，监测队列、训练日志及GPU；异常即停，不自行改源码、预算或重试。
完成后核验35轮与blind选点，报告两任务指标、联合配对区间和独立的路线/升级判断；不使用test。
主表材料bootstrap为2000次、seed20261016；谱差仅作诊断，复用历史组成组统计的2000次、seed42。
I1/I2的12份预测/摘要导出逐字节复用；不自动替换I1，不扩展宽度、残差、I4或新公式。
