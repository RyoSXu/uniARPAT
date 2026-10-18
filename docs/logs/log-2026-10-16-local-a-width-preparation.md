# 2026-10-16 完整 A 内部容量对照准备

对应[两档容量设计](../design/design-2026-10-16-local-a-width.md)。本记录为10月16日正式训练前的准备证据；
10月18日已确认W1完成9/35轮后中断，当前因平台迁移暂停，见[迁移交接](log-2026-10-18-platform-migration-handoff.md)。
本轮只准备A64（标量64/MLP隐藏128）与W1（128/256）；同时改变两种宽度，不能分别归因。
公式、残差、两次局部调用×两个内部块、随后六层全局、联合损失与35轮配方固定；I1仍为性能参照。

## 改动与数据流

| 调用位置 | 文件、符号与职责 |
|---|---|
| 局部计算与接入 | `model/periodic_messages.py` 的 `InvariantMessageBlock`、`PeriodicLocalMessage`：参数化A内部宽度；`LocalResidualLayer`、`LocalGlobalEncoder`、`Transformer`传递宽度，对外512维不变 |
| 初始化与权重身份 | `utils/local_width_experiment.py` 的 `local_width_config`、`initialize_local_width_model`、`local_width_contract`：W1独立契约，复制A64未训练的全部非局部消息张量，整个新局部分支独立初始化 |
| 训练、缓存与运行 | `tools/train.py`组装W1显式宽度并记录执行设置；`PeriodicGeometryCache`按宽度预留工作区；`tools/eval/local_width_compare.py`认证参考、运行单候选、复用12份参考导出并报告 |
| 短检与回归 | `tools/checks/check_local_width.py`比较相同真实批次；`tests/pipeline/test_local_width.py`覆盖直接公式、初始化、兼容、缓存、恢复和实验边界，登记既有静态入口 |

旧入口仍为64/128，参数键、默认数值实现、旧配置与checkpoint契约保持；不同宽度或选点不能混用权重。
共同初值排除**整个**局部消息分支，包括形状未变的反投影bias；外层LN、ZP、全局、Decoder与输出头共同初值相同。
不读取训练权重作初始化；历史checkpoint只认证身份。新预检锁当前源码，历史预检及导出不改写。
选点为valid两任务blind中位R²等权平均最大值，并列取较早轮次；Oracle与train/valid同组成谱差仅诊断。

## 真实短检

证据：[preflight_final.json](../../results/runs/local_a_width_20261016/preflight_final.json)。
V100 32GB，123.46秒；临时模型有有限批次更新，正式目录没有checkpoint或epoch历史。
完整train/valid为18,704/2,313，未使用test；批序、训练起始RNG和A64未训练初始化均匹配历史记录。

两档均通过真实训练梯度有限检查；W1通过blind/oracle输出有限检查、FP32/FP64完整输出及物理谱的
平移/旋转/反射与原子重排检查、权重/AdamW/RNG恢复检查，拒绝A64及异选点契约。
扩宽后的角度消息与独立直接拼接公式的输出、参数/标量/晶格/坐标梯度一致；缓存保持输出、梯度及随机流。
这不提供新的准确率结论，也不替代完整等价晶胞/超胞验收。

- 预检SHA256：`fd67553c307feb2f75a43f786fd3ad922614e71911b6b721705cb2122af2c910`；运行入口重新认证通过，当前60份源码指纹一致。
- W1 E(3)最大相对误差：FP32为2.00e−6、FP64为4.17e−15；原子重排分别为1.34e−6、3.47e−15，低于1e−4/1e−9容限。
- 实际共同非局部参数75,554,116个；局部消息参数由523,136增至1,766,144，逐张量初值与存储隔离通过。
- 全仓静态入口通过：Ruff、内存语法编译、463项测试（355.78秒，含新增12项）及818个候选文件边界检查。
  4份当轮文档的75处本地引用存在，差异空白检查通过；容量实现随10月18日迁移整理入库。

## 成本与限制

同批FP32完整AdamW步，预热后取3次中位数；两档均关闭重计算，保留原微批策略。

| 批次（结构/原子/槽宽） | A64秒 | W1秒 | 耗时比 | A64峰值MiB | W1峰值MiB |
|---|---:|---:|---:|---:|---:|
| 32/352/40 | 0.363 | 0.460 | 1.27 | 7,834 | 11,692 |
| 32/380/94，微批23+9 | 0.514 | 0.644 | 1.25 | 7,514 | 11,335 |
| 32/510/160，微批31+1 | 0.593 | 0.775 | 1.31 | 7,786 | 11,680 |

A64总参数76,077,252，W1为77,320,260；扩宽只增加局部容量。
以A64实测35轮队列4.64 h乘本次步耗时比，W1点估计约5.8–6.1 h，规划预留**5.5–7 h单卡**。
这是少量批次校准的推断，整轮耗时、缓存命中与尾部角度对分布仍可改变成本。
本次W1短测峰值11.42 GiB不是整轮上限；A64历史整轮峰值24.56 GiB。
W1沿用16 GiB几何缓存上限，但按128/256估算临时工作区并在压力下淘汰；仍需监测OOM，不自动改配方。

## 交接

产物根目录 `results/runs/local_a_width_20261016/`，标签前缀 `width20261016`。
只新增W1训练；A64复用已核验I3（best ep30），I1复用best ep33，不重新训练或推理参考模型。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/tefscloud/miniconda3/envs/torch/bin/python -B tools/eval/local_width_compare.py \
  --tag-prefix width20261016 \
  --preflight results/runs/local_a_width_20261016/preflight_final.json \
  --out results/runs/local_a_width_20261016 --run
```

OpenCode记录PID、起止时间与退出码，监测队列、日志、显存和有限损失；异常即停，不自动重试或续跑不完整候选。
完成后核验35轮与blind选点，报告两任务blind R²/失败率、联合分数与材料配对区间、时间与显存。
W1−A64的容量判断与W1−I1的基线升级判断分开；均沿用联合区间下界>0及双任务保护。
材料bootstrap 2,000次、seed20261016，两任务共用索引；谱差诊断按组成组bootstrap 2,000次、seed42。
不自动替换I1、追加公式/宽度/预算、修改源码或提交Git。
