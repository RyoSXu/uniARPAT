# 2026-10-20 完整 A 内部归一化对照准备

本页记录正式训练前的实现与短检；35轮现已完成并核验，见[运行结果](log-2026-10-21-local-internal-norm-run.md)。
对应[设计](../design/design-2026-10-20-local-internal-norm.md)。
仅新增LN1（内部pre-norm），复用迁移后A64（内部post-norm）；局部其余设置及全局头数/FFN是控制条件，尚未定型。

## 实现与调用链

`local_norm_compare` → `tools/train.py --local_route LN1` → `initialize_local_norm_model` →
`LocalGlobalEncoder` → `PeriodicLocalMessage(A)` → 四个 `InvariantMessageBlock(norm_first=True)`。

| 文件与符号 | 职责 |
|---|---|
| `model/periodic_messages.py`：`InvariantMessageBlock`、`PeriodicLocalMessage` | 新增默认关闭的内部pre-norm；归一化覆盖中心/邻居标量及完整更新分支，残差保留原状态 |
| `model/local_global_encoder.py`、`model/transformer.py`：局部组装 | LN1沿用A64的完整消息、宽度、外层LN/残差及六层修正全局；无新增参数或末端LN |
| `utils/local_norm_experiment.py`：`initialize_local_norm_model`、`local_norm_contract` | 复制全部未训练A64张量，保留调用方RNG；新契约拒绝与post-norm或旧选点checkpoint混用 |
| `tools/eval/local_norm_compare.py`：`reference_identity`、`verify_completed`、`analyze` | 认证历史参考，核对完成时的实际配置，只推理新候选并复用A64导出，形成主表和谱差诊断 |
| `tools/checks/check_local_norm.py`：`post_norm_compatibility`、`check` | 执行Git中原版本核对默认行为，重放参考预测，完成真实批次与资源短测 |
| `tools/train.py`、`utils/local_global_experiment.py` | 接入CLI、初始化、模型契约、实际归一化记录与训练起始RNG记录；旧入口行为保持 |
| `tests/pipeline/test_local_norm.py` | 12项新增检查，已登记既有静态检查入口 |

## 实际核验

最终预检：`results/runs/local_internal_norm_20261020/preflight_final.json`，耗时83.36 s，锁定66份当前源码。
SHA256：`da410f1e84408cdbb3fe38d5ed029193c27cf9f0a6f44453dc683bf297c23adb`。
初轮探针保留为同目录 `preflight_initial.json`；新增完成时配置门槛后重新运行最终预检，没有改写初轮指纹。

- 历史参考的63份源码逐项匹配Git提交 `1c29ba0d50ec2479f837624b6836ad98e2f453f0`，
  参考配置、best/latest、35轮history及14份train/valid导出与独立核验一致；不要求重训A64或保留旧I1。
- 完整post-norm模型与历史源码在CPU FP32/FP64下，输出及287份参数梯度逐位一致。
  所选A64权重重放32个真实valid材料，预测/目标谱与冻结导出匹配，最大相对误差2.48e−6。
- 两组全部初始张量相同，参数量均76,077,252；初始状态哈希为
  `2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。
  训练起始RNG与批序匹配，数据为完整train/valid 18,704/2,313。
- 新候选真实批次损失及全部可训练参数梯度有限，blind/oracle输出有限；FP32/FP64完整输出、物理谱的E(3)/原子重排通过。
  最大相对误差分别为1.52e−6、5.23e−15。权重/AdamW/RNG恢复、异契约拒绝通过。
- 88项相关CPU检查通过（新增12项、既有76项）；250份Python源码Ruff与内存编译通过。
  仅更新随后丢弃的短测模型；未创建正式checkpoint或epoch历史，未评估test，未提交Git。

## 成本

V100 32GB、FP32，预热后完整AdamW步：

| 批次（结构/原子/槽宽） | 中位秒 | 峰值MiB |
|---|---:|---:|
| 32/352/40 | 0.359 | 7,805 |
| 32/380/94，微批23+9 | 0.508 | 7,533 |
| 32/510/160，微批31+1 | 0.584 | 7,795 |

短测最高约7.62 GiB，不覆盖整轮缓存增长；历史A64整轮为24.56 GiB。
新增单组35轮连导出分析仍规划4.5–5.5 h；前三轮后更新ETA，不据短测承诺整轮显存或性能。

## OpenCode交接

使用当前安装项目依赖的torch环境，在仓库根执行；入口会重新校验预检、源码、环境与参考身份。
任一不匹配或发现部分训练产物即停止报告，不覆盖预检、不自动续跑或重训参考。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_norm_compare.py --tag-prefix norm20261020 \
  --preflight results/runs/local_internal_norm_20261020/preflight_final.json \
  --out results/runs/local_internal_norm_20261020 --run
```

独立会话后台运行，保存PID、起止时间及退出码；每5–10分钟检查进程、history、日志和GPU，前三轮估算ETA。
OOM、NaN/Inf、退出异常或契约不符立即停止报告，不自动重试、改配方或追加预算。
结束后核验35轮连续、latest=35及best选点，报告两任务blind R²/失败率、联合差值及95%区间、
train/valid谱差诊断和实际耗时/显存；主判据与双任务保护见设计，不自动替换I1或宣称定位到具体机制。
