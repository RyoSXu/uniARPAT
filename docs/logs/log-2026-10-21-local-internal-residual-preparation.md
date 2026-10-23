# 日志：2026-10-21 — 完整 A 内部残差对照准备

## 范围

按[设计](../design/design-2026-10-21-local-internal-residual.md)，只把四个内部块的`LN64(s+F(s,R))`
改为`LN64(F(s,R))`，保留完整消息、内部post-norm、外层pre-norm、完整状态读出和外层残差。
`s`仍进入消息与更新MLP。这项与LR1增量读出分开，不组合未达采用条件的候选。

调用链：训练/配置派发 → `local_residual_experiment`共同初值/模型验证/独立契约 →
`LocalGlobalEncoder`的LR2路由 → `PeriodicLocalMessage.internal_residual=False` →
四个`InvariantMessageBlock.residual_add=False`。训练日志记录实际加法开关及归一化/读出。
`local_residual_compare.py`、`check_local_residual.py`以`INTERNAL_RESIDUAL`复用现有受控对照流程。

## 验证证据

预检：`results/runs/local_internal_residual_20261021/preflight_final.json`，`passed=true`，耗时80.49秒。
SHA256：`2a187dd0abee0b731063ba311018bea0142d435884d079ecd7cd5ab931495811`。
候选标签`_lres_resid20261021_noadd_k2_g6_staged_s42_e35`；正式训练尚未启动。

- 认证69份当前源码、63份历史源码及A64配置/数据/history/best/latest/冻结导出。
  默认A64的完整模型固定夹具输出和287组参数梯度在FP32/FP64下与历史代码逐位一致；32个真实valid材料回放最大相对误差`1.98e−6`。
- 参数76,077,252个、287个张量，全部未训练初值与A64相同；状态SHA256为
  `2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。
  起始RNG和首批材料顺序一致；初始化未读取历史训练checkpoint。
- 两个真实训练批的全部参数梯度有限、valid输出有限；权重/优化器/RNG恢复及异契约拒绝通过。
  完整输出和物理谱的E(3)/原子重排最大相对误差为FP32 `1.22e−6`、FP64 `3.51e−15`。
- 11项新测试、34项既有候选测试、76项相关回归，共121项通过；262份Python源码Ruff/内存编译、文档引用和仓库载荷检查通过。
- 本轮69份源码另存`source_snapshot/`，与预检哈希一致；此前LR1的69份快照在修改前已核对，旧预检与训练产物保持原样。

V100 FP32实测训练批成本：

| 批次 | 结构/原子/padding宽度 | 微批 | 中位秒/批 | 峰值分配GiB |
|---|---|---|---:|---:|
| 普通1 | 32/352/40 | 32 | 0.359 | 7.62 |
| 普通2 | 32/380/94 | 23+9 | 0.507 | 7.36 |
| 最大结构 | 32/510/160 | 31+1 | 0.586 | 7.62 |

成本与此前短检接近。短检缓存尚未填满，正式显存参考约24.56GiB分配值，仍须监测实际占用。
预计一组35轮加完整诊断4.5–5.5小时，前三轮校正ETA；本次检查尚无LR2正式精度结论。

## OpenCode交接

只新增LR2一组35轮、seed42，复用A64参考，不重训、不用任何历史训练checkpoint初始化。
先核对设计和本机预检。入口检查源码、环境、数据、参考、共同初值和配方；不匹配或已有部分产物即停，不自动续跑。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_residual_compare.py --tag-prefix resid20261021 \
  --preflight results/runs/local_internal_residual_20261021/preflight_final.json \
  --out results/runs/local_internal_residual_20261021 --run
```

后台运行，保存PID、起止时间与退出码；每5–10分钟检查日志、history、进程和GPU，前三轮估算ETA。
OOM、NaN/Inf、异常退出或契约不符即停止报告，不自动重试、改配置或追加预算；不读test。
结束后核验35轮连续、latest=35、best按联合blind最大值保存；报告两任务blind R²/失败率、
LR2−A64联合差值与95%区间、完整train/valid组成谱差诊断、耗时与显存。
按设计判断数值采用条件，不自动替换I1或启动后续实验。写独立运行日志，简要更新`docs/status.md`，不执行Git提交。
