# 日志：2026-10-21 — 完整 A 增量读出准备

本页保留执行前的准备与交接；正式35轮及独立核验已完成，见[运行结果](log-2026-10-21-local-increment-run.md)。

## 范围

按[设计](../design/design-2026-10-21-local-increment-readout.md)，在A64共同条件下仅将`P_out(s2)`改为
`P_out(s2-s0)`，保留偏置、不detach，保留完整消息、内外加法残差及原归一化位置。
这属于独立接口候选，当前实现不是错误；不采用LN1/LN2，也不把增量称作纯几何信息。

调用链：`tools/train.py`与配置派发 → `local_increment_experiment`共同初值/验证/独立契约 →
`LocalGlobalEncoder`的LR1路由 → `PeriodicLocalMessage.readout_delta=True`。
训练实际记录四个内部post-norm块、两个外层pre-norm及两次增量读出。
`local_increment_compare.py`、`check_local_increment.py`以`INCREMENT_READOUT`复用既有局部对照流程。

## 验证证据

预检：`results/runs/local_increment_20261021/preflight_final.json`，`passed=true`，耗时80.73秒。
SHA256：`e5522ef07b27034895c4c2fd90ed9504074d5f0793d54a6c1b12d7a8f045c65a`。
候选标签`_lri_res20261021_delta_k2_g6_staged_s42_e35`；正式训练尚未启动。

- 认证69份当前源码、63份历史源码，以及A64配置/数据/history/best/latest/冻结导出；恢复参考行为的真实valid回放覆盖32个材料，最大相对误差`1.87e−6`。
- 参数76,077,252个、287个张量，全部未训练初值与A64相同，状态SHA256为
  `2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。
  默认A64对历史完整模型的输出及全部参数梯度在FP32/FP64下逐位相同；起始RNG与材料批序匹配。
- 两个真实训练批的全部参数梯度有限、valid输出有限；权重/优化器/RNG恢复和异契约拒绝通过。
  完整输出与物理谱的E(3)/原子重排最大相对误差为FP32 `1.05e−6`、FP64 `3.46e−15`。
- 11项新测试、23项原归一化测试、76项相关回归，共110项通过；258份Python源码Ruff/内存编译、文档引用和仓库载荷检查通过。
- LR1与前一轮LN2各自绑定的69份源码另存对应运行目录的`source_snapshot/`，分别与原预检哈希完全匹配。
  这保留未提交阶段的实验版本；正式入口仍要求当前源码匹配，不放宽保护或改写旧预检。

V100 FP32实测训练批成本：

| 批次 | 结构/原子/padding宽度 | 微批 | 中位秒/批 | 峰值分配GiB |
|---|---|---|---:|---:|
| 普通1 | 32/352/40 | 32 | 0.359 | 7.62 |
| 普通2 | 32/380/94 | 23+9 | 0.505 | 7.36 |
| 最大结构 | 32/510/160 | 31+1 | 0.583 | 7.62 |

与A64/LN2短检接近，未见额外成本。短检缓存尚未填满；正式峰值参考约24.56GiB，须监测实际运行。
预计一组35轮加完整诊断4.5–5.5小时，前三轮校正ETA。短检只支持开始对照，尚无LR1正式精度结果。

## OpenCode交接

只新增LR1一组35轮、seed42；保留预检及所有既有结果，不重训A64，不用任何历史训练checkpoint初始化。
先核对设计及本机预检，入口会复核源码、环境、数据、参考、共同初值和配方；不匹配或部分产物即停止，不自动续跑。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_increment_compare.py --tag-prefix res20261021 \
  --preflight results/runs/local_increment_20261021/preflight_final.json \
  --out results/runs/local_increment_20261021 --run
```

后台运行，保存PID、起止时间和退出码；每5–10分钟检查日志、history、进程和GPU，前三轮估算ETA。
OOM、NaN/Inf、异常退出或契约不符即停止报告，不自动重试、改配置或追加预算；不读test。
完成后核验35轮连续、latest=35、best按联合blind最大值保存；报告两任务blind R²/失败率、
LR1−A64联合差值与95%区间、完整train/valid组成谱差诊断、耗时与显存。
按设计判断数值采用条件，不自动替换I1。写独立运行日志、简要更新`docs/status.md`，不执行Git提交。
