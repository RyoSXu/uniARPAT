# 日志：2026-10-21 — 完整 A 外层归一化对照准备

## 范围

按[设计](../design/design-2026-10-21-local-outer-norm.md)，只把两次局部调用的外层512维归一化由pre移到post；
内部四个64维块继续post-norm。此项与已完成的LN1内部归一化实验分开，A64参考继续复用。

调用链：训练/配置派发 → `local_outer_norm_experiment`验证、共同初值和独立契约 → `LocalGlobalEncoder`
的LN2路由 → `LocalResidualLayer.outer_norm_first=False`。训练日志记录实际内部及外层归一化位置。
`local_outer_norm_compare.py`/`check_local_outer_norm.py`复用原归一化队列与GPU短检，以`OUTER_NORM`显式选择协议。

## 证据

预检：`results/runs/local_outer_norm_20261021/preflight_final.json`，`passed=true`，耗时80.81秒。
SHA256：`bf13e3145189e6f1ba19406be1d787f6e140efe48792b8a56893557121fdf509`。
正式候选标签为`_lon_outer20261021_post_k2_g6_staged_s42_e35`；本轮未启动正式训练。

- 认证69份当前源码、63份历史源码及A64配置/数据/history/best/latest/冻结导出，历史参考固定到`1c29ba0d50ec2479f837624b6836ad98e2f453f0`。
- 参数76,077,252个、287个张量；全部未训练初值SHA256与A64相同：
  `2a2f9a507cf9a6955fe6c299be995b9fec78e5fe3b5d254148eb27aa680fe3fa`。
  当前默认A64相对历史完整模型的输出及287组参数梯度在FP32/FP64下逐位相同；32个真实valid材料回放最大相对误差`3.95e−6`。
- 起始RNG和首批材料顺序匹配；两个真实训练批的全部参数梯度有限，验证输出有限；权重/优化器/RNG恢复与异契约拒绝通过。
- 完整输出及物理谱的E(3)/原子重排：FP32最大相对误差`1.45e−6`，FP64为`3.96e−15`，均低于预设阈值。
- 11项新测试、12项原归一化测试、76项相关回归，共99项通过；254份Python源码Ruff/内存编译、文档引用及仓库载荷检查通过。

V100 FP32实测训练批成本，包含原有自动分批与几何缓存：

| 批次 | 结构/原子/padding宽度 | 微批 | 中位秒/批 | 峰值分配GiB |
|---|---|---|---:|---:|
| 普通1 | 32/352/40 | 32 | 0.360 | 7.62 |
| 普通2 | 32/380/94 | 23+9 | 0.507 | 7.35 |
| 最大结构 | 32/510/160 | 31+1 | 0.584 | 7.61 |

与LN1短检成本接近，未见明显额外计算成本。短检缓存未填满，不把7.62GiB当作正式峰值；
A64/LN1完整运行峰值约24.56GiB。正式35轮加诊断仍预计4.5–5.5小时，需以前三轮实际进度校正。

## 结论

状态：**准备关卡通过，训练待执行**。实现与短检支持开始本项对照，尚无LN2的35轮精度结果，不能宣称外层post-norm更好或定位到退化机制。

## OpenCode交接

先核对设计和下述预检，只允许新增LN2一组35轮、seed42。A64不重训；不用LN1或A64的训练权重初始化。
入口会复核源码、环境、数据、参考身份、共同初值和配方；部分训练产物或任何不匹配均停止，不覆盖、不自动恢复。

```bash
PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B tools/eval/local_outer_norm_compare.py --tag-prefix outer20261021 \
  --preflight results/runs/local_outer_norm_20261021/preflight_final.json \
  --out results/runs/local_outer_norm_20261021 --run
```

独立后台会话运行，保存PID、起止时间与退出码；每5–10分钟检查进程、history、日志及GPU，前三轮估算ETA。
OOM、NaN/Inf、异常退出或契约不符即停止报告，不自动重试、改配置或追加预算，不读test。
结束后核验35轮连续、latest=35、best严格按联合blind最大值；报告两任务blind R²/失败率、联合差值及95%区间、
train/valid同组成谱差诊断和实际时间/显存。区间只反映材料重采样；不自动替换I1、不追加新候选、不执行Git提交。
将证据与结论写入独立运行日志并简要更新`docs/status.md`。
