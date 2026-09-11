# 工作日志 2026-09-11：PhononDB全量复算到货验收

## 到货
- 10,034/10,034 unique，零错误，184MB（`phonondb_recomputed.jsonl`，6分片已合并）。
- 抽检：maxfreq中位22.7THz、p99 119.5THz（与Delta实测4721cm⁻¹上限一致）；零全零谱。
- 单位：THz（Si标尺逻辑复核通过）；加工时统一转cm⁻¹。

## 过程记录
- mesh传参bug（conf文件写法）修后pilot 50/50；6分片并行（OMP=1），实测~1900条/h（单进程610条/h的3倍，未达6倍：NFS虚高负载+worker争抢）。
- 教训：批量接口先做批量大小探测（census 200上限事件同类）。

## 下一步
A3c覆盖映射（MDR文件名mp引用→mpid）→ A2b三极gap → A4加工。
