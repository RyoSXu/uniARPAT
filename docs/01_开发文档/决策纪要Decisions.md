# uniARPAT 关键决策与结论纪要（聊天共识归档）

**建档**：2026-09-09 ｜ **性质**：只记录已定结论，不记录过程 ｜ **关联**：Backlog总清单（执行）、合并评估（依据）、加工规范（数据）、Hygiene记录（工程）

---

## 一、已冻结决策（违反需重新上会）

**数据与评估**
1. MinMax换归一化优先于扩embedding（目标错了表征越强越跑偏）。
2. 双谱先行；Raman Phase-R；IR/介电暂缓。
3. v1冻结归档（v0-legacy）；v2重标定后再谈涨点。
4. 混训记账制：源标签+分源统计+冲突审计；禁止`concat.py`式裸拼接。
5. 选型/报告一律median R² + fail率；mean R²停用；oracle/blind双轨报告。
6. 缺失记mask永不填0；禁外推；离群按归因表判决（winsorize/降权/剔除）。
7. 网格：生产冻结 E0+P0（eDOS [-6,6]/128 + phDOS [-280,980]/64，C2b 2026-09-14收官、09-15改判）；
   P2 仅实验记录；P1 死因=卷积头假设等距网格，非均匀网格必须等 C5 新读出。

**实验纪律**
8. 三层对照（超参/设计配置/数据配置）一视同仁；pilot初筛→胜者进100轮；单因子；预写判决；等算力。
9. h1重跑完成前不动架构、不动性能代码（AMP/分桶/重构全排队）。

**工程**
10. 训练环境冻结；抓数走独立venv（本次破例已记录在案，下不为例）。
11. 存储双轨：Parquet canonical + npy训练缓存；raw JSONL不可变；manifest sha。
12. Checkpoint统一全量dict；ablation落盘有效config；环境lockfile。

**数据口径（2026-09-16增补）**
13. Q1截断隔离：trunc=C_win/C_full<0.5（+14个不可读且γ_label<0.1）计1682出池，train/valid/test同步；
    旧缓存归档、新缓存同名接管、旧基线标注pre-Q；此后成绩一律Q1口径。离群归因表（#6）新增“截断（NBANDS不足）→剔除”条目。

**盲推与评估（2026-09-15增补）**
14. 盲推正统=H1 η/γ有界头（`Scale=3N·η̂/Δ`、`N_val·γ̂/Δ`）；0.92全局常数退役、maxfreq硬guard删除；
    掩膜永留评估端禁入loss；选型用 median+fail，mean停用；盲推必报gap分布p50/p90/p99。

## 二、实测数字（证据，不随意见改变；现行=B7 Q1，其余存档勿引）

- 现行（B7，Q1 2287样本，2026-09-17）：eDOS med 0.518/fail 5.73%，phDOS med 0.741/fail 3.50%，Cv MAE 0.30。
- 存档（v0-legacy，1371样本，2026-09-08，oracle）：M3 eDOS med 0.493 / phDOS med 0.682；M5 blind崩（eDOS med 0.279 / mean -0.130，fail 33.5%）。
- 数据画像：eDOS max 10618（单bin伪影候选）、80样本max>200；phDOS 63%零值；He训练零样本。
- 成分重叠13.6%；切分mp-id零泄漏；角度/1c/哨兵链条一致。
- 死参6.56M（8.6%）；RP上提6次→1次（117s→111s/轮）；数据加载0ms；前向113ms（encoder占61%）。
- MP Delta：eDOS 691,978行（GGA 498k/GGA+U 178k/HSE 10.8k）、phDOS 27,914行；本地4.4GB。
- A2对齐：MP谱↔v1 bins中位相关 **0.9935**；83%单自旋通道；3%细胞选择不一致（原胞/常规胞）。
- JARVIS pilot：77条（eDOS全中、phDOS 62）；占位审计1450格点0部分占位。
- phDOS全球公开天花板约33–35k unique；eDOS MP池27万文档。

## 三、数据源 verdict

- MP主源（结构+eDOS+phDOS同源，REST+Delta直读已打通，S3无签名公开读）。
- JARVIS交叉+备份（eDOS bulk 55k本地；网页双自旋原始）。
- PhononDB已入库（A3b收官：10,034/10,034零错误；Kyoto原站注销，MDR包已全下）。
- CRD Phase-R已到货（`getdata/raw/crd/*.zip`，5101条，mpid直连；频率轴二选一待定）。
- MP REST谱数组为null是官方迁移（v2026.04 Delta化，有公告），非我方查错；`mp-api`全谱经Delta表，需pyarrow/deltalake（已装好）。

## 四、已完成归档（原待办，2026-09-18确认）
- h1重跑、census、全量清单确认、v2加工规范执行均已完成，见工作日志。

## 五、网络通路记录（网络组 2026-09-09）

- `archive.materialscloud.org` proxy直连旁路已生效（毫秒级），**不要撤**（Phase-R依赖）。
- Kyoto站证实死亡，非代理问题；MDR通路明确，维持backlog。
