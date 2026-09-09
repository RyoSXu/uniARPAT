# uniARPAT 待开发总清单（Backlog，主干文档）

**建档**：2026-09-09 ｜ **状态**：待开发 ｜ **前置完成**：对照Hygiene五项（`3e6df8f`）、工程E1–E4（`5b3d280`）
**关联文档**：双报告合并评估（统一路线）、Hygiene实施记录、v2加工规范（待建）
**冻结基线**：v0-legacy（旧成绩）→ h1（hygiene重跑，待执行）

---

## A. 数据v2线（主线）

- [ ] **A1 Delta整表下载**（进行中，后台）：eDOS 4.35GB + phDOS 0.35GB，文件级断点续传，manifest校验。验收：本地DeltaTable可查。
- [ ] **A2 JARVIS pilot对齐验证**：网格/结构/标尺/离群四项。门：中位相关>0.9、结构全同>95%。
- [ ] **A3 census精确计数**：双谱构成28–32k锁定；原子数分布定cap（候选96/128）；晶系/金属性/元素覆盖四张表。
- [ ] **A4 v2加工规范定稿**：CIF六项清单（结构JSON/canonical CIF/对称性/占位/谱原始/provenance）、grids.yaml版本化、winsorize规则、round-trip校验1000条。
- [ ] **A5 processed落盘**：Parquet全集（含split列）+ `build_v2_cache.py` → npy/memmap训练缓存；删哨兵126/127；float32；mask唯一化；24维原子特征预计算；晶格/坐标分离。
- [ ] **A6 分层切分v2**：成分去重隔离 + 难度/晶系分层 + seed固定 + 旧索引保留。
- [ ] **A7 映射v2**：L1 reference直连 + L2同成分StructureMatcher（记分）+ L3诚实标记；新旧一致性校验。
- [ ] **A8 CIF特征完备性**：占位审计→规则；对称性注入；CIF round-trip diff=0。

## B. 对照与基线线

- [x] **B0 hygiene五项 + E1–E4**（已完成，已提交）
- [ ] **B1 h1重跑 M1–M5**（100轮，输出`h1`后缀）：新对照基线。门：111s/轮量级、收敛形态正常。
- [ ] **B2 v2基线重标定**：v2上M1重跑 + 数据量消融（5k/10k/20k/30k）→ 学习曲线决定是否加码数据。

## C. 模型线（B1/B2开门后）

- [ ] **C1 Phase 1四件套**：24维特征MLP投影、能量坐标编码+RFF（仅eDOS）、TV/梯度loss+物理加权、旋转+声子位移增强。逐个消融验收。
- [ ] **C2 Phase 1.5**：SumNorm-KL双轨+Huber、守恒缩放替代M5 scale（P0-0修复）、截断mask+距离衰减。
- [ ] **C2b 网格对照试验**（eDOS/phDOS分辨率与分配）：
  - G0 基准：eDOS[-6,6]/128等距 + phDOS[-280,980]/64（v2默认，与v1可比）；
  - G1 加密：eDOS 256等距（head输出通道128→256，参数+~0.4M，记录在案）；
  - G2 非均匀：总数仍128，Fermi ±1.5eV内占64 bins、尾部分64 bins（参数量与G0完全一致，最公平）；
  - G3 phDOS非均匀：低频声学支加密（总数仍64）。
  - 控制：同数据、同切分、同seed、varied仅网格+head末层；指标 median R² + fail率 + 峰区MAE（Fermi±1eV / 声学支<200cm⁻¹）。
  - 决策规则：G2/G3若赢→分辨率分配是瓶颈，永久采用；G1赢但G2不赢→纯容量问题，加head；都不赢→网格非瓶颈，结案。
- [ ] **C3 Phase 3**：PhysMoE晶体级路由 vs 能量级路由 A/B；[CLS] Cell Token；PotNet长程后置。
- [ ] **C4 AMP加速臂**（默认关`--amp`，h1后验证）：预期1.4–1.8x；NaN冒烟先行；成绩单独立记录。

## D. 数据扩量线（phDOS天花板对策）

- [ ] **D1 uMLIP伪标签**：MACE/Orb力→声子谱，无限近似phDOS预训练，30k DFT微调（Phase 2首位）。
- [ ] **D2 Raman Phase-R**：CRD 5k接入（Materials Cloud `ze-58`），第三解码器预留。
- [ ] **D3 eDOS-only辅助臂**（M-aux，等计算量对照，Phase 1.5后）：掩码多任务，只教encoder。
- [ ] **D4 IR/介电**：暂缓，已记录。

## E. 工程线（h1跑完后）

- [ ] **E5 结构重构批**：12元组→dict；ConfigBuilder拆分；metrics气象遗留清理；评估入口收敛；model.py拆分；constants.py；注释规范。以h1成绩为回归网（R²对齐到1e-3）。
- [ ] **E6 数据加载升级**：长度分桶batching（数学等价，省30–50%）；v2 dataset（dict batch）。
- [ ] **E7 lint/CI**：ruff + 单测CI。
- [ ] **E8 入口去重**：`test_cif.py` vs `cif2dos.py` 留一；`test.py` stale修复或删除。

## F. Backlog

- [ ] **F1 PhononDB Kyoto重试**（本站503，延后）。
- [ ] **F2 MP Delta访问方式归档**（已打通，写进v2手册：REST+Delta直读+重试策略）。
- [ ] **F3 He/稀有元素样本策略**（补样本 or 单列统计，待定）。

## 已冻结决策（不再讨论）

1. MinMax换归一化优先于扩embedding；2. 双谱先行、Raman/IR延后；3. v1冻结归档、v2重标定；4. 混训记账制（源标签+分源统计+冲突审计）；5. 选型/报告用median+fail；6. 训练环境冻结，抓数走独立venv；7. 存储双轨（Parquet canonical + npy缓存）；8. phDOS拼存量到顶，后续靠洗+借。

## 对照实验总原则（2026-09-09冻结）

- **三层对照**：(a)超参（lr/batch/dropout/λ权重）→ (b)设计配置（网格/归一化/head/loss形式/切分策略/源配比）→ (c)数据配置（源混合、增强开关、winsorize阈值）。三层一视同仁：**凡是有≥2个合理选项、且会影响结果的，都配消融臂**。
- **防组合爆炸**：10-epoch pilot初筛（便宜），胜者进100轮；一次只动一个因子；每臂预写判决规则（赢了如何、输了如何），不做开放式实验。
- **等算力对照**：凡涉及"更多数据/更多步数"的臂，总步数拉平后再比，杜绝把算力涨点当方法涨点。
