# uniARPAT 待开发总清单（Backlog，主干文档）

**建档**：2026-09-09 ｜ **状态**：待开发 ｜ **前置完成**：对照Hygiene五项（`3e6df8f`）、工程E1–E4（`5b3d280`）
**关联文档**：[技术路线Roadmap.md](./技术路线Roadmap.md)、archive/Hygiene实施记录、[数据加工规范DataSpec.md](./数据加工规范DataSpec.md)
**冻结基线**：v0-legacy（旧成绩）→ h1（hygiene重跑，待执行）

---

## A. 数据v2线（主线，详见 Design-A*.md）

- [x] **A1 Delta整表下载**（eDOS 4.35GB/phDOS 0.35GB已落地）。
- [x] **A2 对齐验证**（中位0.9935；83%单自旋；3%细胞不一致）。
- [x] **A3 census收尾** → [Design-A3](2026-09-10-Design-A3census收尾.md)：有效全集154,373；声子26,609；**真双谱18,644**（7,965个有声子无eDOS文档，已逐个复核）；unresolved=0。清单`getdata/raw/dual_spectra_ids.json`（26,609声子全集，真双谱以A4抽取`edos_absent.json`为准）。
- [ ] **A3b PhononDB复算入库**（PhononDB 10,034包第三源，PBEsol；包内yaml嵌forces，phonopy mesh DOS直算，四面体+密度标定）：复算脚本已验证（pilot 50/50），全量跑中；结构/Born/介电副产品同步入库。A4输入三源至此齐：MP双谱18.6k + JARVIS pilot + PhononDB复算。
- [ ] **A4 v2加工执行** → [Design-A4](2026-09-10-Design-A4v2加工执行.md)（规范见DataSpec）。
- [ ] **A5 processed落盘**（随A4）。
- [ ] **A6 分层切分v2** → [Design-A6](2026-09-10-Design-A6分层切分.md)（8:1:1+三层分层+硬隔离+seed42）。
- [ ] **A7 映射v2** → [Design-A7](2026-09-10-Design-A7映射.md)（L1✅/L2待议/L3标记）。
- [ ] **A2b 三源gap分析**（phdb到货后）：MP↔JARVIS↔PhononDB三极分歧矩阵（eDOS两极/phDOS三极）；分歧样本→冲突审计输入；源权重设计依据。
- [ ] **A3-supplement PhononDB覆盖映射**（phdb到货后）：MDR文件名mp引用→mpid，与双谱集取交集；新增覆盖数+分布（低对称复杂氧化物验证）；未命中走L2 matcher。
- [ ] **A8 CIF特征完备性**（随A4验收）。

## B. 对照与基线线

- [x] **B0 hygiene五项 + E1–E4**（已完成，已提交）
- [x] **B1 h1重跑 M1–M4**（日志[2026-09-11-h1收官](../03_工作日志/日志-2026-09-11-h1收官.md)）：
  M1 0.519/12.3%，0.678/10.0% ｜ M2 0.526/13.3%，0.681/8.5% ｜
  M3 0.491/16.2%，0.650/11.6% ｜ M4 0.481/17.4%，0.649/12.2%（eDOS med/fail，phDOS med/fail）。
  判决：解耦≈零（砍回共享，省~35M）；对称头负（回退轻量）；门控负（判死刑转MoE）。M5待P0-0修复后单跑。
- [x] **B3 batch/lr放量臂**：M1@96（lr×3）med对齐（+0.002/−0.003）但eDOS fail 12.3%→15.7%恶化 → **不换轨，全员留32**；96作未来大模型备选。附带修：`--lr`曾被吞掉（只记不用，现已生效）；`--tag`隔离输出。教训：固定轮数下大batch不省时间（FLOP不变），只看动力学。
- [ ] **B4 超参扫描**（B3定轨后）：dropout / weight_decay（现AdamW默认0.01，未显式调过）/ warmup长度 / λ_ph / 梯度裁剪阈值——一次一个，10轮pilot初筛，胜者进100轮。预期二阶增益，不前置。
- [ ] **B2 v2基线重标定** → [Design-B2](2026-09-10-Design-B2v2基线重标定.md)。

## 0. 最高优先级（2026-09-10置顶）
- [ ] **E9 Encoder v2** → [Design-E](2026-09-10-Design-E编码器v2.md)（已决断P0冻结，轮数另定）：自研稀疏周期图+低阶等变（G1必做/G2G3选做）+坐标query（Q1Q2必做/Q3选做）+守恒前面积审计，接口`[B,L,512]+[B,512]`不变。开工门槛A6+B1+B2；与C1/C2见Design-E§5分工。

## C. 模型线（B1/B2开门后）

- [ ] **C1 Phase 1四件套**：24维特征MLP投影、能量坐标编码+RFF（仅eDOS）、TV/梯度loss+物理加权、旋转+声子位移增强。逐个消融验收。
- [ ] **C2 Phase 1.5**：SumNorm-KL双轨+Huber、守恒缩放替代M5 scale（P0-0修复）、截断mask+距离衰减。
- [ ] **C2b 网格对照试验**（2026-09-09重排：不依赖遗产，按物理+SOTA设臂，分两阶段）：
  - **Stage 1（eDOS四臂，phDOS固定为宽非均匀P1）**：
    - E0 anchor [-6,6]/128等距（仅用于v1归因，非设计中心）；
    - E1 窄密 [-4,4]/160等距（XBERT ±4eV先例，0.05eV/bin）；
    - E2 非均匀160（Fermi ±2eV内96 bins≈0.042eV，尾部64 bins）；
    - E3 宽密 [-6,6]/256等距（容量上限探测）。
  - **Stage 2（phDOS三臂，eDOS用Stage 1胜者）**：
    - P0 anchor [-280,980]/64；
    - P1 宽非均匀 [-300,4000]/112（0~500按10cm⁻¹≈50，500~4000按~60cm⁻¹≈58，负频~10）；
    - P2 宽等距 [-300,4000]/128（测非均匀是否必要）。
  - 依据：线宽采样律 + XBERT窗口 + DOSTransformer能量模态（Fermi精细）+ PET-MAD-DOS宽谱+CNN读出。
  - 指标：median R² + fail率 + 峰区MAE；判决：Stage胜者永久采用，两阶段总计7臂≈25h。
- [ ] **C3 Phase 3**：PhysMoE晶体级路由 vs 能量级路由 A/B；[CLS] Cell Token；PotNet长程后置。
- [ ] **C5 解码器MoE** → [Design-C5解码器MoE.md]（待讨论：token级4专家Top-2+负载均衡 vs 晶体级PhysMoE；位置/宽度/层数）。
- [ ] **P0 预训练线** → [Design-P0预训练.md]（2026-09-10已决断见Design-E§5：仅保留(a)/(b)，(c)删除；放E9-P0之后）。
- [ ] **C4 AMP加速臂**（默认关`--amp`，h1后验证）：预期1.4–1.8x；NaN冒烟先行；成绩单独立记录。

## D. 数据扩量线（phDOS天花板对策）

- [ ] **D1 uMLIP伪标签**：暂停（2026-09-10撞无外部权重决断，见Design-E§5；原MACE/Orb力→声子谱方案归档备查）。
- [ ] **D2 Raman Phase-R**：CRD已到货（`getdata/raw/crd/*.zip`，5101条，mpid直连；频率轴二选一待定），第三解码器预留。
- [ ] **D3 eDOS-only辅助臂**（M-aux，等计算量对照，Phase 1.5后）：掩码多任务，只教encoder。
- [ ] **D4 IR/介电**：暂缓，已记录。
- [ ] **D5 PhononDB混训权重**（复算入库后）：PBEsol第三源分源监控/配比，ablation备而不用。

## E. 工程线（h1跑完后）

- [ ] **E5 结构重构批**：12元组→dict；ConfigBuilder拆分；metrics气象遗留清理；评估入口收敛；model.py拆分；constants.py；注释规范。以h1成绩为回归网（R²对齐到1e-3）。
- [ ] **E6 数据加载升级**：长度分桶batching（数学等价，省30–50%）；v2 dataset（dict batch）。
- [ ] **E7 lint/CI**：ruff + 单测CI。
- [ ] **E8 入口去重**：`test_cif.py` vs `cif2dos.py` 留一；`test.py` stale修复或删除。

## F. Backlog

- [x] **F1 PhononDB Kyoto重试**（已转为A3b执行；原站注销，MDR包已全下）。
- [ ] **F2 MP Delta访问方式归档**（已打通，写进v2手册：REST+Delta直读+重试策略）。
- [ ] **F3 He/稀有元素样本策略**（已定 2026-09-09，v2双谱口径修正）：v1不动（含mp-1019742）；
  v2 census实测MP全库He仅6材料且0个有phDOS → **v2双谱核心集不含He**（口径一致）；
  6个He-eDOS留作未来M-aux臂（eDOS-only辅助池）专用，不进双谱训练/评估；phDOS侧He主表剔除+单列探针。

## 已冻结决策（不再讨论）

1. MinMax换归一化优先于扩embedding；2. 双谱先行、Raman/IR延后；3. v1冻结归档、v2重标定；4. 混训记账制（源标签+分源统计+冲突审计）；5. 选型/报告用median+fail；6. 训练环境冻结，抓数走独立venv；7. 存储双轨（Parquet canonical + npy缓存）；8. phDOS拼存量到顶，后续靠洗+借。

## 对照实验总原则（2026-09-09冻结）

- **三层对照**：(a)超参（lr/batch/dropout/λ权重）→ (b)设计配置（网格/归一化/head/loss形式/切分策略/源配比）→ (c)数据配置（源混合、增强开关、winsorize阈值）。三层一视同仁：**凡是有≥2个合理选项、且会影响结果的，都配消融臂**。
- **防组合爆炸**：10-epoch pilot初筛（便宜），胜者进100轮；一次只动一个因子；每臂预写判决规则（赢了如何、输了如何），不做开放式实验。
- **等算力对照**：凡涉及"更多数据/更多步数"的臂，总步数拉平后再比，杜绝把算力涨点当方法涨点。
