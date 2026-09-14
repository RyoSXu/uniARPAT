# uniARPAT 待开发总清单（Backlog，主干文档）

**建档**：2026-09-09 ｜ **状态**：待开发 ｜ **前置完成**：对照Hygiene五项（`3e6df8f`）、工程E1–E4（`5b3d280`）
**关联文档**：[技术路线Roadmap.md](./技术路线Roadmap.md)、archive/Hygiene实施记录、[数据加工规范DataSpec.md](./数据加工规范DataSpec.md)
**冻结基线**：v0-legacy（旧成绩）→ h1（hygiene重跑，待执行）

---

## A. 数据v2线（按执行顺序排列）

- [x] **A1 Delta整表下载**（eDOS 4.35GB/phDOS 0.35GB已落地）。
- [x] **A2 对齐验证**（MP↔v1中位0.9935；83%单自旋；3%细胞不一致；MP↔JARVIS gap：eDOS 0.54/phDOS 0.85）。
- [x] **A3 census收尾** → [Design-A3](2026-09-10-Design-A3census收尾.md)：有效全集154,373；声子26,609；**真双谱18,644**（7,965有声子无eDOS，已逐个复核，见`edos_absent.json`）；unresolved=0。
- [x] **A3b PhononDB复算入库**（10,034/10,034零错误，184MB；maxfreq中位22.7THz；日志见03工作日志）。
- [x] **A3c PhononDB覆盖映射**（2026-09-11收官，日志见03工作日志）：
  10,034 serial与复算1:1；9,938译出唯一canonical（96死ID挂unresolved，`mp-867xxx`系为主）；
  有效全集内9,936；与MP声子集交5,545/真双谱交4,660；**增量核心4,391**（有eDOS但MP无声子，真双谱18,644→~23,035）；
  分布验证通过（元素80vs80/氧化物差0.091/maxfreq中位+19.8%贴线待A2b复核/晶系全覆盖略偏低对称）。
- [x] **A2b 三源gap分析**（2026-09-12收官，1,960声子对/3,352电子对）：
  声子MP-PhononDB r=0.893互换+轻校准；MP-JARVIS r=0.658源标签加权
  （JARVIS强度任意单位，A4前按3N求和规则重归一）；eDOS配准后r=0.79，
  +1eV为能量零点系统差（已验证非伪影）；结构MPvsPhononDB dV 2.7%/空间群98.6%一致 →
  增量双谱暂定MP结构输入+源标签。权重与选型见日志。
- [x] **Top-up双缺口**（2026-09-12）：MP增量结构全回+eDOS归属（REST 67%+Delta结构归属，
  241真无出池）；JVASP结构经figshare整包解决；结构选型**转正**（JVASPvsMP dV 1.7%/空间群98.0%），
  A4统一MP结构+源标签。**谱落实双谱池26,002（+39.5%）**。
- [x] **A4 v2加工执行**（2026-09-12收官，`v2_processed.parquet` 24,988行）：
  网格eDOS[-6,6]/128+phDOS[-280,980]/64盒平均；THz密度除33.356（Si对上v1）；
  JARVIS按3N重归一；原胞symprec0.1+82格式；标签MP>pheasy/dfpt>PhononDB(w1)>JARVIS(w0.5)；
  winsorize clip1,229/宽峰保留511；CIF round-trip 1000/1000。
- [x] **A5 processed落盘**（`data/train4ARPAT-v2/` 20,040/2,477/2,471，--check+forward冒烟通过）。
- [x] **A6 分层切分v2**（8:1:1 seed42，三层±2%，成分零跨集断言，探针1条，He跳过）。
- [ ] **A7 映射v2** → [Design-A7](2026-09-10-Design-A7映射.md)（L1✅49,942零冲突；L2待议；L3标记）。
  - 2026-09-12决议：死ID（PhononDB 96 / JARVIS死leg 4,686+昵称37，MP已删库）**永久出池**，
    L2不再覆盖（raw保留备查，不删）；集外6条（2+4）同样出池。
- [x] **A7b JARVIS覆盖映射**（2026-09-11收官，ID级，谱未入库）：
  45,154/49,942 JVASP已映射（43,872唯一canonical；死leg 4,686+昵称37挂unresolved）；
  有效内43,868；MP声子交15,815/真双谱交13,142；**缺口天花板28,053**（ID级，需谱清单确认）；
  谱拉取按缺口/交集分区抓（见日志），不全抓。
- [x] **A7c JARVIS缺口谱拉取**（2026-09-12收官）：
  三源子集3,601全回（eDOS 99.9%/phDOS 58.4%，A2b输入）；
  缺口28,852全回，phDOS命中3,768→3,600唯一canonical（13.1%）。
- [x] **Top-up#1 MP增量结构+eDOS归属**（2026-09-12）：REST半小时7,724全回；
  REST dos指针只覆盖67%（es与dos-route口径不一致，dos-route 271k为准），
  余量本地Delta结构归属（归属索引`delta_edos_index.parquet`，nsites+物种+dV三级+matcher终判）；
  MP-eDOS落实：PhononDB增量4,049/4,391、JARVIS增量3,572/3,600；
  **并池（谱落实口径）：新增7,358，真双谱 18,644 → 26,002（+39.5%）**；
  370无MP-eDOS（241 Delta真无+129待定）暂出双谱池（28个可用JVASP-eDOS回补，待定）。
- [x] **A8 CIF特征完备性**（2026-09-12闭环）：24,988逐样本canonical CIF（原胞symprec=0.1）
  全生成零失败（`v2_cifs.zip` 20.4M），抽检可解析。
- [x] **A9 数据报告v2**（2026-09-13）：descriptor 体例（Background/来源/方法/文件清单/验证/使用/代码/引用），
  与数据同目录：`getdata/v2_release/README.md`（发布包 ~90MB：主表去split列 + CIF + 映射 + 统计；
  网格与切分不属数据层）。
  （2026-09-13：A10 开源发布项删除，发布包已随数据留存，上传另议。）

## B. 对照与基线线

- [x] **B0 hygiene五项 + E1–E4**（已完成，已提交）
- [x] **B1 h1重跑 M1–M4**（日志[2026-09-11-h1收官](../03_工作日志/日志-2026-09-11-h1收官.md)）：
  M1 0.519/12.3%，0.678/10.0% ｜ M2 0.526/13.3%，0.681/8.5% ｜
  M3 0.491/16.2%，0.650/11.6% ｜ M4 0.481/17.4%，0.649/12.2%（eDOS med/fail，phDOS med/fail）。
  判决：解耦≈零（砍回共享，省~35M）；对称头负（回退轻量）；门控负（判死刑转MoE）。M5待P0-0修复后单跑。
- [x] **B3 batch/lr放量臂**：M1@96（lr×3）med对齐（+0.002/−0.003）但eDOS fail 12.3%→15.7%恶化 → **不换轨，全员留32**；96作未来大模型备选。附带修：`--lr`曾被吞掉（只记不用，现已生效）；`--tag`隔离输出。教训：固定轮数下大batch不省时间（FLOP不变），只看动力学。
- [ ] **B4 超参扫描**（B3定轨后）：dropout / weight_decay（现AdamW默认0.01，未显式调过）/ warmup长度 / λ_ph / 梯度裁剪阈值——一次一个，10轮pilot初筛，胜者进100轮。预期二阶增益，不前置。
- [x] **B2 v2基线重标定** → [Design-B2](2026-09-10-Design-B2v2基线重标定.md)（2026-09-14由C2b终考兼任，不另跑：
  M1@E0P2-100：eDOS med 0.438/fail 17.1%，phDOS med 0.833/fail 2.5%。后续涨点位归C1/E9）。

## 0. 最高优先级（2026-09-10置顶）
- [ ] **E9 Encoder v2** → [Design-E](2026-09-10-Design-E编码器v2.md)（已决断P0冻结，轮数另定）：自研稀疏周期图+低阶等变（G1必做/G2G3选做）+坐标query（Q1Q2必做/Q3选做）+守恒前面积审计，接口`[B,L,512]+[B,512]`不变。开工门槛A6+B1+B2；与C1/C2见Design-E§5分工。

## C. 模型线（B1/B2开门后）

- [ ] **C1 Phase 1四件套**：24维特征MLP投影、能量坐标编码+RFF（仅eDOS）、TV/梯度loss+物理加权、旋转+声子位移增强。逐个消融验收。
- [ ] **C2 Phase 1.5**：SumNorm-KL双轨+Huber、守恒缩放替代M5 scale（P0-0修复）、截断mask+距离衰减。
  - [x] C2.1 SumNorm-KL/W+Huber（2026-09-14合并，默认；见日志）。
  - [ ] C2.1b attribution臂（排队）：SumNorm+SmoothL1，拆归一与loss的各自贡献。
  - [ ] C2.2 守恒缩放（2026-09-14机制验证通过，声子部分合并）：
    blind-3N×0.92达oracle水平（0.839 vs 0.833）；公式`Y=d·3N·0.92/Δ`冻结；
    M5 ScaleHead可删（E阶段）；eDOS价电子标定需全谱窗，另起臂。
    推理接线（cif2dos盲模式）归E阶段。
- [x] **C2b 网格对照试验**（2026-09-14收官；臂定义见 Design 原文，已折叠）：
  pilot：E0 0.371唯一胜（E1 0.299/E2 0.321/E3 0.283）；P2 0.786/2.1%大胜（P0 0.667/P1 0.609）；
  100轮（E0+P2，best ep60）：eDOS med 0.438/fail 17.1%，phDOS med 0.833/fail 2.5%，Cv 0.35；
  **训练网格冻结=E0+P2**。数据`data/grids_c2b/`（split复用）。
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
