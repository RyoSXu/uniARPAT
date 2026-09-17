# uniARPAT 待开发总清单（Backlog，主干文档）

**建档**：2026-09-09 ｜ **状态**：待开发 ｜ **前置完成**：对照Hygiene五项（`3e6df8f`）、工程E1–E4（`5b3d280`）
**关联文档**：[技术路线Roadmap.md](./技术路线Roadmap.md)、archive/Hygiene实施记录、[数据加工规范DataSpec.md](./数据加工规范DataSpec.md)
**冻结基线**：v0-legacy（旧成绩）→ h1（hygiene重跑，待执行）

---

## ★ 总指挥计划（2026-09-15冻结：按序走完 = 最强模型，全串行）

| Phase | 事项 | 资源 | Gate |
|---|---|---|---|
| Phase 0 | Z0 ZVAL对照表 + H1 η/γ头实现+冒烟 | CPU半天 | 表冻结、冒烟过 |
| Phase 1 | H1 η pilot（M1+η/γ vs 对照） | GPU~2h | 总体+盲gap分布见 verdict |
| Phase 2 | B5 生产百轮（sumnorm+dropout0+E0P0+η若合并） | GPU~7h | 一趟还三债：dropout长跑验证+B2生产基线+E9对照基线 |
| Phase 3 | S1 边界标量+98/2 flag → 固定窗盲推全闭环 | GPU~2h | blind pipeline运营化 |
| Phase 4 | E9-P0（合同§1–4），后Q/G/L串行，P0预训练再后 | 另议 | 拿B5当对照基线 |

- **Gate铁律**：上一步verdict落盘才开下一步；挂起项进队列标注，不返工已冻部分。
- **明确不排（E9后队列）**：C5新读出（含逐点解码验收）→ warp网格pilot → 非均匀大窗mechanization（C5门槛）→ 密度臂（备选）→ C1会师/D3/C3 → 连续谱场（E9-R&D）。
- **主线零数据返工**：Phase 0–4标签一根不动（η/标量信号全从现有标签+掩膜+ZVAL现算）；重切只发生在E9后队列，纯CPU活，不挡路。
- **执行状态（2026-09-17）**：Phase 0✓（Z0冻结+H1实现）→ Phase 1✓（η合并）→ Phase 2✓（B5收官+B6改dropout0.05）
  → Phase 3半（S1：wmax/Eval已验证未合并，Eg死刑，flag兑现为Q1）→ **Q1截断隔离已执行（缓存18706/2313/2287）** → Phase 4 E9待开（配方sumnorm+E0P0+eta+dropout0.05，对照须Q1重跑）。

---

## A. 数据v2线（按执行顺序排列）

- [x] **A1 Delta整表下载**（eDOS 4.35GB/phDOS 0.35GB已落地）。
- [x] **A2 对齐验证**（MP↔v1中位0.9935；83%单自旋；3%细胞不一致；MP↔JARVIS gap：eDOS 0.54/phDOS 0.85）。
- [x] **A3 census收尾** → [Design-A3](2026-09-10-Design-A3census收尾.md)：有效全集154,373；声子26,609；**真双谱18,644**（7,965有声子无eDOS，已逐个复核，见`edos_absent.json`）；unresolved=0。
- [x] **A3b PhononDB复算入库**（10,034/10,034零错误，184MB；maxfreq中位22.7THz；日志见03工作日志）。
- [x] **A3c PhononDB覆盖映射**（2026-09-11收官，日志见03工作日志）：
  10,034 serial与复算1:1；9,938译出唯一canonical（96死ID挂unresolved，`mp-867xxx`系为主）；
  有效全集内9,936；与MP声子集交5,545/真双谱交4,660；**增量核心4,391**（ID口径；
  谱落实4,049，见Top-up#1；分布验证略）。
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
  winsorize clip1,612/宽峰保留530（单点bin修复后重跑值）；CIF round-trip 1000/1000。
- [x] **A5 processed落盘**（`data/train4ARPAT/` 20,040/2,477/2,471，--check+forward冒烟通过；
  2026-09-15：v2转正接管该名，v1归档`data/archive/`；
  2026-09-16：Q1隔离后同名缓存为干净口径18706/2313/2287，旧缓存归档`data/archive/train4ARPAT_20260916_preQ1/`）。
- [x] **A6 分层切分v2**（8:1:1 seed42，三层±2%，成分零跨集断言，探针1条，He跳过）。
- [x] **Q1 截断隔离**（2026-09-16，名单`data/quarantine_q1.json`，机制见`index/z0_REPORT.md` D1-D4）：
  MP-DOS任务NBANDS不足（截断集胞中位54原子 vs 健康8原子），全谱电子数<50%（中位9%），
  eDOS形状 supervision 毒药（学对被罚）；phDOS独立任务健康、结构健康。
  口径：trunc<0.5计1668 + delta不可读且γ_label<0.1计14，共**1682出池**（train-1334/valid-164/test-184）；
  2个不可读但γ健康保留（无罪推定）。旧缓存归档`data/archive/train4ARPAT_20260916_preQ1/`，
  新缓存`data/train4ARPAT/`（**18706/2313/2287**，γ地板0.002→0.106+，零泄漏，CPU冒烟过），
  物化脚本`tools/getdata/q1_rebuild.py`（A5同构）。此后成绩均为Q1口径；旧基线（B5等）标注pre-Q。
- [ ] **A7 映射v2** → [Design-A7](2026-09-10-Design-A7映射.md)（L1✅49,942零冲突；L2待议；L3标记）。
  - 2026-09-12决议：死ID（PhononDB 96 / JARVIS死leg 4,686+昵称37，MP已删库）**永久出池**，
    L2不再覆盖（raw保留备查，不删）；集外6条（2+4）同样出池。
- [x] **A7b JARVIS覆盖映射**（2026-09-11收官，ID级，谱未入库）：
  45,154/49,942 JVASP已映射（43,872唯一canonical；死leg 4,686+昵称37挂unresolved）；
  有效内43,868；MP声子交15,815/真双谱交13,142；**缺口天花板28,053**（ID口径；
  谱落实3,600，见A7c/Top-up）；
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
- [x] **B4 超参扫描**（2026-09-15收官，M1@E0P2+sumnorm，10轮/臂，旧网格pilot口径）：
  dropout 0（0.466/4.9%）唯一胜出并入默认；wd0.1/wu2/clip平；λ0.5边缘(+0.01)不追，λ2.0负向；
  C1.3重审（T2系数+lr1e-5）：崩塌解除但仍逊对照0.06，维持挂起。过拟合警告：下次长跑val监控守。
- [x] **B2 v2基线重标定** → [Design-B2](2026-09-10-Design-B2v2基线重标定.md)（2026-09-14由C2b终考兼任，不另跑：
  M1@E0P2-100：eDOS med 0.438/fail 17.1%，phDOS med 0.833/fail 2.5%。**此为旧网格成绩，存档**；生产E0P0基线见B5）。
- [x] **B5 生产百轮**（Phase 2，2026-09-16收官，M1×100 tag _b5，E0P0+sumnorm+dropout0+eta）：
  best ep33（balanced-score选型）；test：eDOS med 0.463/fail 8.86%，phDOS med 0.696/fail 4.69%；
  盲测test：声子 0.686（gap p50/p90/p99=0.0006/0.044/0.52），电子 0.394（gap 0.023/0.54/尾巴）。
  三债已还：dropout长跑（见下过拟合审计）+ B2生产基线（本条，**pre-Q口径存档**）+ E9对照基线（须Q1重跑，不可直引）。
- [x] **B6 dropout复赛**（2026-09-16收官：0/0.05/0.1 × 35轮，test同口径）：
  dropout0（B5）：eDOS 0.463/8.86，phDOS 0.696/4.69；
  **0.05**：eDOS 0.510/5.67，phDOS 0.738/3.16，盲e 0.448/盲p 0.730；
  0.1：eDOS 0.504/5.79，phDOS 0.746/2.75，盲e 0.438/盲p 0.738。
  判决：0.05与0.1双杀0（eDOS med +0.05、fail砍半）；0.05 vs 0.1完美对称分裂
  （0.05全取电子、0.1全取声子），按电子瓶颈优先+最小有效正则化取**0.05**进默认，
  0.1记为已验证备选。B4-dropout0 verdict正式限定为10轮 pilot 口径。
  E9解锁：配方=sumnorm+E0P0+eta+dropout0.05，E9对照跑~35轮。

## 0. 最高优先级（2026-09-10置顶）
- [ ] **E9 Encoder v2** → [Design-E](2026-09-10-Design-E编码器v2.md)（**Phase 4**，P0合同§1–4先行，Q/G/L串行，轮数另定）：自研稀疏周期图+低阶等变（G1必做/G2G3选做）+坐标query（Q1Q2必做/Q3选做）+守恒前面积审计，接口`[B,L,512]+[B,512]`不变。开工门槛B5；与C1/C2见Design-E§5分工。训练与对照一律Q1干净池，旧基线对比须标注pre-Q。

## C. 模型线（B1/B2开门后）

- [ ] **C1 Phase 1四件套**（2026-09-14/15四轮pilot全挂，见日志；挂起条件各记）：
  C1.1（24维+静默tok，打平，机制验证通过，待C5会师/100轮）/
  C1.2（能量+RFF，打平，E2无翻转）/ C1.3（TV三档崩塌→B4对齐重审仍逊，关闭单独立项）/
  C1.4（位移增强，全平；旋转在82格式下证伪为no-op，已删）。
- [ ] **C2 Phase 1.5**：SumNorm-KL双轨+Huber、η/γ有界覆盖头、截断mask（已判留评估端）。
  - [x] C2.1 SumNorm-KL/W+Huber（2026-09-14合并，默认；见日志）。
  - [ ] C2.1b attribution臂（排队，E9后）：SumNorm+SmoothL1，拆归一与loss的各自贡献。
  - [ ] C2.3 截断mask进loss（2026-09-15挂起→判死刑：声子-0.028，掩膜零bin是边界监督；**掩膜永留评估端**）。
  - [x] **H1 η/γ有界覆盖头**（盲推最终式，2026-09-15 pilot通过并合并为默认）：
    声子`Scale=3N·η̂/Δ`、电子`Scale=N_val·γ̂/Δ`，Sigmoid[0,1]有监督头（`scale_mode=eta`，
    监督η=S_win·Δ_ph/3N、γ=S_win·Δ_e/N_val，和槽是盒平均之和、差一个Δ——pilot v1实锤教训）；
    前置Z0 ZVAL表（`index/z0_nvalence.parquet` + 训练侧车，用量82.2%自源/17.8%表回退）。
    pilot verdict（M1×10，对照_h1ctl vs _h1eta2）：形状不劣（eDOS 0.466/4.94 vs 0.468/5.34；
    phDOS 0.726/1.50 vs 0.717/1.38）；声子盲 gap p50/p90/p99 = 0.001/0.029/0.39
   （退役0.92公式：0.008/0.33/1.80，全分位碾压）；电子盲 med 0.403 vs oracle 0.466，
    gap p50 0.023，p99 尾巴=窗内仅0.01~0.1 e⁻/原子的E_F错位标签（~8%，S1 flag首个客户，见下）。
    旧案归档：C2.4对数头`N×exp(MLP)`被替代；c24A臂因跨归一化污染废弃；`_h1eta`（漏Δ版）作废保留。
  - [x] **Z0 POTCAR-ZVAL对照表**（Phase 0，H1-eDOS前置，2026-09-15冻结见`index/z0_REPORT.md`）：MP默认POTCAR→逐元素ZVAL→池内逐材料N_valence冻结；
    元素表关键判定：_d系（Ga/Ge/In/Sn/Pb/Tl）/ _pv系（Ti/V/Cr/Mo/W/Tc/Ru/Rh/Re/Os）/
    _sv系（Sc/Y/Zr/K/Ca/Sr/Ba）、Na=7；暂定Mn=13/Sm=11（conf-0）。
    `_sv/_pv`半芯坑按对照表口径统一；矩（μ₀/μ₁/μ₂）只进loss insult，不做生成。
  - [ ] **S1 边界标量+复核队列运营化**（Phase 3，pilot判完一半）：
    wmax/Eval头已验证（35轮test MAE 58cm⁻¹/0.69eV，形状无伤，合并不急，默认仍off）；
    Eg回归 formulation 死刑（95%零目标+MSE塌缩，改分类+量级另起队列）；
    flag已兑现为Q1隔离（上）。固定窗盲推闭环=形状+η/γ（已合并）+Q1干净池，达成。
  - [x] C2.2 守恒缩放（机制验证通过即退役，2026-09-15）：
    blind-3N×0.92达oracle水平（0.839 vs 0.833）证伪"总量不可学"，但**0.92全局常数退役**（仅52%样本在±5%内）、
    maxfreq硬guard删除，统一由H1 η/γ接管；M5 ScaleHead可删（E阶段）；推理接线（cif2dos盲模式）归E阶段。
- [x] **C2b 网格对照试验**（2026-09-14收官；臂定义见 Design 原文，已折叠）：
  pilot：E0 0.371唯一胜（E1 0.299/E2 0.321/E3 0.283）；P2 0.786/2.1%大胜（P0 0.667/P1 0.609）；
  100轮（E0+P2，best ep60）：eDOS med 0.438/fail 17.1%，phDOS med 0.833/fail 2.5%，Cv 0.35；
  **生产训练网格冻结=E0+P0**（2026-09-15改判：P2 82%空转+稀释核心区，仅作实验记录；P1死因=卷积头假设等距网格，非均匀网格必须等C5新读出）。
  数据`data/grids_c2b/`（split复用）。覆盖实测：E0全覆盖93.9%/平均98.1%；P0全覆盖3.8%/平均52.2%（左缘oversize，掩膜已无害化）。
  声子原生包络CDF（MP raw）：minfreq中位-70/p5-434/p1-855；maxfreq中位515/p80-966/p95-1627/p99-3700。
  P0左缘微调（-280→-450量级）列为可选，不挡主线。
- [ ] **C3 Phase 3**：PhysMoE晶体级路由 vs 能量级路由 A/B；[CLS] Cell Token；PotNet长程后置。
- [ ] **C5 解码器MoE** → [Design-C5解码器MoE.md]（E9后：token级4专家Top-2+负载均衡 vs 晶体级PhysMoE；位置/宽度/层数；
  新增验收项：P1式非均匀网格+逐点/能量条件式读出 vs 均匀+卷积——预测前者反超，此为P1死因解释的可证伪版本；
  warp网格pilot与宽窗mechanization挂C5门槛之后）。
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
9. 盲推正统=H1 η/γ有界头（2026-09-15）：0.92全局常数退役、maxfreq硬guard删除（C2.2/C2.4决议）。
10. 掩膜永留评估端，禁入loss（C2.3判死刑）；缺失记mask永不填0。
11. 生产网格E0+P0冻结；P2仅实验记录；网格重切只随verdict走，不预跑。
12. 网格设计双规则（2026-09-15）：窗口缘卡95%材料包络、尾部交掩膜+η；bin宽≤min（DFT展宽，典型线宽/2），工程（2的幂/head维度）排后。
13. 非均匀网格⇔读出头绑定：卷积头只配均匀网格；非均匀/宽窗必须等C5新读出（P1教训）；直接放大窗口若不保密度则无意义。
14. 终极形态=逐点/连续解码（query任意能量，杀死窗口/分箱问题），经C1.2 query → C5读出 → E9-R&D连续谱场转正。

## 对照实验总原则（2026-09-09冻结）

- **三层对照**：(a)超参（lr/batch/dropout/λ权重）→ (b)设计配置（网格/归一化/head/loss形式/切分策略/源配比）→ (c)数据配置（源混合、增强开关、winsorize阈值）。三层一视同仁：**凡是有≥2个合理选项、且会影响结果的，都配消融臂**。
- **防组合爆炸**：10-epoch pilot初筛（便宜），胜者进100轮；一次只动一个因子；每臂预写判决规则（赢了如何、输了如何），不做开放式实验。
- **等算力对照**：凡涉及"更多数据/更多步数"的臂，总步数拉平后再比，杜绝把算力涨点当方法涨点。
- **跨归一化对口径**（2026-09-15增补，c24A教训）：复用checkpoint或跨臂对比前，
  先核`config_used.yaml`的norm/loss键；minmax骨干上跑sumnorm监督一律视为污染，直接废弃。
- **盲推verdict看gap分布**（2026-09-15增补，C2.2教训）：凡涉及blind/oracle的臂，判决不只看中位，
  必须报gap分布（p90/p99），防止中位达标、尾部崩盘。
