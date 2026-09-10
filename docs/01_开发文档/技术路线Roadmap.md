# uniARPAT 双报告合并评估与统一攻坚路线（含 embedding / 数据专项意见）

**日期**：2026-09-09
**归档路径**：`docs/04_前沿探索/2026-09-09_uniARPAT_双报告合并评估与统一攻坚路线.md`
**输入A**：徐尚 & Antigravity 组《精度跃升前沿文献调研与架构级审查突破方案》（同目录，2026-09-09）
**输入B**：Muse Spark 独立审查（文献2024–2026 + 代码实测 + 数据量化）
**基线实测**：100-epoch复跑 `results/test_m*_summary.csv`（1371测试样本）——M3 eDOS R² med 0.493 / phDOS med 0.682 为最高；M4声子失败率最低 5.40%；M5盲测崩塌（eDOS med 0.279 / mean -0.130，失败率33.5%）。
**约束红线**：纯 CIF→DOS 端到端（输入仅晶体结构）；锁定 13,707 黄金基准（10965/1371/1371）；单卡 V100，参数 ≤85M，显存 ≤12GB，100轮 15–20h。

---

## 1. 核验结论（已实测）

- `utils/periodic_table_v2.csv` 确为 **29 列**（输入A的3.1节事实成立）。
- 死参数实测：**每 encoder 层约 1.09M（`self_attn`+`rel_proj`+`dir_proj`），6 层合计约 6.5M**——比输入A估计的 ~3.5M 更严重；另 `rbf_encoder` 全程未用，`RPEncoding` 无参却每层每batch重复计算球谐 6 次。
- eDOS 逐样本 max：中位 19.5 / p90 62.6 / p99 167.7 / 最大 10618；**80 个样本 max>200，4 个 >500**。
- 元素长尾：训练集 86 种元素；**He（Z=2）训练集计数为 0、测试集出现**；Ne/Ar/Kr/Gd 训练集仅 1–14 个原子。
- 成分重叠：测试集 1362 个去重成分中 **185 个（13.6%）** 在训练集出现过（同成分、可能不同结构；不算泄漏，但报告需分层）。
- phDOS 负频区（前14 bins <0）：训练集平均 1.9% 谱权重在内，**490 个样本 >20%**（虚频/展宽尾待定，需画谱确认）。
- **评估口径警告**：M1–M4 用真值 min/max 逆归一化（`model/model.py:230-240`），成绩为 **oracle**；唯一盲测 M5 已崩。**当前没有可用的盲测模型**，后续每个变体必须同时报 oracle / blind 两套 R² med + fail。

---

## 2. 输入A五大突破口裁决

| # | 断言 | 裁决 | 备注 |
|---|---|---|---|
| 3.1 | `Linear(3,512)` 秩≤3；扩 24 维+MLP | **成立，价值最高**，列合并路线 P0 | 补：`tok_emb` 对长尾元素（He零样本）是噪声，物理特征做主体、可学习表做残差；动工前先做 fail 富集统计（He/稀有元素/大max 是否富集） |
| 3.3 | Query 无能量坐标；连续傅里叶编码 | **成立，与输入B收敛** | 补：eDOS/phDOS 网格物理尺度不同，编码须分任务、带物理单位、标零点（Fermi bin / ν=0）；RFF 高频项只给 eDOS |
| 3.5 | TV/梯度 loss + 物理加权 | **半成立** | TV+加权对且便宜；但 Mat2Spec 真核是 **SumNorm+KL/W 双轨** 而非 TV。MinMax 本身被离群 max 劫持，只加 TV 是给错误目标精修。分两步：先 TV+加权（零风险），再 SumNorm-KL（换目标） |
| 3.2 | 晶格宏观量断层；[CLS] Cell Token | **成立** | V/ρ/V_atom/晶系确实从未进网。体积→幅值只作 Token 先验、不作硬约束 |
| 3.4 | 死参数 + RP 重复计算；上提缓存 | **成立，输入B漏检** | 纯工程红利（省 ~30% 显存 / 20% 时间），无精度风险，进 Phase 1 |

---

## 3. 输入A的三个漏项（输入B增量）

- **漏项1（最严重）：M5 ScaleHead 零梯度未被发现。** `train_one_step`（`model/model.py:148-173`）只监督 `outputs['edos/phdos']`，`phys_*/log_scale_*` 无梯度；`compute_shape_loss` 定义未调用。任何延续 shape-scale 的实验必复现 blind 转负。合并路线加 **P0-0**：先修 scale 监督，或改用解析守恒缩放（PET-MAD-DOS 2026：eDOS 按价电子数、phDOS 按 3N 模式数重缩放，零参数）。
- **漏项2：无预训练/增强/数据效率设计。** 补：去噪预训练或 DeNS 辅助头（二选一）、旋转增强（必做）、声子本征矢加权位移增强（与"未弛豫输入"噪声同分布）。全部在红线内（自监督/增强，不引外部数据与 Oracle 标签）。
- **漏项3：编码器只谈工程、不谈表示完备性。** 补：5–6Å 截断 mask + 距离衰减（Crystalformer 2024）先行，角度/二面角与周期不变表示（ComFormer ICLR 2024 / PerCNet 2025）后置；`1/c` 存储 hack 应改为显式不变格子表示。输入A的 PotNet-Ewald 长程排到截断之后。
- **风险**：M6–M8 预期数字（M8 eDOS 0.62–0.66）无实测锚点（Week-2 规划值已有前车之鉴），须经 10-epoch pilot 验证；PhysMoE（晶体级路由）与 token-wise 能量级路由（MGT 2025 依据）应做 A/B，不单押。

---

## 4. 文献依据对照（2024–2026 为主）

- PET-MAD-DOS（Ceriotti, Digital Discovery 2026）：无约束 Transformer + 增强学等变；电子数守恒缩放；派生量走 1D-CNN（带隙 MAE 0.19 eV）；LoRA 小数据微调打平 bespoke；逐能量通道 UQ。
- Orb / Orb-v3（Orbital Materials, 2024–2025）：非等变 GNS + 去噪扩散预训练 + 旋转增强 + equigrad；Huber(δ=0.01)；速度快 3–6 倍且高阶导数性质可用。
- EquiformerV2 + OMat24 + DeNS（Meta FAIR, ICLR 2024）：eSCN 高阶表示；OMat24 预训练→MP 微调 F1 破 0.9；声子 benchmark 上 OMat 版最强。
- ComFormer（ICLR 2024）/ PerCNet（2025）/ Crystalformer（ICLR 2024）：周期不变 + 几何完备（角度/二面角/距离衰减）。
- MGT（2025）：MoE 路由融合不变/等变双视角，MAE 降至多 21%。
- 声子位移增强（Benítez et al. 2025）：声子知情采样以更少数据胜随机（R² 0.85 vs 0.72）。
- FEDONet / RFF：MLP 谱偏置，随机傅里叶特征降高频误差（eDOS Van Hove 尖峰的机理级解释）。
- 冻结 uMLIP 迁移（npj Comput. Mater. 2025）：10% 数据微调即超从零训练——列为 P2 可选（涉外部权重，需红线特批）。

---

## 5. 统一施工顺序（红线内）

- **Phase 1（零风险，10-epoch pilot 验收）**：24 维原子特征 + MLP 投影 → 连续能量编码 + RFF（仅 eDOS）→ TV/梯度 loss + 物理加权 → 死参数清理 + RP 缓存 → 旋转 / 声子位移增强。
- **Phase 1.5（换目标，pilot 归因）**：SumNorm-KL 双轨 + Huber + 守恒缩放替代 M5 scale（P0-0）。
- **Phase 2**：[CLS] Cell Token → 截断 mask + 距离衰减（前置）→ PotNet 式长程（后置）。
- **Phase 3**：PhysMoE 晶体级路由 vs 能量级路由 A/B，决出旗舰。
- **贯穿要求**：fail 按晶系 / 带隙 / 成分见过与否 / 峰均比分层统计；每变体双报 oracle + blind。

## 7. 数据重取决策记录（2026-09-09）

- Key：沿用老 key（直连 REST 验证 200 有效）；`mp-api` 包因 pyarrow/numpy 构建链与 torch 环境冲突，**弃用，改直连 REST**（`X-API-KEY` + `requests`，已验证）。
- 范围：先双谱（eDOS + phDOS）；Raman（CRD 5099）只做 ID 交集摸底与抓取，不进本轮训练——分阶段归因。
- IR/介电：暂缓，已记录。
- v1 数据冻结归档；v2 上先重跑 M1 标定数据偏移，再谈架构涨点。

---

## 6. 定价

输入A是"Phase 1–3 施工图"，价值在可执行性；输入B是"漏项清单 + 2026 锚点"，价值在堵住 M5 崩塌、无监督 scale、数据效率三坑。合并后 Phase 1 可直接开工；Phase 1.5 与 Phase 3 的 A/B 为新增关键决策点。
