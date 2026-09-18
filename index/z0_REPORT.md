# Z0 冻结报告（2026-09-15）：池内逐材料 N_valence（原胞口径）

## 交付物
- `index/z0_nvalence.parquet`：mpid, N_val, method, T_occ, ratio, confmin（24,988 行，零缺失）
- `index/z0_zval.json`：96 元素 PBE-ZVAL 表 + 置信度（2 精确 / 1 模式 / 0 暂定）
- `data/train4ARPAT/{train,valid,test}/nvalence_{split}.npy`：训练侧车（对齐 split index 顺序）

## 方法
- 逐材料自源占据积分（E≤Ef），census 18,644（mp_edos_raw.jsonl）+ 增量 6,340（Delta total-dos，task-id 直连），原胞归一经 task/prim 整数比。
- 缺陷修正史：全谱积分错计能带数（改占据截断）；record-Ef 跨任务错位（MgO 实锤：record-Ef 落导带，改能隙感知 plateau-median）；DOS-cell≠存储结构（整数比校验）；插值振铃（plateau-median + 整数 snap）。
- 信任规则：snap（全占据+近整数+元素相容）> 自源（能隙/金属）> 元素表回退；金属-only 接受带 25%。

## 质量
- own 20,546（82.2%：gap+snap 9,808 / metal+snap 7,334 / unsnapped 3,404），其中 84% 距整数 <0.02；
- fallback 4,442（17.8%：元素表精确值，含 bad_ratio 311 / comp_mismatch 14 / dev 过大）；
- 元素表关键判定：MP 用 _d（Ga/Ge/In/Sn/Pb/Tl=13/14/13/14/14/13）、_pv（Ti/V/Cr/Mo/W/Tc/Ru/Rh/Re/Os=10/11/12/12/12/13/14/15/13/14）、_sv（Sc/Y/Zr/K/Ca/Sr/Ba=11/11/12/9/10/10/10/10）、Na=7；Fe/Co/Ni/Cu/Zn 标准（8/9/10/11/12）；稀土轻 11（Ce 12）重 9（Yb 8）；U/Th/Pa=14/12/13。
- 暂定（conf-0，回退时标注）：Mn=13、Sm=11、Pm=11、Np=15、Eu/Gd 占位。

## 用途
- H1 η_e 监督分母：γ_true = S_win_e / N_val（S_win_e 取 sumnorm 和槽）。
- 注意：N_val 口径 = MP-PBE-DOS 任务实际价电子数；corrupt-DOS 样本已回退，标签形状不受影响（sumnorm），仅 γ 目标用表值。

## 附录：Q1 截断隔离判定（D1–D4）
> Backlog 与 `data/quarantine_q1.json` 引用的 D1–D4 即本节。此前版本缺失，现补齐。

- D1 定义：截断率 `trunc = C_win / C_full`。`C_win` 为固定窗 E0 内电子数（盒平均和槽×Δ_e），`C_full` 为全谱电子数；`trunc < 0.5` 判截断（MP-DOS 任务 NBANDS 不足，截断集胞中位 54 原子 vs 健康 8 原子，全谱电子数中位仅 9%）。
- D2 名单：`trunc < 0.5` 计 1668 条 + Delta 不可读且 `γ_label < 0.1` 计 14 条，共 1682 出池（train 1334 / valid 164 / test 184）；2 个不可读但 γ 健康保留（无罪推定）。名单见 `data/quarantine_q1.json`，物化脚本 `tools/getdata/q1_rebuild.py`。
- D3 口径：新缓存 `data/train4ARPAT/` 为干净口径 18706/2313/2287，γ 地板 0.002→0.106+，零泄漏；旧缓存归档 `data/archive/train4ARPAT_20260916_preQ1/`（20040/2477/2471），旧成绩一律标注 pre-Q。
- D4 影响：eDOS 形状监督毒药出池后，B7 电子 med 0.463→0.518；phDOS 为独立任务，保持健康。此后成绩一律 Q1 口径。
