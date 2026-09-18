# uniARPAT: Unified Ab-initio Representation for Phonon and Electron Density of States

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#-license)

**uniARPAT** is the next-generation unified physical deep learning framework for the joint, end-to-end prediction of **electronic density of states (eDOS)** and **phonon density of states (phDOS)** directly from unrelaxed crystal structures.

Built upon the foundations of **ARPAT**, uniARPAT introduces leakage-free Shape-Scale inference (H1 η/γ heads), self-healing bandgap preservation, and material-specific thermodynamic integration. Week-2 的解耦/对称头/门控三假设已被 h1 否决（见下），当前默认回退 M1。

---

## 🌟 Key Architectural Advancements

```
                      [Crystal Structure (Atoms + Coords + Cell)]
                                         │
                                         ▼
                            [Geometric Crystal Encoder]
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [eDOS Decoder]                  [phDOS Decoder]
                         │                               │
                         └──────► [Post-Decoder] ◄───────┘
                                  [Gated Cross-]
                                  [ Attention  ]
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
               [MultiScale Head]                 [DeepConv1d Head]
                  (~0.788M)                          (~0.788M)
                         │                               │
                         ▼                               ▼
                  [eDOS Shape [0,1]]              [phDOS Shape [0,1]]
                          │                               │
                          └──────────────┬────────────────┘
                                         ▼
                     [H1 η/γ Heads from global features]
                      声子 Scale=3N·η̂/Δ，电子 Scale=N_val·γ̂/Δ
                         （图中汇合箭头仅示意相乘，不表示因果）
                                         │
                                         ▼
                         [True Physical Absolute Spectra]
                                        │
                                        ▼
                     [Macroscopic Thermal & Thermo Properties]
                     (Julian-Slack κ_L, Debye Temp, Cv, Sv, Fvib)
```

1. **Decoupled Dual-Decoder Architecture**:
   Eliminates inter-task negative transfer by separating electronic and vibrational latent representations, allowing specialized queries to converge without gradient interference.
2. **Post-Decoder Zero-Initialized Gated Cross-Attention**:
   Employs learnable gating scalars ($\alpha_e, \alpha_p$) initialized to $0.0$. The model starts as strictly decoupled and smoothly learns electron-phonon feature modulation as training progresses.
3. **Capacity-Symmetric Output Heads (~0.788M params)**:
   Resolves historical capacity skew (30.7M vs 1.5k) by balancing the phDOS `DeepConv1dHead` (787,969 params) and the eDOS `MultiScaleResidualHead` (788,225 params) to within 0.032% parameter parity.
4. **Leakage-Free Shape-Scale Decoupled Prediction**:
   The `ScaleHead` MLP autonomously predicts physical energy scales from global crystal features. This eliminates reliance on ground-truth min/max labels during inference, enabling true blind deployment on unknown materials.
5. **Self-Healing Bandgap Preservation (`safe_shape_norm`)**:
   Guarantees mathematically flat zero-density in semiconductor bandgaps while dynamically protecting neurons against Dying ReLU stagnation through smooth Softplus fallback.
6. **Material-Specific Julian-Slack Lattice Thermal Conductivity**:
   Integrates unit cell volume $V_{\text{atom}}$, average atomic mass $\bar{M}$, and optical phonon suppression factor $n_{\text{atoms}}^{2/3}$ with the calibrated Julian-Slack equation ($A=3.1\times 10^{-6}$), aligning ML predictions with solid-state physics.

---

## 📁 Repository Structure

```text
uniARPAT/
├── README.md / AGENTS.md / docs/STATUS.md   # 入口：项目说明 / agent导航 / 活状态
├── requirements.txt / requirements-lock.txt
├── model/ / datasets/ / utils/              # 骨干+头 / 数据集 / 特征与调度
├── tools/getdata/ / tools/eval/ / tools/legacy/  # 抓数加工 / verdict脚本（禁/tmp）/ 退役入口存档
├── run_ablation_experiments.py / cif2dos.py / thermo_props.py  # 唯一训练入口 / 盲推 / 热力学库（根目录仅此3个py）
├── configs/config.yaml                      # 被runner改写；可复现看 output/ablation_*/config_used.yaml
├── data/train4ARPAT/                        # Q1干净缓存 18706/2313/2287；旧缓存 data/archive/*_preQ1/
├── index/z0_*.parquet+json + z0_REPORT.md   # ZVAL表+Q1 D1–D4
├── tests/                                   # 单测（截至09-18为34项，见下）
├── docs/INDEX.md+GLOSSARY.md                # 文档地图+术语；中文历史目录名冻结保留
├── results/                                 # history_*.csv + test_*_summary.csv（入库）
└── output/                                  # checkpoint+config（不入库，以results为准）
```

---

## 🚀 Quickstart

### 0. 数据前置（必做）
本仓库训练依赖 Q1 干净缓存。先校验：
```bash
ls data/train4ARPAT/manifest.json   # 期望 train 18706 / valid 2313 / test 2287
```
缺数时用 `tools/getdata/q1_rebuild.py` 物化，不要重跑 A1–A6。旧缓存只在 `data/archive/*_preQ1/` 存档。

### 1. Environment Installation
Ensure Python 3.10+ and CUDA are installed (V100 实测用 CUDA 11 系镜像；CUDA 12 仅在新卡验证过）。Then install requirements:

```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
Verify model parameters, self-healing activations, and physical equations:

```bash
python3 -m unittest discover tests
```
*(截至 09-18 为 34 项，约 1 分钟；CPU 机约 75 秒。）*

### 3. Pilot 初筛（10 Epochs）
```bash
python3 run_ablation_experiments.py --model M1 --epochs 10 --tag _xxx
```
- Q1 实测（V100，M1）：约 190 s/epoch，峰值显存约 7.2 GB。旧 `107.8 s / 9.16 GB` 为 v1 存档值，勿引。

### 4. 生产对照（B7 配方，不要裸跑）
```bash
# 唯一合法对照（Q1，sumnorm+E0P0+eta+dropout0.05）
python3 run_ablation_experiments.py --model M1 --epochs 35 --tag _e9ctl
```
禁 ` --model all --epochs 100`：会用旧默认跑出污染成绩并覆盖结果。`--tag` 必加，跨归一化禁复用 checkpoint。

---

## 📊 Baseline Benchmarks & Ablation Design

### 1. 现行基线 B7（Q1 干净池，2,287 测试样本，2026-09-17）
生产配方（sumnorm + E0P0 + H1 η/γ + dropout 0.05），best ep33：

| Physical Target | eDOS med / fail | phDOS med / fail | Notes |
| :--- | :---: | :---: | :--- |
| **B7 `_e9ctl`（唯一合法对照）** | **0.518 / 5.73%** | **0.741 / 3.50%** | Cv MAE 0.30 J/(mol-atom·K)；盲声子 0.735（gap p50 0.0008），盲电子 0.480（gap p50 0.010） |
| 存档 pre-Q（B5/B6/H1，旧缓存） | 0.463–0.510 / 5.67–8.86% | 0.696–0.738 / 3.16–4.69% | 跨池禁直接比涨点 |

旧 1,371 样本 oracle 值（phDOS 0.694 / eDOS 0.521，Cv 0.364）为 v0-legacy 存档，勿引。

### 2. Ablation Planning Matrix (Table 1；Week-2 规划值，参数为规划值)

> NOTE（现行对照是 B7 Q1，见上表）：下表 M1 的 0.521/0.694 为 Week-2 规划初值；
> h1（09-11）已收官，B5/B6（pre-Q）已存档。M2–M5 "Ablation run" 为空表示未测得可用值——不要引用为成绩。
> 参数列为规划值（hygiene 后实测约 71.1M，相对关系仍有效）。

| Variant | Decoder | Cross-Modal Interaction | eDOS Head | phDOS Head | Loss & Scale Scheme | Params | eDOS $R^2$ (med/mean) | phDOS $R^2$ (med/mean) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 (Baseline)** | Shared | None | 1-layer (1.5k) | 6-layer (30.7M) | MSE (Oracle scale) | **77.63M** | 0.521 / 0.374 | 0.694 / 0.585 |
| **M2** | Decoupled | None | 1-layer (1.5k) | 3-layer (0.78M) | MSE (Oracle scale) | **72.96M** | *Ablation run* | *Ablation run* |
| **M3** | Decoupled | None | Multi-Scale (0.78M) | 3-layer (0.78M) | MSE (Oracle scale) | **73.75M** | *Ablation run* | *Ablation run* |
| **M4** | Decoupled | Zero-Init Gated | Multi-Scale (0.78M) | 3-layer (0.78M) | MSE (Oracle scale) | **75.85M** | *Ablation run* | *Ablation run* |
| **M5 (Full)** | Decoupled | Zero-Init Gated | Multi-Scale (0.78M) | 3-layer (0.78M) | Physical Loss + Shape-Scale | **75.93M** | 旧ScaleHead零梯度已作废，勿引（盲测曾崩） | — |

> h1 verdict（09-11）：解耦≈零（砍回共享）、对称头负（回退轻量）、门控负（判死刑转MoE）。
> 当前默认回退 M1，上表 M2–M4 为历史规划假设，不代表现行最优。

---

## 📖 Citation

Manuscript in preparation. ARPAT base reference to be added after peer-review confirmation.

---

## 📜 License
License 待补（原 MIT 链接无文件，暂按内部使用）。
