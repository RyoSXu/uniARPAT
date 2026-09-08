# uniARPAT: Unified Ab-initio Representation for Phonon and Electron Density of States

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Base Paper](https://img.shields.io/badge/npj%20Comput%20Mater-2026%20In%20Press-red.svg)](https://doi.org/10.1038/s41524-026-02199-3)

**uniARPAT** is the next-generation unified physical deep learning framework for the joint, end-to-end prediction of **electronic density of states (eDOS)** and **phonon density of states (phDOS)** directly from unrelaxed crystal structures.

Built upon the foundations of **ARPAT** (*npj Computational Materials*, 2026, Article in Press, [DOI: 10.1038/s41524-026-02199-3](https://doi.org/10.1038/s41524-026-02199-3)), uniARPAT introduces architectural decoupling, zero-initialized cross-modal gated modulation, leakage-free Shape-Scale inference, and material-specific thermodynamic property integration.

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
                            [ScaleHead (Log-Scales)]
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
├── README.md                      # Publication-grade documentation
├── requirements.txt               # Strict package dependencies
├── .gitignore                     # Git tracking configuration
│
├── model/                         # Core neural network architectures
│   ├── transformer.py             # Decoupled Transformer with safe_shape_norm
│   ├── heads.py                   # DeepConv1dHead, MultiScaleResidualHead, GatedAttention
│   └── model.py                   # Multi-task loss functions & vectorized metrics
│
├── datasets/                      # Data loaders and dataset definitions
│   └── dataset.py                 # Multi-task crystal & DOS dataset
│
├── tests/                         # Automated unit test suite (10/10 passing)
│   ├── test_model_heads.py        # Output head capacity and parameter verification
│   ├── test_scale_norm.py         # Shape-Scale & Dying ReLU self-healing tests
│   ├── test_gated_attention.py    # Zero-init identity & gate parameter tests
│   └── test_thermo_props.py       # Julian-Slack and high-T Dulong-Petit tests
│
├── docs/                          # Comprehensive technical documentation
│   ├── 01_开发文档/                # Core R&D proposal & Pilot acceptance report
│   ├── 02_审查复核/                # 4-stage independent peer review records
│   └── 03_工作日志/                # Full chronological development log
│
├── figures/                       # Publication-quality benchmark figures (Fig 1 - Fig 4)
├── results/                       # Per-sample test metrics CSV and pilot checkpoints
├── run_ablation_experiments.py    # Week-2 end-to-end ablation runner (M1 - M5)
├── run_pilot_10epochs.py          # Week-1 10-epoch pilot verification script
├── evaluate_and_plot.py           # Evaluation analysis and figure plotting script
└── thermo_props.py                # Thermodynamic integration & Julian-Slack calculator
```

---

## 🚀 Quickstart

### 1. Environment Installation
Ensure Python 3.10+ and CUDA 12.0+ are installed. Then install requirements:

```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
Verify model parameters, self-healing activations, and physical equations:

```bash
python3 -m unittest discover tests
```
*(All 10 tests should pass in under 0.2 seconds).*

### 3. Week-1 Pilot Verification (10 Epochs)
Quickly test training stability, VRAM consumption, and convergence:

```bash
python3 run_pilot_10epochs.py
```
- **Tested Performance (Tesla V100)**: 107.8 s/epoch, 9.16 GB peak VRAM, Loss drops by 43.7%.

### 4. Week-2 Full Ablation Suite (100 Epochs)
Launch the 5 ablation variants (M1 through M5) sequentially:

```bash
# Run all variants sequentially (approx. 16 hours on Tesla V100)
python3 run_ablation_experiments.py --model all --epochs 100

# Or run a single variant
python3 run_ablation_experiments.py --model M5 --epochs 100
```

---

## 📊 Baseline Benchmarks & Ablation Design

### 1. Independent Baseline Audit (1,371 Unseen Test Materials)
Evaluated strictly per-sample on completely unseen test crystals:

| Physical Target | Metric | Value (Mean ± Std / Median) | Notes |
| :--- | :--- | :---: | :--- |
| **phDOS** (64 bins) | MAE<br>$R^2$ | $0.0170 \pm 0.016$ / **0.0129**<br>$0.585 \pm 0.42$ / **0.694** | Failure rate ($R^2 < 0$): **7.37%** (101 / 1,371) |
| **eDOS** (128 bins) | MAE<br>$R^2$ | $2.70 \pm 2.82$ / **1.77**<br>$0.374 \pm 0.65$ / **0.521** | Failure rate ($R^2 < 0$): **14.66%** (201 / 1,371) |
| **Lattice Thermodynamics** | $\Theta_D$ MAE<br>$C_v$ MAE (300K)<br>$\kappa_L$ Pred (True) | **41.09 K**<br>**0.364 J/(mol-atom·K)**<br>**22.44 (19.56) W/(m·K)** | Relative error on $C_v$ vs Dulong-Petit is only 1.4% |

### 2. Ablation Planning Matrix (Table 1)

| Variant | Decoder | Cross-Modal Interaction | eDOS Head | phDOS Head | Loss & Scale Scheme | Params | eDOS $R^2$ (med/mean) | phDOS $R^2$ (med/mean) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 (Baseline)** | Shared | None | 1-layer (1.5k) | 6-layer (30.7M) | MSE (Oracle scale) | **77.63M** | 0.521 / 0.374 | 0.694 / 0.585 |
| **M2** | Decoupled | None | 1-layer (1.5k) | 3-layer (0.78M) | MSE (Oracle scale) | **72.96M** | *Ablation run* | *Ablation run* |
| **M3** | Decoupled | None | Multi-Scale (0.78M) | 3-layer (0.78M) | MSE (Oracle scale) | **73.75M** | *Ablation run* | *Ablation run* |
| **M4** | Decoupled | Zero-Init Gated | Multi-Scale (0.78M) | 3-layer (0.78M) | MSE (Oracle scale) | **75.85M** | *Ablation run* | *Ablation run* |
| **M5 (Full)** | Decoupled | Zero-Init Gated | Multi-Scale (0.78M) | 3-layer (0.78M) | Physical Loss + Shape-Scale | **75.93M** | Pilot (val/Blind): -0.346 / -0.659 | Pilot (val/Blind): 0.461 / 0.121 |

---

## 📖 Citation

If you use uniARPAT or ARPAT in your research, please cite:

```bibtex
@article{arpat2026,
  title={Unified Representation and Joint Learning for Electronic and Vibrational Band Structures of Crystals},
  author={uniARPAT Research Team},
  journal={npj Computational Materials},
  year={2026},
  note={Article in Press},
  doi={10.1038/s41524-026-02199-3}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
