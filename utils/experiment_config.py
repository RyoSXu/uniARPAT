"""Experiment configuration dataclass.

Single source of truth for all experiment hyperparameters.
Adding a new knob: (1) add a field here, (2) add argparse line in runner. Done.
"""
from dataclasses import dataclass, field, fields
from typing import Optional
import argparse


@dataclass
class ExperimentConfig:
    """All experiment hyperparameters in one place."""
    
    # --- Core ---
    model_name: str = "M1"
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-5
    seed: int = 42
    tag: str = ""
    skip_existing: bool = False
    data_dir: str = "./data/train4ARPAT"
    
    # --- Grid / Output dimensions ---
    edos_num: int = 128
    phdos_num: int = 64
    # R2a: shared Transformer decoder depth.  Six is the frozen B7 default.
    decoder_layers: int = 6
    # R2b: replace only the phDOS readout with atom-additive fixed-grid logits.
    use_atom_additive_phdos: bool = False
    
    # --- Feature encoding ---
    atom_feat: str = "legacy3"
    energy_code: str = "none"
    edos_grid: str = ""
    use_macro_lattice: bool = False
    
    # --- Shape-loss weights ---
    tv_w: float = 0.0
    grad_w: float = 0.0
    edos_slope_ratio: float = 0.0
    peak_w: float = 1.0
    tail_w: float = 1.0
    tail_start: int = -1
    
    # --- Training augmentation ---
    augment: bool = False
    disp_sigma: float = 0.01
    
    # --- Target normalization and masks ---
    norm: str = "sumnorm"
    use_mask: bool = False
    skip_test_eval: bool = False

    # --- C4 CUDA automatic mixed precision (default off) ---
    use_amp: bool = False

    # --- E6 fixed-window length buckets plus dynamic padding trim ---
    use_bucket_batch: bool = False
    
    # --- Optimizer and regularization overrides ---
    dropout: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_epochs: Optional[int] = None
    lambda_ph: Optional[float] = None
    grad_clip: Optional[float] = None
    
    # --- Distribution-loss weights ---
    w_w1: Optional[float] = None
    w_huber: Optional[float] = None
    
    # --- Blind-inference scale heads ---
    scale_mode: str = "eta"
    scale_sup_w: float = 1.0
    eta_sup_w: float = 1.0
    delta_edos: float = 0.09375
    delta_phdos: float = 19.6875
    freeze_backbone: bool = False
    init_ckpt: str = ""
    # Candidate-1 pilot fairness shim: when set, re-seed the training RNG right
    # after an init checkpoint is loaded, so arms with different module counts
    # enter the first iteration from the same random stream. Default off keeps
    # the existing B7 trajectory byte-identical.
    reset_rng_after_init: bool = False
    
    # --- Optional boundary scalars ---
    scalar_mode: str = "none"
    scalar_sup_w: float = 1.0
    
    # --- Optional sparse graph ---
    use_g1: bool = False
    g1_r_cut: float = 5.5
    g1_max_neighbors: int = 48

    # --- G2a periodic multi-image edge-conditioned message (fixed R=5.5) ---
    use_g2: bool = False
    g2_r_cut: float = 5.5
    # Candidate-1 edge content function: "radial" keeps G2a; "joint" is only
    # valid together with use_g2=True and is never silently ignored.
    g2_content_mode: str = "radial"
    
    # --- Optional coordinate-conditioned output trunks ---
    q1_coord: bool = False
    q1_hidden: int = 128
    q2_fourier: bool = False

    # --- C5 token-level decoder MoE (frozen pilot design) ---
    c5_moe: bool = False
    c5_moe_balance_w: float = 0.01

    # --- R1a parameter-matched pointwise MLP readout ---
    r1a_point: bool = False

    # --- R1b coordinate-generated decoder query ---
    r1b_coord: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Build config from parsed argparse namespace."""
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in vars(args).items() if k in valid}
        # argparse uses 'model' but dataclass uses 'model_name'
        if 'model' in vars(args) and 'model_name' not in d:
            d['model_name'] = args.model
        return cls(**d)
