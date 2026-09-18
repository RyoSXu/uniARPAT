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
    
    # --- Feature encoding ---
    atom_feat: str = "legacy3"
    energy_code: str = "none"
    edos_grid: str = ""
    
    # --- Shape-loss weights ---
    tv_w: float = 0.0
    grad_w: float = 0.0
    peak_w: float = 1.0
    tail_w: float = 1.0
    tail_start: int = -1
    
    # --- Training augmentation ---
    augment: bool = False
    disp_sigma: float = 0.01
    
    # --- Target normalization and masks ---
    norm: str = "sumnorm"
    use_mask: bool = False
    
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
    
    # --- Optional boundary scalars ---
    scalar_mode: str = "none"
    scalar_sup_w: float = 1.0
    
    # --- Optional sparse graph ---
    use_g1: bool = False
    g1_r_cut: float = 5.5
    g1_max_neighbors: int = 48
    
    # --- Optional coordinate-conditioned output trunks ---
    q1_coord: bool = False
    q1_hidden: int = 128
    q2_fourier: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig":
        """Build config from parsed argparse namespace."""
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in vars(args).items() if k in valid}
        # argparse uses 'model' but dataclass uses 'model_name'
        if 'model' in vars(args) and 'model_name' not in d:
            d['model_name'] = args.model
        return cls(**d)
