"""Configuration, matched initialization and identity for local/global trials.

No training is dispatched here. Historical ZP100 fingerprints are deliberately
absent: a canonical untrained A model supplies only the common components.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import re
import random

import numpy as np
import torch

from utils.experiment_config import ExperimentConfig

INIT_PROTOCOL = "local-global-init-v1"
SELECTION_METRICS = ("balanced_oracle_mae", "balanced_blind_mae")


def validate_local_global_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route == "none":
        if cfg.local_layout is not None or cfg.local_depth is not None or cfg.selection_metric is not None:
            raise ValueError("local layout/depth require --local_route A or B+")
        return
    if cfg.local_route not in ("A", "B+") or cfg.local_layout not in ("staged", "alternating"):
        raise ValueError("explicit A/B+ route and staged/alternating layout required")
    if isinstance(cfg.local_depth, bool) or not isinstance(cfg.local_depth, int) or cfg.local_depth < 1:
        raise ValueError("--local_depth must be a positive integer")
    if cfg.model_name != "M1" or cfg.atom_feat != "z_only_proj":
        raise ValueError("current local/global comparison requires M1 and ZP (z_only_proj)")
    if cfg.norm != "sumnorm" or cfg.scale_mode != "eta" or not cfg.skip_test_eval:
        raise ValueError("local/global trials require SumNorm/H1 and valid-only selection")
    if cfg.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"choose --selection_metric explicitly from {SELECTION_METRICS}")
    conflicts = {
        "use_g1": cfg.use_g1, "use_g2": cfg.use_g2,
        "use_periodic_manybody": cfg.use_periodic_manybody,
        "use_macro_lattice": cfg.use_macro_lattice, "use_amp": cfg.use_amp,
        "init_ckpt": bool(cfg.init_ckpt), "freeze_backbone": cfg.freeze_backbone,
        "reset_rng_after_init": cfg.reset_rng_after_init,
        "pair_aux": cfg.pair_aux_arm != "none" or cfg.pair_ratio != 0,
        "slope_aux": cfg.edos_slope_ratio != 0, "scalar_aux": cfg.scalar_mode != "none",
        "c5_moe": cfg.c5_moe, "atom_additive": cfg.use_atom_additive_phdos,
        "r1a_point": cfg.r1a_point, "r1b_coord": cfg.r1b_coord,
    }
    enabled = sorted(key for key, value in conflicts.items() if value)
    if enabled:
        raise ValueError(f"local/global comparison conflicts with {enabled}")
    if not cfg.tag or cfg.epochs < 1 or cfg.batch_size < 1 or not np.isfinite(cfg.lr) or cfg.lr <= 0:
        raise ValueError("local/global trials need an isolated tag and positive training configuration")


def comparison_configs(base: ExperimentConfig, *, depth: int, seeds: list[int],
                       tag_prefix: str) -> list[ExperimentConfig]:
    """Four equal-count arms per seed, with explicit budget supplied by caller."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
        raise ValueError("four-arm comparison needs depth >= 2; depth 1 layouts coincide")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("nonnegative, distinct seeds required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    configs = []
    for seed in seeds:
        for route, label in (("A", "a"), ("B+", "bp")):
            for layout in ("staged", "alternating"):
                cfg = replace(base, seed=seed, local_route=route, local_layout=layout,
                              local_depth=depth,
                              tag=f"_lg_{tag_prefix}_k{depth}_{label}_{layout}_s{seed}")
                validate_local_global_config(cfg)
                configs.append(cfg)
    return configs


def _is_local_parameter(key: str) -> bool:
    return key.startswith("encoder.local_layers.") and ".message." in key


def initialize_comparison_model(transformer, params: dict, seed: int) -> dict:
    """Match all common weights, keep route-specific local weights independent.

    Build the selected route in canonical staged order under a scoped CPU RNG.
    B+ copies ZP/global/readout/external norms from a same-seed untrained A
    reference. References never share Parameters with the trainable target.
    Constructor behavior (including zero biases/trunks) remains authoritative.
    """
    from model.transformer import Transformer
    from utils.zproj_alignment import state_hash

    if params.get("local_route") not in ("A", "B+"):
        raise ValueError("matched initialization requires an A/B+ model")
    selected = {**params, "local_layout": "staged"}
    with torch.random.fork_rng(devices=[]):
        # Seed CPU only; reference construction must not change CUDA RNG state.
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**selected)
        if selected["local_route"] == "B+":
            torch.random.default_generator.manual_seed(seed)
            canonical = Transformer(**{**selected, "local_route": "A"})
            source = canonical.state_dict()
            state = reference.state_dict()
            common = {key: source[key] for key in state if not _is_local_parameter(key)}
            reference.load_state_dict({**state, **common}, strict=True)
            del canonical
        state = reference.state_dict()
        expected = transformer.state_dict()
        if expected.keys() != state.keys() or any(expected[k].shape != state[k].shape for k in state):
            raise ValueError("initialization parameters do not describe the target model")
        transformer.load_state_dict(state, strict=True)
        common_state = {key: value for key, value in state.items() if not _is_local_parameter(key)}
        result = {"protocol": INIT_PROTOCOL, "seed": seed,
                  "common_state_sha256": state_hash(common_state),
                  "state_sha256": state_hash(state),
                  "update_counts": transformer.encoder.update_counts}
    return result


def comparison_contract(cfg: ExperimentConfig, transformer_params: dict) -> dict | None:
    if cfg.local_route == "none":
        return None
    return {"schema": "local-global-model-v1", "init_protocol": INIT_PROTOCOL,
            "seed": cfg.seed, "transformer": deepcopy(transformer_params),
            "normalization": cfg.norm, "selection_metric": cfg.selection_metric,
            "training": {key: value for key, value in asdict(cfg).items()
                         if key not in ('tag', 'skip_existing')}}


def capture_training_rng() -> dict:
    """Checkpoint-safe Python/NumPy/Torch streams (no NumPy pickle objects)."""
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_training_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("comparison resume requires matching CUDA RNG devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
