"""I2 replaces I1's radial messages with full B+ on the corrected backbone."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

import torch

from utils.attention_projection_experiment import (
    d1_reference_parameters, initialize_attention_projection_model, validate_attention_projection_config,
    validate_projection_model,
)
from utils.experiment_config import ExperimentConfig
from utils.local_structure_experiment import is_shared_control_parameter, parameter_counts

BPLUS_INIT = "i1-bplus-init-v1"
BPLUS_SCHEMA = "i1-bplus-model-v1"
DEFAULT_I1_REFERENCE = "results/runs/ablation_m1_ap_proj20261013_i1_k2_g6_staged_s42_e35"
I1_PREFLIGHT = "results/runs/attention_projection_attribution_20261013/preflight_final.json"


def validate_i1_bplus_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "I2":
        raise ValueError("the corrected-backbone B+ candidate requires I2")
    validate_attention_projection_config(replace(cfg, local_route="I1"))


def i1_bplus_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    cfg = replace(reference, local_route="I2", skip_existing=False,
                  tag=f"_ibp_{tag_prefix}_i2_k2_g6_staged_s42_e35")
    validate_i1_bplus_config(cfg)
    return cfg


def i1_reference_parameters(params: dict) -> dict:
    if params.get("local_route") != "I2":
        raise ValueError("B+ refinement initialization requires I2 parameters")
    reference = {**params, "local_route": "I1"}
    d1_reference_parameters(reference)
    return reference


def validate_i2_model(transformer, params: dict, *, require_identity=False) -> None:
    reference = i1_reference_parameters(params)
    # Validate the real module without changing its route, parameter keys or
    # trainability. I1's projection checks apply to the same six scalar layers.
    if (transformer.local_route != "I2" or transformer.encoder.route != "I2"
            or transformer.encoder.geometry_route != "B+"):
        raise ValueError("I2 must use full B+ physical geometry")
    if (transformer.encoder.depth != 2 or transformer.encoder.global_depth != 6
            or transformer.encoder.layout != "staged"):
        raise ValueError("I2 requires staged L2/G6")
    from types import SimpleNamespace
    validate_projection_model(SimpleNamespace(
        local_route="I1", encoder=transformer.encoder,
        named_parameters=transformer.named_parameters), reference, require_identity=require_identity)
    if any(layer.message.route != "B+" or not layer.message.include_high_order
           or len(layer.message.blocks) != 2 for layer in transformer.encoder.local_layers):
        raise ValueError("I2 local messages must use full two-block B+")


def initialize_i1_bplus_model(transformer, params: dict, seed: int) -> dict:
    """Copy I1 common values and the independently initialized B+ content.

    No trained weights are loaded. Equivalent local projections and block LNs
    are shared as well as ZP, global, decoder and heads. B+ content takes the
    established untrained B+ reference values. Scoped CPU RNG leaves the
    caller's CPU/CUDA streams intact.
    """
    from model.transformer import Transformer
    from utils.local_global_experiment import deep_reference_state
    from utils.zproj_alignment import state_hash

    reference_params = i1_reference_parameters(params)
    validate_i2_model(transformer, params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_attention_projection_model(reference, reference_params, seed)
    i1_state = reference.state_dict()
    bplus_state, _ = deep_reference_state(seed, {**params, "local_route": "B+"})
    target = transformer.state_dict()
    if target.keys() != bplus_state.keys() or any(target[k].shape != bplus_state[k].shape for k in target):
        raise ValueError("I2 must retain the full B+ tensor layout")
    shared_keys = {key for key in target if is_shared_control_parameter(key)}
    if (shared_keys != {key for key in i1_state if is_shared_control_parameter(key)}
            or any(target[key].shape != i1_state[key].shape for key in shared_keys)):
        raise ValueError("I2 and I1 common tensor layouts differ")
    shared = {key: i1_state[key] for key in shared_keys}
    transformer.load_state_dict({**bplus_state, **shared}, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in shared.items()):
        raise ValueError("I2 common initial values do not match I1")
    validate_i2_model(transformer, params, require_identity=True)
    content = {key: value for key, value in current.items() if key not in shared_keys}
    return {"protocol": BPLUS_INIT, "seed": seed, "route": "I2",
            "i1_initialization": reference_info, "state_sha256": state_hash(current),
            "shared_control_state_sha256": state_hash(shared),
            "bplus_content_state_sha256": state_hash(content),
            "n_shared_tensors": len(shared), "n_tensors": len(current),
            "parameter_counts": parameter_counts(transformer),
            "reference_parameter_counts": parameter_counts(reference),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def i1_bplus_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_i1_bplus_config(cfg)
    i1_reference_parameters(params)
    return {"schema": BPLUS_SCHEMA, "init_protocol": BPLUS_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, "training": training_record(cfg),
            "ablation": {"reference": "I1", "local_content": "B+", "global_depth": 6,
                         "local_depth": 2, "global_normalization": "post-norm",
                         "attention_probability_dropout": 0.0, "geometry_scoring": False,
                         "projection_initialization": "Q=K=V=O=identity; zero biases",
                         "projection_trainable": True, "common_initial_tensors_identical": True,
                         "scope": "full local-content package, including capacity; not angle-only"}}
