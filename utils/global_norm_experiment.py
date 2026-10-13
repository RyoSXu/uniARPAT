"""N1 changes only C1's global normalization placement, from pre to post."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

from utils.experiment_config import ExperimentConfig
from utils.local_structure_experiment import initialize_structure_control, validate_structure_control_config

NORM_INIT = "global-norm-init-v1"
NORM_SCHEMA = "global-norm-model-v1"


def validate_global_norm_config(cfg: ExperimentConfig) -> None:
    if (cfg.local_route != "N1" or isinstance(cfg.local_depth, bool)
            or not isinstance(cfg.local_depth, int) or cfg.local_depth != 2
            or isinstance(cfg.global_depth, bool) or not isinstance(cfg.global_depth, int)
            or cfg.global_depth != 6 or cfg.local_layout != "staged" or not cfg.tag):
        raise ValueError("N1 requires an isolated tag, integer L2/G6 and staged layout")
    validate_structure_control_config(replace(cfg, local_route="C1"))


def global_norm_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    cfg = replace(reference, local_route="N1", skip_existing=False,
                  tag=f"_gn_{tag_prefix}_n1_k2_g6_staged_s42_e35")
    validate_global_norm_config(cfg)
    return cfg


def c1_reference_parameters(params: dict) -> dict:
    if (params.get("local_route") != "N1" or params.get("local_depth") != 2
            or params.get("global_depth") != 6 or params.get("local_layout") != "staged"
            or params.get("normalize_before", False)):
        raise ValueError("N1 changes global normalization only, with the original decoder unchanged")
    return {**params, "local_route": "C1"}


def initialize_global_norm_model(transformer, params: dict, seed: int) -> dict:
    """Use the complete untrained C1 state, including all biases and local tensors."""
    from utils.zproj_alignment import state_hash

    reference_params = c1_reference_parameters(params)
    if any(layer.norm_first for layer in transformer.encoder.global_layers):
        raise ValueError("N1 must have post-norm in every global layer")
    # C1 and N1 own the same parameter keys/shapes. Reuse its exact initializer;
    # do not seed a separately parameterized constructor or read trained weights.
    reference_info = initialize_structure_control(transformer, reference_params, seed)
    state = transformer.state_dict()
    if state_hash(state) != reference_info["state_sha256"]:
        raise ValueError("N1 did not take the complete untrained C1 state")
    return {"protocol": NORM_INIT, "seed": seed, "route": "N1",
            "c1_initialization": reference_info, "state_sha256": state_hash(state),
            "n_tensors": len(state), "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def global_norm_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_global_norm_config(cfg)
    c1_reference_parameters(params)
    return {"schema": NORM_SCHEMA, "init_protocol": NORM_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, "training": training_record(cfg),
            "ablation": {"reference": "C1", "changed": "global pre-norm to post-norm",
                         "global_depth": 6, "local_depth": 2, "local_content": "C1",
                         "learned_qkvo": True, "attention_probability_dropout": params["dropout"],
                         "geometry_scoring": False, "all_initial_tensors_identical": True}}
