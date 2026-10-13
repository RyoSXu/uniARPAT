"""D1 disables only N1's global attention-probability dropout."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

from utils.experiment_config import ExperimentConfig
from utils.global_norm_experiment import (
    c1_reference_parameters, initialize_global_norm_model, validate_global_norm_config,
)

DROPOUT_INIT = "attention-dropout-init-v1"
DROPOUT_SCHEMA = "attention-dropout-model-v1"


def validate_attention_dropout_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "D1":
        raise ValueError("attention dropout control requires D1")
    validate_global_norm_config(replace(cfg, local_route="N1"))


def attention_dropout_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    cfg = replace(reference, local_route="D1", skip_existing=False,
                  tag=f"_ad_{tag_prefix}_d1_k2_g6_staged_s42_e35")
    validate_attention_dropout_config(cfg)
    return cfg


def n1_reference_parameters(params: dict) -> dict:
    if params.get("local_route") != "D1":
        raise ValueError("D1 initialization requires D1 parameters")
    reference = {**params, "local_route": "N1"}
    c1_reference_parameters(reference)
    return reference


def initialize_attention_dropout_model(transformer, params: dict, seed: int) -> dict:
    """Copy every untrained N1 tensor; dropout is behavior, not a weight."""
    reference = n1_reference_parameters(params)
    layers = transformer.encoder.global_layers
    if len(layers) != 6 or any(
        layer.norm_first or layer.attn.dropout != 0.0
        or layer.dropout_attn.p != params["dropout"]
        or layer.dropout_ffn.p != params["dropout"] or layer.ffn[2].p != params["dropout"]
        for layer in layers
    ):
        raise ValueError("D1 must change only attention-probability dropout in the six post-norm layers")
    n1_info = initialize_global_norm_model(transformer, reference, seed)
    return {"protocol": DROPOUT_INIT, "seed": seed, "route": "D1",
            "n1_initialization": n1_info, "state_sha256": n1_info["state_sha256"],
            "n_tensors": n1_info["n_tensors"], "parameter_count": n1_info["parameter_count"],
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def attention_dropout_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_attention_dropout_config(cfg)
    n1_reference_parameters(params)
    return {"schema": DROPOUT_SCHEMA, "init_protocol": DROPOUT_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, "training": training_record(cfg),
            "ablation": {"reference": "N1", "changed": "global attention-probability dropout to zero",
                         "global_depth": 6, "local_depth": 2, "local_content": "C1",
                         "global_normalization": "post-norm", "learned_qkvo": True,
                         "attention_probability_dropout": 0.0,
                         "reference_attention_probability_dropout": params["dropout"],
                         "residual_and_ffn_dropout": params["dropout"],
                         "geometry_scoring": False, "all_initial_tensors_identical": True}}
