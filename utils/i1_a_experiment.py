"""I3 compares full A with completed I2/B+ at the corrected joint backbone."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from utils.attention_projection_experiment import validate_projection_model
from utils.experiment_config import ExperimentConfig
from utils.i1_bplus_experiment import (
    i1_reference_parameters, initialize_i1_bplus_model, validate_i1_bplus_config,
)
from utils.local_structure_experiment import is_shared_control_parameter, parameter_counts

A_INIT = "i1-a-init-v1"
A_SCHEMA = "i1-a-model-v1"
DEFAULT_I2_REFERENCE = "results/runs/ablation_m1_ibp_enc20261014_i2_k2_g6_staged_s42_e35"
I2_EVIDENCE = "results/runs/i1_bplus_encoder_20261014"


def validate_i1_a_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "I3":
        raise ValueError("the corrected joint-backbone A candidate requires I3")
    validate_i1_bplus_config(replace(cfg, local_route="I2"))


def i1_a_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    cfg = replace(reference, local_route="I3", skip_existing=False,
                  tag=f"_ia_{tag_prefix}_i3_k2_g6_staged_s42_e35")
    validate_i1_a_config(cfg)
    return cfg


def i2_reference_parameters(params: dict) -> dict:
    if params.get("local_route") != "I3":
        raise ValueError("A refinement initialization requires I3 parameters")
    reference = {**params, "local_route": "I2"}
    i1_reference_parameters(reference)
    return reference


def validate_i3_model(transformer, params: dict, *, require_identity=False) -> None:
    from model.local_global_encoder import LOCAL_PAIR_CHUNK

    reference = i1_reference_parameters(i2_reference_parameters(params))
    encoder = transformer.encoder
    if (transformer.local_route != "I3" or encoder.route != "I3" or encoder.geometry_route != "A"
            or encoder.depth != 2 or encoder.global_depth != 6 or encoder.layout != "staged"):
        raise ValueError("I3 must use full A with staged L2/G6")
    validate_projection_model(SimpleNamespace(local_route="I1", encoder=encoder,
        named_parameters=transformer.named_parameters), reference, require_identity=require_identity)
    if any(layer.message.route != "A" or len(layer.message.blocks) != 2
           for layer in encoder.local_layers):
        raise ValueError("I3 requires full two-block A messages")
    if any(block.recompute_angles or block.pair_chunk != LOCAL_PAIR_CHUNK
           for layer in encoder.local_layers for block in layer.message.blocks):
        raise ValueError("I3 must preserve the approved no-recompute A execution settings")


def initialize_i1_a_model(transformer, params: dict, seed: int) -> dict:
    """Use I2 common tensors and canonical untrained A-specific content."""
    from model.transformer import Transformer
    from utils.local_global_experiment import deep_reference_state
    from utils.zproj_alignment import state_hash

    reference_params = i2_reference_parameters(params)
    validate_i3_model(transformer, params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_i1_bplus_model(reference, reference_params, seed)
    reference_state = reference.state_dict()
    a_state, _ = deep_reference_state(seed, {**params, "local_route": "A"})
    target = transformer.state_dict()
    if target.keys() != a_state.keys() or any(target[k].shape != a_state[k].shape for k in target):
        raise ValueError("I3 must retain the complete A tensor layout")
    shared_keys = {key for key in target if is_shared_control_parameter(key)}
    if (shared_keys != {key for key in reference_state if is_shared_control_parameter(key)}
            or any(target[key].shape != reference_state[key].shape for key in shared_keys)):
        raise ValueError("I3 and I2 common tensor layouts differ")
    shared = {key: reference_state[key] for key in shared_keys}
    transformer.load_state_dict({**a_state, **shared}, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in shared.items()):
        raise ValueError("I3 common initial values do not match I2")
    validate_i3_model(transformer, params, require_identity=True)
    return {"protocol": A_INIT, "seed": seed, "route": "I3",
            "i2_initialization": reference_info, "state_sha256": state_hash(current),
            "shared_control_state_sha256": state_hash(shared), "n_shared_tensors": len(shared),
            "a_content_state_sha256": state_hash({key: value for key, value in current.items()
                                                 if key not in shared_keys}),
            "n_tensors": len(current), "parameter_counts": parameter_counts(transformer),
            "reference_parameter_counts": parameter_counts(reference),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def i1_a_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_i1_a_config(cfg)
    i2_reference_parameters(params)
    return {"schema": A_SCHEMA, "init_protocol": A_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, "training": training_record(cfg),
            "ablation": {"reference": "I2", "secondary_reference": "I1", "local_content": "A",
                         "global_depth": 6, "local_depth": 2, "global_normalization": "post-norm",
                         "attention_probability_dropout": 0.0, "geometry_scoring": False,
                         "projection_initialization": "Q=K=V=O=identity; zero biases",
                         "projection_trainable": True, "common_initial_tensors_identical": True,
                         "joint_training": True,
                         "scope": "complete A versus B+ local packages, including capacity; staged only"}}
