"""Frozen joint-blind I1/I2/I4 comparison of the local-to-readout interface."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from utils.checkpoint_selection import BLIND_R2_METRIC, selection_contract
from utils.experiment_config import ExperimentConfig
from utils.i1_bplus_experiment import (
    initialize_i1_bplus_model, validate_i1_bplus_config, validate_i2_model,
)
from utils.local_structure_experiment import parameter_counts

READOUT_INIT = "local-readout-init-v1"
READOUT_SCHEMA = "local-readout-model-v1"
READOUT_ROUTES = ("I1", "I2", "I4")
GATE_KEY = "encoder.local_readout_gate"


def validate_local_readout_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "I4" or cfg.selection_metric != BLIND_R2_METRIC:
        raise ValueError("I4 requires the joint blind R2 checkpoint selection protocol")
    validate_i1_bplus_config(replace(cfg, local_route="I2", selection_metric="balanced_oracle_mae"))


def local_readout_configs(reference: ExperimentConfig, tag_prefix: str) -> list[ExperimentConfig]:
    from utils.local_global_experiment import validate_local_global_config

    if reference.local_route != "I1" or not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("local readout requires the I1 recipe and an isolated alphanumeric tag prefix")
    controls = [replace(reference, local_route=route, selection_metric=BLIND_R2_METRIC,
                        skip_existing=False, tag=f"_lri_{tag_prefix}_{route.lower()}_k2_g6_staged_s42_e35")
                for route in READOUT_ROUTES]
    for cfg in controls:
        validate_local_global_config(cfg)
    return controls


def i2_reference_parameters(params: dict) -> dict:
    if (params.get("local_route") != "I4" or params.get("local_layout") != "staged"
            or params.get("local_depth") != 2 or params.get("global_depth") != 6):
        raise ValueError("local readout initialization requires I4 staged L2/G6")
    return {**params, "local_route": "I2"}


def validate_i4_model(transformer, params: dict, *, require_zero=False, require_identity=False) -> None:
    reference = i2_reference_parameters(params)
    encoder = transformer.encoder
    if transformer.local_route != "I4" or encoder.route != "I4":
        raise ValueError("the local readout candidate must use I4")
    # Reuse I2's complete-content and global-layer checks without changing any
    # attributes on the real module or relaxing historical I2 validation.
    proxy = SimpleNamespace(route="I2", **{name: getattr(encoder, name) for name in
        ("geometry_route", "depth", "global_depth", "layout", "local_layers", "global_layers")})
    validate_i2_model(SimpleNamespace(local_route="I2", encoder=proxy,
                                      named_parameters=transformer.named_parameters),
                      reference, require_identity=require_identity)
    gate = getattr(encoder, "local_readout_gate", None)
    if (not isinstance(gate, torch.nn.Parameter) or gate.ndim != 0
            or not gate.requires_grad or not torch.isfinite(gate)):
        raise ValueError("I4 requires one finite trainable scalar readout gate")
    if require_zero and gate.detach().item() != 0.:
        raise ValueError("I4 must start with a zero readout gate")


def initialize_local_readout_model(transformer, params: dict, seed: int) -> dict:
    from model.transformer import Transformer
    from utils.zproj_alignment import state_hash

    reference_params = i2_reference_parameters(params)
    validate_i4_model(transformer, params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_i1_bplus_model(reference, reference_params, seed)
    source = reference.state_dict()
    target = transformer.state_dict()
    if (set(target) != set(source) | {GATE_KEY}
            or any(target[key].shape != value.shape for key, value in source.items())):
        raise ValueError("I4 may add only its scalar gate to the complete I2 tensor layout")
    transformer.load_state_dict({**source, GATE_KEY: torch.zeros_like(target[GATE_KEY])}, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in source.items()):
        raise ValueError("I4 common initial values differ from untrained I2")
    validate_i4_model(transformer, params, require_zero=True, require_identity=True)
    common = {key: current[key] for key in source}
    return {"protocol": READOUT_INIT, "seed": seed, "route": "I4",
            "bplus_initialization": reference_info, "common_state_sha256": state_hash(common),
            "state_sha256": state_hash(current), "n_common_tensors": len(common),
            "n_tensors": len(current), "gate_initial_value": 0.,
            "parameter_counts": parameter_counts(transformer),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def local_readout_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_local_readout_config(cfg)
    i2_reference_parameters(params)
    return {"schema": READOUT_SCHEMA, "init_protocol": READOUT_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, **selection_contract(cfg.selection_metric),
            "training": training_record(cfg), "ablation": {
                "reference": "I2", "local_content": "B+", "local_depth": 2, "global_depth": 6,
                "memory": "h_global + gate * (h_local - h_ZP)",
                "gate": {"parameter": GATE_KEY, "initial_value": 0., "shared_by_tasks": True},
                "consumers": ["shared_decoder_edos", "shared_decoder_phdos", "H1_eta_gamma"],
                "extra_normalization": False, "common_initial_tensors_identical": True,
                "scope": "forward and gradient-path intervention; not a proof of information loss"}}
