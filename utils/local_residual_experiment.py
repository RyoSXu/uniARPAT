"""Complete A: remove the explicit internal scalar additions, retaining the outer residual."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from utils.checkpoint_selection import BLIND_R2_METRIC, selection_contract
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import initialize_i1_a_model, validate_i1_a_config, validate_i3_model

RESIDUAL_INIT = "local-residual-init-v1"
RESIDUAL_SCHEMA = "local-residual-model-v1"


def validate_residual_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "LR2" or cfg.selection_metric != BLIND_R2_METRIC:
        raise ValueError("LR2 requires complete A without internal additions and joint blind R2 selection")
    validate_i1_a_config(replace(cfg, local_route="I3", selection_metric="balanced_oracle_mae"))


def residual_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if reference.local_route != "I3" or reference.selection_metric != BLIND_R2_METRIC:
        raise ValueError("internal residual comparison requires the joint-blind A64 recipe")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("isolated alphanumeric tag prefix required")
    cfg = replace(reference, local_route="LR2", skip_existing=False,
                  tag=f"_lres_{tag_prefix}_noadd_k2_g6_staged_s42_e35")
    validate_residual_config(cfg)
    return cfg


def additive_parameters(params: dict) -> dict:
    if params.get("local_route") != "LR2":
        raise ValueError("internal residual comparison initialization requires LR2 parameters")
    return {**params, "local_route": "I3"}


def validate_residual_model(transformer, params: dict, *, require_identity=False) -> None:
    reference = additive_parameters(params)
    encoder = transformer.encoder
    if transformer.local_route != "LR2" or encoder.route != "LR2":
        raise ValueError("internal residual candidate must use the isolated LR2 route")
    proxy = SimpleNamespace(route="I3", **{name: getattr(encoder, name) for name in
        ("geometry_route", "depth", "global_depth", "layout", "local_layers", "global_layers")})
    validate_i3_model(SimpleNamespace(local_route="I3", encoder=proxy,
                                     named_parameters=transformer.named_parameters),
                      reference, require_identity=require_identity)
    if hasattr(encoder, "local_readout_gate"):
        raise ValueError("internal residual comparison does not include an I4 readout gate")
    for layer in encoder.local_layers:
        message = layer.message
        if (layer.outer_norm_first is not True or message.readout_delta is not False
                or message.internal_residual is not False or any(block.residual_add is not False for block in message.blocks)
                or (message.scalar_channels, message.hidden_channels) != (64, 128)
                or message.norm_first is not False or any(block.norm_first is not False for block in message.blocks)
                or message.local_output.bias is None or not message.local_output.bias.requires_grad):
            raise ValueError("LR2 removes only four internal additions; retain norms, full-state readout and outer residual")


def initialize_residual_model(transformer, params: dict, seed: int) -> dict:
    """Copy every untrained A64 tensor without loading trained weights or consuming caller RNG."""
    from model.transformer import Transformer
    from utils.zproj_alignment import state_hash

    reference_params = additive_parameters(params)
    validate_residual_model(transformer, params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_i1_a_model(reference, reference_params, seed)
    source, target = reference.state_dict(), transformer.state_dict()
    if source.keys() != target.keys() or any(source[key].shape != target[key].shape for key in source):
        raise ValueError("internal residual comparison must preserve every A64 tensor key and shape")
    transformer.load_state_dict(source, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key]))
           or current[key].data_ptr() == value.data_ptr() for key, value in source.items()):
        raise ValueError("LR2 must copy all A64 initial tensors without storage aliases")
    validate_residual_model(transformer, params, require_identity=True)
    return {"protocol": RESIDUAL_INIT, "seed": seed, "route": "LR2",
            "additive_initialization": reference_info, "state_sha256": state_hash(current),
            "n_tensors": len(current), "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "all_initial_tensors_identical": True, "internal_residual_add": False,
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def residual_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_residual_config(cfg)
    additive_parameters(params)
    return {"schema": RESIDUAL_SCHEMA, "init_protocol": RESIDUAL_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, **selection_contract(cfg.selection_metric),
            "training": training_record(cfg), "ablation": {
                "reference": "A64", "local_content": "A", "widths": [64, 128],
                "internal_normalization": "post-norm", "outer_normalization": "pre-norm512",
                "internal_formula": "LN64(F(s,R))", "reference_internal_formula": "LN64(s+F(s,R))",
                "internal_residual_add": False, "scalar_still_enters_message_and_update": True,
                "outer_residual": "h+P_out(s2)", "local_readout": "W_out*s2+b_out",
                "local_depth": 2, "internal_blocks_per_call": 2, "global_depth": 6, "layout": "staged",
                "all_initial_tensors_identical": True, "joint_training": True,
                "scope": "four explicit internal scalar additions only; outer residual and complete A messages retained"}}
