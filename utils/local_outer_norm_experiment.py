"""Complete A: outer post-norm versus A64's unchanged outer pre-norm."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from utils.checkpoint_selection import BLIND_R2_METRIC, selection_contract
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import initialize_i1_a_model, validate_i1_a_config, validate_i3_model

OUTER_NORM_INIT = "local-outer-norm-init-v1"
OUTER_NORM_SCHEMA = "local-outer-norm-model-v1"


def validate_outer_norm_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "LN2" or cfg.selection_metric != BLIND_R2_METRIC:
        raise ValueError("LN2 requires complete A outer post-norm and joint blind R2 selection")
    validate_i1_a_config(replace(cfg, local_route="I3", selection_metric="balanced_oracle_mae"))


def outer_norm_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if reference.local_route != "I3" or reference.selection_metric != BLIND_R2_METRIC:
        raise ValueError("outer normalization comparison requires the joint-blind A64 recipe")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("isolated alphanumeric tag prefix required")
    cfg = replace(reference, local_route="LN2", skip_existing=False,
                  tag=f"_lon_{tag_prefix}_post_k2_g6_staged_s42_e35")
    validate_outer_norm_config(cfg)
    return cfg


def pre_norm_parameters(params: dict) -> dict:
    if params.get("local_route") != "LN2":
        raise ValueError("outer post-norm initialization requires LN2 parameters")
    return {**params, "local_route": "I3"}


def validate_outer_norm_model(transformer, params: dict, *, require_identity=False) -> None:
    reference = pre_norm_parameters(params)
    encoder = transformer.encoder
    if transformer.local_route != "LN2" or encoder.route != "LN2":
        raise ValueError("outer post-norm candidate must use the isolated LN2 route")
    proxy = SimpleNamespace(route="I3", **{name: getattr(encoder, name) for name in
        ("geometry_route", "depth", "global_depth", "layout", "local_layers", "global_layers")})
    validate_i3_model(SimpleNamespace(local_route="I3", encoder=proxy,
                                     named_parameters=transformer.named_parameters),
                      reference, require_identity=require_identity)
    if hasattr(encoder, "local_readout_gate"):
        raise ValueError("outer normalization comparison does not include a readout gate")
    for layer in encoder.local_layers:
        message = layer.message
        if (layer.outer_norm_first is not False or message.readout_delta is not False or message.internal_residual is not True
                or any(block.residual_add is not True for block in message.blocks)
                or (message.scalar_channels, message.hidden_channels) != (64, 128)
                or message.norm_first is not False or any(block.norm_first is not False for block in message.blocks)):
            raise ValueError("LN2 requires two outer post-norm layers and four unchanged internal post-norm blocks")


def initialize_outer_norm_model(transformer, params: dict, seed: int) -> dict:
    """Match every untrained A64 tensor; trained checkpoints are never read here."""
    from model.transformer import Transformer
    from utils.zproj_alignment import state_hash

    reference_params = pre_norm_parameters(params)
    validate_outer_norm_model(transformer, params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_i1_a_model(reference, reference_params, seed)
    source, target = reference.state_dict(), transformer.state_dict()
    if source.keys() != target.keys() or any(source[key].shape != target[key].shape for key in source):
        raise ValueError("outer normalization must preserve every A64 tensor key and shape")
    transformer.load_state_dict(source, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key]))
           or current[key].data_ptr() == value.data_ptr() for key, value in source.items()):
        raise ValueError("LN2 must copy all A64 initial tensors without storage aliases")
    validate_outer_norm_model(transformer, params, require_identity=True)
    return {"protocol": OUTER_NORM_INIT, "seed": seed, "route": "LN2",
            "pre_norm_initialization": reference_info, "state_sha256": state_hash(current),
            "n_tensors": len(current), "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "all_initial_tensors_identical": True, "internal_normalization": "post-norm",
            "outer_normalization": "post-norm", "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def outer_norm_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_outer_norm_config(cfg)
    pre_norm_parameters(params)
    return {"schema": OUTER_NORM_SCHEMA, "init_protocol": OUTER_NORM_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, **selection_contract(cfg.selection_metric),
            "training": training_record(cfg), "ablation": {
                "reference": "A64", "local_content": "A", "widths": [64, 128],
                "internal_normalization": "post-norm", "internal_residual": "s+F(s,R)",
                "outer_normalization": "post-norm512", "reference_outer_normalization": "pre-norm512",
                "outer_formula": "LN512(h+L(h,R))", "reference_outer_formula": "h+L(LN512(h),R)",
                "local_readout": "P_out(s2)", "extra_terminal_norm": False,
                "local_depth": 2, "internal_blocks_per_call": 2, "global_depth": 6, "layout": "staged",
                "all_initial_tensors_identical": True, "joint_training": True,
                "scope": "outer normalization placement only; remaining architecture is a control, not selected"}}
