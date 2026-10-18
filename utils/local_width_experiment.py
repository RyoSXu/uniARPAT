"""Frozen full-A capacity package with common nonlocal initialization."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace

import torch

from utils.checkpoint_selection import BLIND_R2_METRIC, selection_contract
from utils.experiment_config import ExperimentConfig
from utils.i1_a_experiment import initialize_i1_a_model, validate_i1_a_config, validate_i3_model

BASE_WIDTHS = (64, 128)
WIDE_WIDTHS = (128, 256)
WIDTH_INIT = "local-a-width-init-v1"
WIDTH_SCHEMA = "local-a-width-model-v1"


def validate_local_width_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route != "W1" or cfg.selection_metric != BLIND_R2_METRIC:
        raise ValueError("W1 requires the fixed full-A width candidate and joint blind R2 selection")
    validate_i1_a_config(replace(cfg, local_route="I3", selection_metric="balanced_oracle_mae"))


def local_width_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if reference.local_route != "I3" or reference.selection_metric != BLIND_R2_METRIC:
        raise ValueError("width comparison requires the completed joint-blind full-A recipe")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    cfg = replace(reference, local_route="W1", skip_existing=False,
                  tag=f"_law_{tag_prefix}_w128_h256_k2_g6_staged_s42_e35")
    validate_local_width_config(cfg)
    return cfg


def narrow_reference_parameters(params: dict) -> dict:
    if (params.get("local_route") != "W1"
            or (params.get("local_scalar_channels"), params.get("local_hidden_channels")) != WIDE_WIDTHS):
        raise ValueError("W1 must explicitly record scalar128 / hidden256")
    reference = {**params, "local_route": "I3"}
    del reference["local_scalar_channels"], reference["local_hidden_channels"]
    return reference


def validate_width_model(transformer, params: dict, *, require_identity=False) -> None:
    reference = narrow_reference_parameters(params)
    encoder = transformer.encoder
    if transformer.local_route != "W1" or encoder.route != "W1":
        raise ValueError("the full-A capacity candidate must use W1")
    proxy = SimpleNamespace(route="I3", **{name: getattr(encoder, name) for name in
        ("geometry_route", "depth", "global_depth", "layout", "local_layers", "global_layers")})
    validate_i3_model(SimpleNamespace(local_route="I3", encoder=proxy,
                                     named_parameters=transformer.named_parameters),
                      reference, require_identity=require_identity)
    if hasattr(encoder, "local_readout_gate"):
        raise ValueError("width comparison does not include the local-readout gate")
    for layer in encoder.local_layers:
        message = layer.message
        if ((message.scalar_channels, message.hidden_channels) != WIDE_WIDTHS
                or message.local_input.out_features != 128 or message.local_output.in_features != 128
                or any((block.scalar_channels, block.hidden_channels) != WIDE_WIDTHS
                       for block in message.blocks)):
            raise ValueError("W1 must widen the complete local message to 128/256")


def width_parameter_counts(transformer) -> dict:
    from utils.local_global_experiment import _is_local_parameter

    total = sum(value.numel() for value in transformer.parameters())
    local = sum(value.numel() for name, value in transformer.named_parameters() if _is_local_parameter(name))
    return {"total": total, "local_message": local, "matched_nonlocal": total - local}


def initialize_local_width_model(transformer, params: dict, seed: int) -> dict:
    """Copy only nonlocal tensors; initialize all widened local tensors fresh."""
    from model.transformer import Transformer
    from utils.local_global_experiment import _is_local_parameter
    from utils.zproj_alignment import state_hash

    reference_params = narrow_reference_parameters(params)
    validate_width_model(transformer, params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_i1_a_model(reference, reference_params, seed)
        torch.random.default_generator.manual_seed(seed)
        canonical = Transformer(**params)
    source, local_source, target = reference.state_dict(), canonical.state_dict(), transformer.state_dict()
    if (source.keys() != target.keys() or local_source.keys() != target.keys()
            or any(local_source[key].shape != target[key].shape for key in target)):
        raise ValueError("the width candidate must retain A's parameter keys")
    common = {key: value for key, value in source.items() if not _is_local_parameter(key)}
    if any(target[key].shape != value.shape for key, value in common.items()):
        raise ValueError("the width candidate changed nonlocal tensor shapes")
    local = {key: value for key, value in local_source.items() if _is_local_parameter(key)}
    transformer.load_state_dict({**common, **local}, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key]))
           or current[key].data_ptr() == value.data_ptr() for key, value in {**common, **local}.items()):
        raise ValueError("width initialization must copy canonical tensors without storage aliases")
    validate_width_model(transformer, params, require_identity=True)
    return {"protocol": WIDTH_INIT, "seed": seed, "route": "W1", "widths": list(WIDE_WIDTHS),
            "narrow_initialization": reference_info,
            "state_sha256": state_hash(current), "matched_nonlocal_state_sha256": state_hash(common),
            "local_message_state_sha256": state_hash(local), "n_common_tensors": len(common),
            "n_tensors": len(current), "parameter_counts": width_parameter_counts(transformer),
            "reference_parameter_counts": width_parameter_counts(reference),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def local_width_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_local_width_config(cfg)
    narrow_reference_parameters(params)
    return {"schema": WIDTH_SCHEMA, "init_protocol": WIDTH_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, **selection_contract(cfg.selection_metric),
            "training": training_record(cfg), "ablation": {
                "reference": "A64", "secondary_reference": "I1", "local_content": "A",
                "reference_widths": list(BASE_WIDTHS), "candidate_widths": list(WIDE_WIDTHS),
                "local_depth": 2, "internal_blocks_per_call": 2, "global_depth": 6, "layout": "staged",
                "common_initial_tensors": "all tensors outside encoder.local_layers.*.message.*",
                "all_local_message_tensors_fresh": True, "joint_training": True,
                "scope": "combined scalar/MLP capacity package; no separate width attribution"}}
