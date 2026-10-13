"""I1/I0 separate identity projection initialization from its trainability."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

import torch

from utils.attention_dropout_experiment import (
    initialize_attention_dropout_model, n1_reference_parameters, validate_attention_dropout_config,
)
from utils.experiment_config import ExperimentConfig

PROJECTION_ROUTES = ("I1", "I0")
PROJECTION_INIT = "attention-projection-init-v1"
PROJECTION_SCHEMA = "attention-projection-model-v1"


def validate_attention_projection_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route not in PROJECTION_ROUTES:
        raise ValueError("identity projection controls require I1/I0")
    validate_attention_dropout_config(replace(cfg, local_route="D1"))


def attention_projection_configs(reference: ExperimentConfig, tag_prefix: str) -> list[ExperimentConfig]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    configs = [replace(reference, local_route=route, skip_existing=False,
                       tag=f"_ap_{tag_prefix}_{route.lower()}_k2_g6_staged_s42_e35")
               for route in PROJECTION_ROUTES]
    for cfg in configs:
        validate_attention_projection_config(cfg)
    return configs


def d1_reference_parameters(params: dict) -> dict:
    if params.get("local_route") not in PROJECTION_ROUTES:
        raise ValueError("identity projection initialization requires I1/I0 parameters")
    reference = {**params, "local_route": "D1"}
    n1_reference_parameters(reference)
    return reference


def is_projection_parameter(name: str) -> bool:
    return re.fullmatch(r"encoder\.global_layers\.\d+\.attn\."
                        r"(in_proj_(weight|bias)|out_proj\.(weight|bias))", name) is not None


def set_identity_projections(transformer) -> None:
    """Make Q=K=V=input and the attention output projection an identity."""
    with torch.no_grad():
        for layer in transformer.encoder.global_layers:
            attn = layer.attn
            eye = torch.eye(attn.embed_dim, device=attn.in_proj_weight.device,
                            dtype=attn.in_proj_weight.dtype)
            attn.in_proj_weight.copy_(eye.repeat(3, 1))
            attn.in_proj_bias.zero_()
            attn.out_proj.weight.copy_(eye)
            attn.out_proj.bias.zero_()


def validate_projection_model(transformer, params: dict, *, require_identity: bool = False) -> None:
    route = params["local_route"]
    d1_reference_parameters(params)
    if transformer.local_route != route or len(transformer.encoder.global_layers) != 6:
        raise ValueError("projection target must own the configured I1/I0 route and six global layers")
    for layer in transformer.encoder.global_layers:
        if (layer.norm_first or layer.attn.dropout != 0.0
                or layer.dropout_attn.p != params["dropout"]
                or layer.dropout_ffn.p != params["dropout"] or layer.ffn[2].p != params["dropout"]):
            raise ValueError("identity projection controls must preserve D1 normalization and dropout")
        if require_identity:
            attn = layer.attn
            eye = torch.eye(attn.embed_dim, device=attn.in_proj_weight.device,
                            dtype=attn.in_proj_weight.dtype)
            if (not torch.equal(attn.in_proj_weight, eye.repeat(3, 1))
                    or not torch.equal(attn.out_proj.weight, eye)
                    or torch.count_nonzero(attn.in_proj_bias) or torch.count_nonzero(attn.out_proj.bias)):
                raise ValueError("fixed/initial identity projection tensors changed")
    frozen = {name for name, p in transformer.named_parameters() if not p.requires_grad}
    expected = ({name for name, _ in transformer.named_parameters() if is_projection_parameter(name)}
                if route == "I0" else set())
    if frozen != expected:
        raise ValueError("I0 freezes only QKVO; I1 must retain every trainable D1 parameter")


def initialize_attention_projection_model(transformer, params: dict, seed: int) -> dict:
    from model.transformer import Transformer
    from utils.zproj_alignment import state_hash

    reference_params = d1_reference_parameters(params)
    validate_projection_model(transformer, params)
    # Rebuild a trainable D1 reference even for I0: its historical initialization
    # metadata counts trainable parameters and must remain exactly reproducible.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_attention_dropout_model(reference, reference_params, seed)
    source = reference.state_dict()
    expected = transformer.state_dict()
    if expected.keys() != source.keys() or any(expected[k].shape != source[k].shape for k in expected):
        raise ValueError("I1/I0 must retain all D1 tensor keys and shapes")
    transformer.load_state_dict(source, strict=True)
    set_identity_projections(transformer)
    validate_projection_model(transformer, params, require_identity=True)
    current = transformer.state_dict()
    retained = {key: value for key, value in source.items() if not is_projection_parameter(key)}
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in retained.items()):
        raise ValueError("non-projection D1 initial tensors changed")
    projections = {key: value for key, value in current.items() if is_projection_parameter(key)}
    return {"protocol": PROJECTION_INIT, "seed": seed, "route": params["local_route"],
            "d1_initialization": reference_info, "state_sha256": state_hash(current),
            "retained_state_sha256": state_hash(retained),
            "projection_state_sha256": state_hash(projections),
            "n_tensors": len(current), "n_retained_tensors": len(retained),
            "changed_initial_tensors": [key for key in current
                                        if not torch.equal(current[key], source[key].to(current[key]))],
            "frozen_parameter_names": [name for name, p in transformer.named_parameters() if not p.requires_grad],
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in transformer.parameters() if p.requires_grad),
            "projection_parameter_count": sum(value.numel() for value in projections.values()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def attention_projection_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_attention_projection_config(cfg)
    d1_reference_parameters(params)
    return {"schema": PROJECTION_SCHEMA, "init_protocol": PROJECTION_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, "training": training_record(cfg),
            "ablation": {"reference": "D1", "local_content": "C1", "global_depth": 6, "local_depth": 2,
                         "global_normalization": "post-norm", "attention_probability_dropout": 0.0,
                         "residual_and_ffn_dropout": params["dropout"], "geometry_scoring": False,
                         "projection_initialization": "Q=K=V=O=identity; zero biases",
                         "projection_trainable": cfg.local_route == "I1",
                         "non_projection_initial_tensors_identical": True,
                         "all_initial_tensors_identical_between_I0_I1": True,
                         "scope": "I1-D1: projection initialization; I0-I1: projection trainability "
                                  "and effective trainable capacity at identity initialization"}}
