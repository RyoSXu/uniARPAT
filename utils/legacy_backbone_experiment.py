"""O0/O1: old content attention, with the geometric scoring path removed."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

import torch

from utils.experiment_config import ExperimentConfig
from utils.local_structure_experiment import validate_structure_control_config

LEGACY_ROUTES = ("O0", "O1")
LEGACY_INIT = "legacy-backbone-init-v1"
LEGACY_SCHEMA = "legacy-backbone-model-v1"
LOCAL_PREFIX = "encoder.local_layers."


def validate_legacy_backbone_config(cfg: ExperimentConfig) -> None:
    depth = {"O0": 0, "O1": 2}.get(cfg.local_route)
    if depth is None or isinstance(cfg.local_depth, bool) \
            or not isinstance(cfg.local_depth, int) or cfg.local_depth != depth:
        raise ValueError("O0/O1 require integer local_depth=0/2 respectively")
    if isinstance(cfg.global_depth, bool) or not isinstance(cfg.global_depth, int) or cfg.global_depth != 6:
        raise ValueError("O0/O1 require integer global_depth=6")
    validate_structure_control_config(replace(cfg, local_route="C1", local_depth=2))
    if not cfg.tag:
        raise ValueError("O0/O1 require isolated experiment tags")


def legacy_backbone_configs(reference: ExperimentConfig, tag_prefix: str) -> list[ExperimentConfig]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    configs = [replace(reference, local_route=route, local_depth=depth, skip_existing=False,
                       tag=f"_oldg_{tag_prefix}_{route.lower()}_k{depth}_g6_staged_s42_e35")
               for route, depth in (("O0", 0), ("O1", 2))]
    for cfg in configs:
        validate_legacy_backbone_config(cfg)
    return configs


def baseline_parameters(params: dict) -> dict:
    """Reconstruct the original baseline constructor, before any trained weights."""
    if (params.get("local_route") not in LEGACY_ROUTES
            or params.get("local_depth") != (0 if params["local_route"] == "O0" else 2)
            or params.get("local_layout") != "staged" or params.get("global_depth") != 6
            or params.get("normalize_before", False)):
        raise ValueError("old backbone attribution requires staged O0/O1, G6 and original post-norm")
    base = {key: value for key, value in params.items()
            if key not in ("local_route", "local_depth", "local_layout", "global_depth", "lg_baseline")}
    base.update(local_route="none", lg_baseline=True)
    return base


def old_source_key(key: str) -> str:
    """Map an O0/O1 content layer to its retained original-baseline tensor."""
    return re.sub(r"^encoder\.global_layers\.(\d+)\.layer\.", r"encoder.layers.\1.", key)


def initialize_legacy_backbone_model(transformer, params: dict, seed: int) -> dict:
    from model.transformer import Transformer
    from utils.lg_baseline_experiment import initialize_baseline_model
    from utils.local_structure_experiment import initialize_structure_control
    from utils.zproj_alignment import state_hash

    base_params = baseline_parameters(params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        baseline = Transformer(**base_params)
        baseline_info = initialize_baseline_model(baseline, base_params, seed)
    source = baseline.state_dict()
    local, local_info = {}, None
    if params["local_route"] == "O1":
        c1_params = {**params, "local_route": "C1", "local_depth": 2}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            c1 = Transformer(**c1_params)
            local_info = initialize_structure_control(c1, c1_params, seed)
        local = {key: value for key, value in c1.state_dict().items() if key.startswith(LOCAL_PREFIX)}
    expected = transformer.state_dict()
    shared = {key: source[old_source_key(key)] for key in expected if not key.startswith(LOCAL_PREFIX)}
    mapped = {**shared, **local}
    if mapped.keys() != expected.keys() or any(mapped[key].shape != expected[key].shape for key in expected):
        raise ValueError("O0/O1 tensors do not match retained old baseline and unchanged C1")
    if any("rp_proj" in key or "rp_encoder" in key for key in expected):
        raise ValueError("geometric scoring parameters must not exist in O0/O1")
    transformer.load_state_dict(mapped, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in mapped.items()):
        raise ValueError("old backbone / C1 untrained tensors were not copied verbatim")
    return {"protocol": LEGACY_INIT, "seed": seed, "route": params["local_route"],
            "baseline_initialization": baseline_info, "c1_initialization": local_info,
            "shared_old_state_sha256": state_hash(shared),
            "local_state_sha256": state_hash(local) if local else None,
            "state_sha256": state_hash(current), "n_shared_tensors": len(shared),
            "n_local_tensors": len(local),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def legacy_backbone_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_legacy_backbone_config(cfg)
    baseline_parameters(params)
    return {"schema": LEGACY_SCHEMA, "init_protocol": LEGACY_INIT, "seed": cfg.seed,
            "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, "training": training_record(cfg),
            "ablation": {"global_backbone": "old q=k=v, post-norm",
                         "geometry_scoring": False, "local_content": "C1" if cfg.local_route == "O1" else None,
                         "global_depth": 6, "local_depth": cfg.local_depth,
                         "scope": "old/new backbone comparison includes encoder parameterization and initial values"}}
