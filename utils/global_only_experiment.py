"""Global-only control: keep the G6 backbone and remove every local update."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

import torch

from utils.experiment_config import ExperimentConfig
from utils.local_structure_experiment import validate_structure_control_config

GLOBAL_ONLY_INIT = "global-only-init-v1"
GLOBAL_ONLY_SCHEMA = "global-only-model-v1"
LOCAL_PREFIX = "encoder.local_layers."


def validate_global_only_config(cfg: ExperimentConfig) -> None:
    if (cfg.local_route != "G0" or isinstance(cfg.local_depth, bool)
            or not isinstance(cfg.local_depth, int) or cfg.local_depth != 0):
        raise ValueError("global-only control requires G0 and local_depth=0")
    if isinstance(cfg.global_depth, bool) or not isinstance(cfg.global_depth, int) or cfg.global_depth != 6:
        raise ValueError("global-only control requires integer global_depth=6")
    # Reuse the exact C0/C1 recipe, differing only in removal of local updates.
    validate_structure_control_config(replace(cfg, local_route="C0", local_depth=2))
    if not cfg.tag:
        raise ValueError("global-only control requires an isolated tag")


def global_only_config(reference: ExperimentConfig, tag_prefix: str) -> ExperimentConfig:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    cfg = replace(reference, local_route="G0", local_depth=0, skip_existing=False,
                  tag=f"_g0_{tag_prefix}_k0_g6_staged_s42_e35")
    validate_global_only_config(cfg)
    return cfg


def global_only_reference_params(params: dict) -> dict:
    if (params.get("local_route") != "G0" or params.get("local_depth") != 0
            or params.get("global_depth") != 6 or params.get("local_layout") != "staged"):
        raise ValueError("global-only initialization requires G0, staged K0/G6")
    return {**params, "local_route": "B+", "local_depth": 2}


def initialize_global_only_model(transformer, params: dict, seed: int) -> dict:
    """Copy all surviving tensors from the untrained B+ L2/G6 reference.

    No trained weights or local parameters are retained. The source and target
    own separate tensors; reference construction preserves caller RNG streams.
    """
    from utils.local_global_experiment import deep_reference_state
    from utils.zproj_alignment import state_hash

    reference, _ = deep_reference_state(seed, global_only_reference_params(params))
    shared = {key: value for key, value in reference.items() if not key.startswith(LOCAL_PREFIX)}
    target = transformer.state_dict()
    if target.keys() != shared.keys() or any(target[key].shape != shared[key].shape for key in target):
        raise ValueError("global-only target must exactly match the reference without local layers")
    transformer.load_state_dict(shared, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in shared.items()):
        raise ValueError("global-only shared tensors were not copied verbatim")
    return {"protocol": GLOBAL_ONLY_INIT, "seed": seed, "route": "G0",
            "reference_init_protocol": "local-global-depth-init-v1",
            "reference_state_sha256": state_hash(reference),
            "shared_state_sha256": state_hash(shared), "state_sha256": state_hash(current),
            "n_shared_tensors": len(shared),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def global_only_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    validate_global_only_config(cfg)
    global_only_reference_params(params)
    return {"schema": GLOBAL_ONLY_SCHEMA, "init_protocol": GLOBAL_ONLY_INIT,
            "reference_init_protocol": "local-global-depth-init-v1",
            "seed": cfg.seed, "transformer": deepcopy(params),
            "normalization": cfg.norm, "selection_metric": cfg.selection_metric,
            "training": training_record(cfg),
            "ablation": {"removed": "all local updates", "reference_local_depth": 2,
                         "global_depth": 6, "geometry_consumed": False}}
