"""Frozen C0/C1 protocol, untrained B+ initialization and isolated run identity."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import re

import torch
import yaml

from utils.experiment_config import ExperimentConfig
from utils.paths import file_sha256, input_path

CONTROL_INIT_PROTOCOL = "local-structure-control-init-v1"
CONTROL_CONTRACT_SCHEMA = "local-structure-control-model-v1"
DEFAULT_REFERENCE = ("results/runs/ablation_m1_lgd_s42g6sel20261008"
                     "_k2_g6_bp_staged_s42")
FROZEN = dict(model_name="M1", epochs=35, batch_size=32, lr=5e-5, seed=42,
              atom_feat="z_only_proj", norm="sumnorm", scale_mode="eta",
              skip_test_eval=True, local_layout="staged", local_depth=2, global_depth=6,
              selection_metric="balanced_oracle_mae", q1_coord=True, q1_hidden=128,
              decoder_layers=6, edos_num=128, phdos_num=64)


def validate_structure_control_config(cfg: ExperimentConfig) -> None:
    """Only the agreed same-backbone, seed-42, 35-epoch attribution experiment."""
    defaults = ExperimentConfig()
    expected_values = {field.name: FROZEN.get(field.name, getattr(defaults, field.name))
                       for field in fields(ExperimentConfig)
                       if field.name not in ("local_route", "tag", "skip_existing")}
    differences = {key: [getattr(cfg, key), expected] for key, expected in expected_values.items()
                   if getattr(cfg, key) != expected}
    if differences:
        raise ValueError(f"structure controls require the frozen C0/C1 recipe: {differences}")
    variants = ("augment", "use_mask", "use_bucket_batch", "q2_fourier", "lg_baseline")
    if any(getattr(cfg, key) for key in variants) or cfg.energy_code != "none":
        raise ValueError("structure controls cannot add augmentation, masks, buckets or readout variants")


def is_shared_control_parameter(key: str) -> bool:
    """Share everything outside message content, plus equivalent projection/LN."""
    prefix = "encoder.local_layers."
    if not key.startswith(prefix) or ".message." not in key:
        return True
    suffix = key.split(".message.", 1)[1]
    return (suffix.startswith(("local_input.", "local_output."))
            or re.fullmatch(r"blocks\.[01]\.norm\.(weight|bias)", suffix) is not None)


def parameter_counts(transformer) -> dict:
    named = {key: value.numel() for key, value in transformer.named_parameters()
             if value.requires_grad}
    message = {key: value for key, value in named.items()
               if key.startswith("encoder.local_layers.") and ".message." in key}
    return {"total": sum(named.values()), "local_message": sum(message.values()),
            "local_content": sum(value for key, value in message.items()
                                 if not is_shared_control_parameter(key)),
            "shared": sum(value for key, value in named.items()
                          if is_shared_control_parameter(key))}


def initialize_structure_control(transformer, params: dict, seed: int) -> dict:
    """Copy common values from the canonical untrained B+ G6 reference.

    Control-specific parameters are drawn under their own same-seed constructor.
    No trained checkpoint is loaded; all reference tensors are copied, not shared.
    The caller's CPU/CUDA random streams are left untouched. The training runner
    already resets all streams to the experiment seed immediately before training.
    """
    from model.transformer import Transformer
    from utils.local_global_experiment import deep_reference_state, _is_local_parameter
    from utils.zproj_alignment import state_hash

    if (params.get("local_route") not in ("C0", "C1")
            or params.get("local_layout") != "staged" or params.get("local_depth") != 2
            or params.get("global_depth") != 6):
        raise ValueError("control initialization requires C0/C1, staged L2/G6")
    reference, _ = deep_reference_state(seed, {**params, "local_route": "B+"})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        independent = Transformer(**params).state_dict()
    target = transformer.state_dict()
    if target.keys() != independent.keys() or any(target[k].shape != independent[k].shape for k in target):
        raise ValueError("control parameters do not describe the target model")
    shared = {key: reference[key] for key in target if is_shared_control_parameter(key)}
    if any(target[key].shape != value.shape for key, value in shared.items()):
        raise ValueError("control common tensor shapes differ from B+")
    transformer.load_state_dict({**independent, **shared}, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in shared.items()):
        raise ValueError("common control tensors were not copied verbatim")
    return {"protocol": CONTROL_INIT_PROTOCOL, "seed": seed, "route": params["local_route"],
            "reference_init_protocol": "local-global-depth-init-v1",
            "reference_state_sha256": state_hash(reference),
            "common_state_sha256": state_hash({key: value for key, value in current.items()
                                                if not _is_local_parameter(key)}),
            "shared_control_state_sha256": state_hash(shared), "n_shared_tensors": len(shared),
            "state_sha256": state_hash(current), "parameter_counts": parameter_counts(transformer),
            "update_counts": transformer.encoder.update_counts,
            "execution_order": [list(step) for step in transformer.encoder.execution_order]}


def structure_control_contract(cfg: ExperimentConfig, params: dict) -> dict:
    from utils.local_global_experiment import training_record

    return {"schema": CONTROL_CONTRACT_SCHEMA, "init_protocol": CONTROL_INIT_PROTOCOL,
            "reference_init_protocol": "local-global-depth-init-v1",
            "seed": cfg.seed, "transformer": deepcopy(params),
            "normalization": cfg.norm, "selection_metric": cfg.selection_metric,
            "training": training_record(cfg),
            "local_content": {"route": cfg.local_route, "version": 1,
                              "point_widths": [208, 129], "radial_width": 128,
                              "radial_channels": 16, "scalar_channels": 64, "radius": 6.0}}


def load_control_reference(run_dir: Path) -> tuple[dict, ExperimentConfig]:
    """Read the complete existing C2 identity; refuse recipe drift before running."""
    from tools.eval.lg_valid_export import load_run, select_best
    from utils.local_global_experiment import (
        DEEP_CONTRACT_SCHEMA, DEEP_INIT_PROTOCOL, comparison_contract, validate_local_global_config,
    )

    saved, params, contract = load_run(run_dir)
    names = {field.name for field in fields(ExperimentConfig)}
    training = contract["training"]
    if set(training) - names:
        raise ValueError("reference training fields are not supported by this runner")
    suffix = run_dir.name.removeprefix("ablation_m1")
    cfg = ExperimentConfig(**{**training, "tag": suffix})
    if (contract["schema"] != DEEP_CONTRACT_SCHEMA or contract["init_protocol"] != DEEP_INIT_PROTOCOL
            or cfg.local_route != "B+" or comparison_contract(cfg, params) != contract):
        raise ValueError("C2 must be the saved B+ G6 staged comparison")
    validate_local_global_config(cfg)
    validate_structure_control_config(replace(cfg, local_route="C1"))
    data_dir = Path(input_path(cfg.data_dir))
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_contract = saved["cli"]["data_contract"]
    if (file_sha256(manifest_path) != data_contract["manifest_sha256"]
            or manifest["schema"] != data_contract["schema"]
            or data_contract["normalization"] != cfg.norm
            or data_contract["pair_budget"] != cfg.pair_budget):
        raise ValueError("current data manifest/config differs from the C2 reference")
    # Verify train/valid payloads. The unchanged manifest binds all split metadata;
    # this planning/analysis helper does not open test arrays or build a test loader.
    for name, entry in manifest["files"].items():
        if Path(name).parts[0] in ("train", "valid") and file_sha256(data_dir / name) != entry["sha256"]:
            raise ValueError("current train/valid data differ from the reference: " + name)
    for key in names - {"tag", "skip_existing"}:
        if key in saved["cli"] and saved["cli"][key] != getattr(cfg, key):
            raise ValueError(f"reference CLI and contract differ for {key}")
    picked = select_best(run_dir, cfg.selection_metric, saved)
    if int(picked["history"].epoch.max()) != cfg.epochs:
        raise ValueError("C2 reference has not completed its exact training budget")
    latest = torch.load(run_dir / "checkpoint_latest.pth", map_location="cpu", weights_only=True)
    if latest.get("model_contract") != contract or int(latest["epoch"]) != cfg.epochs:
        raise ValueError("C2 latest checkpoint is not the completed reference")
    if latest.get("data_contract") != saved["cli"].get("data_contract"):
        raise ValueError("C2 checkpoint/config data contracts differ")
    return saved, cfg


def structure_control_configs(reference: ExperimentConfig, tag_prefix: str) -> list[ExperimentConfig]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    from utils.local_global_experiment import validate_local_global_config

    controls = [replace(reference, local_route=route, skip_existing=False,
                        tag=f"_lsc_{tag_prefix}_k2_g6_{route.lower()}_staged_s42_e35")
                for route in ("C0", "C1")]
    for cfg in controls:
        validate_local_global_config(cfg)
    return controls
