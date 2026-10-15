"""Baseline comparison arm: original six-layer encoder with the ZP entrance.

The arm answers exactly one question: what changes when the whole encoder is
replaced by the local/global family, at seed 42 and 35 epochs? It is not an
attribution of any effect to local aggregation alone, and it never replaces the
historical ZP100 baseline or its fingerprint protocol.

Initialization is aligned to the four-arm ``local-global-init-v1`` protocol. The
untrained A/staged/K=2/seed=42 reference supplies every parameter and buffer
outside ``encoder.*``; the original encoder keeps its own independent
construction. The four arms define their common part differently (everything
except the local message weights), so both hashes are recorded: the
reference-level one is directly comparable with the arms' ``common_state_sha256``.

Everything else — model/loss/optimizer/scheduler/dropout/grids/data order — is
the four-arm configuration. The contract schema is deliberately distinct so a
checkpoint from another arm is rejected instead of silently resumed.
"""
from __future__ import annotations

from copy import deepcopy

import torch

from utils.experiment_config import ExperimentConfig
from utils.checkpoint_selection import selection_contract
from utils.local_global_experiment import (
    INIT_PROTOCOL, SELECTION_METRICS, _is_local_parameter, training_record,
    untrained_reference_state)

BASELINE_INIT_PROTOCOL = "lg-baseline-init-v1"
BASELINE_CONTRACT_SCHEMA = "lg-baseline-model-v1"
BASELINE_REFERENCE_ROUTE = "A"
BASELINE_REFERENCE_LAYOUT = "staged"
BASELINE_REFERENCE_DEPTH = 2
# Fixed by the reference arm; not a tunable of this comparison.
REFERENCE_IDENTITY = {"route": BASELINE_REFERENCE_ROUTE, "layout": BASELINE_REFERENCE_LAYOUT,
                      "depth": BASELINE_REFERENCE_DEPTH, "protocol": INIT_PROTOCOL}


def _is_encoder_parameter(key: str) -> bool:
    return key.startswith("encoder.")


def validate_baseline_config(cfg: ExperimentConfig) -> None:
    """Frozen contract of the baseline comparison arm.

    Same restrictions as the local/global trials: M1 with the fixed ZP entrance,
    SumNorm/H1, valid-only selection and an explicit budget and tag. No
    historical trained weights, no AMP and no side geometry paths, so the only
    difference against the four arms stays the encoder.
    """
    if not cfg.lg_baseline:
        return
    if cfg.local_route != "none" or cfg.local_layout is not None or cfg.local_depth is not None \
            or cfg.global_depth is not None:
        raise ValueError("lg_baseline is the old-encoder arm and requires local_route='none'")
    if cfg.model_name != "M1" or cfg.atom_feat != "z_only_proj":
        raise ValueError("the baseline comparison requires M1 and ZP (z_only_proj)")
    if cfg.norm != "sumnorm" or cfg.scale_mode != "eta" or not cfg.skip_test_eval:
        raise ValueError("the baseline comparison requires SumNorm/H1 and valid-only selection")
    if cfg.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"choose --selection_metric explicitly from {SELECTION_METRICS}")
    conflicts = {
        "use_g1": cfg.use_g1, "use_g2": cfg.use_g2,
        "use_periodic_manybody": cfg.use_periodic_manybody,
        "use_macro_lattice": cfg.use_macro_lattice, "use_amp": cfg.use_amp,
        "init_ckpt": bool(cfg.init_ckpt), "freeze_backbone": cfg.freeze_backbone,
        "reset_rng_after_init": cfg.reset_rng_after_init,
        "pair_aux": cfg.pair_aux_arm != "none" or cfg.pair_ratio != 0,
        "slope_aux": cfg.edos_slope_ratio != 0, "scalar_aux": cfg.scalar_mode != "none",
        "c5_moe": cfg.c5_moe, "atom_additive": cfg.use_atom_additive_phdos,
        "r1a_point": cfg.r1a_point, "r1b_coord": cfg.r1b_coord,
    }
    enabled = sorted(key for key, value in conflicts.items() if value)
    if enabled:
        raise ValueError(f"the baseline comparison conflicts with {enabled}")
    if not cfg.tag or cfg.epochs < 1 or cfg.batch_size < 1 or not cfg.lr > 0:
        raise ValueError("the baseline comparison needs an isolated tag and a positive training configuration")


def baseline_reference_params(params: dict) -> dict:
    """Four-arm A/staged/K=2 reference parameters for these non-encoder values.

    Only the route knobs change: every other value, including the forced
    ``token_num``, must already be identical to the comparison arms so the
    copied common part is meaningful.
    """
    if params.get("local_route", "none") != "none":
        raise ValueError("the baseline target must use local_route='none' parameters")
    reference = {key: value for key, value in params.items()
                 if key not in ("local_route", "local_layout", "local_depth",
                                "global_depth", "lg_baseline")}
    reference.update({"local_route": BASELINE_REFERENCE_ROUTE,
                      "local_layout": BASELINE_REFERENCE_LAYOUT,
                      "local_depth": BASELINE_REFERENCE_DEPTH})
    return reference


def initialize_baseline_model(transformer, params: dict, seed: int) -> dict:
    """Copy the reference's non-encoder state and keep the original encoder.

    Keys and shapes are checked before any copy, the copy is verified value by
    value afterwards, and the encoder tensors are confirmed untouched. The
    reference never shares tensors with the trainable target.
    """
    from utils.zproj_alignment import state_hash

    if getattr(transformer, "local_route", "none") != "none" or not transformer.lg_baseline:
        raise ValueError("baseline initialization requires the old encoder with lg_baseline=True")
    if params.get("local_route", "none") != "none":
        raise ValueError("baseline initialization requires local_route='none' parameters")
    state = untrained_reference_state(seed, baseline_reference_params(params))
    expected = transformer.state_dict()

    reference_common = {key: value for key, value in state.items()
                        if not _is_encoder_parameter(key)}
    target_encoder = {key for key in expected if _is_encoder_parameter(key)}
    target_common = {key for key in expected if not _is_encoder_parameter(key)}
    if not target_encoder:
        raise ValueError("baseline initialization requires an original encoder to keep")
    if not (target_encoder | target_common) == set(expected) or target_encoder & target_common:
        raise ValueError("baseline state is not a clean encoder/common split")
    missing = target_common - set(reference_common)
    unexpected = set(reference_common) - target_common
    if missing or unexpected:
        raise ValueError("baseline common keys do not match the local/global reference: "
                         f"missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}")
    mismatched = sorted(key for key in reference_common
                        if reference_common[key].shape != expected[key].shape)
    if mismatched:
        raise ValueError(f"baseline common tensors have incompatible shapes: {mismatched[:5]}")

    encoder_before = {key: expected[key].clone() for key in sorted(target_encoder)}
    merged = {key: reference_common[key] if key in reference_common else expected[key]
              for key in expected}
    transformer.load_state_dict(merged, strict=True)
    current = transformer.state_dict()

    shared = sorted(key for key in state if key in current and current[key] is state[key])
    if shared:
        raise ValueError(f"baseline must not share tensors with the reference: {shared[:5]}")
    inconsistent = sorted(key for key in reference_common
                          if not torch.equal(current[key], reference_common[key]))
    if inconsistent:
        raise ValueError(f"baseline common tensors were not copied verbatim: {inconsistent[:5]}")
    touched = sorted(key for key in encoder_before
                     if not torch.equal(current[key], encoder_before[key]))
    if touched:
        raise ValueError(f"the copy step modified the original encoder: {touched[:5]}")

    return {
        "protocol": BASELINE_INIT_PROTOCOL,
        "seed": seed,
        "reference": dict(REFERENCE_IDENTITY),
        "n_common_tensors": len(target_common),
        "n_encoder_tensors": len(target_encoder),
        # Non-encoder part actually carried by this model.
        "common_state_sha256": state_hash({key: current[key] for key in sorted(target_common)}),
        # The independently initialized original encoder.
        "encoder_state_sha256": state_hash({key: current[key] for key in sorted(target_encoder)}),
        # Whole model.
        "state_sha256": state_hash(current),
        # Same subset definition as local-global-init-v1's common_state_sha256,
        # so it can be compared directly with the recorded four-arm value.
        "reference_common_state_sha256": state_hash(
            {key: value for key, value in state.items() if not _is_local_parameter(key)}),
    }


def baseline_contract(cfg: ExperimentConfig, transformer_params: dict) -> dict:
    """Model/data contract of the baseline arm; distinct schema rejects others."""
    return {"schema": BASELINE_CONTRACT_SCHEMA, "init_protocol": BASELINE_INIT_PROTOCOL,
            "seed": cfg.seed, "transformer": deepcopy(transformer_params),
            "reference": dict(REFERENCE_IDENTITY),
            "normalization": cfg.norm, "selection_metric": cfg.selection_metric,
            "training": training_record(cfg), **selection_contract(cfg.selection_metric)}
