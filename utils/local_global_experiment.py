"""Configuration, matched initialization and identity for local/global trials.

No training is dispatched here. Historical ZP100 fingerprints are deliberately
absent: a canonical untrained A model supplies only the common components.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import re
import random

import numpy as np
import torch

from utils.experiment_config import ExperimentConfig
from utils.checkpoint_selection import BLIND_R2_METRIC, SELECTION_METRICS, selection_contract

INIT_PROTOCOL = "local-global-init-v1"
# Decoupled-depth arms copy the K=2 reference and extend only the global stack.
DEEP_INIT_PROTOCOL = "local-global-depth-init-v1"
DEEP_CONTRACT_SCHEMA = "local-global-depth-model-v1"


def is_deep_depths(local_depth, global_depth) -> bool:
    """True only for the supported decoupling: a longer global stack."""
    return global_depth is not None and global_depth != local_depth


def validate_local_global_config(cfg: ExperimentConfig) -> None:
    if cfg.local_route == "LN1":
        from utils.local_norm_experiment import validate_local_norm_config
        validate_local_norm_config(cfg)
        return
    if cfg.local_route == "W1":
        from utils.local_width_experiment import validate_local_width_config
        validate_local_width_config(cfg)
        return
    if cfg.local_route == "I4":
        from utils.local_readout_experiment import validate_local_readout_config
        validate_local_readout_config(cfg)
        return
    if cfg.selection_metric == BLIND_R2_METRIC:
        # Keep frozen experiment factories intact: the architecture/training
        # recipe is checked as before; only the selection protocol changes.
        validate_local_global_config(replace(cfg, selection_metric="balanced_oracle_mae"))
        return
    if cfg.local_route == "I3":
        from utils.i1_a_experiment import validate_i1_a_config
        validate_i1_a_config(cfg)
        return
    if cfg.local_route == "I2":
        from utils.i1_bplus_experiment import validate_i1_bplus_config
        validate_i1_bplus_config(cfg)
        return
    if cfg.local_route in ("I1", "I0"):
        from utils.attention_projection_experiment import validate_attention_projection_config
        validate_attention_projection_config(cfg)
        return
    if cfg.local_route == "D1":
        from utils.attention_dropout_experiment import validate_attention_dropout_config
        validate_attention_dropout_config(cfg)
        return
    if cfg.local_route == "N1":
        from utils.global_norm_experiment import validate_global_norm_config
        validate_global_norm_config(cfg)
        return
    if cfg.local_route in ("O0", "O1"):
        from utils.legacy_backbone_experiment import validate_legacy_backbone_config
        validate_legacy_backbone_config(cfg)
        return
    if cfg.local_route == "G0":
        from utils.global_only_experiment import validate_global_only_config
        validate_global_only_config(cfg)
        return
    if cfg.local_route == "none":
        if (cfg.local_layout is not None or cfg.local_depth is not None
                or cfg.global_depth is not None):
            raise ValueError("local layout/depth require --local_route A or B+")
        # The baseline arm selects checkpoints with the same metric without
        # being a local/global route; every other arm still needs a route.
        if cfg.selection_metric is not None and not cfg.lg_baseline:
            raise ValueError("explicit selection metrics require --local_route A or B+ or --lg_baseline")
        return
    if cfg.local_route not in ("A", "B+", "C0", "C1") or cfg.local_layout not in ("staged", "alternating"):
        raise ValueError("explicit local route and staged/alternating layout required")
    if cfg.local_route in ("C0", "C1"):
        from utils.local_structure_experiment import validate_structure_control_config
        validate_structure_control_config(cfg)
    if isinstance(cfg.local_depth, bool) or not isinstance(cfg.local_depth, int) or cfg.local_depth < 1:
        raise ValueError("--local_depth must be a positive integer")
    if cfg.global_depth is not None:
        if (isinstance(cfg.global_depth, bool) or not isinstance(cfg.global_depth, int)
                or cfg.global_depth < 1):
            raise ValueError("--global_depth must be a positive integer")
        if not is_deep_depths(cfg.local_depth, cfg.global_depth):
            raise ValueError("--global_depth must differ from --local_depth")
        if cfg.global_depth < cfg.local_depth:
            raise ValueError("the matched initialization protocol extends only the global "
                             "stack: --global_depth must exceed --local_depth")
    if cfg.model_name != "M1" or cfg.atom_feat != "z_only_proj":
        raise ValueError("current local/global comparison requires M1 and ZP (z_only_proj)")
    if cfg.norm != "sumnorm" or cfg.scale_mode != "eta" or not cfg.skip_test_eval:
        raise ValueError("local/global trials require SumNorm/H1 and valid-only selection")
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
        raise ValueError(f"local/global comparison conflicts with {enabled}")
    if not cfg.tag or cfg.epochs < 1 or cfg.batch_size < 1 or not np.isfinite(cfg.lr) or cfg.lr <= 0:
        raise ValueError("local/global trials need an isolated tag and positive training configuration")


def comparison_configs(base: ExperimentConfig, *, depth: int, seeds: list[int],
                       tag_prefix: str) -> list[ExperimentConfig]:
    """Four equal-count arms per seed, with explicit budget supplied by caller."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
        raise ValueError("four-arm comparison needs depth >= 2; depth 1 layouts coincide")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("nonnegative, distinct seeds required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    configs = []
    for seed in seeds:
        for route, label in (("A", "a"), ("B+", "bp")):
            for layout in ("staged", "alternating"):
                cfg = replace(base, seed=seed, local_route=route, local_layout=layout,
                              local_depth=depth,
                              tag=f"_lg_{tag_prefix}_k{depth}_{label}_{layout}_s{seed}")
                validate_local_global_config(cfg)
                configs.append(cfg)
    return configs


def depth_comparison_configs(base: ExperimentConfig, *, depth: int, global_depth: int,
                             seeds: list[int], tag_prefix: str,
                             routes: tuple[str, ...] = ("B+",)) -> list[ExperimentConfig]:
    """Decoupled-depth arms: the same local calls, a longer global stack.

    Each arm gets its own tag and its own contract schema; the two layouts share
    one initial state, so an observed difference is the orchestration only.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth comparison needs a positive local depth")
    if isinstance(global_depth, bool) or not isinstance(global_depth, int) \
            or global_depth <= depth:
        raise ValueError("depth comparison needs global_depth > local_depth")
    if not routes or any(route not in ("A", "B+") for route in routes):
        raise ValueError("depth comparison routes must be A or B+")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("nonnegative, distinct seeds required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag_prefix):
        raise ValueError("tag prefix must contain only letters, digits, underscores or hyphens")
    configs = []
    for seed in seeds:
        for route in routes:
            label = "a" if route == "A" else "bp"
            for layout in ("staged", "alternating"):
                cfg = replace(base, seed=seed, local_route=route, local_layout=layout,
                              local_depth=depth, global_depth=global_depth,
                              tag=(f"_lgd_{tag_prefix}_k{depth}_g{global_depth}"
                                   f"_{label}_{layout}_s{seed}"))
                validate_local_global_config(cfg)
                configs.append(cfg)
    return configs


def _is_local_parameter(key: str) -> bool:
    return key.startswith("encoder.local_layers.") and ".message." in key


def _is_extended_global_parameter(key: str, local_depth: int) -> bool:
    """Global layers past the reference's K: initialized separately, never copied."""
    prefix = "encoder.global_layers."
    if not key.startswith(prefix):
        return False
    return int(key[len(prefix):].split(".", 1)[0]) >= local_depth


def untrained_reference_state(seed: int, params: dict) -> dict:
    """Build the canonical untrained local/global reference state.

    Construction runs under a scoped CPU RNG so the caller's random stream is
    untouched. ``params`` must describe an A/B+ model; the canonical reference
    always uses the staged layout. A B+ reference receives every non-local
    component from a same-seed untrained A reference, so the common part is one
    shared value across all comparison arms. Returned tensors stay owned by the
    discarded reference module, so no parameter is ever shared with a target.
    """
    from model.transformer import Transformer

    if params.get("local_route") not in ("A", "B+"):
        raise ValueError("matched initialization requires an A/B+ model")
    selected = {**params, "local_layout": "staged"}
    with torch.random.fork_rng(devices=[]):
        # Seed CPU only; reference construction must not change CUDA RNG state.
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**selected)
        if selected["local_route"] == "B+":
            torch.random.default_generator.manual_seed(seed)
            canonical = Transformer(**{**selected, "local_route": "A"})
            source = canonical.state_dict()
            state = reference.state_dict()
            common = {key: source[key] for key in state if not _is_local_parameter(key)}
            reference.load_state_dict({**state, **common}, strict=True)
            del canonical
        return reference.state_dict()


def deep_reference_state(seed: int, params: dict) -> tuple[dict, dict]:
    """K=2 reference state plus independently initialized extra global layers.

    The shared part is the complete ``local-global-init-v1`` reference of the
    same seed and route, so it carries both local updates, the first
    ``local_depth`` global layers and every downstream module verbatim. Only the
    extra global layers are new: they are drawn once from a scoped untrained
    canonical construction, which makes them identical for every layout of the
    same experiment. No trained weight is read.
    """
    from model.transformer import Transformer

    local_depth, global_depth = params.get("local_depth"), params.get("global_depth")
    if params.get("local_route") not in ("A", "B+") or not isinstance(local_depth, int) \
            or not is_deep_depths(local_depth, global_depth):
        raise ValueError("deep initialization requires an A/B+ model with "
                         "global_depth larger than local_depth")
    shared = untrained_reference_state(seed, {**params, "local_layout": "staged",
                                             "global_depth": local_depth})
    with torch.random.fork_rng(devices=[]):
        # Seed CPU only; reference construction must not change CUDA RNG state.
        torch.random.default_generator.manual_seed(seed)
        source = Transformer(**{**params, "local_route": "A", "local_layout": "staged"})
        extended = {key: value for key, value in source.state_dict().items()
                    if _is_extended_global_parameter(key, local_depth)}
        del source
    if not extended or set(shared) & set(extended):
        raise ValueError("deep reference did not split into shared and extended tensors")
    return {**shared, **extended}, shared


def initialize_deep_model(transformer, params: dict, seed: int) -> dict:
    """Copy the shared tensors from the K=2 reference and extend the globals.

    Every shared tensor is copied value by value and verified afterwards; the
    extended global layers are checked to survive the copy untouched. Both
    layouts receive the same state, recorded here as hashes.
    """
    from utils.zproj_alignment import state_hash

    local_depth = params.get("local_depth")
    state, shared_reference = deep_reference_state(seed, params)
    expected = transformer.state_dict()
    if expected.keys() != state.keys() or any(expected[k].shape != state[k].shape for k in state):
        raise ValueError("initialization parameters do not describe the target model")
    shared = {key: value for key, value in state.items()
              if not _is_extended_global_parameter(key, local_depth)}
    extended = {key: value for key, value in state.items()
                if _is_extended_global_parameter(key, local_depth)}
    transformer.load_state_dict(state, strict=True)
    current = transformer.state_dict()
    inconsistent = sorted(key for key in shared
                          if not torch.equal(current[key], shared[key]))
    if inconsistent:
        raise ValueError(f"shared tensors were not copied verbatim: {inconsistent[:5]}")
    relaid = sorted(key for key in extended if not torch.equal(current[key], extended[key]))
    if relaid:
        raise ValueError(f"extended global layers did not take the canonical init: {relaid[:5]}")
    borrowed = sorted(key for key in state if current[key] is state[key])
    if borrowed:
        raise ValueError(f"the target must not share tensors with the reference: {borrowed[:5]}")
    return {
        "protocol": DEEP_INIT_PROTOCOL,
        "shared_init_protocol": INIT_PROTOCOL,
        "seed": seed,
        "reference": {"protocol": INIT_PROTOCOL, "layout": "staged",
                      "local_depth": local_depth, "global_depth": local_depth},
        "local_depth": local_depth,
        "global_depth": params.get("global_depth"),
        "execution_order": [list(step) for step in transformer.encoder.execution_order],
        "n_shared_tensors": len(shared),
        "n_extended_global_tensors": len(extended),
        "shared_state_sha256": state_hash(shared),
        "extended_global_state_sha256": state_hash(extended),
        "common_state_sha256": state_hash(
            {key: value for key, value in current.items() if not _is_local_parameter(key)}),
        # Same subset definition as local-global-init-v1, so it can be compared
        # directly with the recorded four-arm common_state_sha256.
        "reference_common_state_sha256": state_hash(
            {key: value for key, value in shared_reference.items() if not _is_local_parameter(key)}),
        "state_sha256": state_hash(current),
        "update_counts": transformer.encoder.update_counts,
    }


def initialize_comparison_model(transformer, params: dict, seed: int) -> dict:
    """Match all common weights, keep route-specific local weights independent.

    Build the selected route in canonical staged order under a scoped CPU RNG.
    B+ copies ZP/global/readout/external norms from a same-seed untrained A
    reference. References never share Parameters with the trainable target.
    Constructor behavior (including zero biases/trunks) remains authoritative.
    A decoupled global stack extends this rule instead of replacing it.
    """
    from utils.zproj_alignment import state_hash

    if params.get("local_route") == "LN1":
        from utils.local_norm_experiment import initialize_local_norm_model
        return initialize_local_norm_model(transformer, params, seed)
    if params.get("local_route") == "W1":
        from utils.local_width_experiment import initialize_local_width_model
        return initialize_local_width_model(transformer, params, seed)
    if params.get("local_route") == "I4":
        from utils.local_readout_experiment import initialize_local_readout_model
        return initialize_local_readout_model(transformer, params, seed)
    if params.get("local_route") == "I3":
        from utils.i1_a_experiment import initialize_i1_a_model
        return initialize_i1_a_model(transformer, params, seed)
    if params.get("local_route") == "I2":
        from utils.i1_bplus_experiment import initialize_i1_bplus_model
        return initialize_i1_bplus_model(transformer, params, seed)
    if params.get("local_route") in ("I1", "I0"):
        from utils.attention_projection_experiment import initialize_attention_projection_model
        return initialize_attention_projection_model(transformer, params, seed)
    if params.get("local_route") == "D1":
        from utils.attention_dropout_experiment import initialize_attention_dropout_model
        return initialize_attention_dropout_model(transformer, params, seed)
    if params.get("local_route") == "N1":
        from utils.global_norm_experiment import initialize_global_norm_model
        return initialize_global_norm_model(transformer, params, seed)
    if params.get("local_route") in ("O0", "O1"):
        from utils.legacy_backbone_experiment import initialize_legacy_backbone_model
        return initialize_legacy_backbone_model(transformer, params, seed)
    if params.get("local_route") == "G0":
        from utils.global_only_experiment import initialize_global_only_model
        return initialize_global_only_model(transformer, params, seed)
    if params.get("local_route") in ("C0", "C1"):
        from utils.local_structure_experiment import initialize_structure_control
        return initialize_structure_control(transformer, params, seed)
    if params.get("local_route") not in ("A", "B+"):
        raise ValueError("matched initialization requires an A/B+ model")
    if is_deep_depths(params.get("local_depth"), params.get("global_depth")):
        return initialize_deep_model(transformer, params, seed)
    state = untrained_reference_state(seed, params)
    expected = transformer.state_dict()
    if expected.keys() != state.keys() or any(expected[k].shape != state[k].shape for k in state):
        raise ValueError("initialization parameters do not describe the target model")
    transformer.load_state_dict(state, strict=True)
    common_state = {key: value for key, value in state.items() if not _is_local_parameter(key)}
    result = {"protocol": INIT_PROTOCOL, "seed": seed,
              "common_state_sha256": state_hash(common_state),
              "state_sha256": state_hash(state),
              "update_counts": transformer.encoder.update_counts}
    return result


def training_record(cfg: ExperimentConfig) -> dict:
    """Training knobs carried by a model contract.

    ``global_depth`` is recorded only when set, so a default-depth contract
    stays byte-identical to the contracts already saved by earlier arms.
    """
    return {key: value for key, value in asdict(cfg).items()
            if key not in ('tag', 'skip_existing')
            and (key != 'global_depth' or value is not None)}


def comparison_contract(cfg: ExperimentConfig, transformer_params: dict) -> dict | None:
    if cfg.local_route == "LN1":
        from utils.local_norm_experiment import local_norm_contract
        return local_norm_contract(cfg, transformer_params)
    if cfg.local_route == "W1":
        from utils.local_width_experiment import local_width_contract
        return local_width_contract(cfg, transformer_params)
    if cfg.local_route == "I4":
        from utils.local_readout_experiment import local_readout_contract
        return local_readout_contract(cfg, transformer_params)
    if cfg.selection_metric == BLIND_R2_METRIC:
        contract = comparison_contract(replace(cfg, selection_metric="balanced_oracle_mae"), transformer_params)
        if contract is None:
            return None
        return {**contract, "selection_metric": cfg.selection_metric, "training": training_record(cfg),
                **selection_contract(cfg.selection_metric)}
    if cfg.local_route == "I3":
        from utils.i1_a_experiment import i1_a_contract
        return i1_a_contract(cfg, transformer_params)
    if cfg.local_route == "I2":
        from utils.i1_bplus_experiment import i1_bplus_contract
        return i1_bplus_contract(cfg, transformer_params)
    if cfg.local_route in ("I1", "I0"):
        from utils.attention_projection_experiment import attention_projection_contract
        return attention_projection_contract(cfg, transformer_params)
    if cfg.local_route == "D1":
        from utils.attention_dropout_experiment import attention_dropout_contract
        return attention_dropout_contract(cfg, transformer_params)
    if cfg.local_route == "N1":
        from utils.global_norm_experiment import global_norm_contract
        return global_norm_contract(cfg, transformer_params)
    if cfg.local_route in ("O0", "O1"):
        from utils.legacy_backbone_experiment import legacy_backbone_contract
        return legacy_backbone_contract(cfg, transformer_params)
    if cfg.local_route == "G0":
        from utils.global_only_experiment import global_only_contract
        return global_only_contract(cfg, transformer_params)
    if cfg.local_route == "none":
        return None
    if cfg.local_route in ("C0", "C1"):
        from utils.local_structure_experiment import structure_control_contract
        return structure_control_contract(cfg, transformer_params)
    training = training_record(cfg)
    if is_deep_depths(cfg.local_depth, cfg.global_depth):
        return {"schema": DEEP_CONTRACT_SCHEMA, "init_protocol": DEEP_INIT_PROTOCOL,
                "shared_init_protocol": INIT_PROTOCOL,
                "seed": cfg.seed, "transformer": deepcopy(transformer_params),
                "reference": {"protocol": INIT_PROTOCOL, "layout": "staged",
                              "local_depth": cfg.local_depth, "global_depth": cfg.local_depth},
                "normalization": cfg.norm, "selection_metric": cfg.selection_metric,
                "training": training}
    return {"schema": "local-global-model-v1", "init_protocol": INIT_PROTOCOL,
            "seed": cfg.seed, "transformer": deepcopy(transformer_params),
            "normalization": cfg.norm, "selection_metric": cfg.selection_metric,
            "training": training}


def capture_training_rng() -> dict:
    """Checkpoint-safe Python/NumPy/Torch streams (no NumPy pickle objects)."""
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_training_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("comparison resume requires matching CUDA RNG devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
