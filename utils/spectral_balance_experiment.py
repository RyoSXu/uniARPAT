"""Isolated balanced-spectrum protocol on the unchanged complete-A64 backbone."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC, energy_selection_contract, load_balance_protocol


def legacy_balance_config(cfg):
    from utils.experiment_config import ExperimentConfig

    # Legacy factories compare this path as a string. The new protocol checks
    # the actual manifest/payload hashes independently, so equivalent absolute
    # paths need not be rejected by the historical recipe's spelling.
    return replace(cfg, spectral_balance_config="", scale_mode="eta",
                   selection_metric="balanced_blind_r2", data_dir=ExperimentConfig().data_dir)


def validate_balance_config(cfg):
    from utils.local_global_experiment import validate_local_global_config

    if (not cfg.spectral_balance_config or cfg.local_route != "I3"
            or cfg.scale_mode != "per_atom" or cfg.selection_metric != ENERGY_METRIC):
        raise ValueError("balanced spectra require I3/A64, per_atom scale and balanced_blind_energy")
    validate_local_global_config(legacy_balance_config(cfg))
    protocol = load_balance_protocol(cfg.spectral_balance_config, cfg.data_dir)
    if (protocol["reference"]["label"] != "A64"
            or protocol["tasks"]["edos"]["bins"] != cfg.edos_num
            or protocol["tasks"]["phdos"]["bins"] != cfg.phdos_num
            or protocol["tasks"]["edos"]["bin_width"] != cfg.delta_edos
            or protocol["tasks"]["phdos"]["bin_width"] != cfg.delta_phdos):
        raise ValueError("balanced trial requires the fixed A64 guard reference and E0/P0 grids")
    return protocol


def balance_model_contract(cfg, params):
    from utils.local_global_experiment import comparison_contract, training_record

    protocol = validate_balance_config(cfg)
    base_params = {**params, "scale_mode": "eta"}
    base = comparison_contract(legacy_balance_config(cfg), base_params)
    return {"schema": "joint-spectral-balance-model-v1", "init_protocol": "joint-spectral-balance-init-v1",
            "seed": cfg.seed, "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": ENERGY_METRIC, **energy_selection_contract(protocol),
            "spectral_balance": protocol,
            "spectral_balance_config_sha256": file_sha256(input_path(cfg.spectral_balance_config)),
            "training": training_record(cfg), "backbone_reference": base,
            "scope": "scale parameterization, blind objective and checkpoint protocol package; "
                     "complete A64 encoder/decoder unchanged; not a single-factor encoder test"}


def initialize_balance_model(transformer, params, seed):
    """Preserve A64 initial tensors; map eta MLP weights to the raw mass MLP."""
    from model.transformer import Transformer
    from utils.i1_a_experiment import initialize_i1_a_model, validate_i3_model
    from utils.zproj_alignment import state_hash

    if params.get("scale_mode") != "per_atom" or params.get("local_route") != "I3":
        raise ValueError("balanced initialization requires per_atom complete A64")
    base_params = {**params, "scale_mode": "eta"}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**base_params)
        info = initialize_i1_a_model(reference, base_params, seed)
    state = reference.state_dict()
    mapped = {key.replace("eta_head.", "mass_head.", 1) if key.startswith("eta_head.") else key: value
              for key, value in state.items()}
    expected = transformer.state_dict()
    if expected.keys() != mapped.keys() or any(expected[key].shape != mapped[key].shape for key in expected):
        raise ValueError("balanced mass head must preserve A64 MLP tensor shapes")
    transformer.load_state_dict(mapped, strict=True)
    validate_i3_model(transformer, params, require_identity=True)
    actual = transformer.state_dict()
    if any(not torch.equal(actual[key], value.to(actual[key])) for key, value in mapped.items()):
        raise ValueError("balanced initialization did not preserve mapped initial tensors")
    return {"protocol": "joint-spectral-balance-init-v1", "seed": seed,
            "a64_initialization": info, "state_sha256": state_hash(actual),
            "mass_head_mapping": "eta_head.mlp -> mass_head.mlp; raw e/p mass logits replace bounded ph/e ratios",
            "backbone_initial_tensors_identical": True,
            "update_counts": transformer.encoder.update_counts}
