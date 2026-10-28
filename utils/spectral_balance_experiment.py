"""Isolated balanced-spectrum protocol on the unchanged complete-A64 backbone."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC, energy_selection_contract, load_balance_protocol


def shape_supervision_contract(enabled):
    """One fixed candidate; no validation-tuned coefficient or weight search."""
    if not isinstance(enabled, bool):
        raise ValueError("spectral_shape_aux must be boolean")
    return {} if not enabled else {"shape_supervision": {
        "schema": "equal-task-sumnorm-shape-aux-v1", "weight": 1.,
        "formula": "0.5*mean(E_edos+S_edos) + 0.5*mean(E_phdos+S_phdos)",
        "shape_loss": "KL + W1 + Huber", "w_w1": 1., "w_huber": 1., "huber_delta": .02,
        "zero_target": "zero auxiliary loss; retain blind energy; average over all materials",
        "checkpoint_metric_includes_auxiliary": False,
    }}


def legacy_balance_config(cfg):
    from utils.experiment_config import ExperimentConfig

    # Legacy factories compare this path as a string. The new protocol checks
    # the actual manifest/payload hashes independently, so equivalent absolute
    # paths need not be rejected by the historical recipe's spelling.
    return replace(cfg, spectral_balance_config="", spectral_shape_aux=False, split_decoder=False,
                   scale_mode="eta", selection_metric="balanced_blind_r2",
                   data_dir=ExperimentConfig().data_dir)


def validate_balance_config(cfg):
    from utils.local_global_experiment import validate_local_global_config

    if (not cfg.spectral_balance_config or cfg.local_route != "I3"
            or cfg.scale_mode != "per_atom" or cfg.selection_metric != ENERGY_METRIC):
        raise ValueError("balanced spectra require I3/A64, per_atom scale and balanced_blind_energy")
    if cfg.split_decoder and not cfg.spectral_shape_aux:
        raise ValueError("split decoder requires the fixed shape auxiliary on the current recipe")
    shape_supervision_contract(cfg.spectral_shape_aux)
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
    base_params = {**params, "scale_mode": "eta", "decoupled_decoder": False}
    base = comparison_contract(legacy_balance_config(cfg), base_params)
    contract = {"schema": "joint-spectral-balance-model-v1", "init_protocol": "joint-spectral-balance-init-v1",
            "seed": cfg.seed, "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": ENERGY_METRIC, **energy_selection_contract(protocol),
            "spectral_balance": protocol,
            "spectral_balance_config_sha256": file_sha256(input_path(cfg.spectral_balance_config)),
            "training": training_record(cfg), "backbone_reference": base,
            "scope": "scale parameterization, blind objective and checkpoint protocol package; "
                     "complete A64 encoder/decoder unchanged; not a single-factor encoder test"}
    if cfg.spectral_shape_aux:
        contract.update(schema="joint-spectral-balance-shape-model-v1",
                        scope="fixed shape auxiliary added to the completed blind-energy control; "
                              "same A64 tensors, mass reconstruction, data, budget and checkpoint rule",
                        **shape_supervision_contract(True))
    if cfg.split_decoder:
        if not params.get("decoupled_decoder"):
            raise ValueError("split decoder candidate must construct two task decoders")
        contract.update(schema="joint-spectral-balance-split-decoder-v1",
                        scope="two full 6-layer decoders copied from one shared initial decoder; "
                              "queries, heads, encoder, loss, data and checkpoint rule unchanged; "
                              "extra parameters are the second decoder and are not capacity-matched",
                        **shape_supervision_contract(True))
    elif params.get("decoupled_decoder"):
        raise ValueError("decoupled decoder weights require the split decoder candidate")
    return contract


def _install_split_decoder(shared_state, target):
    """Copy one shared decoder into both task decoders. Other tensors stay put."""
    expected = target.state_dict()
    mapped = {}
    for key, value in shared_state.items():
        destinations = (f"edos_decoder.{key[len('decoder.'):]}", f"phdos_decoder.{key[len('decoder.'):]}") \
            if key.startswith("decoder.") else (key,)
        for dest in destinations:
            if dest not in expected or expected[dest].shape != value.shape:
                raise ValueError(f"split decoder tensor does not match the shared decoder: {dest}")
            mapped[dest] = value.detach().clone()
    if set(mapped) != set(expected):
        raise ValueError("split decoder leaves tensors outside the shared initialization")
    return mapped


def initialize_balance_model(transformer, params, seed):
    """Preserve A64 initial tensors; map eta MLP weights to the raw mass MLP."""
    from model.transformer import Transformer
    from utils.i1_a_experiment import initialize_i1_a_model, validate_i3_model
    from utils.zproj_alignment import state_hash

    if params.get("scale_mode") != "per_atom" or params.get("local_route") != "I3":
        raise ValueError("balanced initialization requires per_atom complete A64")
    decoupled = bool(params.get("decoupled_decoder"))
    base_params = {**params, "scale_mode": "eta", "decoupled_decoder": False}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**base_params)
        info = initialize_i1_a_model(reference, base_params, seed)
    state = reference.state_dict()
    mapped = {key.replace("eta_head.", "mass_head.", 1) if key.startswith("eta_head.") else key: value
              for key, value in state.items()}
    if decoupled:
        mapped = _install_split_decoder(mapped, transformer)
    expected = transformer.state_dict()
    if expected.keys() != mapped.keys() or any(expected[key].shape != mapped[key].shape for key in expected):
        raise ValueError("balanced mass head must preserve A64 MLP tensor shapes")
    transformer.load_state_dict(mapped, strict=True)
    validate_i3_model(transformer, params, require_identity=True)
    actual = transformer.state_dict()
    if any(not torch.equal(actual[key], value.to(actual[key])) for key, value in mapped.items()):
        raise ValueError("balanced initialization did not preserve mapped initial tensors")
    if decoupled:
        for key, value in actual.items():
            if not key.startswith("edos_decoder."):
                continue
            other = actual["phdos_decoder." + key[len("edos_decoder."):]]
            if not torch.equal(value, other):
                raise ValueError("task decoders diverged at initialization")
    return {"protocol": "joint-spectral-balance-init-v1", "seed": seed,
            "a64_initialization": info, "state_sha256": state_hash(actual),
            "mass_head_mapping": "eta_head.mlp -> mass_head.mlp; raw e/p mass logits replace bounded ph/e ratios",
            "backbone_initial_tensors_identical": True,
            "split_decoder": decoupled,
            "decoder_copies_identical_at_init": decoupled,
            "update_counts": transformer.encoder.update_counts}
