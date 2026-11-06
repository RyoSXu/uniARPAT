"""Balanced-spectrum protocols and isolated candidates on complete A64."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from utils.paths import file_sha256, input_path
from utils.spectral_balance import ENERGY_METRIC, load_balance_protocol

BALANCE_RESIDUAL_SCHEMA = "joint-spectral-balance-residual-model-v1"
STRUCTURE_BLIND_SCHEMA = "joint-spectral-balance-structure-blind-v1"
REFERENCE_CONTROL_SCHEMA = "lr2-es-reference-control-v1"
PHDOS_HEAD_SCHEMA = "lr2-es-phdos-head-depth-v1"
PHDOS_HEAD_DESIGN = "docs/design/design-2026-11-06-phdos-head.md"
PHDOS_HEAD_DEPTHS = (1, 2)


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
    # Reference controls validate the frozen seed-42 structural recipe; their
    # own seed, data order, route and supervision are recorded in the contract.
    structure_blind = cfg.local_route == "S0"
    return replace(cfg, local_route="I3" if cfg.local_route in ("LR2", "S0") else cfg.local_route,
                   local_depth=2 if structure_blind else cfg.local_depth,
                   seed=42, data_order_seed=None, single_task=None, strain_augment=False, local_drop_prob=None, train_subset="",
                   phdos_head_layers=None,
                   local_layout="staged" if cfg.local_layout == "midpoint" else cfg.local_layout,
                   spectral_balance_config="", spectral_shape_aux=False, split_decoder=False,
                   phonon_head=False, phonon_correction=False, scale_mode="eta",
                   selection_metric="balanced_blind_r2", data_dir=ExperimentConfig().data_dir)


def _backbone_params(params):
    """A64 reference parameters without the opt-in readout changes."""
    base = {**params, "scale_mode": "eta", "decoupled_decoder": False}
    if base.get("local_route") == "S0":
        base["local_depth"] = 2
    if base.get("local_route") in ("LR2", "S0"):
        base["local_route"] = "I3"
    if base.get("local_layout") == "midpoint":
        base["local_layout"] = "staged"
    base.pop("phonon_head", None)
    base.pop("phonon_correction", None)
    base.pop("local_drop_prob", None)
    base.pop("phdos_head_layers", None)
    return base


def _final_phdos_conv(state):
    """Keys of the last convolution in the original phDOS readout."""
    indices = {int(key.split(".")[2]) for key in state
               if key.startswith("phdos_out_head.layers.") and key.endswith(".weight")}
    if not indices:
        raise ValueError("phonon correction needs the original convolutional phDOS readout")
    last = max(indices)
    return f"phdos_out_head.layers.{last}.weight", f"phdos_out_head.layers.{last}.bias"


def reference_control_kind(cfg):
    """None for every historical trial; otherwise the LR2 E+S reference control."""
    if cfg.local_route == "S0":
        return "structure_blind"
    if cfg.single_task is not None:
        return f"{cfg.single_task}_only"
    if cfg.seed != 42 or cfg.data_order_seed is not None:
        return "seed_replicate"
    return None


def expected_selection_metric(cfg):
    kind = reference_control_kind(cfg)
    if kind == "structure_blind":
        return "unguarded_blind_energy"
    if kind in ("edos_only", "phdos_only"):
        return f"{cfg.single_task}_blind_energy"
    return ENERGY_METRIC


def validate_reference_control(cfg):
    kind = reference_control_kind(cfg)
    if kind is None:
        return None
    if (cfg.local_route not in ("LR2", "S0") or cfg.local_layout != "staged" or not cfg.spectral_shape_aux
            or cfg.split_decoder or cfg.phonon_head or cfg.phonon_correction):
        raise ValueError("reference controls apply only to the adopted staged LR2 E+S recipe")
    if cfg.single_task is not None and (cfg.single_task not in ("edos", "phdos") or cfg.local_route != "LR2"):
        raise ValueError("single-task controls keep the full LR2 architecture")
    if cfg.local_route == "S0" and (cfg.local_depth != 0 or cfg.global_depth != 6):
        raise ValueError("structure-blind S0 requires zero local calls and the six corrected global layers")
    for name in ("seed", "data_order_seed"):
        value = getattr(cfg, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    return kind


def regularization_kind(cfg):
    """None for every historical trial; otherwise the 2026-11-05 generalization arm."""
    if cfg.strain_augment and cfg.local_drop_prob is not None:
        raise ValueError("geometry perturbation and local drop are separate arms")
    return "strain_augment" if cfg.strain_augment else "local_drop" if cfg.local_drop_prob is not None else None


def validate_regularization(cfg):
    from utils.structure_regularization import LOCAL_DROP_PROB

    kind = regularization_kind(cfg)
    if kind is None:
        return None
    if (cfg.local_route != "LR2" or cfg.local_layout != "staged" or not cfg.spectral_shape_aux
            or cfg.split_decoder or cfg.phonon_head or cfg.phonon_correction or cfg.single_task is not None
            or cfg.augment or cfg.selection_metric != ENERGY_METRIC):
        raise ValueError("generalization arms apply only to the adopted staged LR2 E+S recipe")
    if kind == "local_drop" and cfg.local_drop_prob != LOCAL_DROP_PROB:
        raise ValueError(f"local drop probability is fixed at {LOCAL_DROP_PROB}; no scan")
    return kind


def validate_learning_curve(cfg):
    """The learning-curve points change only the train rows of the seed-42 staged LR2 E+S recipe."""
    if not cfg.train_subset:
        return None
    if (cfg.local_route != "LR2" or cfg.local_layout != "staged" or not cfg.spectral_shape_aux
            or cfg.split_decoder or cfg.phonon_head or cfg.phonon_correction or cfg.single_task is not None
            or cfg.augment or regularization_kind(cfg) is not None or reference_control_kind(cfg) is not None
            or cfg.selection_metric != ENERGY_METRIC):
        raise ValueError("learning-curve subsets apply only to the adopted seed-42 staged LR2 E+S recipe")
    from utils.train_subset import load_train_subset

    return load_train_subset(input_path(cfg.train_subset), input_path(cfg.data_dir))


def validate_phdos_head(cfg):
    """The head study changes only the phDOS readout depth of the seed-42 staged LR2 E+S recipe."""
    if cfg.phdos_head_layers is None:
        return None
    if cfg.phdos_head_layers not in PHDOS_HEAD_DEPTHS:
        raise ValueError(f"phDOS head depth must be one of {PHDOS_HEAD_DEPTHS}; the historical 6 is the default")
    if (cfg.local_route != "LR2" or cfg.local_layout != "staged" or not cfg.spectral_shape_aux
            or cfg.split_decoder or cfg.phonon_head or cfg.phonon_correction or cfg.single_task is not None
            or cfg.augment or regularization_kind(cfg) is not None or reference_control_kind(cfg) is not None
            or cfg.train_subset or cfg.selection_metric != ENERGY_METRIC):
        raise ValueError("the phDOS head study applies only to the adopted seed-42 staged LR2 E+S recipe")
    return cfg.phdos_head_layers


def phdos_head_state(d_model, layers, seed):
    """Seeded independent head tensors, initialized as the Transformer constructor does."""
    from model.heads import CNN

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = CNN(d_model, d_model * 3, output_dim=1, num_layers=layers)
        for parameter in head.parameters():
            if parameter.dim() > 1:
                torch.nn.init.xavier_normal_(parameter)
    return {f"phdos_out_head.{key}": value for key, value in head.state_dict().items()}


def validate_balance_config(cfg):
    from utils.local_global_experiment import validate_local_global_config

    validate_reference_control(cfg)
    validate_regularization(cfg)
    validate_learning_curve(cfg)
    validate_phdos_head(cfg)
    if (not cfg.spectral_balance_config or cfg.local_route not in ("I3", "LR2", "S0")
            or cfg.scale_mode != "per_atom" or cfg.selection_metric != expected_selection_metric(cfg)):
        raise ValueError("balanced spectra require I3/A64, LR2 or S0, per_atom scale and the matching blind-energy selection")
    if cfg.local_layout == "midpoint" and cfg.local_route != "LR2":
        raise ValueError("midpoint changes scheduling only on the fixed LR2 E+S recipe")
    if cfg.local_route in ("LR2", "S0") and (not cfg.spectral_shape_aux or cfg.split_decoder
                                    or cfg.phonon_head or cfg.phonon_correction):
        raise ValueError("balanced LR2 changes only internal additions on the shared-decoder E+S recipe")
    if cfg.split_decoder and not cfg.spectral_shape_aux:
        raise ValueError("split decoder requires the fixed shape auxiliary on the current recipe")
    if cfg.phonon_head and (not cfg.spectral_shape_aux or cfg.split_decoder):
        raise ValueError("phonon head is a single candidate on the shape-auxiliary recipe")
    if cfg.phonon_correction and not cfg.phonon_head:
        raise ValueError("phonon correction needs the phonon head")
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
    from utils.checkpoint_selection import selection_contract
    from utils.local_global_experiment import comparison_contract, training_record

    protocol = validate_balance_config(cfg)
    if params.get("local_route") != cfg.local_route:
        raise ValueError("balanced configuration and transformer routes disagree")
    if params.get("local_layout") != cfg.local_layout:
        raise ValueError("balanced configuration and transformer layouts disagree")
    base = comparison_contract(legacy_balance_config(cfg), _backbone_params(params))
    contract = {"schema": "joint-spectral-balance-model-v1", "init_protocol": "joint-spectral-balance-init-v1",
            "seed": cfg.seed, "transformer": deepcopy(params), "normalization": cfg.norm,
            "selection_metric": cfg.selection_metric, **selection_contract(cfg.selection_metric, protocol),
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
    if (cfg.phonon_head != bool(params.get("phonon_head"))
            or cfg.phonon_correction != bool(params.get("phonon_correction"))):
        raise ValueError("phonon head flag and model parameters disagree")
    if cfg.phonon_head:
        contract.update(schema="joint-spectral-balance-phonon-head-v1",
                        scope="phDOS shape from learned Born-von Karman springs and exact harmonic "
                              "lattice dynamics; total stays with the per-atom mass head; eDOS path, "
                              "encoder, loss, data and checkpoint rule unchanged",
                        phonon_head={"schema": "learned-spring-harmonic-dos-v1", "radius": 6.0,
                                     "spring_form": "k_par r r^T + k_perp (I - r r^T); quintic cutoff",
                                     "acoustic_sum_rule": "on-site block equals neighbor sum",
                                     "mesh": "shifted Monkhorst-Pack, ceil(20 A / plane spacing), <= 512 points",
                                     "broadening": "one learned Gaussian width, initial 1 bin, floor 0.25 bin",
                                     "masses": "pymatgen standard atomic masses",
                                     "output": "in-window modes per 3N, floored, as logits"},
                        **shape_supervision_contract(True))
    if cfg.phonon_correction:
        contract["phonon_head"]["correction"] = ("original phDOS decoder and convolutional readout added to the "
                                                 "physics logits; its final convolution starts at zero")
        contract.update(schema="joint-spectral-balance-phonon-prior-v1",
                        scope="phDOS shape = physics prior from learned springs times a learned correction; "
                              "start equals the physics shape; total, eDOS path, encoder, loss, data and "
                              "checkpoint rule unchanged")
    if cfg.local_route == "LR2":
        from utils.local_residual_experiment import residual_ablation_contract

        contract.update(schema=BALANCE_RESIDUAL_SCHEMA, ablation=residual_ablation_contract(),
                        scope="four internal scalar additions removed from complete A64 E+S; "
                              "identical untrained tensors, outer residual, data, joint loss, "
                              "shared decoder, budget and checkpoint guards")
        if cfg.local_layout == "midpoint":
            from utils.spectral_layout_experiment import MIDPOINT_SCHEMA, layout_ablation_contract

            contract.update(schema=MIDPOINT_SCHEMA, ablation=layout_ablation_contract(),
                            scope="only second-local-call placement differs from staged LR2 E+S; "
                                  "L1 G1 G2 G3 L2 G4 G5 G6; all modules, initial tensors, "
                                  "loss, data, budget and checkpoint guards unchanged")
    if cfg.local_route == "S0":
        contract.update(schema=STRUCTURE_BLIND_SCHEMA,
                        scope="no local update and no geometry; the six corrected global layers, ZP, "
                              "shared decoder, per-atom total head, E+S loss and data unchanged; "
                              "removing the local branch also removes its parameters")
    regularization = regularization_kind(cfg)
    if (params.get("local_drop_prob") is not None) != (cfg.local_drop_prob is not None) or (
            cfg.local_drop_prob is not None and params.get("local_drop_prob") != cfg.local_drop_prob):
        raise ValueError("local drop flag and model parameters disagree")
    if regularization is not None:
        from utils.structure_regularization import STRAIN_AUGMENT_CONTRACT, local_drop_contract

        contract["regularization"] = (deepcopy(STRAIN_AUGMENT_CONTRACT) if regularization == "strain_augment"
                                      else local_drop_contract(cfg.local_drop_prob))
        contract["regularization"].update(
            kind=regularization, design="docs/design/design-2026-11-05-structure-generalization.md",
            scope="training-only regularizer on staged LR2 E+S; architecture, initial tensors, loss, data, "
                  "budget, evaluation path and checkpoint guards unchanged")
    subset = validate_learning_curve(cfg)
    if subset is not None:
        from utils.train_subset import subset_contract

        contract["train_subset"] = subset_contract(subset)
    depth = validate_phdos_head(cfg)
    if params.get("phdos_head_layers") != depth:
        raise ValueError("phDOS head depth flag and model parameters disagree")
    if depth is not None:
        contract["phdos_head"] = {
            "schema": PHDOS_HEAD_SCHEMA, "design": PHDOS_HEAD_DESIGN, "layers": depth, "reference_layers": 6,
            "hidden_channels": 3 * int(params.get("d_model", 512)), "kernel_size": 3,
            "initialization": "seeded independent head; every other tensor copied from the A64/LR2 initial state",
            "scope": "phDOS readout depth only; encoder, shared decoder, queries, eDOS head, mass head, loss, "
                     "data, budget, evaluation path and checkpoint guards unchanged"}
    kind = reference_control_kind(cfg)
    if kind is not None:
        contract["reference_control"] = {
            "schema": REFERENCE_CONTROL_SCHEMA, "kind": kind, "seed": cfg.seed,
            "data_order_seed": 0 if cfg.data_order_seed is None else cfg.data_order_seed,
            "supervised_tasks": [cfg.single_task] if cfg.single_task else ["edos", "phdos"],
            "loss": ("mean(E_t+S_t) for the supervised task; other task coefficient 0"
                     if cfg.single_task else "0.5*mean(E_edos+S_edos) + 0.5*mean(E_phdos+S_phdos)"),
            "backbone_reference": "validated against the frozen seed-42 staged recipe",
            "selection": "guards applied" if cfg.selection_metric == ENERGY_METRIC
                         else "own blind objective; guards recorded, not applied"}
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


LOCAL_PREFIX = "encoder.local_layers."


def initialize_structure_blind_model(transformer, params, seed):
    """S0 copies every non-local tensor of the same-seed untrained LR2 E+S model."""
    from model.transformer import Transformer
    from utils.zproj_alignment import state_hash

    if (params.get("local_route") != "S0" or params.get("local_depth") != 0
            or params.get("global_depth") != 6 or params.get("local_layout") != "staged"
            or any(params.get(key) for key in ("decoupled_decoder", "phonon_head", "phonon_correction"))):
        raise ValueError("structure-blind initialization requires S0, staged K0/G6 and the shared decoder")
    reference_params = {**params, "local_route": "LR2", "local_depth": 2}
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**reference_params)
        reference_info = initialize_balance_model(reference, reference_params, seed)
    shared = {key: value for key, value in reference.state_dict().items() if not key.startswith(LOCAL_PREFIX)}
    target = transformer.state_dict()
    if target.keys() != shared.keys() or any(target[key].shape != shared[key].shape for key in target):
        raise ValueError("S0 must equal the LR2 reference without its local layers")
    transformer.load_state_dict(shared, strict=True)
    current = transformer.state_dict()
    if any(not torch.equal(current[key], value.to(current[key])) for key, value in shared.items()):
        raise ValueError("S0 shared tensors were not copied verbatim")
    encoder = transformer.encoder
    if encoder.local_layers or len(encoder.global_layers) != 6:
        raise ValueError("S0 must have no local layers and six global layers")
    for layer in encoder.global_layers:
        attn = layer.attn
        eye = torch.eye(attn.embed_dim, dtype=attn.in_proj_weight.dtype)
        if (layer.norm_first or attn.dropout != 0.0 or not torch.equal(attn.in_proj_weight, eye.repeat(3, 1))
                or not torch.equal(attn.out_proj.weight, eye) or torch.count_nonzero(attn.in_proj_bias)
                or torch.count_nonzero(attn.out_proj.bias)):
            raise ValueError("S0 must keep post-norm, zero attention dropout and identity Q/K/V/O")
    return {"protocol": "lr2-es-structure-blind-init-v1", "seed": seed, "route": "S0",
            "lr2_initialization": reference_info, "reference_state_sha256": state_hash(reference.state_dict()),
            "state_sha256": state_hash(current), "n_shared_tensors": len(shared),
            "parameter_count": sum(p.numel() for p in transformer.parameters()),
            "reference_parameter_count": sum(p.numel() for p in reference.parameters()),
            "update_counts": encoder.update_counts,
            "execution_order": [list(step) for step in encoder.execution_order]}


def initialize_balance_model(transformer, params, seed):
    """Preserve A64 initial tensors; map eta MLP weights to the raw mass MLP."""
    if params.get("local_route") == "S0":
        return initialize_structure_blind_model(transformer, params, seed)
    from model.transformer import Transformer
    from utils.i1_a_experiment import initialize_i1_a_model, validate_i3_model
    from utils.zproj_alignment import state_hash

    if params.get("scale_mode") != "per_atom" or params.get("local_route") not in ("I3", "LR2"):
        raise ValueError("balanced initialization requires per_atom complete A64")
    residual = params.get("local_route") == "LR2"
    if residual and any(params.get(key) for key in ("decoupled_decoder", "phonon_head", "phonon_correction")):
        raise ValueError("balanced LR2 cannot combine untested decoder or phonon candidates")
    decoupled = bool(params.get("decoupled_decoder"))
    phonon = bool(params.get("phonon_head"))
    base_params = _backbone_params(params)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = Transformer(**base_params)
        info = initialize_i1_a_model(reference, base_params, seed)
    state = reference.state_dict()
    mapped = {key.replace("eta_head.", "mass_head.", 1) if key.startswith("eta_head.") else key: value
              for key, value in state.items()}
    if decoupled:
        mapped = _install_split_decoder(mapped, transformer)
    if phonon:
        from model.local_global_encoder import LOCAL_RADIUS
        from model.phonon_head import PhononSpringHead

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = PhononSpringHead(transformer.d_model, LOCAL_RADIUS)
        mapped.update({f"phonon_head.{key}": value for key, value in head.state_dict().items()})
    depth = params.get("phdos_head_layers")
    if depth is not None:
        mapped = {key: value for key, value in mapped.items() if not key.startswith("phdos_out_head.")}
        mapped.update(phdos_head_state(transformer.d_model, depth, seed))
    correction = bool(params.get("phonon_correction"))
    if correction:
        for key in _final_phdos_conv(mapped):
            mapped[key] = torch.zeros_like(mapped[key])
    expected = transformer.state_dict()
    if expected.keys() != mapped.keys() or any(expected[key].shape != mapped[key].shape for key in expected):
        raise ValueError("balanced mass head must preserve A64 MLP tensor shapes")
    transformer.load_state_dict(mapped, strict=True)
    if residual:
        if params.get("local_layout") == "midpoint":
            from utils.spectral_layout_experiment import validate_midpoint_model as validate_residual_model
        else:
            from utils.local_residual_experiment import validate_residual_model

        validate_residual_model(transformer, params, require_identity=True)
    else:
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
            **({"internal_residual_add": False, "route": "LR2"} if residual else {}),
            **({"local_layout": "midpoint", "execution_order": [list(step) for step in transformer.encoder.execution_order]}
               if params.get("local_layout") == "midpoint" else {}),
            **({"phonon_head": "seeded independent initialization; all A64 tensors copied"} if phonon else {}),
            **({"phdos_head": {"layers": depth, "state_sha256": state_hash(
                {k: v for k, v in actual.items() if k.startswith("phdos_out_head.")}),
                "parameters": sum(v.numel() for k, v in actual.items() if k.startswith("phdos_out_head."))}}
               if depth is not None else {}),
            **({"phonon_correction": "final phDOS convolution zeroed; start equals the physics shape"}
               if correction else {}),
            "update_counts": transformer.encoder.update_counts}
