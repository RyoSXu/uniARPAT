import os
import time
import argparse
import hashlib
import json
import random
import yaml
import torch
import numpy as np
import pandas as pd
import logging
from utils.builder import ConfigBuilder
from model.model import basemodel
from utils.experiment_config import ExperimentConfig
from utils.ablation_checkpoint import (
    atomic_torch_save, build_ablation_checkpoint, restore_ablation_checkpoint,
)
from utils.pair_aux_batches import (
    PairPlanBatchSampler, build_pair_universe, schedule_auxiliary_batches,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('ablation')


def setup_ablation_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducible comparisons.

    Deterministic cuDNN can reduce throughput, but reproducibility is more
    important than speed for an ablation experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


G2_CONTENT_MODES = ("radial", "joint")
PAIR_B7_CHECKPOINT = os.path.abspath("./output/ablation_m1_e9ctl/checkpoint_best.pth")
PAIR_B7_CHECKPOINT_SHA256 = "cbbf94f227c9a3b4802c09bad7c1058ea015283b0acbd868831a829368d9ad40"


def load_joint_initial_state(transformer, state):
    """Accept complete joint weights or a complete B7 backbone, never partial G2.

    Check keys and shapes before copying any weights. This is only the explicit
    joint initialization path; ordinary checkpoint restoration stays strict.
    """
    expected = transformer.state_dict()
    branch = {key for key in expected if key.startswith("encoder.g2_msgs.")}
    if not branch or transformer.g2_content_mode != "joint":
        raise ValueError("joint initialization requires a joint G2 model")
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    if unexpected or (missing and missing != branch):
        raise ValueError(
            "joint init requires a complete joint state or B7 backbone; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    bad_shapes = [key for key in state
                  if not isinstance(state[key], torch.Tensor)
                  or state[key].shape != expected[key].shape]
    if bad_shapes:
        raise ValueError(f"joint init has incompatible tensors: {sorted(bad_shapes)}")
    return transformer.load_state_dict(state, strict=not missing)


def validate_g2_config(cfg: ExperimentConfig) -> None:
    """Reject illegal G2 combinations before any directory or data access.

    Candidate-1 contract: ``g2_content_mode="joint"`` must be explicit and run
    together with ``--use_g2``; G2 never combines with G1. Silent fallback to
    radial is forbidden, so invalid combinations raise here instead of being
    ignored after artifacts and data loaders already exist.
    """
    if cfg.g2_content_mode not in G2_CONTENT_MODES:
        raise ValueError(
            f"g2_content_mode must be one of {G2_CONTENT_MODES}, got {cfg.g2_content_mode!r}")
    if cfg.g2_content_mode == "joint" and not cfg.use_g2:
        raise ValueError(
            "g2_content_mode='joint' requires --use_g2; silent fallback to radial is forbidden")
    if cfg.use_g1 and cfg.use_g2:
        raise ValueError(
            "G2 is a single-factor module on B7; --use_g1 and --use_g2 are mutually exclusive")


def validate_reset_rng_config(cfg: ExperimentConfig) -> None:
    """Reject ``--reset_rng_after_init`` without an initialization checkpoint.

    Resetting the RNG only has meaning after an init checkpoint has been
    consumed. The check runs alongside ``validate_g2_config`` so an unusable
    flag fails before any directory, config template, or data loader exists.
    """
    if cfg.reset_rng_after_init and not cfg.init_ckpt:
        raise ValueError("--reset_rng_after_init requires --init_ckpt")


def validate_pair_aux_config(cfg: ExperimentConfig) -> None:
    """Keep the pair auxiliary path opt-in and isolated to its frozen recipe."""
    if cfg.pair_aux_arm not in {"none", "control", "candidate"}:
        raise ValueError("pair_aux_arm must be none, control, or candidate")
    if not np.isfinite(cfg.pair_ratio) or cfg.pair_ratio < 0:
        raise ValueError("pair_ratio must be finite and nonnegative")
    if cfg.pair_aux_arm == "none":
        if cfg.pair_ratio != 0:
            raise ValueError("pair_ratio must be zero when pair auxiliary mode is disabled")
        return
    if cfg.pair_ratio != 0.10:
        raise ValueError("the frozen pair auxiliary design requires pair_ratio=0.10")
    if not cfg.skip_test_eval:
        raise ValueError("pair auxiliary runs require --skip_test_eval")
    if os.path.abspath(cfg.init_ckpt) != PAIR_B7_CHECKPOINT or not cfg.reset_rng_after_init:
        raise ValueError("pair auxiliary runs require the frozen B7 checkpoint and RNG reset")
    expected_tag = "_pcctl" if cfg.pair_aux_arm == "control" else "_pcaux"
    if cfg.tag != expected_tag:
        raise ValueError(f"{cfg.pair_aux_arm} pair arm requires tag {expected_tag}")
    frozen_values = {
        "model_name": (cfg.model_name, "M1"),
        "epochs": (cfg.epochs, 10),
        "batch_size": (cfg.batch_size, 32),
        "lr": (cfg.lr, 5e-5),
        "seed": (cfg.seed, 42),
        "data_dir": (os.path.abspath(cfg.data_dir), os.path.abspath("./data/train4ARPAT")),
        "decoder_layers": (cfg.decoder_layers, 6),
        "atom_feat": (cfg.atom_feat, "legacy3"),
        "energy_code": (cfg.energy_code, "none"),
        "norm": (cfg.norm, "sumnorm"),
        "scale_mode": (cfg.scale_mode, "eta"),
        "scalar_mode": (cfg.scalar_mode, "none"),
        "tv_w": (cfg.tv_w, 0.0),
        "grad_w": (cfg.grad_w, 0.0),
        "peak_w": (cfg.peak_w, 1.0),
        "tail_w": (cfg.tail_w, 1.0),
        "tail_start": (cfg.tail_start, -1),
        "eta_sup_w": (cfg.eta_sup_w, 1.0),
        "scale_sup_w": (cfg.scale_sup_w, 1.0),
    }
    mismatches = {
        name: values for name, values in frozen_values.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"pair auxiliary frozen recipe mismatch: {mismatches}")
    optional_defaults = {
        "dropout": (cfg.dropout, {None, 0.05}),
        "warmup_epochs": (cfg.warmup_epochs, {None, 5}),
        "weight_decay": (cfg.weight_decay, {None}),
        "lambda_ph": (cfg.lambda_ph, {None, 1.0}),
        "grad_clip": (cfg.grad_clip, {None, 0.0}),
        "w_w1": (cfg.w_w1, {None, 1.0}),
        "w_huber": (cfg.w_huber, {None, 1.0}),
    }
    bad_optional = sorted(
        name for name, (value, allowed) in optional_defaults.items() if value not in allowed
    )
    if bad_optional:
        raise ValueError(f"pair auxiliary frozen optional settings differ: {bad_optional}")
    conflicts = {
        "use_amp": cfg.use_amp,
        "use_bucket_batch": cfg.use_bucket_batch,
        "augment": cfg.augment,
        "use_mask": cfg.use_mask,
        "freeze_backbone": cfg.freeze_backbone,
        "use_g1": cfg.use_g1,
        "use_g2": cfg.use_g2,
        "q1_coord": cfg.q1_coord,
        "q2_fourier": cfg.q2_fourier,
        "c5_moe": cfg.c5_moe,
        "r1a_point": cfg.r1a_point,
        "r1b_coord": cfg.r1b_coord,
        "use_macro_lattice": cfg.use_macro_lattice,
        "use_atom_additive_phdos": cfg.use_atom_additive_phdos,
        "edos_slope_ratio": cfg.edos_slope_ratio != 0,
    }
    enabled = sorted(name for name, value in conflicts.items() if value)
    if enabled:
        raise ValueError(f"pair auxiliary is a single-factor experiment; conflicts={enabled}")


def validate_pair_aux_init_checkpoint(cfg: ExperimentConfig) -> None:
    """Verify the exact B7 epoch-33 initialization before creating artifacts."""
    if cfg.pair_aux_arm == "none":
        return
    digest = hashlib.sha256()
    with open(cfg.init_ckpt, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != PAIR_B7_CHECKPOINT_SHA256:
        raise ValueError("pair auxiliary init checkpoint SHA-256 mismatch")
    checkpoint = torch.load(cfg.init_ckpt, map_location="cpu", weights_only=True)
    identity = (checkpoint.get("epoch"), checkpoint.get("model_name"), checkpoint.get("seed"))
    if identity != (33, "M1", 42) or checkpoint.get("use_amp", False):
        raise ValueError(f"pair auxiliary requires the FP32 B7 epoch-33 checkpoint: {identity}")


MODEL_CONFIGS = {
    'M1': {
        'desc': 'Shared decoder with asymmetric eDOS and phDOS output heads.',
        'transformer_params': {
            'decoupled_decoder': False,
            'use_gated_cross_attn': False,
            'head_type': 'legacy',
            'predict_scale': False
        }
    },
    'M2': {
        'desc': 'Historical decoupled decoder with a smaller phDOS head.',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': False,
            'head_type': 'ph_trimmed',
            'predict_scale': False
        }
    },
    'M3': {
        'desc': 'Historical decoupled decoder with symmetric output heads.',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': False,
            'head_type': 'symmetric',
            'predict_scale': False
        }
    },
    'M4': {
        'desc': 'Historical decoupled decoder with gated cross-attention.',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': True,
            'head_type': 'symmetric',
            'predict_scale': False
        }
    },
    'M5': {
        'desc': 'Historical shape-and-scale variant with a learned scale head.',
        'transformer_params': {
            'decoupled_decoder': True,
            'use_gated_cross_attn': True,
            'head_type': 'symmetric',
            'predict_scale': True
        }
    }
}

def _resolve_edos_grid(spec: str, edos_num: int):
    """Resolve an eDOS grid name or a file of bin centers to a Python list."""
    import os as _os
    if _os.path.exists(spec):
        return np.load(spec).tolist()
    with open('./data/grids_c2b/grids.json') as f:
        grids = json.load(f)
    if spec in grids:
        e = np.asarray(grids[spec], dtype=float)
        assert len(e) - 1 == edos_num, f"{spec} bins {len(e)-1} != {edos_num}"
        return ((e[:-1] + e[1:]) / 2).tolist()
    raise ValueError(f"unknown edos_grid spec: {spec}")


def _ph_grid_centers(phdos_num: int):
    """Return phDOS bin centers matching the requested output length, if known."""
    try:
        with open('./data/grids_c2b/grids.json') as f:
            grids = json.load(f)
    except Exception:
        return None
    for arm in ("P0", "P1", "P2"):
        e = np.asarray(grids.get(arm, []), dtype=float)
        if len(e) - 1 == phdos_num:
            return ((e[:-1] + e[1:]) / 2).tolist()
    return None


def maybe_get_test_loader(builder, cfg, sumnorm):
    if cfg.skip_test_eval:
        return None
    return builder.get_dataloader(
        split="test",
        dos_minmax=True,
        batch_size=cfg.batch_size,
        dos_sumnorm=sumnorm,
    )


def write_edos_slope_calibration(config_path, calibration):
    with open(config_path) as stream:
        config_used = yaml.safe_load(stream)
    runtime = config_used.setdefault("runtime", {})
    runtime.update({
        "edos_slope_ratio": calibration["target_gradient_ratio"],
        "edos_slope_lambda": calibration["lambda"],
        "edos_slope_base_gradient_norm": calibration["base_gradient_norm"],
        "edos_slope_gradient_norm": calibration["slope_gradient_norm"],
    })
    with open(config_path, "w") as stream:
        yaml.safe_dump(config_used, stream, sort_keys=False)


def write_pair_aux_calibration(config_path, calibration, frozen_plan_hash, epoch_plan_hash):
    with open(config_path) as stream:
        config_used = yaml.safe_load(stream)
    runtime = config_used.setdefault("runtime", {})
    runtime.update({
        "pair_ratio": calibration["target_gradient_ratio"],
        "pair_lambda": calibration["lambda"],
        "pair_base_gradient_norm": calibration["base_gradient_norm"],
        "pair_gradient_norm": calibration["pair_gradient_norm"],
        "pair_plan_hash": frozen_plan_hash,
        "pair_epoch_plan_hash": epoch_plan_hash,
    })
    with open(config_path, "w") as stream:
        yaml.safe_dump(config_used, stream, sort_keys=False)


def maybe_get_pair_aux_loader(train_loader, cfg):
    """Build the label-free plan only for an explicitly active pair arm."""
    if cfg.pair_aux_arm == "none":
        return None, None
    from torch.utils.data import DataLoader
    index_path = os.path.join(cfg.data_dir, "train", "train_index.npy")
    sample_ids = np.load(index_path)
    universe = build_pair_universe(train_loader.dataset.elements, sample_ids)
    sampler = PairPlanBatchSampler(universe, cfg.seed, pairs_per_batch=16)
    loader = DataLoader(
        train_loader.dataset,
        batch_sampler=sampler,
        collate_fn=train_loader.collate_fn,
        num_workers=0,
        pin_memory=train_loader.pin_memory,
    )
    return loader, sampler


def train_and_eval(cfg: ExperimentConfig):
    if cfg.model_name not in MODEL_CONFIGS :
        raise ValueError (f"Unknown model name: {cfg.model_name }. Available: {list (MODEL_CONFIGS .keys ())}")
    # Explicit early rejection of illegal G2 combinations: before save_dir is
    # created and before any data or config template is loaded.
    validate_g2_config(cfg)
    # Same early boundary: an RNG reset without an init checkpoint is meaningless.
    validate_reset_rng_config(cfg)
    validate_pair_aux_config(cfg)
    validate_pair_aux_init_checkpoint(cfg)

    setup_ablation_seed (cfg.seed )
    # A tag isolates artifacts from runs with a different configuration.
    suffix =cfg.model_name .lower ()+cfg.tag 
    summary_file =f"./results/test_{suffix }_summary.csv"
    if cfg.skip_existing and os .path .exists (summary_file ):
        logger .info (f"[{cfg.model_name }] Already completed ({summary_file } exists). Skipping.")
        return pd .read_csv (summary_file ).to_dict (orient ='records')[0 ]

    config_info =MODEL_CONFIGS [cfg.model_name ]
    logger .info ("="*70 )
    logger .info (f"   STARTING ABLATION EXPERIMENT: {cfg.model_name }")
    logger .info (f"   Description: {config_info ['desc']}")
    logger .info (f"   Epochs: {cfg.epochs } | Batch Size: {cfg.batch_size } | LR: {cfg.lr } | Seed: {cfg.seed }")
    logger .info ("="*70 )

    save_dir =f"./output/ablation_{suffix }"
    os .makedirs (save_dir ,exist_ok =True )
    os .makedirs ('./results',exist_ok =True )

    with open ('configs/default.yaml')as f :
        yaml_cfg =yaml .load (f ,Loader =yaml .FullLoader )

    yaml_cfg ['model']['params']['sub_model']['transformer'].update (config_info ['transformer_params'])
    # Command-line grid settings override the template dimensions and data path.
    yaml_cfg ['model']['params']['sub_model']['transformer']['edos_num']=cfg.edos_num 
    yaml_cfg ['model']['params']['sub_model']['transformer']['phdos_num']=cfg.phdos_num 
    yaml_cfg ['model']['params']['sub_model']['transformer']['num_decoder_layers']=int (cfg.decoder_layers )
    yaml_cfg ['model']['params']['sub_model']['transformer']['use_atom_additive_phdos']=bool (cfg.use_atom_additive_phdos )
    yaml_cfg ['model']['params']['sub_model']['transformer']['atom_feat_mode']=cfg.atom_feat 
    yaml_cfg ['model']['params']['sub_model']['transformer']['energy_code']=cfg.energy_code 
    yaml_cfg ['model']['params']['sub_model']['transformer']['use_macro_lattice']=bool (cfg.use_macro_lattice )
    for _k ,_v in (("tv_w",cfg.tv_w ),("grad_w",cfg.grad_w ),("peak_w",cfg.peak_w ),
    ("tail_w",cfg.tail_w ),("tail_start",cfg.tail_start )):
        yaml_cfg ['model']['params'][_k ]=_v 
    yaml_cfg ['model']['params']['edos_slope_ratio']=float (cfg.edos_slope_ratio )
    yaml_cfg ['model']['params']['pair_aux_arm']=str (cfg.pair_aux_arm )
    yaml_cfg ['model']['params']['pair_ratio']=float (cfg.pair_ratio )
        # Sum normalization selects the distribution-based loss below.
    yaml_cfg ['model']['params']['use_mask']=bool (cfg.use_mask )
    yaml_cfg ['model']['params']['use_amp']=bool (cfg.use_amp )
    # Optional scale-prediction heads.
    yaml_cfg ['model']['params']['sub_model']['transformer']['scale_mode']=cfg.scale_mode 
    yaml_cfg ['model']['params']['scale_sup_w']=float (cfg.scale_sup_w )
    yaml_cfg ['model']['params']['eta_sup_w']=float (cfg.eta_sup_w )
    yaml_cfg ['model']['params']['delta_edos']=float (cfg.delta_edos )
    yaml_cfg ['model']['params']['delta_phdos']=float (cfg.delta_phdos )
    # Optional boundary-scalar heads.
    yaml_cfg ['model']['params']['sub_model']['transformer']['scalar_mode']=cfg.scalar_mode 
    yaml_cfg ['model']['params']['scalar_sup_w']=float (cfg.scalar_sup_w )
    # Optional sparse periodic graph; disabled leaves the base path unchanged.
    yaml_cfg ['model']['params']['sub_model']['transformer']['use_g1']=bool (cfg.use_g1 )
    yaml_cfg ['model']['params']['sub_model']['transformer']['g1_r_cut']=float (cfg.g1_r_cut )
    yaml_cfg ['model']['params']['sub_model']['transformer']['g1_max_neighbors']=int (cfg.g1_max_neighbors )
    # G2a periodic multi-image message; single fixed cutoff, default off.
    # No neighbor-count or shift-range knobs are exposed for scanning.
    yaml_cfg ['model']['params']['sub_model']['transformer']['use_g2']=bool (cfg.use_g2 )
    yaml_cfg ['model']['params']['sub_model']['transformer']['g2_r_cut']=5.5
    # Candidate-1 edge content function; joint requires use_g2 (validated above)
    # and reaches the model through the same transformer params channel.
    yaml_cfg ['model']['params']['sub_model']['transformer']['g2_content_mode']=str (cfg.g2_content_mode )
    # Optional coordinate-conditioned output trunks.
    yaml_cfg ['model']['params']['sub_model']['transformer']['q1_coord']=bool (cfg.q1_coord )
    yaml_cfg ['model']['params']['sub_model']['transformer']['q1_hidden']=int (cfg.q1_hidden )
    # Fourier variant of the coordinate-conditioned trunk.
    yaml_cfg ['model']['params']['sub_model']['transformer']['q2_fourier']=bool (cfg.q2_fourier )
    # C5 fixed token-level MoE design; only the enable flag is tunable.
    yaml_cfg ['model']['params']['sub_model']['transformer']['c5_moe']=bool (cfg.c5_moe )
    yaml_cfg ['model']['params']['c5_moe_balance_w']=float (cfg.c5_moe_balance_w )
    # R1a parameter-matched pointwise MLP readout heads.
    yaml_cfg ['model']['params']['sub_model']['transformer']['r1a_point']=bool (cfg.r1a_point or cfg.r1b_coord )
    # R1b coordinate-generated decoder query.
    yaml_cfg ['model']['params']['sub_model']['transformer']['r1b_coord']=bool (cfg.r1b_coord )
    # Optional optimization overrides; None keeps the template value.
    if cfg.dropout is not None :
        yaml_cfg ['model']['params']['sub_model']['transformer']['dropout']=float (cfg.dropout )
    if cfg.weight_decay is not None :
        yaml_cfg ['model']['params']['optimizer']['transformer']['params']['weight_decay']=float (cfg.weight_decay )
    if cfg.lambda_ph is not None :
        yaml_cfg ['model']['params']['lambda_ph']=float (cfg.lambda_ph )
    if cfg.grad_clip is not None :
        yaml_cfg ['model']['params']['grad_clip']=float (cfg.grad_clip )
        # Optional W1 and Huber loss weights; None keeps the template value.
    if cfg.w_w1 is not None :
        yaml_cfg ['model']['params']['w_w1']=float (cfg.w_w1 )
    if cfg.w_huber is not None :
        yaml_cfg ['model']['params']['w_huber']=float (cfg.w_huber )
    _wu =cfg.warmup_epochs 
    yaml_cfg ['model']['params']['loss_form']="sumnorm_klw"if cfg.norm =="sumnorm"else "smoothl1"
    _sn =(cfg.norm =="sumnorm")
    yaml_cfg ['model']['params']['sub_model']['transformer']['energy_code']=cfg.energy_code 
    if cfg.energy_code =="edos":
        yaml_cfg ['model']['params']['sub_model']['transformer']['edos_grid']=_resolve_edos_grid (cfg.edos_grid ,cfg.edos_num )
    yaml_cfg ['model']['params']['dos_minmax']=True 
    yaml_cfg ['model']['params']['save_best']='balanced_score'
    yaml_cfg ['dataset']['train']['data_dir']=cfg.data_dir 
    yaml_cfg ['dataset']['valid']['data_dir']=cfg.data_dir 
    yaml_cfg ['dataset']['test']['data_dir']=cfg.data_dir 
    # Apply augmentation only to the training split.
    yaml_cfg ['dataset']['train']['augment']=bool (cfg.augment )
    yaml_cfg ['dataset']['train']['disp_sigma']=float (cfg.disp_sigma )

    builder =ConfigBuilder (**yaml_cfg )

    train_loader =builder .get_dataloader (split ='train',dos_minmax =True ,batch_size =cfg.batch_size ,dos_sumnorm =_sn ,use_bucket_batch =cfg .use_bucket_batch )
    pair_loader ,pair_sampler =maybe_get_pair_aux_loader (train_loader ,cfg )
    val_loader =builder .get_dataloader (split ='valid',dos_minmax =True ,batch_size =cfg.batch_size ,dos_sumnorm =_sn )
    test_loader =maybe_get_test_loader(builder, cfg, _sn)

    model =builder .get_model ()
    device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')
    model .to (device )

    total_params =sum (p .numel ()for p in model .model ['transformer'].parameters ()if p .requires_grad )
    logger .info (f"[{cfg.model_name }] Verified Trainable Parameters: {total_params :,} ({total_params /1e6 :.3f}M)")

    # Optionally initialize from a backbone checkpoint and train only new heads.
    if cfg.init_ckpt :
        _ck =torch .load (cfg.init_ckpt ,map_location ='cpu')
        _st =_ck ['model']if isinstance (_ck ,dict )and 'model'in _ck else _ck 
        if cfg.g2_content_mode == 'joint':
            _miss, _unexp = load_joint_initial_state(model.model['transformer'], _st)
        else:
            _strict_init =cfg.pair_aux_arm !="none"
            _miss ,_unexp =model .model ['transformer'].load_state_dict (
                _st ,strict =_strict_init )
        logger .info (f"[{cfg.model_name }] init_ckpt loaded: missing={list (_miss )[:5 ]} unexpected={list (_unexp )[:5 ]}")
        model .to (device )
    # Candidate-1 fairness shim: after the init checkpoint is consumed and
    # before the first training iteration, realign the Python/NumPy/PyTorch RNG.
    # Architecture-dependent parameter initialization consumes a different
    # number of random draws, so a shared seed alone does not align the dropout
    # stream. Only runs when explicitly requested; default path is unchanged.
    if cfg.reset_rng_after_init:
        setup_ablation_seed(cfg.seed)
        logger.info(
            f"[{cfg.model_name}] RNG reset after init_ckpt load "
            f"(--reset_rng_after_init, seed={cfg.seed})")
    pair_latest_path =os .path .join (save_dir ,'checkpoint_latest.pth')
    if pair_sampler is not None and not os .path .exists (pair_latest_path ):
        pair_sampler .set_epoch (0 )
        first_pair_batch =next (iter (pair_loader ))
        model ._calibrate_edos_pair_weight (first_pair_batch )
    if cfg.freeze_backbone :
        model .model ['transformer'].requires_grad_ (False )
        ntr =0 
        for n ,p in model .model ['transformer'].named_parameters ():
            if 'scale_head_c24'in n or 'eta_head'in n or 'scalar_head'in n :
                p .requires_grad_ (True )
                ntr +=p .numel ()
        logger .info (f"[{cfg.model_name }] backbone frozen, trainable scale params: {ntr :,}")

    optimizer =model .optimizer ['transformer']
    # Apply the command-line learning rate before building the scheduler.
    for pg in optimizer .param_groups :
        pg ['lr']=cfg.lr 
        pg ['initial_lr']=cfg.lr 
    logger .info (f"[{cfg.model_name }] Effective optimizer LR set to {cfg.lr :.2e}")
    # Use one warmup-plus-cosine schedule for every runner invocation.
    from utils .builder import build_warmup_cosine_scheduler 
    scheduler =build_warmup_cosine_scheduler (optimizer ,cfg.epochs )if _wu is None else build_warmup_cosine_scheduler (optimizer ,cfg.epochs ,warmup_epochs =int (_wu ))

    best_val_score =float ('inf')
    best_epoch =0 
    history =[]
    start_epoch =0 

    # Resume long runs from the latest full checkpoint when available.
    # checkpoint_latest.pth carries {epoch, model, optimizer, best_val_score}.
    latest_p =os .path .join (save_dir ,'checkpoint_latest.pth')
    hist_p =f"./results/history_{suffix }.csv"
    config_path =os .path .join (save_dir ,"config_used.yaml")
    if os .path .exists (latest_p ):
        try :
            ck =torch .load (latest_p ,map_location ='cpu')
            expected_pair_plan_hash =None
            expected_pair_lambda =None
            if pair_sampler is not None:
                saved_epoch =int (ck .get ('epoch',0 ))
                if saved_epoch <=0:
                    raise ValueError("pair auxiliary checkpoint must follow a completed epoch")
                expected_pair_plan_hash =pair_sampler .frozen_plan_hash ()
                if not os .path .exists (config_path ):
                    raise ValueError("pair auxiliary resume requires config_used.yaml")
                with open (config_path )as stream:
                    saved_runtime =yaml .safe_load (stream ).get ("runtime",{})
                expected_pair_lambda =saved_runtime .get ("pair_lambda")
                if saved_runtime .get ("pair_plan_hash")!=expected_pair_plan_hash:
                    raise ValueError("pair auxiliary config has a different pair plan")
            resume_meta =restore_ablation_checkpoint (
                ck ,model .model ['transformer'],optimizer ,cfg .use_amp ,model .gscaler,
                edos_slope_ratio=model.edos_slope_ratio,
                pair_aux_arm=cfg.pair_aux_arm,pair_ratio=cfg.pair_ratio,
                pair_lambda=expected_pair_lambda,
                pair_plan_hash=expected_pair_plan_hash)
            if model.edos_slope_ratio > 0.0:
                if resume_meta["edos_slope_lambda"] is None:
                    raise ValueError("slope-loss checkpoint is missing its calibrated lambda")
                model.edos_slope_lambda = float(resume_meta["edos_slope_lambda"])
                model.edos_slope_calibrated = True
                model.edos_slope_calibration = None
            if pair_sampler is not None:
                model.pair_lambda = float(resume_meta["pair_lambda"])
                model.pair_aux_calibrated = True
                model.pair_aux_calibration = None
            start_epoch =resume_meta ['epoch']
            best_val_score =resume_meta ['best_val_score']
            # restore best_epoch + history (history file optional: killed runs
            # only have checkpoints; it is rewritten incrementally below).
            if os .path .exists (hist_p ):
                dfh =pd .read_csv (hist_p )
                history =dfh .to_dict (orient ='records')
                if start_epoch <=0 and len (dfh ):
                    start_epoch =int (dfh ['epoch'].max ())
                if history :
                    best_epoch =int (dfh .loc [dfh ['balanced_score'].idxmin (),'epoch'])
                    # fast-forward cosine part of scheduler to start_epoch
            for _ in range (start_epoch ):
                scheduler .step ()
            model .to (device )
            logger .info (f"[{cfg.model_name }] Resumed from epoch {start_epoch } "
            f"(best ep {best_epoch }, score {best_val_score :.4f})")
        except Exception:
            logger.exception("[%s] Resume failed; existing run artifacts are preserved", cfg.model_name)
            raise

    # Only persist configuration after recovery has succeeded. Preserve an
    # existing resume config, including its recorded slope calibration.
    if not os.path.exists(latest_p) or not os.path.exists(config_path):
        with open(config_path, "w") as f:
            yaml .dump ({'cli':{'model':cfg.model_name ,'epochs':cfg.epochs ,
            'batch_size':cfg.batch_size ,'lr':cfg.lr ,'seed':cfg.seed ,
            'data_dir':cfg.data_dir ,'edos_num':cfg.edos_num ,
            'phdos_num':cfg.phdos_num ,'atom_feat':cfg.atom_feat ,
            'decoder_layers':cfg.decoder_layers ,
            'use_atom_additive_phdos':cfg.use_atom_additive_phdos ,
            'energy_code':cfg.energy_code ,'edos_grid':cfg.edos_grid ,
            'use_macro_lattice':cfg.use_macro_lattice ,
            'tv_w':cfg.tv_w ,'grad_w':cfg.grad_w ,'peak_w':cfg.peak_w ,
            'edos_slope_ratio':cfg.edos_slope_ratio ,
            'pair_aux_arm':cfg.pair_aux_arm ,'pair_ratio':cfg.pair_ratio ,
            'tail_w':cfg.tail_w ,'tail_start':cfg.tail_start ,
            'augment':cfg.augment ,'disp_sigma':cfg.disp_sigma ,
            'norm':cfg.norm ,'use_mask':cfg.use_mask ,'use_amp':cfg.use_amp ,'use_bucket_batch':cfg .use_bucket_batch ,'skip_test_eval':cfg.skip_test_eval ,'dropout':cfg.dropout ,
            'weight_decay':cfg.weight_decay ,'warmup_epochs':_wu ,
            'lambda_ph':cfg.lambda_ph ,'grad_clip':cfg.grad_clip ,
            'w_w1':cfg.w_w1 ,'w_huber':cfg.w_huber ,
            'scale_mode':cfg.scale_mode ,'freeze_backbone':cfg.freeze_backbone ,
            'eta_sup_w':cfg.eta_sup_w ,
            'delta_edos':cfg.delta_edos ,'delta_phdos':cfg.delta_phdos ,
            'scalar_mode':cfg.scalar_mode ,'scalar_sup_w':cfg.scalar_sup_w ,
            'use_g1':cfg.use_g1 ,'g1_r_cut':cfg.g1_r_cut ,
            'g1_max_neighbors':cfg.g1_max_neighbors ,
            'use_g2':cfg.use_g2 ,'g2_r_cut':5.5 ,
            'g2_content_mode':cfg.g2_content_mode ,
            'q1_coord':cfg.q1_coord ,'q1_hidden':cfg.q1_hidden ,
            'q2_fourier':cfg.q2_fourier ,
            'c5_moe':cfg.c5_moe ,'c5_moe_balance_w':cfg.c5_moe_balance_w ,
            'r1a_point':cfg.r1a_point ,
            'r1b_coord':cfg.r1b_coord ,
            'init_ckpt':cfg.init_ckpt ,'scale_sup_w':cfg.scale_sup_w ,
            'reset_rng_after_init':bool (cfg.reset_rng_after_init )},
            'config':yaml_cfg },f ,indent =2 ,sort_keys =False ,
            default_flow_style =False )

    if getattr(model, "pair_aux_calibration", None) is not None:
        write_pair_aux_calibration(
            config_path,
            model.pair_aux_calibration,
            pair_sampler.frozen_plan_hash(),
            pair_sampler.plan_hash(0),
        )
        model.pair_aux_calibration = None


    for epoch in range (start_epoch ,cfg.epochs ):
    # Advance the distributed sampler so each epoch receives a new order.
        sampler =getattr (train_loader ,"sampler",None )
        if not hasattr (sampler,"set_epoch"):
            sampler =getattr (train_loader ,"batch_sampler",sampler )
        if sampler is not None and hasattr (sampler ,"set_epoch"):
            sampler .set_epoch (epoch )
        pair_schedule ={}
        pair_iterator =None
        if pair_sampler is not None:
            pair_sampler .set_epoch (epoch )
            pair_schedule =dict (schedule_auxiliary_batches (len (train_loader ),len (pair_loader )))
            pair_iterator =iter (pair_loader )
        model .model ['transformer'].train ()
        train_loss =0.0 
        sub_loss_accum ={}
        c5_load_accum =None
        n_batches =len (train_loader )
        t_start =time .time ()

        for step ,batch in enumerate (train_loader ):
            pair_batch =next (pair_iterator )if step in pair_schedule else None
            loss_dict =model .train_one_step (batch ,step =step ,pair_batch =pair_batch )
            if model.edos_slope_calibration is not None:
                write_edos_slope_calibration(
                    os.path.join(save_dir, "config_used.yaml"),
                    model.edos_slope_calibration,
                )
                model.edos_slope_calibration = None
            if getattr(model, "pair_aux_calibration", None) is not None:
                write_pair_aux_calibration(
                    os.path.join(save_dir, "config_used.yaml"),
                    model.pair_aux_calibration,
                    pair_sampler.frozen_plan_hash(),
                    pair_sampler.plan_hash(epoch),
                )
                model.pair_aux_calibration = None
            train_loss +=loss_dict ['loss']
            for k ,v in loss_dict .items ():
                sub_loss_accum [k ]=sub_loss_accum .get (k ,0.0 )+v 
            if cfg.c5_moe:
                c5_load =model .model ['transformer'].last_c5_moe_load
                if c5_load is not None and c5_load.numel ():
                    c5_load_accum =c5_load.detach ().cpu ()if c5_load_accum is None else c5_load_accum +c5_load.detach ().cpu ()
            if (step +1 )%100 ==0 or (step +1 )==n_batches :
                logger .info (
                f"[{cfg.model_name }] Epoch [{epoch +1 :03d}/{cfg.epochs :03d}] Step [{step +1 :03d}/{n_batches :03d}] | "
                f"Loss: {loss_dict ['loss']:.4f} (eDOS: {loss_dict ['loss_edos']:.4f}, phDOS: {loss_dict ['loss_phdos']:.4f})"
                )

        scheduler .step ()
        torch .cuda .synchronize ()if torch .cuda .is_available ()else None 
        ep_time =time .time ()-t_start 
        train_loss /=n_batches 
        avg_sub_losses ={f"train_{k }":v /n_batches for k ,v in sub_loss_accum .items ()}
        c5_load_metrics ={}
        if c5_load_accum is not None:
            c5_load =c5_load_accum /n_batches
            c5_load_metrics ={f'c5_expert_load_{i }':float (v )for i ,v in enumerate (c5_load )}

        # Record peak GPU memory for this epoch in the history CSV.
        peak_vram_mb =torch .cuda .max_memory_allocated ()/(1024 **2 )if torch .cuda .is_available ()else 0.0 

        # Extract gate scalars if gated cross-attention is present
        alpha_e ,alpha_p =0.0 ,0.0 
        if model .model ['transformer'].use_gated_cross_attn :
            alpha_e =model .model ['transformer'].gated_cross_attn .alpha_e .item ()
            alpha_p =model .model ['transformer'].gated_cross_attn .alpha_p .item ()

            # Validation reports blind metrics; older scale variants also report oracle metrics.
        val_metrics =evaluate_split (model ,val_loader ,is_m5 =(cfg.model_name =='M5'),ph_grid =_ph_grid_centers (cfg.phdos_num ),sumnorm =(cfg.norm =='sumnorm'))
        balanced =0.5 *val_metrics ['mae_edos_median']+0.5 *val_metrics ['mae_phdos_median']

        log_str =(
        f"[{cfg.model_name }] Epoch [{epoch +1 :03d}/{cfg.epochs :03d}] ({ep_time :.1f}s) | "
        f"Train Loss: {train_loss :.4f} | "
        f"Val eDOS R2(med): {val_metrics ['r2_edos_median']:.3f} (MAE: {val_metrics ['mae_edos_median']:.3f}) | "
        f"Val phDOS R2(med): {val_metrics ['r2_phdos_median']:.3f} (MAE: {val_metrics ['mae_phdos_median']:.4f}) | "
        f"Balanced Score: {balanced :.4f}"
        )
        if cfg.model_name =='M5':
            log_str +=f" | Oracle eDOS R2: {val_metrics ['oracle_r2_edos_median']:.3f}, phDOS R2: {val_metrics ['oracle_r2_phdos_median']:.3f}"
        if model .model ['transformer'].use_gated_cross_attn :
            log_str +=f" | Gate (a_e={alpha_e :.4f}, a_p={alpha_p :.4f})"
        if c5_load_metrics :
            load_text =','.join (f"{v :.2f}"for v in c5_load_metrics .values ())
            log_str +=f" | C5 load ({load_text})"
        logger .info (log_str )

        history .append ({
        'epoch':epoch +1 ,
        'seed':cfg.seed ,
        'epoch_time_s':ep_time ,
        'peak_vram_mb':peak_vram_mb ,
        'alpha_e':alpha_e ,
        'alpha_p':alpha_p ,
        **avg_sub_losses ,
        **c5_load_metrics ,
        **val_metrics ,
        'balanced_score':balanced 
        })

        # Reset the counter to measure each epoch independently.
        if torch .cuda .is_available ():
            torch .cuda .reset_peak_memory_stats ()

            # Save a complete state atomically so interrupted runs remain recoverable.
        def _ckpt (epoch_ ,best_ ):
            return build_ablation_checkpoint (
                epoch_,cfg .model_name,cfg .seed,cfg .use_amp,
                model .model ['transformer'],optimizer,best_,model .gscaler,
                edos_slope_ratio=(model.edos_slope_ratio if model.edos_slope_ratio > 0.0 else None),
                edos_slope_lambda=(model.edos_slope_lambda if model.edos_slope_ratio > 0.0 else None),
                pair_aux_arm=(cfg.pair_aux_arm if pair_sampler is not None else None),
                pair_ratio=(model.pair_ratio if pair_sampler is not None else None),
                pair_lambda=(model.pair_lambda if pair_sampler is not None else None),
                pair_plan_hash=(pair_sampler.frozen_plan_hash() if pair_sampler is not None else None))

        if balanced <best_val_score :
            best_val_score =balanced 
            best_epoch =epoch +1 
            atomic_torch_save (_ckpt (epoch +1 ,balanced ),os .path .join (save_dir ,'checkpoint_best.pth'))
            logger .info (f"[{cfg.model_name }] New best model saved at Epoch {epoch +1 } (Score: {balanced :.4f})")

        atomic_torch_save (_ckpt (epoch +1 ,best_val_score ),os .path .join (save_dir ,'checkpoint_latest.pth'))
        # Incremental history (killed runs resume from checkpoint + history).
        pd .DataFrame (history ).to_csv (f"./results/history_{suffix }.csv",index =False )

        # Save training history
    df_history =pd .DataFrame (history )
    df_history .to_csv (f"./results/history_{suffix }.csv",index =False )

    if cfg.skip_test_eval:
        logger.info("Test loader and automatic test evaluation skipped by request")
        return {"test_skipped": True}

    # Load best checkpoint and evaluate on Test set
    logger .info (f"\nEvaluating Best Model ({cfg.model_name }, Epoch {best_epoch }) on Test Set ({len (test_loader .dataset )} materials)...")
    best_ckpt =torch .load (os .path .join (save_dir ,'checkpoint_best.pth'))
    best_state =best_ckpt ['model']if isinstance (best_ckpt ,dict )and 'model'in best_ckpt else best_ckpt 
    model .model ['transformer'].load_state_dict (best_state )

    test_metrics =evaluate_split (model ,test_loader ,is_m5 =(cfg.model_name =='M5'),return_sample_level =True ,
    ph_grid =_ph_grid_centers (cfg.phdos_num ),sumnorm =(cfg.norm =='sumnorm'))
    df_samples =test_metrics .pop ('sample_df')
    pred_p =test_metrics .pop ('pred_phdos')
    pred_e =test_metrics .pop ('pred_edos')
    tgt_p =test_metrics .pop ('tgt_phdos')
    tgt_e =test_metrics .pop ('tgt_edos')

    np .save (f"./results/pred_phdos_{suffix }.npy",pred_p )
    np .save (f"./results/pred_edos_{suffix }.npy",pred_e )
    if cfg.model_name =='M5':
        np .save ('./results/pred_phdos.npy',pred_p )
        np .save ('./results/pred_edos.npy',pred_e )
        np .save ('./results/tgt_phdos.npy',tgt_p )
        np .save ('./results/tgt_edos.npy',tgt_e )
        df_samples .to_csv ('./results/test_evaluation_summary.csv',index =False )

    df_samples .to_csv (f"./results/samples_{suffix }_test.csv",index =False )

    df_test_summary =pd .DataFrame ([test_metrics ])
    df_test_summary .to_csv (f"./results/test_{suffix }_summary.csv",index =False )

    logger .info ("="*70 )
    logger .info (f"   TEST EVALUATION SUMMARY ({cfg.model_name })")
    logger .info ("="*70 )
    for k ,v in test_metrics .items ():
        logger .info (f"   {k :28s}: {v :.4f}"if isinstance (v ,float )else f"   {k :28s}: {v }")
    logger .info ("="*70 )
    return test_metrics 

def evaluate_split (model ,dataloader ,is_m5 :bool =False ,return_sample_level :bool =False ,ph_grid =None ,sumnorm :bool =False ):
    model .model ['transformer'].eval ()
    records =[]
    # Thermodynamic integration requires a uniform frequency grid. Skip it for
    # non-uniform grids; spectral verdict metrics remain available.
    skip_thermo ,_calc_override =False ,None 
    if ph_grid is not None :
        import numpy as _np 
        _g =_np .asarray (ph_grid ,dtype =float )
        _d =_np .diff (_g )
        if _np .allclose (_d ,_d [0 ],rtol =1e-3 ):
            from thermo_props import ThermodynamicCalculator as _TC 
            _calc_override =_TC (ph_freq_min =float (_g [0 ]),ph_freq_max =float (_g [-1 ]),
            ph_bins =len (_g ))
        else :
            skip_thermo =True 
    if return_sample_level :
        all_p_p ,all_t_p =[],[]
        all_p_e ,all_t_e =[],[]

    with torch .no_grad ():
        for batch in dataloader :
            inp ,pos ,mask ,edos_tgt ,phdos_tgt ,edos_m ,edos_s ,edos_min ,edos_max ,phdos_m ,phdos_s ,phdos_min ,phdos_max ,edos_cov ,phdos_cov ,_nvalence ,edos_x ,phdos_x =model .data_preprocess (batch )

            with model .amp_autocast ():
                outputs =model .model ['transformer'](inp ,mask ,pos ,edos_x ,phdos_x )
            outputs =model .fp32_outputs (outputs)

            # Targets in true physical space
            t_e =edos_tgt *(edos_max -edos_min )+edos_min 
            t_p =phdos_tgt *(phdos_max -phdos_min )+phdos_min 

            if is_m5 :
            # 1. Blind physical prediction
                p_e =torch .clamp (outputs ['phys_edos'],min =0.0 )
                p_p =torch .clamp (outputs ['phys_phdos'],min =0.0 )
                # 2. Oracle denormalization (with ground-truth min/max)
                p_e_orc =torch .clamp (outputs ['shape_edos']*(edos_max -edos_min )+edos_min ,min =0.0 )
                p_p_orc =torch .clamp (outputs ['shape_phdos']*(phdos_max -phdos_min )+phdos_min ,min =0.0 )
            else :
            # Direct-output variants use oracle denormalization. For SumNorm,
            # convert logits to a distribution before restoring its total.
                if sumnorm :
                    import torch .nn .functional as _F 
                    p_e =torch .clamp (_F .softmax (outputs ['edos'],dim =-1 )*(edos_max -edos_min )+edos_min ,min =0.0 )
                    p_p =torch .clamp (_F .softmax (outputs ['phdos'],dim =-1 )*(phdos_max -phdos_min )+phdos_min ,min =0.0 )
                else :
                    p_e =torch .clamp (outputs ['edos']*(edos_max -edos_min )+edos_min ,min =0.0 )
                    p_p =torch .clamp (outputs ['phdos']*(phdos_max -phdos_min )+phdos_min ,min =0.0 )
                p_e_orc ,p_p_orc =None ,None 

                # Sample-wise primary metrics (H2 hygiene: shared fn, bit-identical math)
            from utils .metrics import per_sample_spectral_metrics 
            _me =per_sample_spectral_metrics (p_e ,t_e )
            mae_e ,mse_e ,r2_e =_me ['mae'],_me ['mse'],_me ['r2']
            _mp =per_sample_spectral_metrics (p_p ,t_p )
            mae_p ,mse_p ,r2_p =_mp ['mae'],_mp ['mse'],_mp ['r2']

            # Sample-wise Oracle metrics if M5
            if is_m5 :
                _meo =per_sample_spectral_metrics (p_e_orc ,t_e )
                mae_e_orc ,mse_e_orc ,r2_e_orc =_meo ['mae'],_meo ['mse'],_meo ['r2']
                _mpo =per_sample_spectral_metrics (p_p_orc ,t_p )
                mae_p_orc ,mse_p_orc ,r2_p_orc =_mpo ['mae'],_mpo ['mse'],_mpo ['r2']

            if return_sample_level :
                all_p_p .append (p_p .cpu ().numpy ())
                all_t_p .append (t_p .cpu ().numpy ())
                all_p_e .append (p_e .cpu ().numpy ())
                all_t_e .append (t_e .cpu ().numpy ())

            B =inp .shape [0 ]
            for i in range (B ):
                rec ={
                'mae_edos':mae_e [i ].item (),
                'mse_edos':mse_e [i ].item (),
                'r2_edos':r2_e [i ].item (),
                'mae_phdos':mae_p [i ].item (),
                'mse_phdos':mse_p [i ].item (),
                'r2_phdos':r2_p [i ].item (),
                }
                if is_m5 :
                    rec .update ({
                    'oracle_mae_edos':mae_e_orc [i ].item (),
                    'oracle_mse_edos':mse_e_orc [i ].item (),
                    'oracle_r2_edos':r2_e_orc [i ].item (),
                    'oracle_mae_phdos':mae_p_orc [i ].item (),
                    'oracle_mse_phdos':mse_p_orc [i ].item (),
                    'oracle_r2_phdos':r2_p_orc [i ].item (),
                    })
                records .append (rec )

    df =pd .DataFrame (records )
    summary ={
    'mae_edos_mean':float (df ['mae_edos'].mean ()),
    'mae_edos_std':float (df ['mae_edos'].std ()),
    'mae_edos_median':float (df ['mae_edos'].median ()),
    'mse_edos_mean':float (df ['mse_edos'].mean ()),
    'mse_edos_std':float (df ['mse_edos'].std ()),
    'r2_edos_mean':float (df ['r2_edos'].mean ()),
    'r2_edos_std':float (df ['r2_edos'].std ()),
    'r2_edos_median':float (df ['r2_edos'].median ()),
    'fail_rate_edos':float ((df ['r2_edos']<0 ).mean ()*100.0 ),

    'mae_phdos_mean':float (df ['mae_phdos'].mean ()),
    'mae_phdos_std':float (df ['mae_phdos'].std ()),
    'mae_phdos_median':float (df ['mae_phdos'].median ()),
    'mse_phdos_mean':float (df ['mse_phdos'].mean ()),
    'mse_phdos_std':float (df ['mse_phdos'].std ()),
    'r2_phdos_mean':float (df ['r2_phdos'].mean ()),
    'r2_phdos_std':float (df ['r2_phdos'].std ()),
    'r2_phdos_median':float (df ['r2_phdos'].median ()),
    'fail_rate_phdos':float ((df ['r2_phdos']<0 ).mean ()*100.0 ),
    }

    if is_m5 :
        summary .update ({
        'oracle_mae_edos_mean':float (df ['oracle_mae_edos'].mean ()),
        'oracle_mae_edos_median':float (df ['oracle_mae_edos'].median ()),
        'oracle_r2_edos_mean':float (df ['oracle_r2_edos'].mean ()),
        'oracle_r2_edos_median':float (df ['oracle_r2_edos'].median ()),
        'oracle_mae_phdos_mean':float (df ['oracle_mae_phdos'].mean ()),
        'oracle_mae_phdos_median':float (df ['oracle_mae_phdos'].median ()),
        'oracle_r2_phdos_mean':float (df ['oracle_r2_phdos'].mean ()),
        'oracle_r2_phdos_median':float (df ['oracle_r2_phdos'].median ()),
        })

    if return_sample_level :
        from thermo_props import ThermodynamicCalculator 
        calc =_calc_override if _calc_override is not None else ThermodynamicCalculator ()
        p_p_arr =np .concatenate (all_p_p ,axis =0 )
        t_p_arr =np .concatenate (all_t_p ,axis =0 )
        p_e_arr =np .concatenate (all_p_e ,axis =0 )
        t_e_arr =np .concatenate (all_t_e ,axis =0 )

        cv_preds ,cv_tgts =[],[]
        debye_preds ,debye_tgts =[],[]
        if skip_thermo :
            n =len (p_p_arr )
            cv_preds ,cv_tgts =[float ("nan")]*n ,[float ("nan")]*n 
            debye_preds ,debye_tgts =[float ("nan")]*n ,[float ("nan")]*n 
        else :
            for i in range (len (p_p_arr )):
                d_p =calc .compute_Debye_T (p_p_arr [i ])
                d_t =calc .compute_Debye_T (t_p_arr [i ])
                debye_preds .append (d_p )
                debye_tgts .append (d_t )
                cv_p =calc .compute_Cv (p_p_arr [i ],T_range =[300.0 ])[0 ]
                cv_t =calc .compute_Cv (t_p_arr [i ],T_range =[300.0 ])[0 ]
                cv_preds .append (cv_p )
                cv_tgts .append (cv_t )

        df ['cv_pred']=cv_preds 
        df ['cv_true']=cv_tgts 
        df ['debye_pred']=debye_preds 
        df ['debye_true']=debye_tgts 

        valid_debye =(np .array (debye_tgts )>50 )&(np .array (debye_preds )>50 )
        if np .sum (valid_debye )>0 :
            cv_mae =float (np .mean (np .abs (np .array (cv_preds )[valid_debye ]-np .array (cv_tgts )[valid_debye ])))
        elif np .all (~np .isfinite (np .array (cv_preds ,dtype =float ))):
            cv_mae =float ("nan")# thermo skipped (nonuniform grid)
        else :
            cv_mae =float (np .mean (np .abs (np .array (cv_preds )-np .array (cv_tgts ))))
        summary ['cv_mae']=cv_mae 
        summary ['sample_df']=df 
        summary ['pred_phdos']=p_p_arr 
        summary ['pred_edos']=p_e_arr 
        summary ['tgt_phdos']=t_p_arr 
        summary ['tgt_edos']=t_e_arr 

    return summary 

def build_arg_parser():
    """CLI parser (exposed so contract tests can exercise the exact options)."""
    parser = argparse.ArgumentParser(description='uniARPAT Ablation Experiments Runner')
    parser.add_argument('--model', type=str, default='M1', choices=['M1', 'M2', 'M3', 'M4', 'M5', 'all'], help='Model variant to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--skip_existing', action='store_true', help='Skip variant if test summary already exists')
    parser.add_argument('--seed', type=int, default=42, help='Random seed recorded in the history CSV')
    parser.add_argument('--tag', type=str, default='', help='Unique suffix for output and result files')
    parser.add_argument('--data_dir', type=str, default='./data/train4ARPAT', help='Dataset root')
    parser.add_argument('--edos_num', type=int, default=128, help='Number of eDOS output bins')
    parser.add_argument('--phdos_num', type=int, default=64, help='Number of phDOS output bins')
    parser.add_argument('--decoder_layers', type=int, default=6,
                        help='Shared Transformer decoder depth; 6 is the B7 default')
    parser.add_argument('--use_atom_additive_phdos', action='store_true',
                        help='R2b: sum nonnegative fixed-grid phDOS contributions from atom tokens')
    parser.add_argument('--atom_feat', type=str, default='legacy3', choices=['legacy3', 'mendeleev24'], help='Atomic feature table')
    parser.add_argument('--energy_code', type=str, default='none', choices=['none', 'edos'], help='Add an eDOS bin-energy encoding')
    parser.add_argument('--edos_grid', type=str, default='', help='Named eDOS grid or path to bin centers')
    parser.add_argument('--use_macro_lattice', action='store_true', help='Enable E10 CIF-only macro lattice residual')
    parser.add_argument('--tv_w', type=float, default=0.0, help='Total-variation loss weight')
    parser.add_argument('--grad_w', type=float, default=0.0, help='Gradient-matching loss weight')
    parser.add_argument('--edos_slope_ratio', type=float, default=0.0, help='Calibrated eDOS slope-loss gradient ratio; zero disables it')
    parser.add_argument('--pair_aux_arm', type=str, default='none',
                        choices=['none', 'control', 'candidate'],
                        help='Opt-in same-composition eDOS pair auxiliary arm')
    parser.add_argument('--pair_ratio', type=float, default=0.0,
                        help='Frozen pair-loss gradient ratio; zero disables pair planning')
    parser.add_argument('--peak_w', type=float, default=1.0, help='Weight for high-density eDOS bins')
    parser.add_argument('--tail_w', type=float, default=1.0, help='Weight for high-frequency phDOS bins')
    parser.add_argument('--tail_start', type=int, default=-1, help='First high-frequency phDOS bin; -1 disables the region')
    parser.add_argument('--augment', action='store_true', help='Apply training-only phonon coordinate displacement')
    parser.add_argument('--disp_sigma', type=float, default=0.01, help='Coordinate-displacement standard deviation in fractional units')
    parser.add_argument('--norm', type=str, default='sumnorm', choices=['minmax', 'sumnorm'], help='Target normalization; sumnorm is the default')
    parser.add_argument('--use_mask', action='store_true', help='Experimental: mask unsupported bins in the loss')
    parser.add_argument('--skip_test_eval', action='store_true', help='Do not construct or evaluate the test split')
    parser.add_argument('--use_amp', action='store_true', help='C4: CUDA FP16 automatic mixed precision')
    parser.add_argument('--use_bucket_batch', action='store_true', help='E6: fixed-window length bucket plus dynamic padding trim')
    parser.add_argument('--dropout', type=float, default=None, help='Transformer dropout; None uses the template value (0.05)')
    parser.add_argument('--weight_decay', type=float, default=None, help='AdamW weight decay; None uses the template value')
    parser.add_argument('--warmup_epochs', type=int, default=None, help='Warmup duration; None uses the template value')
    parser.add_argument('--lambda_ph', type=float, default=None, help='Relative phDOS loss weight')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient-norm limit; None disables clipping')
    parser.add_argument('--w_w1', type=float, default=None, help='One-dimensional Wasserstein loss weight')
    parser.add_argument('--w_huber', type=float, default=None, help='Huber loss weight')
    parser.add_argument('--scale_mode', type=str, default='eta', choices=['none', 'decoupled', 'eta'], help='Blind-inference scale head; eta is the default')
    parser.add_argument('--eta_sup_w', type=float, default=1.0, help='Eta/gamma auxiliary-loss weight')
    parser.add_argument('--delta_edos', type=float, default=0.09375, help='eDOS bin width in eV for eta/gamma supervision')
    parser.add_argument('--delta_phdos', type=float, default=19.6875, help='phDOS bin width in cm^-1 for eta/gamma supervision')
    parser.add_argument('--scalar_mode', type=str, default='none', choices=['none', 's1'], help='Optional boundary-scalar heads')
    parser.add_argument('--scalar_sup_w', type=float, default=1.0, help='Boundary-scalar auxiliary-loss weight')
    parser.add_argument('--use_g1', action='store_true', help='Enable the experimental sparse periodic graph')
    parser.add_argument('--g1_r_cut', type=float, default=5.5, help='Sparse-graph cutoff in Å')
    parser.add_argument('--g1_max_neighbors', type=int, default=48, help='Maximum graph neighbors per atom')
    parser.add_argument('--use_g2', action='store_true', help='Enable G2a periodic multi-image Value residual (fixed R=5.5)')
    parser.add_argument('--g2_content_mode', type=str, default='radial', choices=['radial', 'joint'],
                        help='G2a edge content function; joint requires --use_g2 (candidate 1)')
    parser.add_argument('--q1_coord', action='store_true', help='Enable coordinate-conditioned output trunks')
    parser.add_argument('--q1_hidden', type=int, default=128, help='Hidden size of coordinate trunks')
    parser.add_argument('--q2_fourier', action='store_true', help='Use Fourier features in coordinate trunks')
    parser.add_argument('--c5_moe', action='store_true', help='Enable C5 fixed token-level Top-2 decoder MoE')
    parser.add_argument('--r1a_point', action='store_true', help='Enable R1a parameter-matched pointwise MLP readout heads')
    parser.add_argument('--r1b_coord', action='store_true', help='Enable R1b coordinate-generated decoder query')
    parser.add_argument('--freeze_backbone', action='store_true', help='Train only auxiliary heads after initialization')
    parser.add_argument('--init_ckpt', type=str, default='', help='Checkpoint used to initialize the model')
    parser.add_argument('--reset_rng_after_init', action='store_true',
                        help='Re-seed the training RNG right after --init_ckpt is loaded '
                             '(candidate-1 fairness shim; requires --init_ckpt)')
    parser.add_argument('--scale_sup_w', type=float, default=1.0, help='Weight of scale-prediction supervision')
    return parser


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.model == 'all':
        for m in ['M1', 'M2', 'M3', 'M4', 'M5']:
            args.model = m
            cfg = ExperimentConfig.from_args(args)
            train_and_eval(cfg)
    else:
        cfg = ExperimentConfig.from_args(args)
        train_and_eval(cfg)
